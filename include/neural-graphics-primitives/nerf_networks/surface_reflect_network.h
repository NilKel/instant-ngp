/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   surface_reflect_network.h
 *  @brief  Surface rendering with encoded reflection vectors
 */

#pragma once

#include "surface_network.h"

namespace ngp {

template <typename T>
class SurfaceReflectNetwork : public SurfaceNetwork<T> {
public:
	using json = nlohmann::json;

	SurfaceReflectNetwork(
		uint32_t n_pos_dims,
		uint32_t n_dir_dims,
		uint32_t n_extra_dims,
		uint32_t dir_offset,
		const json& pos_encoding,
		const json& dir_encoding,
		const json& density_network,
		const json& rgb_network,
		bool use_sdf = false
	) : SurfaceNetwork<T>(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
	                      pos_encoding, dir_encoding, density_network, rgb_network,
	                      use_sdf) {
		
		// Override method name
		this->m_method = "surface_reflect";
		
		// RGB network input: 16 surface features + encoded view dirs + encoded reflection vectors
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		uint32_t total_before_padding = 16 + 
		                                 this->m_dir_encoding->padded_output_width() + 
		                                 this->m_dir_encoding->padded_output_width();
		this->m_rgb_network_input_width = tcnn::next_multiple(total_before_padding, rgb_alignment);
		
		// Recreate RGB network with updated input width
		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = this->m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		this->m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
	}

	// Override inference to add reflection vector encoding
	void inference_mixed_precision_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		uint32_t batch_size = input.n();
		
		tcnn::GPUMatrixDynamic<T> density_network_input{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		tcnn::GPUMatrixDynamic<T> rgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		
		tcnn::GPUMatrixDynamic<T> density_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0, rgb_network_input.n_bytes(), stream));
		
		// Position encoding -> density network
		this->m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);
		
		this->m_density_network->inference_mixed_precision(
			stream, density_network_input, density_network_output, use_inference_params
		);
		
		// Compute analytical normals
		tcnn::GPUMatrixDynamic<float> analytical_normals = this->compute_analytical_normals_inference(
			stream, batch_size, input, density_network_input, density_network_output, use_inference_params
		);
		
		// Compute surface features
		linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			density_network_output.data(),
			analytical_normals.data(),
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1,
			rgb_network_input.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr
		);
		
		// Direction encoding
		auto dir_out = rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		this->m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);
		
		// Compute and encode reflection vectors
		uint32_t reflect_start_idx = 16 + this->m_dir_encoding->padded_output_width();
		auto reflect_out = rgb_network_input.slice_rows(reflect_start_idx, this->m_dir_encoding->padded_output_width());
		
		auto view_dirs_input = input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width());
		tcnn::GPUMatrixDynamic<float> reflection_vectors{3, batch_size, stream, analytical_normals.layout()};
		
		linear_kernel(calculate_reflection_vector_kernel<T>, 0, stream,
			batch_size,
			view_dirs_input.data(),
			analytical_normals.data(),
			reflection_vectors.data(),
			this->m_dir_encoding->input_width(),
			view_dirs_input.layout() == tcnn::AoS ? view_dirs_input.m() : 1,
			analytical_normals.layout() == tcnn::AoS ? 3 : 1,
			reflection_vectors.layout() == tcnn::AoS ? 3 : 1
		);
		
		this->m_dir_encoding->inference_mixed_precision(
			stream,
			reflection_vectors,
			reflect_out,
			use_inference_params
		);
		
		// RGB network
		tcnn::GPUMatrixDynamic<T> rgb_network_output{
			output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};
		this->m_rgb_network->inference_mixed_precision(
			stream, rgb_network_input, rgb_network_output, use_inference_params
		);
		
		// Extract density
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? this->padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr,
			false
		);
	}

	// Override forward to add reflection encoding context
	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		uint32_t batch_size = input.n();
		auto forward = std::make_unique<ForwardContext>();
		
		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(forward->rgb_network_input.data(), 0,
		                                  forward->rgb_network_input.n_bytes(), stream));
		
		forward->pos_encoding_ctx = this->m_pos_encoding->forward(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			true
		);
		
		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output,
			use_inference_params, true
		);
		
		// Compute analytical normals
		forward->analytical_normals = this->compute_analytical_normals_forward(
			stream, batch_size, input, forward, use_inference_params
		);
		
		// Compute surface features
		auto surface_features_slice = forward->rgb_network_input.slice_rows(0, 16);
		linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
			batch_size,
			forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
			forward->density_network_output.data(),
			forward->analytical_normals.data(),
			surface_features_slice.layout() == tcnn::AoS ? surface_features_slice.stride() : 1,
			surface_features_slice.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr
		);
		
		// Direction encoding
		auto dir_out = forward->rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = this->m_dir_encoding->forward(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);
		
		// Reflection vector encoding
		uint32_t reflect_start_idx = 16 + this->m_dir_encoding->padded_output_width();
		auto reflection_section = forward->rgb_network_input.slice_rows(reflect_start_idx, this->m_dir_encoding->padded_output_width());
		
		auto view_dirs_input = input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width());
		tcnn::GPUMatrixDynamic<float> reflection_vectors{3, batch_size, stream, forward->analytical_normals.layout()};
		
		linear_kernel(calculate_reflection_vector_kernel<T>, 0, stream,
			batch_size,
			view_dirs_input.data(),
			forward->analytical_normals.data(),
			reflection_vectors.data(),
			this->m_dir_encoding->input_width(),
			view_dirs_input.layout() == tcnn::AoS ? view_dirs_input.m() : 1,
			forward->analytical_normals.layout() == tcnn::AoS ? 3 : 1,
			reflection_vectors.layout() == tcnn::AoS ? 3 : 1
		);
		
		forward->reflection_encoding_ctx = this->m_dir_encoding->forward(
			stream,
			reflection_vectors,
			&reflection_section,
			use_inference_params,
			prepare_input_gradients
		);
		
		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrix<T>{
				output->data(), this->m_rgb_network->padded_output_width(), batch_size, output->layout()
			};
		}
		
		forward->rgb_network_ctx = this->m_rgb_network->forward(
			stream, forward->rgb_network_input,
			output ? &forward->rgb_network_output : nullptr,
			use_inference_params, prepare_input_gradients
		);
		
		if (output) {
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
				output->layout() == tcnn::AoS ? this->padded_output_width() : 1,
				forward->density_network_output.data(),
				output->data() + 3 * (output->layout() == tcnn::AoS ? 1 : batch_size),
				this->m_use_sdf,
				this->m_variance_network ? this->m_variance_network->params() : nullptr,
				false
			);
		}
		
		return forward;
	}

private:
	struct ForwardContext : public SurfaceNetwork<T>::ForwardContext {
		std::unique_ptr<tcnn::Context> reflection_encoding_ctx;
	};
};

} // namespace ngp

