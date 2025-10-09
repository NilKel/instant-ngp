/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   hash_surface_network.h
 *  @brief  Hash-based surface features with compact representation
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

template <typename T>
class HashSurfaceNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	HashSurfaceNetwork(
		uint32_t n_pos_dims,
		uint32_t n_dir_dims,
		uint32_t n_extra_dims,
		uint32_t dir_offset,
		const json& pos_encoding,
		const json& dir_encoding,
		const json& density_network,
		const json& rgb_network,
		bool use_sdf = false
	) : NerfNetworkBase<T>(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
	                       pos_encoding, dir_encoding, density_network, rgb_network,
	                       "hash_surface", use_sdf) {
		
		// Hash surface: Extract n_levels density features, MLP outputs 1D density
		uint32_t hash_output_width = this->m_pos_encoding->padded_output_width();
		uint32_t n_levels = hash_output_width / 4; // e.g., 32 / 4 = 8 levels
		
		json local_density_network_config = density_network;
		
		// Density MLP input: extracted density features (padded to alignment)
		uint32_t density_alignment = tcnn::minimum_alignment(density_network);
		uint32_t padded_density_input_dims = tcnn::next_multiple(n_levels, density_alignment);
		local_density_network_config["n_input_dims"] = padded_density_input_dims;
		
		// Density MLP output: 1D density
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 1;
		}
		
		this->m_density_network.reset(tcnn::create_network<T>(local_density_network_config));
		
		// RGB network input: n_levels surface features + 1D density + direction encoding
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		this->m_rgb_network_input_width = tcnn::next_multiple(
			n_levels + 1 + this->m_dir_encoding->padded_output_width(),
			rgb_alignment
		);
		
		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = this->m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		this->m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
		
		this->m_density_model = std::make_shared<tcnn::NetworkWithInputEncoding<T>>(
			this->m_pos_encoding, this->m_density_network
		);
	}

	void inference_mixed_precision_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		uint32_t batch_size = input.n();
		uint32_t n_levels = this->m_pos_encoding->padded_output_width() / 4;
		
		// Hash features matrix (full 32D from position encoding)
		tcnn::GPUMatrixDynamic<T> hash_features{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		// Density network input (extracted + padded features)
		tcnn::GPUMatrixDynamic<T> density_network_input{
			this->m_density_network->input_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		tcnn::GPUMatrixDynamic<T> rgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		
		tcnn::GPUMatrixDynamic<T> density_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0, rgb_network_input.n_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(density_network_input.data(), 0, density_network_input.n_bytes(), stream));
		
		// Position encoding -> hash features
		this->m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			hash_features,
			use_inference_params
		);
		
		// Extract density features from hash
		linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
			batch_size, n_levels, 4,
			hash_features.data(),
			hash_features.layout() == tcnn::AoS ? hash_features.stride() : 1,
			density_network_input.data(),
			density_network_input.layout() == tcnn::AoS ? density_network_input.stride() : 1
		);
		
		// Density MLP
		this->m_density_network->inference_mixed_precision(
			stream, density_network_input, density_network_output, use_inference_params
		);
		
		// Compute analytical normals from density
		tcnn::GPUMatrixDynamic<float> analytical_normals = compute_analytical_normals_inference(
			stream, batch_size, input, density_network_input, density_network_output, hash_features, use_inference_params
		);
		
		// Compute hash surface features
		uint32_t surface_features = n_levels;
		linear_kernel(compute_hash_surface_features_kernel<T>, 0, stream,
			batch_size, n_levels, 4,
			hash_features.data(),
			hash_features.layout() == tcnn::AoS ? hash_features.stride() : 1,
			analytical_normals.data(),
			rgb_network_input.data(),
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1
		);
		
		// Copy density to RGB input after surface features
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1,
			density_network_output.data(),
			rgb_network_input.data() + (rgb_network_input.layout() == tcnn::AoS ? surface_features : surface_features * batch_size),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr,
			false
		);
		
		// Direction encoding
		auto dir_out = rgb_network_input.slice_rows(surface_features + 1, this->m_dir_encoding->padded_output_width());
		this->m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);
		
		// RGB network
		tcnn::GPUMatrixDynamic<T> rgb_network_output{
			output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};
		this->m_rgb_network->inference_mixed_precision(
			stream, rgb_network_input, rgb_network_output, use_inference_params
		);
		
		// Extract final density
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

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		uint32_t batch_size = input.n();
		uint32_t n_levels = this->m_pos_encoding->padded_output_width() / 4;
		
		auto forward = std::make_unique<ForwardContext>();
		
		// Full hash features from position encoding
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
			true  // Always prepare gradients for hash_surface
		);
		
		// Extract density features
		tcnn::GPUMatrixDynamic<T> extracted_density_features{
			this->m_density_network->input_width(), batch_size, stream,
			forward->density_network_input.layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(extracted_density_features.data(), 0,
		                                  extracted_density_features.n_bytes(), stream));
		
		linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
			batch_size, n_levels, 4,
			forward->density_network_input.data(),
			forward->density_network_input.layout() == tcnn::AoS ? forward->density_network_input.stride() : 1,
			extracted_density_features.data(),
			extracted_density_features.layout() == tcnn::AoS ? extracted_density_features.stride() : 1
		);
		
		// Density MLP forward
		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		forward->density_network_ctx = this->m_density_network->forward(
			stream, extracted_density_features, &forward->density_network_output,
			use_inference_params, true
		);
		
		// Store extracted features for backward
		forward->extracted_density_features = std::move(extracted_density_features);
		
		// Compute normals and surface features (simplified - full impl needs helper methods)
		// ... (similar to inference but with context saving)
		
		// RGB forward
		auto dir_out = forward->rgb_network_input.slice_rows(n_levels + 1, this->m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = this->m_dir_encoding->forward(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			&dir_out,
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
			// Extract density to output
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

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::GradientMode param_gradients_mode = tcnn::GradientMode::Overwrite
	) override {
		// Hash surface backward is complex - implement based on original nerf_network.h lines 1249-1356
		// For now, stub implementation
		throw std::runtime_error("HashSurfaceNetwork::backward_impl not yet fully implemented");
	}

	void density(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("HashSurfaceNetwork::density input must be in column major format.");
		}
		
		uint32_t batch_size = output.n();
		uint32_t n_levels = this->m_pos_encoding->padded_output_width() / 4;
		
		tcnn::GPUMatrixDynamic<T> hash_features{
			this->m_pos_encoding->padded_output_width(), batch_size, stream, input.layout()
		};
		
		this->m_pos_encoding->inference_mixed_precision(
			stream, input.slice_rows(0, this->m_pos_encoding->input_width()),
			hash_features, use_inference_params
		);
		
		uint32_t padded_density_input_width = this->m_density_network->input_width();
		tcnn::GPUMatrixDynamic<T> padded_density_features{
			padded_density_input_width, batch_size, stream, input.layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(padded_density_features.data(), 0,
		                                  padded_density_features.n_bytes(), stream));
		
		linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
			batch_size, n_levels, 4,
			hash_features.data(),
			hash_features.layout() == tcnn::AoS ? hash_features.stride() : 1,
			padded_density_features.data(),
			padded_density_features.layout() == tcnn::AoS ? padded_density_features.stride() : 1
		);
		
		this->m_density_network->inference_mixed_precision(
			stream, padded_density_features, output, use_inference_params
		);
	}

private:
	struct ForwardContext : public NerfNetworkBase<T>::ForwardContextBase {
		tcnn::GPUMatrixDynamic<T> extracted_density_features;
		tcnn::GPUMatrixDynamic<float> analytical_normals;
	};

	tcnn::GPUMatrixDynamic<float> compute_analytical_normals_inference(
		cudaStream_t stream,
		uint32_t batch_size,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& density_network_input,
		const tcnn::GPUMatrixDynamic<T>& density_network_output,
		const tcnn::GPUMatrixDynamic<T>& hash_features,
		bool use_inference_params
	) {
		// Simplified - full implementation would replicate original logic
		tcnn::GPUMatrixDynamic<float> normals{3, batch_size, stream, tcnn::AoS};
		// TODO: Implement full normal computation
		return normals;
	}
};

} // namespace ngp

