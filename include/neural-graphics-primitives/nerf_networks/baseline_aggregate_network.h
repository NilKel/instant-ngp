/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   baseline_aggregate_network.h
 *  @brief  NeRF baseline aggregate network implementation
 *          Outputs 8D from density MLP [density, diffuse_RGB, features]
 *          Alpha blends 7D per pixel, then applies RGB MLP to features
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

// Helper kernels for baseline_aggregate mode (must be defined before class)

template <typename T>
__global__ void copy_density_network_output(
	const uint32_t n_elements,
	const uint32_t src_stride,
	const uint32_t dst_stride,
	const T* __restrict__ src,
	T* __restrict__ dst,
	const uint32_t n_channels
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	for (uint32_t ch = 0; ch < n_channels; ++ch) {
		dst[i * dst_stride + ch] = src[i * src_stride + ch];
	}
}

template <typename T>
__global__ void copy_gradients_to_density_output(
	const uint32_t n_elements,
	const uint32_t src_stride,
	const uint32_t dst_stride,
	const T* __restrict__ src,
	T* __restrict__ dst,
	const uint32_t n_channels
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	for (uint32_t ch = 0; ch < n_channels; ++ch) {
		dst[i * dst_stride + ch] = src[i * src_stride + ch];
	}
}

template <typename T>
__global__ void copy_features_to_rgb_input(
	const uint32_t n_elements,
	const float* __restrict__ features,
	T* __restrict__ rgb_input,
	const uint32_t rgb_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Copy 4D features to first 4 channels of RGB input
	for (uint32_t ch = 0; ch < 4; ++ch) {
		rgb_input[i * rgb_stride + ch] = T(features[i * 4 + ch]);
	}
}

template <typename T>
class BaselineAggregateNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	BaselineAggregateNetwork(
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
	                       "baseline_aggregate", use_sdf) {
		
		// Configure density network output: 8D = [density, diffuse_RGB(3), features(4)]
		json local_density_network_config = density_network;
		local_density_network_config["n_output_dims"] = 8;
		local_density_network_config["n_input_dims"] = this->m_pos_encoding->padded_output_width();
		this->m_density_network.reset(tcnn::create_network<T>(local_density_network_config));
		
		// RGB network input: 4D features + direction encoding (16D) = 20D -> padded to 32D
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		this->m_rgb_network_input_width = tcnn::next_multiple(
			4 + this->m_dir_encoding->padded_output_width(),
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

	// Override to return 8D output width instead of standard 4D
	uint32_t padded_output_width() const override {
		return 8;
	}

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
		
		tcnn::GPUMatrixDynamic<T> density_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream,
			tcnn::AoS
		};
		
		// Forward: position encoding -> density network -> 8D output
		this->m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);
		
		this->m_density_network->inference_mixed_precision(
			stream, density_network_input, density_network_output, use_inference_params
		);
		
		// Copy all 8 channels to output: [density, diffuse_RGB(3), features(4)]
		linear_kernel(copy_density_network_output<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? this->padded_output_width() : 1,
			density_network_output.data(),
			output.data(),
			8  // Copy all 8 channels
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
		auto forward = std::make_unique<ForwardContext>();
		
		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{
			this->m_density_network->padded_output_width(), batch_size, stream,
			tcnn::AoS
		};
		
		forward->pos_encoding_ctx = this->m_pos_encoding->forward(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);
		
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output,
			use_inference_params, false
		);
		
		if (output) {
			// Copy all 8D output
			linear_kernel(copy_density_network_output<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
				output->layout() == tcnn::AoS ? this->padded_output_width() : 1,
				forward->density_network_output.data(),
				output->data(),
				8
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
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		uint32_t batch_size = input.n();
		
		// Gradients for 8D output: [density, diffuse_RGB(3), features(4)]
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream,
			tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0,
		                                  dL_ddensity_network_output.n_bytes(), stream));
		
		// Copy gradients from output (all 8 channels)
		linear_kernel(copy_gradients_to_density_output<T>, 0, stream,
			batch_size,
			dL_doutput.layout() == tcnn::AoS ? this->padded_output_width() : 1,
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1,
			dL_doutput.data(),
			dL_ddensity_network_output.data(),
			8  // All 8 channels
		);
		
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (this->m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{
				this->m_pos_encoding->padded_output_width(), batch_size, stream,
				this->m_pos_encoding->preferred_output_layout()
			};
		}
		
		this->m_density_network->backward(
			stream, *forward.density_network_ctx, forward.density_network_input,
			forward.density_network_output, dL_ddensity_network_output,
			dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr,
			use_inference_params, param_gradients_mode
		);
		
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, this->m_pos_encoding->input_width());
			}
			
			this->m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, this->m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	// Per-pixel RGB MLP evaluation (for post-accumulation)
	// This is called after volume rendering to process accumulated features
	void apply_directional_mlp(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& accumulated_features,  // [4D features per pixel]
		const tcnn::GPUMatrixDynamic<float>& view_directions,        // [3D view direction per pixel]
		tcnn::GPUMatrixDynamic<T>& directional_rgb,                  // [3D RGB output per pixel]
		bool use_inference_params = true
	) {
		uint32_t n_pixels = accumulated_features.n();
		
		// Allocate RGB network input buffer
		tcnn::GPUMatrixDynamic<T> rgb_network_input{
			this->m_rgb_network_input_width, n_pixels, stream,
			this->m_dir_encoding->preferred_output_layout()
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0,
		                                  rgb_network_input.n_bytes(), stream));
		
		// Copy features to RGB input [0:4]
		linear_kernel(copy_features_to_rgb_input<T>, 0, stream,
			n_pixels,
			accumulated_features.data(),
			rgb_network_input.data(),
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1
		);
		
		// Encode view directions and copy to RGB input [4:20]
		auto dir_out = rgb_network_input.slice_rows(4, this->m_dir_encoding->padded_output_width());
		this->m_dir_encoding->inference_mixed_precision(
			stream,
			view_directions,
			dir_out,
			use_inference_params
		);
		
		// Run RGB MLP
		this->m_rgb_network->inference_mixed_precision(
			stream, rgb_network_input, directional_rgb, use_inference_params
		);
	}

private:
	struct ForwardContext : public NerfNetworkBase<T>::ForwardContextBase {
		// baseline_aggregate uses base context without extensions
	};
};

} // namespace ngp

