/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   volume_network.h
 *  @brief  Volume rendering with divergence-based features
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

template <typename T>
class VolumeNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	VolumeNetwork(
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
	                       "volume", use_sdf) {
		
		// Configure density network: 48D output (1D density + 45D Φ features for 15 x 3D vector fields)
		json local_density_network_config = density_network;
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 48;
		}
		local_density_network_config["n_input_dims"] = this->m_pos_encoding->padded_output_width();
		this->m_density_network.reset(tcnn::create_network<T>(local_density_network_config));
		
		// RGB network input: 16 divergence features + direction encoding
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		this->m_rgb_network_input_width = tcnn::next_multiple(
			16 + this->m_dir_encoding->padded_output_width(),
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
		// Volume mode: Compute divergences from 15 vector fields
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
		
		// Compute volume divergences for inference
		tcnn::GPUMatrixDynamic<float> volume_divergences = compute_volume_divergences_inference(
			stream, batch_size, input, density_network_input, density_network_output, use_inference_params
		);
		
		// Compute volume features using divergences (all 16 channels in one kernel)
		linear_kernel(compute_volume_divergence_kernel<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			density_network_output.data(),
			volume_divergences.data(),  // Pass volume divergences
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1,
			rgb_network_input.data()
		);
		
		// Direction encoding goes after the 16D features
		auto dir_out = rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		
		// Encode view directions
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
		
		// Extract density (from channel 0 of density network output)
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
		// Volume mode: Use same density network approach but with divergence computation
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
		
		// CRITICAL: For volume mode, use AoS layout for density buffer to ensure copy compatibility
		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		// CRITICAL: Enable gradient computation for divergence computation
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output,
			use_inference_params, true
		);
		
		// Compute volume divergences using the same pattern as analytical normals (no gradient flow to parameters)
		forward->volume_divergences = compute_volume_divergences_forward(
			stream, batch_size, input, forward, use_inference_params
		);
		
		// Compute volume features directly into RGB slice using divergences
		auto volume_features_slice = forward->rgb_network_input.slice_rows(0, 16);
		
		// Compute volume features using divergences
		linear_kernel(compute_volume_divergence_kernel<T>, 0, stream,
			batch_size,
			forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
			forward->density_network_output.data(),
			forward->volume_divergences.data(),  // Pass volume divergences
			volume_features_slice.layout() == tcnn::AoS ? volume_features_slice.stride() : 1,
			volume_features_slice.data()
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
			// Extract density (from channel 0 of density network output)
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
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		uint32_t batch_size = input.n();
		
		// Standard RGB network backward
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0,
		                                  dL_drgb_network_input.n_bytes(), stream));
		
		this->m_rgb_network->backward(
			stream, *forward.rgb_network_ctx, forward.rgb_network_input,
			forward.rgb_network_output, dL_doutput.slice_rows(0, 3),
			&dL_drgb_network_input, use_inference_params, param_gradients_mode
		);
		
		// Direction encoding backward
		auto dL_ddir_out = dL_drgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
		if (dL_dinput) {
			dL_ddir_encoding_input = dL_dinput->slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width());
		}
		
		this->m_dir_encoding->backward(
			stream, *forward.dir_encoding_ctx,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			dL_ddir_out.slice_rows(0, this->m_dir_encoding->padded_output_width()),
			dL_ddir_out, dL_dinput ? &dL_ddir_encoding_input : nullptr,
			use_inference_params, param_gradients_mode
		);
		
		// Backward pass: gradients from RGB slice back to 48D density output using VOLUME DIVERGENCES
		auto dL_dvolume_slice = dL_drgb_network_input.slice_rows(0, 16);
		
		// Density network output gradients
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0,
		                                  dL_ddensity_network_output.n_bytes(), stream));
		
		// Compute gradients w.r.t. divergences from volume features
		tcnn::GPUMatrixDynamic<float> dL_ddivergences{
			15, batch_size, stream, forward.volume_divergences.layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddivergences.data(), 0, dL_ddivergences.n_bytes(), stream));
		
		// Compute gradients from volume features to density output and divergences
		linear_kernel(volume_divergence_backward_kernel<T>, 0, stream,
			batch_size,
			dL_dvolume_slice.layout() == tcnn::AoS ? dL_dvolume_slice.stride() : 1,
			dL_dvolume_slice.data(),
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1,
			dL_ddensity_network_output.data(),
			dL_ddivergences.data()  // Collect gradients w.r.t. divergences
		);
		
		// Backpropagate gradients from divergences back to density network parameters
		accumulate_volume_divergence_gradients(
			stream, batch_size, input, forward, dL_ddivergences,
			dL_ddensity_network_output, use_inference_params, param_gradients_mode
		);
		
		// Accumulate density gradient from output[3]
		linear_kernel(add_density_gradient_from_output<T>, 0, stream,
			batch_size,
			dL_doutput.layout() == tcnn::AoS ? dL_doutput.m() : 1,
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1,
			dL_ddensity_network_output.data()
		);
		
		// Density network backward
		tcnn::GPUMatrixDynamic<T> dL_ddensity_input;
		if (this->m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_input = tcnn::GPUMatrixDynamic<T>{
				this->m_pos_encoding->padded_output_width(), batch_size, stream,
				this->m_pos_encoding->preferred_output_layout()
			};
		}
		
		this->m_density_network->backward(
			stream, *forward.density_network_ctx, forward.density_network_input,
			forward.density_network_output, dL_ddensity_network_output,
			dL_ddensity_input.data() ? &dL_ddensity_input : nullptr,
			use_inference_params, param_gradients_mode
		);
		
		// Position encoding backward
		if (dL_ddensity_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, this->m_pos_encoding->input_width());
			}
			
			this->m_pos_encoding->backward(
				stream, *forward.pos_encoding_ctx,
				input.slice_rows(0, this->m_pos_encoding->input_width()),
				forward.density_network_input, dL_ddensity_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params, param_gradients_mode
			);
		}
	}

private:
	struct ForwardContext : public NerfNetworkBase<T>::ForwardContextBase {
		tcnn::GPUMatrixDynamic<float> volume_divergences;  // 15D divergence values per sample
	};

	// Helper methods for divergence computation (see lines 1884-2027 in original)
	tcnn::GPUMatrixDynamic<float> compute_volume_divergences_inference(
		cudaStream_t stream,
		uint32_t batch_size,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& density_network_input,
		const tcnn::GPUMatrixDynamic<T>& density_network_output,
		bool use_inference_params
	);

	tcnn::GPUMatrixDynamic<float> compute_volume_divergences_forward(
		cudaStream_t stream,
		uint32_t batch_size,
		const tcnn::GPUMatrixDynamic<float>& input,
		std::unique_ptr<ForwardContext>& forward,
		bool use_inference_params
	);
};

} // namespace ngp

