/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   baseline_network.h
 *  @brief  Standard NeRF baseline network implementation
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

template <typename T>
class BaselineNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	BaselineNetwork(
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
	                       "baseline", use_sdf) {
		
		// Configure density network output
		json local_density_network_config = density_network;
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = use_sdf ? 16 : 16;
		}
		local_density_network_config["n_input_dims"] = this->m_pos_encoding->padded_output_width();
		this->m_density_network.reset(tcnn::create_network<T>(local_density_network_config));
		
		// RGB network input: density output + direction encoding
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		this->m_rgb_network_input_width = tcnn::next_multiple(
			this->m_dir_encoding->padded_output_width() + 
			std::max(16u, this->m_density_network->padded_output_width()),
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
		
		tcnn::GPUMatrixDynamic<T> density_network_input{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		tcnn::GPUMatrixDynamic<T> rgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream,
			this->m_dir_encoding->preferred_output_layout()
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0, rgb_network_input.n_bytes(), stream));
		
		auto density_network_output = rgb_network_input.slice_rows(0, this->m_density_network->padded_output_width());
		tcnn::GPUMatrixDynamic<T> rgb_network_output{
			output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};
		
		// Forward: position encoding -> density network -> density output
		this->m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);
		
		this->m_density_network->inference_mixed_precision(
			stream, density_network_input, density_network_output, use_inference_params
		);
		
		// Direction encoding
		auto dir_out = rgb_network_input.slice_rows(
			this->m_density_network->padded_output_width(),
			this->m_dir_encoding->padded_output_width()
		);
		
		this->m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);
		
		// RGB network
		this->m_rgb_network->inference_mixed_precision(
			stream, rgb_network_input, rgb_network_output, use_inference_params
		);
		
		// Extract density to output
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? this->padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr,
			false  // MLP already has activation
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
		
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{
			this->m_rgb_network_input_width, batch_size, stream,
			this->m_dir_encoding->preferred_output_layout()
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(forward->rgb_network_input.data(), 0,
		                                  forward->rgb_network_input.n_bytes(), stream));
		
		forward->pos_encoding_ctx = this->m_pos_encoding->forward(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);
		
		// Density network output is a slice of RGB input
		forward->density_network_output = forward->rgb_network_input.slice_rows(
			0, this->m_density_network->padded_output_width()
		);
		
		auto dir_out = forward->rgb_network_input.slice_rows(
			this->m_density_network->padded_output_width(),
			this->m_dir_encoding->padded_output_width()
		);
		
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output,
			use_inference_params, false
		);
		
		forward->dir_encoding_ctx = this->m_dir_encoding->forward(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);
		
		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{
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
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		uint32_t batch_size = input.n();
		
		tcnn::GPUMatrix<T> dL_drgb{this->m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		
		linear_kernel(extract_rgb<T>, 0, stream,
			batch_size * 3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);
		
		const tcnn::GPUMatrixDynamic<T> rgb_network_output{
			(T*)output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};
		
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream,
			this->m_dir_encoding->preferred_output_layout()
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0,
		                                  dL_drgb_network_input.n_bytes(), stream));
		
		this->m_rgb_network->backward(
			stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output,
			dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode
		);
		
		// Backprop through dir encoding
		if (this->m_dir_encoding->n_params() > 0 || dL_dinput) {
			auto dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(
				this->m_density_network->padded_output_width(),
				this->m_dir_encoding->padded_output_width()
			);
			
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(
					this->m_dir_offset, this->m_dir_encoding->input_width()
				);
			}
			
			auto dir_encoding_forward_output = forward.rgb_network_input.slice_rows(
				this->m_density_network->padded_output_width(),
				this->m_dir_encoding->padded_output_width()
			);
			
			this->m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
				dir_encoding_forward_output,
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
		
		// Gradient from density output
		auto dL_ddensity_network_output = dL_drgb_network_input.slice_rows(
			0, this->m_density_network->padded_output_width()
		);
		
		// Add density gradient from final output
		linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
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

private:
	struct ForwardContext : public NerfNetworkBase<T>::ForwardContextBase {
		// Baseline uses base context without extensions
	};
};

} // namespace ngp



