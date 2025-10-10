/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   baseline_explicit_network.h
 *  @brief  Baseline rendering with explicit learnable density grid
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

template <typename T>
class BaselineExplicitNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	BaselineExplicitNetwork(
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
	                       "baseline_explicit", use_sdf) {
		
		// Configure density network: 15D output (features for RGB input[1-15])
		json local_density_network_config = density_network;
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 15;
		}
		local_density_network_config["n_input_dims"] = this->m_pos_encoding->padded_output_width();
		this->m_density_network.reset(tcnn::create_network<T>(local_density_network_config));
		
		// Initialize explicit density grid
		uint32_t grid_res = 128;
		if (density_network.contains("explicit_grid_resolution")) {
			grid_res = density_network["explicit_grid_resolution"];
		}
		
		json dense_grid_config = {
			{"otype", "DenseGrid"},
			{"n_levels", 1},
			{"n_features_per_level", 1},
			{"base_resolution", grid_res},
			{"per_level_scale", 1.0f},
			{"interpolation", "Linear"}
		};
		
		this->m_density_grid.reset(tcnn::create_encoding<T>(3, dense_grid_config, 1));
		
		// Initialize grid to low density
		std::vector<T> init_params(this->m_density_grid->n_params());
		pcg32 rng(42);
		for (size_t i = 0; i < init_params.size(); ++i) {
			init_params[i] = T(-5.0f + (rng.next_float() - 0.5f));
		}
		this->m_density_grid->set_params(init_params.data(), init_params.data(), init_params.data());
		
		// RGB network input: density + 15 features + direction encoding
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
		
		// Check for separate density grid optimizer config
		if (density_network.contains("density_grid_optimizer")) {
			m_density_grid_optimizer_config = density_network["density_grid_optimizer"];
			m_use_separate_grid_optimizer = true;
		} else {
			m_use_separate_grid_optimizer = false;
		}
	}
	
	// Keep all params in main trainer even with separate optimizer
	// The grid will just use a different learning rate via the separate trainer
	size_t n_params() const override {
		return NerfNetworkBase<T>::n_params();
	}
	
	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		// Always use base implementation - grid is part of main param buffer
		NerfNetworkBase<T>::set_params_impl(params, inference_params, gradients);
	}
	
	// Methods for separate density grid parameter management
	bool uses_separate_grid_optimizer() const { return m_use_separate_grid_optimizer; }
	
	size_t n_grid_params() const { return this->m_density_grid ? this->m_density_grid->n_params() : 0; }
	
	void set_grid_params(T* params, T* inference_params, T* gradients) {
		if (this->m_density_grid) {
			this->m_density_grid->set_params(params, inference_params, gradients);
		}
	}
	
	json grid_optimizer_config() const { return m_density_grid_optimizer_config; }

	void inference_mixed_precision_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		// baseline_explicit: density from grid, features from MLP (no surface computations)
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
		
		// Position encoding
		this->m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);
		
		// Step 1: Get density from grid
		tcnn::GPUMatrixDynamic<T> grid_density_explicit{
			this->m_density_grid->padded_output_width(), batch_size, stream,
			this->m_density_grid->preferred_output_layout()
		};
		auto grid_ctx = this->m_density_grid->forward(
			stream,
			input.slice_rows(0, 3),  // positions
			&grid_density_explicit,
			use_inference_params,
			false  // don't need input gradients
		);
		
		// Step 2: MLP forward for 15D features
		this->m_density_network->inference_mixed_precision(
			stream, density_network_input, density_network_output, use_inference_params
		);
		
		// Step 3: Copy grid density to RGB input[0] (NO activation - grid stores log-density)
		linear_kernel(replace_first_channel_kernel<T>, 0, stream,
			batch_size,
			grid_density_explicit.data(),
			grid_density_explicit.layout() == tcnn::AoS ? grid_density_explicit.stride() : 1,
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1,
			rgb_network_input.data(),
			false  // NO activation: grid stores log-density, renderer applies exp()
		);
		
		// Step 4: Copy MLP features[0-14] to RGB input[1-15]
		linear_kernel(copy_channels_kernel<T>, 0, stream,
			batch_size,
			15,  // number of channels to copy
			density_network_output.data(),
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			rgb_network_input.data(),
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1,
			1  // destination offset (skip channel 0)
		);
		
		// Step 5: Direction encoding
		auto dir_out = rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
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
		
		// Extract density to output[3] (from grid) - NO activation, grid stores log-density
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			grid_density_explicit.layout() == tcnn::AoS ? grid_density_explicit.stride() : 1,
			output.layout() == tcnn::AoS ? this->padded_output_width() : 1,
			grid_density_explicit.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr,
			false  // NO activation: grid stores log-density, renderer applies exp()
		);
	}

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		// baseline_explicit: density from grid + 15D features from MLP (no surface computation)
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
		
		// Step 1: Get density from grid
		forward->grid_density = tcnn::GPUMatrixDynamic<T>{
			this->m_density_grid->padded_output_width(), batch_size, stream,
			this->m_density_grid->preferred_output_layout()
		};
		forward->density_grid_ctx = this->m_density_grid->forward(
			stream,
			input.slice_rows(0, 3),  // positions only
			&forward->grid_density,
			use_inference_params,
			false  // don't need input gradients
		);
		
		// Step 2: MLP forward pass (15D features)
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output, 
			use_inference_params, false
		);
		
		// Step 3: Copy grid density to RGB input[0] (apply exp() since grid stores log-density)
		uint32_t grid_src_stride = forward->grid_density.layout() == tcnn::AoS ? forward->grid_density.stride() : 1;
		uint32_t rgb_dst_stride = forward->rgb_network_input.layout() == tcnn::AoS ? forward->rgb_network_input.stride() : 1;
		
		linear_kernel(replace_first_channel_kernel<T>, 0, stream,
			batch_size,
			forward->grid_density.data(),
			grid_src_stride,
			rgb_dst_stride,
			forward->rgb_network_input.data(),
			false  // NO activation: grid stores log-density, renderer applies exp()
		);
		
		// Step 4: Copy MLP features[0-14] to RGB input[1-15]
		uint32_t mlp_src_stride = forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1;
		
		linear_kernel(copy_channels_kernel<T>, 0, stream,
			batch_size,
			15,  // number of channels to copy
			forward->density_network_output.data(),
			mlp_src_stride,
			forward->rgb_network_input.data(),
			rgb_dst_stride,
			1  // destination offset
		);
		
		// Step 5: Direction encoding
		auto dir_out = forward->rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
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
			// Extract density to output[3] (from grid) - NO activation, grid stores log-density
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				forward->grid_density.layout() == tcnn::AoS ? forward->grid_density.stride() : 1,
				output->layout() == tcnn::AoS ? this->padded_output_width() : 1,
				forward->grid_density.data(),
				output->data() + 3 * (output->layout() == tcnn::AoS ? 1 : batch_size),
				this->m_use_sdf,
				this->m_variance_network ? this->m_variance_network->params() : nullptr,
				false  // NO activation: grid stores log-density, renderer applies exp()
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
		// baseline_explicit: backprop to both grid and MLP (simple feature copy, no surface features)
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		uint32_t batch_size = input.n();
		
		// Extract RGB gradients with proper padding (match old nerf_network.h implementation)
		tcnn::GPUMatrix<T> dL_drgb{this->m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		
		linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);
		
		// Standard RGB network backward
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0,
		                                  dL_drgb_network_input.n_bytes(), stream));
		
		const tcnn::GPUMatrixDynamic<T> rgb_network_output{
			(T*)output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};
		
		this->m_rgb_network->backward(
			stream, *forward.rgb_network_ctx, forward.rgb_network_input,
			rgb_network_output, dL_drgb,
			&dL_drgb_network_input, use_inference_params, param_gradients_mode
		);
		
		// Direction encoding backward
		auto dL_ddir_out = dL_drgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		auto dir_out_forward = forward.rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		
		tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
		if (dL_dinput) {
			dL_ddir_encoding_input = dL_dinput->slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width());
		}
		
		this->m_dir_encoding->backward(
			stream, *forward.dir_encoding_ctx,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			dir_out_forward,
			dL_ddir_out, dL_dinput ? &dL_ddir_encoding_input : nullptr,
			use_inference_params, param_gradients_mode
		);
		
		// Step 1: Extract gradients for MLP features from RGB input[1-15]
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0,
		                                  dL_ddensity_network_output.n_bytes(), stream));
		
		uint32_t rgb_src_stride = dL_drgb_network_input.layout() == tcnn::AoS ? dL_drgb_network_input.stride() : 1;
		uint32_t mlp_dst_stride = dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1;
		
		linear_kernel(copy_channels_with_src_offset_kernel<T>, 0, stream,
			batch_size,
			15,  // number of channels to copy
			dL_drgb_network_input.data(),
			rgb_src_stride,
			1,  // source offset (skip channel 0, read from RGB[1-15])
			dL_ddensity_network_output.data(),
			mlp_dst_stride
		);
		
		// Step 2: Extract gradient for grid density from RGB input[0]
		tcnn::GPUMatrixDynamic<T> dL_dgrid_density{
			this->m_density_grid->padded_output_width(), batch_size, stream,
			this->m_density_grid->preferred_output_layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dgrid_density.data(), 0, dL_dgrid_density.n_bytes(), stream));
		
		uint32_t grid_dst_stride = dL_dgrid_density.layout() == tcnn::AoS ? dL_dgrid_density.stride() : 1;
		
		// DISABLED: Do not backprop gradient from RGB input[0] to grid
		// linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
		// 	batch_size,
		// 	dL_drgb_network_input.data(),
		// 	rgb_src_stride,
		// 	grid_dst_stride,
		// 	dL_dgrid_density.data(),
		// 	nullptr,  // No forward values needed - no activation to chain through
		// 	false  // NO activation: pass gradients through unchanged
		// );
		
		// Step 2b: ACCUMULATE density gradient from RGBD output (channel 3) to grid
		uint32_t rgbd_src_stride = dL_doutput.layout() == tcnn::AoS ? dL_doutput.m() : 1;
		
		linear_kernel(accumulate_density_gradient_to_grid_kernel<T>, 0, stream,
			batch_size,
			dL_doutput.data(),
			rgbd_src_stride,  // For AoS: stride=4, for SoA: stride=1
			dL_dgrid_density.data(),
			grid_dst_stride,
			dL_doutput.m(),  // Number of rows (channels) = 4 for RGBD
			nullptr,  // No forward values needed - no activation to chain through
			false  // NO activation: pass gradients through unchanged
		);
		
		// Step 3: Backprop to MLP
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
		
		// Step 4: Backprop to grid
		if (this->m_density_grid->n_params() > 0) {
			this->m_density_grid->backward(
				stream, *forward.density_grid_ctx, 
				input.slice_rows(0, 3),
				forward.grid_density,
				dL_dgrid_density,
				nullptr,  // Don't need input gradients
				use_inference_params,
				param_gradients_mode
			);
		}
		
		// Step 5: Backprop through hash encoding
		if (dL_ddensity_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, this->m_pos_encoding->input_width());
			}
			
			this->m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, this->m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

private:
	struct ForwardContext : public NerfNetworkBase<T>::ForwardContextBase {
		tcnn::GPUMatrixDynamic<T> grid_density;
		std::unique_ptr<tcnn::Context> density_grid_ctx;
	};
	
	// Separate optimizer support for density grid
	bool m_use_separate_grid_optimizer = false;
	json m_density_grid_optimizer_config;
};

} // namespace ngp

