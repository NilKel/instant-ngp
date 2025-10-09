/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   surface_explicit_network.h
 *  @brief  Surface rendering with explicit learnable density grid
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

template <typename T>
class SurfaceExplicitNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	SurfaceExplicitNetwork(
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
	                       "surface_explicit", use_sdf) {
		
		// Configure density network: 48D output (1 dummy + 45D Φ features for 15 x 3D vectors)
		json local_density_network_config = density_network;
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 48;
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
		
		// RGB network input: 16 surface features + direction encoding
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
		// surface_explicit: density from grid + features from MLP + normals from grid gradients
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
		
		// Step 1: Get density from grid (need forward context for normals)
		// Use grid's preferred layout (SoA) to match hash encoding
		tcnn::GPUMatrixDynamic<T> grid_density_explicit{
			this->m_density_grid->padded_output_width(), batch_size, stream,
			this->m_density_grid->preferred_output_layout()
		};
		auto grid_ctx = this->m_density_grid->forward(
			stream,
			input.slice_rows(0, 3),  // positions
			&grid_density_explicit,
			use_inference_params,
			true  // prepare_input_gradients = true for normal computation
		);
		
		// Step 2: Compute normals from grid gradients using autodiff
		tcnn::GPUMatrixDynamic<float> normals = compute_normals_from_grid_gradients(
			stream, batch_size, input.slice_rows(0, 3), this->m_density_grid,
			*grid_ctx, grid_density_explicit, use_inference_params
		);
		
		// Step 3: MLP forward for features
		this->m_density_network->inference_mixed_precision(
			stream, density_network_input, density_network_output, use_inference_params
		);
		
		// Step 4: Compute surface features
		// MLP outputs 45D vectors (15 x 3D) starting from channel 0
		auto surface_features_slice = rgb_network_input.slice_rows(1, 15);
		linear_kernel(compute_surface_features_from_vectors_kernel<T>, 0, stream,
			batch_size,
			density_network_output.data(),  // Use channels 0-44 (45 features for 15 3D vectors)
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			normals.data(),
			normals.layout() == tcnn::AoS ? 3 : 1,
			surface_features_slice.data(),
			surface_features_slice.layout() == tcnn::AoS ? surface_features_slice.stride() : 1
		);
		
		// Step 5: Replace channel 0 with grid density (extract from padded grid output)
		linear_kernel(replace_first_channel_kernel<T>, 0, stream,
			batch_size,
			grid_density_explicit.data(),
			grid_density_explicit.layout() == tcnn::AoS ? grid_density_explicit.stride() : 1,  // grid stride (padded)
			rgb_network_input.layout() == tcnn::AoS ? rgb_network_input.stride() : 1,  // output stride
			rgb_network_input.data()
		);
		
		// Step 6: Direction encoding (same as surface mode)
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
		
		// Extract density to output[3] (from grid)
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			grid_density_explicit.layout() == tcnn::AoS ? grid_density_explicit.stride() : 1,
			output.layout() == tcnn::AoS ? this->padded_output_width() : 1,
			grid_density_explicit.data(),
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
		// surface_explicit: density from grid + features from MLP
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
		
		// Step 1: Get density from grid (with gradients for normal computation)
		// Use grid's preferred layout (SoA) to match hash encoding
		forward->grid_density = tcnn::GPUMatrixDynamic<T>{
			this->m_density_grid->padded_output_width(), batch_size, stream,
			this->m_density_grid->preferred_output_layout()
		};
		forward->density_grid_ctx = this->m_density_grid->forward(
			stream,
			input.slice_rows(0, 3),  // positions only
			&forward->grid_density,
			use_inference_params,
			true  // prepare_input_gradients for normal computation
		);
		
		// Step 2: MLP forward pass (48D output: 1 dummy + 45 features for 15 vectors)
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output, 
			use_inference_params, false
		);
		
		// Step 3: Compute normals from grid gradients using autodiff
		forward->analytical_normals = compute_normals_from_grid_gradients(
			stream, batch_size, input.slice_rows(0, 3), this->m_density_grid,
			*forward->density_grid_ctx, forward->grid_density, use_inference_params,
			forward.get()  // Pass context to enable gradient saving for finite diff
		);
		
		// Step 4: Compute 15 surface features using kernel
		// MLP outputs 45D vectors (15 x 3D) starting from channel 0
		auto surface_features_slice = forward->rgb_network_input.slice_rows(1, 15);
		linear_kernel(compute_surface_features_from_vectors_kernel<T>, 0, stream,
			batch_size,
			forward->density_network_output.data(),  // Use channels 0-44 (45 features for 15 3D vectors)
			forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
			forward->analytical_normals.data(),
			forward->analytical_normals.layout() == tcnn::AoS ? 3 : 1,
			surface_features_slice.data(),
			surface_features_slice.layout() == tcnn::AoS ? surface_features_slice.stride() : 1
		);
		
		// Step 5: Replace channel 0 with grid density (extract from padded grid output)
		linear_kernel(replace_first_channel_kernel<T>, 0, stream,
			batch_size,
			forward->grid_density.data(),
			forward->grid_density.layout() == tcnn::AoS ? forward->grid_density.stride() : 1,  // grid stride (padded)
			forward->rgb_network_input.layout() == tcnn::AoS ? forward->rgb_network_input.stride() : 1,  // output stride
			forward->rgb_network_input.data()
		);
		
		// Step 6: Direction encoding (same as surface mode)
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
			// Extract density to output[3] (from grid)
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				forward->grid_density.layout() == tcnn::AoS ? forward->grid_density.stride() : 1,
				output->layout() == tcnn::AoS ? this->padded_output_width() : 1,
				forward->grid_density.data(),
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
		// surface_explicit: backprop to both grid and MLP
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
		
		// Step 1: Backprop through surface features to get gradients for 45-D MLP output
		auto dL_dsurface_slice = dL_drgb_network_input.slice_rows(1, 15);
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0,
		                                  dL_ddensity_network_output.n_bytes(), stream));
		
		linear_kernel(backprop_surface_features_kernel<T>, 0, stream,
			batch_size,
			dL_dsurface_slice.data(),
			dL_dsurface_slice.layout() == tcnn::AoS ? dL_dsurface_slice.stride() : 1,
			forward.analytical_normals.data(),
			forward.analytical_normals.layout() == tcnn::AoS ? 3 : 1,
			dL_ddensity_network_output.data(),  // Output to features (skip channel 0)
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1
		);
		
		// Step 2: Extract gradient for channel 0 (goes to grid) - allocate with padding
		// Use grid's preferred layout (SoA) to match hash encoding
		tcnn::GPUMatrixDynamic<T> dL_dgrid_density{
			this->m_density_grid->padded_output_width(), batch_size, stream,
			this->m_density_grid->preferred_output_layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dgrid_density.data(), 0, dL_dgrid_density.n_bytes(), stream));
		
		// Step 2a: Extract gradient from RGB network input (channel 0)
		linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
			batch_size,
			dL_drgb_network_input.data(),
			dL_drgb_network_input.layout() == tcnn::AoS ? dL_drgb_network_input.stride() : 1,
			dL_dgrid_density.layout() == tcnn::AoS ? dL_dgrid_density.stride() : 1,  // grid stride
			dL_dgrid_density.data(),
			forward.grid_density.data(),  // Forward grid values (not used when apply_exp=false)
			false  // Don't apply exp() - gradients already w.r.t. log-density from fused kernels
		);
		
		// Step 2b: ACCUMULATE density gradient from RGBD output (channel 3) to grid
		// This is the gradient from the rendering loss w.r.t. density
		linear_kernel(accumulate_density_gradient_to_grid_kernel<T>, 0, stream,
			batch_size,
			dL_doutput.data(),
			dL_doutput.layout() == tcnn::AoS ? dL_doutput.m() : 1,  // RGBD output stride
			dL_dgrid_density.data(),
			dL_dgrid_density.layout() == tcnn::AoS ? dL_dgrid_density.stride() : 1,  // grid stride
			dL_doutput.m(),  // Number of rows (channels) = 4 for RGBD
			forward.grid_density.data(),  // Forward grid values (not used when apply_exp=false)
			false  // Don't apply exp() - gradients already w.r.t. log-density from fused kernels
		);
		
		// Step 2.5: Backprop gradients from normals to grid (for finite differences)
		// This computes dL/dnormals from the surface features and backprops through finite diff
		tcnn::GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, forward.analytical_normals.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals.data(), 0, dL_dnormals.n_bytes(), stream));
		
		linear_kernel(backprop_surface_features_to_normals_kernel<T>, 0, stream,
			batch_size,
			dL_dsurface_slice.data(),
			dL_dsurface_slice.layout() == tcnn::AoS ? dL_dsurface_slice.stride() : 1,
			forward.density_network_output.data(),
			forward.density_network_output.layout() == tcnn::AoS ? forward.density_network_output.stride() : 1,
			dL_dnormals.data(),
			dL_dnormals.layout() == tcnn::AoS ? 3 : 1
		);
		
		// Check if we're using finite differences and have saved contexts
		const char* grad_method_env = std::getenv("NGP_GRAD_METHOD");
		bool use_finite_diff = (grad_method_env && std::string(grad_method_env) == "finite");
		
		if (use_finite_diff && !forward.finite_diff_contexts.empty()) {
			backprop_normals_from_finite_differences(
				stream, batch_size, dL_dnormals, forward.analytical_normals,
				this->m_density_grid, forward, use_inference_params, param_gradients_mode
			);
		}
		
		// NOTE: For surface_explicit, we do NOT call accumulate_analytical_normal_gradients()
		// because the normals come from the GRID, not from the MLP density output.
		// The gradient flow is:
		//   - dL/dnormals → grid (via backprop_normals_from_finite_differences or grid->backward)
		//   - dL/dnormals → MLP vectors (already handled by backprop_surface_features_to_normals_kernel)
		// The MLP does NOT output density (channel 0), only feature vectors (channels 1-47).
		
		// Step 3: Backprop to density network parameters
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
		
		// Step 4: Backprop to grid parameters
		// Note: We don't need dL_dinput from grid since we already get it from hash encoding
		if (this->m_density_grid->n_params() > 0) {
			this->m_density_grid->backward(
				stream, *forward.density_grid_ctx, 
				input.slice_rows(0, 3),  // positions
				forward.grid_density,  // grid output from forward pass
				dL_dgrid_density,  // gradients from channel 0
				nullptr,  // Don't need input gradients - hash encoding handles that
				use_inference_params,
				param_gradients_mode
			);
		}
		
		// Step 5: Backprop through pos encoding
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
		tcnn::GPUMatrixDynamic<float> analytical_normals;
		
		// For finite differences
		std::vector<std::unique_ptr<tcnn::Context>> finite_diff_contexts;
		std::vector<tcnn::GPUMatrixDynamic<T>> finite_diff_densities;
		std::vector<tcnn::GPUMatrixDynamic<float>> finite_diff_positions;
		int finite_diff_genus = 1;
	};
};

} // namespace ngp

