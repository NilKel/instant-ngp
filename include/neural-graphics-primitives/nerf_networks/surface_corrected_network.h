/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   surface_corrected_network.h
 *  @brief  Surface rendering with analytical normals and surface features (corrected version)
 */

#pragma once

#include "nerf_network_base.h"

namespace ngp {

template <typename T>
class SurfaceCorrectedNetwork : public NerfNetworkBase<T> {
public:
	using json = nlohmann::json;

	SurfaceCorrectedNetwork(
		uint32_t n_pos_dims,
		uint32_t n_dir_dims,
		uint32_t n_extra_dims,
		uint32_t dir_offset,
		const json& pos_encoding,
		const json& dir_encoding,
		const json& density_network,
		const json& rgb_network,
		bool use_sdf = false,
		const json& correction_encoding = json::object(),
		const json& correction_network = json::object()
	) : NerfNetworkBase<T>(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
	                       pos_encoding, dir_encoding, density_network, rgb_network,
	                       "surface_corrected", use_sdf) {
		
		// Configure density network: 48D output (1D density + 45D Φ features for 15 x 3D vectors)
		json local_density_network_config = density_network;
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 48;
		}
		local_density_network_config["n_input_dims"] = this->m_pos_encoding->padded_output_width();
		this->m_density_network.reset(tcnn::create_network<T>(local_density_network_config));
		
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
		
		// STEP 1: Initialize correction encoding but do not use it
		json local_correction_encoding_config = correction_encoding;
		if (!correction_encoding.contains("n_dims_to_encode")) {
			local_correction_encoding_config["n_dims_to_encode"] = n_pos_dims;
		}
		m_correction_encoding.reset(tcnn::create_encoding<T>(
			n_pos_dims, local_correction_encoding_config, 1
		));
		
		// STEP 2: Initialize correction MLP but do not use it
		uint32_t correction_alignment = tcnn::minimum_alignment(correction_network);
		m_correction_network_input_width = tcnn::next_multiple(
			m_correction_encoding->padded_output_width(),
			correction_alignment
		);

		json local_correction_network_config = correction_network;
		if (!correction_network.contains("n_output_dims")) {
			local_correction_network_config["n_output_dims"] = 3;  // 3D correction vector
		}
		local_correction_network_config["n_input_dims"] = m_correction_network_input_width;
		m_correction_network.reset(tcnn::create_network<T>(local_correction_network_config));
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
		
		// Force AoS layout for surface modes
		tcnn::GPUMatrixDynamic<T> rgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		
		tcnn::GPUMatrixDynamic<T> density_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0, rgb_network_input.n_bytes(), stream));
		
		tcnn::GPUMatrixDynamic<T> rgb_network_output{
			output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};
		
		// Forward: position encoding -> density network
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
		tcnn::GPUMatrixDynamic<float> analytical_normals = compute_analytical_normals_inference(
			stream, batch_size, input, density_network_input, density_network_output, use_inference_params
		);
		
		// Compute surface features using analytical normals
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
		
		// Direction encoding (after surface features)
		auto dir_out = rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		
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
			true  // Always prepare gradients for surface mode (needed for normals)
		);
		
		// Separate buffer for density output in surface mode
		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		auto dir_out = forward->rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
		
		forward->density_network_ctx = this->m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output,
			use_inference_params, true  // Enable gradients for normal computation
		);
		
		// Compute analytical normals using autodiff
		forward->analytical_normals = compute_analytical_normals_forward(
			stream, batch_size, input, forward, use_inference_params
		);
		
		// STEP 4: Correction network forward pass (but don't use output)
		forward->correction_network_input = tcnn::GPUMatrixDynamic<T>{
			m_correction_network_input_width, batch_size, stream, tcnn::AoS
		};
		
		forward->correction_network_output = tcnn::GPUMatrixDynamic<T>{
			m_correction_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(forward->correction_network_input.data(), 0,
		                                  forward->correction_network_input.n_bytes(), stream));
		
		auto correction_encoding_output = forward->correction_network_input.slice_rows(
			0, m_correction_encoding->padded_output_width()
		);
		
		forward->correction_encoding_ctx = m_correction_encoding->forward(
			stream,
			input.slice_rows(0, m_correction_encoding->input_width()),
			&correction_encoding_output,
			use_inference_params,
			prepare_input_gradients
		);
		
		forward->correction_network_ctx = m_correction_network->forward(
			stream,
			forward->correction_network_input,
			&forward->correction_network_output,
			use_inference_params,
			prepare_input_gradients
		);
		
		// DEBUG: Skip correction and use analytical normals directly
		forward->corrected_normals = tcnn::GPUMatrixDynamic<float>{3, batch_size, stream, tcnn::AoS};
		CUDA_CHECK_THROW(cudaMemcpyAsync(forward->corrected_normals.data(), forward->analytical_normals.data(),
		                                  forward->corrected_normals.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		
		// Compute surface features from normals
		auto surface_features_slice = forward->rgb_network_input.slice_rows(0, 16);
		
		linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
			batch_size,
			forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
			forward->density_network_output.data(),
			forward->corrected_normals.data(),  // Use corrected normals (which are analytical normals)
			surface_features_slice.layout() == tcnn::AoS ? surface_features_slice.stride() : 1,
			surface_features_slice.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr
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
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0,
		                                  dL_drgb_network_input.n_bytes(), stream));
		
		this->m_rgb_network->backward(
			stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output,
			dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode
		);
		
		// Backprop through dir encoding
		if (this->m_dir_encoding->n_params() > 0 || dL_dinput) {
			auto dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
			
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width());
			}
			
			auto dir_encoding_forward_output = forward.rgb_network_input.slice_rows(16, this->m_dir_encoding->padded_output_width());
			
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
		
		// Backprop through surface features
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0,
		                                  dL_ddensity_network_output.n_bytes(), stream));
		
		auto dL_dsurface_slice = dL_drgb_network_input.slice_rows(0, 16);
		
		tcnn::GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, forward.analytical_normals.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals.data(), 0, dL_dnormals.n_bytes(), stream));
		
		linear_kernel(surface_features_slice_backward_kernel<T>, 0, stream,
			batch_size,
			dL_dsurface_slice.layout() == tcnn::AoS ? dL_dsurface_slice.stride() : 1,
			dL_dsurface_slice.data(),
			forward.analytical_normals.data(),
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1,
			forward.density_network_output.data(),
			dL_ddensity_network_output.data(),
			dL_dnormals.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr
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

protected:
	struct ForwardContext : public NerfNetworkBase<T>::ForwardContextBase {
		tcnn::GPUMatrixDynamic<float> raw_gradients;
		tcnn::GPUMatrixDynamic<float> analytical_normals;
		
		// STEP 3: Correction network fields (added but not used)
		tcnn::GPUMatrixDynamic<float> corrected_normals;
		tcnn::GPUMatrixDynamic<T> correction_network_input;
		tcnn::GPUMatrixDynamic<T> correction_network_output;
		std::unique_ptr<tcnn::Context> correction_encoding_ctx;
		std::unique_ptr<tcnn::Context> correction_network_ctx;
	};

	// Helper: Compute analytical normals for forward pass
	tcnn::GPUMatrixDynamic<float> compute_analytical_normals_forward(
		cudaStream_t stream,
		uint32_t batch_size,
		const tcnn::GPUMatrixDynamic<float>& input,
		std::unique_ptr<ForwardContext>& forward,
		bool use_inference_params
	) {
		// Create gradient seed for density channel
		tcnn::GPUMatrixDynamic<T> dL_dsdf_seed{
			this->m_density_network->padded_output_width(), batch_size, stream,
			forward->density_network_output.layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf_seed.data(), 0, dL_dsdf_seed.n_bytes(), stream));
		
		linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
			batch_size, T(1.0f),
			dL_dsdf_seed.layout() == tcnn::AoS ? dL_dsdf_seed.stride() : 1,
			dL_dsdf_seed.data()
		);
		
		// Backward through density network
		tcnn::GPUMatrixDynamic<T> dL_ddensity_input{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		this->m_density_network->backward(
			stream,
			*forward->density_network_ctx,
			forward->density_network_input,
			forward->density_network_output,
			dL_dsdf_seed,
			&dL_ddensity_input,
			use_inference_params,
			tcnn::GradientMode::Ignore
		);
		
		// Backward through position encoding
		tcnn::GPUMatrixDynamic<float> dSDF_dpos{this->m_pos_encoding->input_width(), batch_size, stream, tcnn::AoS};
		
		this->m_pos_encoding->backward(
			stream,
			*forward->pos_encoding_ctx,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			forward->density_network_input,
			dL_ddensity_input,
			&dSDF_dpos,
			use_inference_params,
			tcnn::GradientMode::Ignore
		);
		
		// Store raw gradients
		auto raw_grads = dSDF_dpos.slice_rows(0, 3);
		forward->raw_gradients = tcnn::GPUMatrixDynamic<float>{3, batch_size, stream, raw_grads.layout()};
		CUDA_CHECK_THROW(cudaMemcpyAsync(forward->raw_gradients.data(), raw_grads.data(),
		                                  forward->raw_gradients.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		
		// Process gradients (normalize or clamp based on settings)
		tcnn::GPUMatrixDynamic<float> normals{3, batch_size, stream, tcnn::AoS};
		linear_kernel(process_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data(),
			this->m_normalize_normals,
			this->m_clamp_gradients,
			this->m_max_gradient_magnitude
		);
		
		return normals;
	}

	// Helper: Compute analytical normals for inference
	tcnn::GPUMatrixDynamic<float> compute_analytical_normals_inference(
		cudaStream_t stream,
		uint32_t batch_size,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& density_network_input,
		const tcnn::GPUMatrixDynamic<T>& density_network_output,
		bool use_inference_params
	) {
		// Create temporary contexts
		auto temp_pos_ctx = this->m_pos_encoding->forward(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			const_cast<tcnn::GPUMatrixDynamic<T>*>(&density_network_input),
			use_inference_params,
			true
		);
		
		auto temp_density_ctx = this->m_density_network->forward(
			stream,
			density_network_input,
			const_cast<tcnn::GPUMatrixDynamic<T>*>(&density_network_output),
			use_inference_params,
			true
		);
		
		// Gradient seed
		tcnn::GPUMatrixDynamic<T> dL_dsdf_seed{
			this->m_density_network->padded_output_width(), batch_size, stream,
			density_network_output.layout()
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf_seed.data(), 0, dL_dsdf_seed.n_bytes(), stream));
		
		linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
			batch_size, T(1.0f),
			dL_dsdf_seed.layout() == tcnn::AoS ? dL_dsdf_seed.stride() : 1,
			dL_dsdf_seed.data()
		);
		
		tcnn::GPUMatrixDynamic<T> dL_ddensity_input{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};
		
		this->m_density_network->backward(
			stream,
			*temp_density_ctx,
			density_network_input,
			density_network_output,
			dL_dsdf_seed,
			&dL_ddensity_input,
			use_inference_params,
			tcnn::GradientMode::Ignore
		);
		
		tcnn::GPUMatrixDynamic<float> dSDF_dpos{this->m_pos_encoding->input_width(), batch_size, stream, tcnn::AoS};
		
		this->m_pos_encoding->backward(
			stream,
			*temp_pos_ctx,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			density_network_input,
			dL_ddensity_input,
			&dSDF_dpos,
			use_inference_params,
			tcnn::GradientMode::Ignore
		);
		
		tcnn::GPUMatrixDynamic<float> normals{3, batch_size, stream, tcnn::AoS};
		linear_kernel(process_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data(),
			this->m_normalize_normals,
			this->m_clamp_gradients,
			this->m_max_gradient_magnitude
		);
		
		return normals;
	}

	// Parameter management methods
	size_t n_params() const override {
		size_t total_params = 0;
		if (this->m_pos_encoding) total_params += this->m_pos_encoding->n_params();
		if (this->m_density_network) total_params += this->m_density_network->n_params();
		if (this->m_dir_encoding) total_params += this->m_dir_encoding->n_params();
		if (this->m_rgb_network) total_params += this->m_rgb_network->n_params();
		if (m_correction_encoding) total_params += m_correction_encoding->n_params();
		if (m_correction_network) total_params += m_correction_network->n_params();
		return total_params;
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		// Call base class implementation first
		this->NerfNetworkBase<T>::set_params_impl(params, inference_params, gradients);
		
		// Add correction network parameters
		size_t offset = this->NerfNetworkBase<T>::n_params();
		
		if (m_correction_encoding) {
			m_correction_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_correction_encoding->n_params();
		}
		
		if (m_correction_network) {
			m_correction_network->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_correction_network->n_params();
		}
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Call base class implementation first
		this->NerfNetworkBase<T>::initialize_params(rnd, params_full_precision, scale);
		
		// Add correction network parameters
		size_t offset = this->NerfNetworkBase<T>::n_params();
		
		if (m_correction_encoding) {
			m_correction_encoding->initialize_params(rnd, params_full_precision + offset, scale);
			offset += m_correction_encoding->n_params();
		}
		
		if (m_correction_network) {
			m_correction_network->initialize_params(rnd, params_full_precision + offset, scale);
			offset += m_correction_network->n_params();
		}
	}

private:
	// STEP 1: Correction encoding (initialized but not used)
	std::shared_ptr<tcnn::Encoding<T>> m_correction_encoding;
	
	// STEP 2: Correction MLP (initialized but not used)
	std::shared_ptr<tcnn::Network<T>> m_correction_network;
	uint32_t m_correction_network_input_width;
};

} // namespace ngp