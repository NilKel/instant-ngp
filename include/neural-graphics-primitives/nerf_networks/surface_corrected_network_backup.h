/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   surface_corrected_network.h
 *  @brief  Surface rendering with learned normal correction
 */

#pragma once

#include "surface_network.h"

namespace ngp {

template <typename T>
class SurfaceCorrectedNetwork : public SurfaceNetwork<T> {
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
	) : SurfaceNetwork<T>(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
	                      pos_encoding, dir_encoding, density_network, rgb_network,
	                      use_sdf) {

		this->m_method = "surface_corrected";

		// Create correction encoding (separate hashgrid for normal correction)
		json local_correction_encoding_config = correction_encoding;
		if (!correction_encoding.contains("n_dims_to_encode")) {
			local_correction_encoding_config["n_dims_to_encode"] = n_pos_dims;
		}
		m_correction_encoding.reset(tcnn::create_encoding<T>(
			n_pos_dims, local_correction_encoding_config, 1
		));

		// Create correction network (16D output, use first 3 for correction vector)
		// Ensure input width is properly aligned for the network type
		uint32_t correction_alignment = tcnn::minimum_alignment(correction_network);
		m_correction_network_input_width = tcnn::next_multiple(
			m_correction_encoding->padded_output_width(),
			correction_alignment
		);

		json local_correction_network_config = correction_network;
		if (!correction_network.contains("n_output_dims")) {
			local_correction_network_config["n_output_dims"] = 16;  // Padded to 16
		}
		local_correction_network_config["n_input_dims"] = m_correction_network_input_width;
		m_correction_network.reset(tcnn::create_network<T>(local_correction_network_config));

		// Combined model for correction path
		m_correction_model = std::make_shared<tcnn::NetworkWithInputEncoding<T>>(
			m_correction_encoding, m_correction_network
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
		auto forward = std::make_unique<ForwardContextCorrected>();

		// Step 1: Allocate buffers (following baseline pattern)
		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{
			this->m_pos_encoding->padded_output_width(), batch_size, stream,
			this->m_pos_encoding->preferred_output_layout()
		};

		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};

		CUDA_CHECK_THROW(cudaMemsetAsync(forward->rgb_network_input.data(), 0,
		                                  forward->rgb_network_input.n_bytes(), stream));

		// DISABLED - Skip correction network buffers for debugging
		// forward->correction_network_input = tcnn::GPUMatrixDynamic<T>{
		// 	m_correction_network_input_width, batch_size, stream, tcnn::AoS
		// };

		// CUDA_CHECK_THROW(cudaMemsetAsync(forward->correction_network_input.data(), 0,
		//                                   forward->correction_network_input.n_bytes(), stream));

		// forward->correction_network_output = tcnn::GPUMatrixDynamic<T>{
		// 	m_correction_network->padded_output_width(), batch_size, stream, tcnn::AoS
		// };

		forward->analytical_normals = tcnn::GPUMatrixDynamic<float>{3, batch_size, stream, tcnn::AoS};
		forward->corrected_normals = tcnn::GPUMatrixDynamic<float>{3, batch_size, stream, tcnn::AoS};

		// Step 2: Position encoding for main density network
		forward->pos_encoding_ctx = this->m_pos_encoding->forward(
			stream,
			input.slice_rows(0, this->m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			true  // Always prepare gradients for surface mode (needed for normals)
		);

		// Step 3: Separate buffer for density output (like surface network)
		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{
			this->m_density_network->padded_output_width(), batch_size, stream, tcnn::AoS
		};

		forward->density_network_ctx = this->m_density_network->forward(
			stream,
			forward->density_network_input,
			&forward->density_network_output,
			use_inference_params,
			true  // Enable gradients for normal computation
		);

		// Step 4: Compute analytical normals using the same helper as SurfaceNetwork
		// Cast to base class for compatibility
		auto& base_forward = reinterpret_cast<std::unique_ptr<typename SurfaceNetwork<T>::ForwardContext>&>(forward);
		forward->analytical_normals = this->compute_analytical_normals_forward(
			stream, batch_size, input, base_forward, use_inference_params
		);

		// Step 5: DISABLED - Skip correction encoding and network for debugging
		// auto correction_encoding_output = forward->correction_network_input.slice_rows(
		// 	0, m_correction_encoding->padded_output_width()
		// );

		// forward->correction_encoding_ctx = m_correction_encoding->forward(
		// 	stream,
		// 	input.slice_rows(0, m_correction_encoding->input_width()),
		// 	&correction_encoding_output,
		// 	use_inference_params,
		// 	prepare_input_gradients
		// );

		// forward->correction_network_ctx = m_correction_network->forward(
		// 	stream,
		// 	forward->correction_network_input,
		// 	&forward->correction_network_output,
		// 	use_inference_params,
		// 	prepare_input_gradients
		// );

		// Step 6: Compute surface features using analytical normals (like surface network)
		auto surface_features_slice = forward->rgb_network_input.slice_rows(0, 16);
		
		linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
			batch_size,
			forward->density_network_output.layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
			forward->density_network_output.data(),
			forward->analytical_normals.data(),  // Use analytical normals directly
			surface_features_slice.layout() == tcnn::AoS ? surface_features_slice.stride() : 1,
			surface_features_slice.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr
		);

		// Step 8: Direction encoding
		auto dir_slice = forward->rgb_network_input.slice_rows(
			16, this->m_dir_encoding->padded_output_width()
		);
		forward->dir_encoding_ctx = this->m_dir_encoding->forward(
			stream,
			input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
			&dir_slice,
			use_inference_params,
			prepare_input_gradients
		);

		// Step 9: RGB network
		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{
				output->data(), this->m_rgb_network->padded_output_width(), batch_size, output->layout()
			};
		}

		forward->rgb_network_ctx = this->m_rgb_network->forward(
			stream,
			forward->rgb_network_input,
			output ? &forward->rgb_network_output : nullptr,
			use_inference_params,
			prepare_input_gradients
		);

		// Step 10: Extract density to output
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
	) {
		const auto& forward = dynamic_cast<const ForwardContextCorrected&>(ctx);
		uint32_t batch_size = input.n();

		// Step 1: Extract RGB gradients (following baseline pattern)
		tcnn::GPUMatrix<T> dL_drgb{this->m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));

		linear_kernel(extract_rgb<T>, 0, stream,
			batch_size * 3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{
			(T*)output.data(), this->m_rgb_network->padded_output_width(), batch_size, output.layout()
		};

		// Step 2: Allocate gradient buffer for RGB network input
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{
			this->m_rgb_network_input_width, batch_size, stream, tcnn::AoS
		};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0,
		                                  dL_drgb_network_input.n_bytes(), stream));

		// Step 3: RGB network backward
		this->m_rgb_network->backward(
			stream,
			*forward.rgb_network_ctx,
			forward.rgb_network_input,
			rgb_network_output,
			dL_drgb,
			&dL_drgb_network_input,
			use_inference_params,
			param_gradients_mode
		);

		// Step 4: Direction encoding backward
		if (this->m_dir_encoding->n_params() > 0 || dL_dinput) {
			auto dL_ddir_slice = dL_drgb_network_input.slice_rows(
				16, this->m_dir_encoding->padded_output_width()
			);

			tcnn::GPUMatrixDynamic<float> dL_ddir_input;
			if (dL_dinput) {
				dL_ddir_input = dL_dinput->slice_rows(
					this->m_dir_offset, this->m_dir_encoding->input_width()
				);
			}

			auto dir_out = forward.rgb_network_input.slice_rows(
				16, this->m_dir_encoding->padded_output_width()
			);

			this->m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(this->m_dir_offset, this->m_dir_encoding->input_width()),
				dir_out,
				dL_ddir_slice,
				dL_dinput ? &dL_ddir_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		// Step 5: Surface features backward (to analytical normals and density output)
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
			forward.analytical_normals.data(),  // Use analytical normals directly
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1,
			forward.density_network_output.data(),
			dL_ddensity_network_output.data(),
			dL_dnormals.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr
		);

		// Step 6: Analytical normals backward (no correction gradients)
		// The dL_dnormals from surface features backward is already the analytical normals gradient

		// Step 7: DISABLED - Skip correction network backward for debugging
		// tcnn::GPUMatrixDynamic<T> dL_dcorrection_network_input;
		// if (m_correction_encoding->n_params() > 0 || dL_dinput) {
		// 	dL_dcorrection_network_input = tcnn::GPUMatrixDynamic<T>{
		// 		m_correction_network_input_width, batch_size, stream, tcnn::AoS
		// 	};
		// }

		// m_correction_network->backward(
		// 	stream,
		// 	*forward.correction_network_ctx,
		// 	forward.correction_network_input,
		// 	forward.correction_network_output,
		// 	dL_dcorrection_output,
		// 	dL_dcorrection_network_input.data() ? &dL_dcorrection_network_input : nullptr,
		// 	use_inference_params,
		// 	param_gradients_mode
		// );

		// Step 8: DISABLED - Skip correction encoding backward for debugging
		// if (dL_dcorrection_network_input.data()) {
		// 	// Extract the slice corresponding to the actual encoding output
		// 	auto dL_dcorrection_encoding_output = dL_dcorrection_network_input.slice_rows(
		// 		0, m_correction_encoding->padded_output_width()
		// 	);

		// 	// Get the encoding output from forward pass
		// 	auto correction_encoding_output = forward.correction_network_input.slice_rows(
		// 		0, m_correction_encoding->padded_output_width()
		// 	);

		// 	tcnn::GPUMatrixDynamic<float> dL_dcorrection_pos_input;
		// 	if (dL_dinput) {
		// 		dL_dcorrection_pos_input = dL_dinput->slice_rows(
		// 			0, m_correction_encoding->input_width()
		// 		);
		// 	}

		// 	m_correction_encoding->backward(
		// 		stream,
		// 		*forward.correction_encoding_ctx,
		// 		input.slice_rows(0, m_correction_encoding->input_width()),
		// 		correction_encoding_output,
		// 		dL_dcorrection_encoding_output,
		// 		dL_dinput ? &dL_dcorrection_pos_input : nullptr,
		// 		use_inference_params,
		// 		param_gradients_mode
		// 	);
		// }

		// Step 9: Add density gradient from final output
		linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::AoS ? dL_ddensity_network_output.stride() : 1,
			dL_ddensity_network_output.data()
		);

		// Step 10: Density network backward
		tcnn::GPUMatrixDynamic<T> dL_ddensity_input;
		if (this->m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_input = tcnn::GPUMatrixDynamic<T>{
				this->m_pos_encoding->padded_output_width(), batch_size, stream,
				this->m_pos_encoding->preferred_output_layout()
			};
		}

		this->m_density_network->backward(
			stream,
			*forward.density_network_ctx,
			forward.density_network_input,
			forward.density_network_output,
			dL_ddensity_network_output,
			dL_ddensity_input.data() ? &dL_ddensity_input : nullptr,
			use_inference_params,
			param_gradients_mode
		);

		// Step 11: Position encoding backward
		if (dL_ddensity_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_input;
			if (dL_dinput) {
				dL_dpos_input = dL_dinput->slice_rows(0, this->m_pos_encoding->input_width());
			}

			this->m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, this->m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_input,
				dL_dinput ? &dL_dpos_input : nullptr,
				use_inference_params,
				param_gradients_mode == tcnn::GradientMode::Overwrite ?
					tcnn::GradientMode::Accumulate : tcnn::GradientMode::Accumulate
			);
		}
	}

	void density(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error{"SurfaceCorrectedNetwork::density input must be in column-major layout"};
		}

		this->m_density_model->inference_mixed_precision(stream, input, output, use_inference_params);

		linear_kernel(extract_density<T>, 0, stream,
			output.n(),
			output.layout() == tcnn::AoS ? output.stride() : 1,
			output.layout() == tcnn::AoS ? output.stride() : 1,
			output.data(),
			output.data(),
			this->m_use_sdf,
			this->m_variance_network ? this->m_variance_network->params() : nullptr,
			false
		);
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		// Set density model (wraps pos_encoding + density_network)
		this->m_density_model->set_params(params, inference_params, gradients);

		// Set correction model (wraps correction_encoding + correction_network)
		m_correction_model->set_params(params, inference_params, gradients);

		size_t offset = 0;
		// Individual components in same order as n_params()
		this->m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += this->m_density_network->n_params();

		m_correction_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_correction_network->n_params();

		this->m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += this->m_rgb_network->n_params();

		this->m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += this->m_pos_encoding->n_params();

		m_correction_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_correction_encoding->n_params();

		this->m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += this->m_dir_encoding->n_params();

		if (this->m_variance_network) {
			this->m_variance_network->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += this->m_variance_network->n_params();
		}
	}

	size_t n_params() const override {
		// Count individual components, same order as set_params_impl
		size_t params = this->m_pos_encoding->n_params() +
		                this->m_density_network->n_params() +
		                m_correction_encoding->n_params() +
		                m_correction_network->n_params() +
		                this->m_dir_encoding->n_params() +
		                this->m_rgb_network->n_params();
		if (this->m_variance_network) {
			params += this->m_variance_network->n_params();
		}
		return params;
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Initialize individual components in same order as n_params() and set_params_impl
		this->m_density_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += this->m_density_network->n_params();

		m_correction_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_correction_network->n_params();

		this->m_rgb_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += this->m_rgb_network->n_params();

		this->m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += this->m_pos_encoding->n_params();

		m_correction_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_correction_encoding->n_params();

		this->m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += this->m_dir_encoding->n_params();

		if (this->m_variance_network) {
			this->m_variance_network->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += this->m_variance_network->n_params();
		}
	}

	uint32_t padded_output_width() const override {
		return std::max(this->m_rgb_network->padded_output_width() + 1, (uint32_t)4);
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = this->m_density_network->layer_sizes();
		auto correction_layers = m_correction_network->layer_sizes();
		auto rgb_layers = this->m_rgb_network->layer_sizes();
		layers.insert(layers.end(), correction_layers.begin(), correction_layers.end());
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	json hyperparams() const override {
		json result = SurfaceNetwork<T>::hyperparams();
		result["correction_encoding"] = m_correction_encoding->hyperparams();
		result["correction_network"] = m_correction_network->hyperparams();
		return result;
	}

private:
	struct ForwardContextCorrected : public SurfaceNetwork<T>::ForwardContext {
		tcnn::GPUMatrixDynamic<T> correction_network_input;
		tcnn::GPUMatrixDynamic<T> correction_network_output;
		tcnn::GPUMatrixDynamic<float> corrected_normals;

		std::unique_ptr<tcnn::Context> correction_encoding_ctx;
		std::unique_ptr<tcnn::Context> correction_network_ctx;
	};

	std::shared_ptr<tcnn::Encoding<T>> m_correction_encoding;
	std::shared_ptr<tcnn::Network<T>> m_correction_network;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<T>> m_correction_model;
	uint32_t m_correction_network_input_width;
};

}
