/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network_base.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Base class for mode-specific NeRF network implementations
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/nerf_helpers.h>

namespace ngp {

template <typename T>
class NerfNetworkBase : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetworkBase(
		uint32_t n_pos_dims,
		uint32_t n_dir_dims,
		uint32_t n_extra_dims,
		uint32_t dir_offset,
		const json& pos_encoding,
		const json& dir_encoding,
		const json& density_network,
		const json& rgb_network,
		const std::string& method,
		bool use_sdf = false
	) : m_n_pos_dims{n_pos_dims},
	    m_n_dir_dims{n_dir_dims},
	    m_dir_offset{dir_offset},
	    m_n_extra_dims{n_extra_dims},
	    m_method{method},
	    m_use_sdf{use_sdf} {
		
		// Check for HashPot vector features mode
		const char* hashpot_env = std::getenv("NGP_HASHPOT");
		m_hashpot_mode = (hashpot_env && std::string(hashpot_env) == "1");
		
		// Create encodings
		m_pos_encoding.reset(tcnn::create_encoding<T>(
			n_pos_dims,
			pos_encoding,
			density_network.contains("otype") &&
			(tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") ||
			 tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u
		));
		
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(
			m_n_dir_dims + m_n_extra_dims,
			dir_encoding,
			rgb_alignment
		));
		
		// Note: Density and RGB networks are created in derived classes
		// Each derived class modifies the config (adds n_input_dims, n_output_dims, etc.)
		// before creating the networks to avoid passing unexpected config parameters
		
		// Initialize variance network for SDF mode
		if (m_use_sdf) {
			std::array<int, 1> resolution{1};
			m_variance_network = std::make_shared<TrainableBuffer<1, 1, T>>(resolution);
		}
		
		// Note: Density model (fused pos encoding + density network) is created in derived classes
		// after the density network is initialized
	}

	virtual ~NerfNetworkBase() = default;

	// Configuration setters (common across all modes)
	void set_backprop_normals(bool v) { m_backprop_normals = v; }
	void set_use_analytical_normals(bool v) { m_use_analytical_normals = v; }
	void set_use_eikonal_loss(bool v) { m_use_eikonal_loss = v; }
	void set_eikonal_weight(float weight) { m_eikonal_weight = weight; }
	void set_normalize_normals(bool v) { m_normalize_normals = v; }
	void set_clamp_gradients(bool v) { m_clamp_gradients = v; }
	void set_max_gradient_magnitude(float mag) { m_max_gradient_magnitude = mag; }
	bool hashpot_mode() const { return m_hashpot_mode; }

	// Common parameter management
	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		m_density_model->set_params(params, inference_params, gradients);

		size_t offset = 0;
		m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_dir_encoding->n_params();

		if (m_variance_network) {
			m_variance_network->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_variance_network->n_params();
		}
		
		if (m_density_grid) {
			m_density_grid->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_density_grid->n_params();
		}
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_density_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_density_network->n_params();

		m_rgb_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_dir_encoding->n_params();

		if (m_variance_network) {
			m_variance_network->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_variance_network->n_params();
		}
	}

	size_t n_params() const override {
		size_t params = m_pos_encoding->n_params() + m_density_network->n_params() +
		                m_dir_encoding->n_params() + m_rgb_network->n_params();
		if (m_variance_network) {
			params += m_variance_network->n_params();
		}
		if (m_density_grid) {
			params += m_density_grid->n_params();
		}
		return params;
	}

	// Common accessors
	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}
	
	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	uint32_t width(uint32_t layer) const override {
		auto density_layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		
		if (layer < density_layers.size()) {
			return density_layers[layer].second;
		} else {
			uint32_t rgb_layer = layer - density_layers.size();
			if (rgb_layer < rgb_layers.size()) {
				return rgb_layers[rgb_layer].second;
			}
		}
		return 0;
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->layer_sizes().size() + m_rgb_network->layer_sizes().size();
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		return std::make_pair(nullptr, tcnn::RM);
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& pos_encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	const std::shared_ptr<tcnn::Network<T>>& density_network() const {
		return m_density_network;
	}

	const std::shared_ptr<tcnn::Network<T>>& rgb_network() const {
		return m_rgb_network;
	}

	json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"method", m_method},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
		};
	}

	// Variance monitoring for SDF mode
	float get_current_variance() const {
		if (!m_variance_network || !m_variance_network->params()) {
			return 0.12f;
		}
		return float(m_variance_network->params()[0]);
	}

	float get_current_s_value() const {
		float variance = get_current_variance();
		return expf(variance * 10.0f);
	}

	// Density query (common interface, but may be overridden)
	virtual void density(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetworkBase::density input must be in column major format.");
		}
		m_density_model->set_jit_fusion(false);
		m_density_model->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			output,
			use_inference_params
		);
	}

protected:
	// Protected members accessible to derived classes
	std::shared_ptr<tcnn::Network<T>> m_density_network;
	std::shared_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;
	std::shared_ptr<TrainableBuffer<1, 1, T>> m_variance_network;
	std::shared_ptr<tcnn::Encoding<T>> m_density_grid;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<T>> m_density_model;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims;
	uint32_t m_dir_offset;
	
	bool m_backprop_normals = false;
	bool m_use_analytical_normals = true;
	bool m_use_eikonal_loss = false;
	float m_eikonal_weight = 0.01f;
	bool m_normalize_normals = true;
	bool m_clamp_gradients = false;
	float m_max_gradient_magnitude = 1.0f;
	bool m_hashpot_mode = false;
	bool m_use_sdf = false;

	std::string m_method;

	// Forward context base (can be extended by derived classes)
	struct ForwardContextBase : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrixDynamic<T> density_network_output;
		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrixDynamic<T> rgb_network_output;

		std::unique_ptr<tcnn::Context> pos_encoding_ctx;
		std::unique_ptr<tcnn::Context> dir_encoding_ctx;
		std::unique_ptr<tcnn::Context> density_network_ctx;
		std::unique_ptr<tcnn::Context> rgb_network_ctx;
	};
};

} // namespace ngp



