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
		// See original lines 477-505
		throw std::runtime_error("VolumeNetwork::inference_mixed_precision_impl not yet fully implemented");
	}

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		// See original lines 661-687
		// Computes divergences: ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z
		throw std::runtime_error("VolumeNetwork::forward_impl not yet fully implemented");
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
		// See original lines 1356-1378
		throw std::runtime_error("VolumeNetwork::backward_impl not yet fully implemented");
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

