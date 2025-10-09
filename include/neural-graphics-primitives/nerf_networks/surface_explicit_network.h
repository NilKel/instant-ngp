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
		// Surface explicit: density from grid, features from MLP, normals from grid gradients
		// See original lines 378-429
		throw std::runtime_error("SurfaceExplicitNetwork::inference_mixed_precision_impl not yet fully implemented");
	}

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		// See original lines 688-738
		throw std::runtime_error("SurfaceExplicitNetwork::forward_impl not yet fully implemented");
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
		// See original lines 1379-1503
		throw std::runtime_error("SurfaceExplicitNetwork::backward_impl not yet fully implemented");
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

