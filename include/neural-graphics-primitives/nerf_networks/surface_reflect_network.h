/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   surface_reflect_network.h
 *  @brief  Surface rendering with encoded reflection vectors
 */

#pragma once

#include "surface_network.h"

namespace ngp {

template <typename T>
class SurfaceReflectNetwork : public SurfaceNetwork<T> {
public:
	using json = nlohmann::json;

	SurfaceReflectNetwork(
		uint32_t n_pos_dims,
		uint32_t n_dir_dims,
		uint32_t n_extra_dims,
		uint32_t dir_offset,
		const json& pos_encoding,
		const json& dir_encoding,
		const json& density_network,
		const json& rgb_network,
		bool use_sdf = false
	) : SurfaceNetwork<T>(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
	                      pos_encoding, dir_encoding, density_network, rgb_network,
	                      use_sdf) {
		
		// Override method name
		this->m_method = "surface_reflect";
		
		// RGB network input: 16 surface features + encoded view dirs + encoded reflection vectors
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		uint32_t total_before_padding = 16 + 
		                                 this->m_dir_encoding->padded_output_width() + 
		                                 this->m_dir_encoding->padded_output_width();
		this->m_rgb_network_input_width = tcnn::next_multiple(total_before_padding, rgb_alignment);
		
		// Recreate RGB network with updated input width
		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = this->m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		this->m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
	}

	// Surface reflect computes R = 2(N·V)N - V and encodes it
	// Forward: lines 346-376 in original
	// Backward: lines 1175-1242 in original
};

} // namespace ngp

