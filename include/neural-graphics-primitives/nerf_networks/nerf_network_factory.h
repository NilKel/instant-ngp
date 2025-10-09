/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 */

/** @file   nerf_network_factory.h
 *  @brief  Factory function to create mode-specific NeRF networks
 */

#pragma once

#include "nerf_network_base.h"
#include "baseline_network.h"
#include "surface_network.h"
#include "surface_normal_network.h"
#include "surface_reflect_network.h"
#include "surface_explicit_network.h"
#include "baseline_explicit_network.h"
#include "hash_surface_network.h"
#include "volume_network.h"

namespace ngp {

/**
 * @brief Factory function to create a NeRF network based on the method string
 * 
 * @tparam T Network precision type (typically __half or float)
 * @param n_pos_dims Number of position dimensions (typically 3)
 * @param n_dir_dims Number of direction dimensions (typically 3)
 * @param n_extra_dims Number of extra input dimensions
 * @param dir_offset Offset in input where direction starts
 * @param pos_encoding Position encoding configuration (JSON)
 * @param dir_encoding Direction encoding configuration (JSON)
 * @param density_network Density MLP configuration (JSON)
 * @param rgb_network RGB MLP configuration (JSON)
 * @param method Rendering method: "baseline", "surface", "surface_normal", "surface_reflect",
 *               "surface_explicit", "baseline_explicit", "hash_surface", "volume"
 * @param use_sdf Whether to use SDF-to-density conversion (NeuS2 style)
 * @return std::shared_ptr<NerfNetworkBase<T>> Pointer to mode-specific network implementation
 * 
 * @throws std::runtime_error if method is unknown
 */
template <typename T>
std::shared_ptr<NerfNetworkBase<T>> create_nerf_network(
	uint32_t n_pos_dims,
	uint32_t n_dir_dims,
	uint32_t n_extra_dims,
	uint32_t dir_offset,
	const nlohmann::json& pos_encoding,
	const nlohmann::json& dir_encoding,
	const nlohmann::json& density_network,
	const nlohmann::json& rgb_network,
	const std::string& method = "baseline",
	bool use_sdf = false
) {
	printf("=== Creating NeRF network: method=%s, use_sdf=%d ===\n", method.c_str(), use_sdf);
	fflush(stdout);
	
	// Create mode-specific network based on method string
	if (method == "baseline") {
		return std::make_shared<BaselineNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "surface") {
		return std::make_shared<SurfaceNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "surface_normal") {
		return std::make_shared<SurfaceNormalNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "surface_reflect") {
		return std::make_shared<SurfaceReflectNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "surface_explicit") {
		return std::make_shared<SurfaceExplicitNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "baseline_explicit") {
		return std::make_shared<BaselineExplicitNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "hash_surface") {
		return std::make_shared<HashSurfaceNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else if (method == "volume") {
		return std::make_shared<VolumeNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
	else {
		// Fallback to baseline for unknown methods (with warning)
		printf("WARNING: Unknown method '%s', falling back to baseline\n", method.c_str());
		fflush(stdout);
		return std::make_shared<BaselineNetwork<T>>(
			n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
			pos_encoding, dir_encoding, density_network, rgb_network,
			use_sdf
		);
	}
}

/**
 * @brief Helper to get list of supported methods
 */
inline std::vector<std::string> get_supported_methods() {
	return {
		"baseline",
		"surface",
		"surface_normal",
		"surface_reflect",
		"surface_explicit",
		"baseline_explicit",
		"hash_surface",
		"volume"
	};
}

/**
 * @brief Check if a method is supported
 */
inline bool is_method_supported(const std::string& method) {
	auto methods = get_supported_methods();
	return std::find(methods.begin(), methods.end(), method) != methods.end();
}

} // namespace ngp



