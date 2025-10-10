/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
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
class NerfNetwork : public Network<float, T> {
private:
	// Forward declaration of private struct
	struct ForwardContext;
	
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network, const std::string& method = "baseline", bool use_sdf = false) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims}, m_method{method}, m_use_sdf{use_sdf} {
		
		// Check for HashPot vector features mode
		const char* hashpot_env = std::getenv("NGP_HASHPOT");
		m_hashpot_mode = (hashpot_env && std::string(hashpot_env) == "1");
		
		m_pos_encoding.reset(create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = minimum_alignment(rgb_network);
		m_dir_encoding.reset(create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));
		
		// Variance parameter is now integrated into the density network output
		// Channel 1 of density network will contain the learnable variance parameter

		json local_density_network_config = density_network;
		
	// HashPot mode: encoding output is 3x larger (conceptually N,F,3 vectors)
	// but density network input size is just the encoding output width
	uint32_t density_input_dims = m_pos_encoding->padded_output_width();
	printf("=== NerfNetwork CONSTRUCTOR REACHED, method=%s ===\n", m_method.c_str());
	fflush(stdout);
		if (m_hashpot_mode) {
			uint32_t encoding_output_width = m_pos_encoding->padded_output_width();
		} else if (m_method == "hash_surface") {
			// hash_surface: Density MLP takes extracted density features (n_levels)
			// With 4 features per level and 8 levels: 32 total features, extract 8 density features
			uint32_t n_levels = density_input_dims / 4; // 32 / 4 = 8 levels
			
			// Get density network alignment and pad if needed
			uint32_t density_alignment = minimum_alignment(density_network);
			uint32_t padded_density_input_dims = next_multiple(n_levels, density_alignment);
			
			density_input_dims = padded_density_input_dims; // Use padded dimensions
		}
		
	local_density_network_config["n_input_dims"] = density_input_dims;
	if (!density_network.contains("n_output_dims")) {
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume") {
			// 48D: 1D density + 45D Φ features (15 x 3D vectors) for surface or volume features
			local_density_network_config["n_output_dims"] = 48;
		} else if (m_method == "surface_explicit") {
			// surface_explicit: 48D output (1 dummy + 45D Φ features for 15 vectors)
			// Density comes from grid, not MLP
			local_density_network_config["n_output_dims"] = 48;
		} else if (m_method == "baseline_explicit") {
			// baseline_explicit: 15D output (15 features for RGB input[1-15])
			// Density comes from grid (goes to RGB input[0])
			local_density_network_config["n_output_dims"] = 15;
		} else if (m_method == "hash_surface") {
			// hash_surface: Density MLP takes n_levels density features and outputs 1D density
			local_density_network_config["n_output_dims"] = 1;
		} else if (m_use_sdf) {
			// SDF mode: 1D SDF in channel 0, rest can be features for color
			local_density_network_config["n_output_dims"] = 16;
		} else {
			local_density_network_config["n_output_dims"] = 16;
		}
	}
	m_density_network.reset(create_network<T>(local_density_network_config));
	
	// Initialize explicit density grid for surface_explicit and baseline_explicit modes
	if (m_method == "surface_explicit" || m_method == "baseline_explicit") {
		// Get grid resolution from config (default 128)
		uint32_t grid_res = 128;
		if (density_network.contains("explicit_grid_resolution")) {
			grid_res = density_network["explicit_grid_resolution"];
		}
		
		printf("GRID RESOLUTION: %d\n", grid_res);
		// Create DenseGrid encoding config
		// NOTE: For single-level grids, must set per_level_scale=1.0 to avoid division by zero
		// NOTE: Can't specify both n_features and n_levels - use n_levels instead
		json dense_grid_config = {
			{"otype", "DenseGrid"},
			{"n_levels", 1},  // Single level grid
			{"n_features_per_level", 1},  // 1 feature per level
			{"base_resolution", grid_res},
			{"per_level_scale", 1.0f},  // Must be 1.0 to avoid div-by-zero with n_levels=1
			{"interpolation", "Linear"}
		};
		
		printf("Dense grid config: %s\n", dense_grid_config.dump().c_str());
		fflush(stdout);
		
		m_density_grid.reset(create_encoding<T>(3, dense_grid_config, 1));  // Use alignment=1 to avoid padding
		
		printf("%s: Created %dx%dx%d density grid with %zu parameters\n",
			m_method.c_str(), 
			grid_res, grid_res, grid_res, m_density_grid->n_params());
		printf("  padded_output_width: %u, n_output_dims: %u\n", 
			m_density_grid->padded_output_width(), m_density_grid->output_width());
		fflush(stdout);
		
		// Initialize grid to NEGATIVE log-density values for low-density start
		// Standard NeRF initializes to ~exp(-5) ≈ 0.007 density (nearly transparent)
		// We'll initialize uniformly to around -5 with small noise
		std::vector<T> init_params(m_density_grid->n_params());
		pcg32 rng(42);
		for (size_t i = 0; i < init_params.size(); ++i) {
			// Uniform in [-5.5, -4.5] → after exp(): [0.004, 0.011] - nearly transparent
			init_params[i] = T(-5.0f + (rng.next_float() - 0.5f));  // -5 ± 0.5
		}
		m_density_grid->set_params(init_params.data(), init_params.data(), init_params.data());
		printf("Initialized grid to log-density ~[-5.5, -4.5] (density ~[0.004, 0.011] after exp)\n");
		fflush(stdout);
	}

	if (m_method == "surface_normal") {
		// Surface_normal: 16 surface features + encoded view dirs + encoded normals
		uint32_t total_before_padding = 16 + m_dir_encoding->padded_output_width() + m_dir_encoding->padded_output_width();
		m_rgb_network_input_width = next_multiple(total_before_padding, rgb_alignment);
	} else if (m_method == "surface_reflect") {
		// Surface_reflect: 16 surface features + encoded view dirs + encoded reflection vectors
		uint32_t total_before_padding = 16 + m_dir_encoding->padded_output_width() + m_dir_encoding->padded_output_width();
		m_rgb_network_input_width = next_multiple(total_before_padding, rgb_alignment);
	} else if (m_method == "surface" || m_method == "surface_explicit") {
		// Surface/Surface_explicit: 16 surface features + direction encoding
		m_rgb_network_input_width = next_multiple(16 + m_dir_encoding->padded_output_width(), rgb_alignment);
	} else if (m_method == "volume") {
		// Volume: 16 divergence features + direction encoding (same as surface)
		m_rgb_network_input_width = next_multiple(16 + m_dir_encoding->padded_output_width(), rgb_alignment);
	} else if (m_method == "hash_surface") {
		// hash_surface: n_levels surface features + 1D density + direction encoding
		uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // 32 / 4 = 8 levels
		uint32_t surface_features = n_levels; // 8 hash surface features
		m_rgb_network_input_width = next_multiple(surface_features + 1 + m_dir_encoding->padded_output_width(), rgb_alignment);
	} else {
		// Baseline: density output + direction encoding  
		m_rgb_network_input_width = next_multiple(m_dir_encoding->padded_output_width() + std::max(16u, m_density_network->padded_output_width()), rgb_alignment);
	}

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(create_network<T>(local_rgb_network_config));

		// Initialize variance network for SDF mode
		if (m_use_sdf) {
			std::array<int, 1> resolution{1};
			m_variance_network = std::make_shared<TrainableBuffer<1, 1, T>>(resolution);
		}


		m_density_model = std::make_shared<NetworkWithInputEncoding<T>>(m_pos_encoding, m_density_network);
	}

	virtual ~NerfNetwork() { }

	void set_backprop_normals(bool v) { m_backprop_normals = v; }
	void set_use_analytical_normals(bool v) { m_use_analytical_normals = v; }
	void set_use_eikonal_loss(bool v) { m_use_eikonal_loss = v; }
	void set_eikonal_weight(float weight) { m_eikonal_weight = weight; }
	void set_normalize_normals(bool v) { m_normalize_normals = v; }
	void set_clamp_gradients(bool v) { m_clamp_gradients = v; }
	void set_max_gradient_magnitude(float mag) { m_max_gradient_magnitude = mag; }
	
	bool hashpot_mode() const { return m_hashpot_mode; }

void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
	uint32_t batch_size = input.n();
	GPUMatrixDynamic<T> density_network_input;

	
	// For hash_surface, we need separate matrices for hash features and density features
		GPUMatrixDynamic<T> hash_features_matrix;
		if (m_method == "hash_surface") {
			// Create separate 32D matrix for hash features from position encoding
			hash_features_matrix = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			// Create 16D matrix for extracted density features (will be populated later)
			density_network_input = GPUMatrixDynamic<T>{m_density_network->input_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		} else {
			density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}
		
	// CRITICAL: For surface modes, force AoS layout to match Frequency encoding behavior
	MatrixLayout surface_layout = (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "hash_surface" || m_method == "surface_explicit") ? AoS : m_dir_encoding->preferred_output_layout();
	// FIXED: Use the same RGB network input width as training (m_rgb_network_input_width) for all modes
	GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, surface_layout};

		// CRITICAL FIX: Zero out the RGB network input buffer in inference mode too
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0, rgb_network_input.n_bytes(), stream));

		GPUMatrixDynamic<T> density_network_output;
	if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "surface_explicit") {
		// CRITICAL: For surface modes, use AoS layout for density buffer to ensure copy compatibility
		// This forces SphericalHarmonics to behave like Frequency encoding
		density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
	} else if (m_method == "hash_surface") {
		// hash_surface: Separate buffer for 1D density output
		density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
	} else if (m_method == "baseline_explicit") {
		// baseline_explicit: Separate buffer for 15D MLP features (not a slice!)
		density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
	} else {
		// Baseline mode (including SDF mode - use same pattern)
		density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
	}

	GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
	
	// For surface_explicit: grid density needs to persist until final extraction
	GPUMatrixDynamic<T> grid_density_explicit;

	// Standard forward pass
	if (m_method == "hash_surface") {
			// For hash_surface, position encoding outputs to hash_features_matrix (32D)
			m_pos_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				hash_features_matrix,
				use_inference_params
			);
		} else {
			// For other methods, position encoding outputs to density_network_input
			m_pos_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input,
				use_inference_params
			);
		}


		if (m_method == "hash_surface") {

			// hash_surface: Extract density features from hash interpolation, then pass to density MLP
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // Dynamic levels calculation
			


			
			// Clear the density_network_input matrix (set to zero for padding)
			CUDA_CHECK_THROW(cudaMemsetAsync(density_network_input.data(), 0, density_network_input.n_bytes(), stream));
			
			// Extract density features from hash_features_matrix to density_network_input (first n_levels rows)
			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels, // Dynamic number of levels
				4, // 4 features per level
				hash_features_matrix.data(),
				hash_features_matrix.layout() == AoS ? hash_features_matrix.stride() : 1,
				density_network_input.data(), // Write to first n_levels rows (with padding)
				density_network_input.layout() == AoS ? density_network_input.stride() : 1
			);
			
			// Pass density_network_input (8D features + padding) to density MLP
			
			m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);
		} else {
			m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);
		}

		// Set up direction encoding
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
			// Surface modes: Compute analytical normals and use them for surface features
			
			// Compute analytical normals (normalized/raw based on settings) for inference
			GPUMatrixDynamic<float> analytical_normals = compute_analytical_normals_inference_unified(
				stream, batch_size, input, density_network_input, density_network_output, use_inference_params
			);
			
			// Compute surface features directly using analytical normals (all 16 channels in one kernel)
			linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				density_network_output.data(),
				analytical_normals.data(),  // Pass analytical normals
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				rgb_network_input.data(),
				m_use_sdf,  // SDF mode flag
				m_variance_network ? m_variance_network->params() : nullptr  // Variance params
			);
			
			// Direction encoding goes after the 16D surface features
			auto dir_out = rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			
			// Encode view directions
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
			
			// For surface_normal mode, encode normals in inference too
			if (m_method == "surface_normal") {
				// Get slice for the encoded normals (same width as direction encoding)
				uint32_t normal_start_idx = 16 + m_dir_encoding->padded_output_width();
				auto normal_out = rgb_network_input.slice_rows(normal_start_idx, m_dir_encoding->padded_output_width());
				
				// Create a float matrix view of the analytical normals compatible with encoding input
				GPUMatrixDynamic<float> normals_for_encoding{3, batch_size, stream, analytical_normals.layout()};
				CUDA_CHECK_THROW(cudaMemcpyAsync(normals_for_encoding.data(), analytical_normals.data(), 
					analytical_normals.n_bytes(), cudaMemcpyDeviceToDevice, stream));
				
				// Encode normals using the same encoding as view directions (no gradients needed for inference)
				m_dir_encoding->inference_mixed_precision(
					stream,
					normals_for_encoding,
					normal_out,
					use_inference_params
				);
			}
			
			// For surface_reflect mode, encode reflection vectors in inference too
			if (m_method == "surface_reflect") {
				// Get slice for the encoded reflection vectors (same width as direction encoding)
				uint32_t reflect_start_idx = 16 + m_dir_encoding->padded_output_width();
				auto reflect_out = rgb_network_input.slice_rows(reflect_start_idx, m_dir_encoding->padded_output_width());
				
				// Extract view directions from input (use directly without copying)
				auto view_dirs_input = input.slice_rows(m_dir_offset, m_dir_encoding->input_width());
				
				// Compute reflection vectors directly from input view directions
				GPUMatrixDynamic<float> reflection_vectors{3, batch_size, stream, analytical_normals.layout()};
				
				linear_kernel(calculate_reflection_vector_kernel<T>, 0, stream,
					batch_size,
					view_dirs_input.data(),                      // Use view direction input directly
					analytical_normals.data(),                  // Analytical normals
					reflection_vectors.data(),                  // Output reflection vectors
					m_dir_encoding->input_width(),               // Width of view direction input
					view_dirs_input.layout() == AoS ? view_dirs_input.m() : 1,  // View stride
					analytical_normals.layout() == AoS ? 3 : 1, // Normal stride
					reflection_vectors.layout() == AoS ? 3 : 1  // Reflection stride
				);
				
				// Encode reflection vectors using the same encoding as view directions
				m_dir_encoding->inference_mixed_precision(
					stream,
					reflection_vectors,
					reflect_out,
			use_inference_params
		);
	}
	
} else if (m_method == "surface_explicit") {
		// surface_explicit: density from grid + features from MLP + normals from grid gradients
		
	// Step 1: Get density from grid (need forward context for normals)
	// Use grid's preferred layout (SoA) to match hash encoding
	grid_density_explicit = GPUMatrixDynamic<T>{m_density_grid->padded_output_width(), batch_size, stream, m_density_grid->preferred_output_layout()};
	auto grid_ctx = m_density_grid->forward(
		stream,
		input.slice_rows(0, 3),  // positions
		&grid_density_explicit,
		use_inference_params,
		true  // prepare_input_gradients = true for normal computation
	);
		
		// Step 2: Compute normals from grid gradients using autodiff
		GPUMatrixDynamic<float> normals = compute_normals_from_grid_gradients(
			stream, batch_size, input.slice_rows(0, 3), m_density_grid, *grid_ctx, grid_density_explicit, use_inference_params
		);
		
		// Step 3: MLP forward for features
		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);
		
	// Step 4: Compute surface features
	// MLP outputs 45D vectors (15 x 3D) starting from channel 0
	auto surface_features_slice = rgb_network_input.slice_rows(1, 15);
	linear_kernel(compute_surface_features_from_vectors_kernel<T>, 0, stream,
		batch_size,
		density_network_output.data(),  // Use channels 0-44 (45 features for 15 3D vectors)
		density_network_output.layout() == AoS ? density_network_output.stride() : 1,
		normals.data(),
		normals.layout() == AoS ? 3 : 1,
		surface_features_slice.data(),
		surface_features_slice.layout() == AoS ? surface_features_slice.stride() : 1
	);
		
		// Step 5: Replace channel 0 with grid density (extract from padded grid output)
		linear_kernel(replace_first_channel_kernel<T>, 0, stream,
			batch_size,
			grid_density_explicit.data(),
			grid_density_explicit.layout() == AoS ? grid_density_explicit.stride() : 1,  // grid stride (padded)
			rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,  // output stride
			rgb_network_input.data(),
			false  // Old monolithic implementation - surface_explicit uses grid for normals
		);
		
		// Step 6: Direction encoding (same as surface mode)
		auto dir_out = rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);
	
} else if (m_method == "baseline_explicit") {
	// baseline_explicit: density from grid, features from MLP (no surface computations)
	
	
	// Step 1: Get density from grid
	grid_density_explicit = GPUMatrixDynamic<T>{m_density_grid->padded_output_width(), batch_size, stream, m_density_grid->preferred_output_layout()};
	auto grid_ctx = m_density_grid->forward(
		stream,
		input.slice_rows(0, 3),  // positions
		&grid_density_explicit,
		use_inference_params,
		false  // don't need input gradients
	);
	
	// Step 2: MLP forward for 15D features
	m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);
	
	// Step 3: Copy grid density to RGB input[0]
	linear_kernel(replace_first_channel_kernel<T>, 0, stream,
		batch_size,
		grid_density_explicit.data(),
		grid_density_explicit.layout() == AoS ? grid_density_explicit.stride() : 1,
		rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
		rgb_network_input.data(),
		true  // Old monolithic: baseline_explicit applies ReLU (grid stores raw density)
	);
	
	// Step 4: Copy MLP features[0-14] to RGB input[1-15]
	linear_kernel(copy_channels_kernel<T>, 0, stream,
		batch_size,
		15,  // number of channels to copy
		density_network_output.data(),
		density_network_output.layout() == AoS ? density_network_output.stride() : 1,
		rgb_network_input.data(),
		rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
		1  // destination offset (skip channel 0)
	);
	
	// Step 5: Direction encoding
	auto dir_out = rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
	m_dir_encoding->inference_mixed_precision(
		stream,
		input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
		dir_out,
		use_inference_params
	);

} else if (m_method == "volume") {
	// Volume mode: Compute divergences and use them for volume features
			
			// Compute volume divergences for inference
			GPUMatrixDynamic<float> volume_divergences = compute_volume_divergences_inference(
				stream, batch_size, input, density_network_input, density_network_output, use_inference_params
			);
			
			// Compute volume features using divergences (all 16 channels in one kernel)
			linear_kernel(compute_volume_divergence_kernel<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				density_network_output.data(),
				volume_divergences.data(),  // Pass volume divergences
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				rgb_network_input.data()
			);
			
			// Direction encoding goes after the 16D features (surface or volume)
			auto dir_out = rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			
			// Encode view directions
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
			
		} else if (m_method == "hash_surface") {
			// hash_surface mode: Compute analytical normals from density MLP output, then compute surface features

			// Compute analytical normals from density MLP output (1D density)
			GPUMatrixDynamic<float> analytical_normals = compute_analytical_normals_inference_unified(
				stream, batch_size, input, density_network_input, density_network_output, use_inference_params
			);
			
			// Compute hash surface features using vector potential from hash interpolation and analytical normals
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // Dynamic levels calculation
			uint32_t surface_features = n_levels; // n_levels hash surface features (no density)
			
			linear_kernel(compute_hash_surface_features_kernel<T>, 0, stream,
				batch_size,
				n_levels, // Dynamic number of levels
				4, // 4 features per level
				hash_features_matrix.data(), // Use full 32D hash features to compute surface features
				hash_features_matrix.layout() == AoS ? hash_features_matrix.stride() : 1,
				analytical_normals.data(), // 3D analytical normals
				rgb_network_input.data(), // Output: n_levels surface features (no density)
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1
			);
			
			// Copy density MLP output[0] to RGB input after surface features
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				density_network_output.data(),
				rgb_network_input.data() + (rgb_network_input.layout() == AoS ? surface_features : surface_features * batch_size),
				m_use_sdf,  // SDF mode flag
				m_variance_network ? m_variance_network->params() : nullptr,  // Variance params
				false  // MLP already has activation
			);
			
			// Direction encoding goes after the surface features + density
			auto dir_out = rgb_network_input.slice_rows(surface_features + 1, m_dir_encoding->padded_output_width());

			// Encode view directions
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
			
		} else {
			// Baseline mode
			
			
			auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
			
			
		}

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		

// Extract density to output - use correct source for each mode
if (m_method == "surface_explicit" || m_method == "baseline_explicit") {
	// surface_explicit/baseline_explicit: Extract density from grid, not MLP
	linear_kernel(extract_density<T>, 0, stream,
		batch_size,
		grid_density_explicit.layout() == AoS ? grid_density_explicit.stride() : 1,  // Use actual stride from padded grid
		output.layout() == AoS ? padded_output_width() : 1,
		grid_density_explicit.data(),
		output.data() + 3 * (output.layout() == AoS ? 1 : batch_size),
		false,  // grid stores log-density, not SDF
		nullptr,
		false  // Don't apply exp() - rendering kernels will do it (consistent with baseline mode)
	);
} else {
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == AoS ? density_network_output.stride() : 1,
			output.layout() == AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == AoS ? 1 : batch_size),
			m_use_sdf,
			m_variance_network ? m_variance_network->params() : nullptr,  // Variance params
			false  // MLP already has activation
		);
	}
}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
	uint32_t batch_size = input.n();
	auto forward = std::make_unique<ForwardContext>();
	GPUMatrixDynamic<T> density_network_input;
		if (m_method == "hash_surface") {
			forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		} else {
			forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}
		
	// CRITICAL: For surface modes, force AoS layout to match Frequency encoding behavior
	MatrixLayout surface_layout = (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "hash_surface" || m_method == "surface_explicit") ? AoS : m_dir_encoding->preferred_output_layout();
	forward->rgb_network_input = GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, surface_layout};

		// CRITICAL FIX: Zero out the RGB network input buffer to prevent garbage in unused sections
		// This is especially important for surface_normal mode which has larger buffers
		CUDA_CHECK_THROW(cudaMemsetAsync(forward->rgb_network_input.data(), 0, forward->rgb_network_input.n_bytes(), stream));

	forward->pos_encoding_ctx = m_pos_encoding->forward(
		stream,
		input.slice_rows(0, m_pos_encoding->input_width()),
		&forward->density_network_input,
		use_inference_params,
		prepare_input_gradients || m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "hash_surface" || m_method == "surface_explicit" // Always prepare gradients for surface and volume modes
	);

		GPUMatrixDynamic<T> dir_out;
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
			// Surface modes: Use baseline-style slicing for density, custom kernel for surface features
			// CRITICAL: For surface modes, use AoS layout for density buffer to ensure copy compatibility
			// This forces SphericalHarmonics to behave like Frequency encoding
			forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
			
			dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			
			// CRITICAL: Enable gradient computation for analytical normals
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, true);
			
			// Compute analytical normals (normalized/raw based on settings) using NeuS2 pattern (no gradient flow to parameters)
			forward->analytical_normals = compute_analytical_normals_forward_unified(
				stream, batch_size, input, forward, use_inference_params
			);
			
			// Compute surface features directly into RGB slice using analytical normals
			auto surface_features_slice = forward->rgb_network_input.slice_rows(0, 16);
			
			// For surface modes, compute surface features  
			linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				forward->density_network_output.data(),
				forward->analytical_normals.data(),  // Pass normalized analytical normals
				surface_features_slice.layout() == AoS ? surface_features_slice.stride() : 1,
				surface_features_slice.data(),
				m_use_sdf,  // SDF mode flag
				m_variance_network ? m_variance_network->params() : nullptr  // Variance params
			);
			
		} else if (m_method == "volume") {
			// Volume mode: Use same density network approach but with divergence computation
			// CRITICAL: For volume mode, use AoS layout for density buffer to ensure copy compatibility
			forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
			
			dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			
			// CRITICAL: Enable gradient computation for divergence computation
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, true);
			
			// Compute volume divergences using the same pattern as analytical normals (no gradient flow to parameters)
			forward->volume_divergences = compute_volume_divergences_forward(
				stream, batch_size, input, forward, use_inference_params
			);
			
			// Compute volume features directly into RGB slice using divergences
			auto volume_features_slice = forward->rgb_network_input.slice_rows(0, 16);
			
		// Compute volume features using divergences
		linear_kernel(compute_volume_divergence_kernel<T>, 0, stream,
			batch_size,
			forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
			forward->density_network_output.data(),
			forward->volume_divergences.data(),  // Pass volume divergences
			volume_features_slice.layout() == AoS ? volume_features_slice.stride() : 1,
			volume_features_slice.data()
		);
} else if (m_method == "surface_explicit") {
	// surface_explicit: density from grid + features from MLP
	forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
	
// Step 1: Get density from grid (with gradients for normal computation)
// Use grid's preferred layout (SoA) to match hash encoding
forward->grid_density = GPUMatrixDynamic<T>{m_density_grid->padded_output_width(), batch_size, stream, m_density_grid->preferred_output_layout()};
forward->density_grid_ctx = m_density_grid->forward(
	stream,
	input.slice_rows(0, 3),  // positions only
	&forward->grid_density,
	use_inference_params,
	true  // prepare_input_gradients for normal computation
);
	
	// Step 2: MLP forward pass (48D output: 1 dummy + 45 features for 15 vectors)
	forward->density_network_ctx = m_density_network->forward(
		stream, forward->density_network_input, &forward->density_network_output, 
		use_inference_params, false
	);
	
	// Step 3: Compute normals from grid gradients using autodiff
	forward->analytical_normals = compute_normals_from_grid_gradients(
		stream, batch_size, input.slice_rows(0, 3), m_density_grid, *forward->density_grid_ctx, forward->grid_density, use_inference_params,
		forward.get()  // Pass context to enable gradient saving for finite diff
	);
		
	// Step 4: Compute 15 surface features using kernel
	// MLP outputs 45D vectors (15 x 3D) starting from channel 0
	auto surface_features_slice = forward->rgb_network_input.slice_rows(1, 15);
	linear_kernel(compute_surface_features_from_vectors_kernel<T>, 0, stream,
		batch_size,
		forward->density_network_output.data(),  // Use channels 0-44 (45 features for 15 3D vectors)
		forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
		forward->analytical_normals.data(),
		forward->analytical_normals.layout() == AoS ? 3 : 1,
		surface_features_slice.data(),
		surface_features_slice.layout() == AoS ? surface_features_slice.stride() : 1
	);
		
	// Step 5: Replace channel 0 with grid density (extract from padded grid output)
	linear_kernel(replace_first_channel_kernel<T>, 0, stream,
		batch_size,
		forward->grid_density.data(),
		forward->grid_density.layout() == AoS ? forward->grid_density.stride() : 1,  // grid stride (padded)
		forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1,  // output stride
		forward->rgb_network_input.data(),
		false  // Old monolithic - surface_explicit uses grid for normals
	);
		
		// Step 6: Direction encoding (same as surface mode)
		dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
	} else if (m_method == "baseline_explicit") {
		// baseline_explicit: density from grid + 15D features from MLP (no surface computation)
		
		
		forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
		
		// Step 1: Get density from grid
		forward->grid_density = GPUMatrixDynamic<T>{m_density_grid->padded_output_width(), batch_size, stream, m_density_grid->preferred_output_layout()};
		forward->density_grid_ctx = m_density_grid->forward(
			stream,
			input.slice_rows(0, 3),  // positions only
			&forward->grid_density,
			use_inference_params,
			false  // don't need input gradients
		);
		

		
		// Step 2: MLP forward pass (15D features)
		forward->density_network_ctx = m_density_network->forward(
			stream, forward->density_network_input, &forward->density_network_output, 
			use_inference_params, false
		);
		

		
		// Step 3: Copy grid density to RGB input[0]
		uint32_t grid_src_stride = forward->grid_density.layout() == AoS ? forward->grid_density.stride() : 1;
		uint32_t rgb_dst_stride = forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1;

		
		linear_kernel(replace_first_channel_kernel<T>, 0, stream,
			batch_size,
			forward->grid_density.data(),
			grid_src_stride,
			rgb_dst_stride,
			forward->rgb_network_input.data(),
			true  // Old monolithic: baseline_explicit applies ReLU (grid stores raw density)
		);
		
		// Step 4: Copy MLP features[0-14] to RGB input[1-15]
		uint32_t mlp_src_stride = forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1;

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
		dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
	} else if (m_method == "hash_surface") {
			// hash_surface mode: Extract density features from hash, pass to density MLP, then compute surface features
			
			// Extract density features from hash interpolation
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // 32 / 4 = 8 levels
			uint32_t padded_density_input_width = m_density_network->input_width(); // Use actual network input width (should be 16)
			

			
			// Create padded matrix for density features
			GPUMatrixDynamic<T> padded_density_features{padded_density_input_width, batch_size, stream, forward->density_network_input.layout()};
			
			// Clear the matrix (set to zero)
			CUDA_CHECK_THROW(cudaMemsetAsync(padded_density_features.data(), 0, padded_density_features.n_bytes(), stream));
			

			
			// Extract density features to first n_levels rows (every 4th feature from hash interpolation)
			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels, // 8 levels
				4, // 4 features per level
				forward->density_network_input.data(),
				forward->density_network_input.layout() == AoS ? forward->density_network_input.stride() : 1,
				padded_density_features.data(), // Write to first n_levels rows
				padded_density_features.layout() == AoS ? padded_density_features.stride() : 1
			);
			
			// Create separate buffer for 1D density output
			forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
			
			// Pass padded density features to density MLP
			
			forward->density_network_ctx = m_density_network->forward(stream, padded_density_features, &forward->density_network_output, use_inference_params, true);
			
			// Compute analytical normals from density MLP output
			forward->analytical_normals = compute_analytical_normals_forward_unified(
				stream, batch_size, input, forward, use_inference_params
			);
			
			// Compute hash surface features: ReLU(-vector_potential_k · normals) for each level (NO density)
			uint32_t surface_features = n_levels; // n_levels hash surface features (no density)
			auto hash_surface_features_slice = forward->rgb_network_input.slice_rows(0, surface_features);
			
			linear_kernel(compute_hash_surface_features_kernel<T>, 0, stream,
				batch_size,
				n_levels, // Dynamic number of levels
				4, // 4 features per level
				forward->density_network_input.data(), // Use full 32D hash features from pos encoding
				forward->density_network_input.layout() == AoS ? forward->density_network_input.stride() : 1,
				forward->analytical_normals.data(), // 3D analytical normals
				hash_surface_features_slice.data(), // Output: n_levels surface features (no density)
				hash_surface_features_slice.layout() == AoS ? hash_surface_features_slice.stride() : 1
			);
			
			// Copy density MLP output[0] to RGB input after surface features
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1,
				forward->density_network_output.data(),
				forward->rgb_network_input.data() + (forward->rgb_network_input.layout() == AoS ? surface_features : surface_features * batch_size),
				m_use_sdf,  // SDF mode flag
				m_variance_network ? m_variance_network->params() : nullptr,  // Variance params
				false  // MLP already has activation
			);
			
			// Direction encoding goes after the surface features + density
			dir_out = forward->rgb_network_input.slice_rows(surface_features + 1, m_dir_encoding->padded_output_width());
		} else {
			// Baseline mode (including SDF mode - use same pattern)
			forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
			dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());


			
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, false);
			

		}

		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);
		
		

		// For surface_normal mode, encode normals using direction encoding
		if (m_method == "surface_normal") {
			
			
			// Get slice for the encoded normals (same width as direction encoding)
			uint32_t normal_start_idx = 16 + m_dir_encoding->padded_output_width();
			auto normals_section = forward->rgb_network_input.slice_rows(normal_start_idx, m_dir_encoding->padded_output_width());
			
			// Create a float matrix view of the analytical normals compatible with encoding input
			GPUMatrixDynamic<float> normals_for_encoding{3, batch_size, stream, forward->analytical_normals.layout()};
			CUDA_CHECK_THROW(cudaMemcpyAsync(normals_for_encoding.data(), forward->analytical_normals.data(), 
				forward->analytical_normals.n_bytes(), cudaMemcpyDeviceToDevice, stream));
			
			// Encode normals using the same encoding as view directions
			forward->normal_encoding_ctx = m_dir_encoding->forward(
				stream,
				normals_for_encoding,
				&normals_section,
				use_inference_params,
				prepare_input_gradients  // Enable gradients for normal encoding
			);

			
		}

		// For surface_reflect mode, encode reflection vectors using direction encoding
		if (m_method == "surface_reflect") {
			// Get slice for the encoded reflection vectors (same width as direction encoding)
			uint32_t reflect_start_idx = 16 + m_dir_encoding->padded_output_width();
			auto reflection_section = forward->rgb_network_input.slice_rows(reflect_start_idx, m_dir_encoding->padded_output_width());
			
			// Extract view directions from input (use directly without copying)
			auto view_dirs_input = input.slice_rows(m_dir_offset, m_dir_encoding->input_width());
			
			// Compute reflection vectors directly from input view directions
			GPUMatrixDynamic<float> reflection_vectors{3, batch_size, stream, forward->analytical_normals.layout()};
			
			linear_kernel(calculate_reflection_vector_kernel<T>, 0, stream,
				batch_size,
				view_dirs_input.data(),                      // Use view direction input directly
				forward->analytical_normals.data(),         // Analytical normals
				reflection_vectors.data(),                   // Output reflection vectors
				m_dir_encoding->input_width(),               // Width of view direction input
				view_dirs_input.layout() == AoS ? view_dirs_input.m() : 1,  // View stride
				forward->analytical_normals.layout() == AoS ? 3 : 1, // Normal stride
				reflection_vectors.layout() == AoS ? 3 : 1   // Reflection stride
			);
			
			// Encode reflection vectors using the same encoding as view directions
			forward->reflection_encoding_ctx = m_dir_encoding->forward(
				stream,
				reflection_vectors,
				&reflection_section,
				use_inference_params,
				prepare_input_gradients  // Enable gradients for reflection encoding
			);
		}

		if (output) {
			forward->rgb_network_output = GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}
		
		

		
		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		

	if (output) {
		// Extract density to output
		if (m_method == "surface_explicit" || m_method == "baseline_explicit") {
			// surface_explicit/baseline_explicit: Extract density from grid, not MLP
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				forward->grid_density.layout() == AoS ? forward->grid_density.stride() : 1,  // Use actual stride from padded grid
				output->layout() == AoS ? padded_output_width() : 1,
				forward->grid_density.data(),
				output->data() + 3 * (output->layout() == AoS ? 1 : batch_size),
				false,  // grid stores log-density, not SDF
				nullptr,
				true  // Apply exp() activation to grid density
			);
		} else {
			// Other modes extract density from density_network_output
			linear_kernel(extract_density<T>, 0, stream,
				batch_size, 
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				output->layout() == AoS ? padded_output_width() : 1,
				forward->density_network_output.data(), 
				output->data() + 3 * (output->layout() == AoS ? 1 : batch_size),
				m_use_sdf,
				m_variance_network ? m_variance_network->params() : nullptr,  // Variance params
				false  // MLP already has activation
			);
		}
	}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
) override {
	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
	uint32_t batch_size = input.n();
	
	GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));

		linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);
		
		const GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		
	// CRITICAL: For surface modes, force AoS layout to match Frequency encoding behavior
	MatrixLayout surface_layout = (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "hash_surface" || m_method == "surface_explicit") ? AoS : m_dir_encoding->preferred_output_layout();
	GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, surface_layout};
		
		// CRITICAL FIX: Zero out the RGB network input gradient buffer to prevent accumulation into uninitialized memory
		// This is especially important for surface_normal mode which has larger buffers with potentially unused sections
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0, dL_drgb_network_input.n_bytes(), stream));
		
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
		
		// Backprop through dir encoding
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			GPUMatrixDynamic<T> dL_ddir_encoding_output;
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "surface_explicit") {
			dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
		} else if (m_method == "hash_surface") {
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // Dynamic levels calculation
			uint32_t surface_features = n_levels; // n_levels hash surface features (no density)
			dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(surface_features + 1, m_dir_encoding->padded_output_width());
		} else {
			dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		}
			
			GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			GPUMatrixDynamic<T> dir_encoding_forward_output;
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "surface_explicit") {
			dir_encoding_forward_output = forward.rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
		} else if (m_method == "hash_surface") {
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // Dynamic levels calculation
			uint32_t surface_features = n_levels; // n_levels hash surface features (no density)
			dir_encoding_forward_output = forward.rgb_network_input.slice_rows(surface_features, m_dir_encoding->padded_output_width());
		} else {
			dir_encoding_forward_output = forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_encoding_forward_output,
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
			
		}
		


		// Map gradients from surface features back to density outputs
		GPUMatrixDynamic<T> dL_ddensity_network_output;
	if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" || m_method == "volume" || m_method == "surface_explicit") {
		// CRITICAL: For surface/volume modes, use AoS layout for density gradient buffer to ensure copy compatibility
		// This forces SphericalHarmonics to behave like Frequency encoding
		dL_ddensity_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
	} else if (m_method == "hash_surface") {
		// hash_surface: Separate buffer for 1D density gradients
		dL_ddensity_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
	} else if (m_method == "baseline_explicit") {
		// baseline_explicit: Separate buffer with AoS layout to match forward pass
		dL_ddensity_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
	} else {
		// Baseline mode (including SDF mode - use same pattern)
			dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		}

		// SDF mode: Apply SDF-to-density gradient correction for density path
		// Both normal gradients and density gradients should accumulate into SDF[0]
		if (m_use_sdf) {
			// Get gradients w.r.t. density from the output (channel 3)
			auto dL_ddensity_from_output = dL_doutput.slice_rows(3, 1);
			
			// Create temporary buffer for SDF gradients from density path
			GPUMatrixDynamic<T> dL_dsdf_from_density{m_density_network->padded_output_width(), batch_size, stream, forward.density_network_output.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf_from_density.data(), 0, dL_dsdf_from_density.n_bytes(), stream));
			
			// Apply SDF backward transformation to convert density gradients to SDF gradients
			linear_kernel(extract_density_backward<T>, 0, stream,
				batch_size,
				dL_dsdf_from_density.layout() == AoS ? dL_dsdf_from_density.stride() : 1,
				dL_ddensity_from_output.layout() == AoS ? padded_output_width() : 1,
				dL_ddensity_from_output.data(),
				dL_dsdf_from_density.data(), // Write to temporary buffer
				forward.density_network_output.data(), // Original SDF values for derivative computation
				nullptr, // TODO: Implement variance gradients later
				true,  // sdf_mode
				m_variance_network ? m_variance_network->params() : nullptr  // Variance params
			);
			
			// ACCUMULATE SDF gradients from density path into main buffer (only channel 0)
			linear_kernel(accumulate_sdf_gradients_kernel<T>, 0, stream,
				batch_size,
				dL_dsdf_from_density.layout() == AoS ? dL_dsdf_from_density.stride() : 1, // Source stride
				dL_dsdf_from_density.data(),  // Source: SDF gradients from density
				dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1, // Target stride  
				dL_ddensity_network_output.data()  // Target: accumulate into channel 0
			);
		}

		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
			// Backward pass: gradients from RGB slice back to 48D density output using ANALYTICAL NORMALS
			// NOTE: Now WITH gradient flow through normals back to density network parameters
			auto dL_dsurface_slice = dL_drgb_network_input.slice_rows(0, 16);
			
			// Compute gradients w.r.t. normals from surface features
			GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, forward.analytical_normals.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals.data(), 0, dL_dnormals.n_bytes(), stream));
			
			// For both surface and surface_normal modes, compute gradients from surface features
			linear_kernel(surface_features_slice_backward_kernel<T>, 0, stream,
				batch_size,
				dL_dsurface_slice.layout() == AoS ? dL_dsurface_slice.stride() : 1,
				dL_dsurface_slice.data(),
				forward.analytical_normals.data(),  // Pass analytical normals
				dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1,
				forward.density_network_output.data(),  // Pass density output for ReLU condition
				dL_ddensity_network_output.data(),
				dL_dnormals.data(),  // Collect gradients w.r.t. normals
				m_use_sdf,  // SDF mode flag
				m_variance_network ? m_variance_network->params() : nullptr  // Variance params
			);
			
			// For surface_normal mode, add gradients from encoded normals
			if (m_method == "surface_normal") {
				// Get gradients from the encoded normals section
				uint32_t normal_start_idx = 16 + m_dir_encoding->padded_output_width();
				auto dL_dencoded_normals = dL_drgb_network_input.slice_rows(normal_start_idx, m_dir_encoding->padded_output_width());
				
				// Create gradient buffer for normal encoding input (3D normals)
				GPUMatrixDynamic<float> dL_dnormal_encoding_input{3, batch_size, stream, forward.analytical_normals.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormal_encoding_input.data(), 0, dL_dnormal_encoding_input.n_bytes(), stream));
				
				// Backpropagate through normal encoding
				GPUMatrixDynamic<float> normals_for_encoding{3, batch_size, stream, forward.analytical_normals.layout()};
				CUDA_CHECK_THROW(cudaMemcpyAsync(normals_for_encoding.data(), forward.analytical_normals.data(), 
					forward.analytical_normals.n_bytes(), cudaMemcpyDeviceToDevice, stream));
				
				// Get the encoded normals from forward pass
				auto encoded_normals_forward = forward.rgb_network_input.slice_rows(normal_start_idx, m_dir_encoding->padded_output_width());
				
				m_dir_encoding->backward(
					stream,
					*forward.normal_encoding_ctx,
					normals_for_encoding,
					encoded_normals_forward,
					dL_dencoded_normals,
					&dL_dnormal_encoding_input,
					use_inference_params,
					GradientMode::Ignore  // Don't affect encoding parameters, just get gradients
				);
				
				// Accumulate gradients from normal encoding to main normal gradients
				linear_kernel(add_to_buffer_kernel<float>, 0, stream,
					dL_dnormal_encoding_input.n_elements(),
					dL_dnormal_encoding_input.data(),
					dL_dnormals.data()
				);
				
			}
			
			// For surface_reflect mode, add gradients from encoded reflection vectors
			if (m_method == "surface_reflect") {
				// Get gradients from the encoded reflection vectors section
				uint32_t reflect_start_idx = 16 + m_dir_encoding->padded_output_width();
				auto dL_dencoded_reflection = dL_drgb_network_input.slice_rows(reflect_start_idx, m_dir_encoding->padded_output_width());
				
				// Create gradient buffer for reflection encoding input (3D reflection vectors)
				GPUMatrixDynamic<float> dL_dreflection_encoding_input{3, batch_size, stream, forward.analytical_normals.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dreflection_encoding_input.data(), 0, dL_dreflection_encoding_input.n_bytes(), stream));
				
				// Extract view directions from input (use directly without copying)
				auto view_dirs_input = input.slice_rows(m_dir_offset, m_dir_encoding->input_width());
				
				// Recompute reflection vectors using the same direct approach as forward pass
				GPUMatrixDynamic<float> reflection_vectors{3, batch_size, stream, forward.analytical_normals.layout()};
				linear_kernel(calculate_reflection_vector_kernel<T>, 0, stream,
					batch_size,
					view_dirs_input.data(),                      // Use view direction input directly
					forward.analytical_normals.data(),
					reflection_vectors.data(),
					m_dir_encoding->input_width(),               // Width of view direction input
					view_dirs_input.layout() == AoS ? view_dirs_input.m() : 1,  // View stride
					forward.analytical_normals.layout() == AoS ? 3 : 1,
					reflection_vectors.layout() == AoS ? 3 : 1
				);
				
				// Get the encoded reflection vectors from forward pass
				auto encoded_reflection_forward = forward.rgb_network_input.slice_rows(reflect_start_idx, m_dir_encoding->padded_output_width());
				
				// Backpropagate through reflection encoding
				m_dir_encoding->backward(
					stream,
					*forward.reflection_encoding_ctx,
					reflection_vectors,
					encoded_reflection_forward,
					dL_dencoded_reflection,
					&dL_dreflection_encoding_input,
					use_inference_params,
					GradientMode::Ignore  // Don't affect encoding parameters, just get gradients
				);
				
				// Create gradient buffer for normals from reflection computation
				GPUMatrixDynamic<float> dL_dnormals_from_reflection{3, batch_size, stream, forward.analytical_normals.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals_from_reflection.data(), 0, dL_dnormals_from_reflection.n_bytes(), stream));
				
				// Backward through reflection vector computation using direct view direction input (no copying)
				linear_kernel(reflection_vector_backward_kernel<T>, 0, stream,
					batch_size,
					view_dirs_input.data(),                      // Use view direction input directly
					forward.analytical_normals.data(),         // Analytical normals
					dL_dreflection_encoding_input.data(),      // Gradients w.r.t. reflection vectors
					m_dir_encoding->input_width(),               // Width of view direction input
					view_dirs_input.layout() == AoS ? view_dirs_input.m() : 1,  // View stride
					forward.analytical_normals.layout() == AoS ? 3 : 1,        // Normal stride
					dL_dreflection_encoding_input.layout() == AoS ? 3 : 1,     // Reflection gradient stride
					dL_dnormals_from_reflection.data()         // Output: gradients w.r.t. normals
				);
				
				// Accumulate gradients from reflection to main normal gradients
				linear_kernel(add_to_buffer_kernel<float>, 0, stream,
					dL_dnormals_from_reflection.n_elements(),
					dL_dnormals_from_reflection.data(),
					dL_dnormals.data()
				);
				
				// Note: View direction gradients are not computed in backward pass to avoid complexity
				// This is acceptable since view directions are typically fixed inputs (camera rays)
			}
			
			// Backpropagate gradients through analytical normals to density network parameters
			// accumulate_analytical_normal_gradients(
			// 	stream, batch_size, input, forward, dL_dnormals,
			// 	dL_ddensity_network_output, use_inference_params, param_gradients_mode
			// );
		} else if (m_method == "hash_surface") {
			// Backward pass: gradients from hash surface features back to hash interpolation and density
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // Dynamic levels calculation
			uint32_t surface_features = n_levels; // n_levels hash surface features (no density)
			auto dL_dhash_surface_slice = dL_drgb_network_input.slice_rows(0, surface_features);
			
			// Create gradient buffers
			GPUMatrixDynamic<T> dL_dhash_features{m_pos_encoding->padded_output_width(), batch_size, stream, forward.density_network_input.layout()};
			GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, forward.analytical_normals.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dhash_features.data(), 0, dL_dhash_features.n_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals.data(), 0, dL_dnormals.n_bytes(), stream));
			
			// Backward through hash surface features computation

			linear_kernel(hash_surface_features_backward_kernel<T>, 0, stream,
				batch_size,
				n_levels, // Dynamic number of levels
				4, // 4 features per level
				forward.density_network_input.data(), // Hash interpolation output (n_levels * 4)
				forward.density_network_input.layout() == AoS ? forward.density_network_input.stride() : 1,
				forward.analytical_normals.data(), // 3D analytical normals
				dL_dhash_surface_slice.data(), // Gradients w.r.t. hash surface features (n_levels)
				dL_dhash_surface_slice.layout() == AoS ? dL_dhash_surface_slice.stride() : 1,
				dL_dhash_features.data(), // Output: gradients w.r.t. hash features (n_levels * 4)
				dL_dnormals.data() // Output: gradients w.r.t. normals (3D)
			);

			// Extract gradient w.r.t. density from RGB input and accumulate to density MLP output
			auto dL_ddensity_from_rgb = dL_drgb_network_input.slice_rows(surface_features, 1);
			linear_kernel(add_to_buffer_kernel<T>, 0, stream,
				batch_size,
				dL_ddensity_from_rgb.data(),
				dL_ddensity_network_output.data()
			);



			// Backpropagate gradients through analytical normals to density network parameters
			accumulate_analytical_normal_gradients(
				stream, batch_size, input, forward, dL_dnormals,
				dL_ddensity_network_output, use_inference_params, param_gradients_mode
			);
			
			// Backpropagate through density MLP (from 1D density to extracted density features)
			uint32_t extracted_density_features_size = m_density_network->input_width(); // 16D padded (8 actual + 8 padding)
			GPUMatrixDynamic<T> dL_dextracted_density_features{extracted_density_features_size, batch_size, stream, forward.density_network_input.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dextracted_density_features.data(), 0, dL_dextracted_density_features.n_bytes(), stream));
			
			// Extract density features again for backward pass
			GPUMatrixDynamic<T> extracted_density_features{extracted_density_features_size, batch_size, stream, forward.density_network_input.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(extracted_density_features.data(), 0, extracted_density_features.n_bytes(), stream));

			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels,
				4,
				forward.density_network_input.data(),
				forward.density_network_input.layout() == AoS ? forward.density_network_input.stride() : 1,
				extracted_density_features.data(),
				extracted_density_features.layout() == AoS ? extracted_density_features.stride() : 1
			);
			
			// Backward through density MLP
			
			m_density_network->backward(
				stream,
				*forward.density_network_ctx,
				extracted_density_features,
				forward.density_network_output,
				dL_ddensity_network_output,
				&dL_dextracted_density_features,
				use_inference_params,
				param_gradients_mode
			);

			// Accumulate gradients from extracted density features back to hash features (every 4th feature)
			for (uint32_t level = 0; level < n_levels; ++level) {
				linear_kernel(accumulate_density_gradient_to_hash_kernel<T>, 0, stream,
					batch_size,
					level, // level index
					4, // features per level
					dL_dextracted_density_features.data(),
					dL_dextracted_density_features.layout() == AoS ? dL_dextracted_density_features.stride() : 1,
					level, // gradient index in extracted features
					dL_dhash_features.data(),
					dL_dhash_features.layout() == AoS ? dL_dhash_features.stride() : 1
				);
			}
			
			// Backpropagate through position encoding for hash_surface
			if (m_pos_encoding->n_params() > 0 || dL_dinput) {
				GPUMatrixDynamic<float> dL_dpos_encoding_input;
				if (dL_dinput) {
					dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
				}
				
				m_pos_encoding->backward(
					stream,
					*forward.pos_encoding_ctx,
					input.slice_rows(0, m_pos_encoding->input_width()),
					forward.density_network_input,
					dL_dhash_features,  // Use hash features gradients instead of density network input gradients
					dL_dinput ? &dL_dpos_encoding_input : nullptr,
					use_inference_params,
					param_gradients_mode
				);
			}
		} else if (m_method == "volume") {
			// Backward pass: gradients from RGB slice back to 48D density output using VOLUME DIVERGENCES
			auto dL_dvolume_slice = dL_drgb_network_input.slice_rows(0, 16);
			
			// Compute gradients w.r.t. divergences from volume features
			GPUMatrixDynamic<float> dL_ddivergences{15, batch_size, stream, forward.volume_divergences.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddivergences.data(), 0, dL_ddivergences.n_bytes(), stream));
			
			// Compute gradients from volume features to density output and divergences
			linear_kernel(volume_divergence_backward_kernel<T>, 0, stream,
				batch_size,
				dL_dvolume_slice.layout() == AoS ? dL_dvolume_slice.stride() : 1,
				dL_dvolume_slice.data(),
				dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1,
				dL_ddensity_network_output.data(),
				dL_ddivergences.data()  // Collect gradients w.r.t. divergences
			);
			
		// Backpropagate gradients from divergences back to density network parameters
		accumulate_volume_divergence_gradients(
			stream, batch_size, input, forward, dL_ddivergences,
			dL_ddensity_network_output, use_inference_params, param_gradients_mode
		);
	} else if (m_method == "surface_explicit") {
		// surface_explicit: backprop to both grid and MLP
		
		// Step 1: Backprop through surface features to get gradients for 45-D MLP output
		auto dL_dsurface_slice = dL_drgb_network_input.slice_rows(1, 15);
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
		
		linear_kernel(backprop_surface_features_kernel<T>, 0, stream,
			batch_size,
			dL_dsurface_slice.data(),
			dL_dsurface_slice.layout() == AoS ? dL_dsurface_slice.stride() : 1,
			forward.analytical_normals.data(),
			forward.analytical_normals.layout() == AoS ? 3 : 1,
			dL_ddensity_network_output.data(),  // Output to features (skip channel 0)
			dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1
		);
		
		// Step 2: Extract gradient for channel 0 (goes to grid) - allocate with padding
		// Use grid's preferred layout (SoA) to match hash encoding
		GPUMatrixDynamic<T> dL_dgrid_density{m_density_grid->padded_output_width(), batch_size, stream, m_density_grid->preferred_output_layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dgrid_density.data(), 0, dL_dgrid_density.n_bytes(), stream));

		// Step 2a: Extract gradient from RGB network input (channel 0)
		linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
			batch_size,
			dL_drgb_network_input.data(),
			dL_drgb_network_input.layout() == AoS ? dL_drgb_network_input.stride() : 1,
			dL_dgrid_density.layout() == AoS ? dL_dgrid_density.stride() : 1,  // grid stride
			dL_dgrid_density.data(),
			forward.grid_density.data(),  // Forward grid values (not used when apply_exp=false)
			false  // Don't apply exp() - gradients already w.r.t. log-density from fused kernels
		);

		// Step 2b: ACCUMULATE density gradient from RGBD output (channel 3) to grid
		// This is the gradient from the rendering loss w.r.t. density
		linear_kernel(accumulate_density_gradient_to_grid_kernel<T>, 0, stream,
			batch_size,
			dL_doutput.data(),
			dL_doutput.layout() == AoS ? dL_doutput.m() : 1,  // RGBD output stride
			dL_dgrid_density.data(),
			dL_dgrid_density.layout() == AoS ? dL_dgrid_density.stride() : 1,  // grid stride
			dL_doutput.m(),  // Number of rows (channels) = 4 for RGBD
			forward.grid_density.data(),  // Forward grid values (not used when apply_exp=false)
			false  // Don't apply exp() - gradients already w.r.t. log-density from fused kernels
		);

		// Step 2.5: Backprop gradients from normals to grid (for finite differences)
		// This computes dL/dnormals from the surface features and backprops through finite diff
		GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, forward.analytical_normals.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals.data(), 0, dL_dnormals.n_bytes(), stream));
		
		linear_kernel(backprop_surface_features_to_normals_kernel<T>, 0, stream,
			batch_size,
			dL_dsurface_slice.data(),
			dL_dsurface_slice.layout() == AoS ? dL_dsurface_slice.stride() : 1,
			forward.density_network_output.data() ,
			forward.density_network_output.layout() == AoS ? forward.density_network_output.stride() : 1,
			dL_dnormals.data(),
			dL_dnormals.layout() == AoS ? 3 : 1
		);
		
		// Check if we're using finite differences and have saved contexts
		const char* grad_method_env = std::getenv("NGP_GRAD_METHOD");
		bool use_finite_diff = (grad_method_env && std::string(grad_method_env) == "finite");
		
		if (use_finite_diff && !forward.finite_diff_contexts.empty()) {
			backprop_normals_from_finite_differences(
				stream, batch_size, dL_dnormals, forward.analytical_normals,
				m_density_grid, forward, use_inference_params, param_gradients_mode
			);
		}
		
		// NOTE: For surface_explicit, we do NOT call accumulate_analytical_normal_gradients()
		// because the normals come from the GRID, not from the MLP density output.
		// The gradient flow is:
		//   - dL/dnormals → grid (via backprop_normals_from_finite_differences or grid->backward)
		//   - dL/dnormals → MLP vectors (already handled by backprop_surface_features_to_normals_kernel)
		// The MLP does NOT output density (channel 0), only feature vectors (channels 1-47).
		
		// Step 3: Backprop to density network parameters
		GPUMatrixDynamic<T> dL_ddensity_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}
		m_density_network->backward(
			stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output,
			dL_ddensity_network_output, dL_ddensity_input.data() ? &dL_ddensity_input : nullptr, use_inference_params, param_gradients_mode
		);
				
		// Step 4: Backprop to grid parameters
		// Note: We don't need dL_dinput from grid since we already get it from hash encoding
		if (m_density_grid->n_params() > 0) {
			m_density_grid->backward(
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
			GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}
			
			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
		
		// Don't do the standard density network backward for surface_explicit
		// Skip the standard backprop below by zeroing dL_ddensity_network_output
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
	} else if (m_method == "baseline_explicit") {
		// baseline_explicit: backprop to both grid and MLP (simple feature copy, no surface features)

		
		// Step 1: Extract gradients for MLP features from RGB input[1-15]
		// NOTE: dL_ddensity_network_output is now allocated separately with AoS layout (not zeroed by allocation above)
		
		uint32_t rgb_src_stride = dL_drgb_network_input.layout() == AoS ? dL_drgb_network_input.stride() : 1;
		uint32_t mlp_dst_stride = dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1;

		
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
		GPUMatrixDynamic<T> dL_dgrid_density{m_density_grid->padded_output_width(), batch_size, stream, m_density_grid->preferred_output_layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dgrid_density.data(), 0, dL_dgrid_density.n_bytes(), stream));
		
		
		
		uint32_t grid_dst_stride = dL_dgrid_density.layout() == AoS ? dL_dgrid_density.stride() : 1;
		
		
		linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
			batch_size,
			dL_drgb_network_input.data(),
			rgb_src_stride,
			grid_dst_stride,
			dL_dgrid_density.data(),
			forward.grid_density.data(),  // Forward grid values (not used when apply_exp=false)
			false  // Don't apply exp() - gradients already w.r.t. log-density from fused kernels
		);
		
		// Step 2b: ACCUMULATE density gradient from RGBD output (channel 3) to grid
		uint32_t rgbd_src_stride = dL_doutput.layout() == AoS ? dL_doutput.m() : 1;
		
		
		
		linear_kernel(accumulate_density_gradient_to_grid_kernel<T>, 0, stream,
			batch_size,
			dL_doutput.data(),
			rgbd_src_stride,  // For AoS: stride=4, for SoA: stride=1
			dL_dgrid_density.data(),
			grid_dst_stride,
			dL_doutput.m(),  // Number of rows (channels) = 4 for RGBD
			forward.grid_density.data(),  // Forward grid values (not used when apply_exp=false)
			false  // Don't apply exp() - gradients already w.r.t. log-density from fused kernels
		);
		
		// Step 3: Backprop to MLP
		GPUMatrixDynamic<T> dL_ddensity_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}
		
		
		
		m_density_network->backward(
			stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output,
			dL_ddensity_network_output, dL_ddensity_input.data() ? &dL_ddensity_input : nullptr, use_inference_params, param_gradients_mode
		);
		

		
		// Step 4: Backprop to grid
		if (m_density_grid->n_params() > 0) {
			
			
			m_density_grid->backward(
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
			GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}
			
			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
		
		// Skip standard backprop
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
	}
	
	// Add gradient from final RGBD output (alpha blending)
		// if (m_method == "surface") {
		// 	// In surface mode, alpha gradient goes to RGB input[0]
		// 	linear_kernel(add_density_gradient<T>, 0, stream,
		// 		batch_size,
		// 		dL_doutput.m(),
		// 		dL_doutput.data(),
		// 		dL_drgb_network_input.layout() == RM ? 1 : dL_drgb_network_input.stride(),
		// 		dL_drgb_network_input.data()
		// 	);
		// } else {
		// 	// In baseline mode, alpha gradient goes to density output[0]  
		// 	linear_kernel(add_density_gradient<T>, 0, stream,
		// 		batch_size,
		// 		dL_doutput.m(),
		// 		dL_doutput.data(),
		// 		dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
		// 		dL_ddensity_network_output.data()
		// 	);
	// }
	
	// CRITICAL: For surface_explicit and baseline_explicit, density comes from GRID, not MLP
	// So density gradients from RGBD output should NOT go to dL_ddensity_network_output
	if (m_method != "surface_explicit" && m_method != "baseline_explicit") {

		
		linear_kernel(add_density_gradient<T>, 0, stream,
				batch_size,
				dL_doutput.m(),
				dL_doutput.data(),
				dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
				dL_ddensity_network_output.data()
			);
		

	}
	// NOTE: No gradient clipping needed - working NeuS2 implementations don't use it

	GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput || m_backprop_normals) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

	// Skip density network backward for hash_surface, surface_explicit, and baseline_explicit since they were already handled
	if (m_method != "hash_surface" && m_method != "surface_explicit" && m_method != "baseline_explicit") {

		
		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

	}

	// Backprop through pos encoding (skip for hash_surface, surface_explicit, and baseline_explicit since they're handled specially)
	if (dL_ddensity_network_input.data() && m_method != "hash_surface" && m_method != "surface_explicit" && m_method != "baseline_explicit") {
					GPUMatrixDynamic<float> dL_dpos_encoding_input;
		if (dL_dinput) {
			dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
		}
		
			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		// OPTIONAL: Accumulate gradients w.r.t. variance parameter
		if (m_variance_network && m_use_sdf) {
			// Create a simple kernel to accumulate variance gradients from SDF-to-density conversion
			// This is a simplified version - full implementation would require more complex gradient computation
			
			// For now, add a small regularization term to keep variance reasonable
			T regularization_weight = T(0.001f);
			T target_variance = T(0.12f);  // Target log-variance
			
			// Simple L2 regularization: weight * (variance - target)²
			if (m_variance_network->params()) {
				T current_variance = m_variance_network->params()[0];
				T variance_error = current_variance - target_variance;
				T variance_grad = T(2.0f) * regularization_weight * variance_error;
				
				// This is a simplified approach - in practice you'd want proper gradient accumulation
				// from the SDF-to-density conversion chain rule
				// For now, we just add regularization to the gradients if they exist
				if (m_variance_network->gradients()) {
					m_variance_network->gradients()[0] += variance_grad;
				}
			}
		}
	}

	/**
	 * @brief Computes analytical normals during forward pass using automatic differentiation.
	 * 
	 * Uses the chain rule to compute ∇SDF (gradients of SDF w.r.t. position) by:
	 * 1. Seeding gradients at SDF output (channel 0 = 1.0)
	 * 2. Backpropagating through density network
	 * 3. Backpropagating through position encoding
	 * 
	 * Handles hash_surface method specially by extracting/distributing density features.
	 * Stores raw gradients in forward context for Eikonal loss computation.
	 * 
	 * @param stream CUDA stream for operations
	 * @param batch_size Number of samples to process
	 * @param input Full network input (position + direction + extras)
	 * @param forward Forward context containing network outputs and contexts (reused)
	 * @param use_inference_params Whether to use inference or training parameters
	 * @return Processed normals (normalized or clamped based on settings)
	 * 
	 * @note Use this during forward pass when you have a ForwardContext.
	 *       For inference-only, use compute_analytical_normals_inference_unified().
	 */
	GPUMatrixDynamic<float> compute_analytical_normals_forward_unified(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		std::unique_ptr<ForwardContext>& forward,
		bool use_inference_params
	) {
		// Step 1: Create gradient seed for SDF channel (channel 0 = 1.0, others = 0.0)
		GPUMatrixDynamic<T> dL_dsdf_seed{m_density_network->padded_output_width(), batch_size, stream, forward->density_network_output.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf_seed.data(), 0, dL_dsdf_seed.n_bytes(), stream));
		
		// Set first channel to 1.0 for all batch elements (NeuS2 pattern)
		linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
			batch_size, T(1.0f),
			dL_dsdf_seed.layout() == AoS ? dL_dsdf_seed.stride() : 1,
			dL_dsdf_seed.data()
		);
		
		// Step 2 & 3: Handle backward pass differently for hash_surface vs other methods
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};  // Use AoS for consistency
		
		if (m_method == "hash_surface") {
			// For hash_surface, we need to handle the gradient flow through extraction process
			// Create gradient buffer for the extracted density features (16D)
			GPUMatrixDynamic<T> dL_dextracted_density{m_density_network->input_width(), batch_size, stream, forward->density_network_input.layout()};
			
			// Get the actual extracted density features used in forward pass
			// We need to re-extract them since forward->density_network_input contains full 32D features
			GPUMatrixDynamic<T> temp_extracted_density{m_density_network->input_width(), batch_size, stream, forward->density_network_input.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(temp_extracted_density.data(), 0, temp_extracted_density.n_bytes(), stream));
			
			// Extract density features again (same as in forward pass)
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4;
			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels,
				4, // 4 features per level
				forward->density_network_input.data(), // This contains the full 32D hash features
				forward->density_network_input.layout() == AoS ? forward->density_network_input.stride() : 1,
				temp_extracted_density.data(),
				temp_extracted_density.layout() == AoS ? temp_extracted_density.stride() : 1
			);
			
			// Backward through density network with extracted features
			m_density_network->backward(
				stream, 
				*forward->density_network_ctx,
				temp_extracted_density,  // Use extracted density features
				forward->density_network_output,
				dL_dsdf_seed,
				&dL_dextracted_density,
				use_inference_params, 
				GradientMode::Ignore  // Don't affect parameter gradients
			);
			
			// Create full 32D gradient buffer for position encoding backward
			GPUMatrixDynamic<T> dL_dhash_features{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dhash_features.data(), 0, dL_dhash_features.n_bytes(), stream));
			
			// Distribute gradients from extracted density features back to full hash features
			// Only the 4th component of each level gets gradients (the density features)
			// Use kernel instead of cudaMemcpy2DAsync for CUDA graph compatibility
			linear_kernel(distribute_hash_density_gradients_kernel<T>, 0, stream,
				batch_size,
				n_levels, // number of levels
				dL_dextracted_density.data(),
				dL_dextracted_density.layout() == AoS ? dL_dextracted_density.stride() : 1,
				dL_dhash_features.data(),
				dL_dhash_features.layout() == AoS ? dL_dhash_features.stride() : 1
			);
			
			// Backward through position encoding with full 32D gradients
			m_pos_encoding->backward(
				stream,
				*forward->pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward->density_network_input,  // This contains the full 32D hash features
				dL_dhash_features,
				&dSDF_dpos,
				use_inference_params,
				GradientMode::Ignore  // Don't affect parameter gradients
			);
		} else {
			// Standard approach for other methods
			// Step 2: Backward through density network with GradientMode::Ignore
			GPUMatrixDynamic<T> dL_ddensity_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			
			m_density_network->backward(
				stream, 
				*forward->density_network_ctx,
				forward->density_network_input,
				forward->density_network_output,
				dL_dsdf_seed,
				&dL_ddensity_input,
				use_inference_params, 
				GradientMode::Ignore  // Don't affect parameter gradients
			);
			
			// Step 3: Backward through position encoding with GradientMode::Ignore
			m_pos_encoding->backward(
				stream,
				*forward->pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward->density_network_input,
				dL_ddensity_input,
				&dSDF_dpos,
				use_inference_params,
				GradientMode::Ignore  // Don't affect parameter gradients
			);
		}
		
		// Store raw gradients for backward pass (including Eikonal loss)
		GPUMatrixDynamic<float> raw_grads = dSDF_dpos.slice_rows(0, 3);
		forward->raw_gradients = GPUMatrixDynamic<float>{3, batch_size, stream, raw_grads.layout()};
		CUDA_CHECK_THROW(cudaMemcpyAsync(forward->raw_gradients.data(), raw_grads.data(), 
			forward->raw_gradients.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		
		// Process gradients according to user settings (normalized or raw with optional clamping)
		GPUMatrixDynamic<float> normals{3, batch_size, stream, AoS};
		linear_kernel(process_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data(),
			m_normalize_normals,      // Whether to normalize to unit vectors
			m_clamp_gradients,        // Whether to clamp gradient magnitude
			m_max_gradient_magnitude  // Maximum allowed gradient magnitude
		);
		
		return normals;
	}

	/**
	 * @brief Computes volume divergences during inference (without existing ForwardContext).
	 * 
	 * Computes ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z for each of 15 3D vector fields.
	 * 
	 * Algorithm:
	 * - For each of 15 vector fields:
	 *   1. Seed gradients for all 3 components [Φ_kx, Φ_ky, Φ_kz]
	 *   2. Backpropagate to get ∂Φ_k/∂position
	 *   3. Extract diagonal: ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z
	 * 
	 * @param stream CUDA stream for operations
	 * @param batch_size Number of samples to process
	 * @param input Full network input
	 * @param density_network_input Output from position encoding
	 * @param density_network_output Output from density network (48D)
	 * @param use_inference_params Whether to use inference parameters
	 * @return 15D divergence values per sample
	 * 
	 * @note Creates temporary contexts. Use compute_volume_divergences_forward() if you
	 *       already have a ForwardContext to reuse.
	 * @note Efficiency: 15 passes through network stack (one per vector field)
	 */
	GPUMatrixDynamic<float> compute_volume_divergences_inference(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& density_network_input,
		const GPUMatrixDynamic<T>& density_network_output,
		bool use_inference_params
	) {
		// IMPROVED: Still 15 gradient computations (one per vector field) but better organized
		// ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z for each of the 15 3D vectors
		// This version reuses contexts efficiently
		
		GPUMatrixDynamic<float> divergences{15, batch_size, stream, AoS};
		CUDA_CHECK_THROW(cudaMemsetAsync(divergences.data(), 0, divergences.n_bytes(), stream));
		
		// For each of the 15 vector fields, compute divergence efficiently
		for (uint32_t k = 0; k < 15; ++k) {
			// Create gradient seed for all 3 components of vector field k: [Φ_kx, Φ_ky, Φ_kz]
			GPUMatrixDynamic<T> dL_dphi_seed{m_density_network->padded_output_width(), batch_size, stream, density_network_output.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dphi_seed.data(), 0, dL_dphi_seed.n_bytes(), stream));
			
			// Set gradient seed for all 3 components of vector field k simultaneously
			for (uint32_t comp = 0; comp < 3; ++comp) {
				uint32_t phi_channel = 1 + k * 3 + comp;  // Phi_k components: [x, y, z]
				linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
					batch_size, T(1.0f),
					dL_dphi_seed.layout() == AoS ? dL_dphi_seed.stride() : 1,
					dL_dphi_seed.data() + phi_channel * (dL_dphi_seed.layout() == AoS ? 1 : batch_size)
				);
			}
			
			// Create temporary contexts for gradient computation (reusing inference setup)
			auto temp_density_ctx = m_density_network->forward(
				stream, density_network_input, const_cast<GPUMatrixDynamic<T>*>(&density_network_output), use_inference_params, true
			);
			
			auto temp_pos_ctx = m_pos_encoding->forward(
				stream, input.slice_rows(0, m_pos_encoding->input_width()), const_cast<GPUMatrixDynamic<T>*>(&density_network_input), use_inference_params, true
			);
			
			// Single backward pass through density network for this vector field
			GPUMatrixDynamic<T> dL_ddensity_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			m_density_network->backward(stream, *temp_density_ctx, density_network_input, density_network_output, dL_dphi_seed, &dL_ddensity_input, use_inference_params, GradientMode::Ignore);
			
			// Single backward pass through position encoding for this vector field
			GPUMatrixDynamic<float> dPhi_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};
			m_pos_encoding->backward(stream, *temp_pos_ctx, input.slice_rows(0, m_pos_encoding->input_width()), density_network_input, dL_ddensity_input, &dPhi_dpos, use_inference_params, GradientMode::Ignore);
			
			// Compute divergence: ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z
			linear_kernel(compute_divergence_diagonal_kernel, 0, stream,
				batch_size,
				dPhi_dpos.data(),  // [3 x batch_size] gradients of [Φ_kx, Φ_ky, Φ_kz] w.r.t. [x, y, z]
				divergences.data() + k * (divergences.layout() == AoS ? 1 : batch_size), // divergence_k output
				dPhi_dpos.layout() == AoS ? dPhi_dpos.stride() : 1
			);
		}
		
		return divergences;
	}

	/**
	 * @brief Computes volume divergences during forward pass (reuses ForwardContext).
	 * 
	 * Computes ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z for each of 15 3D vector fields.
	 * 
	 * Algorithm (maximally efficient):
	 * - For each spatial dimension (x, y, z):
	 *   - For each vector field k (0-14):
	 *     1. Seed gradient for single component Φ_{k,dim}
	 *     2. Backpropagate to get ∂Φ_{k,dim}/∂dim
	 *     3. Accumulate to divergence[k]
	 * 
	 * @param stream CUDA stream for operations
	 * @param batch_size Number of samples to process
	 * @param input Full network input
	 * @param forward Forward context containing network outputs (reused)
	 * @param use_inference_params Whether to use inference parameters
	 * @return 15D divergence values per sample
	 * 
	 * @note Reuses existing contexts from forward pass.
	 * @note Efficiency: 45 passes (3 spatial dims × 15 vector fields)
	 * @warning Memory intensive due to 45 gradient computations
	 */
	GPUMatrixDynamic<float> compute_volume_divergences_forward(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		std::unique_ptr<ForwardContext>& forward,
		bool use_inference_params
	) {
		// EFFICIENT: Only 3×15 = 45 gradient computations total (instead of 15×3 = 45 in the old way)
		// But organized much more efficiently: 3 spatial passes × 15 vector fields per pass
		// ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z for each of the 15 3D vectors
		
		GPUMatrixDynamic<float> divergences{15, batch_size, stream, AoS};
		CUDA_CHECK_THROW(cudaMemsetAsync(divergences.data(), 0, divergences.n_bytes(), stream));
		
		// For each spatial dimension (x=0, y=1, z=2), compute gradients for all vector fields
		for (uint32_t spatial_dim = 0; spatial_dim < 3; ++spatial_dim) {
			
			// For each vector field within this spatial dimension
			for (uint32_t k = 0; k < 15; ++k) {
				// Create gradient seed for ONLY the specific component Φ_{k,spatial_dim}
				GPUMatrixDynamic<T> dL_dphi_seed{m_density_network->padded_output_width(), batch_size, stream, forward->density_network_output.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dphi_seed.data(), 0, dL_dphi_seed.n_bytes(), stream));
				
				// Set gradient seed for only this specific component
				uint32_t phi_channel = 1 + k * 3 + spatial_dim;  // Φ_{k,spatial_dim}
				linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
					batch_size, T(1.0f),
					dL_dphi_seed.layout() == AoS ? dL_dphi_seed.stride() : 1,
					dL_dphi_seed.data() + phi_channel * (dL_dphi_seed.layout() == AoS ? 1 : batch_size)
				);
				
				// Single backward pass through density network for this component
				GPUMatrixDynamic<T> dL_ddensity_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
				m_density_network->backward(
					stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, 
					dL_dphi_seed, &dL_ddensity_input, use_inference_params, GradientMode::Ignore
				);
				
				// Single backward pass through position encoding for this component
				GPUMatrixDynamic<float> dPhi_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};
				m_pos_encoding->backward(
					stream, *forward->pos_encoding_ctx, input.slice_rows(0, m_pos_encoding->input_width()), 
					forward->density_network_input, dL_ddensity_input, &dPhi_dpos, use_inference_params, GradientMode::Ignore
				);
				
				// Extract ∂Φ_{k,spatial_dim}/∂spatial_dim and accumulate to divergence
				// dPhi_dpos[spatial_dim] contains the gradient we want: ∂Φ_{k,spatial_dim}/∂spatial_dim
				linear_kernel(extract_spatial_gradient_kernel, 0, stream,
					batch_size,
					dPhi_dpos.data(),                                // Position gradients [3+ x batch_size]
					spatial_dim,                                     // Which spatial dimension (0=x, 1=y, 2=z)
					dPhi_dpos.layout() == AoS ? dPhi_dpos.stride() : 1,  // Gradient stride
					divergences.layout() == AoS ? divergences.stride() : 1,  // Divergence stride
					k,                                               // Vector field index (0-14)
					divergences.data()                               // Output divergences [15 x batch_size]
				);
			}
		}
		
		return divergences;
	}

	/**
	 * @brief Computes analytical normals during inference (without existing ForwardContext).
	 * 
	 * Uses the chain rule to compute ∇SDF (gradients of SDF w.r.t. position) by:
	 * 1. Creating temporary forward contexts
	 * 2. Seeding gradients at SDF output (channel 0 = 1.0)
	 * 3. Backpropagating through density network
	 * 4. Backpropagating through position encoding
	 * 
	 * Handles hash_surface method specially by extracting/distributing density features.
	 * 
	 * @param stream CUDA stream for operations
	 * @param batch_size Number of samples to process
	 * @param input Full network input (position + direction + extras)
	 * @param density_network_input Output from position encoding
	 * @param density_network_output Output from density network (48D)
	 * @param use_inference_params Whether to use inference parameters
	 * @return Processed normals (normalized or clamped based on settings)
	 * 
	 * @note Creates temporary contexts. Use compute_analytical_normals_forward_unified()
	 *       if you already have a ForwardContext to reuse.
	 * @note ~80% code duplication with forward version - consider refactoring
	 */
	GPUMatrixDynamic<float> compute_analytical_normals_inference_unified(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& density_network_input,
		const GPUMatrixDynamic<T>& density_network_output,
		bool use_inference_params
	) {
		// For hash_surface, we need to handle the position encoding → density extraction differently
		std::unique_ptr<Context> temp_pos_ctx;
		std::unique_ptr<Context> temp_density_ctx;
		
		// Declare these outside the if block so they're accessible in backward pass
		GPUMatrixDynamic<T> temp_hash_features;
		GPUMatrixDynamic<T> temp_density_input;
		
		if (m_method == "hash_surface") {
			// Create separate 32D buffer for full hash features
			temp_hash_features = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			
			// Position encoding outputs to full 32D hash features
			temp_pos_ctx = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				&temp_hash_features,
				use_inference_params,
				true  // prepare_input_gradients
			);
			
			// Create temporary padded density input and extract features
			temp_density_input = GPUMatrixDynamic<T>{m_density_network->input_width(), batch_size, stream, density_network_input.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(temp_density_input.data(), 0, temp_density_input.n_bytes(), stream));
			
			// Extract density features (4th component of each level) into first 8 rows
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4;
			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels,
				4, // 4 features per level
				temp_hash_features.data(),
				temp_hash_features.layout() == AoS ? temp_hash_features.stride() : 1,
				temp_density_input.data(),
				temp_density_input.layout() == AoS ? temp_density_input.stride() : 1
			);
			
			// Density network forward with extracted features
			temp_density_ctx = m_density_network->forward(
				stream,
				temp_density_input,
				const_cast<GPUMatrixDynamic<T>*>(&density_network_output),
				use_inference_params,
				true  // prepare_input_gradients
			);
		} else {
			// Standard approach for other methods
			temp_pos_ctx = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				const_cast<GPUMatrixDynamic<T>*>(&density_network_input),
				use_inference_params,
				true  // prepare_input_gradients
			);
			
			temp_density_ctx = m_density_network->forward(
				stream,
				density_network_input,
				const_cast<GPUMatrixDynamic<T>*>(&density_network_output),
				use_inference_params,
				true  // prepare_input_gradients
			);
		}
		
		// Compute gradients using same pattern as forward
		GPUMatrixDynamic<T> dL_dsdf_seed{m_density_network->padded_output_width(), batch_size, stream, density_network_output.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf_seed.data(), 0, dL_dsdf_seed.n_bytes(), stream));
		
		linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
			batch_size, T(1.0f),
			dL_dsdf_seed.layout() == AoS ? dL_dsdf_seed.stride() : 1,
			dL_dsdf_seed.data()
		);
		
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};  // Use AoS for consistency
		
		if (m_method == "hash_surface") {
			// For hash_surface, we need to handle the gradient flow differently
			// Create gradient buffer for the extracted density features (16D)
			GPUMatrixDynamic<T> dL_dextracted_density{m_density_network->input_width(), batch_size, stream, density_network_input.layout()};
			
			
			// Backward through density network (16D input/output)
			m_density_network->backward(
				stream, 
				*temp_density_ctx,
				temp_density_input,  // Use the temporary extracted density input from forward pass
				density_network_output,
				dL_dsdf_seed,
				&dL_dextracted_density,
				use_inference_params, 
				GradientMode::Ignore
			);
			
			// Create full 32D gradient buffer for position encoding backward
			GPUMatrixDynamic<T> dL_dhash_features{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dhash_features.data(), 0, dL_dhash_features.n_bytes(), stream));
			
			// Distribute gradients from extracted density features back to full hash features
			// Only the 4th component of each level gets gradients (the density features)
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4;
			
			// Use kernel to distribute gradients back to hash features (CUDA graph compatible)
			// This is the reverse of extract_hash_density_features_kernel
			linear_kernel(distribute_hash_density_gradients_kernel<T>, 0, stream,
				batch_size,
				n_levels, // Number of levels
				dL_dextracted_density.data(), // Source: gradients from extracted density features
				dL_dextracted_density.layout() == AoS ? dL_dextracted_density.stride() : 1,
				dL_dhash_features.data(), // Target: gradients to hash features
				dL_dhash_features.layout() == AoS ? dL_dhash_features.stride() : 1
			);
			
			// Backward through position encoding with full 32D gradients
			m_pos_encoding->backward(
				stream,
				*temp_pos_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				temp_hash_features,  // Use the temporary hash features from forward pass
				dL_dhash_features,
				&dSDF_dpos,
				use_inference_params,
				GradientMode::Ignore
			);
		} else {
			// Standard approach for other methods
			GPUMatrixDynamic<T> dL_ddensity_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			
			m_density_network->backward(
				stream, 
				*temp_density_ctx,
				density_network_input,
				density_network_output,
				dL_dsdf_seed,
				&dL_ddensity_input,
				use_inference_params, 
				GradientMode::Ignore
			);
			
			m_pos_encoding->backward(
				stream,
				*temp_pos_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input,
				dL_ddensity_input,
				&dSDF_dpos,
				use_inference_params,
				GradientMode::Ignore
			);
		}
		
		// Process gradients according to user settings (normalized or raw with optional clamping)
		GPUMatrixDynamic<float> normals{3, batch_size, stream, AoS};
		linear_kernel(process_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data(),
			m_normalize_normals,      // Whether to normalize to unit vectors
			m_clamp_gradients,        // Whether to clamp gradient magnitude
			m_max_gradient_magnitude  // Maximum allowed gradient magnitude
		);
		
		return normals;
	}

	/**
	 * @brief Backpropagates gradients from divergence values to network parameters.
	 * 
	 * For each of 15 vector fields and 3 spatial components:
	 * 1. Creates gradient seed from divergence gradients
	 * 2. Backpropagates through position encoding (second-order)
	 * 3. Backpropagates through density network (second-order)
	 * 4. Accumulates to main gradient buffer
	 * 
	 * This computes second-order gradients: d(∇·Φ)/d(params)
	 * 
	 * @param stream CUDA stream for operations
	 * @param batch_size Number of samples
	 * @param input Network input from forward pass
	 * @param forward Forward context containing cached values
	 * @param dL_ddivergences Gradients w.r.t. 15D divergences
	 * @param dL_ddensity_network_output Gradient buffer to accumulate into
	 * @param use_inference_params Whether to use inference parameters
	 * @param param_gradients_mode Gradient mode (typically Ignore for second-order)
	 * 
	 * @note Computationally expensive: 45 second-order gradient computations
	 * @warning Memory intensive due to temporary context creation
	 */
	void accumulate_volume_divergence_gradients(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		const ForwardContext& forward,
		const GPUMatrixDynamic<float>& dL_ddivergences,
		GPUMatrixDynamic<T>& dL_ddensity_network_output,
		bool use_inference_params,
		GradientMode param_gradients_mode
	) {
		// For each of the 15 vector fields, backpropagate gradients through divergence computation
		// We need to compute gradients w.r.t. each Phi component from divergence gradients
		
		for (uint32_t k = 0; k < 15; ++k) {
			// Get gradient w.r.t. divergence_k: dL/d(∇·Φ_k)
			float* dL_ddiv_k = const_cast<float*>(dL_ddivergences.data()) + k * (dL_ddivergences.layout() == AoS ? 1 : batch_size);
			
			// For each component (x, y, z) of vector field k, backpropagate through gradient computation
			for (uint32_t comp = 0; comp < 3; ++comp) {
				// Create gradient w.r.t. spatial gradient: dL/d(∂Φ_k_comp/∂spatial_comp)
				GPUMatrixDynamic<float> dL_dspatial_grad{m_pos_encoding->input_width(), batch_size, stream, AoS};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dspatial_grad.data(), 0, dL_dspatial_grad.n_bytes(), stream));
				
				// Copy divergence gradient to spatial gradient component: dL/d(∂Φ_k_comp/∂spatial_comp) = dL/d(∇·Φ_k)
				linear_kernel(copy_float_to_T_kernel<float>, 0, stream,
					batch_size,
					dL_ddiv_k,
					dL_dspatial_grad.data() + comp * (dL_dspatial_grad.layout() == AoS ? 1 : batch_size)
				);
				
				// Create temporary contexts for second-order backward pass
				auto temp_pos_ctx = m_pos_encoding->forward(
					stream, input.slice_rows(0, m_pos_encoding->input_width()), 
					const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_input), use_inference_params, true
				);
				
				auto temp_density_ctx = m_density_network->forward(
					stream, forward.density_network_input, 
					const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_output), use_inference_params, true
				);
				
				// Backward through position encoding (second-order approximation)
				GPUMatrixDynamic<T> dL_ddensity_input_from_div{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
				m_pos_encoding->backward(
					stream, *temp_pos_ctx, input.slice_rows(0, m_pos_encoding->input_width()), 
					forward.density_network_input, dL_ddensity_input_from_div, &dL_dspatial_grad, 
					use_inference_params, GradientMode::Ignore
				);
				
				// Backward through density network (second-order approximation)
				GPUMatrixDynamic<T> dL_ddensity_output_from_div{m_density_network->padded_output_width(), batch_size, stream, forward.density_network_output.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_output_from_div.data(), 0, dL_ddensity_output_from_div.n_bytes(), stream));
				m_density_network->backward(
					stream, *temp_density_ctx, forward.density_network_input, forward.density_network_output, 
					dL_ddensity_output_from_div, &dL_ddensity_input_from_div, use_inference_params, GradientMode::Ignore
				);
				
				// Accumulate gradients into main gradient buffer for the specific Phi component
				uint32_t phi_channel = 1 + k * 3 + comp;  // Phi_k component (x=0, y=1, z=2)
				linear_kernel(add_to_buffer_kernel<T>, 0, stream,
					batch_size,
					dL_ddensity_output_from_div.data() + phi_channel * (dL_ddensity_output_from_div.layout() == AoS ? 1 : batch_size),
					dL_ddensity_network_output.data() + phi_channel * (dL_ddensity_network_output.layout() == AoS ? 1 : batch_size)
				);
			}
		}
	}

	/**
	 * @brief Backpropagates gradients from normals to network parameters.
	 * 
	 * Applies chain rule through normalization: n = -∇SDF / ||∇SDF||
	 * 1. Computes dL/d(∇SDF) using Jacobian of normalization
	 * 2. Optionally adds Eikonal regularization gradients
	 * 3. Backpropagates through position encoding (second-order)
	 * 4. Backpropagates through density network (second-order)
	 * 5. Accumulates to main gradient buffer with reduced weight (0.1)
	 * 
	 * Handles hash_surface method specially by extracting/distributing density features.
	 * 
	 * @param stream CUDA stream for operations
	 * @param batch_size Number of samples
	 * @param input Network input from forward pass
	 * @param forward Forward context containing cached values (including raw_gradients)
	 * @param dL_dnormals Gradients w.r.t. normalized normals
	 * @param dL_ddensity_network_output Gradient buffer to accumulate into
	 * @param use_inference_params Whether to use inference parameters
	 * @param param_gradients_mode Gradient mode (typically Ignore for second-order)
	 * 
	 * @note Uses reduced weight (0.1) to prevent gradient explosion from second-order terms
	 * @note Adds Eikonal loss if enabled: enforces ||∇SDF|| ≈ 1
	 * @warning Hash_surface requires special handling for feature extraction
	 */
	void accumulate_analytical_normal_gradients(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		const ForwardContext& forward,
		const GPUMatrixDynamic<float>& dL_dnormals,
		GPUMatrixDynamic<T>& dL_ddensity_network_output,
		bool use_inference_params,
		GradientMode param_gradients_mode
	) {
		// Chain rule through normalization: dL/d(∇SDF) = dL/dnormals · d(normals)/d(∇SDF)
		// For normalization: normals = -∇SDF / ||∇SDF||
		// We need to compute: dL/d(∇SDF) via the chain rule through normalization
		
		// Step 1: Apply chain rule through normalization
		// Compute dL/d(raw_gradients) = dL/dnormals · d(normals)/d(raw_gradients)
		GPUMatrixDynamic<float> dL_draw_gradients{3, batch_size, stream, forward.raw_gradients.layout()};
		linear_kernel(chain_rule_through_normalization_kernel<float>, 0, stream,
			batch_size,
			dL_dnormals.data(),
			forward.raw_gradients.data(),
			dL_draw_gradients.data()
		);
		
		// Step 1.5: Add Eikonal regularization to raw gradients (if enabled)
		if (m_use_eikonal_loss && m_eikonal_weight > 0.0f) {
			linear_kernel(add_eikonal_gradients_kernel<float>, 0, stream,
				batch_size,
				forward.raw_gradients.data(),
				m_eikonal_weight,
				dL_draw_gradients.data()  // Accumulate into existing gradient buffer
			);
		}
		
		// Step 2: Create position gradient buffer for backpropagation
		GPUMatrixDynamic<float> dL_dpos_from_normals{m_pos_encoding->input_width(), batch_size, stream, input.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dpos_from_normals.data(), 0, dL_dpos_from_normals.n_bytes(), stream));
		
		// Copy gradients w.r.t. raw gradients to position gradient buffer (first 3 components)
		linear_kernel(copy_normal_gradients_to_pos_kernel<float>, 0, stream,
			batch_size,
			dL_draw_gradients.data(),
			dL_dpos_from_normals.data()
		);
		
		// Step 2: Create temporary contexts for second-order backward pass - handle hash_surface properly
		// (This is the limitation - ideally we'd use backward_backward_input)
		std::unique_ptr<Context> temp_pos_ctx;
		std::unique_ptr<Context> temp_density_ctx;
		GPUMatrixDynamic<T> temp_hash_features_for_normals;
		GPUMatrixDynamic<T> temp_extracted_density_for_normals;
		GPUMatrixDynamic<T> dL_ddensity_input_from_normals;
		
		if (m_method == "hash_surface") {
			// For hash_surface, create separate buffers and extract density features
			temp_hash_features_for_normals = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			
			temp_pos_ctx = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				&temp_hash_features_for_normals,
				use_inference_params,
				true  // prepare_input_gradients for second-order
			);
			
			// Extract density features again
			temp_extracted_density_for_normals = GPUMatrixDynamic<T>{m_density_network->input_width(), batch_size, stream, forward.density_network_input.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(temp_extracted_density_for_normals.data(), 0, temp_extracted_density_for_normals.n_bytes(), stream));
			
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4;
			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels,
				4, // 4 features per level
				temp_hash_features_for_normals.data(),
				temp_hash_features_for_normals.layout() == AoS ? temp_hash_features_for_normals.stride() : 1,
				temp_extracted_density_for_normals.data(),
				temp_extracted_density_for_normals.layout() == AoS ? temp_extracted_density_for_normals.stride() : 1
			);
			
			temp_density_ctx = m_density_network->forward(
				stream,
				temp_extracted_density_for_normals,
				const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_output),
				use_inference_params,
				true  // prepare_input_gradients for second-order
			);
			
			// Create 32D gradient buffer for position encoding
			dL_ddensity_input_from_normals = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		} else {
			// Standard approach for other methods
			temp_pos_ctx = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_input),
				use_inference_params,
				true  // prepare_input_gradients for second-order
			);
			
			temp_density_ctx = m_density_network->forward(
				stream,
				forward.density_network_input,
				const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_output),
				use_inference_params,
				true  // prepare_input_gradients for second-order
			);
			
			dL_ddensity_input_from_normals = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}
		
		// Step 3: Backward through position encoding (second-order approximation)
		if (m_method == "hash_surface") {
			m_pos_encoding->backward(
				stream,
				*temp_pos_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				temp_hash_features_for_normals,
				dL_ddensity_input_from_normals,
				&dL_dpos_from_normals,
				use_inference_params,
				GradientMode::Ignore  // Don't affect parameters, just compute gradients
			);
		} else {
			m_pos_encoding->backward(
				stream,
				*temp_pos_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_input_from_normals,
				&dL_dpos_from_normals,
				use_inference_params,
				GradientMode::Ignore  // Don't affect parameters, just compute gradients
			);
		}
		
		// Step 4: Backward through density network (second-order approximation)
		GPUMatrixDynamic<T> dL_ddensity_output_from_normals{m_density_network->padded_output_width(), batch_size, stream, forward.density_network_output.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_output_from_normals.data(), 0, dL_ddensity_output_from_normals.n_bytes(), stream));
		
		if (m_method == "hash_surface") {
			// Create 16D gradient buffer for extracted density features
			GPUMatrixDynamic<T> dL_dextracted_density_from_normals{m_density_network->input_width(), batch_size, stream, forward.density_network_input.layout()};
			
			m_density_network->backward(
				stream,
				*temp_density_ctx,
				temp_extracted_density_for_normals,
				forward.density_network_output,
				dL_ddensity_output_from_normals,
				&dL_dextracted_density_from_normals,
				use_inference_params,
				GradientMode::Ignore  // Don't affect parameters, just compute gradients
			);
			
			// Distribute gradients from extracted density features back to hash features
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_input_from_normals.data(), 0, dL_ddensity_input_from_normals.n_bytes(), stream));
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4;
			
			// Use kernel to distribute gradients (CUDA graph compatible)
			linear_kernel(distribute_hash_density_gradients_kernel<T>, 0, stream,
				batch_size,
				n_levels, // Number of levels
				dL_dextracted_density_from_normals.data(), // Source: gradients from extracted density features
				dL_dextracted_density_from_normals.layout() == AoS ? dL_dextracted_density_from_normals.stride() : 1,
				dL_ddensity_input_from_normals.data(), // Target: gradients to hash features
				dL_ddensity_input_from_normals.layout() == AoS ? dL_ddensity_input_from_normals.stride() : 1
			);
		} else {
			m_density_network->backward(
				stream,
				*temp_density_ctx,
				forward.density_network_input,
				forward.density_network_output,
				dL_ddensity_output_from_normals,
				&dL_ddensity_input_from_normals,
				use_inference_params,
				GradientMode::Ignore  // Don't affect parameters, just compute gradients
			);
		}
		
		// Step 5: Accumulate second-order gradients into main gradient buffer
		// Focus on SDF channel (channel 0) since that's what affects normals
		linear_kernel(accumulate_second_order_gradients_kernel<T>, 0, stream,
			batch_size,
			dL_ddensity_output_from_normals.layout() == AoS ? dL_ddensity_output_from_normals.stride() : 1,
			dL_ddensity_output_from_normals.data(),
			dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1,
			dL_ddensity_network_output.data()
		);
	}

	// Rest of the implementation (density, set_params, etc.) - same as original
	void density(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		
		if (m_method == "hash_surface") {
			// For hash_surface: position encoding → extract density features → density network
			GPUMatrixDynamic<T> hash_features{m_pos_encoding->padded_output_width(), batch_size, stream, input.layout()};
			
			// Step 1: Position encoding (3D → 32D hash features)
			m_pos_encoding->inference_mixed_precision(stream, input.slice_rows(0, m_pos_encoding->input_width()), hash_features, use_inference_params);
			
			// Step 2: Extract and pad density features (32D → 8D → 16D padded)
			uint32_t n_levels = m_pos_encoding->padded_output_width() / 4; // 32 / 4 = 8 levels
			uint32_t padded_density_input_width = m_density_network->input_width(); // Should be 16
			
			GPUMatrixDynamic<T> padded_density_features{padded_density_input_width, batch_size, stream, input.layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(padded_density_features.data(), 0, padded_density_features.n_bytes(), stream));
			
			// Extract density features (every 4th feature from hash interpolation)
			linear_kernel(extract_hash_density_features_kernel<T>, 0, stream,
				batch_size,
				n_levels,
				4, // 4 features per level
				hash_features.data(),
				hash_features.layout() == AoS ? hash_features.stride() : 1,
				padded_density_features.data(),
				padded_density_features.layout() == AoS ? padded_density_features.stride() : 1
			);
			
			// Step 3: Pass padded density features to density network
			m_density_network->inference_mixed_precision(stream, padded_density_features, output, use_inference_params);
		} else {
			// Original logic for other methods
			m_density_model->set_jit_fusion(false);
			m_density_model->inference_mixed_precision(stream, input.slice_rows(0, m_pos_encoding->input_width()), output, use_inference_params);
		}
	}

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
		
		// if (m_density_grid) {
		// 	// Initialize to near-empty: scale=0.01 gives [-0.01, 0.01], then subtract 3 to get [-3.01, -2.99]
		// 	// After exp(), this gives density ~ [0.05, 0.05] which is near-empty
		// 	size_t n = m_density_grid->n_params();
		// 	std::vector<float> temp(n);
		// 	for (size_t i = 0; i < n; ++i) {
		// 		temp[i] = (rnd.next_float() * 2.0f - 1.0f) * 0.01f - 3.0f;  // ~[-3.01, -2.99], exp() ~ 0.05
		// 	}
		// 	CUDA_CHECK_THROW(cudaMemcpy(params_full_precision, temp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
		// 	params_full_precision += n;
		// }
	}

	size_t n_params() const override {
		size_t params = m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
		if (m_variance_network) {
			params += m_variance_network->n_params();
		}
		if (m_density_grid) {
			params += m_density_grid->n_params();
		}
		return params;
	}

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

	uint32_t required_input_alignment() const override {
		return 1;
	}

	// Required pure virtual functions from Network<float, T>
	uint32_t width(uint32_t layer) const override {
		// Return width of specified layer in the network stack
		auto density_layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		
		if (layer < density_layers.size()) {
			return density_layers[layer].second;  // Return output width of density layer
		} else {
			uint32_t rgb_layer = layer - density_layers.size();
			if (rgb_layer < rgb_layers.size()) {
				return rgb_layers[rgb_layer].second;  // Return output width of RGB layer
			}
		}
		return 0;  // Invalid layer
	}

	uint32_t num_forward_activations() const override {
		// Return total number of layers in the network stack
		return m_density_network->layer_sizes().size() + m_rgb_network->layer_sizes().size();
	}

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		// This function would return intermediate activations for visualization
		// For now, return nullptr as it's not critical for surface reconstruction
		return std::make_pair(nullptr, RM);
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	const std::shared_ptr<Encoding<T>>& pos_encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	const std::shared_ptr<Network<T>>& density_network() const {
		return m_density_network;
	}

	const std::shared_ptr<Network<T>>& rgb_network() const {
		return m_rgb_network;
	}

	json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
		};
	}

	// NEW: Expose analytical normals for visualization
	GPUMatrixDynamic<float> get_analytical_normals_for_visualization(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		bool use_inference_params = true
	) {
		
		if (m_method != "surface" && m_method != "surface_normal" && m_method != "surface_reflect" && m_method != "hash_surface") {
			throw std::runtime_error("Analytical normals only available for surface methods");
		}
		
		uint32_t batch_size = input.n();
		
		// Prepare network inputs and outputs
		GPUMatrixDynamic<T> density_network_input;
		if (m_method == "hash_surface") {
			density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		} else {
			density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}
		GPUMatrixDynamic<T> density_network_output{m_density_network->padded_output_width(), batch_size, stream, AoS};
		
		// Forward pass through position encoding
		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);
		
		// Forward pass through density network
		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);
		
		// Compute analytical normals using inference pattern
		return compute_analytical_normals_inference_unified(
			stream, batch_size, input, density_network_input, density_network_output, use_inference_params
		);
	}

	// Compute analytical normals for visualization purposes  
	// This method can be used instead of the default input_gradient for superior normal quality
	void compute_analytical_normals_for_rendering(
		cudaStream_t stream,
		const GPUMatrix<float>& input,
		GPUMatrix<float>& normals_output
	) {
		// Only available for surface methods
		if (m_method != "surface" && m_method != "surface_normal" && m_method != "surface_reflect" && m_method != "hash_surface") {
			throw std::runtime_error("Analytical normals only available for surface methods");
		}
		
		uint32_t batch_size = input.n();
		
		// Convert input to match our analytical normal computation requirements
		GPUMatrixDynamic<float> input_dynamic{input.data(), input.m(), batch_size, stream, input.layout()};
		
		// Compute analytical normals using our specialized method
		GPUMatrixDynamic<float> analytical_normals = get_analytical_normals_for_visualization(
			stream, input_dynamic, true /* use_inference_params */
		);
		
		// Copy analytical normals to output buffer
		if (normals_output.m() >= 3 && analytical_normals.m() >= 3) {
			// Copy only the first 3 components (x, y, z normals)
			uint32_t copy_elements = std::min({3u, normals_output.m(), analytical_normals.m()});
			
			// Handle layout differences
			if (normals_output.layout() == analytical_normals.layout()) {
				// Same layout - direct copy
				CUDA_CHECK_THROW(cudaMemcpy2DAsync(
					normals_output.data(),
					normals_output.stride() * sizeof(float),
					analytical_normals.data(),
					analytical_normals.stride() * sizeof(float),
					copy_elements * sizeof(float),
					batch_size,
					cudaMemcpyDeviceToDevice,
					stream
				));
			} else {
				// Different layouts - use kernel
				linear_kernel(copy_float_to_T_kernel<float>, 0, stream,
					copy_elements * batch_size,
					analytical_normals.data(),
					normals_output.data()
				);
			}
			
			// Zero out remaining components if normals_output has more than 3 components
			if (normals_output.m() > 3) {
				CUDA_CHECK_THROW(cudaMemset2DAsync(
					normals_output.data() + 3 * (normals_output.layout() == AoS ? 1 : batch_size),
					normals_output.stride() * sizeof(float),
					0,
					(normals_output.m() - 3) * sizeof(float),
					batch_size,
					stream
				));
			}
		}
		
		
	}

	// Variance monitoring functions for debugging and optimization
	float get_current_variance() const {
		if (!m_variance_network || !m_variance_network->params()) {
			return 0.12f;  // Default value
		}
		return float(m_variance_network->params()[0]);
	}

	float get_current_s_value() const {
		float variance = get_current_variance();
		return expf(variance * 10.0f);
	}

private:
	std::shared_ptr<Network<T>> m_density_network;
	std::shared_ptr<Network<T>> m_rgb_network;
	std::shared_ptr<Encoding<T>> m_pos_encoding;
	std::shared_ptr<Encoding<T>> m_dir_encoding;
	std::shared_ptr<TrainableBuffer<1, 1, T>> m_variance_network;  // For SDF mode
	std::shared_ptr<Encoding<T>> m_density_grid;  // Explicit density grid (for surface_explicit mode only)

	std::shared_ptr<NetworkWithInputEncoding<T>> m_density_model;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims;
	uint32_t m_dir_offset;
	bool m_backprop_normals = false;
	bool m_use_analytical_normals = true;
	bool m_use_eikonal_loss = false;
	float m_eikonal_weight = 0.01f;
	bool m_normalize_normals = true;  // Default: use normalized normals (backward compatible)
	bool m_clamp_gradients = false;   // Default: no gradient clamping
	float m_max_gradient_magnitude = 1.0f;  // Default clamp magnitude
	bool m_hashpot_mode = false;      // Default: disabled hashpot vector features mode
	bool m_use_sdf = false;           // Whether to use SDF-to-density conversion

	std::string m_method;

	// Storage of forward pass data
	struct ForwardContext : public Context {
		GPUMatrixDynamic<T> density_network_input;
		GPUMatrixDynamic<T> density_network_output;
		GPUMatrixDynamic<T> rgb_network_input;
		GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;
		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
		std::unique_ptr<Context> normal_encoding_ctx;     // For surface_normal mode
		std::unique_ptr<Context> reflection_encoding_ctx; // For surface_reflect mode
		std::unique_ptr<Context> density_grid_ctx;        // For surface_explicit mode grid encoding

		// Analytical normals (∂SDF/∂xyz) - stored in forward context for backward pass
		GPUMatrixDynamic<float> dSDF_dPos;
		GPUMatrixDynamic<float> raw_gradients;     // Raw ∇SDF before normalization
		GPUMatrixDynamic<float> analytical_normals;
		
	// For volume mode: divergences of 15 3D vector fields
	GPUMatrixDynamic<float> volume_divergences;      // 15D divergence values per sample
	
	// For surface_normal mode: encoded normals
	GPUMatrixDynamic<T> encoded_normals_out;         // Output slice for encoded normals
	
	// For surface_explicit mode: grid density output
	GPUMatrixDynamic<T> grid_density;                // Grid density output for backward pass
	
	// For finite difference gradient computation (surface_explicit --grad finite)
	std::vector<std::unique_ptr<Context>> finite_diff_contexts;  // Variable number based on genus
	std::vector<GPUMatrixDynamic<T>> finite_diff_densities;      // Variable number based on genus
	std::vector<GPUMatrixDynamic<float>> finite_diff_positions;  // Variable number based on genus
	int finite_diff_genus = 1;  // Genus used for finite differences (1 or 2)
};

	
};

} 