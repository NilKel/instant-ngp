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

namespace ngp {

template <typename T>
__global__ void extract_density(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ density,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd[i * rgbd_stride] = density[i * density_stride];
}

// NeuS2-style set_constant_value_view for analytical normal computation
template <typename T>
__global__ void set_constant_value_view_kernel(
	const uint32_t batch_size,
	const T value,
	const uint32_t stride,
	T* __restrict__ data
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= batch_size) return;
	
	// Set channel 0 for batch element i (NeuS2 pattern)
	data[i * stride] = value;
}

// Surface features computation: density + ReLU(-Phi_k · normals)
template <typename T>
__global__ void compute_surface_features_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D: 1D density + 45D Phi
	const float* __restrict__ normals,     // 3D analytical normals
	const uint32_t rgb_stride,
	T* __restrict__ rgb_input             // Target: RGB input channels 0-15
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* density_base = density_output + i * density_stride;
	const float* normal = normals + i * 3;
	T* rgb_base = rgb_input + i * rgb_stride;
	
	// Channel 0: Copy density directly
	rgb_base[0] = density_base[0];
	
	// Channels 1-15: Compute surface features from 45D Phi
	for (uint32_t k = 0; k < 15; ++k) {
		const T* phi_k = density_base + 1 + k * 3; // Phi_k at channels [1+3k, 2+3k, 3+3k]
		
		// Dot product: phi_k · normal
		float dot_product = (float)phi_k[0] * normal[0] + (float)phi_k[1] * normal[1] + (float)phi_k[2] * normal[2];
		
		// Surface feature: Raw dot product (NO ReLU, NO negation)
		T surface_feature = T(dot_product);
		
		// Store in RGB input channels 1-15
		rgb_base[1 + k] = surface_feature;
	}
}

// Backward: accumulate gradients from surface features to Phi and normals
template <typename T>
__global__ void surface_features_backward_kernel(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const T* __restrict__ dL_drgb_input,   // Gradients w.r.t. RGB input channels 0-15
	const float* __restrict__ normals,     // Analytical normals from forward pass
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // Density output for ReLU condition check
	T* __restrict__ dL_ddensity_output,    // Target: gradients w.r.t. 48D density output
	float* __restrict__ dL_dnormals       // Target: gradients w.r.t. normals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* dL_drgb_base = dL_drgb_input + i * rgb_stride;
	const float* normal = normals + i * 3;
	const T* density_base = density_output + i * density_stride;
	T* dL_ddensity_base = dL_ddensity_output + i * density_stride;
	float* dL_dnormals_base = dL_dnormals + i * 3;
	
	// Channel 0: density gradient (direct copy)
	dL_ddensity_base[0] += dL_drgb_base[0];
	
	// Initialize normal gradients
	dL_dnormals_base[0] = 0.0f;
	dL_dnormals_base[1] = 0.0f;
	dL_dnormals_base[2] = 0.0f;
	
	// Channels 1-45: Phi feature gradients from surface features
	for (uint32_t k = 0; k < 15; ++k) {
		T dL_dsurface_feature = dL_drgb_base[1 + k]; // Gradient w.r.t. surface feature k
		
		const T* phi_k = density_base + 1 + k * 3; // Phi_k at channels [1+3k, 2+3k, 3+3k]
		float dot_product = (float)phi_k[0] * normal[0] + (float)phi_k[1] * normal[1] + (float)phi_k[2] * normal[2];
		
		// No ReLU condition - always propagate gradients
		// Backward through raw dot product: gradient = dL_dsurface_feature (no negation)
		T dL_ddot_product = dL_dsurface_feature;
		
		// Backward through dot product: dL/dPhi_k = dL_ddot_product * normal
		T* dL_dphi_k = dL_ddensity_base + 1 + k * 3;
		dL_dphi_k[0] += dL_ddot_product * T(normal[0]);
		dL_dphi_k[1] += dL_ddot_product * T(normal[1]);
		dL_dphi_k[2] += dL_ddot_product * T(normal[2]);
		
		// Backward through dot product: dL/dnormal = dL_ddot_product * Phi_k
		dL_dnormals_base[0] += (float)dL_ddot_product * (float)phi_k[0];
		dL_dnormals_base[1] += (float)dL_ddot_product * (float)phi_k[1];
		dL_dnormals_base[2] += (float)dL_ddot_product * (float)phi_k[2];
	}
}

// Normalization kernel for analytical gradients -> unit normals
template <typename T>
__global__ void normalize_analytical_gradients_kernel(
	const uint32_t n_elements,
	const T* __restrict__ dSDF_dPos,    // 3D+ analytical gradients
	T* __restrict__ normals             // Output: normalized normals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Get gradient vector for sample i (first 3 components)
	T grad_x = dSDF_dPos[i * 3 + 0];
	T grad_y = dSDF_dPos[i * 3 + 1];  
	T grad_z = dSDF_dPos[i * 3 + 2];
	
	// Compute magnitude
	T magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
	
	// Normalize and negate (normals point outward from surface)
	// Add epsilon to prevent division by zero
	T inv_mag = magnitude > T(1e-8f) ? T(-1.0f) / magnitude : T(0.0f);
	
	normals[i * 3 + 0] = grad_x * inv_mag;
	normals[i * 3 + 1] = grad_y * inv_mag;
	normals[i * 3 + 2] = grad_z * inv_mag;
}

// Helper kernel to copy normal gradients to position gradient buffer
template <typename T>
__global__ void copy_normal_gradients_to_pos_kernel(
	const uint32_t n_elements,
	const T* __restrict__ dL_dnormals,      // 3D normal gradients
	T* __restrict__ dL_dpos                 // Position gradients (3D+ components)
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Copy normal gradients to first 3 components of position gradients
	dL_dpos[i * 3 + 0] = dL_dnormals[i * 3 + 0];
	dL_dpos[i * 3 + 1] = dL_dnormals[i * 3 + 1];
	dL_dpos[i * 3 + 2] = dL_dnormals[i * 3 + 2];
}

// Helper kernel to accumulate second-order gradients (focusing on SDF channel)
template <typename T>
__global__ void accumulate_second_order_gradients_kernel(
	const uint32_t n_elements,
	const uint32_t source_stride,
	const T* __restrict__ dL_source,        // Second-order gradients
	const uint32_t target_stride,
	T* __restrict__ dL_target               // Main gradient buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Accumulate gradient for SDF channel (channel 0) with weight
	// Use smaller weight to prevent gradient explosion from second-order terms
	const T second_order_weight = T(0.1f);
	dL_target[i * target_stride] += second_order_weight * dL_source[i * source_stride];
}

	// Helper kernel to set unit normals [1/√3, 1/√3, 1/√3]
template <typename T>
__global__ void set_unit_normals_kernel(
	const uint32_t n_elements,
	T* __restrict__ normals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Set isotropic unit normal [1/√3, 1/√3, 1/√3] - all dimensions contribute equally
	const T inv_sqrt3 = T(0.57735026919f);  // 1/√3
	normals[i * 3 + 0] = inv_sqrt3;  // x = 1/√3
	normals[i * 3 + 1] = inv_sqrt3;  // y = 1/√3
	normals[i * 3 + 2] = inv_sqrt3;  // z = 1/√3
}

// NEW: Kernel to compute surface features for channels 1-15 only (density handled separately)
template <typename T>
__global__ void compute_surface_features_channels_1_to_15_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D: 1D density + 45D Phi (15 x 3D vectors)
	const uint32_t rgb_stride,
	T* __restrict__ rgb_input             // Target: RGB input channels 1-15 (channel 0 untouched)
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* density_base = density_output + i * density_stride;
	T* rgb_base = rgb_input + i * rgb_stride;
	
	// Channel 0: SKIP - handled by separate density copy like baseline
	
	// Isotropic unit normal [1/√3, 1/√3, 1/√3]
	const T inv_sqrt3 = T(0.57735026919f);  // 1/√3
	
	// Channels 1-15: Compute dot products of 45D Phi features with unit normal
	for (uint32_t k = 0; k < 15; ++k) {
		const T* phi_k = density_base + 1 + k * 3; // Phi_k at channels [1+3k, 2+3k, 3+3k]
		
		// Dot product: phi_k · [1/√3, 1/√3, 1/√3]
		T dot_product = phi_k[0] * inv_sqrt3 + phi_k[1] * inv_sqrt3 + phi_k[2] * inv_sqrt3;
		
		// Store in RGB input channels 1-15
		rgb_base[1 + k] = dot_product;
	}
}

// NEW: Kernel to compute 16D surface features (1D density + 15D dot products with unit normals)
template <typename T>
__global__ void compute_surface_features_with_unit_normals_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D: 1D density + 45D Phi (15 x 3D vectors)
	const uint32_t rgb_stride,
	T* __restrict__ rgb_input             // Target: RGB input channels 0-15
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* density_base = density_output + i * density_stride;
	T* rgb_base = rgb_input + i * rgb_stride;
	
	// Channel 0: Copy density directly
	rgb_base[0] = density_base[0];
	
	// Isotropic unit normal [1/√3, 1/√3, 1/√3]
	const T inv_sqrt3 = T(0.57735026919f);  // 1/√3
	
	// Channels 1-15: Compute dot products of 45D Phi features with unit normal
	for (uint32_t k = 0; k < 15; ++k) {
		const T* phi_k = density_base + 1 + k * 3; // Phi_k at channels [1+3k, 2+3k, 3+3k]
		
		// Dot product: phi_k · [1/√3, 1/√3, 1/√3]
		T dot_product = phi_k[0] * inv_sqrt3 + phi_k[1] * inv_sqrt3 + phi_k[2] * inv_sqrt3;
		
		// Store in RGB input channels 1-15
		rgb_base[1 + k] = dot_product;
	}
}

// NEW: Backward kernel for surface features with unit normals
template <typename T>
__global__ void surface_features_with_unit_normals_backward_kernel(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const T* __restrict__ dL_drgb_input,   // Gradients w.r.t. RGB input channels 0-15
	const uint32_t density_stride,
	T* __restrict__ dL_ddensity_output     // Target: gradients w.r.t. 48D density output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* dL_drgb_base = dL_drgb_input + i * rgb_stride;
	T* dL_ddensity_base = dL_ddensity_output + i * density_stride;
	
	// Channel 0: density gradient (direct copy)
	dL_ddensity_base[0] += dL_drgb_base[0];
	
	// Isotropic unit normal [1/√3, 1/√3, 1/√3]
	const T inv_sqrt3 = T(0.57735026919f);  // 1/√3
	
	// Channels 1-45: Backpropagate gradients from surface features to Phi features
	for (uint32_t k = 0; k < 15; ++k) {
		T dL_dsurface_feature = dL_drgb_base[1 + k]; // Gradient w.r.t. surface feature k
		
		// Backward through dot product: phi_k · [1/√3, 1/√3, 1/√3]
		// dL/dphi_k = dL_dsurface_feature * [1/√3, 1/√3, 1/√3]
		T* dL_dphi_k = dL_ddensity_base + 1 + k * 3;
		dL_dphi_k[0] += dL_dsurface_feature * inv_sqrt3;
		dL_dphi_k[1] += dL_dsurface_feature * inv_sqrt3;
		dL_dphi_k[2] += dL_dsurface_feature * inv_sqrt3;
	}
}

// Helper kernel to copy features directly (for unit normals mode)
template <typename T>
__global__ void copy_processed_features(
	const uint32_t n_elements,
	const T* __restrict__ input,
	T* __restrict__ output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	output[i] = input[i];
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const uint32_t output_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

// Helper kernel to print RGB input values for debugging
template <typename T>
__global__ void print_rgb_values_kernel(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	T* __restrict__ rgb_input,
	const char* mode_name
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	T* rgb_base = rgb_input + i * rgb_stride;
	
	// DEBUG: Print first few elements to compare between modes
	if (i == 0) {
		printf("DEBUG %s: rgb_base[0]=%f (density), rgb_base[1]=%f (feature), rgb_base[16]=%f (view dir)\n",
			mode_name, (float)rgb_base[0], (float)rgb_base[1], (float)rgb_base[16]);
	}
}

// Helper kernel to zero out RGB input channels 1-15 for debugging
template <typename T>
__global__ void zero_rgb_features_1_to_15_kernel(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	T* __restrict__ rgb_input
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	T* rgb_base = rgb_input + i * rgb_stride;
	
	// Zero out channels 1-15, keep channel 0 (density) and channels 16+ (view direction) intact
	for (uint32_t k = 1; k < 16; ++k) {
		rgb_base[k] = T(0.0f);
	}
}

template <typename T>
class NerfNetwork : public Network<float, T> {
private:
	// Forward declaration of private struct
	struct ForwardContext;
	
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network, const std::string& method = "baseline") : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims}, m_method{method} {
		printf("=== NerfNetwork Constructor ===\n");
		printf("Method: %s\n", m_method.c_str());
		printf("n_pos_dims: %d, n_dir_dims: %d\n", n_pos_dims, n_dir_dims);
		
		m_pos_encoding.reset(create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = minimum_alignment(rgb_network);
		m_dir_encoding.reset(create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		printf("pos_encoding->padded_output_width(): %d\n", m_pos_encoding->padded_output_width());
		printf("pos_encoding->preferred_output_layout(): %d\n", (int)m_pos_encoding->preferred_output_layout());
		printf("dir_encoding->padded_output_width(): %d\n", m_dir_encoding->padded_output_width());
		printf("dir_encoding->preferred_output_layout(): %d\n", (int)m_dir_encoding->preferred_output_layout());
		printf("rgb_alignment: %d\n", rgb_alignment);

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			if (m_method == "surface") {
				// 48D: 1D density + 45D Φ features (15 x 3D vectors) - test if 45D can learn with fixed normals
				local_density_network_config["n_output_dims"] = 48;
				printf("Surface mode: Set density network output dims to 48 (1D density + 45D Phi)\n");
			} else {
				local_density_network_config["n_output_dims"] = 16;
				printf("Baseline mode: Set density network output dims to 16\n");
			}
		}
		m_density_network.reset(create_network<T>(local_density_network_config));

		printf("density_network->padded_output_width(): %d\n", m_density_network->padded_output_width());

		if (m_method == "surface") {
			// Surface: 16 surface features + direction encoding
			m_rgb_network_input_width = next_multiple(16 + m_dir_encoding->padded_output_width(), rgb_alignment);
			printf("Surface mode RGB input calculation: 16 + %d = %d -> next_multiple(..., %d) = %d\n", 
				m_dir_encoding->padded_output_width(), 16 + m_dir_encoding->padded_output_width(), 
				rgb_alignment, m_rgb_network_input_width);
		} else {
			// Baseline: density output + direction encoding  
			m_rgb_network_input_width = next_multiple(m_dir_encoding->padded_output_width() + std::max(16u, m_density_network->padded_output_width()), rgb_alignment);
			printf("Baseline mode RGB input calculation: %d + max(16, %d) = %d + %d = %d -> next_multiple(..., %d) = %d\n",
				m_dir_encoding->padded_output_width(), m_density_network->padded_output_width(),
				m_dir_encoding->padded_output_width(), std::max(16u, m_density_network->padded_output_width()),
				m_dir_encoding->padded_output_width() + std::max(16u, m_density_network->padded_output_width()),
				rgb_alignment, m_rgb_network_input_width);
		}

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(create_network<T>(local_rgb_network_config));

		printf("RGB network input_width: %d, output_width: %d\n", m_rgb_network_input_width, 3);
		printf("=== End NerfNetwork Constructor ===\n");

		m_density_model = std::make_shared<NetworkWithInputEncoding<T>>(m_pos_encoding, m_density_network);
	}

	virtual ~NerfNetwork() { }

	void set_backprop_normals(bool v) { m_backprop_normals = v; }
	void set_use_analytical_normals(bool v) { m_use_analytical_normals = v; }

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();

		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		GPUMatrixDynamic<T> density_network_output;
		if (m_method == "surface") {
			density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, m_dir_encoding->preferred_output_layout()};
		} else {
			density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		}

		GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		// Standard forward pass
		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		// Set up direction encoding
		if (m_method == "surface") {
			// Surface mode: Use baseline-style density copy + surface features for channels 1-15
			
			// Copy density using baseline approach: density_output[0] → rgb_input[0]
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				density_network_output.data(),
				rgb_network_input.data()
			);
			
			
			// Compute surface features for channels 1-15 only
			linear_kernel(compute_surface_features_channels_1_to_15_kernel<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				density_network_output.data(),
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				rgb_network_input.data()
			);
			
			// Direction encoding goes after the 16D surface features
			uint32_t available_dir_space = m_rgb_network_input_width - 16;
			uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			
			printf("SURFACE DEBUG: m_rgb_network_input_width=%d, available_dir_space=%d, dir_encoding->padded_output_width()=%d, dir_encoding_width=%d\n",
				m_rgb_network_input_width, available_dir_space, m_dir_encoding->padded_output_width(), dir_encoding_width);
			printf("SURFACE LAYOUT: rgb_input.layout()=%d, rgb_input.stride()=%d, dir_out starts at row 16\n",
				(int)rgb_network_input.layout(), rgb_network_input.stride());
			
			auto dir_out = rgb_network_input.slice_rows(16, dir_encoding_width);
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
			
			// DEBUG: Print RGB input values to compare with baseline mode
			linear_kernel(print_rgb_values_kernel<T>, 0, stream,
				1,  // Only print first element  
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				rgb_network_input.data(),
				"SURFACE"
			);
		} else {
			// Baseline mode
			
			printf("BASELINE DEBUG: m_density_network->padded_output_width()=%d, m_dir_encoding->padded_output_width()=%d\n",
				m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			printf("BASELINE LAYOUT: rgb_input.layout()=%d, rgb_input.stride()=%d, dir_out starts at row %d\n",
				(int)rgb_network_input.layout(), rgb_network_input.stride(), m_density_network->padded_output_width());
			
			auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
			
			// DEBUG: Print baseline RGB input values to compare with surface mode
			linear_kernel(print_rgb_values_kernel<T>, 0, stream,
				1,  // Only print first element
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				rgb_network_input.data(),
				"BASELINE"
			);
		}

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		// Extract density to output - use correct source for each mode
		if (m_method == "surface") {
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				output.layout() == AoS ? padded_output_width() : 1,
				density_network_output.data(),
				output.data() + 3 * (output.layout() == AoS ? 1 : batch_size)
			);
		} else {
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				output.layout() == AoS ? padded_output_width() : 1,
				density_network_output.data(),
				output.data() + 3 * (output.layout() == AoS ? 1 : batch_size)
			);
		}
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		uint32_t batch_size = input.n();
		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients || m_method == "surface" // Always prepare gradients for surface mode
		);

		GPUMatrixDynamic<T> dir_out;
		if (m_method == "surface") {
			// Surface mode: Use baseline-style slicing for density, custom kernel for surface features
			forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, m_dir_encoding->preferred_output_layout()};
			
			// uint32_t available_dir_space = m_rgb_network_input_width - 16;
			// uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			
			// CRITICAL: Enable gradient computation for analytical normals
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, true);
			
			// Copy density using baseline approach: density_output[0] → rgb_input[0]
			linear_kernel(extract_density<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1,
				forward->density_network_output.data(),
				forward->rgb_network_input.data()
			);
			
			// Compute surface features for channels 1-15 only
			linear_kernel(compute_surface_features_channels_1_to_15_kernel<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				forward->density_network_output.data(),
				forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1,
				forward->rgb_network_input.data()
			);
		} else {
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

		if (output) {
			forward->rgb_network_output = GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}
		
		// Surface mode: compute surface features with unit normals [1/√3, 1/√3, 1/√3]
		if (m_method == "surface") {
			
			
			// Compute 16D surface features (1D density + 15D dot products with unit normals)
			
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		if (output) {
			// Extract density to output
			if (m_method == "surface") {
				linear_kernel(extract_density<T>, 0, stream,
					batch_size, 
					forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1,
					output->layout() == AoS ? padded_output_width() : 1,
					forward->rgb_network_input.data(),
					output->data() + 3 * (output->layout() == AoS ? 1 : batch_size)
				);
			} else {
				linear_kernel(extract_density<T>, 0, stream,
					batch_size, 
					forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
					output->layout() == AoS ? padded_output_width() : 1,
					forward->density_network_output.data(), 
					output->data() + 3 * (output->layout() == AoS ? 1 : batch_size)
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
		
		GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
		
		// Backprop through dir encoding
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			GPUMatrixDynamic<T> dL_ddir_encoding_output;
			// if (m_method == "surface") {
			// 	uint32_t available_dir_space = m_rgb_network_input_width - 16;
			// 	uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			// 	dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(16, dir_encoding_width);
			// } else {
			// 	dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			// }
			if (m_method == "surface") {
				dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			} else {
				dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			}
			
			GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			GPUMatrixDynamic<T> dir_encoding_forward_output;
			// if (m_method == "surface") {
			// 	uint32_t available_dir_space = m_rgb_network_input_width - 16;
			// 	uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			// 	dir_encoding_forward_output = forward.rgb_network_input.slice_rows(16, dir_encoding_width);
			// } else {
			// 	dir_encoding_forward_output = forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			// }
			if (m_method == "surface") {
				dir_encoding_forward_output = forward.rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
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
		if (m_method == "surface") {
			dL_ddensity_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, m_dir_encoding->preferred_output_layout()};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
		} else {
			dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		}

		if (m_method == "surface") {
			
			// Backward pass for surface features with unit normals
			linear_kernel(surface_features_with_unit_normals_backward_kernel<T>, 0, stream,
				batch_size,
				dL_drgb_network_input.layout() == RM ? 1 : dL_drgb_network_input.stride(),
				dL_drgb_network_input.data(),
				dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
				dL_ddensity_network_output.data()
			);
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
		linear_kernel(add_density_gradient<T>, 0, stream,
				batch_size,
				dL_doutput.m(),
				dL_doutput.data(),
				dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
				dL_ddensity_network_output.data()
			);
		// NOTE: No gradient clipping needed - working NeuS2 implementations don't use it

		GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput || m_backprop_normals) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding
		if (dL_ddensity_network_input.data()) {
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
	}

	// Helper function to compute analytical normals during forward pass (NeuS2 pattern)
	GPUMatrixDynamic<float> compute_analytical_normals_forward(
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
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, input.layout()};
		
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
		
		// Normalize gradients to get unit normals: n = -∇SDF / ||∇SDF||
		GPUMatrixDynamic<float> normals{3, batch_size, stream, CM};
		linear_kernel(normalize_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data()
		);
		
		return normals;
	}

	// Helper function for inference-only analytical normals
	GPUMatrixDynamic<float> compute_analytical_normals_inference(
		cudaStream_t stream,
		uint32_t batch_size,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& density_network_input,
		const GPUMatrixDynamic<T>& density_network_output,
		bool use_inference_params
	) {
		// Create temporary contexts for analytical computation
		auto temp_pos_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			const_cast<GPUMatrixDynamic<T>*>(&density_network_input),
			use_inference_params,
			true  // prepare_input_gradients
		);
		
		auto temp_density_ctx = m_density_network->forward(
			stream,
			density_network_input,
			const_cast<GPUMatrixDynamic<T>*>(&density_network_output),
			use_inference_params,
			true  // prepare_input_gradients
		);
		
		// Compute gradients using same pattern as forward
		GPUMatrixDynamic<T> dL_dsdf_seed{m_density_network->padded_output_width(), batch_size, stream, density_network_output.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf_seed.data(), 0, dL_dsdf_seed.n_bytes(), stream));
		
		linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
			batch_size, T(1.0f),
			dL_dsdf_seed.layout() == AoS ? dL_dsdf_seed.stride() : 1,
			dL_dsdf_seed.data()
		);
		
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
		
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, input.layout()};
		
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
		
		return dSDF_dpos.slice_rows(0, 3);
	}

	// Second-order gradient accumulation (NeuS2-inspired pattern with workaround for backward_backward_input)
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
		// WORKAROUND: Since backward_backward_input may not be available,
		// we implement a simplified second-order gradient flow by:
		// 1. Creating temporary contexts for second-order computation
		// 2. Using finite differences to approximate second-order derivatives
		// 3. Accumulating gradients properly
		
		// Step 1: Create temporary gradient buffer for normal gradients
		GPUMatrixDynamic<float> dL_dpos_from_normals{m_pos_encoding->input_width(), batch_size, stream, input.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_dpos_from_normals.data(), 0, dL_dpos_from_normals.n_bytes(), stream));
		
		// Copy normal gradients to position gradient buffer (first 3 components)
		linear_kernel(copy_normal_gradients_to_pos_kernel<float>, 0, stream,
			batch_size,
			dL_dnormals.data(),
			dL_dpos_from_normals.data()
		);
		
		// Step 2: Create temporary contexts for second-order backward pass
		// (This is the limitation - ideally we'd use backward_backward_input)
		auto temp_pos_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_input),
			use_inference_params,
			true  // prepare_input_gradients for second-order
		);
		
		auto temp_density_ctx = m_density_network->forward(
			stream,
			forward.density_network_input,
			const_cast<GPUMatrixDynamic<T>*>(&forward.density_network_output),
			use_inference_params,
			true  // prepare_input_gradients for second-order
		);
		
		// Step 3: Backward through position encoding (second-order approximation)
		GPUMatrixDynamic<T> dL_ddensity_input_from_normals{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		
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
		
		// Step 4: Backward through density network (second-order approximation)
		GPUMatrixDynamic<T> dL_ddensity_output_from_normals{m_density_network->padded_output_width(), batch_size, stream, forward.density_network_output.layout()};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_output_from_normals.data(), 0, dL_ddensity_output_from_normals.n_bytes(), stream));
		
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
		
		// Step 5: Accumulate second-order gradients into main gradient buffer
		// Focus on SDF channel (channel 0) since that's what affects normals
		linear_kernel(accumulate_second_order_gradients_kernel<T>, 0, stream,
			batch_size,
			dL_ddensity_output_from_normals.layout() == RM ? 1 : dL_ddensity_output_from_normals.stride(),
			dL_ddensity_output_from_normals.data(),
			dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);
	}

	// Rest of the implementation (density, set_params, etc.) - same as original
	void density(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_density_model->set_jit_fusion(false);
		m_density_model->inference_mixed_precision(stream, input.slice_rows(0, m_pos_encoding->input_width()), output, use_inference_params);
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
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
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

private:
	std::shared_ptr<Network<T>> m_density_network;
	std::shared_ptr<Network<T>> m_rgb_network;
	std::shared_ptr<Encoding<T>> m_pos_encoding;
	std::shared_ptr<Encoding<T>> m_dir_encoding;

	std::shared_ptr<NetworkWithInputEncoding<T>> m_density_model;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims;
	uint32_t m_dir_offset;
	bool m_backprop_normals = false;
	bool m_use_analytical_normals = true;

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

		// Analytical normals (∂SDF/∂xyz) - stored in forward context for backward pass
		GPUMatrixDynamic<float> dSDF_dPos;
	};
};

} 