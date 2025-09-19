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
	// Add larger epsilon to prevent division by zero and numerical instability
	T inv_mag = magnitude > T(1e-6f) ? T(-1.0f) / magnitude : T(0.0f);
	
	// Additional safety: clamp the result to prevent extreme values
	T norm_x = fmaxf(-1.0f, fminf(1.0f, grad_x * inv_mag));
	T norm_y = fmaxf(-1.0f, fminf(1.0f, grad_y * inv_mag));
	T norm_z = fmaxf(-1.0f, fminf(1.0f, grad_z * inv_mag));
	
	normals[i * 3 + 0] = norm_x;
	normals[i * 3 + 1] = norm_y;
	normals[i * 3 + 2] = norm_z;
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

// Helper kernel to accumulate normal gradients from encoding
template <typename T>
__global__ void accumulate_normal_gradients_kernel(
	const uint32_t n_elements,
	const T* __restrict__ dL_dnormals_from_encoding,  // Gradients from normal encoding
	T* __restrict__ dL_dnormals_total                 // Total normal gradients (accumulate into)
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Accumulate gradients from normal encoding
	dL_dnormals_total[i * 3 + 0] += dL_dnormals_from_encoding[i * 3 + 0];
	dL_dnormals_total[i * 3 + 1] += dL_dnormals_from_encoding[i * 3 + 1];
	dL_dnormals_total[i * 3 + 2] += dL_dnormals_from_encoding[i * 3 + 2];
}

// Simple kernel to add one buffer to another (element-wise)
template <typename T>
__global__ void add_to_buffer_kernel(
	const uint32_t n_elements,
	const T* __restrict__ source,
	T* __restrict__ destination
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	destination[i] += source[i];
}

// Scaled accumulation kernel for balancing gradient contributions
template <typename T>
__global__ void add_scaled_to_buffer_kernel(
	const uint32_t n_elements,
	const T scale_factor,
	const T* __restrict__ source,
	T* __restrict__ destination
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	destination[i] += scale_factor * source[i];
}

// Helper kernel to compute gradients w.r.t. raw gradients via chain rule through normalization
template <typename T>
__global__ void chain_rule_through_normalization_kernel(
	const uint32_t n_elements,
	const float* __restrict__ dL_dnormals,     // Gradients w.r.t. normals [3×N]
	const float* __restrict__ raw_gradients,   // Raw ∇SDF from forward pass [3×N]
	float* __restrict__ dL_draw_gradients      // Output: gradients w.r.t. raw gradients [3×N]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Get gradients for sample i
	const float* dL_dn = dL_dnormals + i * 3;      // [3]
	const float* grad = raw_gradients + i * 3;     // [3] (∇SDF)
	float* dL_dg = dL_draw_gradients + i * 3;      // [3] (output)
	
	// Compute gradient magnitude
	float grad_mag_sq = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
	float grad_mag = sqrtf(grad_mag_sq + 1e-12f);  // Add epsilon for stability
	float inv_grad_mag = 1.0f / grad_mag;
	float inv_grad_mag3 = inv_grad_mag * inv_grad_mag * inv_grad_mag;
	
	// Chain rule for normalization: normals = -∇SDF / ||∇SDF||
	// d(normals)/d(∇SDF) = -1/||∇SDF|| * I + (∇SDF ⊗ ∇SDF) / ||∇SDF||³
	// where I is identity matrix and ⊗ is outer product
	
	// Compute dot product: dL_dnormals · ∇SDF
	float dot_product = dL_dn[0] * grad[0] + dL_dn[1] * grad[1] + dL_dn[2] * grad[2];
	
	// Apply chain rule
	dL_dg[0] = -dL_dn[0] * inv_grad_mag + grad[0] * dot_product * inv_grad_mag3;
	dL_dg[1] = -dL_dn[1] * inv_grad_mag + grad[1] * dot_product * inv_grad_mag3;
	dL_dg[2] = -dL_dn[2] * inv_grad_mag + grad[2] * dot_product * inv_grad_mag3;
}

// Eikonal loss regularization kernel - adds gradients to enforce ||∇SDF|| ≈ 1
template <typename T>
__global__ void add_eikonal_gradients_kernel(
	const uint32_t n_elements,
	const float* __restrict__ raw_gradients,     // Raw ∇SDF from forward pass [3×N]
	const float eikonal_weight,
	float* __restrict__ dL_dpos_gradients        // Accumulate into position gradients [3×N]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Get gradient vector for sample i
	const float* grad = raw_gradients + i * 3;
	float* dL_dgrad = dL_dpos_gradients + i * 3;
	
	// Compute gradient magnitude: ||∇SDF||
	float grad_norm_sq = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
	float grad_norm = sqrtf(grad_norm_sq + 1e-6f);
	
	// Eikonal gradient: dL/d(∇SDF) = 2 * weight * (||∇SDF|| - 1) * ∇SDF / ||∇SDF||
	float eikonal_error = grad_norm - 1.0f;
	float eikonal_grad_coeff = 2.0f * eikonal_weight * eikonal_error / (grad_norm + 1e-8f);
	
	// Accumulate eikonal gradients
	dL_dgrad[0] += eikonal_grad_coeff * grad[0];
	dL_dgrad[1] += eikonal_grad_coeff * grad[1];
	dL_dgrad[2] += eikonal_grad_coeff * grad[2];
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
		
		// Scale up surface features to match baseline magnitude
		const T surface_scale = T(3.0f);  // Experimental scaling factor
		
		// Store in RGB input channels 1-15
		rgb_base[1 + k] = dot_product * surface_scale;
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
		
		// Backward through scaling factor
		const T surface_scale = T(3.0f);  // Must match forward pass scaling
		T dL_ddot_product = dL_dsurface_feature * surface_scale;
		
		// Backward through dot product: phi_k · [1/√3, 1/√3, 1/√3]
		// dL/dphi_k = dL_ddot_product * [1/√3, 1/√3, 1/√3]
		T* dL_dphi_k = dL_ddensity_base + 1 + k * 3;
		dL_dphi_k[0] += dL_ddot_product * inv_sqrt3;
		dL_dphi_k[1] += dL_ddot_product * inv_sqrt3;
		dL_dphi_k[2] += dL_ddot_product * inv_sqrt3;
	}
}

// NEW: Kernel to compute surface features directly into RGB slice WITH ANALYTICAL NORMALS
template <typename T>
__global__ void compute_surface_features_to_slice_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D: 1D density + 45D Phi
	const float* __restrict__ analytical_normals,  // 3D analytical normals per sample
	const uint32_t slice_stride,
	T* __restrict__ rgb_slice             // Target: 16D RGB slice [0:15]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Handle both AoS and SoA layouts properly
	// AoS: density_stride = 48, each sample has all 48 channels contiguous
	// SoA: density_stride = 1, each channel has all samples contiguous
	
	// Channel 0: Copy density directly
	const T density_val = density_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)];
	rgb_slice[i * slice_stride + 0 * (slice_stride == 1 ? n_elements : 1)] = density_val;
	
	// Get analytical normal for this sample
	const float* normal = analytical_normals + i * 3;
	
	// Scaling factor to match baseline magnitude
	const T surface_scale = T(3.0f);
	
	// Channels 1-15: Compute ReLU(-phi_k · analytical_normal)
	for (uint32_t k = 0; k < 15; ++k) {
		// Extract 3D Phi vector k from channels [1+3k, 2+3k, 3+3k]
		T phi_x = density_output[i * density_stride + (1 + k * 3 + 0) * (density_stride == 1 ? n_elements : 1)];
		T phi_y = density_output[i * density_stride + (1 + k * 3 + 1) * (density_stride == 1 ? n_elements : 1)];
		T phi_z = density_output[i * density_stride + (1 + k * 3 + 2) * (density_stride == 1 ? n_elements : 1)];
		
		// Dot product: phi_k · analytical_normal
		T dot_product = phi_x * T(normal[0]) + phi_y * T(normal[1]) + phi_z * T(normal[2]);
		
		// Surface feature: ReLU(-dot_product) = max(0, -dot_product)
		T surface_feature = fmaxf(T(0.0f), -dot_product);
		
		// Store in RGB slice channels 1-15
		rgb_slice[i * slice_stride + (1 + k) * (slice_stride == 1 ? n_elements : 1)] = surface_feature * surface_scale;
	}
}

// NEW: Backward kernel for slice-based surface features WITH ANALYTICAL NORMALS
template <typename T>
__global__ void surface_features_slice_backward_kernel(
	const uint32_t n_elements,
	const uint32_t slice_stride,
	const T* __restrict__ dL_drgb_slice,   // Gradients w.r.t. RGB slice [0:15]
	const float* __restrict__ analytical_normals,  // Analytical normals from forward pass
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D density output (needed for ReLU condition)
	T* __restrict__ dL_ddensity_output,    // Target: gradients w.r.t. 48D density output
	float* __restrict__ dL_dnormals        // Target: gradients w.r.t. normals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Handle both AoS and SoA layouts properly
	// AoS: stride = width, each sample has all channels contiguous
	// SoA: stride = 1, each channel has all samples contiguous
	
	// Channel 0: density gradient (direct copy)
	const T dL_ddensity = dL_drgb_slice[i * slice_stride + 0 * (slice_stride == 1 ? n_elements : 1)];
	dL_ddensity_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)] += dL_ddensity;
	
	// Get analytical normal for this sample
	const float* normal = analytical_normals + i * 3;
	float* dL_dnormals_base = dL_dnormals + i * 3;
	
	// NOTE: Don't initialize normal gradients here - they should be accumulated
	// The buffer is zero-initialized at allocation time
	// dL_dnormals_base[0] = 0.0f;  // REMOVED: This prevents accumulation
	// dL_dnormals_base[1] = 0.0f;  // REMOVED: This prevents accumulation  
	// dL_dnormals_base[2] = 0.0f;  // REMOVED: This prevents accumulation
	
	// Scaling factor to match forward pass
	const T surface_scale = T(3.0f);
	
	// Channels 1-45: Backpropagate gradients from surface features to Phi features AND normals
	for (uint32_t k = 0; k < 15; ++k) {
		const T dL_dsurface_feature = dL_drgb_slice[i * slice_stride + (1 + k) * (slice_stride == 1 ? n_elements : 1)];
		
		// Backward through scaling
		const T dL_dsurface_feature_unscaled = dL_dsurface_feature * surface_scale;
		
		// Recompute dot product from forward pass to check ReLU condition
		const T* density_base = density_output + i * density_stride;
		const T* phi_k = density_base + 1 + k * 3;
		T dot_product = phi_k[0] * T(normal[0]) + phi_k[1] * T(normal[1]) + phi_k[2] * T(normal[2]);
		
		// Backward through ReLU: gradient flows only if -dot_product > 0 (i.e., dot_product < 0)
		T dL_ddot_product = T(0.0f);
		if (dot_product < T(0.0f)) {
			// ReLU derivative: d/dx ReLU(-x) = -1 when -x > 0 (i.e., x < 0)
			dL_ddot_product = -dL_dsurface_feature_unscaled;
		}
		// If dot_product >= 0, then ReLU(-dot_product) = 0, so gradient is 0
		
		// Backward through dot product: phi_k · normal
		// dL/dphi_k = dL_ddot_product * normal
		const T dL_dphi_x = dL_ddot_product * T(normal[0]);
		const T dL_dphi_y = dL_ddot_product * T(normal[1]);
		const T dL_dphi_z = dL_ddot_product * T(normal[2]);
		
		// dL/dnormal = dL_ddot_product * phi_k
		dL_dnormals_base[0] += (float)dL_ddot_product * (float)phi_k[0];
		dL_dnormals_base[1] += (float)dL_ddot_product * (float)phi_k[1];
		dL_dnormals_base[2] += (float)dL_ddot_product * (float)phi_k[2];
		
		// Accumulate gradients to 3D Phi vector k at channels [1+3k, 2+3k, 3+3k]
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 0) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_x;
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 1) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_y;
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 2) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_z;
	}
}

// ORIGINAL: Kernel to compute surface features directly into RGB slice WITH UNIT NORMALS (kept as fallback)
template <typename T>
__global__ void compute_surface_features_to_slice_unit_normals_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D: 1D density + 45D Phi
	const uint32_t slice_stride,
	T* __restrict__ rgb_slice             // Target: 16D RGB slice [0:15]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Handle both AoS and SoA layouts properly
	// AoS: density_stride = 48, each sample has all 48 channels contiguous
	// SoA: density_stride = 1, each channel has all samples contiguous
	
	// Channel 0: Copy density directly
	const T density_val = density_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)];
	rgb_slice[i * slice_stride + 0 * (slice_stride == 1 ? n_elements : 1)] = density_val;
	
	// Isotropic unit normal [1/√3, 1/√3, 1/√3]
	const T inv_sqrt3 = T(0.57735026919f);  // 1/√3
	const T surface_scale = T(3.0f);  // Scaling factor
	
	// Channels 1-15: Compute ReLU(-phi_k · unit_normal)
	for (uint32_t k = 0; k < 15; ++k) {
		// Extract 3D Phi vector k from channels [1+3k, 2+3k, 3+3k]
		T phi_x = density_output[i * density_stride + (1 + k * 3 + 0) * (density_stride == 1 ? n_elements : 1)];
		T phi_y = density_output[i * density_stride + (1 + k * 3 + 1) * (density_stride == 1 ? n_elements : 1)];
		T phi_z = density_output[i * density_stride + (1 + k * 3 + 2) * (density_stride == 1 ? n_elements : 1)];
		
		// Dot product: phi_k · [1/√3, 1/√3, 1/√3]
		T dot_product = phi_x * inv_sqrt3 + phi_y * inv_sqrt3 + phi_z * inv_sqrt3;
		
		// Surface feature: ReLU(-dot_product) = max(0, -dot_product)
		T surface_feature = fmaxf(T(0.0f), -dot_product);
		
		// Store in RGB slice channels 1-15
		rgb_slice[i * slice_stride + (1 + k) * (slice_stride == 1 ? n_elements : 1)] = surface_feature * surface_scale;
	}
}

// ORIGINAL: Backward kernel for slice-based surface features WITH UNIT NORMALS (kept as fallback)
template <typename T>
__global__ void surface_features_slice_unit_normals_backward_kernel(
	const uint32_t n_elements,
	const uint32_t slice_stride,
	const T* __restrict__ dL_drgb_slice,   // Gradients w.r.t. RGB slice [0:15]
	const uint32_t density_stride,
	const T* __restrict__ density_output,  // 48D density output (needed for ReLU condition)
	T* __restrict__ dL_ddensity_output     // Target: gradients w.r.t. 48D density output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Handle both AoS and SoA layouts properly
	// AoS: stride = width, each sample has all channels contiguous
	// SoA: stride = 1, each channel has all samples contiguous
	
	// Channel 0: density gradient (direct copy)
	const T dL_ddensity = dL_drgb_slice[i * slice_stride + 0 * (slice_stride == 1 ? n_elements : 1)];
	dL_ddensity_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)] += dL_ddensity;
	
	// Isotropic unit normal and scaling factor
	const T inv_sqrt3 = T(0.57735026919f);  // 1/√3
	const T surface_scale = T(3.0f);  // Must match forward pass scaling
	
	// Channels 1-45: Backpropagate gradients from surface features to Phi features
	for (uint32_t k = 0; k < 15; ++k) {
		const T dL_dsurface_feature = dL_drgb_slice[i * slice_stride + (1 + k) * (slice_stride == 1 ? n_elements : 1)];
		
		// Backward through scaling
		const T dL_dsurface_feature_unscaled = dL_dsurface_feature * surface_scale;
		
		// Recompute dot product from forward pass to check ReLU condition
		const T* density_base = density_output + i * density_stride;
		const T* phi_k = density_base + 1 + k * 3;
		T dot_product = phi_k[0] * inv_sqrt3 + phi_k[1] * inv_sqrt3 + phi_k[2] * inv_sqrt3;
		
		// Backward through ReLU: gradient flows only if -dot_product > 0 (i.e., dot_product < 0)
		T dL_ddot_product = T(0.0f);
		if (dot_product < T(0.0f)) {
			// ReLU derivative: d/dx ReLU(-x) = -1 when -x > 0 (i.e., x < 0)
			dL_ddot_product = -dL_dsurface_feature_unscaled;
		}
		// If dot_product >= 0, then ReLU(-dot_product) = 0, so gradient is 0
		
		// dL/dphi_k = dL_ddot_product * [1/√3, 1/√3, 1/√3]
		const T dL_dphi_component = dL_ddot_product * inv_sqrt3;
		
		// Accumulate gradients to 3D Phi vector k at channels [1+3k, 2+3k, 3+3k]
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 0) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_component;
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 1) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_component;
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 2) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_component;
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

// Helper kernel to copy float values to T values with proper type conversion
template <typename T>
__global__ void copy_float_to_T_kernel(
	const uint32_t n_elements,
	const float* __restrict__ source,
	T* __restrict__ destination
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	destination[i] = T(source[i]);  // Proper type conversion
}



// Helper kernel to accumulate T gradients into float gradients with type conversion
template <typename T>
__global__ void accumulate_T_to_float_kernel(
	const uint32_t n_elements,
	const T* __restrict__ source,        // T* gradients
	float* __restrict__ destination      // float* gradients (accumulate into)
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	destination[i] += float(source[i]);  // Type conversion and accumulation
}

// Layout-aware kernel to copy 3D float normals to T normals slice (handles AoS/SoA)
template <typename T>
__global__ void copy_3D_normals_to_slice_kernel(
	const uint32_t n_elements,
	const float* __restrict__ normals,        // 3D normals [3 x batch_size] 
	const uint32_t slice_stride,              // Stride of the target slice
	T* __restrict__ output_slice              // Target slice [3 x batch_size]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Handle both AoS and SoA layouts properly (same pattern as surface features)
	const float* normal_base = normals + i * 3;  // Source: always assumes normals are [nx, ny, nz] per sample
	
	// Copy 3 components with proper layout handling and NaN protection
	for (uint32_t ch = 0; ch < 3; ++ch) {
		uint32_t target_idx = i * slice_stride + ch * (slice_stride == 1 ? n_elements : 1);
		// Safety check: replace NaN/Inf with zero
		float val = normal_base[ch];
		if (!isfinite(val)) {
			val = 0.0f;
		}
		output_slice[target_idx] = T(val);
	}
}

// Layout-aware kernel to accumulate T gradients from slice to float normal gradients
template <typename T>
__global__ void accumulate_3D_slice_to_normals_kernel(
	const uint32_t n_elements,
	const uint32_t slice_stride,              // Stride of the source slice  
	const T* __restrict__ dL_dslice,          // T* gradients from slice [3 x batch_size]
	float* __restrict__ dL_dnormals           // float* normal gradients [3 x batch_size]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Handle both AoS and SoA layouts properly (same pattern as surface features)
	float* normal_grad_base = dL_dnormals + i * 3;  // Target: always [nx, ny, nz] per sample
	
	// Accumulate 3 components with proper layout handling
	for (uint32_t ch = 0; ch < 3; ++ch) {
		uint32_t source_idx = i * slice_stride + ch * (slice_stride == 1 ? n_elements : 1);
		normal_grad_base[ch] += float(dL_dslice[source_idx]);
	}
}

// NEW: Layout-aware kernel to calculate reflection vectors from view directions and normals
template <typename T>
__global__ void calculate_reflection_vector_kernel(
	const uint32_t n_elements,
	const float* __restrict__ view_dirs,     // View direction input (may have >3 components)
	const float* __restrict__ normals,       // 3D analytical normal per sample
	float* __restrict__ reflection_vectors,  // Output: 3D reflection vector per sample
	const uint32_t view_width,               // Width of view direction input (e.g., 3 for basic, more for encodings)
	const uint32_t view_stride,              // Stride for view directions
	const uint32_t normal_stride,            // Stride for normals (always 3)
	const uint32_t reflect_stride            // Stride for reflection vectors (always 3)
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Layout-aware indexing for view directions (only use first 3 components)
	// For AoS: view_dirs has layout [sample0: x,y,z,extra..., sample1: x,y,z,extra..., ...]
	// For SoA: view_dirs has layout [x_all..., y_all..., z_all..., extra_all...]
	uint32_t v_x_idx, v_y_idx, v_z_idx;
	if (view_stride == view_width) {
		// AoS layout: each sample has all components contiguous
		v_x_idx = i * view_width + 0;
		v_y_idx = i * view_width + 1;
		v_z_idx = i * view_width + 2;
	} else {
		// SoA layout: each component has all samples contiguous
		v_x_idx = 0 * n_elements + i;
		v_y_idx = 1 * n_elements + i;
		v_z_idx = 2 * n_elements + i;
	}
	
	// Normal indexing (always 3D)
	const uint32_t n_x_idx = (normal_stride == 3) ? (i * 3 + 0) : (0 * n_elements + i);
	const uint32_t n_y_idx = (normal_stride == 3) ? (i * 3 + 1) : (1 * n_elements + i);
	const uint32_t n_z_idx = (normal_stride == 3) ? (i * 3 + 2) : (2 * n_elements + i);
	
	// Reflection vector indexing (always 3D)
	const uint32_t r_x_idx = (reflect_stride == 3) ? (i * 3 + 0) : (0 * n_elements + i);
	const uint32_t r_y_idx = (reflect_stride == 3) ? (i * 3 + 1) : (1 * n_elements + i);
	const uint32_t r_z_idx = (reflect_stride == 3) ? (i * 3 + 2) : (2 * n_elements + i);
	
	// View directions in NeRF point FROM surface TO camera
	// For reflection formula, we need vector pointing TO surface, so negate
	const float v_x = -view_dirs[v_x_idx];
	const float v_y = -view_dirs[v_y_idx];
	const float v_z = -view_dirs[v_z_idx];
	
	const float n_x = normals[n_x_idx];
	const float n_y = normals[n_y_idx];
	const float n_z = normals[n_z_idx];
	
	// Dot product: v · n
	const float dot_vn = v_x * n_x + v_y * n_y + v_z * n_z;
	
	// Reflection formula: R = v - 2 * (v · n) * n
	float r_x = v_x - 2.0f * dot_vn * n_x;
	float r_y = v_y - 2.0f * dot_vn * n_y;
	float r_z = v_z - 2.0f * dot_vn * n_z;
	
	// Normalize the reflection vector (like view directions)
	float r_mag = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);
	float inv_r_mag = (r_mag > 1e-8f) ? (1.0f / r_mag) : 0.0f;
	
	reflection_vectors[r_x_idx] = r_x * inv_r_mag;
	reflection_vectors[r_y_idx] = r_y * inv_r_mag;
	reflection_vectors[r_z_idx] = r_z * inv_r_mag;
}

// NEW: Backward kernel for reflection vector computation
template <typename T>
__global__ void reflection_vector_backward_kernel(
	const uint32_t n_elements,
	const float* __restrict__ view_dirs,           // 3D view direction per sample
	const float* __restrict__ normals,             // 3D analytical normal per sample
	const float* __restrict__ dL_dreflection,      // Gradients w.r.t. reflection vectors [3×N]
	float* __restrict__ dL_dview_dirs,             // Output: gradients w.r.t. view directions [3×N]
	float* __restrict__ dL_dnormals               // Output: gradients w.r.t. normals [3×N]
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Negated view directions (as used in forward pass)
	const float v_x = -view_dirs[i * 3 + 0];
	const float v_y = -view_dirs[i * 3 + 1];
	const float v_z = -view_dirs[i * 3 + 2];
	
	const float n_x = normals[i * 3 + 0];
	const float n_y = normals[i * 3 + 1];
	const float n_z = normals[i * 3 + 2];
	
	// Gradients w.r.t. reflection vector
	const float dL_dR_x = dL_dreflection[i * 3 + 0];
	const float dL_dR_y = dL_dreflection[i * 3 + 1];
	const float dL_dR_z = dL_dreflection[i * 3 + 2];
	
	// Dot product: v · n
	const float dot_vn = v_x * n_x + v_y * n_y + v_z * n_z;
	
	// Backward through reflection formula: R = v - 2 * (v · n) * n
	// dR/dv = I - 2 * (n ⊗ n + (v · n) * 0)  = I - 2 * (n ⊗ n)
	// dR/dn = -2 * (v ⊗ I + (v · n) * I) = -2 * v - 2 * (v · n) * I
	
	// Gradients w.r.t. negated view directions (v = -view_dirs)
	float dL_dv_x = dL_dR_x - 2.0f * (dL_dR_x * n_x * n_x + dL_dR_y * n_y * n_x + dL_dR_z * n_z * n_x);
	float dL_dv_y = dL_dR_y - 2.0f * (dL_dR_x * n_x * n_y + dL_dR_y * n_y * n_y + dL_dR_z * n_z * n_y);
	float dL_dv_z = dL_dR_z - 2.0f * (dL_dR_x * n_x * n_z + dL_dR_y * n_y * n_z + dL_dR_z * n_z * n_z);
	
	// Convert to gradients w.r.t. original view directions (since v = -view_dirs)
	dL_dview_dirs[i * 3 + 0] = -dL_dv_x;
	dL_dview_dirs[i * 3 + 1] = -dL_dv_y;
	dL_dview_dirs[i * 3 + 2] = -dL_dv_z;
	
	// Gradients w.r.t. normals
	float dL_dn_from_dot = -2.0f * (dL_dR_x * v_x + dL_dR_y * v_y + dL_dR_z * v_z);
	dL_dnormals[i * 3 + 0] = -2.0f * dot_vn * dL_dR_x + dL_dn_from_dot * n_x;
	dL_dnormals[i * 3 + 1] = -2.0f * dot_vn * dL_dR_y + dL_dn_from_dot * n_y;
	dL_dnormals[i * 3 + 2] = -2.0f * dot_vn * dL_dR_z + dL_dn_from_dot * n_z;
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
			if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
				// 48D: 1D density + 45D Φ features (15 x 3D vectors) - test if 45D can learn with fixed normals
				local_density_network_config["n_output_dims"] = 48;
				printf("%s mode: Set density network output dims to 48 (1D density + 45D Phi)\n", m_method.c_str());
			} else {
				local_density_network_config["n_output_dims"] = 16;
				printf("Baseline mode: Set density network output dims to 16\n");
			}
		}
		m_density_network.reset(create_network<T>(local_density_network_config));

		printf("density_network->padded_output_width(): %d\n", m_density_network->padded_output_width());

		if (m_method == "surface_normal") {
			// Surface_normal: 16 surface features + encoded view dirs + encoded normals
			uint32_t total_before_padding = 16 + m_dir_encoding->padded_output_width() + m_dir_encoding->padded_output_width();
			m_rgb_network_input_width = next_multiple(total_before_padding, rgb_alignment);
			printf("Surface_normal mode RGB input calculation: 16 + %d + %d = %d -> next_multiple(..., %d) = %d\n", 
				m_dir_encoding->padded_output_width(), 
				m_dir_encoding->padded_output_width(),
				total_before_padding, rgb_alignment, m_rgb_network_input_width);
			printf("Surface_normal buffer layout: [0:15] surface_features, [16:%d] dir_encoding, [%d:%d] normal_encoding\n",
				15 + m_dir_encoding->padded_output_width(), 16 + m_dir_encoding->padded_output_width(), 
				15 + m_dir_encoding->padded_output_width() + m_dir_encoding->padded_output_width());
		} else if (m_method == "surface_reflect") {
			// Surface_reflect: 16 surface features + encoded view dirs + encoded reflection vectors
			uint32_t total_before_padding = 16 + m_dir_encoding->padded_output_width() + m_dir_encoding->padded_output_width();
			m_rgb_network_input_width = next_multiple(total_before_padding, rgb_alignment);
			printf("Surface_reflect mode RGB input calculation: 16 + %d + %d = %d -> next_multiple(..., %d) = %d\n", 
				m_dir_encoding->padded_output_width(), 
				m_dir_encoding->padded_output_width(),
				total_before_padding, rgb_alignment, m_rgb_network_input_width);
			printf("Surface_reflect buffer layout: [0:15] surface_features, [16:%d] dir_encoding, [%d:%d] reflection_encoding\n",
				15 + m_dir_encoding->padded_output_width(), 16 + m_dir_encoding->padded_output_width(), 
				15 + m_dir_encoding->padded_output_width() + m_dir_encoding->padded_output_width());
		} else if (m_method == "surface") {
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
	void set_use_eikonal_loss(bool v) { m_use_eikonal_loss = v; }
	void set_eikonal_weight(float weight) { m_eikonal_weight = weight; }

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();

		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		
		// CRITICAL: For surface modes, force AoS layout to match Frequency encoding behavior
		MatrixLayout surface_layout = (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") ? AoS : m_dir_encoding->preferred_output_layout();
		// FIXED: Use the same RGB network input width as training (m_rgb_network_input_width) for all modes
		GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, surface_layout};

		// CRITICAL FIX: Zero out the RGB network input buffer in inference mode too
		CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_input.data(), 0, rgb_network_input.n_bytes(), stream));

		GPUMatrixDynamic<T> density_network_output;
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
			// CRITICAL: For surface modes, use AoS layout for density buffer to ensure copy compatibility
			// This forces SphericalHarmonics to behave like Frequency encoding
			density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
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
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
			// Surface modes: Compute analytical normals and use them for surface features
			
			// Compute normalized analytical normals for inference
			GPUMatrixDynamic<float> analytical_normals = compute_analytical_normals_inference_unnormalized(
				stream, batch_size, input, density_network_input, density_network_output, use_inference_params
			);
			
			// Compute surface features directly using normalized analytical normals (all 16 channels in one kernel)
			linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
				batch_size,
				density_network_output.layout() == AoS ? density_network_output.stride() : 1,
				density_network_output.data(),
				analytical_normals.data(),  // Pass normalized analytical normals
				rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
				rgb_network_input.data()
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
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == AoS ? density_network_output.stride() : 1,
			output.layout() == AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == AoS ? 1 : batch_size)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		uint32_t batch_size = input.n();
		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		
		// CRITICAL: For surface modes, force AoS layout to match Frequency encoding behavior
		MatrixLayout surface_layout = (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") ? AoS : m_dir_encoding->preferred_output_layout();
		forward->rgb_network_input = GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, surface_layout};

		// CRITICAL FIX: Zero out the RGB network input buffer to prevent garbage in unused sections
		// This is especially important for surface_normal mode which has larger buffers
		CUDA_CHECK_THROW(cudaMemsetAsync(forward->rgb_network_input.data(), 0, forward->rgb_network_input.n_bytes(), stream));

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients || m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect" // Always prepare gradients for surface modes
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
			
			// Compute normalized analytical normals using NeuS2 pattern (no gradient flow to parameters)
			forward->analytical_normals = compute_analytical_normals_forward_unnormalized(
				stream, batch_size, input, forward, use_inference_params
			);
			
			// Compute surface features directly into RGB slice using normalized analytical normals
			auto surface_features_slice = forward->rgb_network_input.slice_rows(0, 16);
			
			// For both surface and surface_normal modes, compute surface features  
			linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
				batch_size,
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				forward->density_network_output.data(),
				forward->analytical_normals.data(),  // Pass normalized analytical normals
				surface_features_slice.layout() == AoS ? surface_features_slice.stride() : 1,
				surface_features_slice.data()
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
			// Both surface and baseline modes extract density from density_network_output
			linear_kernel(extract_density<T>, 0, stream,
				batch_size, 
				forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
				output->layout() == AoS ? padded_output_width() : 1,
				forward->density_network_output.data(), 
				output->data() + 3 * (output->layout() == AoS ? 1 : batch_size)
			);
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
		MatrixLayout surface_layout = (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") ? AoS : m_dir_encoding->preferred_output_layout();
		GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, surface_layout};
		
		// CRITICAL FIX: Zero out the RGB network input gradient buffer to prevent accumulation into uninitialized memory
		// This is especially important for surface_normal mode which has larger buffers with potentially unused sections
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb_network_input.data(), 0, dL_drgb_network_input.n_bytes(), stream));
		
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
		
		// Backprop through dir encoding
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			GPUMatrixDynamic<T> dL_ddir_encoding_output;
			if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
				dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
			} else {
				dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			}
			
			GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			GPUMatrixDynamic<T> dir_encoding_forward_output;
			if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
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
		if (m_method == "surface" || m_method == "surface_normal" || m_method == "surface_reflect") {
			// CRITICAL: For surface modes, use AoS layout for density gradient buffer to ensure copy compatibility
			// This forces SphericalHarmonics to behave like Frequency encoding
			dL_ddensity_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
		} else {
			dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
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
				dL_dnormals.data()  // Collect gradients w.r.t. normals
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
				
				// Create gradient buffers for view directions and normals from reflection computation
				GPUMatrixDynamic<float> dL_dview_dirs_from_reflection{m_dir_encoding->input_width(), batch_size, stream, view_dirs_input.layout()};
				GPUMatrixDynamic<float> dL_dnormals_from_reflection{3, batch_size, stream, forward.analytical_normals.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dview_dirs_from_reflection.data(), 0, dL_dview_dirs_from_reflection.n_bytes(), stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals_from_reflection.data(), 0, dL_dnormals_from_reflection.n_bytes(), stream));
				
				// Backward through reflection vector computation (need to update the backward kernel signature)
				// For now, we'll accumulate gradients only to normals to avoid view direction gradient complications
				// TODO: Add proper view direction gradient accumulation if needed
				
				// Accumulate gradients from reflection to main normal gradients
				linear_kernel(add_to_buffer_kernel<float>, 0, stream,
					dL_dnormals_from_reflection.n_elements(),
					dL_dnormals_from_reflection.data(),
					dL_dnormals.data()
				);
			}
			
			// Backpropagate gradients through analytical normals to density network parameters
			accumulate_analytical_normal_gradients(
				stream, batch_size, input, forward, dL_dnormals,
				dL_ddensity_network_output, use_inference_params, param_gradients_mode
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
				dL_doutput.layout() == AoS ? dL_doutput.stride() : 1,
				dL_doutput.data(),
				dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1,
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

	// Helper function to compute analytical normals during forward pass (NeuS2 pattern) - UNNORMALIZED
	GPUMatrixDynamic<float> compute_analytical_normals_forward_unnormalized(
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
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};  // FIXED: Use AoS for consistency
		
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
		
		// Store raw gradients for backward pass
		GPUMatrixDynamic<float> raw_grads = dSDF_dpos.slice_rows(0, 3);
		forward->raw_gradients = GPUMatrixDynamic<float>{3, batch_size, stream, raw_grads.layout()};
		CUDA_CHECK_THROW(cudaMemcpyAsync(forward->raw_gradients.data(), raw_grads.data(), 
			forward->raw_gradients.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		
		// Normalize gradients to get unit normals: n = -∇SDF / ||∇SDF||
		GPUMatrixDynamic<float> normals{3, batch_size, stream, AoS};  // FIXED: Use AoS layout to match surface buffers
		linear_kernel(normalize_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data()
		);
		return normals;
		
		// COMMENTED: Unnormalized version
		// return dSDF_dpos.slice_rows(0, 3);
	}

	// Helper function to compute analytical normals during forward pass (NeuS2 pattern) - NORMALIZED
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
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};  // FIXED: Use AoS for consistency
		
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
		GPUMatrixDynamic<float> normals{3, batch_size, stream, AoS};  // FIXED: Use AoS layout to match surface buffers
		linear_kernel(normalize_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data()
		);
		
		return normals;
	}

	// Helper function for inference-only analytical normals - UNNORMALIZED
	GPUMatrixDynamic<float> compute_analytical_normals_inference_unnormalized(
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
		
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};  // FIXED: Use AoS for consistency
		
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
		
		// Normalize gradients to get unit normals: n = -∇SDF / ||∇SDF||
		GPUMatrixDynamic<float> normals{3, batch_size, stream, AoS};  // FIXED: Use AoS layout to match surface buffers
		linear_kernel(normalize_analytical_gradients_kernel<float>, 0, stream,
			batch_size,
			dSDF_dpos.data(),
			normals.data()
		);
		return normals;
		
		// COMMENTED: Unnormalized version
		// return dSDF_dpos.slice_rows(0, 3);
	}

	// Helper function for inference-only analytical normals - NORMALIZED
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
		
		GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, AoS};  // FIXED: Use AoS for consistency
		
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

	// Second-order gradient accumulation with chain rule through normalization
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
	bool m_use_eikonal_loss = false;
	float m_eikonal_weight = 0.01f;

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

		// Analytical normals (∂SDF/∂xyz) - stored in forward context for backward pass
		GPUMatrixDynamic<float> dSDF_dPos;
		GPUMatrixDynamic<float> raw_gradients;     // Raw ∇SDF before normalization
		GPUMatrixDynamic<float> analytical_normals;
		
		// For surface_normal mode: encoded normals
		GPUMatrixDynamic<T> encoded_normals_out;         // Output slice for encoded normals
	};
};

} 