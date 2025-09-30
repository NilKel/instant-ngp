/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_helpers.h
 *  @author Nikolaus Binder, NVIDIA
 *  @brief  CUDA kernel helpers for NeRF network operations.
 *          Contains forward/backward pass kernels for surface features,
 *          analytical normals, divergence computation, and SDF conversion.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

namespace ngp {

// ============================================================================
// FORWARD PASS KERNELS
// ============================================================================

/**
 * @brief Converts SDF values to density (NeuS2 formulation) or passes through density values.
 * 
 * Applies the NeuS2 SDF-to-density conversion: density = s * sigmoid(-sdf*s) * (1 - sigmoid(-sdf*s))
 * where s = exp(variance * 10). Includes NaN protection and value clamping.
 * 
 * @param n_elements Number of samples to process
 * @param density_stride Stride for density input (1 for SoA, width for AoS)
 * @param rgbd_stride Stride for RGBD output (1 for SoA, width for AoS)
 * @param density Input SDF or density values
 * @param rgbd Output density values (written to channel 0)
 * @param sdf_mode If true, apply SDF-to-density conversion; otherwise copy directly
 * @param variance_params Learnable variance parameter for NeuS2 (optional)
 */
template <typename T>
__global__ void extract_density(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ density,
	T* __restrict__ rgbd,
	bool sdf_mode = false,
	const T* __restrict__ variance_params = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	T density_val = density[i * density_stride];
	
	if (sdf_mode) {
		// NeuS2 SDF-to-density conversion with NaN protection
		const float sdf = float(density_val);
		const float sdf_clamped = fmaxf(fminf(sdf, 10.0f), -10.0f);
		
		const float variance = variance_params ? float(variance_params[0]) : 0.12f;
		const float variance_clamped = fmaxf(fminf(variance, 2.0f), -2.0f);
		const float s = expf(variance_clamped * 10.0f);
		const float s_clamped = fminf(s, 1000.0f);
		
		const float sigmoid_arg = -sdf_clamped * s_clamped;
		const float sigmoid_arg_clamped = fmaxf(fminf(sigmoid_arg, 50.0f), -50.0f);
		const float sigmoid_sdf = 1.0f / (1.0f + expf(sigmoid_arg_clamped));
		
		float density_result = s_clamped * sigmoid_sdf * (1.0f - sigmoid_sdf);
		
		if (!isfinite(density_result)) {
			density_result = 0.0f;
		}
		
		density_val = T(density_result);
	}
	
	rgbd[i * rgbd_stride] = density_val;
}

/**
 * @brief Extracts RGB channels from RGBD output.
 * 
 * @param n_elements Total number of RGB values (batch_size * 3)
 * @param rgb_stride Stride for RGB output
 * @param output_stride Stride for RGBD input
 * @param rgbd Input RGBD values
 * @param rgb Output RGB values (channels 0-2)
 */
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

/**
 * @brief Sets a constant value to a specific channel across all samples.
 * 
 * Used for initializing gradient buffers or setting dummy values for autodiff.
 * 
 * @param batch_size Number of samples
 * @param value Constant value to set
 * @param stride Stride between samples (1 for SoA, width for AoS)
 * @param data Output buffer (writes to channel 0 of each sample)
 */
template <typename T>
__global__ void set_constant_value_view_kernel(
	const uint32_t batch_size,
	const T value,
	const uint32_t stride,
	T* __restrict__ data
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= batch_size) return;
	
	data[i * stride] = value;
}

/**
 * @brief Computes 16D surface features from 48D density output and analytical normals.
 * 
 * Feature computation:
 * - Channel 0: Density (optionally converted from SDF)
 * - Channels 1-15: ReLU(-Φ_k · n) for 15 3D vector fields Φ_k from density output
 * 
 * @param n_elements Number of samples
 * @param density_stride Stride for 48D density output
 * @param density_output 48D output: [SDF/density, Φ_0, ..., Φ_14] where each Φ_k is 3D
 * @param analytical_normals 3D analytical normals per sample
 * @param slice_stride Stride for output slice
 * @param rgb_slice Output 16D surface features
 * @param sdf_mode If true, convert SDF to density for channel 0
 * @param variance_params Variance parameters for SDF conversion (optional)
 */
template <typename T>
__global__ void compute_surface_features_to_slice_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,
	const float* __restrict__ analytical_normals,
	const uint32_t slice_stride,
	T* __restrict__ rgb_slice,
	bool sdf_mode = false,
	const T* __restrict__ variance_params = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Channel 0: Copy density/SDF, converting if needed
	T density_val = density_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)];
	
	if (sdf_mode) {
		const float sdf = float(density_val);
		const float sdf_clamped = fmaxf(fminf(sdf, 10.0f), -10.0f);
		
		const float variance = variance_params ? float(variance_params[0]) : 0.12f;
		const float variance_clamped = fmaxf(fminf(variance, 2.0f), -2.0f);
		const float s = expf(variance_clamped * 10.0f);
		const float s_clamped = fminf(s, 1000.0f);
		
		const float sigmoid_arg = -sdf_clamped * s_clamped;
		const float sigmoid_arg_clamped = fmaxf(fminf(sigmoid_arg, 50.0f), -50.0f);
		const float sigmoid_sdf = 1.0f / (1.0f + expf(sigmoid_arg_clamped));
		
		float density_result = s_clamped * sigmoid_sdf * (1.0f - sigmoid_sdf);
		
		if (!isfinite(density_result)) {
			density_result = 0.0f;
		}
		
		density_val = T(density_result);
	}
	
	rgb_slice[i * slice_stride + 0 * (slice_stride == 1 ? n_elements : 1)] = density_val;
	
	// Get analytical normal
	const float* normal = analytical_normals + i * 3;
	const T surface_scale = T(3.0f);
	
	// Channels 1-15: ReLU(-Φ_k · n)
	for (uint32_t k = 0; k < 15; ++k) {
		T phi_x = density_output[i * density_stride + (1 + k * 3 + 0) * (density_stride == 1 ? n_elements : 1)];
		T phi_y = density_output[i * density_stride + (1 + k * 3 + 1) * (density_stride == 1 ? n_elements : 1)];
		T phi_z = density_output[i * density_stride + (1 + k * 3 + 2) * (density_stride == 1 ? n_elements : 1)];
		
		T dot_product = phi_x * T(normal[0]) + phi_y * T(normal[1]) + phi_z * T(normal[2]);
		T surface_feature = fmaxf(T(0.0f), -dot_product);
		
		rgb_slice[i * slice_stride + (1 + k) * (slice_stride == 1 ? n_elements : 1)] = surface_feature * surface_scale;
	}
}

/**
 * @brief Processes analytical gradients (∇SDF) to normals.
 * 
 * Converts raw gradients to normals via: n = -∇SDF / ||∇SDF|| (normalized mode)
 * or n = -∇SDF (raw mode with optional clamping).
 * 
 * @param n_elements Number of samples
 * @param dSDF_dPos Analytical gradients ∇SDF (3D per sample)
 * @param normals Output normals (3D per sample)
 * @param normalize If true, normalize to unit vectors; otherwise use raw gradients
 * @param clamp_magnitude If true, clamp gradient magnitude
 * @param max_magnitude Maximum allowed gradient magnitude (if clamping)
 */
template <typename T>
__global__ void process_analytical_gradients_kernel(
	const uint32_t n_elements,
	const T* __restrict__ dSDF_dPos,
	T* __restrict__ normals,
	const bool normalize = true,
	const bool clamp_magnitude = false,
	const float max_magnitude = 1.0f
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	T grad_x = dSDF_dPos[i * 3 + 0];
	T grad_y = dSDF_dPos[i * 3 + 1];  
	T grad_z = dSDF_dPos[i * 3 + 2];
	
	T magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
	
	if (normalize) {
		T inv_mag = magnitude > T(1e-6f) ? T(-1.0f) / magnitude : T(0.0f);
		
		T norm_x = fmaxf(-1.0f, fminf(1.0f, grad_x * inv_mag));
		T norm_y = fmaxf(-1.0f, fminf(1.0f, grad_y * inv_mag));
		T norm_z = fmaxf(-1.0f, fminf(1.0f, grad_z * inv_mag));
		
		normals[i * 3 + 0] = norm_x;
		normals[i * 3 + 1] = norm_y;
		normals[i * 3 + 2] = norm_z;
	} else {
		grad_x = -grad_x;
		grad_y = -grad_y;
		grad_z = -grad_z;
		
		if (clamp_magnitude && magnitude > max_magnitude) {
			T scale_factor = max_magnitude / (magnitude + 1e-8f);
			grad_x *= scale_factor;
			grad_y *= scale_factor;
			grad_z *= scale_factor;
		}
		
		normals[i * 3 + 0] = grad_x;
		normals[i * 3 + 1] = grad_y;
		normals[i * 3 + 2] = grad_z;
	}
}

/**
 * @brief Calculates reflection vectors from view directions and normals.
 * 
 * Computes R = v - 2(v·n)n where v is the incident ray direction (negated view direction)
 * and n is the surface normal. Output is normalized.
 * 
 * @param n_elements Number of samples
 * @param view_dirs View direction vectors (may have >3 components, only first 3 used)
 * @param normals 3D surface normals
 * @param reflection_vectors Output 3D reflection vectors (normalized)
 * @param view_width Width of view direction input
 * @param view_stride Stride for view directions
 * @param normal_stride Stride for normals (always 3)
 * @param reflect_stride Stride for reflection vectors (always 3)
 */
template <typename T>
__global__ void calculate_reflection_vector_kernel(
	const uint32_t n_elements,
	const float* __restrict__ view_dirs,
	const float* __restrict__ normals,
	float* __restrict__ reflection_vectors,
	const uint32_t view_width,
	const uint32_t view_stride,
	const uint32_t normal_stride,
	const uint32_t reflect_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Layout-aware indexing
	uint32_t v_x_idx, v_y_idx, v_z_idx;
	if (view_stride == view_width) {
		v_x_idx = i * view_width + 0;
		v_y_idx = i * view_width + 1;
		v_z_idx = i * view_width + 2;
	} else {
		v_x_idx = 0 * n_elements + i;
		v_y_idx = 1 * n_elements + i;
		v_z_idx = 2 * n_elements + i;
	}
	
	const uint32_t n_x_idx = (normal_stride == 3) ? (i * 3 + 0) : (0 * n_elements + i);
	const uint32_t n_y_idx = (normal_stride == 3) ? (i * 3 + 1) : (1 * n_elements + i);
	const uint32_t n_z_idx = (normal_stride == 3) ? (i * 3 + 2) : (2 * n_elements + i);
	
	const uint32_t r_x_idx = (reflect_stride == 3) ? (i * 3 + 0) : (0 * n_elements + i);
	const uint32_t r_y_idx = (reflect_stride == 3) ? (i * 3 + 1) : (1 * n_elements + i);
	const uint32_t r_z_idx = (reflect_stride == 3) ? (i * 3 + 2) : (2 * n_elements + i);
	
	// Negate view direction to get incident direction
	const float v_x = -view_dirs[v_x_idx];
	const float v_y = -view_dirs[v_y_idx];
	const float v_z = -view_dirs[v_z_idx];
	
	const float n_x = normals[n_x_idx];
	const float n_y = normals[n_y_idx];
	const float n_z = normals[n_z_idx];
	
	const float dot_vn = v_x * n_x + v_y * n_y + v_z * n_z;
	
	// Reflection: R = v - 2(v·n)n
	float r_x = v_x - 2.0f * dot_vn * n_x;
	float r_y = v_y - 2.0f * dot_vn * n_y;
	float r_z = v_z - 2.0f * dot_vn * n_z;
	
	// Normalize
	float r_mag = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);
	float inv_r_mag = (r_mag > 1e-8f) ? (1.0f / r_mag) : 0.0f;
	
	reflection_vectors[r_x_idx] = r_x * inv_r_mag;
	reflection_vectors[r_y_idx] = r_y * inv_r_mag;
	reflection_vectors[r_z_idx] = r_z * inv_r_mag;
}

/**
 * @brief Computes divergence-based features for volume rendering.
 * 
 * Copies density and 15D divergence values (∇·Φ_k for k=0..14) to RGB input.
 * 
 * @param n_elements Number of samples
 * @param density_stride Stride for 48D density output
 * @param density_output 48D density output
 * @param divergences 15D divergence values per sample
 * @param rgb_stride Stride for RGB input
 * @param rgb_input Output 16D features (density + 15 divergences)
 */
template <typename T>
__global__ void compute_volume_divergence_kernel(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const T* __restrict__ density_output,
	const float* __restrict__ divergences,
	const uint32_t rgb_stride,
	T* __restrict__ rgb_input
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* density_base = density_output + i * density_stride;
	const float* divergence = divergences + i * 15;
	T* rgb_base = rgb_input + i * rgb_stride;
	
	rgb_base[0] = density_base[0];
	
	for (uint32_t k = 0; k < 15; ++k) {
		rgb_base[1 + k] = T(divergence[k]);
	}
}

/**
 * @brief Extracts diagonal elements of divergence: ∇·Φ_k = ∂Φ_kx/∂x + ∂Φ_ky/∂y + ∂Φ_kz/∂z.
 * 
 * @param n_elements Number of samples
 * @param spatial_gradients 3D spatial gradients [∂Φ_kx/∂x, ∂Φ_ky/∂y, ∂Φ_kz/∂z] per sample
 * @param divergence_output Output divergence values (1D per sample)
 * @param grad_stride Stride for spatial gradients
 */
static __global__ void compute_divergence_diagonal_kernel(
	const uint32_t n_elements,
	const float* __restrict__ spatial_gradients,
	float* __restrict__ divergence_output,
	const uint32_t grad_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* grad_base = spatial_gradients + i * grad_stride;
	float divergence = grad_base[0] + grad_base[1] + grad_base[2];
	divergence_output[i] = divergence;
}

/**
 * @brief Extracts and accumulates individual spatial gradient for divergence computation.
 * 
 * Accumulates ∂Φ_k_dim/∂dim to divergences[k] for a specific vector field k and dimension.
 * 
 * @param n_elements Number of samples
 * @param position_gradients Gradients w.r.t. position (3D+ per sample)
 * @param spatial_dim Spatial dimension index (0=x, 1=y, 2=z)
 * @param grad_stride Stride for position gradients
 * @param div_stride Stride for divergences
 * @param vector_field_idx Vector field index k (0-14)
 * @param divergences 15D divergence buffer (accumulate into)
 */
static __global__ void extract_spatial_gradient_kernel(
	const uint32_t n_elements,
	const float* __restrict__ position_gradients,
	const uint32_t spatial_dim,
	const uint32_t grad_stride,
	const uint32_t div_stride,
	const uint32_t vector_field_idx,
	float* __restrict__ divergences
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* grad_base = position_gradients + i * grad_stride;
	float spatial_gradient = grad_base[spatial_dim];
	
	float* div_base = divergences + i * div_stride;
	div_base[vector_field_idx] += spatial_gradient;
}

/**
 * @brief Extracts density features from hash interpolation for hash_surface method.
 * 
 * Hash interpolation has 4 features per level: [vector_x, vector_y, vector_z, density].
 * This kernel extracts the 4th feature (density) from each level.
 * 
 * @param n_elements Number of samples
 * @param n_levels Number of hash levels
 * @param features_per_level Features per level (should be 4)
 * @param hash_features Input hash interpolation output (n_levels * 4 per sample)
 * @param hash_stride Stride for hash features
 * @param density_features Output density features (n_levels per sample)
 * @param density_stride Stride for density features
 */
template <typename T>
__global__ void extract_hash_density_features_kernel(
	const uint32_t n_elements,
	const uint32_t n_levels,
	const uint32_t features_per_level,
	const T* __restrict__ hash_features,
	const uint32_t hash_stride,
	T* __restrict__ density_features,
	const uint32_t density_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	for (uint32_t level = 0; level < n_levels; ++level) {
		uint32_t hash_idx = level * features_per_level + 3;
		uint32_t hash_offset = i * hash_stride + hash_idx * (hash_stride == 1 ? n_elements : 1);
		uint32_t density_offset = i * density_stride + level * (density_stride == 1 ? n_elements : 1);
		
		density_features[density_offset] = hash_features[hash_offset];
	}
}

/**
 * @brief Computes hash surface features from vector potential and analytical normals.
 * 
 * For each hash level, computes ReLU(-vector_potential_k · normals) as surface feature.
 * 
 * @param n_elements Number of samples
 * @param n_levels Number of hash levels
 * @param features_per_level Features per level (should be 4)
 * @param hash_features Input hash interpolation output (n_levels * 4 per sample)
 * @param hash_stride Stride for hash features
 * @param analytical_normals 3D analytical normals per sample
 * @param rgb_features Output surface features (n_levels per sample, NO density)
 * @param rgb_stride Stride for RGB features
 */
template <typename T>
__global__ void compute_hash_surface_features_kernel(
	const uint32_t n_elements,
	const uint32_t n_levels,
	const uint32_t features_per_level,
	const T* __restrict__ hash_features,
	const uint32_t hash_stride,
	const float* __restrict__ analytical_normals,
	T* __restrict__ rgb_features,
	const uint32_t rgb_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* normal = analytical_normals + i * 3;
	uint32_t rgb_base_offset = i * rgb_stride;
	
	for (uint32_t level = 0; level < n_levels; ++level) {
		uint32_t hash_base = i * hash_stride + level * features_per_level * (hash_stride == 1 ? n_elements : 1);
		
		T vector_x = hash_features[hash_base + 0 * (hash_stride == 1 ? n_elements : 1)];
		T vector_y = hash_features[hash_base + 1 * (hash_stride == 1 ? n_elements : 1)];
		T vector_z = hash_features[hash_base + 2 * (hash_stride == 1 ? n_elements : 1)];
		
		float dot_product = (float)vector_x * normal[0] + (float)vector_y * normal[1] + (float)vector_z * normal[2];
		T surface_feature = fmaxf(T(0.0f), -dot_product);
		
		uint32_t rgb_offset = rgb_base_offset + level * (rgb_stride == 1 ? n_elements : 1);
		rgb_features[rgb_offset] = surface_feature;
	}
}

// ============================================================================
// BACKWARD PASS KERNELS
// ============================================================================

/**
 * @brief Backward pass for SDF-to-density conversion.
 * 
 * Applies chain rule: dL/dSDF = dL/ddensity * ddensity/dSDF
 * where ddensity/dSDF = s² * sigmoid * (1-sigmoid) * (1-2*sigmoid).
 * 
 * @param n_elements Number of samples
 * @param density_stride Stride for SDF input
 * @param rgbd_stride Stride for density gradients
 * @param dL_drgbd Gradients w.r.t. density output
 * @param dL_ddensity Gradients w.r.t. SDF input (accumulate into)
 * @param sdf_values Original SDF values from forward pass
 * @param dL_dvariance Gradients w.r.t. variance parameter (optional)
 * @param sdf_mode If true, apply chain rule; otherwise copy directly
 * @param variance_params Variance parameters from forward pass (optional)
 */
template <typename T>
__global__ void extract_density_backward(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ dL_drgbd,
	T* __restrict__ dL_ddensity,
	const T* __restrict__ sdf_values,
	T* __restrict__ dL_dvariance,
	bool sdf_mode = false,
	const T* __restrict__ variance_params = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	T dL_density_val = dL_drgbd[i * rgbd_stride];
	
	if (sdf_mode) {
		const float sdf = float(sdf_values[i * density_stride]);
		const float sdf_clamped = fmaxf(fminf(sdf, 10.0f), -10.0f);
		
		const float variance = variance_params ? float(variance_params[0]) : 0.12f;
		const float variance_clamped = fmaxf(fminf(variance, 2.0f), -2.0f);
		const float s = expf(variance_clamped * 10.0f);
		const float s_clamped = fminf(s, 1000.0f);
		
		const float sigmoid_arg = -sdf_clamped * s_clamped;
		const float sigmoid_arg_clamped = fmaxf(fminf(sigmoid_arg, 50.0f), -50.0f);
		const float sigmoid_sdf = 1.0f / (1.0f + expf(sigmoid_arg_clamped));
		
		float ddensity_dsdf = s_clamped * s_clamped * sigmoid_sdf * (1.0f - sigmoid_sdf) * (1.0f - 2.0f * sigmoid_sdf);
		
		if (!isfinite(ddensity_dsdf)) {
			ddensity_dsdf = 0.0f;
		}
		
		dL_ddensity[i * density_stride] += dL_density_val * T(ddensity_dsdf);
	} else {
		dL_ddensity[i * density_stride] += dL_density_val;
	}
}

/**
 * @brief Backward pass for surface features computation.
 * 
 * Backpropagates gradients from 16D surface features to:
 * - 48D density output (density + 15 3D vector fields Φ_k)
 * - 3D analytical normals
 * Handles ReLU gating and optional SDF-to-density conversion.
 * 
 * @param n_elements Number of samples
 * @param slice_stride Stride for RGB slice gradients
 * @param dL_drgb_slice Gradients w.r.t. 16D surface features
 * @param analytical_normals Normals from forward pass
 * @param density_stride Stride for 48D density output
 * @param density_output Density output from forward pass (for ReLU condition)
 * @param dL_ddensity_output Gradients w.r.t. 48D density output (accumulate into)
 * @param dL_dnormals Gradients w.r.t. normals (accumulate into)
 * @param sdf_mode If true, apply SDF chain rule for channel 0
 * @param variance_params Variance parameters (optional)
 */
template <typename T>
__global__ void surface_features_slice_backward_kernel(
	const uint32_t n_elements,
	const uint32_t slice_stride,
	const T* __restrict__ dL_drgb_slice,
	const float* __restrict__ analytical_normals,
	const uint32_t density_stride,
	const T* __restrict__ density_output,
	T* __restrict__ dL_ddensity_output,
	float* __restrict__ dL_dnormals,
	bool sdf_mode = false,
	const T* __restrict__ variance_params = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	// Channel 0: density/SDF gradient
	const T dL_ddensity = dL_drgb_slice[i * slice_stride + 0 * (slice_stride == 1 ? n_elements : 1)];
	
	if (sdf_mode) {
		const T sdf_val = density_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)];
		const float sdf = float(sdf_val);
		const float sdf_clamped = fmaxf(fminf(sdf, 10.0f), -10.0f);
		
		const float variance = variance_params ? float(variance_params[0]) : 0.12f;
		const float variance_clamped = fmaxf(fminf(variance, 2.0f), -2.0f);
		const float s = expf(variance_clamped * 10.0f);
		const float s_clamped = fminf(s, 1000.0f);
		
		const float sigmoid_arg = -sdf_clamped * s_clamped;
		const float sigmoid_arg_clamped = fmaxf(fminf(sigmoid_arg, 50.0f), -50.0f);
		const float sigmoid_sdf = 1.0f / (1.0f + expf(sigmoid_arg_clamped));
		
		float ddensity_dsdf = s_clamped * s_clamped * sigmoid_sdf * (1.0f - sigmoid_sdf) * (1.0f - 2.0f * sigmoid_sdf);
		
		if (!isfinite(ddensity_dsdf)) {
			ddensity_dsdf = 0.0f;
		}
		
		dL_ddensity_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)] += dL_ddensity * T(ddensity_dsdf);
	} else {
		dL_ddensity_output[i * density_stride + 0 * (density_stride == 1 ? n_elements : 1)] += dL_ddensity;
	}
	
	const float* normal = analytical_normals + i * 3;
	float* dL_dnormals_base = dL_dnormals + i * 3;
	
	const T surface_scale = T(3.0f);
	
	// Channels 1-45: Backpropagate through ReLU(-Φ_k · n)
	for (uint32_t k = 0; k < 15; ++k) {
		const T dL_dsurface_feature = dL_drgb_slice[i * slice_stride + (1 + k) * (slice_stride == 1 ? n_elements : 1)];
		const T dL_dsurface_feature_unscaled = dL_dsurface_feature * surface_scale;
		
		// Recompute dot product for ReLU condition
		const T* density_base = density_output + i * density_stride;
		const T* phi_k = density_base + 1 + k * 3;
		T dot_product = phi_k[0] * T(normal[0]) + phi_k[1] * T(normal[1]) + phi_k[2] * T(normal[2]);
		
		// ReLU gate: gradient flows only if dot_product < 0
		T dL_ddot_product = T(0.0f);
		if (dot_product < T(0.0f)) {
			dL_ddot_product = -dL_dsurface_feature_unscaled;
		}
		
		// Gradients w.r.t. Φ_k
		const T dL_dphi_x = dL_ddot_product * T(normal[0]);
		const T dL_dphi_y = dL_ddot_product * T(normal[1]);
		const T dL_dphi_z = dL_ddot_product * T(normal[2]);
		
		// Gradients w.r.t. normals
		dL_dnormals_base[0] += (float)dL_ddot_product * (float)phi_k[0];
		dL_dnormals_base[1] += (float)dL_ddot_product * (float)phi_k[1];
		dL_dnormals_base[2] += (float)dL_ddot_product * (float)phi_k[2];
		
		// Accumulate to Φ_k
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 0) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_x;
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 1) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_y;
		dL_ddensity_output[i * density_stride + (1 + k * 3 + 2) * (density_stride == 1 ? n_elements : 1)] += dL_dphi_z;
	}
}

/**
 * @brief Backward pass for reflection vector computation.
 * 
 * Computes gradients w.r.t. normals from gradients w.r.t. reflection vectors.
 * Reflection formula: R = v - 2(v·n)n, so dL/dn via chain rule.
 * 
 * @param n_elements Number of samples
 * @param view_dirs View directions from forward pass
 * @param normals Normals from forward pass
 * @param dL_dreflection Gradients w.r.t. reflection vectors
 * @param view_width Width of view direction input
 * @param view_stride Stride for view directions
 * @param normal_stride Stride for normals
 * @param reflect_stride Stride for reflection gradients
 * @param dL_dnormals Gradients w.r.t. normals (output)
 */
template <typename T>
__global__ void reflection_vector_backward_kernel(
	const uint32_t n_elements,
	const float* __restrict__ view_dirs,
	const float* __restrict__ normals,
	const float* __restrict__ dL_dreflection,
	const uint32_t view_width,
	const uint32_t view_stride,
	const uint32_t normal_stride,
	const uint32_t reflect_stride,
	float* __restrict__ dL_dnormals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	uint32_t v_x_idx, v_y_idx, v_z_idx;
	if (view_stride == view_width) {
		v_x_idx = i * view_width + 0;
		v_y_idx = i * view_width + 1;
		v_z_idx = i * view_width + 2;
	} else {
		v_x_idx = 0 * n_elements + i;
		v_y_idx = 1 * n_elements + i;
		v_z_idx = 2 * n_elements + i;
	}
	
	const uint32_t n_x_idx = (normal_stride == 3) ? (i * 3 + 0) : (0 * n_elements + i);
	const uint32_t n_y_idx = (normal_stride == 3) ? (i * 3 + 1) : (1 * n_elements + i);
	const uint32_t n_z_idx = (normal_stride == 3) ? (i * 3 + 2) : (2 * n_elements + i);
	
	const uint32_t r_x_idx = (reflect_stride == 3) ? (i * 3 + 0) : (0 * n_elements + i);
	const uint32_t r_y_idx = (reflect_stride == 3) ? (i * 3 + 1) : (1 * n_elements + i);
	const uint32_t r_z_idx = (reflect_stride == 3) ? (i * 3 + 2) : (2 * n_elements + i);
	
	const float v_x = -view_dirs[v_x_idx];
	const float v_y = -view_dirs[v_y_idx];
	const float v_z = -view_dirs[v_z_idx];
	
	const float n_x = normals[n_x_idx];
	const float n_y = normals[n_y_idx];
	const float n_z = normals[n_z_idx];
	
	const float dL_dR_x = dL_dreflection[r_x_idx];
	const float dL_dR_y = dL_dreflection[r_y_idx];
	const float dL_dR_z = dL_dreflection[r_z_idx];
	
	const float dot_vn = v_x * n_x + v_y * n_y + v_z * n_z;
	
	// Backward through R = v - 2(v·n)n
	float dL_dn_from_dot = -2.0f * (dL_dR_x * v_x + dL_dR_y * v_y + dL_dR_z * v_z);
	dL_dnormals[n_x_idx] = -2.0f * dot_vn * dL_dR_x + dL_dn_from_dot * n_x;
	dL_dnormals[n_y_idx] = -2.0f * dot_vn * dL_dR_y + dL_dn_from_dot * n_y;
	dL_dnormals[n_z_idx] = -2.0f * dot_vn * dL_dR_z + dL_dn_from_dot * n_z;
}

/**
 * @brief Backward pass for hash surface features.
 * 
 * Backpropagates gradients from surface features to:
 * - Hash features (vector potentials)
 * - Analytical normals
 * 
 * @param n_elements Number of samples
 * @param n_levels Number of hash levels
 * @param features_per_level Features per level (should be 4)
 * @param hash_features Hash features from forward pass
 * @param hash_stride Stride for hash features
 * @param analytical_normals Normals from forward pass
 * @param dL_drgb_features Gradients w.r.t. surface features
 * @param rgb_stride Stride for RGB gradients
 * @param dL_dhash_features Gradients w.r.t. hash features (accumulate into)
 * @param dL_dnormals Gradients w.r.t. normals (accumulate into)
 */
template <typename T>
__global__ void hash_surface_features_backward_kernel(
	const uint32_t n_elements,
	const uint32_t n_levels,
	const uint32_t features_per_level,
	const T* __restrict__ hash_features,
	const uint32_t hash_stride,
	const float* __restrict__ analytical_normals,
	const T* __restrict__ dL_drgb_features,
	const uint32_t rgb_stride,
	T* __restrict__ dL_dhash_features,
	float* __restrict__ dL_dnormals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* normal = analytical_normals + i * 3;
	const T* dL_drgb_base = dL_drgb_features + i * rgb_stride;
	T* dL_dhash_base = dL_dhash_features + i * hash_stride;
	float* dL_dnormal = dL_dnormals + i * 3;
	
	for (uint32_t level = 0; level < n_levels; ++level) {
		T dL_dsurface_feature = dL_drgb_base[level * (rgb_stride == 1 ? n_elements : 1)];
		
		uint32_t hash_level_base = level * features_per_level * (hash_stride == 1 ? n_elements : 1);
		T vector_x = hash_features[i * hash_stride + hash_level_base + 0 * (hash_stride == 1 ? n_elements : 1)];
		T vector_y = hash_features[i * hash_stride + hash_level_base + 1 * (hash_stride == 1 ? n_elements : 1)];
		T vector_z = hash_features[i * hash_stride + hash_level_base + 2 * (hash_stride == 1 ? n_elements : 1)];
		
		float dot_product = (float)vector_x * normal[0] + (float)vector_y * normal[1] + (float)vector_z * normal[2];
		
		float relu_derivative = (dot_product < 0.0f) ? 1.0f : 0.0f;
		T dL_ddot_product = -dL_dsurface_feature * T(relu_derivative);
		
		uint32_t hash_x_offset = hash_level_base + 0 * (hash_stride == 1 ? n_elements : 1);
		uint32_t hash_y_offset = hash_level_base + 1 * (hash_stride == 1 ? n_elements : 1);
		uint32_t hash_z_offset = hash_level_base + 2 * (hash_stride == 1 ? n_elements : 1);
		
		dL_dhash_base[hash_x_offset] += dL_ddot_product * T(normal[0]);
		dL_dhash_base[hash_y_offset] += dL_ddot_product * T(normal[1]);
		dL_dhash_base[hash_z_offset] += dL_ddot_product * T(normal[2]);
		
		dL_dnormal[0] += float(dL_ddot_product * vector_x);
		dL_dnormal[1] += float(dL_ddot_product * vector_y);
		dL_dnormal[2] += float(dL_ddot_product * vector_z);
	}
}

/**
 * @brief Backward pass for volume divergence features.
 * 
 * Copies gradients from RGB input back to divergence buffer for further backprop.
 * 
 * @param n_elements Number of samples
 * @param rgb_stride Stride for RGB gradients
 * @param dL_drgb_input Gradients w.r.t. 16D RGB input
 * @param density_stride Stride for density gradients
 * @param dL_ddensity_output Gradients w.r.t. density (accumulate channel 0 only)
 * @param dL_ddivergences Gradients w.r.t. 15D divergences (output)
 */
template <typename T>
__global__ void volume_divergence_backward_kernel(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const T* __restrict__ dL_drgb_input,
	const uint32_t density_stride,
	T* __restrict__ dL_ddensity_output,
	float* __restrict__ dL_ddivergences
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T* dL_drgb_base = dL_drgb_input + i * rgb_stride;
	T* dL_ddensity_base = dL_ddensity_output + i * density_stride;
	float* dL_ddivergence = dL_ddivergences + i * 15;
	
	dL_ddensity_base[0] += dL_drgb_base[0];
	
	for (uint32_t k = 0; k < 15; ++k) {
		dL_ddivergence[k] = float(dL_drgb_base[1 + k]);
	}
}

/**
 * @brief Distributes density feature gradients back to hash features.
 * 
 * Reverse of extract_hash_density_features_kernel. Copies gradients from
 * extracted density features back to the 4th feature (index 3) of each hash level.
 * 
 * @param n_elements Number of samples
 * @param n_levels Number of hash levels
 * @param density_gradients Gradients w.r.t. extracted density features
 * @param density_stride Stride for density gradients
 * @param hash_gradients Gradients w.r.t. hash features (accumulate into)
 * @param hash_stride Stride for hash gradients
 */
template <typename T>
__global__ void distribute_hash_density_gradients_kernel(
	const uint32_t n_elements,
	const uint32_t n_levels,
	const T* __restrict__ density_gradients,
	const uint32_t density_stride,
	T* __restrict__ hash_gradients,
	const uint32_t hash_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	for (uint32_t level = 0; level < n_levels; ++level) {
		uint32_t density_offset = i * density_stride + level * (density_stride == 1 ? n_elements : 1);
		uint32_t hash_idx = level * 4 + 3;
		uint32_t hash_offset = i * hash_stride + hash_idx * (hash_stride == 1 ? n_elements : 1);
		
		hash_gradients[hash_offset] = density_gradients[density_offset];
	}
}

/**
 * @brief Accumulates density gradient to specific hash feature channel.
 * 
 * @param n_elements Number of samples
 * @param level Hash level index
 * @param features_per_level Features per level (should be 4)
 * @param dL_dextracted_density Gradients w.r.t. extracted density
 * @param extracted_stride Stride for extracted density
 * @param density_idx Index in extracted density features
 * @param dL_dhash_features Gradients w.r.t. hash features (accumulate into)
 * @param hash_stride Stride for hash features
 */
template <typename T>
__global__ void accumulate_density_gradient_to_hash_kernel(
	const uint32_t n_elements,
	const uint32_t level,
	const uint32_t features_per_level,
	const T* __restrict__ dL_dextracted_density,
	const uint32_t extracted_stride,
	const uint32_t density_idx,
	T* __restrict__ dL_dhash_features,
	const uint32_t hash_stride
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	uint32_t extracted_offset = i * extracted_stride + density_idx * (extracted_stride == 1 ? n_elements : 1);
	T dL_ddensity = dL_dextracted_density[extracted_offset];
	
	uint32_t hash_idx = level * features_per_level + 3;
	uint32_t hash_offset = i * hash_stride + hash_idx * (hash_stride == 1 ? n_elements : 1);
	dL_dhash_features[hash_offset] += dL_ddensity;
}

/**
 * @brief Adds density gradient from RGBD back to density output.
 * 
 * @param n_elements Number of samples
 * @param rgbd_stride Stride for RGBD gradients
 * @param rgbd Gradients w.r.t. RGBD (channel 3 is density)
 * @param density_stride Stride for density gradients
 * @param density Gradients w.r.t. density (accumulate into)
 */
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

/**
 * @brief Chain rule backward through normalization: n = -∇SDF / ||∇SDF||.
 * 
 * Computes dL/d(∇SDF) from dL/dn using the Jacobian of normalization.
 * Jacobian: ∂n/∂g = -1/||g|| * I + (g ⊗ g) / ||g||³ where g = ∇SDF.
 * 
 * @param n_elements Number of samples
 * @param dL_dnormals Gradients w.r.t. normalized normals
 * @param raw_gradients Raw gradients ∇SDF from forward pass
 * @param dL_draw_gradients Gradients w.r.t. raw gradients (output)
 */
template <typename T>
__global__ void chain_rule_through_normalization_kernel(
	const uint32_t n_elements,
	const float* __restrict__ dL_dnormals,
	const float* __restrict__ raw_gradients,
	float* __restrict__ dL_draw_gradients
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* dL_dn = dL_dnormals + i * 3;
	const float* grad = raw_gradients + i * 3;
	float* dL_dg = dL_draw_gradients + i * 3;
	
	float grad_mag_sq = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
	float grad_mag = sqrtf(grad_mag_sq + 1e-12f);
	float inv_grad_mag = 1.0f / grad_mag;
	float inv_grad_mag3 = inv_grad_mag * inv_grad_mag * inv_grad_mag;
	
	float dot_product = dL_dn[0] * grad[0] + dL_dn[1] * grad[1] + dL_dn[2] * grad[2];
	
	dL_dg[0] = -dL_dn[0] * inv_grad_mag + grad[0] * dot_product * inv_grad_mag3;
	dL_dg[1] = -dL_dn[1] * inv_grad_mag + grad[1] * dot_product * inv_grad_mag3;
	dL_dg[2] = -dL_dn[2] * inv_grad_mag + grad[2] * dot_product * inv_grad_mag3;
}

/**
 * @brief Adds Eikonal regularization gradients: enforce ||∇SDF|| ≈ 1.
 * 
 * Eikonal loss: L = ||∇SDF|| - 1)².
 * Gradient: dL/d(∇SDF) = 2 * weight * (||∇SDF|| - 1) * ∇SDF / ||∇SDF||.
 * 
 * @param n_elements Number of samples
 * @param raw_gradients Raw gradients ∇SDF from forward pass
 * @param eikonal_weight Regularization weight
 * @param dL_dpos_gradients Gradients w.r.t. position gradients (accumulate into)
 */
template <typename T>
__global__ void add_eikonal_gradients_kernel(
	const uint32_t n_elements,
	const float* __restrict__ raw_gradients,
	const float eikonal_weight,
	float* __restrict__ dL_dpos_gradients
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* grad = raw_gradients + i * 3;
	float* dL_dgrad = dL_dpos_gradients + i * 3;
	
	float grad_norm_sq = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
	float grad_norm = sqrtf(grad_norm_sq + 1e-6f);
	
	float eikonal_error = grad_norm - 1.0f;
	float eikonal_grad_coeff = 2.0f * eikonal_weight * eikonal_error / (grad_norm + 1e-8f);
	
	dL_dgrad[0] += eikonal_grad_coeff * grad[0];
	dL_dgrad[1] += eikonal_grad_coeff * grad[1];
	dL_dgrad[2] += eikonal_grad_coeff * grad[2];
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * @brief Copies normal gradients (3D) to position gradient buffer (3D+).
 * 
 * @param n_elements Number of samples
 * @param dL_dnormals 3D normal gradients
 * @param dL_dpos Position gradients (first 3 components)
 */
template <typename T>
__global__ void copy_normal_gradients_to_pos_kernel(
	const uint32_t n_elements,
	const T* __restrict__ dL_dnormals,
	T* __restrict__ dL_dpos
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	dL_dpos[i * 3 + 0] = dL_dnormals[i * 3 + 0];
	dL_dpos[i * 3 + 1] = dL_dnormals[i * 3 + 1];
	dL_dpos[i * 3 + 2] = dL_dnormals[i * 3 + 2];
}

/**
 * @brief Accumulates SDF gradients to density gradient buffer (channel 0 only).
 * 
 * @param n_elements Number of samples
 * @param source_stride Stride for source gradients
 * @param dL_source Source SDF gradients
 * @param target_stride Stride for target gradients
 * @param dL_target Target density gradients (accumulate channel 0)
 */
template <typename T>
__global__ void accumulate_sdf_gradients_kernel(
	const uint32_t n_elements,
	const uint32_t source_stride,
	const T* __restrict__ dL_source,
	const uint32_t target_stride,
	T* __restrict__ dL_target
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	dL_target[i * target_stride] += dL_source[i * source_stride];
}

/**
 * @brief Element-wise buffer addition: destination += source.
 * 
 * @param n_elements Number of elements
 * @param source Source buffer
 * @param destination Destination buffer (accumulate into)
 */
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

/**
 * @brief Accumulates second-order gradients with reduced weight.
 * 
 * Used for backpropagating through analytical normal computation.
 * Applies weight of 0.1 to prevent gradient explosion.
 * 
 * @param n_elements Number of samples
 * @param source_stride Stride for source gradients
 * @param dL_source Second-order gradients
 * @param target_stride Stride for target gradients
 * @param dL_target Main gradient buffer (accumulate channel 0)
 */
template <typename T>
__global__ void accumulate_second_order_gradients_kernel(
	const uint32_t n_elements,
	const uint32_t source_stride,
	const T* __restrict__ dL_source,
	const uint32_t target_stride,
	T* __restrict__ dL_target
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const T second_order_weight = T(0.1f);
	dL_target[i * target_stride] += second_order_weight * dL_source[i * source_stride];
}

/**
 * @brief Converts and copies float values to type T.
 * 
 * @param n_elements Number of elements
 * @param source Source buffer (float)
 * @param destination Destination buffer (type T)
 */
template <typename T>
__global__ void copy_float_to_T_kernel(
	const uint32_t n_elements,
	const float* __restrict__ source,
	T* __restrict__ destination
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	destination[i] = T(source[i]);
}

} // namespace ngp
