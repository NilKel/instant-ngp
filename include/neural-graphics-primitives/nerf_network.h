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

template <typename T>
__global__ void add_density_from_surface_features(
    const uint32_t n_elements,
    const uint32_t surface_stride,
    const T* __restrict__ dL_dsurface_features,
    const uint32_t density_stride,
    T* __restrict__ dL_ddensity
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    // Surface feature[0] corresponds to density channel[0]
    // Check bounds
    if (surface_stride == 0 || density_stride == 0) {
        printf("ERROR: Invalid strides in add_density_from_surface_features: surface_stride=%d, density_stride=%d\n", surface_stride, density_stride);
        return;
    }
    
    
    dL_ddensity[i * density_stride] += dL_dsurface_features[i * surface_stride];
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

// Compute surface features: for each sample, write first 16 rows of rgb_input as [density, 15 x ReLU(-phi_i · n)]
// density_out: (D_out x N), where channel 0 is density/SDF, channels 1..45 contain Phi (15x3)
// normals: (3 x N)
// rgb_in: (>=16 x N) first 16 rows to be filled
// Layouts are row-major in practice depending on preferred_output_layout, so we pass strides explicitly
static __global__ void compute_surface_features_kernel(
    const uint32_t n_elements,
    const uint32_t density_stride,
    const float* __restrict__ density_out,
    const uint32_t normals_stride,
    const float* __restrict__ normals,
    const uint32_t rgb_in_stride,
    float* __restrict__ rgb_in
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    
    const float* dens_base = density_out + i * density_stride;
    const float* nbase = normals + i * normals_stride;
    float* rgb_base = rgb_in + i * rgb_in_stride;

    // Channel 0: density (copy directly)
    rgb_base[0] = dens_base[0];

    // Channels 1-15: Phi features ReLU(-phi_k · n) for k = 0..14
    // Phi starts at channel 1, stored as 15 consecutive vec3s: channels 1-45
    for (uint32_t k = 0; k < 15; ++k) {
        const float* phi_k = dens_base + 1 + k*3; // Phi_k = channels [1+3k, 2+3k, 3+3k]
        float dotp = phi_k[0]*nbase[0] + phi_k[1]*nbase[1] + phi_k[2]*nbase[2];
        float feat = fmaxf(0.01f * (-dotp), -dotp); // Leaky ReLU(-dotp) with slope 0.01
        rgb_base[1 + k] = feat; // Store in surface feature channels 1..15
    }
}

// Backward: accumulate gradients from surface features (rows 1..15 of rgb_in) to Phi (channels 1..45 of density_out)
// Ignores gradients w.r.t. normals unless enabled elsewhere.
static __global__ void surface_features_backward_to_phi_kernel(
    const uint32_t n_elements,
    const uint32_t dL_ddensity_stride,
    float* __restrict__ dL_ddensity_out,
    const float* __restrict__ density_out,
    const uint32_t density_stride,
    const uint32_t normals_stride,
    const float* __restrict__ normals,
    const uint32_t dL_drgb_in_stride,
    const float* __restrict__ dL_drgb_in
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    // Bounds checking
    if (dL_ddensity_stride < 48 || density_stride < 48 || normals_stride != 3 || dL_drgb_in_stride < 16) {
        
        return;
    }
    
    float* d_dens_base = dL_ddensity_out + i * dL_ddensity_stride;
    const float* dens_base = density_out + i * density_stride;
    const float* nbase = normals + i * normals_stride;
    const float* d_rgb_base = dL_drgb_in + i * dL_drgb_in_stride;

    // For each of 15 Phi features, if ReLU was active (dotp < 0), propagate gradient
    for (uint32_t k = 0; k < 15; ++k) {
        // Bounds check for phi access
        uint32_t phi_offset = 1 + k*3;
        if (phi_offset + 2 >= density_stride) {
            return;
        }
        
        const float* phi_k = dens_base + phi_offset; // Phi_k at channels [1+3k, 2+3k, 3+3k]
        float dotp = phi_k[0]*nbase[0] + phi_k[1]*nbase[1] + phi_k[2]*nbase[2];
        
        // Handle leaky ReLU backward: grad = 1.0 if -dotp > 0, else 0.01
        float grad_scale = (-dotp > 0.0f) ? 1.0f : 0.01f;
        
        // Always propagate some gradient with leaky ReLU
        // Bounds check for gradient access
        if (1 + k >= dL_drgb_in_stride) {
            return;
        }
        
        float grad_contribution = d_rgb_base[1 + k] * grad_scale; // Scale by leaky ReLU gradient
        // Gradient w.r.t. phi_k is -normal * grad_contribution
        atomicAdd(&d_dens_base[phi_offset + 0], -nbase[0] * grad_contribution);
        atomicAdd(&d_dens_base[phi_offset + 1], -nbase[1] * grad_contribution);
        atomicAdd(&d_dens_base[phi_offset + 2], -nbase[2] * grad_contribution);
    }
}

// NeuS-style set_constant_value_view - exactly like the reference
template <typename T>
static __global__ void set_constant_value_view_kernel(
	const uint32_t batch_size,
	const T value,
	T* __restrict__ data
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= batch_size) return;
	
	// Set channel 0 for batch element i
	// In column-major layout (which TCNN typically uses), element [0, i] is at index i
	data[i] = value;
}

// NeuS-style view-based seed function - exactly matches reference
template <typename T>
void set_constant_value_view(cudaStream_t stream, uint32_t batch_size, T value, tcnn::GPUMatrixDynamic<T>& matrix) {
	// Zero out the entire matrix first
	CUDA_CHECK_THROW(cudaMemsetAsync(matrix.data(), 0, matrix.n_bytes(), stream));
	
	// Set channel 0 to the value for all batch elements
	// This assumes the matrix is in column-major format (channels x batch_size)
	// where element [0, i] is at index i
	linear_kernel(set_constant_value_view_kernel<T>, 0, stream, 
		batch_size, value, matrix.data());
}

// Normalize first 3 rows (dSDF/dpos) per sample into normals
static __global__ void normalize_normals_kernel(
	const uint32_t n_elements,
	const uint32_t grad_stride,
	const float* __restrict__ dL_dpos,
	const uint32_t normals_stride,
	float* __restrict__ normals
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	const float* g = dL_dpos + i * grad_stride;
	float nx = g[0];
	float ny = g[1];
	float nz = g[2];
	float l = sqrtf(nx*nx + ny*ny + nz*nz) + 1e-12f;
	float* n = normals + i * normals_stride;
	
	// Remove verbose output
	n[0] = nx / l;
	n[1] = ny / l;
	n[2] = nz / l;
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

template <typename T>
class NerfNetwork : public Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network, const std::string& method = "baseline") : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims}, m_method{method} {
		printf("=== NerfNetwork Constructor ===\n");
		printf("Method: %s\n", m_method.c_str());
		printf("n_pos_dims: %d, n_dir_dims: %d\n", n_pos_dims, n_dir_dims);

		m_pos_encoding.reset(create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = minimum_alignment(rgb_network);
		m_dir_encoding.reset(create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			if (m_method == "surface") {
				local_density_network_config["n_output_dims"] = 48; // 1D density + 15x3D Phi
				printf("Surface mode: Set density network output dims to 48\n");
			} else {
				local_density_network_config["n_output_dims"] = 16; // 1D density + 3D RGB
				printf("Baseline mode: Set density network output dims to 16\n");
			}
		}
		m_density_network.reset(create_network<T>(local_density_network_config));

		// ADD THIS DEBUG CHECK:
		printf("=== DENSITY NETWORK DEBUG ===\n");
		printf("Config n_output_dims: %d\n", local_density_network_config["n_output_dims"].get<int>());
		printf("Actual network output_width: %d\n", m_density_network->output_width());
		printf("Actual network padded_output_width: %d\n", m_density_network->padded_output_width());
		printf("===============================\n");
		// m_rgb_network_input_width = next_multiple(m_dir_encoding->padded_output_width() + std::max(16u, m_density_network->padded_output_width()), rgb_alignment);
		if (m_method == "surface") {
			// Surface: 16 surface features + direction encoding
			m_rgb_network_input_width = next_multiple(16 + m_dir_encoding->padded_output_width(), rgb_alignment);
		} else {
			// Baseline: density output + direction encoding  
			m_rgb_network_input_width = next_multiple(m_dir_encoding->padded_output_width() + std::max(16u, m_density_network->padded_output_width()), rgb_alignment);
		}

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(create_network<T>(local_rgb_network_config));

		m_density_model = std::make_shared<NetworkWithInputEncoding<T>>(m_pos_encoding, m_density_network);

		// // Disable JIT fusion to ensure analytic-normal surface features path is used everywhere
		// this->set_jit_fusion(false);
		// // Also disable JIT for the sub-networks
		// if (m_density_network) {
		// 	m_density_network->set_jit_fusion(false);
		// }
		// if (m_rgb_network) {
		// 	m_rgb_network->set_jit_fusion(false);
		// }
		// if (m_pos_encoding) {
		// 	m_pos_encoding->set_jit_fusion(false);
		// }
		// if (m_dir_encoding) {
		// 	m_dir_encoding->set_jit_fusion(false);
		// }
		
		// printf("=== ALL JIT FUSION DISABLED ===\n");

		
	}

	virtual ~NerfNetwork() { }


	// bool jit_fusion() const {
	// 	printf("=== JIT FUSION QUERY: Forcing FALSE for method %s ===\n", m_method.c_str());
	// 	return false; // Force disable JIT fusion
	// }

	void set_backprop_normals(bool v) { m_backprop_normals = v; }

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();

		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		GPUMatrixDynamic<T> density_network_output;
		if (m_method == "surface") {
			// Surface mode: separate buffer to avoid size mismatch
			density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, RM};
		} else {
			// Baseline: slice from rgb input
			density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		}

		GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		// Standard forward pass first
		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		// Set up direction encoding
		if (m_method == "surface") {
			// Handle direction encoding first (before computing surface features)
			uint32_t available_dir_space = m_rgb_network_input_width - 16; // Space after surface features
			uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			auto dir_out = rgb_network_input.slice_rows(16, dir_encoding_width);
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);

			// Compute analytical normals using the same approach as forward_impl
			// Create separate forward pass for normal computation
			GPUMatrixDynamic<T> density_network_input_normals{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			GPUMatrixDynamic<T> density_network_output_normals{m_density_network->padded_output_width(), batch_size, stream, RM};
			
			// Forward pass through pos encoding for normals
			auto pos_encoding_ctx_normals = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				&density_network_input_normals,
				use_inference_params,
				true  // prepare_input_gradients = true for gradient computation
			);
			
			// Forward pass through density network for normals
			auto density_network_ctx_normals = m_density_network->forward(
				stream,
				density_network_input_normals,
				&density_network_output_normals,
				use_inference_params,
				true  // prepare_input_gradients = true for gradient computation
			);
			
			// Set up gradient computation for analytical normals
			GPUMatrixDynamic<T> dL_dsdf{m_density_network->padded_output_width(), batch_size, stream, RM};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf.data(), 0, dL_dsdf.n_bytes(), stream));
			set_constant_value_view(stream, batch_size, T(1.0f), dL_dsdf);
			
			// Compute analytical gradients via backprop with GradientMode::Ignore
			GPUMatrixDynamic<T> dL_ddensity_network_input_normals{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, CM};
			
			// Backward through density network
			m_density_network->backward(
				stream, 
				*density_network_ctx_normals, 
				density_network_input_normals, 
				density_network_output_normals, 
				dL_dsdf,
				&dL_ddensity_network_input_normals,
				use_inference_params, 
				GradientMode::Ignore
			);
			
			// Backward through pos encoding
			m_pos_encoding->backward(
				stream,
				*pos_encoding_ctx_normals,
				input.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input_normals,
				dL_ddensity_network_input_normals,
				&dSDF_dpos,
				use_inference_params,
				GradientMode::Ignore
			);
			
			// Create analytical normals by normalizing dSDF/dpos
			GPUMatrix<float> normals{3, batch_size, stream};
			linear_kernel(normalize_normals_kernel, 0, stream,
				batch_size,
				dSDF_dpos.layout() == RM ? dSDF_dpos.rows() : dSDF_dpos.stride(),
				dSDF_dpos.data(),
				normals.m(),
				normals.data()
			);

			// Compute surface features with analytical normals
			linear_kernel(compute_surface_features_kernel, 0, stream,
				batch_size,
				density_network_output.layout() == RM ? density_network_output.rows() : density_network_output.stride(),
				(float*)density_network_output.data(),
				normals.m(), normals.data(),
				rgb_network_input.layout() == RM ? rgb_network_input.rows() : rgb_network_input.stride(),
				(float*)rgb_network_input.data()
			);
			
		} else {
			// Baseline: use standard inference calls
			auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);
		}

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		// Extract density to output
		uint32_t density_stride = density_network_output.layout() == RM ? 1 : density_network_output.stride();
		uint32_t rgbd_stride = output.layout() == AoS ? padded_output_width() : 1;
		T* output_ptr = output.data() + 3 * (output.layout() == AoS ? 1 : batch_size);
		
		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_stride,
			rgbd_stride,
			density_network_output.data(),
			output_ptr
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		GPUMatrixDynamic<T> dir_out;
		if (m_method == "surface") {
			forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, RM};
			// CRITICAL FIX: Ensure we don't exceed the RGB input buffer bounds
			uint32_t available_dir_space = m_rgb_network_input_width - 16; // Space after surface features
			uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			
			dir_out = forward->rgb_network_input.slice_rows(16, dir_encoding_width);
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, true);
		} else {
			forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
			dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, false);
		}
		// forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, /*prepare_input_gradients=*/true);
		

		
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
		
		// if (m_method == "surface") {
		// 	printf("=== SURFACE FORWARD DEBUG: Zeroing RGB network input ===\n");
			
		// 	// Zero out the RGB network input
		// 	CUDA_CHECK_THROW(cudaMemsetAsync(
		// 		forward->rgb_network_input.data(), 
		// 		0, 
		// 		forward->rgb_network_input.n_bytes(), 
		// 		stream
		// 	));
			
		// 	printf("RGB input zeroed in forward pass\n");
		// } else {
			
		// }
		// Surface mode: compute analytical normals and surface features
		if (m_method == "surface") {
			
			// Create separate forward pass for normal computation using individual networks (like NeuS)
			GPUMatrixDynamic<T> density_network_input_normals{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			GPUMatrixDynamic<T> density_network_output_normals{m_density_network->padded_output_width(), batch_size, stream, RM};
			
			// Forward pass through pos encoding
			auto pos_encoding_ctx_normals = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				&density_network_input_normals,
				use_inference_params,
				true  // prepare_input_gradients = true for gradient computation
			);
			
			// Forward pass through density network
			auto density_network_ctx_normals = m_density_network->forward(
				stream,
				density_network_input_normals,
				&density_network_output_normals,
				use_inference_params,
				true  // prepare_input_gradients = true for gradient computation
			);
			
			// Set "loss gradient" to 1.0 for SDF output (channel 0) - NeuS style
			GPUMatrixDynamic<T> dL_dsdf{m_density_network->padded_output_width(), batch_size, stream, RM};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dsdf.data(), 0, dL_dsdf.n_bytes(), stream));
			set_constant_value_view(stream, batch_size, T(1.0f), dL_dsdf);
			
			// Compute analytical gradients via backprop with GradientMode::Ignore (NeuS approach)
			GPUMatrixDynamic<T> dL_ddensity_network_input_normals{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			GPUMatrixDynamic<float> dSDF_dpos{m_pos_encoding->input_width(), batch_size, stream, CM};
			
			// Step 1: Backward through density network (like NeuS)
			m_density_network->backward(
				stream, 
				*density_network_ctx_normals, 
				density_network_input_normals, 
				density_network_output_normals, 
				dL_dsdf,  // T-typed SDF gradient buffer
				&dL_ddensity_network_input_normals,  // Output: gradients w.r.t. density network input
				use_inference_params, 
				GradientMode::Ignore  // KEY: Don't affect parameter gradients
			);
			
			// Step 2: Backward through pos encoding (like NeuS)
			m_pos_encoding->backward(
				stream,
				*pos_encoding_ctx_normals,
				input.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input_normals,
				dL_ddensity_network_input_normals,
				&dSDF_dpos,  // Output: analytical gradients dSDF/dpos
				use_inference_params,
				GradientMode::Ignore  // KEY: Don't affect parameter gradients
			);
			
			// Store the normal computation data for backward pass (NeuS2 Phase 2)
			forward->dSDF_dpos = std::move(dSDF_dpos);
			
			// Create analytical normals by normalizing dSDF/dpos
			forward->normals = GPUMatrix<float>{3, batch_size, stream};
			linear_kernel(normalize_normals_kernel, 0, stream,
				batch_size,
				forward->dSDF_dpos.layout() == RM ? forward->dSDF_dpos.rows() : forward->dSDF_dpos.stride(),
				forward->dSDF_dpos.data(),
				forward->normals.m(),
				forward->normals.data()
			);
			
			// Compute surface features with analytical normals and actual Phi
			linear_kernel(compute_surface_features_kernel, 0, stream,
				batch_size,
				forward->density_network_output.layout() == RM ? forward->density_network_output.rows() : forward->density_network_output.stride(),
				(float*)forward->density_network_output.data(),
				forward->normals.m(), forward->normals.data(),
				forward->rgb_network_input.layout() == RM ? forward->rgb_network_input.rows() : forward->rgb_network_input.stride(),
				(float*)forward->rgb_network_input.data()
			);
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		if (output) {
			linear_kernel(extract_density<T>, 0, stream,
				batch_size, forward->density_network_output.layout() == RM ? 1 : forward->density_network_output.stride(), padded_output_width(), forward->density_network_output.data(), output->data()+3
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
		
		GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
		

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
		
			GPUMatrixDynamic<T> dL_ddir_encoding_output;
					if (m_method == "surface") {
			// CRITICAL FIX: Use the same direction encoding width as forward pass
			uint32_t available_dir_space = m_rgb_network_input_width - 16;
			uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(16, dir_encoding_width);
		} else {
			dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		}
			GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			GPUMatrixDynamic<T> dir_encoding_forward_output;
					if (m_method == "surface") {
			// CRITICAL FIX: Use the same direction encoding width as forward pass
			uint32_t available_dir_space = m_rgb_network_input_width - 16;
			uint32_t dir_encoding_width = std::min(m_dir_encoding->padded_output_width(), available_dir_space);
			dir_encoding_forward_output = forward.rgb_network_input.slice_rows(16, dir_encoding_width);
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

		// Map gradients from the first 16 rows of rgb input back to density outputs: ch0 and Phi
		GPUMatrixDynamic<T> dL_ddensity_network_output;
		if (m_method == "surface") {
			dL_ddensity_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, RM};
			// CRITICAL: Initialize to zero before accumulating gradients
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_network_output.data(), 0, dL_ddensity_network_output.n_bytes(), stream));
		} else {
			dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		}
		// Start with zeros and add density and Phi contributions
		

		if (m_method == "surface") {
			
			
			// NeuS2 Phase 2: Extract gradients w.r.t. normals from RGB loss
			
			uint32_t rgb_stride = dL_drgb_network_input.layout() == RM ? dL_drgb_network_input.rows() : dL_drgb_network_input.stride();
			uint32_t density_stride = dL_ddensity_network_output.layout() == RM ? dL_ddensity_network_output.rows() : dL_ddensity_network_output.stride();
			
			// First: add gradient from surface features (channel 0 has density gradient)
			linear_kernel(add_density_from_surface_features<T>, 0, stream,
				batch_size,
				rgb_stride,
				dL_drgb_network_input.data(),  // Source: surface feature gradients
				density_stride,
				dL_ddensity_network_output.data()  // Target: density network output gradients
			);
			
			// Second: add gradient from output density channel
			linear_kernel(add_density_gradient<T>, 0, stream,
				batch_size,
				dL_doutput.m(),
				dL_doutput.data(),
				density_stride,
				dL_ddensity_network_output.data()
			);

			// NeuS2 Phase 2: Use analytical normals for Phi gradients (proper gradient flow)
			
			
			if (forward.normals.data() && forward.normals.n() == batch_size) {
				
			uint32_t fwd_density_stride = forward.density_network_output.layout() == RM ? forward.density_network_output.rows() : forward.density_network_output.stride();
			linear_kernel(surface_features_backward_to_phi_kernel, 0, stream,
				batch_size,
				density_stride,
				(float*)dL_ddensity_network_output.data(),
				(float*)forward.density_network_output.data(),
				fwd_density_stride,
				3,
					forward.normals.data(),  // Use ANALYTICAL normals for proper gradient flow
				rgb_stride,
				(float*)dL_drgb_network_input.data()
			);
				
				
				// NeuS2 Phase 2: Compute gradients w.r.t. normals from surface features
				
				// TODO: Implement backward_backward_input for second-order derivatives
				// This would compute how changes in normals affect the surface features
				// For now, we rely on the standard gradient flow through Phi features
				
			} else {
				
			}

		} else {
			// Baseline method: add gradient from channel 3 of dL_doutput to channel 0 of density
			
			linear_kernel(add_density_gradient<T>, 0, stream,
				batch_size,
				dL_doutput.m(),
				dL_doutput.data(),
				dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
				dL_ddensity_network_output.data()
			);
		}

		GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput || m_backprop_normals) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients. Grad through normals is not propagated unless enabled (not implemented here)
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

	void density(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		// m_density_model->set_jit_fusion(this->jit_fusion());
		m_density_model->set_jit_fusion(false);
		m_density_model->inference_mixed_precision(stream, input.slice_rows(0, m_pos_encoding->input_width()), output, use_inference_params);
	}

	std::unique_ptr<Context> density_forward(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density_forward input must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->density_network_output = GPUMatrixDynamic<T>{output->data(), m_density_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

		return forward;
	}

	void density_backward(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) {
		if (input.layout() != CM || (dL_dinput && dL_dinput->layout() != CM)) {
			throw std::runtime_error("NerfNetwork::density_backward input must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
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
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->padded_output_width();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
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

	std::string generate_device_function(const std::string& name) const override {
		if (m_method == "surface") {
			printf("JIT: Using surface method device function\n");
			// You would need to modify the generated CUDA code here
		} else {
			printf("JIT: Using baseline method device function\n");
		}
		std::string density_network = name + "_density_network";
		std::string rgb_network = name + "_rgb_network";
		std::string pos_encoding = name + "_pos_encoding";
		std::string dir_encoding = name + "_dir_encoding";

		std::ostringstream preamble;
		preamble
			<< m_density_network->generate_device_function(density_network) << "\n\n"
			<< m_rgb_network->generate_device_function(rgb_network) << "\n\n"
			<< m_pos_encoding->generate_device_function(pos_encoding) << "\n\n"
			<< m_dir_encoding->generate_device_function(dir_encoding) << "\n\n"
			;

		std::string body = dfmt(1, R"(
				auto pos_enc_out = {POS_ENC}(input.slice<0, {POS_ENC_DIMS_IN}>(), params + {POS_ENC_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {POS_ENC_FWD_CTX_OFFSET} : nullptr);

				{RGB_MLP_IN} rgb_mlp_in;
				rgb_mlp_in.slice<0, {DENSITY_MLP_DIMS_OUT}>() = {DENSITY_MLP}(pos_enc_out, params, fwd_ctx);
				rgb_mlp_in.slice<{DENSITY_MLP_DIMS_OUT}, {DIR_ENC_DIMS_OUT}>() = {DIR_ENC}(input.slice<{DIR_OFFSET}, {DIR_ENC_DIMS_IN}>(), params + {DIR_ENC_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {DIR_ENC_FWD_CTX_OFFSET} : nullptr);

				auto rgb_mlp_out = {RGB_MLP}(rgb_mlp_in, params + {RGB_MLP_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {RGB_MLP_FWD_CTX_OFFSET} : nullptr);

				return {{rgb_mlp_out[0], rgb_mlp_out[1], rgb_mlp_out[2], rgb_mlp_in[0]}};
			)",
			"POS_ENC"_a = pos_encoding,
			"POS_ENC_DIMS_IN"_a = m_pos_encoding->input_width(),
			"POS_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params(),
			"POS_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes(),
			"RGB_MLP"_a = rgb_network,
			"RGB_MLP_IN"_a = m_rgb_network->generate_vec_in(),
			"RGB_MLP_PARAMS_OFFSET"_a = m_density_network->n_params(),
			"RGB_MLP_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes(),
			"DENSITY_MLP"_a = density_network,
			"DENSITY_MLP_DIMS_OUT"_a = m_density_network->output_width(),
			"DIR_ENC"_a = dir_encoding,
			"DIR_ENC_DIMS_IN"_a = m_dir_encoding->input_width(),
			"DIR_ENC_DIMS_OUT"_a = m_dir_encoding->output_width(),
			"DIR_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params() + m_pos_encoding->n_params(),
			"DIR_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes() + m_pos_encoding->device_function_fwd_ctx_bytes(),
			"DIR_OFFSET"_a = m_dir_offset
		);

		return fmt::format("{}{}", preamble.str(), this->generate_device_function_from_body(name, body));
	}

	std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override {
		std::string density_network = name + "_density_network";
		std::string rgb_network = name + "_rgb_network";
		std::string pos_encoding = name + "_pos_encoding";
		std::string dir_encoding = name + "_dir_encoding";

		std::ostringstream preamble;
		preamble
			<< m_density_network->generate_backward_device_function(density_network, n_threads) << "\n\n"
			<< m_rgb_network->generate_backward_device_function(rgb_network, n_threads) << "\n\n"
			<< m_pos_encoding->generate_backward_device_function(pos_encoding, n_threads) << "\n\n"
			<< m_dir_encoding->generate_backward_device_function(dir_encoding, n_threads) << "\n\n"
			;

		std::string body = dfmt(1, R"(
				bool requires_pos_encoding_bwd = {POS_ENC_N_PARAMS} != 0 || dL_dx;
				bool requires_dir_encoding_bwd = {DIR_ENC_N_PARAMS} != 0 || dL_dx;

				{RGB_MLP_IN} dL_drgb_mlp_in;
				{RGB_MLP}(
					{RGB_MLP_OUT}(dL_dy.rgb()),
					params + {RGB_MLP_PARAMS_OFFSET},
					fwd_ctx + WARP_SIZE * {RGB_MLP_FWD_CTX_OFFSET},
					dL_dparams ? dL_dparams + {RGB_MLP_PARAMS_OFFSET} : nullptr,
					&dL_drgb_mlp_in
				);
				dL_drgb_mlp_in[0] = dL_drgb_mlp_in[0] + dL_dy[3];

				if (requires_dir_encoding_bwd) {{
					{DIR_ENC}(
						dL_drgb_mlp_in.slice<{DENSITY_MLP_DIMS_OUT}, {DIR_ENC_DIMS_OUT}>(),
						params + {DIR_ENC_PARAMS_OFFSET},
						fwd_ctx + WARP_SIZE * {DIR_ENC_FWD_CTX_OFFSET},
						dL_dparams ? dL_dparams + {DIR_ENC_PARAMS_OFFSET} : nullptr,
						dL_dx ? &dL_dx->slice<{DIR_OFFSET}, {DIR_ENC_DIMS_IN}>() : nullptr
					);
				}}

				{POS_ENC_OUT} dL_dpos_enc_out;
				{DENSITY_MLP}(
					dL_drgb_mlp_in.slice<0, {DENSITY_MLP_DIMS_OUT}>(),
					params,
					fwd_ctx,
					dL_dparams,
					requires_pos_encoding_bwd ? &dL_dpos_enc_out : nullptr
				);

				if (requires_pos_encoding_bwd) {{
					{POS_ENC}(
						dL_dpos_enc_out,
						params + {POS_ENC_PARAMS_OFFSET},
						fwd_ctx + WARP_SIZE * {POS_ENC_FWD_CTX_OFFSET},
						dL_dparams ? dL_dparams + {POS_ENC_PARAMS_OFFSET} : nullptr,
						dL_dx ? &dL_dx->slice<0, {POS_ENC_DIMS_IN}>() : nullptr
					);
				}}
			)",
			"POS_ENC"_a = pos_encoding,
			"POS_ENC_DIMS_IN"_a = m_pos_encoding->input_width(),
			"POS_ENC_OUT"_a = m_pos_encoding->generate_vec_out(),
			"POS_ENC_N_PARAMS"_a = m_pos_encoding->n_params(),
			"POS_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params(),
			"POS_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes(),
			"RGB_MLP"_a = rgb_network,
			"RGB_MLP_IN"_a = m_rgb_network->generate_vec_in(),
			"RGB_MLP_OUT"_a = m_rgb_network->generate_vec_out(),
			"RGB_MLP_PARAMS_OFFSET"_a = m_density_network->n_params(),
			"RGB_MLP_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes(),
			"DENSITY_MLP"_a = density_network,
			"DENSITY_MLP_DIMS_OUT"_a = m_density_network->output_width(),
			"DIR_ENC"_a = dir_encoding,
			"DIR_ENC_DIMS_IN"_a = m_dir_encoding->input_width(),
			"DIR_ENC_DIMS_OUT"_a = m_dir_encoding->output_width(),
			"DIR_ENC_N_PARAMS"_a = m_dir_encoding->n_params(),
			"DIR_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params() + m_pos_encoding->n_params(),
			"DIR_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes() + m_pos_encoding->device_function_fwd_ctx_bytes(),
			"DIR_OFFSET"_a = m_dir_offset
		);

		return fmt::format("{}{}", preamble.str(), this->generate_backward_device_function_from_body(name, body));
	}

	uint32_t device_function_fwd_ctx_bytes() const override {
		return
			m_density_network->device_function_fwd_ctx_bytes() +
			m_rgb_network->device_function_fwd_ctx_bytes() +
			m_pos_encoding->device_function_fwd_ctx_bytes() +
			m_dir_encoding->device_function_fwd_ctx_bytes()
			;
	}

	bool device_function_fwd_ctx_aligned_per_element() const override {
		return false;
	}

	uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, GradientMode param_gradients_mode) const override {
		return std::max(
			std::max(
				m_density_network->backward_device_function_shmem_bytes(n_threads, param_gradients_mode),
				m_rgb_network->backward_device_function_shmem_bytes(n_threads, param_gradients_mode)
			),
			std::max(
				m_pos_encoding->backward_device_function_shmem_bytes(n_threads, param_gradients_mode),
				m_dir_encoding->backward_device_function_shmem_bytes(n_threads, param_gradients_mode)
			)
		);
	}

	void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		m_density_network->convert_params_to_jit_layout(stream, use_inference_params);
		m_rgb_network->convert_params_to_jit_layout(stream, use_inference_params);
		m_pos_encoding->convert_params_to_jit_layout(stream, use_inference_params);
		m_dir_encoding->convert_params_to_jit_layout(stream, use_inference_params);
	}

	void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		m_density_network->convert_params_from_jit_layout(stream, use_inference_params);
		m_rgb_network->convert_params_from_jit_layout(stream, use_inference_params);
		m_pos_encoding->convert_params_from_jit_layout(stream, use_inference_params);
		m_dir_encoding->convert_params_from_jit_layout(stream, use_inference_params);
	}
private:
	std::shared_ptr<Network<T>> m_density_network;
	std::shared_ptr<Network<T>> m_rgb_network;
	std::shared_ptr<Encoding<T>> m_pos_encoding;
	std::shared_ptr<Encoding<T>> m_dir_encoding;

	// Aggregates m_pos_encoding and m_density_network
	std::shared_ptr<NetworkWithInputEncoding<T>> m_density_model;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;
	bool m_backprop_normals = false;

	std::string m_method;

	// // Storage of forward pass data
	struct ForwardContext : public Context {
		GPUMatrixDynamic<T> density_network_input;
		GPUMatrixDynamic<T> density_network_output;
		GPUMatrixDynamic<T> rgb_network_input;
		GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;

		// NeuS2-style analytic gradients and normals with gradient flow preservation
		GPUMatrixDynamic<float> dSDF_dpos;  // Analytical gradients for normal computation
		GPUMatrix<float> normals;  // Normalized analytical normals
	};
};

}
