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

// Forward declarations for CUDA kernels used in this header
template <typename T>
__global__ void set_first_channel_one_kernel(uint32_t n, T* dL_dout, uint32_t stride);

template <typename T>
__global__ void compute_surface_features_kernel(
	uint32_t batch_size,
	uint32_t D,
	const T* __restrict__ phi, uint32_t phi_stride,
	const float* __restrict__ dpos, uint32_t dpos_stride,
	T* __restrict__ out, uint32_t out_stride
);

template <typename T>
__global__ void set_phi_component_single_kernel(
	uint32_t n, uint32_t component_idx, T* dL_dout, uint32_t stride
);

template <typename T>
__global__ void accumulate_divergence_single_kernel(
	uint32_t n, uint32_t feature_idx, uint32_t component_idx, const float* __restrict__ dL_dpos, uint32_t dpos_stride, T* __restrict__ out, uint32_t out_stride
);

template <typename T>
__global__ void combine_rgb_outputs_kernel(
	uint32_t batch_size,
	float weight_surface, float weight_volume,
	const T* __restrict__ surface_rgb, uint32_t surf_stride,
	const T* __restrict__ volume_rgb, uint32_t vol_stride,
	T* __restrict__ output_rgb, uint32_t out_stride
);

template <typename T>
__global__ void compute_divergence_features_kernel(
	uint32_t batch_size,
	uint32_t D,
	const T* __restrict__ phi, uint32_t phi_stride,
	const float* __restrict__ dphi_dx, uint32_t dphi_dx_stride,
	const float* __restrict__ dphi_dy, uint32_t dphi_dy_stride,
	const float* __restrict__ dphi_dz, uint32_t dphi_dz_stride,
	T* __restrict__ out, uint32_t out_stride
);

template <typename T>
__global__ void set_placeholder_normals_kernel(
	uint32_t batch_size,
	T* __restrict__ normals, uint32_t normal_stride
);

template <typename T>
__global__ void set_single_component_seed_kernel(
	uint32_t batch_size, uint32_t component_idx,
	T* __restrict__ seed_data, uint32_t stride
);

template <typename T>
__global__ void accumulate_single_divergence_kernel(
	uint32_t batch_size, uint32_t feature_idx, uint32_t spatial_dim,
	const T* __restrict__ spatial_gradients, uint32_t grad_stride,
	T* __restrict__ divergence_features, uint32_t div_stride
);

template <typename T>
__global__ void set_spatial_components_kernel(
	uint32_t batch_size, 
	uint32_t spatial_dim, 
	uint32_t num_vectors,
	T* __restrict__ dL_dout, uint32_t stride
);

template <typename T>
__global__ void compute_divergence_accumulate_kernel(
	uint32_t batch_size,
	uint32_t num_features,
	uint32_t spatial_dim,
	const float* __restrict__ spatial_gradients, uint32_t spatial_stride,
	float* __restrict__ divergence_features, uint32_t divergence_stride
);

// Eikonal helper
template <typename T>
__global__ void compute_eikonal_v_kernel(
	uint32_t batch_size,
	uint32_t input_width,
	const float* __restrict__ g, uint32_t g_stride,
	float* __restrict__ v, uint32_t v_stride
);

// Scale array kernel
static __global__ void nerf_scale_array_kernel(uint32_t n, float s, float* x);


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

template <typename T>
class NerfNetwork : public Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = minimum_alignment(rgb_network);
		m_dir_encoding.reset(create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		// Parse radiance head mode if provided in network config (defaults to baseline)
		if (rgb_network.contains("radiance_head_mode") && rgb_network["radiance_head_mode"].is_string()) {
			m_radiance_head_mode = rgb_network["radiance_head_mode"].get<std::string>();
		}

		// Determine feature width based on radiance head mode
		m_feature_width = 0;
		if (m_radiance_head_mode == "baseline") {
			m_feature_width = 16; // default density features
		} else if (m_radiance_head_mode == "surface") {
			m_feature_width = 15; // D = 15 from Phi dot n_hat
		} else if (m_radiance_head_mode == "volume") {
			m_feature_width = 15; // D = 15 divergence
		} else if (m_radiance_head_mode == "hybrid" || m_radiance_head_mode == "dual_merge") {
			m_feature_width = 30; // 15 surface + 15 volume concatenated
		} else if (m_radiance_head_mode == "dual_separate") {
			m_feature_width = 15; // Each separate MLP gets 15D features (like baseline gets 16D)
		} else {
			m_feature_width = 16;
		}

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			// Expand density output to carry vector potential Phi (15x3) in non-baseline modes.
			// Channel 0 remains the raw density channel as before; channels 1..45 reserved for Phi.
			// For dual_separate, we need 1 + 45 = 46 channels total
			local_density_network_config["n_output_dims"] = (m_radiance_head_mode == "baseline") ? 16 : 46;
		}
		m_density_network.reset(create_network<T>(local_density_network_config));

		// RGB input width is [features (mode-dependent)] + [dir encoding]
		// All modes use the same structure: feature_width + direction encoding
		m_rgb_network_input_width = next_multiple(m_feature_width + m_dir_encoding->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		// Standard RGB network always outputs 3D RGB
		local_rgb_network_config["n_output_dims"] = 3;

		m_rgb_network.reset(create_network<T>(local_rgb_network_config));

		// Enable per-sample loss via json flag loss_mode=="surface"
		if (rgb_network.contains("loss_mode") && rgb_network["loss_mode"].is_string()) {
			m_loss_mode = rgb_network["loss_mode"].get<std::string>();
		}

		// Optional second RGB head for dual_separate mode (constructed now, used later)
		if (m_radiance_head_mode == "dual_separate") {
			// Create two identical RGB networks, same structure as main network
			// Each gets 15D features + dir encoding, just like baseline gets 16D features + dir
			
			json local_rgb_network_config2 = rgb_network;
			local_rgb_network_config2["n_input_dims"] = m_rgb_network_input_width;  // Same as main network
			local_rgb_network_config2["n_output_dims"] = 3;  // Both output 3D RGB
			m_rgb_network_surface.reset(create_network<T>(local_rgb_network_config2));
			m_rgb_network_volume.reset(create_network<T>(local_rgb_network_config2));
		}

		m_density_model = std::make_shared<NetworkWithInputEncoding<T>>(m_pos_encoding, m_density_network);
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();
		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

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
	
	void set_sdf_eikonal_lambda(float lambda) {
		m_sdf_eikonal_lambda = lambda;
	}
	
	void set_cumsum_regularization_enabled(bool enabled) {
		m_cumsum_regularization_enabled = enabled;
	}

	const std::string& radiance_head_mode() const {
		return m_radiance_head_mode;
	}
	
	void set_radiance_head_mode(const std::string& mode) {
		m_radiance_head_mode = mode;
	}
	
	// Render with custom surface/volume weights (for dual_separate mode)
	void render_with_dual_weights(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output,
		float weight_surface, 
		float weight_volume,
		bool use_inference_params = true
	) {
		if (m_radiance_head_mode != "dual_separate" || !output) {
			// Fall back to normal forward pass
			auto ctx = forward(stream, input, output, use_inference_params, false);
			return;
		}
		
		uint32_t batch_size = input.n();
		
		// Do forward pass to get surface and volume outputs
		auto forward_ctx = forward(stream, input, output, use_inference_params, false);
		auto& forward = dynamic_cast<const ForwardContext&>(*forward_ctx);
		
		// Re-combine with custom weights
		if (forward.surface_rgb_output.n_elements() > 0 && forward.volume_rgb_output.n_elements() > 0) {
			combine_rgb_outputs_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
				batch_size,
				weight_surface, weight_volume,
				forward.surface_rgb_output.data(),
				forward.surface_rgb_output.layout() == AoS ? forward.surface_rgb_output.stride() : 1,
				forward.volume_rgb_output.data(),
				forward.volume_rgb_output.layout() == AoS ? forward.volume_rgb_output.stride() : 1,
				output->data(),
				output->layout() == AoS ? output->stride() : 1
			);
		}
	}

	// Access cumsum regularization features from trainer context
	T* get_surface_features_from_context(const Context* trainer_ctx) const {
		// Try to access the network's ForwardContext through the trainer context
		// The trainer context should contain a model_ctx which is our ForwardContext
		auto* forward_ctx = dynamic_cast<const ForwardContext*>(trainer_ctx);
		return forward_ctx && forward_ctx->surface_features.n_elements() > 0 ? 
			forward_ctx->surface_features.data() : nullptr;
	}

	T* get_volume_features_from_context(const Context* trainer_ctx) const {
		auto* forward_ctx = dynamic_cast<const ForwardContext*>(trainer_ctx);
		return forward_ctx && forward_ctx->volume_features.n_elements() > 0 ? 
			forward_ctx->volume_features.data() : nullptr;
	}

	T* get_sample_densities_from_context(const Context* trainer_ctx) const {
		auto* forward_ctx = dynamic_cast<const ForwardContext*>(trainer_ctx);
		return forward_ctx && forward_ctx->sample_densities.n_elements() > 0 ? 
			forward_ctx->sample_densities.data() : nullptr;
	}

	// Direct access to global feature storage for training kernel
	T* get_global_surface_features() const { 
		return m_global_surface_features.size() > 0 ? m_global_surface_features.data() : nullptr; 
	}
	T* get_global_volume_features() const { 
		return m_global_volume_features.size() > 0 ? m_global_volume_features.data() : nullptr; 
	}
	T* get_global_sample_densities() const { 
		return m_global_sample_densities.size() > 0 ? m_global_sample_densities.data() : nullptr; 
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

		if (m_radiance_head_mode == "baseline") {
			// For baseline, write density MLP output directly into the feature slice to avoid extra copies
			auto feat_out = forward->rgb_network_input.slice_rows(0, m_feature_width);
			forward->density_network_output = GPUMatrixDynamic<T>{feat_out.data(), feat_out.m(), feat_out.n(), feat_out.layout()};
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);
		} else {
			// Non-baseline: run density MLP into a temporary buffer, then compute features below
			forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, m_dir_encoding->preferred_output_layout()};
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);
			// Compute features depending on mode
			auto feat_out = forward->rgb_network_input.slice_rows(0, m_feature_width);
			if (m_radiance_head_mode == "surface" || m_radiance_head_mode == "dual_separate") {
				// density output layout: [sigma | Phi_flattened (3*D)]
				const uint32_t D = 15;
				auto phi_out = forward->density_network_output.slice_rows(1, 1 + 3 * D);

				// Backprop to get d sigma / d x,y,z per sample
				GPUMatrixDynamic<T> dL_ddensity_out{forward->density_network_output.m(), batch_size, stream, forward->density_network_output.layout()};
				CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_out.data(), 0, dL_ddensity_out.n_bytes(), stream));
				// set grad on sigma channel to 1
				set_first_channel_one_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
					batch_size,
					dL_ddensity_out.data(),
					dL_ddensity_out.layout() == AoS ? dL_ddensity_out.stride() : 1
				);

				// dL/d encoded pos
				GPUMatrixDynamic<T> dL_ddensity_in{forward->density_network_input.m(), batch_size, stream, forward->density_network_input.layout()};
				m_density_network->backward(
					stream,
					*forward->density_network_ctx,
					forward->density_network_input,
					forward->density_network_output,
					dL_ddensity_out,
					&dL_ddensity_in,
					use_inference_params,
					GradientMode::Overwrite
				);

				// dL/d raw pos (xyz)
				GPUMatrixDynamic<float> dL_dpos_input;
				m_pos_encoding->backward(
					stream,
					*forward->pos_encoding_ctx,
					input.slice_rows(0, m_pos_encoding->input_width()),
					forward->density_network_input,
					dL_ddensity_in,
					&dL_dpos_input,
					use_inference_params,
					GradientMode::Overwrite
				);

				// Store analytic gradient w.r.t. positions (used as normals when in SDF mode)
				forward->dSDF_dPos = dL_dpos_input.slice_rows(0, m_pos_encoding->input_width());

				// Compute features = ReLU(-dot(Phi_i, n_hat))
				compute_surface_features_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
					batch_size,
					D,
					phi_out.data(),
					phi_out.layout() == AoS ? phi_out.stride() : 1,
					dL_dpos_input.data(),
					dL_dpos_input.layout() == AoS ? dL_dpos_input.stride() : 1,
					feat_out.data(),
					feat_out.layout() == AoS ? feat_out.stride() : 1
				);
			} else if (m_radiance_head_mode == "volume" || m_radiance_head_mode == "hybrid" || m_radiance_head_mode == "dual_merge" || m_radiance_head_mode == "dual_separate") {
				const uint32_t D = 15;
				auto phi_out = forward->density_network_output.slice_rows(1, 1 + 3 * D);
				// zero features
				CUDA_CHECK_THROW(cudaMemsetAsync(feat_out.data(), 0, feat_out.n_bytes(), stream));

				// If hybrid/dual, first fill surface features into first D
				if (m_radiance_head_mode == "hybrid" || m_radiance_head_mode == "dual_merge" || m_radiance_head_mode == "dual_separate") {
					// Compute normals as in surface
					GPUMatrixDynamic<T> dL_ddensity_out{forward->density_network_output.m(), batch_size, stream, forward->density_network_output.layout()};
					CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_out.data(), 0, dL_ddensity_out.n_bytes(), stream));
					set_first_channel_one_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
						batch_size,
						dL_ddensity_out.data(),
						dL_ddensity_out.layout() == AoS ? dL_ddensity_out.stride() : 1
					);
					GPUMatrixDynamic<T> dL_ddensity_in{forward->density_network_input.m(), batch_size, stream, forward->density_network_input.layout()};
					m_density_network->backward(stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, dL_ddensity_out, &dL_ddensity_in, use_inference_params, GradientMode::Overwrite);
					GPUMatrixDynamic<float> dL_dpos_input;
					m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, input.slice_rows(0, m_pos_encoding->input_width()), forward->density_network_input, dL_ddensity_in, &dL_dpos_input, use_inference_params, GradientMode::Overwrite);
					forward->dSDF_dPos = dL_dpos_input.slice_rows(0, m_pos_encoding->input_width());
					auto surf_slice = feat_out.slice_rows(0, D);
					compute_surface_features_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
						batch_size,
						D,
						phi_out.data(),
						phi_out.layout() == AoS ? phi_out.stride() : 1,
						dL_dpos_input.data(),
						dL_dpos_input.layout() == AoS ? dL_dpos_input.stride() : 1,
						surf_slice.data(),
						surf_slice.layout() == AoS ? surf_slice.stride() : 1
					);
					
					// Store surface features for cumsum regularization if enabled
					if (m_cumsum_regularization_enabled && prepare_input_gradients) {
						// Store in both ForwardContext and global memory for training access
						forward->surface_features = GPUMatrixDynamic<T>{D, batch_size, stream, surf_slice.layout()};
						CUDA_CHECK_THROW(cudaMemcpyAsync(
							forward->surface_features.data(),
							surf_slice.data(),
							surf_slice.n_bytes(),
							cudaMemcpyDeviceToDevice,
							stream
						));
						
						// Also store in global memory for direct training kernel access
						m_global_surface_features.resize(D * batch_size);
						CUDA_CHECK_THROW(cudaMemcpyAsync(
							m_global_surface_features.data(),
							surf_slice.data(),
							surf_slice.n_bytes(),
							cudaMemcpyDeviceToDevice,
							stream
						));
					}
				}

				// Volume divergence features into target slice
				GPUMatrixDynamic<T> volm_slice = (m_radiance_head_mode == "volume") ? GPUMatrixDynamic<T>{feat_out.data(), feat_out.m(), feat_out.n(), feat_out.layout()} : feat_out.slice_rows(D, D);
				// Temporary grads buffers
				GPUMatrixDynamic<T> dL_ddensity_out{forward->density_network_output.m(), batch_size, stream, forward->density_network_output.layout()};
				GPUMatrixDynamic<T> dL_ddensity_in{forward->density_network_input.m(), batch_size, stream, forward->density_network_input.layout()};
				GPUMatrixDynamic<float> dL_dpos_input;

				for (uint32_t i = 0; i < D; ++i) {
					for (uint32_t c = 0; c < 3; ++c) {
						// zero grads
						CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_out.data(), 0, dL_ddensity_out.n_bytes(), stream));
						// set unit grad on row corresponding to Phi component (1 + 3*i + c)
						set_phi_component_single_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
							batch_size,
							1 + 3 * i + c,
							dL_ddensity_out.data(),
							dL_ddensity_out.layout() == AoS ? dL_ddensity_out.stride() : 1
						);
						// Backprop to encoded pos
						m_density_network->backward(stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, dL_ddensity_out, &dL_ddensity_in, use_inference_params, GradientMode::Overwrite);
						// Backprop to raw pos
						m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, input.slice_rows(0, m_pos_encoding->input_width()), forward->density_network_input, dL_ddensity_in, &dL_dpos_input, use_inference_params, GradientMode::Overwrite);
						// Accumulate partial derivative for component c into feature i
						accumulate_divergence_single_kernel<T><<<n_blocks_linear(batch_size), N_THREADS_LINEAR, 0, stream>>>(
							batch_size,
							i,
							c,
							dL_dpos_input.data(),
							dL_dpos_input.layout() == AoS ? dL_dpos_input.stride() : 1,
							volm_slice.data(),
							volm_slice.layout() == AoS ? volm_slice.stride() : 1
						);
					}
				}
				
				// Store volume features for cumsum regularization if enabled
				if (m_cumsum_regularization_enabled && prepare_input_gradients && 
				    (m_radiance_head_mode == "dual_merge" || m_radiance_head_mode == "dual_separate")) {
					// Store in both ForwardContext and global memory for training access
					forward->volume_features = GPUMatrixDynamic<T>{D, batch_size, stream, volm_slice.layout()};
					CUDA_CHECK_THROW(cudaMemcpyAsync(
						forward->volume_features.data(),
						volm_slice.data(),
						volm_slice.n_bytes(),
						cudaMemcpyDeviceToDevice,
						stream
					));
					
					// Also store in global memory for direct training kernel access
					m_global_volume_features.resize(D * batch_size);
					CUDA_CHECK_THROW(cudaMemcpyAsync(
						m_global_volume_features.data(),
						volm_slice.data(),
						volm_slice.n_bytes(),
						cudaMemcpyDeviceToDevice,
						stream
					));
				}
			} else {
				// Unknown mode
				CUDA_CHECK_THROW(cudaMemsetAsync(feat_out.data(), 0, feat_out.n_bytes(), stream));
			}
		}

		// Direction encoding (skip for dual_separate as it handles this internally)
		if (m_radiance_head_mode != "dual_separate") {
			auto dir_out = forward->rgb_network_input.slice_rows(m_feature_width, m_dir_encoding->padded_output_width());
			forward->dir_encoding_ctx = m_dir_encoding->forward(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				&dir_out,
				use_inference_params,
				prepare_input_gradients
			);
		}



 else {
			// Standard behavior for all other modes (use main network)
			if (output) {
				forward->rgb_network_output = GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
			}
			forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);
		}

		if (output) {
			linear_kernel(extract_density<T>, 0, stream,
				batch_size, m_dir_encoding->preferred_output_layout() == AoS ? forward->density_network_output.stride() : 1, padded_output_width(), forward->density_network_output.data(), output->data()+3
			);
			
			// Store density values for cumsum regularization if enabled
			if (m_cumsum_regularization_enabled && prepare_input_gradients && 
			    (m_radiance_head_mode == "dual_merge" || m_radiance_head_mode == "dual_separate")) {
				// Store in both ForwardContext and global memory for training access
				forward->sample_densities = GPUMatrixDynamic<T>{1, batch_size, stream, forward->density_network_output.layout()};
				CUDA_CHECK_THROW(cudaMemcpyAsync(
					forward->sample_densities.data(),
					output->data() + 3,
					batch_size * sizeof(T),
					cudaMemcpyDeviceToDevice,
					stream
				));
				
				// Also store in global memory for direct training kernel access
				m_global_sample_densities.resize(batch_size);
				CUDA_CHECK_THROW(cudaMemcpyAsync(
					m_global_sample_densities.data(),
					output->data() + 3,
					batch_size * sizeof(T),
					cudaMemcpyDeviceToDevice,
					stream
				));
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

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		// Handle dual_separate mode differently
		if (m_radiance_head_mode == "dual_separate") {
			// For dual_separate, we only backprop through the surface network for now
			// This simplifies the gradient flow while we debug the architecture
			
			GPUMatrix<T> dL_drgb{m_rgb_network_surface->padded_output_width(), batch_size, stream};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
			linear_kernel(extract_rgb<T>, 0, stream,
				batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
			);

			// Use surface RGB output for backward pass (channels 0,1,2)
			const GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network_surface->padded_output_width(), batch_size, output.layout()};
			GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
			
			// Only backprop through surface network for now
			if (forward.surface_rgb_ctx) {
				m_rgb_network_surface->backward(stream, *forward.surface_rgb_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
			}
			
			// Continue with standard dir encoding and density backward pass
			// Backprop through dir encoding if it is trainable or if we need input gradients
			if (m_dir_encoding->n_params() > 0 || dL_dinput) {
				GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
				GPUMatrixDynamic<float> dL_ddir_encoding_input;
				if (dL_dinput) {
					dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
				}

				m_dir_encoding->backward(
					stream,
					*forward.dir_encoding_ctx,
					input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
					forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
					dL_ddir_encoding_output,
					dL_dinput ? &dL_ddir_encoding_input : nullptr,
					use_inference_params,
					param_gradients_mode
				);
			}

			GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
			linear_kernel(add_density_gradient<T>, 0, stream,
				batch_size,
				dL_doutput.m(),
				dL_doutput.data(),
				dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
				dL_ddensity_network_output.data()
			);

			GPUMatrixDynamic<T> dL_ddensity_network_input;
			if (m_pos_encoding->n_params() > 0 || dL_dinput) {
				dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			}

			m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

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
			return; // Early return for dual_separate mode
		}

		// Standard backward pass for other modes
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
			GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);

		GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

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

	void density(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_density_model->set_jit_fusion(this->jit_fusion());
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
		
		// Set parameters for dual_separate networks if they exist
		if (m_radiance_head_mode == "dual_separate" && m_rgb_network_surface && m_rgb_network_volume) {
			m_rgb_network_surface->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_rgb_network_surface->n_params();
			
			m_rgb_network_volume->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_rgb_network_volume->n_params();
		}

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
		
		// Initialize parameters for dual_separate networks if they exist
		if (m_radiance_head_mode == "dual_separate" && m_rgb_network_surface && m_rgb_network_volume) {
			m_rgb_network_surface->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_rgb_network_surface->n_params();
			
			m_rgb_network_volume->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_rgb_network_volume->n_params();
		}

		m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		size_t total = m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
		
		// Add parameters for dual_separate networks if they exist
		if (m_radiance_head_mode == "dual_separate" && m_rgb_network_surface && m_rgb_network_volume) {
			total += m_rgb_network_surface->n_params() + m_rgb_network_volume->n_params();
		}
		
		return total;
	}

	uint32_t padded_output_width() const override {
		// For dual_separate, each separate network should have the same output as baseline (4 channels: RGB+density)
		// But the final output needs space for volume RGB in channels 12,13,14
		if (m_radiance_head_mode == "dual_separate") {
			return 16; // Enough space for surface RGB (0,1,2), density (3), and volume RGB (12,13,14)
		}
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
		// For dual_separate mode, disable JIT compilation since it uses separate execution path
		if (m_radiance_head_mode == "dual_separate") {
			return "";  // Empty device function disables JIT compilation
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
		
		// Include dual networks if they exist
		if (m_radiance_head_mode == "dual_separate" && m_rgb_network_surface && m_rgb_network_volume) {
			m_rgb_network_surface->convert_params_to_jit_layout(stream, use_inference_params);
			m_rgb_network_volume->convert_params_to_jit_layout(stream, use_inference_params);
		}
		
		m_pos_encoding->convert_params_to_jit_layout(stream, use_inference_params);
		m_dir_encoding->convert_params_to_jit_layout(stream, use_inference_params);
	}

	void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		m_density_network->convert_params_from_jit_layout(stream, use_inference_params);
		m_rgb_network->convert_params_from_jit_layout(stream, use_inference_params);
		
		// Include dual networks if they exist
		if (m_radiance_head_mode == "dual_separate" && m_rgb_network_surface && m_rgb_network_volume) {
			m_rgb_network_surface->convert_params_from_jit_layout(stream, use_inference_params);
			m_rgb_network_volume->convert_params_from_jit_layout(stream, use_inference_params);
		}
		
		m_pos_encoding->convert_params_from_jit_layout(stream, use_inference_params);
		m_dir_encoding->convert_params_from_jit_layout(stream, use_inference_params);
	}

private:
	std::shared_ptr<Network<T>> m_density_network;
	std::shared_ptr<Network<T>> m_rgb_network;
	std::shared_ptr<Network<T>> m_rgb_network_surface; // Added for dual_separate mode
	std::shared_ptr<Network<T>> m_rgb_network_volume; // Added for dual_separate mode
	std::shared_ptr<Encoding<T>> m_pos_encoding;
	std::shared_ptr<Encoding<T>> m_dir_encoding;

	// Aggregates m_pos_encoding and m_density_network
	std::shared_ptr<NetworkWithInputEncoding<T>> m_density_model;

	uint32_t m_rgb_network_input_width;
	uint32_t m_dual_rgb_input_width = 0; // For dual_separate mode
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	uint32_t m_feature_width = 16;
	std::string m_radiance_head_mode = std::string("baseline");
	std::string m_loss_mode = std::string("");
	
	// SDF mode parameters
	float m_sdf_eikonal_lambda = 0.0f;
	bool m_cumsum_regularization_enabled = false;
	
	// Global storage for cumsum regularization features (accessible during training)
	mutable GPUMemory<T> m_global_surface_features;
	mutable GPUMemory<T> m_global_volume_features; 
	mutable GPUMemory<T> m_global_sample_densities;

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

		// Analytic spatial gradient of the first density-head channel w.r.t. input position (∇SDF in SDF mode)
		GPUMatrixDynamic<float> dSDF_dPos;
		
		// Cumsum regularization feature storage (only allocated when needed)
		GPUMatrixDynamic<T> surface_features;  // 15D surface features per sample
		GPUMatrixDynamic<T> volume_features;   // 15D volume features per sample
		GPUMatrixDynamic<T> sample_densities;  // Density per sample

		// Contexts for dual networks
		std::unique_ptr<Context> surface_rgb_ctx;
		std::unique_ptr<Context> volume_rgb_ctx;
		
		// Separate RGB outputs for dual rendering modes
		GPUMatrixDynamic<T> surface_rgb_output;
		GPUMatrixDynamic<T> volume_rgb_output;
	};
};

// Kernel implementations

// Helper: compute v = dL/d(∇SDF) for eikonal: v = 2*(||g|| - 1) * g / max(||g||, eps)
template <typename T>
__global__ void compute_eikonal_v_kernel(
	uint32_t batch_size,
	uint32_t input_width,
	const float* __restrict__ g, uint32_t g_stride,
	float* __restrict__ v, uint32_t v_stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;
	float gx = g[0 * g_stride + b];
	float gy = input_width > 1 ? g[1 * g_stride + b] : 0.f;
	float gz = input_width > 2 ? g[2 * g_stride + b] : 0.f;
	float norm = sqrtf(gx*gx + gy*gy + gz*gz) + 1e-8f;
	float factor = 2.0f * (norm - 1.0f) / norm;
	if (input_width > 0) v[0 * v_stride + b] = factor * gx;
	if (input_width > 1) v[1 * v_stride + b] = factor * gy;
	if (input_width > 2) v[2 * v_stride + b] = factor * gz;
	for (uint32_t r = 3; r < input_width; ++r) {
		v[r * v_stride + b] = 0.f;
	}
}

template <typename T>
__global__ void set_first_channel_one_kernel(
	uint32_t n, T* dL_dout, uint32_t stride
) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;
	dL_dout[i * stride + 0] = (T)1;
}

template <typename T>
__global__ void compute_surface_features_kernel(
	uint32_t batch_size,
	uint32_t D,
	const T* __restrict__ phi, uint32_t phi_stride,
	const float* __restrict__ dpos, uint32_t dpos_stride,
	T* __restrict__ out, uint32_t out_stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;

	float nx = dpos[b * dpos_stride + 0];
	float ny = dpos[b * dpos_stride + 1];
	float nz = dpos[b * dpos_stride + 2];
	if (!kUseSdf) {
		float norm = sqrtf(nx*nx + ny*ny + nz*nz) + 1e-8f;
		nx /= norm; ny /= norm; nz /= norm;
	}

	for (uint32_t i = 0; i < D; ++i) {
		T px = phi[(i*3+0) * phi_stride + b];
		T py = phi[(i*3+1) * phi_stride + b];
		T pz = phi[(i*3+2) * phi_stride + b];
		float s = -(float(px)*nx + float(py)*ny + float(pz)*nz);
		out[i * out_stride + b] = (T)(s > 0.f ? s : 0.f);
	}
}

template <typename T>
__global__ void set_phi_component_single_kernel(
	uint32_t n, uint32_t component_idx, T* dL_dout, uint32_t stride
) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;
	dL_dout[i * stride + component_idx] = (T)1;
}

template <typename T>
__global__ void set_single_component_seed_kernel(
	uint32_t batch_size, uint32_t component_idx,
	T* __restrict__ seed_data, uint32_t stride
) {
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= batch_size) return;
	
	// Set unit gradient for this specific component
	seed_data[idx * stride + component_idx] = (T)1.0f;
}

template <typename T>
__global__ void accumulate_single_divergence_kernel(
	uint32_t batch_size, uint32_t feature_idx, uint32_t spatial_dim,
	const T* __restrict__ spatial_gradients, uint32_t grad_stride,
	T* __restrict__ divergence_features, uint32_t div_stride
) {
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= batch_size) return;
	
	// Extract the spatial gradient for this dimension
	T spatial_grad = spatial_gradients[idx * grad_stride + spatial_dim];
	
	// Accumulate to this feature's divergence: ∇·Φᵢ += ∂Φᵢ_spatial_dim/∂spatial_dim
	divergence_features[feature_idx * div_stride + idx] += spatial_grad;
}



template <typename T>
__global__ void accumulate_divergence_single_kernel(
	uint32_t n, uint32_t feature_idx, uint32_t component_idx, const float* __restrict__ dL_dpos, uint32_t dpos_stride, T* __restrict__ out, uint32_t out_stride
) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;
	out[i * out_stride + feature_idx] += dL_dpos[i * dpos_stride + component_idx];
}

template <typename T>
__global__ void combine_rgb_outputs_kernel(
	uint32_t batch_size,
	float weight_surface, float weight_volume,
	const T* __restrict__ surface_rgb, uint32_t surf_stride,
	const T* __restrict__ volume_rgb, uint32_t vol_stride,
	T* __restrict__ output_rgb, uint32_t out_stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;

	T w_surf = (T)weight_surface;
	T w_vol = (T)weight_volume;
	
	output_rgb[b * out_stride + 0] = w_surf * surface_rgb[b * surf_stride + 0] + w_vol * volume_rgb[b * vol_stride + 0];
	output_rgb[b * out_stride + 1] = w_surf * surface_rgb[b * surf_stride + 1] + w_vol * volume_rgb[b * vol_stride + 1];
	output_rgb[b * out_stride + 2] = w_surf * surface_rgb[b * surf_stride + 2] + w_vol * volume_rgb[b * vol_stride + 2];
}

template <typename T>
__global__ void set_placeholder_normals_kernel(
	uint32_t batch_size,
	T* __restrict__ normals, uint32_t normal_stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;

	// Set simple placeholder normals (unit vector in X direction)
	// This eliminates the expensive double forward pass for analytical gradients
	normals[0 * normal_stride + b] = T(1.0f);
	normals[1 * normal_stride + b] = T(0.0f);
	normals[2 * normal_stride + b] = T(0.0f);
}

template <typename T>
__global__ void set_spatial_components_kernel(
	uint32_t batch_size, 
	uint32_t spatial_dim, 
	uint32_t num_vectors,
	T* __restrict__ dL_dout, uint32_t stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;

	// Set gradient seeds for all components of the specified spatial dimension
	// For spatial_dim=0: set gradients for components 1, 4, 7, 10, ... (x components)
	// For spatial_dim=1: set gradients for components 2, 5, 8, 11, ... (y components)  
	// For spatial_dim=2: set gradients for components 3, 6, 9, 12, ... (z components)
	for (uint32_t i = 0; i < num_vectors; ++i) {
		uint32_t component_idx = 1 + i * 3 + spatial_dim; // Start at 1 (skip density), then every 3rd component
		dL_dout[b * stride + component_idx] = T(1.0f);
	}
}

template <typename T>
__global__ void compute_divergence_accumulate_kernel(
	uint32_t batch_size,
	uint32_t num_features,
	uint32_t spatial_dim,
	const float* __restrict__ spatial_gradients, uint32_t spatial_stride,
	float* __restrict__ divergence_features, uint32_t divergence_stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;

	// For each of the 15 vector fields, accumulate ∂Φᵢ_spatial_dim/∂spatial_dim into divergence
	// The spatial_gradients contain gradients of all 15 vector field components w.r.t. the spatial dimension
	for (uint32_t i = 0; i < num_features; ++i) {
		// The gradients are organized by component: grad[0] = ∂Φ₀ₓ/∂spatial_dim, grad[1] = ∂Φ₁ₓ/∂spatial_dim, etc.
		float gradient_value = spatial_gradients[i * spatial_stride + b];
		divergence_features[i * divergence_stride + b] += gradient_value;
	}
}

template <typename T>
__global__ void compute_divergence_features_kernel(
	uint32_t batch_size,
	uint32_t D,
	const T* __restrict__ phi, uint32_t phi_stride,
	const float* __restrict__ dphi_dx, uint32_t dphi_dx_stride,
	const float* __restrict__ dphi_dy, uint32_t dphi_dy_stride, 
	const float* __restrict__ dphi_dz, uint32_t dphi_dz_stride,
	T* __restrict__ out, uint32_t out_stride
) {
	uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= batch_size) return;

	// For each of the 15 feature dimensions, compute true divergence of Φ_i
	// Φ is organized as [15, 3]: 15 3D vector fields
	// True divergence of Φ_i = dΦ_i_x/dx + dΦ_i_y/dy + dΦ_i_z/dz
	
	for (uint32_t i = 0; i < D; ++i) {
		// For feature i, get gradients of each component
		float dphix_dx = dphi_dx[(i*3 + 0) * dphi_dx_stride + b]; // d(Φ_i_x)/dx
		float dphiy_dy = dphi_dy[(i*3 + 1) * dphi_dy_stride + b]; // d(Φ_i_y)/dy  
		float dphiz_dz = dphi_dz[(i*3 + 2) * dphi_dz_stride + b]; // d(Φ_i_z)/dz
		
		// True divergence: sum of partial derivatives
		float divergence = dphix_dx + dphiy_dy + dphiz_dz;
		
		out[i * out_stride + b] = (T)divergence;
	}
}

template <typename T>
__global__ void compute_cumsum_regularization_kernel(
	uint32_t n_rays,
	uint32_t n_samples_per_ray,
	uint32_t feature_dim,
	const T* __restrict__ surface_features,   // [feature_dim, n_rays * n_samples_per_ray]
	const T* __restrict__ volume_features,    // [feature_dim, n_rays * n_samples_per_ray] 
	const T* __restrict__ sample_densities,   // [1, n_rays * n_samples_per_ray]
	const float* __restrict__ forward_weights, // [n_rays * n_samples_per_ray]
	float* __restrict__ reg_loss_output,      // [n_rays]
	uint32_t surface_stride,
	uint32_t volume_stride,
	uint32_t density_stride
) {
	uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_idx >= n_rays) return;
	
	float reg_loss = 0.0f;
	
	// Initialize accumulator for density-weighted volume features
	T vol_feat_accum[15] = {0}; // Assuming feature_dim = 15
	
	// Iterate backward through samples (farthest to closest)
	for (int sample_idx = n_samples_per_ray - 1; sample_idx >= 0; sample_idx--) {
		uint32_t global_idx = ray_idx * n_samples_per_ray + sample_idx;
		
		// Get forward weight for this sample
		float weight = forward_weights[global_idx];
		if (weight < 1e-8f) continue; // Skip samples with negligible weight
		
		// Get surface features (anchor)
		T surf_feat[15];
		for (uint32_t d = 0; d < feature_dim; d++) {
			surf_feat[d] = surface_features[d * surface_stride + global_idx];
		}
		
		// Normalize surface features
		float surf_norm = 0.0f;
		for (uint32_t d = 0; d < feature_dim; d++) {
			surf_norm += float(surf_feat[d]) * float(surf_feat[d]);
		}
		surf_norm = sqrtf(surf_norm + 1e-8f);
		
		// Normalize volume feature accumulator
		float vol_norm = 0.0f;
		for (uint32_t d = 0; d < feature_dim; d++) {
			vol_norm += float(vol_feat_accum[d]) * float(vol_feat_accum[d]);
		}
		vol_norm = sqrtf(vol_norm + 1e-8f);
		
		// Compute normalized difference
		float sample_loss = 0.0f;
		if (surf_norm > 1e-8f && vol_norm > 1e-8f) {
			for (uint32_t d = 0; d < feature_dim; d++) {
				float surf_normalized = float(surf_feat[d]) / surf_norm;
				float vol_normalized = float(vol_feat_accum[d]) / vol_norm;
				float diff = surf_normalized - vol_normalized;
				sample_loss += diff * diff;
			}
			sample_loss /= float(feature_dim); // Mean over features
		}
		
		// Accumulate weighted loss
		reg_loss += weight * sample_loss;
		
		// Update accumulator with density-weighted volume features
		T density = sample_densities[global_idx];
		for (uint32_t d = 0; d < feature_dim; d++) {
			T vol_feat = volume_features[d * volume_stride + global_idx];
			vol_feat_accum[d] += density * vol_feat;
		}
	}
	
	reg_loss_output[ray_idx] = reg_loss;
}

static __global__ void nerf_scale_array_kernel(uint32_t n, float s, float* x) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;
	x[i] *= s;
}

}
