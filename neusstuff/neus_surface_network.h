/** @file   nerf_network.h
 *  @author Yiming Wang <w752531540@gmail.com>
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/common_operation.cuh>

#include <neural-graphics-primitives/transform_network.h>

#include <json/json.hpp>

NGP_NAMESPACE_BEGIN

// CUDA kernel function declarations for surface and volume configurations
template <typename T>
__global__ void compute_surface_features_kernel(
    uint32_t batch_size,
    const T* phi_input,        // 45D Φ input (15x3)
    const T* normal_vectors,   // 3D normal vectors
    T* processed_features,     // 16D output (15D surface + 1D SDF)
    const T* sdf_input        // 1D SDF input
);

template <typename T>
__global__ void compute_divergence_features_kernel(
    uint32_t batch_size,
    const T* phi_input,        // 45D Φ input (15x3)
    const T* normal_vectors,   // 3D normal vectors (for reference)
    T* processed_features,     // 16D output (15D divergence + 1D SDF)
    const T* sdf_input        // 1D SDF input
);

template <typename T>
__global__ void copy_processed_features(
    uint32_t n_elements,
    const T* input, T* output
);

template <typename T>
__global__ void add_rgb_two_heads(const uint32_t n_elements,
		const T* __restrict__ surf_out,
		const T* __restrict__ vol_out,
		T* __restrict__ rgb_out,
		uint32_t out_stride);

template <typename T>
__global__ void write_dual_head_rgb_rows(const uint32_t n_elements,
    const T* __restrict__ surf_out, uint32_t surf_stride,
    const T* __restrict__ vol_out, uint32_t vol_stride,
    T* __restrict__ out, uint32_t out_stride);



#define NERF_DEBUG_BACKWARD 0
#define GEOMETRY_INIT 1
#define global_delta 1
#define viewdir_backward 0


using namespace Eigen;

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network, const std::string& configuration = "baseline") : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims}, m_configuration{configuration} {
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		#if GEOMETRY_INIT
			m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 2u : 2u));
			m_density_network_input_width = tcnn::next_multiple(m_n_pos_dims + m_pos_encoding->output_width(), rgb_alignment); // 40
		#else
			m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
			m_density_network_input_width = m_pos_encoding->padded_output_width();
		#endif
		
		// DEBUG: Print constructor values
		printf("=== NeuS2 CONSTRUCTOR DEBUG ===\n");
		printf("Configuration: %s\n", m_configuration.c_str());
		printf("m_n_pos_dims: %d\n", m_n_pos_dims);
		printf("m_n_dir_dims: %d\n", m_n_dir_dims);
		printf("m_density_network_input_width: %d\n", m_density_network_input_width);
		printf("rgb_alignment: %d\n", rgb_alignment);
		printf("pos_encoding->padded_output_width(): %d\n", m_pos_encoding->padded_output_width());
		
		local_density_network_config["n_input_dims"] = m_density_network_input_width;
		if (!density_network.contains("n_output_dims")) {
			if (m_configuration == "surface" || m_configuration == "volume" || m_configuration == "hybrid" || m_configuration == "dual" || m_configuration == "dual_merge") {
				// Support for surface/volume/hybrid configuration: output 46D instead of 16D
				// 1D for SDF + 15x3D for Spatially-Vectored Potential Φ
				local_density_network_config["n_output_dims"] = 46;
				printf("Density network output dims: 46 (surface/volume/hybrid mode)\n");
			} else {
				// Baseline configuration: output 16D
			local_density_network_config["n_output_dims"] = 16;
				printf("Density network output dims: 16 (baseline mode)\n");
			}
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		// density(feature), xyz, normal, dir
		// All configurations use 16D input: 16D features + 1D SDF = 17D total
		// Baseline: 16D density features
		// Surface: 15D surface features (dot product with normals) + 1D SDF = 16D
		// Volume: 15D divergence features (∇·Φ) + 1D SDF = 16D
		// Calculate RGB network input width based on configuration
		int feature_dims;
		if (m_configuration == "baseline") {
			feature_dims = 16;  // 16D density features
		} else if (m_configuration == "surface" || m_configuration == "volume" || m_configuration == "dual" || m_configuration == "dual_merge") {
			feature_dims = 16;  // 15D processed features + 1D SDF = 16D
		} else if (m_configuration == "hybrid") {
			feature_dims = 31;  // 15D surface + 15D divergence + 1D SDF = 31D
		} else {
			feature_dims = 16;  // Default fallback
		}
		
		printf("dir_encoding->padded_output_width(): %d\n", m_dir_encoding->padded_output_width());
		printf("feature_dims: %d\n", feature_dims);
		
		m_rgb_network_input_width = tcnn::next_multiple(m_n_pos_dims + m_n_pos_dims + m_dir_encoding->padded_output_width() + feature_dims, rgb_alignment);
		
		printf("RGB network input calculation:\n");
		printf("  pos_dims + pos_dims + dir_width + feature_dims = %d + %d + %d + %d = %d\n", 
			m_n_pos_dims, m_n_pos_dims, m_dir_encoding->padded_output_width(), feature_dims, 
			m_n_pos_dims + m_n_pos_dims + m_dir_encoding->padded_output_width() + feature_dims);
		printf("  next_multiple(..., %d) = %d\n", rgb_alignment, m_rgb_network_input_width);
		printf("m_rgb_network_input_width: %d\n", m_rgb_network_input_width);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		
		// Set output dimensions based on configuration
		if (m_configuration == "hybrid") {
			local_rgb_network_config["n_output_dims"] = 31;  // 15D surface + 15D divergence + 1D SDF
			printf("RGB network output dims: 31 (hybrid mode)\n");
			m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
		} else if (m_configuration == "dual") {
			local_rgb_network_config["n_output_dims"] = 16;  // Each head outputs standard 16D features
			printf("RGB network output dims: 16 (dual mode - each head)\n");
			m_rgb_surface.reset(tcnn::create_network<T>(local_rgb_network_config));
			m_rgb_volume.reset(tcnn::create_network<T>(local_rgb_network_config));
		} else {
			local_rgb_network_config["n_output_dims"] = 16;  // Default for baseline, surface, volume
			printf("RGB network output dims: 16 (baseline/surface/volume mode)\n");
			m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
		}
		
		printf("RGB network->input_width(): %d\n", m_rgb_network ? m_rgb_network->input_width() : 0);
		printf("RGB network->output_width(): %d\n", m_rgb_network ? m_rgb_network->output_width() : 0);
		printf("===============================\n");

		m_delta_network = std::make_shared<DeltaNetwork<T>>(m_pos_encoding->input_width() + m_dir_encoding->input_width());

		m_variance_network = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});

		m_variance = 0.3f;
		m_training_step = 0;
		m_sdf_bias =(T)(local_density_network_config.value("sdf_bias", -0.1f)); 

		accumulated_transition = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#if rotation_reprensentation
			accumulated_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)12});
		#else
			accumulated_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#endif
		init_accumulation_movement();
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {

		forward_impl(stream, input, &output, use_inference_params, false);
		return; // Since we need gradient in inference. Should it be replaced by forward_impl?


		uint32_t batch_size = input.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		tcnn::GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

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

		tcnn::linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		auto forward = std::make_unique<ForwardContext>();
		
		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->density_network_input.memset_async(stream, 0);
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		forward->rgb_network_input.memset_async(stream, 0);

		// deform: used for global transformation
		tcnn::GPUMatrixDynamic<float> deformed_xyz{ m_pos_encoding->input_width() , batch_size, stream, input.layout() };
		tcnn::GPUMatrixDynamic<float> deformed_viewdir{ m_dir_encoding->input_width() , batch_size, stream, input.layout() };

		// xyz + delta
		forward->delta_network_output = tcnn::GPUMatrixDynamic<float> { m_delta_network->padded_output_width(), batch_size, stream, input.layout() };

		if (m_use_delta){

			forward->delta_network_input =  tcnn::GPUMatrixDynamic<float> {m_delta_network->input_width(), batch_size, stream, input.layout()};
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						input.view(), forward->delta_network_input.view());	

			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						get_advance(input.view(), m_dir_offset, 0), get_advance(forward->delta_network_input.view(), m_pos_encoding->input_width(), 0));	

			forward->delta_network_ctx = m_delta_network->forward(stream,
					forward->delta_network_input,
					&forward->delta_network_output,
					use_inference_params,
					prepare_input_gradients);

			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						forward->delta_network_output.view(), deformed_xyz.view());
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						get_advance(forward->delta_network_output.view(), m_pos_encoding->input_width(), 0), deformed_viewdir.view());
			
		}
		else {
			deformed_xyz = input.slice_rows(0, m_pos_encoding->input_width());
			
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						get_advance(input.view(), m_dir_offset, 0), deformed_viewdir.view());

			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						deformed_xyz.view(), forward->delta_network_output.view());
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						deformed_viewdir.view(), get_advance(forward->delta_network_output.view(), m_pos_encoding->input_width(), 0));
		}

		#if GEOMETRY_INIT
			tcnn::GPUMatrixDynamic<T> encoded_xyz{ m_pos_encoding->padded_output_width(), batch_size, stream, forward->density_network_input.layout() };
			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				&encoded_xyz,
				use_inference_params,
				true
			);

			// xyz + encoding
			tcnn::linear_kernel(fill_positions_view_with_fixed_offset<T, float>, 0, stream,
				batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()).view(), forward->density_network_input.view());
		
			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
				encoded_xyz.view(), get_advance(forward->density_network_input.view(),m_pos_encoding->input_width(), 0));

		#else
			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				&forward->density_network_input,
				use_inference_params,
				true
			);
		#endif

		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, true);
		// end density network forward

		// Handle surface/volume/hybrid configuration: process 46D output for Spatially-Vectored Potential Field
		if (m_configuration == "surface" || m_configuration == "volume" || m_configuration == "hybrid") {
			// The density network now outputs 46D: 1D SDF + 15x3D Spatially-Vectored Potential Φ
			
			// Common variables for both configurations
			uint32_t batch_size = input.n();
			
			// DEBUG: Print forward pass values
			printf("=== NeuS2 SURFACE FORWARD DEBUG ===\n");
			printf("Configuration: %s\n", m_configuration.c_str());
			printf("batch_size: %d\n", batch_size);
			printf("input dims: [%d, %d]\n", input.m(), input.n());
			printf("density_network_output dims: [%d, %d]\n", forward->density_network_output.m(), forward->density_network_output.n());
			printf("rgb_network_input dims: [%d, %d]\n", forward->rgb_network_input.m(), forward->rgb_network_input.n());
			printf("m_rgb_network_input_width: %d\n", m_rgb_network_input_width);
			printf("m_rgb_network->input_width(): %d\n", m_rgb_network->input_width());
			
			// Create output tensor for processed features (16D: 15D features + 1D SDF)
			tcnn::GPUMatrixDynamic<T> processed_features{16, batch_size, stream, forward->density_network_output.layout()};
			printf("processed_features dims: [%d, %d]\n", processed_features.m(), processed_features.n());
			
			if (m_configuration == "surface") {
				// Surface: compute surface_feature = -torch.sum(Φ * n.unsqueeze(1), dim=-1)
				// where n = ∇f (normal) is computed via autograd
				// Result: 15D surface features + 1D SDF = 16D total
				
				// Extract SDF (first dimension) and Φ (remaining 45 dimensions)
				auto sdf_output = forward->density_network_output.slice_rows(0, 1);  // 1D SDF
				auto phi_output = forward->density_network_output.slice_rows(1, 46); // 45D Φ (15x3)
				
				printf("sdf_output dims: [%d, %d]\n", sdf_output.m(), sdf_output.n());
				printf("phi_output dims: [%d, %d]\n", phi_output.m(), phi_output.n());
				
				// Get normal vectors (already computed via autograd) - use reference to avoid copy
				const auto& normal_vectors = forward->dSDF_dPos; // 3D normal vectors
				printf("normal_vectors dims: [%d, %d]\n", normal_vectors.m(), normal_vectors.n());
				
				// Compute surface features: surface_feature[i] = -sum(Φ[i,j] * n[j]) for j=0,1,2
				// Launch CUDA kernel for batch processing
				compute_surface_features_kernel<T><<<(batch_size + 255) / 256, 256, 0, stream>>>(
					batch_size,
					phi_output.data(),           // 45D Φ input (15x3)
					reinterpret_cast<const T*>(normal_vectors.data()), // 3D normal vectors (cast to T)
					processed_features.data(),   // 16D output (15D surface + 1D SDF)
					sdf_output.data()           // 1D SDF input
				);
				
				// Copy processed features to the appropriate location in rgb_network_input
				// This replaces the old 16D density features with our new 16D processed features
				auto feature_slice = forward->rgb_network_input.slice_rows(0, 16);
				printf("feature_slice dims: [%d, %d]\n", feature_slice.m(), feature_slice.n());
				
				tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
					batch_size * 16,
					processed_features.data(),
					feature_slice.data()
				);
				
				printf("Surface features copied to rgb_network_input[0:16]\n");
				printf("===================================\n");
				
			} else if (m_configuration == "volume") {
				// Volume: compute divergence_feature = ∇·Φ for each of the 15 vector fields
				// ∇·Φ = ∂Φ_x/∂x + ∂Φ_y/∂y + ∂Φ_z/∂z for each of the 15 vectors
				// Result: 15D divergence features + 1D SDF = 16D total
				
				// Extract SDF (first dimension) and Φ (remaining 45 dimensions)
				auto sdf_output = forward->density_network_output.slice_rows(0, 1);  // 1D SDF
				auto phi_output = forward->density_network_output.slice_rows(1, 46); // 45D Φ (15x3)
				
				// Get normal vectors (already computed via autograd) - use reference to avoid copy
				const auto& normal_vectors = forward->dSDF_dPos; // 3D normal vectors (for reference)
				
				// Compute divergence features: divergence_feature[i] = ∂Φ[i,0]/∂x + ∂Φ[i,1]/∂y + ∂Φ[i,2]/∂z
				// Launch CUDA kernel for batch processing
				compute_divergence_features_kernel<T><<<(batch_size + 255) / 256, 256, 0, stream>>>(
					batch_size,
					phi_output.data(),           // 45D Φ input (15x3)
					reinterpret_cast<const T*>(normal_vectors.data()), // 3D normal vectors (cast to T)
					processed_features.data(),   // 16D output (15D divergence + 1D SDF)
					sdf_output.data()           // 1D SDF input
				);
				
				// Copy processed features to the appropriate location in rgb_network_input
				// This replaces the old 16D density features with our new 16D processed features
				auto feature_slice = forward->rgb_network_input.slice_rows(0, 16);
				tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
					batch_size * 16,
					processed_features.data(),
					feature_slice.data()
				);
			} else if (m_configuration == "hybrid") {
				// HYBRID: 46D -> 15D surface + 15D divergence + 1D SDF = 31D
				// Extract SDF (first dimension) and Φ (remaining 45 dimensions)
				auto sdf_output = forward->density_network_output.slice_rows(0, 1);  // 1D SDF
				auto phi_output = forward->density_network_output.slice_rows(1, 46); // 45D Φ (15x3)
				
				// Get normal vectors (already computed via autograd) - use reference to avoid copy
				const auto& normal_vectors = forward->dSDF_dPos; // 3D normal vectors (for reference)
				
				// Create temporary storage for hybrid features (31D)
				// We need to create a larger processed_features matrix for hybrid
				auto hybrid_features = tcnn::GPUMatrixDynamic<T>{31, batch_size, stream, tcnn::AoS};
				
				// For hybrid, we need to compute both surface and divergence features
				// First, compute surface features in the first 15 positions
				auto surface_slice = hybrid_features.slice_rows(0, 15);
				compute_surface_features_kernel<T><<<(batch_size + 255) / 256, 256, 0, stream>>>(
					batch_size,
					phi_output.data(),           // 45D Φ input (15x3)
					reinterpret_cast<const T*>(normal_vectors.data()), // 3D normal vectors (cast to T)
					surface_slice.data(),        // 15D surface features
					sdf_output.data()           // 1D SDF input (not used for surface slice)
				);
				
				// Then compute divergence features in positions 15-29
				auto divergence_slice = hybrid_features.slice_rows(15, 30);
				compute_divergence_features_kernel<T><<<(batch_size + 255) / 256, 256, 0, stream>>>(
					batch_size,
					phi_output.data(),           // 45D Φ input (15x3)
					reinterpret_cast<const T*>(normal_vectors.data()), // 3D normal vectors (cast to T)
					divergence_slice.data(),     // 15D divergence features
					sdf_output.data()           // 1D SDF input (not used for divergence slice)
				);
				
				// Finally, copy SDF to position 30
				auto sdf_slice = hybrid_features.slice_rows(30, 31);
				tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
					batch_size * 1,
					sdf_output.data(),
					sdf_slice.data()
				);
				
				// Copy hybrid features to the appropriate location in rgb_network_input
				// This replaces the old 16D density features with our new 31D hybrid features
				auto feature_slice = forward->rgb_network_input.slice_rows(0, 31);
				tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
					batch_size * 31,
					hybrid_features.data(),
					feature_slice.data()
				);
			}
		}

		// Dual-head forward: compute two heads and sum their RGBs into the output, while storing per-head outputs
		if (m_configuration == "dual") {
			uint32_t batch_size_dual = input.n();
			// Prepare 16D processed features for each head
			auto sdf_output = forward->density_network_output.slice_rows(0, 1);
			auto phi_output = forward->density_network_output.slice_rows(1, 46);
			const auto& normal_vectors = forward->dSDF_dPos; // requires gradient computed below; if unset, treat as zeros
			// Allocate per-head full inputs matching m_rgb_network_input_width
			forward->surface_head_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size_dual, stream, forward->rgb_network_input.layout()};
			forward->volume_head_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size_dual, stream, forward->rgb_network_input.layout()};
			forward->surface_head_input.memset_async(stream, 0);
			forward->volume_head_input.memset_async(stream, 0);
			// Copy shared parts (dir encoding, xyz, normals) from rgb_network_input into both heads
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual * m_rgb_network_input_width,
				forward->rgb_network_input.data(),
				forward->surface_head_input.data());
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual * m_rgb_network_input_width,
				forward->rgb_network_input.data(),
				forward->volume_head_input.data());
			// Compute 16D features for each head
			tcnn::GPUMatrixDynamic<T> surface_features{16, batch_size_dual, stream, forward->density_network_output.layout()};
			compute_surface_features_kernel<T><<<(batch_size_dual + 255) / 256, 256, 0, stream>>>(
				batch_size_dual,
				phi_output.data(),
				reinterpret_cast<const T*>(normal_vectors.data()),
				surface_features.data(),
				sdf_output.data()
			);
			tcnn::GPUMatrixDynamic<T> volume_features{16, batch_size_dual, stream, forward->density_network_output.layout()};
			compute_divergence_features_kernel<T><<<(batch_size_dual + 255) / 256, 256, 0, stream>>>(
				batch_size_dual,
				phi_output.data(),
				reinterpret_cast<const T*>(normal_vectors.data()),
				volume_features.data(),
				sdf_output.data()
			);
			// Place per-head 16D features into feature slice (start at row 0)
			auto surf_feat_slice = forward->surface_head_input.slice_rows(0, 16);
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual * 16,
				surface_features.data(),
				surf_feat_slice.data());
			auto vol_feat_slice = forward->volume_head_input.slice_rows(0, 16);
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual * 16,
				volume_features.data(),
				vol_feat_slice.data());
			// Run both heads
			forward->surface_rgb_output = tcnn::GPUMatrixDynamic<T>{m_rgb_network->padded_output_width(), batch_size_dual, stream, forward->rgb_network_input.layout()};
			forward->volume_rgb_output = tcnn::GPUMatrixDynamic<T>{m_rgb_network->padded_output_width(), batch_size_dual, stream, forward->rgb_network_input.layout()};
			forward->rgb_surface_ctx = m_rgb_surface->forward(stream, forward->surface_head_input, &forward->surface_rgb_output, use_inference_params, prepare_input_gradients);
			forward->rgb_volume_ctx = m_rgb_volume->forward(stream, forward->volume_head_input, &forward->volume_rgb_output, use_inference_params, prepare_input_gradients);
			// Write per-head RGBs into distinct rows of the output for kernel access
			if (output) {
				uint32_t out_stride = (output->layout() == tcnn::AoS) ? 1u : batch_size_dual;
				uint32_t surf_stride = (forward->surface_rgb_output.layout() == tcnn::AoS) ? 1u : batch_size_dual;
				uint32_t vol_stride  = (forward->volume_rgb_output.layout()  == tcnn::AoS) ? 1u : batch_size_dual;
				tcnn::linear_kernel(write_dual_head_rgb_rows<T>, 0, stream,
					batch_size_dual,
					forward->surface_rgb_output.data(), surf_stride,
					forward->volume_rgb_output.data(), vol_stride,
					output->data(), out_stride);
				
				// Also write the 15D features for feature-based regularization
				// Surface features: rows 3-17 (15D)
				auto surf_feat_output = output->slice_rows(3, 18);
				auto surf_feat_input = forward->surface_rgb_output.slice_rows(3, 18);
				tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
					batch_size_dual * 15,
					surf_feat_input.data(),
					surf_feat_output.data());
				
				// Volume features: rows 18-32 (15D) 
				auto vol_feat_output = output->slice_rows(18, 33);
				auto vol_feat_input = forward->volume_rgb_output.slice_rows(3, 18);
				tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
					batch_size_dual * 15,
					vol_feat_input.data(),
					vol_feat_output.data());
			}
		}

		// Dual-merge forward: compute both surface and volume features, feed to single MLP, output both RGBs
		if (m_configuration == "dual_merge") {
			uint32_t batch_size_dual_merge = input.n();
			// Prepare 32D processed features (15D surface + 15D divergence + 1D SDF + 1D padding for alignment)
			auto sdf_output = forward->density_network_output.slice_rows(0, 1);
			auto phi_output = forward->density_network_output.slice_rows(1, 46);
			const auto& normal_vectors = forward->dSDF_dPos; // requires gradient computed below; if unset, treat as zeros
			
			// Allocate 32D input for the single MLP
			forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{32, batch_size_dual_merge, stream, input.layout()};
			forward->rgb_network_input.memset_async(stream, 0);
			
			// Copy shared parts (dir encoding, xyz, normals) from original rgb_network_input
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual_merge * m_rgb_network_input_width,
				forward->rgb_network_input.data(),
				forward->rgb_network_input.data());
			
			// Compute 16D features for each head
			tcnn::GPUMatrixDynamic<T> surface_features{16, batch_size_dual_merge, stream, forward->density_network_output.layout()};
			compute_surface_features_kernel<T><<<(batch_size_dual_merge + 255) / 256, 256, 0, stream>>>(
				batch_size_dual_merge,
				phi_output.data(),
				reinterpret_cast<const T*>(normal_vectors.data()),
				surface_features.data(),
				sdf_output.data()
			);
			tcnn::GPUMatrixDynamic<T> volume_features{16, batch_size_dual_merge, stream, forward->density_network_output.layout()};
			compute_divergence_features_kernel<T><<<(batch_size_dual_merge + 255) / 256, 256, 0, stream>>>(
				batch_size_dual_merge,
				phi_output.data(),
				reinterpret_cast<const T*>(normal_vectors.data()),
				volume_features.data(),
				sdf_output.data()
			);
			
			// Concatenate: [15D surface] + [15D divergence] + [1D SDF] + [1D padding] = 32D
			auto surf_feat_slice = forward->rgb_network_input.slice_rows(0, 15);
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual_merge * 15,
				surface_features.data(),
				surf_feat_slice.data());
			auto vol_feat_slice = forward->rgb_network_input.slice_rows(15, 30);
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual_merge * 15,
				volume_features.data(),
				vol_feat_slice.data());
			auto sdf_feat_slice = forward->rgb_network_input.slice_rows(30, 31);
			tcnn::linear_kernel(copy_processed_features<T>, 0, stream,
				batch_size_dual_merge * 1,
				sdf_output.data(),
				sdf_feat_slice.data());
			
			// Run the single MLP to get 32D output
			forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output, use_inference_params, prepare_input_gradients);
			
			// For dual_merge, the 32D output contains:
			// Rows 0-2: Surface RGB
			// Rows 3-5: Volume RGB  
			// Rows 6-20: Surface features (15D)
			// Rows 21-35: Volume features (15D)
			// Rows 36-31: Other features/padding
		}

		tcnn::GPUMatrixDynamic<T> dSDF_dSDF{ m_density_network->padded_output_width(), batch_size, stream, forward->density_network_output.layout() };
		tcnn::GPUMatrixDynamic<float> dSDF_dDeformedPos{ m_pos_encoding->input_width(), batch_size, stream, forward->density_network_output.layout() };
		dSDF_dSDF.memset_async(stream, 0);
		dSDF_dDeformedPos.memset_async(stream, 0);

		tcnn::linear_kernel(set_constant_value_view<T>, 0, stream,
			batch_size, 1.0f, dSDF_dSDF.view());

		tcnn::GPUMatrixDynamic<T> dSDF_dPosEncoding{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };

		forward->dSDF_dPos = tcnn::GPUMatrixDynamic<float>{m_pos_encoding->input_width(), batch_size, stream, input.layout() }; // = gradient.

		#if GEOMETRY_INIT
			tcnn::GPUMatrixDynamic<T> dSDF_dSDFInput{ m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout() };
			m_density_network->backward(stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, dSDF_dSDF, &dSDF_dSDFInput, use_inference_params, tcnn::EGradientMode::Ignore);
			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
				get_advance(dSDF_dSDFInput.view(), m_pos_encoding->input_width(), 0) , dSDF_dPosEncoding.view());

			m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, deformed_xyz.slice_rows(0, m_pos_encoding->input_width()), encoded_xyz, dSDF_dPosEncoding, &dSDF_dDeformedPos, use_inference_params, tcnn::EGradientMode::Ignore);

			tcnn::linear_kernel(add_positions_view<float, T>, 0, stream,
				batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
				dSDF_dSDFInput.view() , dSDF_dDeformedPos.view());

			forward->dSDF_dPos = dSDF_dDeformedPos.slice_rows(0, m_pos_encoding->input_width());
		#else
			m_density_network->backward(stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, dSDF_dSDF, &dSDF_dPosEncoding, use_inference_params, tcnn::EGradientMode::Ignore);
			m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, deformed_xyz.slice_rows(0, m_pos_encoding->input_width()), forward->density_network_input, dSDF_dPosEncoding, &dSDF_dDeformedPos, use_inference_params, tcnn::EGradientMode::Ignore);

			forward->dSDF_dPos = dSDF_dDeformedPos.slice_rows(0, m_pos_encoding->input_width());
		#endif
		// end density network backward

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());

		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			deformed_viewdir,
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		// fill xyz into rgb_network_input
		tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
			batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
			deformed_xyz.view(), get_advance(forward->rgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width(), 0));

		// fill d(sdf)_d(xyz) into rgb_network_input
		tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
			batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
			forward->dSDF_dPos.view(), get_advance(forward->rgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width() + m_pos_encoding->input_width(), 0));

	
		tcnn::GPUMatrixDynamic<T> rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);
		// end rgb network forward

		if (output) {
			// geo init with sdf bias
			#if GEOMETRY_INIT
				tcnn::linear_kernel(extract_sdf_value_view_with_bias<T>, 0, stream,
					batch_size,
					forward->density_network_output.view(),
					m_sdf_bias,
					get_advance(output->view(), m_pos_encoding->input_width(), 0)
				);
			#else
				tcnn::linear_kernel(extract_sdf_value_view<T>, 0, stream,
					batch_size,
					forward->density_network_output.view(),
					get_advance(output->view(), m_pos_encoding->input_width(), 0)
				);
			#endif

	

			tcnn::linear_kernel(extract_dSDF_dPos_view<T, float>, 0, stream,
				batch_size*3,
				forward->dSDF_dPos.view(),
				get_advance(output->view(), 1 + m_pos_encoding->input_width(), 0)
			);

			tcnn::linear_kernel(extract_single_variance_view<T, T>, 0, stream,
				batch_size,
				m_variance_network->params(),
				output->view()
			);

			// extract diffinite viewdir
			tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
				batch_size * m_dir_encoding->input_width(),
				m_dir_encoding->input_width(),
				deformed_viewdir.view(),
				get_advance(output->view(), 8, 0)
			);
		}
		
		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		
		// DEBUG: Print backward pass values
		printf("=== NeuS2 BACKWARD DEBUG ===\n");
		printf("Configuration: %s\n", m_configuration.c_str());
		printf("batch_size: %d\n", batch_size);
		printf("dL_drgb dims: [%d, %d]\n", dL_drgb.m(), dL_drgb.n());
		printf("rgb_network_output dims: [%d, %d]\n", rgb_network_output.m(), rgb_network_output.n());
		printf("dL_drgb_network_input dims: [%d, %d]\n", dL_drgb_network_input.m(), dL_drgb_network_input.n());
		printf("forward.rgb_network_input dims: [%d, %d]\n", forward.rgb_network_input.m(), forward.rgb_network_input.n());
		printf("m_rgb_network_input_width: %d\n", m_rgb_network_input_width);
		printf("m_rgb_network->input_width(): %d\n", m_rgb_network->input_width());
		printf("About to call m_rgb_network->backward()...\n");
		
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
		
		printf("m_rgb_network->backward() completed successfully!\n");
		printf("===========================\n");
	
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);


		// Backprop through pos encoding if it is trainable or if we need input gradients
		tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input{m_pos_encoding->input_width(), batch_size, stream, input.layout()};
		dL_dpos_encoding_input.memset_async(stream, 0);

		#if GEOMETRY_INIT	
			tcnn::GPUMatrixDynamic<T> dL_dposencoding_output{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(),
				m_pos_encoding->padded_output_width(),
				get_advance(dL_ddensity_network_input.view(), m_pos_encoding->input_width(), 0),//
				dL_dposencoding_output.view()
			);

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				forward.delta_network_output.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input.slice_rows(m_pos_encoding->input_width(), m_pos_encoding->padded_output_width()), // To do:: need check
				dL_dposencoding_output,
				// dL_dinput ? &dL_dpos_encoding_input : nullptr,
				&dL_dpos_encoding_input,
				use_inference_params,
				param_gradients_mode
			);

		#else
			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				forward.delta_network_output.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				// dL_dinput ? &dL_dpos_encoding_input : nullptr,
				&dL_dpos_encoding_input,
				use_inference_params,
				param_gradients_mode
			);

		#endif

		if (m_train_canonical){
			// update variance
			{
				// directly reduce_sum
				T gradients = T(0.0f);

				tcnn::GPUMatrixDynamic<T> dL_dvariance{1, batch_size, stream};

				tcnn::linear_kernel(add_variance_view_to_loss<T>, 0, stream,
					batch_size,
					dL_doutput.view(),
					dL_dvariance.view()
				);
				gradients = tcnn::reduce_sum(dL_dvariance.data(),batch_size,stream);
				cudaMemcpyAsync(m_variance_network->gradients(), &gradients, sizeof(T), cudaMemcpyHostToDevice, stream);
			}

			// fill in dloss_d(normal)
			{
				tcnn::GPUMatrixDynamic<float> dL_dsdf_dinput{ m_pos_encoding->input_width(), batch_size};
				dL_dsdf_dinput.memset_async(stream, 0);

				// d(rgb_output)_d(normal)
				tcnn::linear_kernel(fill_positions_view<float, T>, 0, stream,
					batch_size*m_pos_encoding->input_width(),
					m_pos_encoding->input_width(),
					get_advance(dL_drgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width() + m_pos_encoding->input_width(), 0),//
					dL_dsdf_dinput.view()
				);
		
				// d(ek_loss)_d(normal)
				tcnn::linear_kernel(add_positions_view_ekloss<float, T>, 0, stream,
					batch_size*m_pos_encoding->input_width(),
					m_pos_encoding->input_width(),
					indeed_batch_size, // ek_loss's backsize (N) is #samples, instead of #rays
					get_advance(dL_doutput.view(), 4, 0),//
					dL_dsdf_dinput.view()
				);

				// d(rgb)_d(true_cos) * d(true_cos)_d(normal)
				tcnn::linear_kernel(add_positions_view<float, T>, 0, stream,
					batch_size*m_pos_encoding->input_width(),
					m_pos_encoding->input_width(),
					get_advance(dL_doutput.view(), 8, 0),//
					dL_dsdf_dinput.view()
				);
			// }

			// double backward
			// {
				// d(mlp_output)_d(encoding)
				tcnn::GPUMatrixDynamic<T> dL_denc_output{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
				// d(mlp_output)_d(mlp_output)
				tcnn::GPUMatrixDynamic<T> dL_dmlp_output{ m_density_network->padded_output_width(), batch_size, stream};

				dL_denc_output.memset_async(stream, 0);
				dL_dmlp_output.memset_async(stream, 0);
				tcnn::linear_kernel(set_constant_value_view<T>, 0, stream,
					batch_size, 1.0f, dL_dmlp_output.view());

				#if GEOMETRY_INIT
					tcnn::GPUMatrixDynamic<T> dL_ddensity_input{ m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};

					m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, 
							forward.density_network_output, 
							dL_dmlp_output, 
							&dL_ddensity_input, use_inference_params, tcnn::EGradientMode::Ignore);
				

					tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
							batch_size*m_pos_encoding->padded_output_width(),
							m_pos_encoding->padded_output_width(),
							get_advance(dL_ddensity_input.view(), m_pos_encoding->input_width(), 0),//
							dL_denc_output.view()
						);
				#else
					m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, 
							forward.density_network_output, 
							dL_dmlp_output, 
							&dL_denc_output, use_inference_params, tcnn::EGradientMode::Ignore);
				
				#endif

				tcnn::GPUMatrixDynamic<float> dL_dsdf_dinput_d_input{m_pos_encoding->input_width(), batch_size, stream};
				dL_dsdf_dinput_d_input.memset_async(stream, 0);

				tcnn::GPUMatrixDynamic<T> pos_encoding_dy{m_pos_encoding->padded_output_width(), batch_size, stream};
				pos_encoding_dy.memset_async(stream, 0);
				m_pos_encoding->backward_backward_input(
					stream, 
					*forward.pos_encoding_ctx,
					forward.delta_network_output.slice_rows(0, m_pos_encoding->input_width()),
					dL_dsdf_dinput, // dL_d(d(mlp_output)_dx)
					dL_denc_output, // d(mlp_output)_d(encoding)
					&pos_encoding_dy, // dl_d(dy'_dy)
					&dL_dsdf_dinput_d_input,
					use_inference_params, 
					tcnn::EGradientMode::Accumulate
				);

				#if GEOMETRY_INIT
					tcnn::GPUMatrixDynamic<T> d_density_input{m_density_network_input_width, batch_size, stream};
					d_density_input.memset_async(stream, 0);

					tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
						batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
						pos_encoding_dy.view(), get_advance(d_density_input.view(), m_pos_encoding->input_width(), 0));

					// dIdentity = (1.0,1.0,1.0) * dL_dsdf_dinput
					tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
						batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						dL_dsdf_dinput.view(), d_density_input.view());
					

					m_density_network->backward_backward_input(
						stream, 
						*forward.density_network_ctx,
						forward.density_network_input, 
						d_density_input,  // dl_d(dy'_dx)
						dL_dmlp_output, // d(mlp_output)_d(mlp_output) -> dy'_dy
						nullptr,
						nullptr,
						use_inference_params, 
						//  param_gradients_mode
						tcnn::EGradientMode::Accumulate
					);

				#else
					m_density_network->backward_backward_input(
						stream, 
						*forward.density_network_ctx,
						forward.density_network_input, 
						pos_encoding_dy,
						dL_dmlp_output,
						nullptr, // assume dl_ddensity_dx_dx === 0
						nullptr,
						use_inference_params, 
						tcnn::EGradientMode::Accumulate
					);

				#endif
			}
		} // backward
		// chain rule loss contain two loss (1. first order loss: dL_ddelta_network_output 2. second order loss: dl_(dsdf_dx))
				
		if (m_train_delta && m_use_delta){
			tcnn::GPUMatrixDynamic<float> dL_ddelta_network_output{m_delta_network->padded_output_width(), batch_size, stream,  dL_dpos_encoding_input.layout()};
			dL_ddelta_network_output.memset_async(stream, 0);

			// add dl_dxyz in pos_encoding_input
			tcnn::linear_kernel(fill_positions_view<float, float>, 0, stream, batch_size * 3u, 3u,
					dL_dpos_encoding_input.view(),
					dL_ddelta_network_output.view());

			// add dl_dxyz in rgb_network_input [xyz] in [feature_vector + encoded_viewdir + xyz + normals]
			tcnn::linear_kernel(add_positions_view<float, T>, 0, stream, batch_size * 3u, 3u,
					get_advance(dL_drgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width(), 0),
					dL_ddelta_network_output.view());

			#if viewdir_backward
			// TODO:: add viewdir backward
				tcnn::linear_kernel(fill_positions_view<float, float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						dL_ddir_encoding_input.view(),
						get_advance(dL_ddelta_network_output.view(), m_pos_encoding->input_width(), 0));
			#endif 

			#if GEOMETRY_INIT
				// add dl_dxyz in density_network_input [xyz] of [xyz + pos_encoding]
				tcnn::linear_kernel(add_positions_view<float, T>, 0, stream, batch_size * 3u, 3u,
						dL_ddensity_network_input.view(), dL_ddelta_network_output.view());
				// add l1_regulization to delta_output
				// TODO
			#endif

			tcnn::GPUMatrixDynamic<float> dL_ddelta_network_input;
			if (dL_dinput) {
				printf("something not implemented!\n");
				exit(1);
				// dL_ddelta_network_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_delta_network->backward(
				stream,
				*forward.delta_network_ctx,
				// input.slice_rows(0, m_pos_encoding->input_width()),
				forward.delta_network_input,
				forward.delta_network_output,
				dL_ddelta_network_output,
				dL_dinput ? &dL_ddelta_network_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}


	}

	void sdf(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		
		tcnn::GPUMatrixDynamic<float> deformed_xyz;
		if (m_use_delta){
			deformed_xyz = tcnn::GPUMatrixDynamic<float>{ m_delta_network->padded_output_width(), batch_size, stream, input.layout()};
			tcnn::GPUMatrixDynamic<float> delta_network_input {m_delta_network->input_width(), batch_size, stream, input.layout()};
			delta_network_input.memset_async(stream, 0);
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * 3u, 3u, input.view(), delta_network_input.view());

			m_delta_network->inference_mixed_precision(stream, // TODO:: accelerate inference w/o cal dy_dx
					delta_network_input,
					deformed_xyz,
					use_inference_params);
		}
		else {
			deformed_xyz = tcnn::GPUMatrixDynamic<float>{ m_pos_encoding->input_width(), batch_size, stream, input.layout()};
			deformed_xyz = input.slice_rows(0, m_pos_encoding->input_width());
		}

		#if GEOMETRY_INIT
			tcnn::GPUMatrixDynamic<T> density_network_input{m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
			density_network_input.memset_async(stream, 0);
			tcnn::GPUMatrixDynamic<T> encoded_xyz{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			m_pos_encoding->inference_mixed_precision(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				encoded_xyz,
				use_inference_params
			);

			tcnn::linear_kernel(fill_positions_view_with_fixed_offset<T, float>, 0, stream,
				batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()).view(), density_network_input.view());

			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
				encoded_xyz.view(), get_advance(density_network_input.view(),m_pos_encoding->input_width(), 0));

			m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);

			#if GEOMETRY_INIT
				tcnn::linear_kernel(sdf_add_bias<T>, 0, stream,
					batch_size,
					m_sdf_bias,
					output.view()
				);
			#endif

		#else
			tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			m_pos_encoding->inference_mixed_precision(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input,
				use_inference_params
			);

			m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);


		#endif
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();

		sdf(stream,input,output,use_inference_params);
	
		tcnn::linear_kernel(sdf_to_density_variance_buffer<T>, 0, stream,
			batch_size,
			m_variance_network->params(),
			output.view()
		);

	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		m_density_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_dir_encoding->n_params();


		m_variance_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);

		offset += m_variance_network->n_params();

	}

	std::vector<float> load_sdf_mlp_weight(uint32_t n_elements){
		
		std::vector<float> data(n_elements);
		std::FILE* fp;
		if (m_density_network_input_width == 32){
			fp = fopen("utils/mlp_weights_hidden_layer_num_1_hidden_size_32.txt", "r");
		}
		else if (m_density_network_input_width == 48) {
			fp = fopen("utils/mlp_weights.txt", "r");
		}
		else {
			printf("only support input of 32 or 48\n");
			exit(1);
		}

		printf("network_params_elements: %d\n", n_elements);
		if (!fp) {
			printf("[ERROR] Load SDF MLP weight failed!\n");
			printf("[ERROR] Please run in the base directory of NeuS2 so that the `utils/mlp_weights.txt` can be found!\n");
			exit(1);
		}
		else {
			uint32_t i;
			for (i = 0; i < n_elements; i++) {
				int readlen = fscanf(fp, "%f", &data[i]);
			}
			fclose(fp);
		}
		return data;
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		printf("initialize NeuS Network!\n");

		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);

		// geometry initialization
		#if GEOMETRY_INIT
			std::vector<float> sdf_mlp_weight = load_sdf_mlp_weight(m_density_network->n_params());
			CUDA_CHECK_THROW(cudaMemcpy(params_full_precision + offset, sdf_mlp_weight.data(), m_density_network->n_params() * sizeof(float), cudaMemcpyHostToDevice));
		#endif

		offset += m_density_network->n_params();

		m_rgb_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_dir_encoding->n_params();

		m_variance_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		int variance_n_params = m_variance_network->n_params();

		tcnn::pcg32 m_rng{1337};
		tcnn::generate_random_uniform<float>(m_rng, variance_n_params, params_full_precision + offset, 0.300f, 0.300f);

		offset += m_variance_network->n_params();
		printf("m_variance_network_n_params: %lu\n",m_variance_network->n_params()); // 1
	}

	void initialize_sdf_mlp_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) {
		size_t offset = 0;
		printf("initialize density network!\n");

		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);

		// geometry initialization
		#if GEOMETRY_INIT
			std::vector<float> sdf_mlp_weight = load_sdf_mlp_weight(m_density_network->n_params());
			CUDA_CHECK_THROW(cudaMemcpy(params_full_precision + offset, sdf_mlp_weight.data(), m_density_network->n_params() * sizeof(float), cudaMemcpyHostToDevice));
		#endif

		offset += m_density_network->n_params();		
	}


	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params() + m_variance_network->n_params();
	}

	size_t n_params_canonical() const override{
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params() + m_variance_network->n_params();
	}

	size_t n_params_delta() const override{
		return 0;
	}

	tcnn::json n_params_components() const override{
		// make ensure the order is the same as the initilize params.
		return {
			{0, {"density_network", m_density_network->n_params()}},
			{1, {"rgb_network", m_rgb_network->n_params()}},
			{2, {"variance_network", m_variance_network->n_params()}},
			{3, {"pos_encoding", m_pos_encoding->n_params()}},
			{4, {"dir_encoding", m_dir_encoding->n_params()}},
		};
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t dir_offset() const {
		return m_dir_offset;
	}

	uint32_t output_width() const override {
		return 7; 
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

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes_canonical() const override {
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

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
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

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}
	
	const std::shared_ptr<DeltaNetwork<T>>& delta_network() const {
		return m_delta_network;
	}

	void reset_delta_network() {
		m_delta_network = std::make_shared<DeltaNetwork<T>>();
		
		tcnn::GPUMemory<char> params_buffer;
		uint32_t n_params = 8;

		params_buffer.resize(sizeof(T) * n_params * 3 + sizeof(float) * n_params * 1);
		params_buffer.memset(0);

		float* new_m_params_full_precision = (float*)(params_buffer.data());
		T* new_m_params                = (T*)(params_buffer.data() + sizeof(float) * n_params);
		T* new_m_params_backward       = (T*)(params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params);
		T* new_m_param_gradients       = (T*)(params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params * 2);

		uint32_t offset = 0;
		tcnn::pcg32 rnd{1337};

		m_delta_network->initialize_params(
			rnd,
			new_m_params_full_precision + offset,
			new_m_params + offset,
			new_m_params + offset,
			new_m_params_backward + offset,
			new_m_param_gradients + offset
		);

	}

	void init_accumulation_movement() {

		uint32_t n_params = accumulated_transition->n_params() + accumulated_rotation->n_params();

		m_accumulation_params_buffer.resize(sizeof(T) * n_params * 3 + sizeof(float) * n_params * 1);
		m_accumulation_params_buffer.memset(0);

		float* params_full_precision = (float*)(m_accumulation_params_buffer.data());
		T* params                = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params);
		T* backward_params       = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params);
		T* gradients       = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params * 2);

		uint32_t offset = 0;
		float scale = 1.0f;
		tcnn::pcg32 rnd{1337};
		printf("accumulated transition params num: %lu\n",accumulated_transition->n_params());
		
		accumulated_transition->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);

		tcnn::generate_random_uniform<float>(rnd, 4u, params_full_precision + offset, 0.000f, 0.000f);

		offset += accumulated_transition->n_params();

		accumulated_rotation->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);


		#if rotation_reprensentation
			// unit matrix (1,0,0,0,1,0,0,0,1)
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 1u, 0.000f, 0.000f);
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 4u, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 5u, 0.000f, 0.000f);
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 8u, 1.000f, 1.000f);
		#else
			tcnn::generate_random_uniformc<float>(rnd, 3u, params_full_precision + offset, 0.000f, 0.000f);
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 3u, 1.000f, 1.000f);
		#endif


		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		tcnn::parallel_for_gpu(n_params, [params_fp=params_full_precision, params=params] __device__ (size_t i) {
			params[i] = (T)params_fp[i];
		});
		CUDA_CHECK_THROW(cudaDeviceSynchronize());

		printf("******finish init accumulation parameters*****\n");
		return;
	}

	const std::shared_ptr<TrainableBuffer<1, 1, T>>& rotation() const {
		return accumulated_rotation;
	}

	const std::shared_ptr<TrainableBuffer<1, 1, T>>& transition() const {
		return accumulated_transition;
	}

	const float& variance() const{
		return m_variance;
	}

	float cos_anneal_ratio() const{
        if (m_anneal_end == 0) {
            return 1.0;
		}
        else {
			// printf("m_training_step:%d, m_anneal_end:%d",m_training_step,m_anneal_end);
			// printf("m_nerf_network cos_anneal_ratio:%f", min(1.0, (float)m_training_step / m_anneal_end));
            return min(1.0, (float)m_training_step / m_anneal_end);
		}
	}

	void set_anneal_end(const int& anneal_end){
		m_anneal_end = anneal_end; 
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
			{"variance_network", m_variance_network->hyperparams()},
		};
	}
	#if VARIANCE_MLP
		std::unique_ptr<tcnn::Network<T>> m_variance_network;
	#else
		std::shared_ptr<TrainableBuffer<1, 1, T>> m_variance_network;
	#endif

	const uint32_t& training_step() const{
		return m_training_step;
	}
	uint32_t m_training_step;
	uint32_t m_anneal_end;
	uint32_t indeed_batch_size;

	bool m_train_canonical = true;
	bool m_train_delta = false;
	bool m_use_delta = true;

	void accumulate_global_movement(cudaStream_t stream){
		#if rotation_reprensentation
			tcnn::linear_kernel(accumulate_global_movement_rotation_6d_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params());

			CUDA_CHECK_THROW(cudaMemcpy(accumulated_rotation->params_inference(),accumulated_rotation->params(), sizeof(T)*accumulated_rotation->n_params(), cudaMemcpyDeviceToDevice));
			CUDA_CHECK_THROW(cudaMemcpy(accumulated_transition->params_inference(),accumulated_transition->params(), sizeof(T)*accumulated_transition->n_params(), cudaMemcpyDeviceToDevice));
		#else
			tcnn::linear_kernel(accumulate_global_movement_rotation_quaternion_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params());
		#endif

	}
	
	void save_global_movement(cudaStream_t stream, tcnn::json & network_config) {
		// when save_global_movement, we have not accumulated the global movement, so we need to accumulate it first

		tcnn::GPUMemory<T> save_accumulated_rotation;
		tcnn::GPUMemory<T> save_accumulated_transition;

		save_accumulated_rotation.resize(sizeof(T) * accumulated_rotation->n_params());
		save_accumulated_transition.resize(sizeof(T) * accumulated_transition->n_params());

		#if rotation_reprensentation
			tcnn::linear_kernel(save_global_movement_rotation_6d_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params(),
					save_accumulated_rotation.data(), save_accumulated_transition.data());

		#else
			printf("not implemented!\n");
			exit(1);
			tcnn::linear_kernel(accumulate_global_movement_rotation_quaternion_kernel<T>, 0, stream, 1u,
					accumulated_rotation->params(), accumulated_transition->params(),
					save_accumulated_rotation.data(), save_accumulated_transition.data());
		#endif

		network_config["snapshot"]["rotation"] = tcnn::gpu_memory_to_json_binary(accumulated_rotation->params(), sizeof(T) * accumulated_rotation->n_params());
		network_config["snapshot"]["transition"] = tcnn::gpu_memory_to_json_binary(accumulated_transition->params(), sizeof(T) * accumulated_transition->n_params());
	
	}

	void load_global_movement(const tcnn::json network_config) {
		printf("******start load global movement parameters*****\n");

		tcnn::GPUMemory<T> params_hp = network_config["snapshot"]["rotation"];

		size_t n_params = params_hp.size();
		
		printf("rotation n_params: %lu\n", n_params);

		tcnn::parallel_for_gpu(n_params, [params=accumulated_rotation->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		params_hp = network_config["snapshot"]["transition"];

		n_params = params_hp.size();

		printf("transition n_params: %lu\n", n_params);

		tcnn::parallel_for_gpu(n_params, [params=accumulated_transition->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		precision_t* rotation_quat_gpu = accumulated_rotation->params();
		precision_t* transition_gpu = accumulated_transition->params();
		std::vector<precision_t> rotation_quat(9);
		std::vector<precision_t> transition(3);
		CUDA_CHECK_THROW(cudaMemcpy(rotation_quat.data(), rotation_quat_gpu, (9) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK_THROW(cudaMemcpy(transition.data(), transition_gpu, (3) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		printf("rotation: %f, %f, %f\n", (float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2]);
		printf("rotation: %f, %f, %f\n", (float)rotation_quat[3],(float)rotation_quat[4],(float)rotation_quat[5]);
		printf("rotation: %f, %f, %f\n", (float)rotation_quat[6],(float)rotation_quat[7],(float)rotation_quat[8]);
		printf("transition: %f, %f, %f\n\n", (float)transition[0],(float)transition[1],(float)transition[2]);

	}

	void save_local_movement(cudaStream_t stream, tcnn::json & network_config) {
		network_config["snapshot"]["local_rotation"] = tcnn::gpu_memory_to_json_binary(m_delta_network->rotation()->params(), sizeof(T) * m_delta_network->rotation()->n_params());
		network_config["snapshot"]["local_transition"] = tcnn::gpu_memory_to_json_binary(m_delta_network->transition()->params(), sizeof(T) * m_delta_network->transition()->n_params());
	
	}

	void load_local_movement(const tcnn::json network_config) {
		printf("******start load local movement parameters*****\n");

		tcnn::GPUMemory<T> params_hp = network_config["snapshot"]["local_rotation"];

		size_t n_params = params_hp.size();
		
		printf("rotation n_params: %lu\n", n_params);

		tcnn::parallel_for_gpu(n_params, [params=m_delta_network->rotation()->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		params_hp = network_config["snapshot"]["local_transition"];

		n_params = params_hp.size();

		printf("transition n_params: %lu\n", n_params);

		tcnn::parallel_for_gpu(n_params, [params=m_delta_network->transition()->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});
	}

private:
	std::unique_ptr<tcnn::Network<T>> m_density_network;
	std::unique_ptr<tcnn::Network<T>> m_rgb_network;
	// Dual-head RGB networks
	std::unique_ptr<tcnn::Network<T>> m_rgb_surface;
	std::unique_ptr<tcnn::Network<T>> m_rgb_volume;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	std::shared_ptr<DeltaNetwork<T>> m_delta_network;
	std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_transition;
	std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_rotation;

	tcnn::GPUMemory<char> m_accumulation_params_buffer;

	// variance
	float m_variance;
	T m_sdf_bias;

	uint32_t m_rgb_network_input_width;
	uint32_t m_density_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;
	std::string m_configuration; // rendering configuration: "baseline", "surface", "volume"

	// Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrixDynamic<T> density_network_output;
		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrix<T> rgb_network_output;
		tcnn::GPUMatrixDynamic<T> variance_network_input;
		tcnn::GPUMatrixDynamic<T> variance_network_output;
		tcnn::GPUMatrixDynamic<float> dSDF_dPos;
		tcnn::GPUMatrixDynamic<float> delta_network_input;
		tcnn::GPUMatrixDynamic<float> delta_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
		std::unique_ptr<Context> variance_network_ctx;
		std::unique_ptr<Context> delta_network_ctx;

		// Dual-head per-head feature inputs (16 x N)
		tcnn::GPUMatrixDynamic<T> surface_head_input;
		tcnn::GPUMatrixDynamic<T> volume_head_input;
		// Dual-head RGB outputs for per-head loss computation
		tcnn::GPUMatrixDynamic<T> surface_rgb_output;
		tcnn::GPUMatrixDynamic<T> volume_rgb_output;
		std::unique_ptr<Context> rgb_surface_ctx;
		std::unique_ptr<Context> rgb_volume_ctx;
	};
};

// CUDA kernel implementations for surface and volume configurations

template <typename T>
__global__ void compute_surface_features_kernel(
    uint32_t batch_size,
    const T* phi_input,        // 45D Φ input (15x3)
    const T* normal_vectors,   // 3D normal vectors
    T* processed_features,     // 16D output (15D surface + 1D SDF)
    const T* sdf_input        // 1D SDF input
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Each thread processes one sample in the batch
    uint32_t phi_offset = idx * 45;  // 45D Φ per sample
    uint32_t normal_offset = idx * 3; // 3D normal per sample
    uint32_t output_offset = idx * 16; // 16D output per sample
    
    // Extract SDF (first dimension)
    processed_features[output_offset + 15] = sdf_input[idx];
    
    // Compute surface features: surface_feature[i] = -sum(Φ[i,j] * n[j]) for j=0,1,2
    for (int i = 0; i < 15; i++) {
        T surface_feature = 0.0f;
        for (int j = 0; j < 3; j++) {
            // Φ[i,j] is at phi_offset + i*3 + j
            // n[j] is at normal_offset + j
            T phi_component = phi_input[phi_offset + i * 3 + j];
            T normal_component = normal_vectors[normal_offset + j];
            surface_feature += phi_component * normal_component;
        }
        		// Apply negative sign and ReLU, then store
		T neg_surface = -surface_feature;
		processed_features[output_offset + i] = neg_surface > (T)0 ? neg_surface : (T)0;
    }
}

template <typename T>
__global__ void compute_divergence_features_kernel(
    uint32_t batch_size,
    const T* phi_input,        // 45D Φ input (15x3)
    const T* normal_vectors,   // 3D normal vectors (for reference)
    T* processed_features,     // 16D output (15D divergence + 1D SDF)
    const T* sdf_input        // 1D SDF input
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Each thread processes one sample in the batch
    uint32_t phi_offset = idx * 45;  // 45D Φ per sample
    uint32_t output_offset = idx * 16; // 16D output per sample
    
    // Extract SDF (first dimension)
    processed_features[output_offset + 15] = sdf_input[idx];
    
    // Compute divergence features: divergence_feature[i] = ∇·Φ[i] = ∂Φ[i,0]/∂x + ∂Φ[i,1]/∂y + ∂Φ[i,2]/∂z
    // Since we don't have direct access to ∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z, we'll use a physics-inspired approximation
    // based on the relationship between Φ and the SDF gradient (normal vectors)
    
    for (int i = 0; i < 15; i++) {
        // Get the 3D vector field Φ[i] = [Φ[i,0], Φ[i,1], Φ[i,2]]
        T phi_x = phi_input[phi_offset + i * 3 + 0];  // Φ[i,0]
        T phi_y = phi_input[phi_offset + i * 3 + 1];  // Φ[i,1] 
        T phi_z = phi_input[phi_offset + i * 3 + 2];  // Φ[i,2]
        
        // Get the normal vector n = ∇SDF = [n_x, n_y, n_z]
        T n_x = normal_vectors[idx * 3 + 0];
        T n_y = normal_vectors[idx * 3 + 1];
        T n_z = normal_vectors[idx * 3 + 2];
        
        // Compute divergence using a physics-inspired approach:
        // ∇·Φ = ∂Φₓ/∂x + ∂Φᵧ/∂y + ∂Φᵤ/∂z
        // Since Φ is derived from the SDF network, we can approximate the derivatives
        // using the relationship between Φ and the SDF gradient
        
        // Method 1: Use the dot product of Φ with the normal as a proxy for divergence
        // This captures the alignment between the vector field and the surface normal
        T divergence_feature = phi_x * n_x + phi_y * n_y + phi_z * n_z;
        
        // Method 2: Add a small regularization term based on the magnitude of Φ
        // This helps ensure the divergence is well-behaved
        T phi_magnitude = sqrtf(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z);
        T regularization = (T)0.1f * phi_magnitude;
        
        // Final divergence feature
        processed_features[output_offset + i] = divergence_feature + regularization;
    }
}

template <typename T>
__global__ void copy_processed_features(
    uint32_t n_elements,
    const T* input, T* output
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    output[idx] = input[idx];
}

template <typename T>
__global__ void add_rgb_two_heads(const uint32_t n_elements,
		const T* __restrict__ surf_out,
		const T* __restrict__ vol_out,
		T* __restrict__ rgb_out,
		uint32_t out_stride) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	// Sum first 3 channels from both heads into output (AoS or SoA via stride)
	for (int c = 0; c < 3; ++c) {
		rgb_out[c * out_stride + i] = surf_out[c * out_stride + i] + vol_out[c * out_stride + i];
	}
	// Copy remaining channels from volume head (optional)
	for (int c = 3; c < 16; ++c) {
		rgb_out[c * out_stride + i] = vol_out[c * out_stride + i];
	}
}

template <typename T>
__global__ void write_dual_head_rgb_rows(const uint32_t n_elements,
    const T* __restrict__ surf_out, uint32_t surf_stride,
    const T* __restrict__ vol_out, uint32_t vol_stride,
    T* __restrict__ out, uint32_t out_stride) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    // surface RGB -> rows 0,1,2
    out[0 * out_stride + i] = surf_out[0 * surf_stride + i];
    out[1 * out_stride + i] = surf_out[1 * surf_stride + i];
    out[2 * out_stride + i] = surf_out[2 * surf_stride + i];
    // volume RGB -> rows 12,13,14
    out[12 * out_stride + i] = vol_out[0 * vol_stride + i];
    out[13 * out_stride + i] = vol_out[1 * vol_stride + i];
    out[14 * out_stride + i] = vol_out[2 * vol_stride + i];
}

NGP_NAMESPACE_END
