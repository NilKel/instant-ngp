# Implement Explicit Density Grid for surface_explicit Mode

## Goal
Replace the placeholder approach in `surface_explicit` mode with a proper `tcnn::Encoding` dense grid for storing explicit density values. Density will come from the grid, while the MLP outputs features for surface computation.

## Key Principles
1. **ONLY affect surface_explicit mode** - all changes should be guarded by `if (m_method == "surface_explicit")`
2. **Non-invasive** - don't move or restructure existing functions used by other modes
3. **Follow existing patterns** - use the same structure as `m_pos_encoding` and `m_dir_encoding`

## Implementation Steps

### Step 1: Add Member Variable (MINIMAL CHANGE)
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In the private section with other member variables (around line 100)

```cpp
// Explicit density grid (for surface_explicit mode only)
std::shared_ptr<tcnn::Encoding<T>> m_density_grid;
```

**Location**: After the other encoding members like `m_pos_encoding` and `m_dir_encoding`.

### Step 2: Initialize Grid in Constructor (surface_explicit ONLY)
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In the constructor, after creating m_density_network (around line 180)

```cpp
// Initialize explicit density grid for surface_explicit mode
if (m_method == "surface_explicit") {
    // Get grid resolution from config (default 128)
    uint32_t grid_res = 128;
    if (density_network.contains("explicit_grid_resolution")) {
        grid_res = density_network["explicit_grid_resolution"];
    }
    
    // Create DenseGrid encoding config
    json dense_grid_config = {
        {"otype", "Grid"},
        {"type", "Dense"},
        {"n_levels", 1},
        {"n_features_per_level", 1},
        {"base_resolution", grid_res},
        {"interpolation", "Linear"}
    };
    
    m_density_grid.reset(create_encoding<T>(3, dense_grid_config, 1));
    
    printf("surface_explicit: Created %dx%dx%d density grid with %zu parameters\n", 
        grid_res, grid_res, grid_res, m_density_grid->n_params());
}
```

### Step 3: Update Forward Pass (surface_explicit ONLY)
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In `forward_impl()`, find the `if (m_method == "surface_explicit")` block (around line 820)

**Current approach**: Uses MLP output channel 0 as density
**New approach**: Get density from grid, use MLP for features only

```cpp
} else if (m_method == "surface_explicit") {
    // surface_explicit: density from grid + features from MLP
    
    // Step 1: Get density from grid (with gradients for normal computation)
    GPUMatrixDynamic<T> grid_density{1, batch_size, stream, AoS};
    forward->density_grid_ctx = m_density_grid->forward(
        stream,
        input.slice_rows(0, 3),  // positions only
        &grid_density,
        use_inference_params,
        true  // prepare_input_gradients for normal computation
    );
    
    // Step 2: MLP forward pass (48D output: 1 dummy + 45 features for 15 vectors)
    forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
    forward->density_network_ctx = m_density_network->forward(
        stream, forward->density_network_input, &forward->density_network_output, 
        use_inference_params, false
    );
    
    // Step 3: Compute surface features (dot products with normals)
    // Get analytical normals from MLP
    forward->analytical_normals = compute_analytical_normals_forward_unified(
        stream, batch_size, input, forward, use_inference_params
    );
    
    // Compute 15 surface features using existing kernel
    linear_kernel(compute_surface_features_from_vectors_kernel<T>, 0, stream,
        batch_size,
        forward->density_network_output.data() + 1,  // Skip first channel (dummy)
        forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
        forward->analytical_normals.data(),
        forward->analytical_normals.layout() == AoS ? forward->analytical_normals.stride() : 1,
        forward->rgb_network_input.data() + 1,  // Output to channels 1-15
        forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1
    );
    
    // Step 4: Replace channel 0 with grid density
    linear_kernel(replace_first_channel_kernel<T>, 0, stream,
        batch_size,
        grid_density.data(),
        forward->rgb_network_input.layout() == AoS ? forward->rgb_network_input.stride() : 1,
        forward->rgb_network_input.data()
    );
    
    // Step 5: Direction encoding (same as surface mode)
    dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
    forward->dir_encoding_ctx = m_dir_encoding->forward(
        stream, input.slice_rows(m_dir_offset, m_n_dir_dims), &dir_out, use_inference_params, false
    );
```

### Step 4: Update Backward Pass (surface_explicit ONLY)
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In `backward_impl()`, find the `if (m_method == "surface_explicit")` block (around line 1180)

```cpp
} else if (m_method == "surface_explicit") {
    // surface_explicit: backprop to both grid and MLP
    
    // Step 1: Backprop through surface features to get gradients for 45-D MLP output
    GPUMatrixDynamic<T> dL_ddensity_output_features{m_density_network->padded_output_width(), batch_size, stream, forward.density_network_output.layout()};
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensity_output_features.data(), 0, dL_ddensity_output_features.n_bytes(), stream));
    
    linear_kernel(backprop_surface_features_kernel<T>, 0, stream,
        batch_size,
        dL_drgb_network_input.data() + 1,  // Gradients for channels 1-15
        dL_drgb_network_input.layout() == AoS ? dL_drgb_network_input.stride() : 1,
        forward.analytical_normals.data(),
        forward.analytical_normals.layout() == AoS ? forward.analytical_normals.stride() : 1,
        dL_ddensity_output_features.data() + 1,  // Output to features (skip channel 0)
        dL_ddensity_output_features.layout() == AoS ? dL_ddensity_output_features.stride() : 1
    );
    
    // Step 2: Extract gradient for channel 0 (goes to grid)
    GPUMatrixDynamic<T> dL_dgrid_density{1, batch_size, stream, AoS};
    linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
        batch_size,
        dL_drgb_network_input.data(),
        dL_drgb_network_input.layout() == AoS ? dL_drgb_network_input.stride() : 1,
        dL_dgrid_density.data()
    );
    
    // Step 3: Backprop to density network parameters
    m_density_network->backward(
        stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output,
        dL_ddensity_output_features, &dL_ddensity_input, use_inference_params, param_gradients_mode
    );
    
    // Step 4: Backprop to grid parameters
    GPUMatrixDynamic<float> dL_dpositions_from_grid{3, batch_size, stream, AoS};
    m_density_grid->backward(
        stream, *forward.density_grid_ctx, 
        input.slice_rows(0, 3),  // positions
        dL_dgrid_density,  // gradients from channel 0
        &dL_dpositions_from_grid,
        use_inference_params,
        param_gradients_mode
    );
```

### Step 5: Update Inference (surface_explicit ONLY)
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In `inference_mixed_precision()`, find the `if (m_method == "surface_explicit")` block (around line 320)

```cpp
} else if (m_method == "surface_explicit") {
    // surface_explicit: density from grid + features from MLP + normals from grid gradients
    
    // Step 1: Get density from grid with gradients
    GPUMatrixDynamic<T> grid_density{1, batch_size, stream, AoS};
    m_density_grid->inference_mixed_precision(
        stream,
        input.slice_rows(0, 3),  // positions
        grid_density,
        use_inference_params
    );
    
    // Step 2: Compute normals from grid gradients
    GPUMatrixDynamic<float> normals = compute_normals_from_grid_gradients(
        stream, batch_size, input.slice_rows(0, 3), m_density_grid, use_inference_params
    );
    
    // Step 3: MLP forward for features
    m_density_network->inference_mixed_precision(
        stream, density_network_input, density_network_output, use_inference_params
    );
    
    // Step 4: Compute surface features
    linear_kernel(compute_surface_features_from_vectors_kernel<T>, 0, stream,
        batch_size,
        density_network_output.data() + 1,
        density_network_output.layout() == AoS ? density_network_output.stride() : 1,
        normals.data(),
        normals.layout() == AoS ? normals.stride() : 1,
        rgb_network_input.data() + 1,
        rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1
    );
    
    // Step 5: Replace channel 0 with grid density
    linear_kernel(replace_first_channel_kernel<T>, 0, stream,
        batch_size,
        grid_density.data(),
        rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
        rgb_network_input.data()
    );
```

### Step 6: Implement Normal Computation Helper
**File**: `include/neural-graphics-primitives/nerf_helpers.h`

**Location**: Add at the end of the file (around line 1330)

```cpp
/**
 * @brief Compute normals from density grid using autodiff
 * 
 * Uses backward_backward_input_gradients to get ∂density/∂xyz from the grid.
 * Normals are computed as n = -∇density / ||∇density||
 */
template <typename T>
GPUMatrixDynamic<float> compute_normals_from_grid_gradients(
    cudaStream_t stream,
    uint32_t batch_size,
    const GPUMatrixDynamic<float>& positions,
    std::shared_ptr<tcnn::Encoding<T>>& grid_encoding,
    bool use_inference_params
) {
    // Seed gradient: scalar 1.0 for each sample
    GPUMatrixDynamic<T> dL_dgrid_output{1, batch_size, stream, tcnn::AoS};
    linear_kernel(set_constant_value_view_kernel<T>, 0, stream,
        batch_size, T(1.0f), 1, dL_dgrid_output.data()
    );
    
    // Compute spatial gradients ∂density/∂xyz
    GPUMatrixDynamic<float> dDensity_dPos{3, batch_size, stream, tcnn::AoS};
    CUDA_CHECK_THROW(cudaMemsetAsync(dDensity_dPos.data(), 0, dDensity_dPos.n_bytes(), stream));
    
    // Dummy output (not used in backward_backward_input_gradients)
    GPUMatrixDynamic<T> dummy_output{1, batch_size, stream, tcnn::AoS};
    
    // Get gradients via autodiff
    grid_encoding->backward_backward_input_gradients(
        stream,
        positions,      // input positions
        dummy_output,   // not used
        dL_dgrid_output, // seed gradient
        dDensity_dPos,  // output: spatial gradients
        use_inference_params
    );
    
    // Normalize to get unit normals: n = -∇density / ||∇density||
    GPUMatrixDynamic<float> normals{3, batch_size, stream, tcnn::AoS};
    linear_kernel(normalize_grid_gradients_kernel, 0, stream,
        batch_size,
        dDensity_dPos.data(),
        normals.data()
    );
    
    return normals;
}
```

### Step 7: Update ForwardContext Struct
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In the `ForwardContext` struct (around line 2760)

Add this member:
```cpp
std::unique_ptr<Context> density_grid_ctx;  // For surface_explicit grid encoding
```

### Step 8: Add Required Kernels (if not already present)
**File**: `include/neural-graphics-primitives/nerf_helpers.h`

Check if these kernels exist. If not, add them:

```cpp
// Replace first channel with grid density
template <typename T>
__global__ void replace_first_channel_kernel(
    uint32_t n_elements,
    const T* __restrict__ grid_density,
    uint32_t stride,
    T* __restrict__ network_output
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    network_output[i * stride] = grid_density[i];
}

// Extract first channel gradient for grid backprop
template <typename T>
__global__ void extract_first_channel_gradient_kernel(
    uint32_t n_elements,
    const T* __restrict__ dL_dnetwork_output,
    uint32_t stride,
    T* __restrict__ dL_dgrid_density
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    dL_dgrid_density[i] = dL_dnetwork_output[i * stride];
}

// Normalize grid gradients to unit normals
__global__ void normalize_grid_gradients_kernel(
    uint32_t n_elements,
    const float* __restrict__ gradients,  // [3 x n_elements]
    float* __restrict__ normals            // [3 x n_elements]
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    
    // Load gradient components
    float gx = gradients[i * 3 + 0];
    float gy = gradients[i * 3 + 1];
    float gz = gradients[i * 3 + 2];
    
    // Compute magnitude
    float mag = sqrtf(gx*gx + gy*gy + gz*gz);
    
    // Normalize (handle zero gradients)
    if (mag > 1e-8f) {
        normals[i * 3 + 0] = -gx / mag;
        normals[i * 3 + 1] = -gy / mag;
        normals[i * 3 + 2] = -gz / mag;
    } else {
        // Default normal when gradient is zero
        normals[i * 3 + 0] = 0.0f;
        normals[i * 3 + 1] = 0.0f;
        normals[i * 3 + 2] = 1.0f;
    }
}
```

## Testing

### Build Test
```bash
cd /home/nilkel/Projects/instant-ngp
cmake --build build -j$(nproc)
```

### Functional Test
Use surface_explicit mode with a test scene and verify:
1. Density grid parameters are created (~4MB for 128³ grid)
2. Training converges
3. Normals look reasonable
4. Results are similar to surface mode

## Configuration Example

Users can control grid resolution:
```json
{
  "density_network": {
    "explicit_grid_resolution": 128
  }
}
```

## Key Points

1. **All changes are guarded by `if (m_method == "surface_explicit")`**
2. **No impact on other modes** (surface, volume, hash_surface, etc.)
3. **Follows existing tcnn::Encoding patterns** (same as pos_encoding, dir_encoding)
4. **Grid stores density, MLP outputs features** (48D: 1 dummy + 45 features)
5. **Normals computed from grid gradients** via backward_backward_input_gradients

## Memory Usage

- 128³ grid with fp16 = ~4MB parameters
- Comparable to a mid-sized hash encoding

## What NOT to Do

1. **Don't move existing functions** that other modes depend on
2. **Don't restructure ForwardContext** beyond adding the one new member
3. **Don't change function signatures** used by multiple modes
4. **Don't add code outside `if (m_method == "surface_explicit")` blocks**

## Summary

This is a **minimal, targeted change** that:
- Adds one member variable (`m_density_grid`)
- Initializes it for surface_explicit mode only
- Updates forward/backward/inference paths for surface_explicit only  
- Adds helper function for normal computation from grid gradients
- Reuses all existing infrastructure (tcnn::Encoding, kernels, etc.)

Total new code: ~200 lines
Modified existing code: ~3 blocks (forward, backward, inference for surface_explicit)
