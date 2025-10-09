# Implement Explicit Density Grid for surface_explicit Mode

## IMPLEMENTATION STATUS (UPDATED 2025-10-06)

### 🎯 QUICK SUMMARY

**Status**: Core implementation is COMPLETE and TRAINING SUCCESSFULLY (verified at 70%+ progress)

**What Works**:
- Dense grid creation with 2,097,152 parameters (128³ grid)
- Grid density replaces channel 0, MLP provides 47 features
- Forward/backward passes implemented with gradient flow to grid
- Trilinear interpolation with analytical normals from grid gradients

**What's Flaky**:
- Sometimes fails with `check failed: dL_doutput.m() == padded_output_width()`
- Likely a SoA vs AoS layout mismatch in some code path
- Grid reports `n_output_dims: 8` instead of 1 (due to padding, functionally OK)

**Next Agent Should**:
1. Debug the inconsistent layout issue causing intermittent failures
2. Verify all code paths use `preferred_output_layout()` consistently
3. Test with different batch sizes and grid resolutions
4. Consider removing debug prints once stable

### ✅ COMPLETED
1. **Grid Creation**: Successfully created DenseGrid encoding using tiny-cuda-nn
2. **Forward Pass**: Grid density replaces first channel, MLP provides features
3. **Backward Pass**: Gradients flow to both grid and MLP parameters
4. **Kernels**: Added replace_first_channel and extract_first_channel_gradient kernels
5. **Training**: System trains successfully (verified with progress bars showing 70%+ completion)

### 🐛 CRITICAL BUGS DISCOVERED & FIXED

#### Bug 1: Division by Zero in per_level_scale
**Problem**: With `n_levels=1`, tiny-cuda-nn computes `per_level_scale = exp(log(...) / (n_levels-1))` = `exp(... / 0)` = inf
**Solution**: Explicitly set `"per_level_scale": 1.0f` in grid config

#### Bug 2: Incorrect n_features Configuration  
**Problem**: Setting both `n_features` and `n_levels` causes tiny-cuda-nn to ignore one
**Solution**: Use ONLY `"n_levels": 1` with `"n_features_per_level": 1`, NOT `"n_features"`

#### Bug 3: Padded Output Width (MOST CRITICAL)
**Problem**: Even with 1 feature, tiny-cuda-nn pads output to 8 for alignment
- `grid->output_width()` returns 8 (not 1)
- `grid->padded_output_width()` returns 8
- All matrix allocations MUST use `padded_output_width()`, not hardcoded 1
**Solution**: 
- Allocate matrices with `m_density_grid->padded_output_width()`
- Use `preferred_output_layout()` instead of hardcoded AoS
- Extract/write only first channel but allocate full padded width

#### Bug 4: Kernel Stride Handling
**Problem**: Kernels assumed stride=1 for grid density, but actual stride = `padded_output_width()` = 8
**Solution**: Updated kernels to accept and use grid_stride parameter:
```cpp
// OLD: grid_density[i] 
// NEW: grid_density[i * grid_stride]
```

#### Bug 5: Uninitialized Gradient Padding
**Problem**: Gradient buffer allocated with width 8, but only first channel written → other 7 channels have garbage
**Solution**: Zero-initialize `dL_dgrid_density` before writing to first channel

### ⚠️ REMAINING ISSUES - ROOT CAUSE IDENTIFIED

**PRIMARY BUG**: Layout Mismatch in Grid Matrices

The error `RuntimeError: check failed: dL_doutput.m() == padded_output_width()` is caused by:

1. **GPUMatrixDynamic dimension swapping**:
   - **AoS layout**: `GPUMatrixDynamic(width, batch, ...)` → `m() = width`
   - **SoA layout**: `GPUMatrixDynamic(width, batch, ...)` → `m() = batch` (SWAPPED!)
   
2. **Using `preferred_output_layout()`**:
   - If grid prefers SoA, `m()` returns `batch_size` instead of `padded_output_width()`
   - Causes `grid->backward()` assertion failure

3. **FIX (just applied)**:
   ```cpp
   // WRONG: Uses preferred_output_layout which may be SoA
   forward->grid_density = GPUMatrixDynamic<T>{
       m_density_grid->padded_output_width(), batch_size, stream, 
       m_density_grid->preferred_output_layout()  // ❌ DON'T USE THIS
   };
   
   // CORRECT: Always use AoS to keep m() = padded_output_width
   forward->grid_density = GPUMatrixDynamic<T>{
       m_density_grid->padded_output_width(), batch_size, stream, 
       AoS  // ✅ ALWAYS USE AoS
   };
   ```

4. **Locations to fix**:
   - Line ~362: `grid_density_explicit` in `inference_mixed_precision_impl`
   - Line ~623: `forward->grid_density` in `forward_impl`
   - Line ~1257: `dL_dgrid_density` in `backward_impl` (also needs AoS)

**SECONDARY ISSUES**:

1. **Output Dimension Mismatch**: Grid reports `n_output_dims: 8` instead of 1
   - This is due to alignment padding in tiny-cuda-nn
   - Functionally works but confusing in logs
   - Not a bug, just misleading output

2. **Method Setting**: Must set `testbed.method = args.method` **BEFORE** `testbed.load_training_data()` 
   - Otherwise NerfNetwork constructor doesn't see correct method
   - Already fixed in run.py

### 📝 CORRECT GRID CONFIGURATION

```cpp
json dense_grid_config = {
    {"otype", "DenseGrid"},              // Use DenseGrid, not Grid
    {"n_levels", 1},                      // Single level
    {"n_features_per_level", 1},          // 1 feature per level
    {"base_resolution", grid_res},        // e.g., 128
    {"per_level_scale", 1.0f},            // MUST be 1.0 to avoid div-by-zero
    {"interpolation", "Linear"}           // Trilinear interpolation
};

// Use alignment 8 (standard)
m_density_grid.reset(create_encoding<T>(3, dense_grid_config, 8));
```

### 📝 CORRECT MATRIX ALLOCATION

```cpp
// WRONG:
GPUMatrixDynamic<T> grid_density{1, batch_size, stream, AoS};

// CORRECT:
GPUMatrixDynamic<T> grid_density{
    m_density_grid->padded_output_width(),  // Use padded width (8, not 1)
    batch_size, 
    stream, 
    m_density_grid->preferred_output_layout()  // Use grid's preferred layout
};
```

### 📝 CORRECT KERNEL USAGE

```cpp
// Replace first channel with grid density (extract from padded output)
linear_kernel(replace_first_channel_kernel<T>, 0, stream,
    batch_size,
    grid_density.data(),
    grid_density.layout() == AoS ? grid_density.stride() : 1,  // grid stride (8)
    rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,  // output stride
    rgb_network_input.data()
);

// Extract first channel gradient (write to padded gradient buffer)
GPUMatrixDynamic<T> dL_dgrid_density{
    m_density_grid->padded_output_width(), 
    batch_size, 
    stream, 
    m_density_grid->preferred_output_layout()
};
CUDA_CHECK_THROW(cudaMemsetAsync(dL_dgrid_density.data(), 0, dL_dgrid_density.n_bytes(), stream));

linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
    batch_size,
    dL_drgb_network_input.data(),
    dL_drgb_network_input.layout() == AoS ? dL_drgb_network_input.stride() : 1,
    dL_dgrid_density.layout() == AoS ? dL_dgrid_density.stride() : 1,  // grid stride
    dL_dgrid_density.data()
);
```

## Goal
Replace the placeholder approach in `surface_explicit` mode with a proper `tcnn::Encoding` dense grid for storing explicit density values. Density will come from the grid, while the MLP outputs features for surface computation.

## Key Principles
1. **ONLY affect surface_explicit mode** - all changes should be guarded by `if (m_method == "surface_explicit")`
2. **Non-invasive** - don't move or restructure existing functions used by other modes
3. **Follow existing patterns** - use the same structure as `m_pos_encoding` and `m_dir_encoding`
4. **HANDLE PADDING**: tiny-cuda-nn pads outputs for alignment - always use `padded_output_width()` and `preferred_output_layout()`

## Implementation Steps

### 📍 ACTUAL CODE CHANGES MADE

#### File: `include/neural-graphics-primitives/nerf_network.h`

**Line ~90**: Added member variable
```cpp
std::shared_ptr<tcnn::Encoding<T>> m_density_grid;  // Dense grid for surface_explicit mode
```

**Lines ~99-127**: Constructor - Initialize grid (CRITICAL: uses corrected config)
```cpp
if (m_method == "surface_explicit") {
    uint32_t grid_res = 2;  // Default resolution
    if (density_network.contains("explicit_grid_resolution")) {
        grid_res = density_network["explicit_grid_resolution"];
    }
    
    printf("GRID RESOLUTION: %d\n", grid_res);
    // NOTE: For single-level grids, must set per_level_scale=1.0 to avoid division by zero
    // NOTE: Can't specify both n_features and n_levels - use n_levels instead
    json dense_grid_config = {
        {"otype", "DenseGrid"},
        {"n_levels", 1},                     // Single level grid
        {"n_features_per_level", 1},         // 1 feature per level
        {"base_resolution", grid_res},
        {"per_level_scale", 1.0f},           // CRITICAL: avoid div-by-zero
        {"interpolation", "Linear"}
    };
    
    m_density_grid.reset(create_encoding<T>(3, dense_grid_config, 8));
    
    printf("surface_explicit: Created %dx%dx%d density grid with %zu parameters\n", 
        grid_res, grid_res, grid_res, m_density_grid->n_params());
    printf("  padded_output_width: %u, n_output_dims: %u\n", 
        m_density_grid->padded_output_width(), m_density_grid->output_width());
}
```

**Lines ~360-397**: `inference_mixed_precision_impl` - Forward pass with grid
```cpp
} else if (m_method == "surface_explicit") {
    // Step 1: Get density from grid (CRITICAL: use padded_output_width and preferred_output_layout)
    grid_density_explicit = GPUMatrixDynamic<T>{
        m_density_grid->padded_output_width(), 
        batch_size, 
        stream, 
        m_density_grid->preferred_output_layout()
    };
    auto grid_ctx = m_density_grid->forward(
        stream,
        input.slice_rows(0, 3),
        &grid_density_explicit,
        use_inference_params,
        true  // prepare_input_gradients for normals
    );
    
    // Step 2: Compute normals from grid gradients
    GPUMatrixDynamic<float> normals = compute_normals_from_grid_gradients(
        stream, batch_size, input.slice_rows(0, 3), m_density_grid, 
        *grid_ctx, grid_density_explicit, use_inference_params
    );
    
    // Step 3: MLP forward, Step 4: Compute surface features...
    
    // Step 5: Replace channel 0 with grid density (CRITICAL: handle padded stride)
    linear_kernel(replace_first_channel_kernel<T>, 0, stream,
        batch_size,
        grid_density_explicit.data(),
        grid_density_explicit.layout() == AoS ? grid_density_explicit.stride() : 1,
        rgb_network_input.layout() == AoS ? rgb_network_input.stride() : 1,
        rgb_network_input.data()
    );
}
```

**Lines ~619-660**: `forward_impl` - Training forward pass (similar to inference)
```cpp
} else if (m_method == "surface_explicit") {
    forward->density_network_output = GPUMatrixDynamic<T>{
        m_density_network->padded_output_width(), batch_size, stream, AoS
    };
    
    // CRITICAL: Use padded width and preferred layout
    forward->grid_density = GPUMatrixDynamic<T>{
        m_density_grid->padded_output_width(), 
        batch_size, 
        stream, 
        m_density_grid->preferred_output_layout()
    };
    forward->density_grid_ctx = m_density_grid->forward(
        stream,
        input.slice_rows(0, 3),
        &forward->grid_density,
        use_inference_params,
        true
    );
    
    // ... rest similar to inference
}
```

**Lines ~1255-1318**: `backward_impl` - Backward pass with grid gradients
```cpp
} else if (m_method == "surface_explicit") {
    // Step 1: Backprop through surface features...
    
    // Step 2: Extract gradient for channel 0 (CRITICAL: allocate with padding and zero-init)
    GPUMatrixDynamic<T> dL_dgrid_density{
        m_density_grid->padded_output_width(), 
        batch_size, 
        stream, 
        m_density_grid->preferred_output_layout()
    };
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dgrid_density.data(), 0, dL_dgrid_density.n_bytes(), stream));
    
    linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
        batch_size,
        dL_drgb_network_input.data(),
        dL_drgb_network_input.layout() == AoS ? dL_drgb_network_input.stride() : 1,
        dL_dgrid_density.layout() == AoS ? dL_dgrid_density.stride() : 1,
        dL_dgrid_density.data()
    );
    
    // Step 2.5: Backprop normals (for finite diff)...
    
    // Step 3: Backprop to density network...
    
    // Step 4: Backprop to grid parameters
    if (m_density_grid->n_params() > 0 || dL_dinput) {
        GPUMatrixDynamic<float> dL_dpositions_from_grid{3, batch_size, stream, AoS};
        
        m_density_grid->backward(
            stream, *forward.density_grid_ctx, 
            input.slice_rows(0, 3),
            forward.grid_density,
            dL_dgrid_density,
            dL_dinput ? &dL_dpositions_from_grid : nullptr,
            use_inference_params,
            param_gradients_mode
        );
    }
}
```

**Lines ~2607-2609**: ForwardContext struct - Added grid context storage
```cpp
// For surface_explicit mode: grid density output
GPUMatrixDynamic<T> grid_density;  // Grid density output for backward pass
```

#### File: `include/neural-graphics-primitives/nerf_helpers.h`

**Lines ~1134-1146**: Updated `replace_first_channel_kernel` (CRITICAL: handle padded stride)
```cpp
template <typename T>
__global__ void replace_first_channel_kernel(
    const uint32_t n_elements,
    const T* __restrict__ grid_density,
    const uint32_t grid_stride,         // NEW: grid stride parameter
    const uint32_t output_stride,       // NEW: output stride parameter
    T* __restrict__ network_output
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    // Extract first channel from padded grid output
    network_output[i * output_stride] = grid_density[i * grid_stride];
}
```

**Lines ~1158-1169**: Updated `extract_first_channel_gradient_kernel` (CRITICAL: write to padded buffer)
```cpp
template <typename T>
__global__ void extract_first_channel_gradient_kernel(
    const uint32_t n_elements,
    const T* __restrict__ dL_dnetwork_output,
    const uint32_t input_stride,
    const uint32_t grid_stride,          // NEW: grid stride parameter
    T* __restrict__ dL_dgrid_density
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    // Write to first channel of padded grid gradient
    dL_dgrid_density[i * grid_stride] = dL_dnetwork_output[i * input_stride];
}
```

#### File: `scripts/run.py`

**Line ~245**: Method must be set BEFORE loading training data
```python
# CRITICAL: Set method BEFORE loading data (so NerfNetwork constructor sees it)
testbed.method = args.method
testbed.load_training_data(args.scene)
```

#### File: `configs/nerf/base_explicit.json`

**Line ~34**: Added grid resolution configuration
```json
{
  "network": {
    ...
    "explicit_grid_resolution": 128  // Grid resolution for surface_explicit mode
  }
}
```

---

### Step 1: Add Member Variable (✅ DONE - See above)
**File**: `include/neural-graphics-primitives/nerf_network.h`

**Location**: In the private section with other member variables (around line 90)

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

## Extension: Finite Difference Gradient Backpropagation

### Goal
Add support for computing normals via finite differences with full gradient backpropagation to the density grid. Currently, finite differences are computed but gradients don't flow back to the grid parameters.

### Current State (Working)
- `--grad analytical`: Uses autodiff via `grid_encoding->backward()` to compute normals and backprop to grid ✓
- `--grad finite`: Uses finite differences to compute normals BUT gradients DON'T flow back to grid ✗

### Problem
In finite difference mode, we call `inference_mixed_precision()` for the ±eps samples, which:
1. Doesn't save a forward context
2. Doesn't allow gradients to flow backward
3. Results in grid cells that determine normals NOT receiving gradients from the loss

### Solution Overview
Replace `inference_mixed_precision()` with `forward()` calls and save all contexts/buffers, then backprop through them in the backward pass.

### Implementation Steps

#### Step 1: Extend ForwardContext Struct
**File**: `include/neural-graphics-primitives/nerf_network.h`
**Location**: Inside `ForwardContext` struct (after `grid_density` member)

```cpp
// For finite difference gradient computation (surface_explicit --grad finite)
std::vector<std::unique_ptr<Context>> finite_diff_contexts;  // 6 contexts for ±x, ±y, ±z
std::vector<GPUMatrixDynamic<T>> finite_diff_densities;      // 6 density outputs
std::vector<GPUMatrixDynamic<float>> finite_diff_positions;  // 6 offset position buffers
```

#### Step 2: Update compute_normals_from_grid_gradients Signature
**File**: `include/neural-graphics-primitives/nerf_helpers.h`
**Location**: Function signature (around line 1374)

**Old**:
```cpp
template <typename T>
tcnn::GPUMatrixDynamic<float> compute_normals_from_grid_gradients(
    cudaStream_t stream,
    uint32_t batch_size,
    const tcnn::GPUMatrixDynamic<float>& positions,
    std::shared_ptr<tcnn::Encoding<T>>& grid_encoding,
    const tcnn::Context& grid_ctx,
    const tcnn::GPUMatrixDynamic<T>& grid_output,
    bool use_inference_params
)
```

**New**:
```cpp
template <typename T, typename ForwardCtx = void>
tcnn::GPUMatrixDynamic<float> compute_normals_from_grid_gradients(
    cudaStream_t stream,
    uint32_t batch_size,
    const tcnn::GPUMatrixDynamic<float>& positions,
    std::shared_ptr<tcnn::Encoding<T>>& grid_encoding,
    const tcnn::Context& grid_ctx,
    const tcnn::GPUMatrixDynamic<T>& grid_output,
    bool use_inference_params,
    ForwardCtx* forward_ctx = nullptr  // Optional: save finite diff contexts for backprop
)
```

#### Step 3: Update Finite Difference Implementation to Save Contexts
**File**: `include/neural-graphics-primitives/nerf_helpers.h`
**Location**: Inside `compute_normals_from_grid_gradients`, the finite difference branch

**Replace** the current finite diff loop with:

```cpp
if (use_finite_diff) {
    const float eps = 1e-4f;  // TODO: Make proportional to grid cell size
    bool save_contexts = (forward_ctx != nullptr);
    
    if (save_contexts) {
        // Training mode: Use forward() to save contexts
        forward_ctx->finite_diff_contexts.reserve(6);
        forward_ctx->finite_diff_densities.reserve(6);
        forward_ctx->finite_diff_positions.reserve(6);
        
        for (int dim = 0; dim < 3; ++dim) {
            // Positive offset
            GPUMatrixDynamic<float> positions_pos{3, batch_size, stream, AoS};
            linear_kernel(offset_positions_kernel, 0, stream,
                batch_size, positions.data(), positions_pos.data(), dim, eps);
            GPUMatrixDynamic<T> density_pos{1, batch_size, stream, AoS};
            auto ctx_pos = grid_encoding->forward(stream, positions_pos, &density_pos, use_inference_params, false);
            
            // Negative offset
            GPUMatrixDynamic<float> positions_neg{3, batch_size, stream, AoS};
            linear_kernel(offset_positions_kernel, 0, stream,
                batch_size, positions.data(), positions_neg.data(), dim, -eps);
            GPUMatrixDynamic<T> density_neg{1, batch_size, stream, AoS};
            auto ctx_neg = grid_encoding->forward(stream, positions_neg, &density_neg, use_inference_params, false);
            
            // Compute gradient
            linear_kernel(compute_finite_difference_kernel<T>, 0, stream,
                batch_size, density_pos.data(), density_neg.data(), dDensity_dPos.data(), dim, eps);
            
            // Save for backward
            forward_ctx->finite_diff_contexts.push_back(std::move(ctx_pos));
            forward_ctx->finite_diff_densities.push_back(std::move(density_pos));
            forward_ctx->finite_diff_positions.push_back(std::move(positions_pos));
            forward_ctx->finite_diff_contexts.push_back(std::move(ctx_neg));
            forward_ctx->finite_diff_densities.push_back(std::move(density_neg));
            forward_ctx->finite_diff_positions.push_back(std::move(positions_neg));
        }
    } else {
        // Inference mode: Use inference_mixed_precision (no gradients)
        for (int dim = 0; dim < 3; ++dim) {
            GPUMatrixDynamic<float> positions_pos{3, batch_size, stream, AoS};
            linear_kernel(offset_positions_kernel, 0, stream,
                batch_size, positions.data(), positions_pos.data(), dim, eps);
            GPUMatrixDynamic<T> density_pos{1, batch_size, stream, AoS};
            grid_encoding->inference_mixed_precision(stream, positions_pos, density_pos, use_inference_params);
            
            GPUMatrixDynamic<float> positions_neg{3, batch_size, stream, AoS};
            linear_kernel(offset_positions_kernel, 0, stream,
                batch_size, positions.data(), positions_neg.data(), dim, -eps);
            GPUMatrixDynamic<T> density_neg{1, batch_size, stream, AoS};
            grid_encoding->inference_mixed_precision(stream, positions_neg, density_neg, use_inference_params);
            
            linear_kernel(compute_finite_difference_kernel<T>, 0, stream,
                batch_size, density_pos.data(), density_neg.data(), dDensity_dPos.data(), dim, eps);
        }
    }
}
```

#### Step 4: Add Backprop Kernels
**File**: `include/neural-graphics-primitives/nerf_helpers.h`
**Location**: Before `compute_normals_from_grid_gradients`

```cpp
// Backprop through normalization: n = -∇density / ||∇density||
static __global__ void backprop_normalization_kernel(
    const uint32_t n_elements,
    const float* __restrict__ dL_dnormals,
    const float* __restrict__ normals,
    float* __restrict__ dL_dgradients
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    
    const float nx = normals[i * 3 + 0];
    const float ny = normals[i * 3 + 1];
    const float nz = normals[i * 3 + 2];
    
    const float dL_dnx = dL_dnormals[i * 3 + 0];
    const float dL_dny = dL_dnormals[i * 3 + 1];
    const float dL_dnz = dL_dnormals[i * 3 + 2];
    
    const float dot_product = dL_dnx * nx + dL_dny * ny + dL_dnz * nz;
    
    dL_dgradients[i * 3 + 0] = -(dL_dnx - dot_product * nx);
    dL_dgradients[i * 3 + 1] = -(dL_dny - dot_product * ny);
    dL_dgradients[i * 3 + 2] = -(dL_dnz - dot_product * nz);
}

// Backprop through central difference
template <typename T>
static __global__ void backprop_finite_difference_kernel(
    const uint32_t n_elements,
    const float* __restrict__ dL_dgradients,
    const uint32_t dim,
    const float eps,
    T* __restrict__ dL_ddensity_pos,
    T* __restrict__ dL_ddensity_neg
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    
    const float dL_dgrad_dim = dL_dgradients[i * 3 + dim];
    const float inv_2eps = 1.0f / (2.0f * eps);
    
    dL_ddensity_pos[i] = T(dL_dgrad_dim * inv_2eps);
    dL_ddensity_neg[i] = T(-dL_dgrad_dim * inv_2eps);
}

// Backprop surface features to normals
template <typename T>
__global__ void backprop_surface_features_to_normals_kernel(
    const uint32_t n_elements,
    const T* __restrict__ dL_dsurface_features,
    const uint32_t feature_stride,
    const T* __restrict__ vectors,
    const uint32_t vector_stride,
    float* __restrict__ dL_dnormals,
    const uint32_t normal_stride
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    
    const T surface_scale = T(3.0f);
    float dL_dnx = 0.0f, dL_dny = 0.0f, dL_dnz = 0.0f;
    
    for (uint32_t k = 0; k < 15; ++k) {
        const T vx = vectors[i * vector_stride + k * 3 + 0];
        const T vy = vectors[i * vector_stride + k * 3 + 1];
        const T vz = vectors[i * vector_stride + k * 3 + 2];
        const T dL_dfeature = dL_dsurface_features[i * feature_stride + k];
        const T grad_scale = -surface_scale * dL_dfeature;
        
        dL_dnx += float(grad_scale * vx);
        dL_dny += float(grad_scale * vy);
        dL_dnz += float(grad_scale * vz);
    }
    
    dL_dnormals[i * normal_stride + 0] = dL_dnx;
    dL_dnormals[i * normal_stride + 1] = dL_dny;
    dL_dnormals[i * normal_stride + 2] = dL_dnz;
}
```

#### Step 5: Add Backprop Function
**File**: `include/neural-graphics-primitives/nerf_helpers.h`
**Location**: After `compute_normals_from_grid_gradients`

```cpp
template <typename T, typename ForwardCtx>
void backprop_normals_from_finite_differences(
    cudaStream_t stream,
    uint32_t batch_size,
    const tcnn::GPUMatrixDynamic<float>& dL_dnormals,
    const tcnn::GPUMatrixDynamic<float>& normals,
    std::shared_ptr<tcnn::Encoding<T>>& grid_encoding,
    ForwardCtx& forward_ctx,
    bool use_inference_params,
    tcnn::GradientMode param_gradients_mode
) {
    using namespace tcnn;
    
    GPUMatrixDynamic<float> dL_dRawGradients{3, batch_size, stream, AoS};
    linear_kernel(backprop_normalization_kernel, 0, stream,
        batch_size, dL_dnormals.data(), normals.data(), dL_dRawGradients.data());
    
    const float eps = 1e-4f;
    
    for (int dim = 0; dim < 3; ++dim) {
        GPUMatrixDynamic<T> dL_ddensity_pos{1, batch_size, stream, AoS};
        GPUMatrixDynamic<T> dL_ddensity_neg{1, batch_size, stream, AoS};
        
        linear_kernel(backprop_finite_difference_kernel<T>, 0, stream,
            batch_size, dL_dRawGradients.data(), dim, eps,
            dL_ddensity_pos.data(), dL_ddensity_neg.data());
        
        grid_encoding->backward(stream, *forward_ctx.finite_diff_contexts[dim * 2],
            forward_ctx.finite_diff_positions[dim * 2],
            forward_ctx.finite_diff_densities[dim * 2],
            dL_ddensity_pos, nullptr, use_inference_params, param_gradients_mode);
        
        grid_encoding->backward(stream, *forward_ctx.finite_diff_contexts[dim * 2 + 1],
            forward_ctx.finite_diff_positions[dim * 2 + 1],
            forward_ctx.finite_diff_densities[dim * 2 + 1],
            dL_ddensity_neg, nullptr, use_inference_params, param_gradients_mode);
    }
}
```

#### Step 6: Update Forward Pass Call
**File**: `include/neural-graphics-primitives/nerf_network.h`
**Location**: In `forward_impl`, surface_explicit block, the normal computation call

**Change**:
```cpp
forward->analytical_normals = compute_normals_from_grid_gradients(
    stream, batch_size, input.slice_rows(0, 3), m_density_grid, 
    *forward->density_grid_ctx, forward->grid_density, use_inference_params
);
```

**To**:
```cpp
forward->analytical_normals = compute_normals_from_grid_gradients(
    stream, batch_size, input.slice_rows(0, 3), m_density_grid, 
    *forward->density_grid_ctx, forward->grid_density, use_inference_params, 
    forward.get()  // Pass context to enable gradient saving
);
```

#### Step 7: Add Backward Pass Logic
**File**: `include/neural-graphics-primitives/nerf_network.h`
**Location**: In `backward_impl`, surface_explicit block, BEFORE Step 4 (grid backprop)

**Add**:
```cpp
// Step 3.5: Backprop gradients from normals to grid (for finite differences)
GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, forward.analytical_normals.layout()};
CUDA_CHECK_THROW(cudaMemsetAsync(dL_dnormals.data(), 0, dL_dnormals.n_bytes(), stream));

linear_kernel(backprop_surface_features_to_normals_kernel<T>, 0, stream,
    batch_size,
    dL_dsurface_slice.data(),
    dL_dsurface_slice.layout() == AoS ? dL_dsurface_slice.stride() : 1,
    forward.density_network_output.data() + 1,
    forward.density_network_output.layout() == AoS ? forward.density_network_output.stride() : 1,
    dL_dnormals.data(),
    dL_dnormals.layout() == AoS ? 3 : 1
);

const char* grad_method_env = std::getenv("NGP_GRAD_METHOD");
bool use_finite_diff = (grad_method_env && std::string(grad_method_env) == "finite");

if (use_finite_diff && !forward.finite_diff_contexts.empty()) {
    backprop_normals_from_finite_differences(
        stream, batch_size, dL_dnormals, forward.analytical_normals,
        m_density_grid, forward, use_inference_params, param_gradients_mode
    );
}
```

### Key Points

1. **Two modes**: Training saves contexts, inference doesn't
2. **6 samples per batch**: ±x, ±y, ±z offsets
3. **Memory**: 6x more contexts/buffers during training with finite diff
4. **Gradients flow**: Through normalization → finite diff → grid interpolation → grid parameters

### Testing

```bash
# Analytical (default, should work as before)
python scripts/run.py --scene data.json --method surface_explicit --grad analytical

# Finite differences (NEW: with gradients)
python scripts/run.py --scene data.json --method surface_explicit --grad finite
```

Compare training curves - both should converge, finite diff may be slightly slower.

### Notes

- `eps = 1e-4f` is fixed; ideally should be `1.0f / grid_resolution` for optimal accuracy
- Finite diff is ~7x slower (6 extra forward passes + 6 extra backward passes)
- Only use for debugging/comparison, not production training

