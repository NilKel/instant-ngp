# Explicit Density Grid Implementation Plan

## Overview
Use tiny-cuda-nn's Dense Grid encoding directly for density/SDF values, exactly like how HashGrid is used for positional encoding. No custom wrapper needed - just create another `tcnn::Encoding` instance!

## Key Insight

**We already know how to use grids** - the codebase uses `m_pos_encoding` (HashGrid) and `m_dir_encoding` (Frequency). We just need to add `m_density_grid` (DenseGrid) and use it the same way!

```cpp
// Existing code uses:
m_pos_encoding.reset(create_encoding<T>(3, pos_encoding_config, alignment));

// We'll add:
m_density_grid.reset(create_encoding<T>(3, dense_grid_config, 1));
```

## Architecture

### Current Surface Mode
```
Position (3D) 
  → HashGrid encoding (32D)
  → Density MLP (48D output)
      ├─ Channel 0: SDF/density
      └─ Channels 1-47: feature vectors
  → [Autodiff normals from channel 0]
  → Surface features: density + features + normals → 16D
  → + Direction encoding → RGB MLP → RGB
```

### Surface Explicit Mode (--method surface_explicit)
```
Position (3D)
  ├─ DenseGrid (1D) → density value + analytical dy/dx for normals
  └─ HashGrid encoding (32D) → Density MLP (48D features)
  → Surface features: grid_density + MLP_features + grid_normals → 16D
  → + Direction encoding → RGB MLP → RGB
```

**Key advantage**: 
- No autodiff through density network = simpler, more stable gradients
- Direct grid gradients from tiny-cuda-nn's built-in `dy_dx`
- Follows exact same pattern as existing encodings

---

## Implementation Guide

### Step 1: Add Dense Grid Encoding to NerfNetwork

**File**: `include/neural-graphics-primitives/nerf_network.h`

**Add member variable** (next to `m_pos_encoding` and `m_dir_encoding`):
```cpp
std::shared_ptr<tcnn::Encoding<T>> m_density_grid;  // Dense grid for explicit density
```

**In constructor**, after creating `m_dir_encoding`:
```cpp
// Check if using surface_explicit mode
if (m_method == "surface_explicit") {
    // Get grid resolution from config (default 128)
    uint32_t grid_res = 128;
    if (density_network.contains("grid_resolution")) {
        grid_res = density_network["grid_resolution"];
    }
    
    // Create dense grid encoding (1 level, 1 feature per level)
    json dense_grid_config = {
        {"otype", "Grid"},
        {"type", "Dense"},
        {"n_levels", 1},
        {"n_features_per_level", 1},
        {"base_resolution", grid_res},
        {"interpolation", "Linear"}  // trilinear interpolation
    };
    
    m_density_grid.reset(create_encoding<T>(3, dense_grid_config, 1));
    
    printf("Created dense grid for surface_explicit: %d^3 = %zu params\n", 
        grid_res, m_density_grid->n_params());
}
```

**Update n_output_dims** for density network (still 48D like surface mode):
```cpp
if (!density_network.contains("n_output_dims")) {
    if (m_method == "surface" || m_method == "surface_explicit" || ...) {
        local_density_network_config["n_output_dims"] = 48;
    }
}
```

**Update RGB network input width** (same as surface mode):
```cpp
if (m_method == "surface" || m_method == "surface_explicit") {
    // 16 surface features + direction encoding
    m_rgb_network_input_width = next_multiple(16 + m_dir_encoding->padded_output_width(), rgb_alignment);
}
```

---

### Step 2: Forward Pass - Get Density from Grid

**File**: `include/neural-graphics-primitives/nerf_network.h`, in `forward_impl()`

**Follow surface mode pattern exactly**, but replace density source:

```cpp
if (m_method == "surface_explicit") {
    // Same structure as surface mode
    forward->density_network_output = GPUMatrixDynamic<T>{m_density_network->padded_output_width(), batch_size, stream, AoS};
    dir_out = forward->rgb_network_input.slice_rows(16, m_dir_encoding->padded_output_width());
    
    // Get density and features from MLP (all 48 channels)
    forward->density_network_ctx = m_density_network->forward(
        stream, forward->density_network_input, 
        &forward->density_network_output, use_inference_params, false
    );
    
    // Get density from grid (replaces channel 0 from MLP)
    GPUMatrixDynamic<T> grid_density{1, batch_size, stream, AoS};
    forward->density_grid_ctx = m_density_grid->forward(
        stream,
        input.slice_rows(0, 3),  // 3D positions
        &grid_density,
        use_inference_params,
        true  // prepare_input_gradients = true to get dy/dx for normals
    );
    
    // Get normals from dy/dx (grid's analytical gradients)
    forward->analytical_normals = compute_normals_from_grid_gradients(
        stream, batch_size, forward->density_grid_ctx
    );
    
    // Compute surface features (same kernel as surface mode)
    auto surface_features_slice = forward->rgb_network_input.slice_rows(0, 16);
    linear_kernel(compute_surface_features_to_slice_kernel<T>, 0, stream,
        batch_size,
        forward->density_network_output.layout() == AoS ? forward->density_network_output.stride() : 1,
        forward->density_network_output.data(),
        forward->analytical_normals.data(),
        surface_features_slice.layout() == AoS ? surface_features_slice.stride() : 1,
        surface_features_slice.data(),
        false,  // not SDF mode
        nullptr  // no variance
    );
    
    // Replace channel 0 with grid density
    linear_kernel(replace_first_channel_kernel<T>, 0, stream,
        batch_size,
        grid_density.data(),
        surface_features_slice.data(),
        surface_features_slice.stride()
    );
}
```

---

### Step 3: Helper Function for Grid Normals

**File**: `include/neural-graphics-primitives/nerf_helpers.h`

**Add function to extract and normalize dy/dx from grid context**:
```cpp
/**
 * @brief Compute normals from grid dy/dx (analytical gradients from trilinear interpolation)
 * 
 * The grid encoding's dy/dx contains ∂density/∂x, ∂density/∂y, ∂density/∂z.
 * We normalize to get unit normals: n = -dy/dx / ||dy/dx||
 */
template <typename T>
GPUMatrixDynamic<float> compute_normals_from_grid_gradients(
    cudaStream_t stream,
    uint32_t batch_size,
    const std::unique_ptr<tcnn::Context>& grid_ctx
) {
    // Grid encoding stores dy_dx in its context (when prepare_input_gradients=true)
    // Access it through the context's dy_dx member
    auto& forward_ctx = dynamic_cast<const tcnn::GridEncoding::ForwardContext&>(*grid_ctx);
    
    GPUMatrixDynamic<float> normals{3, batch_size, stream, AoS};
    
    // Normalize gradients: n = -∇density / ||∇density||
    linear_kernel(normalize_grid_gradients_kernel, 0, stream,
        batch_size,
        forward_ctx.dy_dx.data(),  // 3 × batch_size gradients
        normals.data()
    );
    
    return normals;
}

/**
 * @brief Kernel to normalize grid gradients to unit normals
 */
__global__ void normalize_grid_gradients_kernel(
    const uint32_t n_elements,
    const float* __restrict__ dy_dx,    // 3 × n_elements
    float* __restrict__ normals         // 3 × n_elements
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    
    // Read gradients (∂density/∂x, ∂density/∂y, ∂density/∂z)
    float gx = dy_dx[i * 3 + 0];
    float gy = dy_dx[i * 3 + 1];
    float gz = dy_dx[i * 3 + 2];
    
    // Normalize: n = -∇density / ||∇density||
    float norm = sqrtf(gx * gx + gy * gy + gz * gz);
    if (norm > 1e-6f) {
        normals[i * 3 + 0] = -gx / norm;
        normals[i * 3 + 1] = -gy / norm;
        normals[i * 3 + 2] = -gz / norm;
    } else {
        // Default normal if gradient is too small
        normals[i * 3 + 0] = 0.0f;
        normals[i * 3 + 1] = 0.0f;
        normals[i * 3 + 2] = 1.0f;
    }
}

/**
 * @brief Replace first channel of output with grid density
 */
template <typename T>
__global__ void replace_first_channel_kernel(
    const uint32_t n_elements,
    const T* __restrict__ grid_density,  // 1 × n_elements
    T* __restrict__ output,              // stride × n_elements
    uint32_t stride
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    
    output[i * stride] = grid_density[i];
}
```

---

### Step 4: Backward Pass - Update Grid Parameters

**File**: `include/neural-graphics-primitives/nerf_network.h`, in `backward_impl()`

**Follow surface mode backward pattern**:

```cpp
if (m_method == "surface_explicit") {
    // Backprop through surface features (same as surface mode)
    auto dL_dsurface_slice = dL_drgb_network_input.slice_rows(0, 16);
    
    GPUMatrixDynamic<float> dL_dnormals{3, batch_size, stream, AoS};
    
    linear_kernel(surface_features_slice_backward_kernel<T>, 0, stream,
        batch_size,
        dL_dsurface_slice.layout() == AoS ? dL_dsurface_slice.stride() : 1,
        dL_dsurface_slice.data(),
        forward.analytical_normals.data(),
        dL_ddensity_network_output.layout() == AoS ? dL_ddensity_network_output.stride() : 1,
        forward.density_network_output.data(),
        dL_ddensity_network_output.data(),
        dL_dnormals.data(),  // gradients w.r.t. normals (will be ignored)
        false,  // not SDF mode
        nullptr
    );
    
    // Extract gradient w.r.t. grid density (first channel)
    GPUMatrixDynamic<T> dL_dgrid_density{1, batch_size, stream, AoS};
    linear_kernel(extract_first_channel_gradient_kernel<T>, 0, stream,
        batch_size,
        dL_dsurface_slice.data(),
        dL_dgrid_density.data(),
        dL_dsurface_slice.stride()
    );
    
    // Backprop to grid parameters using tiny-cuda-nn's backward
    GPUMatrixDynamic<T> grid_density_dummy{1, batch_size, stream, AoS};  // not used
    
    m_density_grid->backward(
        stream,
        *forward.density_grid_ctx,
        input.slice_rows(0, 3),  // positions
        grid_density_dummy,       // output (not used)
        dL_dgrid_density,         // gradients
        nullptr,                  // don't need dL_dinput
        use_inference_params,
        param_gradients_mode
    );
}
```

**Add gradient extraction kernel**:
```cpp
template <typename T>
__global__ void extract_first_channel_gradient_kernel(
    const uint32_t n_elements,
    const T* __restrict__ dL_doutput,    // stride × n_elements
    T* __restrict__ dL_dfirst_channel,   // 1 × n_elements
    uint32_t stride
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    
    dL_dfirst_channel[i] = dL_doutput[i * stride];
}
```

---

### Step 5: Inference Pass

**File**: `include/neural-graphics-primitives/nerf_network.h`, in `inference_mixed_precision_impl()`

**Same pattern as forward, but simpler** (no context needed):

```cpp
if (m_method == "surface_explicit") {
    // Get density from grid
    GPUMatrixDynamic<T> grid_density{1, batch_size, stream, AoS};
    m_density_grid->inference_mixed_precision(
        stream,
        input.slice_rows(0, 3),
        grid_density,
        use_inference_params
    );
    
    // For normals, need to use forward with dy_dx
    // (inference_mixed_precision doesn't compute gradients)
    // ... implement similar to forward pass
}
```

---

### Step 6: Update ForwardContext

**Add grid context storage**:
```cpp
struct ForwardContext : public Context {
    // ... existing members ...
    
    std::unique_ptr<tcnn::Context> density_grid_ctx;  // For surface_explicit
};
```

---

## Testing

### Basic Test
```bash
# Train with explicit density grid
python scripts/run.py \
    --scene data/nerf/fox \
    --method surface_explicit \
    --network configs/nerf/base.json \
    --n_steps 10000
```

### Compare with Surface Mode
```bash
# Compare explicit vs autodiff
python scripts/compare_methods.py \
    --method1 surface \
    --method2 surface_explicit \
    --scene data/nerf/fox
```

---

## Config Example

```json
{
  "network": {
    "otype": "FullyFusedMLP",
    "n_neurons": 64,
    "n_hidden_layers": 1,
    "n_output_dims": 48,
    "grid_resolution": 128
  }
}
```

---

## Key Differences from Surface Mode

| Aspect | Surface Mode | Surface Explicit |
|--------|-------------|------------------|
| Density source | MLP channel 0 | Dense grid encoding |
| Normal computation | Autodiff through MLP | Grid dy/dx |
| MLP output | 48D (1 density + 47 features) | 48D (all features, density ignored) |
| Gradient flow | Through density network | Directly to grid params |
| Memory | MLP params only | MLP params + grid params |

---

## Benefits

1. ✅ **No custom wrapper** - uses tiny-cuda-nn API directly
2. ✅ **Consistent pattern** - same as `m_pos_encoding`, `m_dir_encoding`
3. ✅ **Simpler gradients** - no autodiff complexity
4. ✅ **Built-in dy/dx** - analytical normals from grid interpolation
5. ✅ **Memory efficient** - 128³ × 2 bytes (fp16) = 4MB
6. ✅ **Easy to extend** - can swap Dense for Tiled or Hash later

---

## Summary

**You're not creating a new abstraction** - you're just using tiny-cuda-nn's existing Grid encoding with `type: Dense` instead of `type: Hash`. The entire implementation is:
1. Create encoding with `create_encoding<T>(3, dense_config, 1)`
2. Use `forward()` to get density and `dy_dx` for normals
3. Use `backward()` to update grid parameters
4. Follow exact same pattern as surface mode, just swap density source

**That's it!** No wrapper classes, no new interfaces, just using what's already there.
