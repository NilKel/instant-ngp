# NeRF Network Refactoring Analysis

## What's Currently in nerf_helpers.h

Based on the outline, `nerf_helpers.h` already contains:
- `extract_density` kernel (~line 44-49)
- `compute_divergence_diagonal_kernel` (~line 387)
- `extract_spatial_gradient_kernel` (~line 414)
- `offset_positions_kernel` (~line 1322)
- `DaubechiesStencil` struct (~line 1345)
- `compute_finite_difference_stencil_kernel` (~line 1395)
- `compute_finite_difference_kernel` (~line 1430)
- `normalize_grid_gradients_kernel` (~line 1454)
- `backprop_normalization_kernel` (~line 1672)
- `backprop_finite_difference_kernel` (~line 1722)
- `backprop_daubechies_stencil_kernel` (~line 1753)

## What Should Stay in nerf_network.h (Base/Mode-Specific Classes)

### Keep in Base Class:
1. **Parameter management** - `set_params_impl()`, `initialize_params()`, `n_params()`
2. **Network accessors** - `pos_encoding()`, `dir_encoding()`, `density_network()`, `rgb_network()`
3. **Common configuration** - SDF settings, normal normalization, gradient clamping
4. **Virtual interface** - `forward_impl()`, `backward_impl()`, `inference_mixed_precision_impl()`

### Keep in Mode-Specific Classes:
1. **Mode-specific forward passes** - Each mode has unique data flow
2. **Mode-specific backward passes** - Different gradient routing per mode
3. **Mode-specific helper methods** - e.g., `compute_analytical_normals_forward()` in SurfaceNetwork
4. **ForwardContext extensions** - Each mode stores different intermediate values

## What Should Move to nerf_helpers.h (Kernel Functions)

These CUDA kernels and device functions should move to `nerf_helpers.h` if not already there:

### Surface Feature Kernels (Referenced in surface_network.h):
1. `compute_surface_features_to_slice_kernel<T>` - Computes surface features from Φ vectors and normals
2. `surface_features_slice_backward_kernel<T>` - Backpropagates through surface features
3. `compute_surface_features_from_vectors_kernel<T>` - For surface_explicit mode
4. `backprop_surface_features_kernel<T>` - Backprop for surface_explicit
5. `backprop_surface_features_to_normals_kernel<T>` - Gradient w.r.t. normals

### Normal Computation Kernels:
6. `set_constant_value_view_kernel<T>` - Sets a constant value to a buffer
7. `process_analytical_gradients_kernel<float>` - Normalizes/clamps gradients to get normals
8. `chain_rule_through_normalization_kernel<float>` - Backprop through normalization
9. `add_eikonal_gradients_kernel<float>` - Adds Eikonal loss gradients

### Hash-Based Kernels:
10. `extract_hash_density_features_kernel<T>` - Extracts density features from hash interpolation
11. `distribute_hash_density_gradients_kernel<T>` - Distributes gradients back to hash features
12. `compute_hash_surface_features_kernel<T>` - Computes hash surface features
13. `hash_surface_features_backward_kernel<T>` - Backprop for hash surface features
14. `accumulate_density_gradient_to_hash_kernel<T>` - Accumulates gradients to hash

### Divergence Kernels (Volume Mode):
15. `compute_volume_divergence_kernel<T>` - Computes volume features from divergences
16. `volume_divergence_backward_kernel<T>` - Backprop for volume features

### Reflection Kernels:
17. `calculate_reflection_vector_kernel<T>` - Computes reflection vectors
18. `reflection_vector_backward_kernel<T>` - Backprop through reflection

### Grid-Based Kernels (Explicit Modes):
19. `replace_first_channel_kernel<T>` - Replaces channel 0 with grid density
20. `copy_channels_kernel<T>` - Copies multiple channels
21. `copy_channels_with_src_offset_kernel<T>` - Copies channels with offset
22. `extract_first_channel_gradient_kernel<T>` - Extracts gradient from channel 0
23. `accumulate_density_gradient_to_grid_kernel<T>` - Accumulates density gradients to grid

### Utility Kernels:
24. `extract_rgb<T>` - Extracts RGB from output
25. `add_density_gradient<T>` - Adds density gradient
26. `add_to_buffer_kernel<T>` - Adds one buffer to another
27. `accumulate_second_order_gradients_kernel<T>` - Accumulates second-order gradients
28. `accumulate_sdf_gradients_kernel<T>` - Accumulates SDF gradients
29. `extract_density_backward<T>` - Backward through density extraction
30. `copy_float_to_T_kernel<T>` - Type conversion kernel
31. `copy_normal_gradients_to_pos_kernel<float>` - Copies normal gradients

## Current State

From analyzing the code:
- **✓ Already in nerf_helpers.h**: Basic kernels like `extract_density`, divergence kernels, finite difference kernels
- **✗ Still in nerf_network.h**: Many surface, hash, and grid kernels are likely embedded or need to be declared

## Recommendation

### Immediate Actions:
1. **Keep the original nerf_network.h as backup** ✓ (already done)
2. **Create nerf_network_kernels.h** - Move all CUDA kernels here (cleaner than mixing with helpers)
3. **Update nerf_helpers.h** - Add missing kernel declarations
4. **Include kernels in mode files** - Each mode includes only the kernels it needs

### File Organization:
```
include/neural-graphics-primitives/
├── nerf_network.h (BACKUP - original file)
├── nerf_helpers.h (Common utilities + basic kernels)
├── nerf_kernels.h (NEW - All CUDA kernels used by networks)
├── nerf_networks/
│   ├── README.md
│   ├── nerf_network_base.h (Base class)
│   ├── nerf_network_factory.h (Factory function)
│   ├── baseline_network.h
│   ├── surface_network.h
│   └── ... (other modes)
```

## Migration Path

### Phase 1: Core Modes (DONE)
- ✓ Create directory structure
- ✓ Create base class
- ✓ Create baseline and surface networks
- ✓ Create factory function

### Phase 2: Kernel Extraction (NEXT)
- Extract all CUDA kernels from nerf_network.h
- Create nerf_kernels.h with all kernels
- Update nerf_helpers.h if needed

### Phase 3: Remaining Modes
- Implement surface_normal_network.h
- Implement surface_reflect_network.h
- Implement surface_explicit_network.h
- Implement baseline_explicit_network.h
- Implement hash_surface_network.h
- Implement volume_network.h

### Phase 4: Integration
- Update testbed.cpp to use factory
- Update CMakeLists.txt if needed
- Test all modes
- Remove old nerf_network.h once verified

## Benefits

### Maintainability:
- Each mode: ~300-500 lines (vs 2900+ in monolithic file)
- Clear separation of concerns
- Easy to locate mode-specific logic

### Performance:
- Compiler can optimize each mode independently
- Smaller compilation units
- Better instruction cache utilization

### Extensibility:
- Add new modes without touching existing code
- Easy to experiment with variants
- Can have mode-specific optimizations

### Testing:
- Unit test each mode independently
- Easier to isolate bugs
- Can benchmark modes separately

## Kernel Count Summary

Based on analysis:
- **~31 CUDA kernels** used across all modes
- **~12 already in nerf_helpers.h**
- **~19 need to be extracted/declared**

Most kernels are mode-specific (surface, hash, explicit), which validates the refactoring approach.



