# Migration Guide: From Monolithic to Modular NeRF Networks

## Quick Start

### Old Code (nerf_network.h):
```cpp
#include <neural-graphics-primitives/nerf_network.h>

// Direct constructor call
auto network = std::make_shared<NerfNetwork<T>>(
    n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
    pos_encoding, dir_encoding, density_network, rgb_network,
    method,  // "baseline", "surface", etc.
    use_sdf
);
```

### New Code (factory pattern):
```cpp
#include <neural-graphics-primitives/nerf_networks/nerf_network_factory.h>

// Factory function call
auto network = create_nerf_network<T>(
    n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
    pos_encoding, dir_encoding, density_network, rgb_network,
    method,  // "baseline", "surface", etc.
    use_sdf
);
```

**That's it!** The factory returns a `std::shared_ptr<NerfNetworkBase<T>>` which has the same interface as the old `NerfNetwork<T>`.

## Detailed Migration Steps

### Step 1: Update Includes

**Before:**
```cpp
#include <neural-graphics-primitives/nerf_network.h>
```

**After:**
```cpp
#include <neural-graphics-primitives/nerf_networks/nerf_network_factory.h>
```

### Step 2: Update Network Creation

**Before:**
```cpp
std::shared_ptr<NerfNetwork<T>> m_network;

// In constructor/setup:
m_network = std::make_shared<NerfNetwork<T>>(
    3, 3, 0, 3,  // dims and offsets
    pos_enc_config, dir_enc_config,
    density_mlp_config, rgb_mlp_config,
    "surface",  // method
    false  // use_sdf
);
```

**After:**
```cpp
std::shared_ptr<NerfNetworkBase<T>> m_network;  // Changed type

// In constructor/setup:
m_network = create_nerf_network<T>(
    3, 3, 0, 3,  // dims and offsets
    pos_enc_config, dir_enc_config,
    density_mlp_config, rgb_mlp_config,
    "surface",  // method
    false  // use_sdf
);
```

### Step 3: Update Type Declarations

**Before:**
```cpp
using NetworkType = NerfNetwork<precision_t>;
std::unique_ptr<NerfNetwork<T>> network_ptr;
```

**After:**
```cpp
using NetworkType = NerfNetworkBase<precision_t>;
std::unique_ptr<NerfNetworkBase<T>> network_ptr;
```

## Common Code Locations to Update

### 1. Testbed Class (src/testbed.cpp or similar)

Look for:
```cpp
std::shared_ptr<NerfNetwork<network_precision_t>> m_nerf_network;
```

Change to:
```cpp
std::shared_ptr<NerfNetworkBase<network_precision_t>> m_nerf_network;
```

And update creation:
```cpp
// Old:
m_nerf_network = std::make_shared<NerfNetwork<network_precision_t>>(...);

// New:
m_nerf_network = create_nerf_network<network_precision_t>(...);
```

### 2. Network Configuration Loading

If you have code that creates networks from config files:

```cpp
json config = load_network_config("config.json");
std::string method = config.value("method", "baseline");

// Old:
auto net = std::make_shared<NerfNetwork<T>>(..., method, ...);

// New:
auto net = create_nerf_network<T>(..., method, ...);
```

### 3. Python Bindings (if using pybind11)

```cpp
// Old:
py::class_<NerfNetwork<T>, std::shared_ptr<NerfNetwork<T>>>(m, "NerfNetwork")
    .def(py::init<...>())
    ...;

// New:
py::class_<NerfNetworkBase<T>, std::shared_ptr<NerfNetworkBase<T>>>(m, "NerfNetwork")
    // No need for py::init, use factory in Python:
    .def_static("create", &create_nerf_network<T>, ...)
    ...;
```

## Interface Compatibility

The `NerfNetworkBase<T>` class provides the same public interface as the old `NerfNetwork<T>`:

### Public Methods (unchanged):
- `forward()` / `forward_impl()`
- `backward()` / `backward_impl()`
- `inference_mixed_precision()` / `inference_mixed_precision_impl()`
- `density()`
- `set_params()` / `set_params_impl()`
- `initialize_params()`
- `n_params()`
- `padded_output_width()`
- `input_width()`
- `output_width()`
- All encoding/network accessors

### Configuration Methods (unchanged):
- `set_backprop_normals()`
- `set_use_analytical_normals()`
- `set_use_eikonal_loss()`
- `set_eikonal_weight()`
- `set_normalize_normals()`
- `set_clamp_gradients()`
- `set_max_gradient_magnitude()`

## Gradual Migration Strategy

You can migrate gradually by keeping both old and new code:

```cpp
// Keep old include for now
#include <neural-graphics-primitives/nerf_network.h>
// Add new include
#include <neural-graphics-primitives/nerf_networks/nerf_network_factory.h>

// Use compile-time flag to switch
#ifdef USE_NEW_NETWORK
    auto network = create_nerf_network<T>(...);
#else
    auto network = std::make_shared<NerfNetwork<T>>(...);
#endif
```

Then test with `USE_NEW_NETWORK` defined, and once verified, remove the old code.

## Troubleshooting

### Issue: "NerfNetwork was not declared in this scope"

**Solution:** Change `NerfNetwork` to `NerfNetworkBase` or update include to factory header.

### Issue: "undefined reference to NerfNetwork<T>::NerfNetwork"

**Solution:** You're trying to use the old constructor. Use `create_nerf_network<T>()` instead.

### Issue: Method X is not supported

**Solution:** Check `get_supported_methods()` for currently implemented modes. Not all modes are implemented yet. The factory will fall back to baseline with a warning.

```cpp
auto supported = get_supported_methods();
if (is_method_supported(my_method)) {
    auto net = create_nerf_network<T>(..., my_method, ...);
} else {
    // Handle unsupported method
}
```

### Issue: Link errors with specific modes

**Solution:** Make sure to include the mode-specific header if you're accessing mode-specific functionality:

```cpp
// For mode-specific features, include the specific network:
#include <neural-graphics-primitives/nerf_networks/surface_network.h>

// Then you can cast if needed:
auto surface_net = std::dynamic_pointer_cast<SurfaceNetwork<T>>(network);
if (surface_net) {
    // Access surface-specific methods
}
```

## Testing Your Migration

### 1. Compile Test
```bash
cd build
cmake ..
make -j8
```

### 2. Runtime Test
```bash
# Test with baseline
./instant-ngp --scene data/nerf/lego --method baseline

# Test with surface
./instant-ngp --scene data/nerf/lego --method surface
```

### 3. Verification
- Check that training works
- Verify inference produces same results
- Confirm mesh extraction works (if applicable)
- Test snapshot save/load

## Benefits of Migration

### For Developers:
- **Easier to understand**: Each mode in ~300-500 lines vs 2900+ lines
- **Easier to debug**: Isolated mode logic
- **Easier to modify**: Change one mode without affecting others
- **Better IDE support**: Faster autocomplete, go-to-definition

### For Users:
- **Same interface**: No changes to command-line arguments
- **Same performance**: Each mode optimized independently
- **More reliable**: Isolated testing per mode
- **Easier to experiment**: Can try different modes easily

## Rollback Plan

If you need to rollback:

1. The original `nerf_network.h` is kept as a backup
2. Change includes back to:
   ```cpp
   #include <neural-graphics-primitives/nerf_network.h>
   ```
3. Change types back to `NerfNetwork<T>`
4. Change creation back to `std::make_shared<NerfNetwork<T>>(...)`

## Support

If you encounter issues:

1. Check the `REFACTORING_ANALYSIS.md` for details on the refactoring
2. Consult the `README.md` in `nerf_networks/` for architecture overview
3. Compare your code against the examples in this guide
4. Check if your mode is implemented (see factory function)

## Next Steps

Once migration is complete:

1. Consider implementing additional modes (see `REFACTORING_ANALYSIS.md`)
2. Extract remaining kernels to `nerf_kernels.h`
3. Add mode-specific unit tests
4. Profile and optimize individual modes
5. Eventually remove the old `nerf_network.h` backup



