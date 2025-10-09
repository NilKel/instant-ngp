# NeRF Network Refactoring

This directory contains the refactored NeRF network implementations, split by rendering mode for better maintainability.

## Structure

```
nerf_networks/
├── nerf_network_base.h         # Base class with common functionality
├── nerf_network_factory.h      # Factory to create mode-specific networks
├── baseline_network.h          # Standard NeRF (baseline mode)
├── surface_network.h           # Surface rendering with analytical normals
├── surface_normal_network.h    # Surface + encoded normals
├── surface_reflect_network.h   # Surface + reflection vectors
├── surface_explicit_network.h  # Surface with explicit density grid
├── baseline_explicit_network.h # Baseline with explicit density grid
├── hash_surface_network.h      # Hash-based surface features
├── volume_network.h            # Volume rendering with divergences
└── README.md                   # This file
```

## Usage

Instead of using `nerf_network.h` directly, include `nerf_network_factory.h`:

```cpp
#include <neural-graphics-primitives/nerf_networks/nerf_network_factory.h>

// Create network based on method string
auto network = create_nerf_network<T>(
    n_pos_dims, n_dir_dims, n_extra_dims, dir_offset,
    pos_encoding, dir_encoding, density_network, rgb_network,
    method,  // "baseline", "surface", "hash_surface", etc.
    use_sdf
);
```

## Modes

| Mode | File | Description |
|------|------|-------------|
| `baseline` | `baseline_network.h` | Standard NeRF rendering |
| `surface` | `surface_network.h` | Surface features from analytical normals |
| `surface_normal` | `surface_normal_network.h` | Surface + encoded normals input to RGB network |
| `surface_reflect` | `surface_reflect_network.h` | Surface + encoded reflection vectors |
| `surface_explicit` | `surface_explicit_network.h` | Surface with learnable density grid |
| `baseline_explicit` | `baseline_explicit_network.h` | Baseline with learnable density grid |
| `hash_surface` | `hash_surface_network.h` | Compact hash-based surface features |
| `volume` | `volume_network.h` | Volume features from divergences |

## Common Functionality (Base Class)

The `NerfNetworkBase` class contains:
- Parameter management (`set_params`, `n_params`, `initialize_params`)
- Encoding accessors (`pos_encoding()`, `dir_encoding()`)
- Network accessors (`density_network()`, `rgb_network()`)
- Common helper methods (variance monitoring for SDF mode)
- Virtual interface for mode-specific implementations

## Mode-Specific Implementations

Each derived class implements:
- `forward_impl()` - Forward pass with context
- `backward_impl()` - Backward pass with gradients
- `inference_mixed_precision_impl()` - Inference-only forward pass
- Mode-specific helper methods (normal computation, divergences, etc.)

## Migration from Old Code

The original `nerf_network.h` is kept as a backup. To migrate:

1. Replace `#include <neural-graphics-primitives/nerf_network.h>`
2. With `#include <neural-graphics-primitives/nerf_networks/nerf_network_factory.h>`
3. Use factory function instead of direct constructor

## Benefits

1. **Maintainability**: Each mode is self-contained (~300-500 lines vs 2900+ lines)
2. **Clarity**: No conditional logic scattered throughout
3. **Testing**: Can test each mode independently
4. **Performance**: Compiler can optimize each mode separately
5. **Extensibility**: Easy to add new modes without touching existing code



