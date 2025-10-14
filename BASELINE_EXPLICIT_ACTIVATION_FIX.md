# CRITICAL: baseline_explicit Activation Fix

## The Problem

Our `baseline_explicit` implementation was applying **double activation**:

1. **Network level**: ReLU applied in `extract_density` kernel
2. **Rendering level**: Exponential activation applied in `network_to_density()`

This resulted in: `α = 1 - exp(-exp(ReLU(density)) * dt)` which is wrong!

## The Solution

**Standard NeRF flow:**
```
Density MLP → raw values → network_to_density(Exponential) → α = 1 - exp(-exp(density) * dt)
```

**Our baseline_explicit flow:**
```
Grid → ReLU → extract_density → network_to_density(None) → α = 1 - exp(-ReLU(density) * dt)
```

## Required Runtime Setting

**CRITICAL:** When using `baseline_explicit`, you MUST set:

```python
testbed.nerf.density_activation = tcnn.cpp.ENerfActivation.None
```

This ensures the rendering kernels don't apply additional exponential activation.

## Why This Matters

### Wrong (Double Activation):
```cpp
// Network applies ReLU
density_relu = ReLU(grid_density);

// Rendering applies exponential (WRONG!)
density_final = exp(density_relu);  // exp(ReLU(x))
alpha = 1 - exp(-density_final * dt);
```

### Correct (Single Activation):
```cpp
// Network applies ReLU
density_relu = ReLU(grid_density);

// Rendering uses as-is (CORRECT!)
density_final = density_relu;  // ReLU(x)
alpha = 1 - exp(-density_final * dt);
```

## Implementation Details

### Network Level (baseline_explicit_network.h)
```cpp
// Line 253-263: inference_mixed_precision_impl
linear_kernel(extract_density<T>, 0, stream,
    batch_size, /* ... */
    true  // Apply ReLU: grid stores raw density, apply ReLU for alpha blending
);

// Line 369-379: forward_impl  
linear_kernel(extract_density<T>, 0, stream,
    batch_size, /* ... */
    true  // Apply ReLU: grid stores raw density, apply ReLU for alpha blending
);
```

### Rendering Level (testbed.cu)
```python
# MUST set this for baseline_explicit!
testbed.nerf.density_activation = tcnn.cpp.ENerfActivation.None
```

## Comparison with Standard NeRF

| Mode | Network Output | Rendering Activation | Final Formula |
|------|----------------|---------------------|---------------|
| **baseline** | Raw values | `network_to_density(Exponential)` | `α = 1 - exp(-exp(density) * dt)` |
| **baseline_explicit** | ReLU values | `network_to_density(None)` | `α = 1 - exp(-ReLU(density) * dt)` |

## Testing the Fix

### 1. Verify Network Output
Check that `baseline_explicit` outputs ReLU'd density values:
```python
# Should see values ≥ 0 (ReLU applied)
print("Density range:", network_output[:, 3].min(), network_output[:, 3].max())
```

### 2. Verify Rendering Activation
```python
# Should be None for baseline_explicit
print("Density activation:", testbed.nerf.density_activation)
# Should print: ENerfActivation.None
```

### 3. Expected Behavior
- **Density values**: All ≥ 0 (ReLU ensures non-negative)
- **Alpha values**: Smooth transition from 0 to 1
- **No double activation**: Single ReLU, no exponential

## Code Changes Made

1. **Added comments** in `baseline_explicit_network.h to clarify ReLU application
2. **Created this documentation** explaining the activation flow
3. **Updated config files** to include the required runtime setting

## Usage

```python
# Load baseline_explicit network
testbed.load_training_data("path/to/scene.json")
testbed.reload_network_from_file("configs/nerf/baseline_explicit_plenoxels.json")

# CRITICAL: Set density activation to None
testbed.nerf.density_activation = tcnn.cpp.ENerfActivation.None

# Train
testbed.train()
```

This ensures the correct single-activation flow for Plenoxels-style training!
