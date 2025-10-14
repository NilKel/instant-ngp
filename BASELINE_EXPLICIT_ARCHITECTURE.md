# baseline_explicit Architecture Summary

## Overview
`baseline_explicit` combines an explicit learnable density grid with MLP-based features and RGB prediction, following Plenoxels-style density handling.

## Network Components

### 1. **Explicit Density Grid**
- **Type**: `DenseGrid` encoding (128³ resolution)
- **Features**: 1D density per voxel
- **Initialization**: Uniform 0.1 (configurable via `init_density`)
- **Optimizer**: Separate Adam optimizer with aggressive LR schedule (30.0 → 0.05)

### 2. **Density MLP**
- **Input**: Position encoding (typically 32D)
- **Output**: 15D features (for RGB MLP conditioning)
- **Purpose**: Provides geometric features, NOT density prediction

### 3. **RGB MLP**
- **Input**: `[grid_density(1D), mlp_features(15D), encoded_direction(16D)]` = 32D
- **Output**: 3D RGB
- **Purpose**: Color prediction conditioned on density and features

## Forward Pass Architecture

### Step 1: Position Encoding
```cpp
pos_encoded = PositionEncoding(input_positions)  // 3D → 32D
```

### Step 2: Density Grid Interpolation
```cpp
grid_density_raw = DenseGrid.interpolate(input_positions)  // 3D → 1D raw density
```

### Step 3: Density MLP (Features Only)
```cpp
mlp_features = DensityMLP(pos_encoded)  // 32D → 15D features
```

### Step 4: RGB MLP Input Construction
```cpp
// Apply ReLU to grid density (Plenoxels approach)
grid_density_relu = ReLU(grid_density_raw)

// RGB MLP input: [density, features, direction]
rgb_input = [
    grid_density_relu,           // 1D: ReLU'd density from grid
    mlp_features[0:14],          // 15D: features from density MLP  
    direction_encoding           // 16D: encoded viewing direction
]  // Total: 32D
```

### Step 5: RGB Prediction
```cpp
rgb_output = RGBMLP(rgb_input)  // 32D → 3D RGB
```

### Step 6: Final Output
```cpp
network_output = [
    rgb_output[0],               // R
    rgb_output[1],               // G  
    rgb_output[2],               // B
    grid_density_relu           // Density (for alpha blending)
]  // 4D total
```

## Backward Pass Architecture

### Step 1: RGB Gradients
```cpp
// Extract RGB gradients from final output
dL_drgb = extract_rgb_gradients(dL_doutput)  // 3D gradients
```

### Step 2: RGB MLP Backward
```cpp
// Backward through RGB MLP
dL_drgb_input = RGBMLP.backward(
    rgb_input, rgb_output, dL_drgb
)  // 32D gradients to RGB input
```

### Step 3: Split RGB Input Gradients
```cpp
dL_dgrid_density = dL_drgb_input[0]           // 1D: gradient to grid density
dL_dmlp_features = dL_drgb_input[1:15]        // 15D: gradients to MLP features  
dL_ddirection = dL_drgb_input[16:31]          // 16D: gradients to direction
```

### Step 4: ReLU Chain Rule for Grid Density
```cpp
// Apply ReLU chain rule: gradient = 0 if input ≤ 0
dL_dgrid_density_raw = dL_dgrid_density * (grid_density_raw > 0 ? 1 : 0)
```

### Step 5: Grid Density Gradient Accumulation
```cpp
// Accumulate gradients to density grid
accumulate_density_gradient_to_grid(
    dL_dgrid_density_raw, grid_density_raw,  // Forward values for chain rule
    apply_relu_chain_rule=true
)
```

### Step 6: Density MLP Backward
```cpp
// Backward through density MLP (features only)
dL_dpos_encoded = DensityMLP.backward(
    pos_encoded, mlp_features, dL_dmlp_features
)
```

### Step 7: Position Encoding Backward
```cpp
// Backward through position encoding
dL_dinput_positions = PositionEncoding.backward(
    input_positions, pos_encoded, dL_dpos_encoded
)
```

## Key Implementation Details

### ReLU Application Points
1. **Grid → RGB MLP**: `ReLU(grid_density)` passed to RGB input[0]
2. **Grid → Output**: `ReLU(grid_density)` passed to output[3] for alpha blending
3. **Backward**: ReLU chain rule applied in both gradient paths

### Alpha Blending Formula
```cpp
// In rendering kernels (requires density_activation = None)
alpha = 1.0 - exp(-ReLU(grid_density) * dt)
```

### Optimizer Configuration
```json
{
    "density_grid_optimizer": {
        "otype": "ExponentialDecay",
        "decay_start": 3000,
        "decay_interval": 1,
        "learning_rate_start": 30.0,
        "learning_rate_end": 0.05,
        "total_training_steps": 50000,
        "nested": {
            "otype": "Adam",
            "learning_rate": 30.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-15
        }
    }
}
```

## Critical Runtime Settings

### Python Configuration
```python
# CRITICAL: Set density activation to None
testbed.nerf.density_activation = tcnn.cpp.ENerfActivation.None
```

### Why This Matters
- **Network applies ReLU**: `ReLU(grid_density)`
- **Rendering should NOT apply additional activation**: `network_to_density(density, None)`
- **Final formula**: `α = 1 - exp(-ReLU(density) * dt)`

## Data Flow Summary

### Forward
```
Input(3D) → PosEnc(32D) → DensityMLP(15D) → Features
     ↓
Grid(1D) → ReLU → RGB_Input[0]
     ↓
Dir(3D) → DirEnc(16D) → RGB_Input[16:31]
     ↓
RGB_Input[32D] → RGBMLP → RGB(3D)
     ↓
Output[4D] = [RGB(3D), ReLU(GridDensity)(1D)]
```

### Backward
```
dL_dOutput[4D] → dL_dRGB(3D) + dL_dDensity(1D)
     ↓
RGBMLP.backward → dL_dRGB_Input[32D]
     ↓
Split: dL_dGridDensity(1D) + dL_dFeatures(15D) + dL_dDir(16D)
     ↓
ReLU_Chain_Rule → dL_dGridRaw(1D) → Grid.update()
     ↓
DensityMLP.backward → dL_dPosEnc(32D) → PosEnc.backward → dL_dInput(3D)
```

## Comparison with Plenoxels

| Aspect | Plenoxels | baseline_explicit |
|--------|-----------|-------------------|
| **Density Storage** | Explicit grid | Explicit grid |
| **Density Activation** | ReLU | ReLU |
| **Features** | Spherical harmonics | MLP features |
| **RGB Prediction** | SH coefficients | MLP |
| **Learning Rate** | 30.0 → 0.05 | 30.0 → 0.05 |
| **Initialization** | 0.1 | 0.1 |

## Potential Issues

1. **Double Activation**: Ensure `density_activation = None`
2. **Learning Rate**: Grid LR much higher than MLP LR
3. **Gradient Flow**: ReLU chain rule must be applied correctly
4. **Initialization**: Uniform 0.1 vs random initialization
