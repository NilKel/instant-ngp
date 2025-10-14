# Plenoxels-Style Updates to baseline_explicit

## Summary

Updated `baseline_explicit_network.h` to implement Plenoxels-style density handling with ReLU activation and aggressive learning rates for the explicit density grid.

## Key Changes

### 1. Density Activation: ReLU Instead of Exponential

**Before:** Grid stored log-density, exponential applied during rendering
**After:** Grid stores raw density, ReLU applied before RGB MLP and alpha blending

#### Forward Pass
- `replace_first_channel_kernel`: Now applies `ReLU(density)` when copying to RGB input
- `extract_density`: Now applies `ReLU(density)` when extracting to output for alpha blending
- Density is conditioned through RGB MLP as `ReLU(grid_density)`
- Alpha blending uses `α = 1 - exp(-ReLU(density) * dt)`

#### Backward Pass
- `extract_first_channel_gradient_kernel`: Applies ReLU chain rule (gradient = 0 when input ≤ 0)
- `accumulate_density_gradient_to_grid_kernel`: Applies ReLU chain rule when accumulating gradients

### 2. Grid Initialization

**Before:** Random initialization around -5.0
**After:** Uniform initialization at 0.1 (Plenoxels default)

```cpp
float init_density = 0.1f;  // Configurable via "init_density" in config
```

### 3. Aggressive Learning Rate Schedule (Plenoxels Paper)

**Density Grid Optimizer:**
- **Initial LR:** 30.0 (3000x higher than typical NeRF)
- **Decay:** From 30.0 to 0.05 over 235,000 steps (after 15k delay)
- **Schedule:** Exponential decay every step
- **Optimizer:** Adam with β₁=0.9, β₂=0.999

**Calculation:**
```cpp
decay_base = pow(0.05 / 30.0, 1.0 / 235000.0) ≈ 0.9999727794
```

**Config:**
```json
"density_grid_optimizer": {
    "otype": "ExponentialDecay",
    "decay_start": 15000,        // Initial delay period
    "decay_interval": 1,          // Decay every step
    "decay_base": 0.9999727794,   // 30→0.05 over 235k steps
    "nested": {
        "otype": "Adam",
        "learning_rate": 30.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-15
    }
}
```

### 4. Required Runtime Settings

**IMPORTANT:** When training with baseline_explicit, set:
- `density_activation = ENerfActivation::None` 
  - The network already applies ReLU, so no additional activation is needed
  - The rendering kernels should receive ReLU'd density directly

**In Python/training code:**
```python
testbed.nerf.density_activation = tcnn.cpp.ENerfActivation.None
```

## Architecture Flow

### Forward Pass
1. **Grid Interpolation:** `raw_density = interpolate(grid, position)`
2. **ReLU Activation:** `density = ReLU(raw_density)`
3. **RGB MLP Input:** `[density, mlp_features(15D), encoded_direction(16D)]`
4. **Alpha Blending:** `α = 1 - exp(-density * dt)`

### Backward Pass  
1. **Gradients from RGB MLP:** `dL/d(density)` from RGB input[0]
2. **Gradients from alpha:** `dL/d(density)` from output[3]
3. **ReLU Chain Rule:** `dL/d(raw_density) = dL/d(density) * (1 if raw_density > 0 else 0)`
4. **Grid Update:** Accumulate gradients with ReLU gating

## Usage

### Config File
Use `configs/nerf/baseline_explicit_plenoxels.json`:
```json
{
    "network": {
        "otype": "FullyFusedMLP",
        "explicit_grid_resolution": 128,
        "init_density": 0.1,
        "density_grid_optimizer": { /* aggressive LR */ }
    }
}
```

### Training Command
```bash
python scripts/run.py \
    --scene data/nerf_synthetic/lego \
    --network configs/nerf/baseline_explicit_plenoxels.json \
    --method baseline_explicit \
    --n_steps 50000
```

### Expected Behavior
- **High initial LR (30.0)** helps density grid escape uniform initialization quickly
- **ReLU activation** makes density easier to sparsify (can be exactly zero)
- **Grid stores raw density** (not log-density) for better gradient flow
- **Density conditions RGB MLP** for view-dependent effects

## Comparison with Standard NeRF

| Aspect | Standard NeRF | Plenoxels baseline_explicit |
|--------|---------------|----------------------------|
| Density representation | MLP output | Explicit grid |
| Activation | Exponential | ReLU |
| Grid stores | Log-density | Raw density |
| Learning rate (density) | 0.01 | 30.0 → 0.05 |
| Optimizer | Adam | Adam with aggressive decay |
| Initialization | Random | Uniform 0.1 |
| Sparsification | Via activation | Via ReLU (exact zeros) |

## References

- Plenoxels paper: "Plenoxels: Radiance Fields without Neural Networks" (Yu et al., CVPR 2022)
- Learning rates from Plenoxels supplementary material
- ReLU activation strategy from original Plenoxels codebase

