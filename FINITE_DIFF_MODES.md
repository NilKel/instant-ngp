# Finite Difference Normal Modes

## Overview
Three finite difference modes are now available for computing normals from density grids in surface rendering:

## Modes

### 1. `--grad finite` (Standard FD)
**Command**: `python scripts/run.py --grad finite --method surface`

**Forward**: 
- Samples density at positions ± ε
- Computes gradient: `∇ρ = (ρ(x+ε) - ρ(x-ε)) / (2ε)`
- Normalizes to get normal: `n = -∇ρ / ||∇ρ||`

**Backward**:
- ✅ Gradients flow through finite differences to density grid
- Uses saved contexts from forward pass
- Chain rule: `dL/dρ(x±ε) = ±dL/d(∇ρ) / (2ε)`

**Use case**: Standard differentiable normal computation

---

### 2. `--grad finite_thresh` (Binary Threshold - NON-differentiable)
**Command**: `python scripts/run.py --grad finite_thresh --density-threshold 0.01`

**Forward**:
- Applies binary threshold: `ρ_binary = (ρ > threshold) ? 1.0 : 0.0`
- Computes FD on binary density: `∇ρ_binary`
- Normalizes to get normal

**Backward**:
- ❌ NO gradients flow from normals to density (step function has zero derivative)
- Density only learns from direct alpha blending path
- Useful for: Binary occupancy experiments, non-differentiable density analysis

**Use case**: Binary occupancy grid experiments (no gradient flow from normals)

---

### 3. `--grad finite_sigm` (Sigmoid - Differentiable)  ⭐ NEW
**Command**: `python scripts/run.py --grad finite_sigm --method surface`

**Forward**:
- Applies sigmoid: `ρ_sigmoid = σ(ρ) = 1/(1 + exp(-ρ))`
- Computes FD on sigmoid density: `∇ρ_sigmoid`
- Normalizes to get normal

**Backward**:
- ✅ Gradients flow through sigmoid chain rule to density grid
- Chain rule: `dL/dρ_raw = dL/d(∇ρ_sigmoid) · ∇σ'(ρ_raw)`
- Sigmoid derivative: `σ'(x) = σ(x) · (1 - σ(x))`

**Gradient Flow**:
```
dL/dRGB → dL/d(surface_features) → dL/d(normals) → dL/d(∇ρ_sigmoid) 
  → dL/d(ρ_sigmoid) → dL/d(ρ_raw) [via σ'(ρ_raw)] → density grid updates ✓
```

**Use case**: Smooth, bounded density [0,1] with full gradient flow

---

## Comparison Table

| Mode | Transform | Differentiable? | Gradient Flow | Output Range |
|------|-----------|-----------------|---------------|--------------|
| `finite` | None | ✅ Yes | Full | [-∞, +∞] |
| `finite_thresh` | `ρ > T ? 1 : 0` | ❌ No | None from normals | {0, 1} |
| `finite_sigm` | `1/(1+exp(-ρ))` | ✅ Yes | Full (via chain rule) | [0, 1] |

---

## Implementation Details

### Forward Pass
**File**: `include/neural-graphics-primitives/nerf_helpers.h`

```cpp
// Line 1811-1815: Mode detection
bool use_finite_diff = (grad_method_env && (std::string(grad_method_env) == "finite" || 
                                             std::string(grad_method_env) == "finite_thresh" ||
                                             std::string(grad_method_env) == "finite_sigm"));
bool use_threshold = (grad_method_env && std::string(grad_method_env) == "finite_thresh");
bool use_sigmoid = (grad_method_env && std::string(grad_method_env) == "finite_sigm");

// Line 1435-1444: Kernel applies transform
if (apply_sigmoid) {
    density_plus = 1.0f / (1.0f + expf(-density_plus));
    density_minus = 1.0f / (1.0f + expf(-density_minus));
}
else if (apply_threshold) {
    density_plus = (density_plus > threshold) ? 1.0f : 0.0f;
    density_minus = (density_minus > threshold) ? 1.0f : 0.0f;
}
```

### Backward Pass
**File**: `include/neural-graphics-primitives/nerf_helpers.h`

```cpp
// Line 2053-2054: Sigmoid mode detection
const char* grad_method_env = std::getenv("NGP_GRAD_METHOD");
bool use_sigmoid = (grad_method_env && std::string(grad_method_env) == "finite_sigm");

// Line 1795-1802: Sigmoid chain rule in kernel
if (apply_sigmoid && density_raw_sample) {
    float raw_density = float(density_raw_sample[i * density_stride]);
    float sigmoid_val = 1.0f / (1.0f + expf(-raw_density));
    float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
    grad *= sigmoid_deriv;  // Apply chain rule
}
```

**Files Updated**:
- `surface_explicit_network.h` (line 427-428): Backward check includes `finite_sigm`
- `nerf_network.h` (line 1446-1447): Backward check includes `finite_sigm`

---

## Example Usage

### Standard FD (full gradients)
```bash
python scripts/run.py \
  --scene data/nerf_synthetic/drums/transforms_train.json \
  --network configs/nerf/base.json \
  --method surface \
  --grad finite \
  --genus 1
```

### Binary Threshold (no normal gradients)
```bash
python scripts/run.py \
  --scene data/nerf_synthetic/drums/transforms_train.json \
  --network configs/nerf/base.json \
  --method surface \
  --grad finite_thresh \
  --density-threshold 0.01
```

### Sigmoid (bounded + full gradients)
```bash
python scripts/run.py \
  --scene data/nerf_synthetic/drums/transforms_train.json \
  --network configs/nerf/base.json \
  --method surface \
  --grad finite_sigm \
  --genus 1
```

---

## When to Use Each Mode

**`finite`**: Default choice for most cases
- Full gradient flow
- No bias from nonlinearity
- Works with any density range

**`finite_thresh`**: For binary occupancy analysis
- Experiment with non-differentiable normals
- Test if normals should influence density learning
- Density only learns from alpha blending

**`finite_sigm`**: For smooth, bounded density
- Constrains density to [0,1] range
- Full gradient flow with smooth nonlinearity
- May help with stability/convergence
- Useful when raw density is unbounded

---

## Verification

To verify gradients are flowing:
1. Monitor density grid changes during training
2. Compare loss convergence across modes
3. Visualize learned density distributions

Expected behavior:
- `finite` and `finite_sigm`: Density grid updates from normal gradients
- `finite_thresh`: Density grid only updates from alpha blending (direct output)

