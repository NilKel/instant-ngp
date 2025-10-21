# NeRF Network Architecture Documentation

This document provides detailed architectural specifications for all 8 NeRF network modes.

---

## 1. Baseline Mode (`baseline`)

### Architecture

**Position Encoding:**
- Type: HashGrid (or configurable)
- Input: 3D position (x, y, z)
- Output: Encoded position features (typically 32D with HashGrid)

**Density MLP:**
- Input: Encoded position features
- Output: 16D (channels 0-15)
- Activation: ReLU (configurable)
- Channel 0: Density value σ

**Direction Encoding:**
- Type: SphericalHarmonics (degree 4) or Composite
- Input: 3D view direction
- Output: Encoded direction features (typically 16D)

**RGB MLP:**
- Input: Density MLP output (16D) + Encoded direction (16D) = 32D (padded)
- Output: 3D RGB
- Activation: ReLU → None (output)

**Explicit Grids:** None

### Forward Pass Flow

1. **Position Encoding**: `pos(x,y,z)` → HashGrid → `enc_pos[32D]`
2. **Density MLP**: `enc_pos[32D]` → MLP(ReLU) → `density_out[16D]`
3. **Density Extraction**: `density = density_out[0]` (channel 0)
4. **Direction Encoding**: `dir(θ,φ)` → SphericalHarmonics → `enc_dir[16D]`
5. **RGB Input Assembly**: `rgb_input = [density_out[16D], enc_dir[16D]]` (32D total)
6. **RGB MLP**: `rgb_input[32D]` → MLP(ReLU→None) → `RGB[3D]`
7. **Final Output**: `[R, G, B, σ]` (4D RGBD)

### Backward Pass Flow

1. **RGB Gradient Extraction**: Extract `dL/dRGB` from output gradients
2. **RGB MLP Backward**: 
   - Input: `dL/dRGB[3D]`
   - Output: `dL/d(rgb_input)[32D]`
   - Gradients flow to RGB MLP parameters
3. **Direction Encoding Backward**:
   - Input: `dL/d(enc_dir)` from rgb_input gradients[16:32]
   - Output: `dL/d(dir)` (optional)
   - Gradients flow to direction encoding parameters
4. **Density Gradient Extraction**:
   - Extract `dL/dσ` from output gradients[3]
   - Add to `dL/d(density_out)[0]` from rgb_input gradients[0]
5. **Density MLP Backward**:
   - Input: `dL/d(density_out)[16D]`
   - Output: `dL/d(enc_pos)[32D]`
   - Gradients flow to density MLP parameters
6. **Position Encoding Backward**:
   - Input: `dL/d(enc_pos)[32D]`
   - Gradients flow to HashGrid parameters

---

## 2. Baseline Aggregate Mode (`baseline_aggregate`)

### Architecture

**Position Encoding:**
- Type: HashGrid (or configurable)
- Input: 3D position (x, y, z)
- Output: Encoded position features (typically 32D with HashGrid)

**Density MLP:**
- Input: Encoded position features
- Output: 8D
  - Channel 0: Density value σ
  - Channels 1-3: Diffuse RGB (R_d, G_d, B_d)
  - Channels 4-7: 4D features (f0, f1, f2, f3)
- Activation: ReLU (configurable)

**Direction Encoding:**
- Type: SphericalHarmonics (degree 4) or Composite
- Input: 3D view direction
- Output: Encoded direction features (typically 16D)

**RGB MLP:**
- Input: 4D features + Encoded direction (16D) = 20D (padded to 32D)
- Output: 3D directional RGB
- Activation: ReLU → None (output)

**Explicit Grids:** None

### Forward Pass Flow

1. **Position Encoding**: `pos(x,y,z)` → HashGrid → `enc_pos[32D]`
2. **Density MLP**: `enc_pos[32D]` → MLP(ReLU) → `density_out[8D]`
3. **Per-Sample Extraction**:
   - `density = density_out[0]`
   - `diffuse_RGB = density_out[1:4]`
   - `features = density_out[4:8]`
4. **Volume Rendering (Alpha Blending)**:
   - For each sample along ray:
     - `alpha_i = 1 - exp(-density_i * dt_i)`
     - `weight_i = alpha_i * T_i`
     - `accumulated_diffuse += diffuse_RGB_i * weight_i`
     - `accumulated_features += features_i * weight_i`
5. **Per-Pixel Direction Encoding**: `dir(θ,φ)` → SphericalHarmonics → `enc_dir[16D]`
6. **Per-Pixel RGB MLP**:
   - `rgb_input = [accumulated_features[4D], enc_dir[16D]]` (20D → padded to 32D)
   - `directional_RGB = RGB_MLP(rgb_input)[3D]`
7. **Final Composition**: `final_RGB = accumulated_diffuse + directional_RGB`
8. **Final Output**: `[final_R, final_G, final_B, accumulated_alpha]` (4D RGBD)

### Backward Pass Flow

1. **Final RGB Gradient Extraction**: Extract `dL/d(final_RGB)` from output gradients
2. **Composition Backward**:
   - `dL/d(accumulated_diffuse) = dL/d(final_RGB)`
   - `dL/d(directional_RGB) = dL/d(final_RGB)`
3. **RGB MLP Backward** (per-pixel):
   - Input: `dL/d(directional_RGB)[3D]`
   - Output: `dL/d(rgb_input)[32D]`
   - Gradients flow to RGB MLP parameters
4. **Direction Encoding Backward** (per-pixel):
   - Input: `dL/d(enc_dir)` from rgb_input gradients[4:20]
   - Output: `dL/d(dir)` (optional)
   - Gradients flow to direction encoding parameters
5. **Feature Gradients Accumulation** (per-pixel → per-sample):
   - Extract: `dL/d(accumulated_features)` from rgb_input gradients[0:4]
   - Distribute to samples: `dL/d(features_i) = dL/d(accumulated_features) * weight_i`
6. **Diffuse RGB Gradients** (per-pixel → per-sample):
   - Extract: `dL/d(accumulated_diffuse)`
   - Distribute to samples: `dL/d(diffuse_RGB_i) = dL/d(accumulated_diffuse) * weight_i`
7. **Density Gradient Extraction**:
   - Extract `dL/dσ` from output gradients[3]
   - Combine with gradients from alpha blending weights
8. **Density MLP Backward** (per-sample):
   - Input: `dL/d(density_out)[8D]` (density + diffuse RGB + features)
   - Output: `dL/d(enc_pos)[32D]`
   - Gradients flow to density MLP parameters
9. **Position Encoding Backward**:
   - Input: `dL/d(enc_pos)[32D]`
   - Gradients flow to HashGrid parameters

### Key Differences from Baseline Mode

1. **Two-Stage Processing**: Baseline processes RGB per-sample, baseline_aggregate processes directional RGB per-pixel
2. **Separate Diffuse and Directional**: Diffuse color is view-independent (accumulated), directional is view-dependent (computed per-pixel)
3. **Feature Accumulation**: 4D features are accumulated during volume rendering, then processed by RGB MLP
4. **8D MLP Output**: Density MLP outputs 8D instead of 16D

---

## 3. Surface Mode (`surface`)

### Architecture

**Position Encoding:**
- Type: HashGrid
- Input: 3D position
- Output: Encoded position features (32D)

**Density MLP:**
- Input: Encoded position (32D)
- Output: 48D
  - Channel 0: Density/SDF value
  - Channels 1-46: 45D vector potential Φ (15 × 3D vectors)
- Activation: ReLU

**Direction Encoding:**
- Type: SphericalHarmonics (degree 4)
- Input: 3D view direction
- Output: 16D

**RGB MLP:**
- Input: 16D surface features + 16D encoded direction = 32D (padded)
- Output: 3D RGB
- Activation: ReLU → None

**Explicit Grids:** None

### Forward Pass Flow

1. **Position Encoding**: `pos[3D]` → HashGrid → `enc_pos[32D]`
2. **Density MLP**: `enc_pos[32D]` → MLP(ReLU) → `density_out[48D]`
3. **Analytical Normal Computation** (Autodiff):
   - Create gradient seed: `dL/d(density_out) = [1, 0, 0, ..., 0]`
   - Backward through density MLP: Get `dL/d(enc_pos)`
   - Backward through position encoding: Get `dSDF/dpos[3D]`
   - Normalize: `normals = -normalize(dSDF/dpos)`
4. **Surface Feature Computation**:
   - Reshape `density_out[1:46]` as 15 vectors `Φ_k[3D]` (k=0..14)
   - For each k: `feature_k = ReLU(-Φ_k · normals)`
   - Result: `surface_features[15D]`
5. **Density Extraction**: `density = density_out[0]`
6. **RGB Input Assembly**: 
   - `rgb_input[0] = density`
   - `rgb_input[1:16] = surface_features[15D]`
   - `rgb_input[16:32] = enc_dir[16D]`
7. **Direction Encoding**: `dir[3D]` → SphericalHarmonics → `enc_dir[16D]`
8. **RGB MLP**: `rgb_input[32D]` → MLP(ReLU→None) → `RGB[3D]`
9. **Final Output**: `[R, G, B, σ]`

### Backward Pass Flow

1. **RGB MLP Backward**: `dL/dRGB[3D]` → `dL/d(rgb_input)[32D]`
2. **Direction Encoding Backward**: `dL/d(enc_dir)` from rgb_input[16:32]
3. **Surface Features Backward**:
   - Input: `dL/d(surface_features)[15D]` from rgb_input[1:16]
   - For each k: Compute `dL/d(Φ_k)` and `dL/d(normals)` using chain rule through ReLU(-Φ_k · normals)
   - Accumulate: `dL/d(density_out)[1:46]` from Φ gradients
4. **Density Gradient**: 
   - Add `dL/dσ` from output[3] to `dL/d(density_out)[0]`
   - Add gradient from rgb_input[0]
5. **Normal Gradients Backward** (Second-order, optional):
   - Chain rule through normalization
   - Backward through position encoding (second pass)
   - Backward through density MLP (second pass)
   - Accumulate into `dL/d(density_out)`
6. **Density MLP Backward**: `dL/d(density_out)[48D]` → `dL/d(enc_pos)[32D]`
7. **Position Encoding Backward**: `dL/d(enc_pos)[32D]` → HashGrid gradients

---

## 3. Surface Normal Mode (`surface_normal`)

### Architecture

Extends Surface Mode with additional normal encoding.

**Additional Components:**
- Second direction encoder for normals (shared with view direction encoder)

**RGB MLP Input Expanded:**
- 16D surface features + 16D encoded direction + 16D encoded normals = 48D (padded)

### Forward Pass Flow

Same as Surface Mode, plus:

4.5. **Normal Encoding**:
   - Take computed `normals[3D]` from step 3
   - Encode: `normals[3D]` → SphericalHarmonics → `enc_normals[16D]`
   
6. **RGB Input Assembly** (Modified):
   - `rgb_input[0:16] = [density, surface_features[15D]]`
   - `rgb_input[16:32] = enc_dir[16D]`
   - `rgb_input[32:48] = enc_normals[16D]`

### Backward Pass Flow

Same as Surface Mode, plus:

2.5. **Normal Encoding Backward**:
   - Input: `dL/d(enc_normals)` from rgb_input[32:48]
   - Output: `dL/d(normals)[3D]`
   - Accumulate into normal gradients from surface features

---

## 4. Surface Reflect Mode (`surface_reflect`)

### Architecture

Extends Surface Mode with reflection vector encoding.

**Additional Components:**
- Reflection vector computation
- Second direction encoder for reflections

**RGB MLP Input:**
- 16D surface features + 16D encoded direction + 16D encoded reflection = 48D (padded)

### Forward Pass Flow

Same as Surface Mode, plus:

4.5. **Reflection Vector Computation**:
   - Given `view_dir[3D]` and `normals[3D]`
   - Compute: `reflect = view_dir - 2 * (view_dir · normals) * normals`
   - Encode: `reflect[3D]` → SphericalHarmonics → `enc_reflect[16D]`

6. **RGB Input Assembly** (Modified):
   - `rgb_input[0:16] = [density, surface_features[15D]]`
   - `rgb_input[16:32] = enc_dir[16D]`
   - `rgb_input[32:48] = enc_reflect[16D]`

### Backward Pass Flow

Same as Surface Mode, plus:

2.5. **Reflection Encoding Backward**:
   - Input: `dL/d(enc_reflect)` from rgb_input[32:48]
   - Output: `dL/d(reflect)[3D]`
3.5. **Reflection Vector Backward**:
   - Chain rule through reflection formula
   - Compute: `dL/d(normals)` from reflection gradients
   - Accumulate into normal gradients

---

## 4.5. Surface Corrected Mode (`surface_corrected`)

### Architecture

Extends Surface Mode with a learned normal corrector network for improved surface features.

**Position Encoding:** HashGrid (same as Surface Mode)

**Density MLP:** Same as Surface Mode
- Input: Encoded position (32D)
- Output: 48D (1D density + 45D vector potential Φ)
- Activation: ReLU

**Corrector Network:** NEW
- Input: 3D analytical normals (from autodiff)
- Output: 3D correction vector `δ_n`
- Default: FullyFusedMLP with 64 neurons, 2 hidden layers
- Activation: ReLU → None (output)

**Direction Encoding:** SphericalHarmonics (same as Surface Mode)

**RGB MLP:** Same as Surface Mode
- Input: 16D surface features + 16D encoded direction = 32D
- Output: 3D RGB
- Note: Surface features computed using **corrected normals** instead of base normals

**Explicit Grids:** None

### Forward Pass Flow

1. **Position Encoding**: `pos[3D]` → HashGrid → `enc_pos[32D]`
2. **Density MLP**: `enc_pos[32D]` → MLP(ReLU) → `density_out[48D]`
3. **Analytical Normal Computation** (Autodiff):
   - Create gradient seed: `dL/d(density_out) = [1, 0, 0, ..., 0]`
   - Backward through density MLP: Get `dL/d(enc_pos)`
   - Backward through position encoding: Get `dSDF/dpos[3D]`
   - Normalize: `n_base = -normalize(dSDF/dpos)`
4. **Corrector Network**:
   - Input: `n_base[3D]`
   - Forward: `n_base[3D]` → Corrector MLP → `δ_n[3D]`
5. **Corrected Normal Computation**:
   - Combine: `v = n_base + δ_n`
   - Normalize: `n_corrected = normalize(v)`
6. **Surface Feature Computation**:
   - Reshape `density_out[1:46]` as 15 vectors `Φ_k[3D]` (k=0..14)
   - For each k: `feature_k = ReLU(-Φ_k · n_corrected)`  ← **Uses corrected normals!**
   - Result: `surface_features[15D]`
7. **Density Extraction**: `density = density_out[0]`
8. **RGB Input Assembly**:
   - `rgb_input[0] = density`
   - `rgb_input[1:16] = surface_features[15D]`
   - `rgb_input[16:32] = enc_dir[16D]`
9. **Direction Encoding**: `dir[3D]` → SphericalHarmonics → `enc_dir[16D]`
10. **RGB MLP**: `rgb_input[32D]` → MLP(ReLU→None) → `RGB[3D]`
11. **Final Output**: `[R, G, B, σ]`

### Backward Pass Flow

1. **RGB MLP Backward**: `dL/dRGB[3D]` → `dL/d(rgb_input)[32D]`
2. **Direction Encoding Backward**: `dL/d(enc_dir)` from rgb_input[16:32]
3. **Surface Features Backward**:
   - Input: `dL/d(surface_features)[15D]` from rgb_input[1:16]
   - For each k: Compute `dL/d(Φ_k)` and `dL/d(n_corrected)` using chain rule through ReLU(-Φ_k · n_corrected)
   - Accumulate: `dL/d(density_out)[1:46]` from Φ gradients
   - Collect: `dL/d(n_corrected)[3D]` for corrector backward
4. **Corrected Normal Backward**:
   - Chain rule through normalization: `n_corrected = normalize(n_base + δ_n)`
   - Jacobian: `J = (I/||v|| - vv^T/||v||^3)` where `v = n_base + δ_n`
   - Compute:
     - `dL/d(n_base) = J^T · dL/d(n_corrected)`
     - `dL/d(δ_n) = J^T · dL/d(n_corrected)` (same as dL/d(n_base))
5. **Eikonal Loss (Optional)**:
   - If `use_eikonal_loss` enabled:
   - Loss: `L_eik = (||n_base + δ_n|| - 1)²`
   - Gradient: `dL/dv = 2 * eikonal_weight * (||v|| - 1) * v / ||v||`
   - Accumulate to both:
     - `dL/d(n_base) += dL/dv`
     - `dL/d(δ_n) += dL/dv`
6. **Corrector Network Backward**:
   - Input: `dL/d(δ_n)[3D]` (from rendering loss + Eikonal loss)
   - Backward through corrector MLP
   - Updates corrector network parameters
   - **No input gradients** (n_base comes from autodiff, not backpropagated here)
7. **Density Gradient**:
   - Add `dL/dσ` from output[3] to `dL/d(density_out)[0]`
   - Add gradient from rgb_input[0]
8. **Normal Gradients Backward** (Second-order, optional):
   - If `backprop_normals` enabled:
   - Chain `dL/d(n_base)` through normalization
   - Backward through position encoding (second pass)
   - Backward through density MLP (second pass)
   - Accumulate into `dL/d(density_out)`
9. **Density MLP Backward**: `dL/d(density_out)[48D]` → `dL/d(enc_pos)[32D]`
10. **Position Encoding Backward**: `dL/d(enc_pos)[32D]` → HashGrid gradients

### Key Differences from Surface Mode

1. **Corrector Network**: New component that learns to refine analytical normals
2. **Two-Stage Normal Computation**:
   - Stage 1: Autodiff normals (n_base) from density gradients
   - Stage 2: Learned correction (δ_n) applied to n_base
3. **Improved Surface Features**: Dot products computed with corrected normals
4. **Eikonal Supervision**: Optional geometric regularization on corrected field
5. **Additional Parameters**: ~12K params for corrector network (default config)

### Use Cases

- **Noisy Analytical Normals**: When density gradients are unreliable (coarse hashgrids, early training)
- **Complex Geometry**: Scenes requiring high-fidelity normals for accurate rendering
- **Geometric Consistency**: Eikonal loss helps maintain unit-norm gradients
- **Reflective Surfaces**: Better normals improve view-dependent effects

### Configuration

Add to JSON config:
```json
{
  "method": "surface_corrected",
  "corrector_network": {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2
  }
}
```

Enable Eikonal loss via:
```cpp
network->set_use_eikonal_loss(true);
network->set_eikonal_weight(0.01f);  // Adjust weight as needed
```

---

## 5. Volume Mode (`volume`)

### Architecture

**Position Encoding:** HashGrid (32D)

**Density MLP:**
- Input: 32D encoded position
- Output: 48D
  - Channel 0: Density
  - Channels 1-46: 45D vector field Φ (15 × 3D vectors)
- Activation: ReLU

**Direction Encoding:** SphericalHarmonics (16D)

**RGB MLP:**
- Input: 16D volume features + 16D encoded direction = 32D
- Output: 3D RGB

**Explicit Grids:** None

### Forward Pass Flow

1. **Position Encoding**: `pos[3D]` → HashGrid → `enc_pos[32D]`
2. **Density MLP**: `enc_pos[32D]` → MLP(ReLU) → `density_out[48D]`
3. **Volume Divergence Computation** (Autodiff):
   - For each vector field k (0-14):
     - For each spatial dimension d (x, y, z):
       - Seed: `dL/d(Φ_{k,d}) = 1`
       - Backward through MLP and encoding
       - Extract: `∂Φ_{k,d}/∂x_d`
     - Compute: `div_k = ∂Φ_{k,x}/∂x + ∂Φ_{k,y}/∂y + ∂Φ_{k,z}/∂z`
   - Result: `divergences[15D]`
4. **Volume Feature Computation**:
   - For k in 0-14:
     - `feature_k = ReLU(density_out[0] + divergences[k])`
   - Result: `volume_features[15D]`
   - First feature uses density directly
5. **Direction Encoding**: `dir[3D]` → SphericalHarmonics → `enc_dir[16D]`
6. **RGB Input Assembly**:
   - `rgb_input[0:16] = [density, volume_features[15D]]`
   - `rgb_input[16:32] = enc_dir[16D]`
7. **RGB MLP**: `rgb_input[32D]` → MLP → `RGB[3D]`
8. **Final Output**: `[R, G, B, σ]`

### Backward Pass Flow

1. **RGB MLP Backward**: `dL/dRGB` → `dL/d(rgb_input)[32D]`
2. **Direction Encoding Backward**: `dL/d(enc_dir)` from rgb_input[16:32]
3. **Volume Features Backward**:
   - Input: `dL/d(volume_features)[15D]` from rgb_input[1:16]
   - Chain rule through ReLU: Get `dL/d(divergences)[15D]`
4. **Divergence Gradients Backward** (Second-order):
   - For each k and d: 
     - Chain rule through divergence computation
     - Backward passes to get `dL/d(density_out)[1:46]`
   - Accumulate all gradients
5. **Density Gradient**: Add from output[3] and rgb_input[0]
6. **Density MLP Backward**: `dL/d(density_out)[48D]` → `dL/d(enc_pos)`
7. **Position Encoding Backward**: Gradients to HashGrid

---

## 6. Hash Surface Mode (`hash_surface`)

### Architecture

**Position Encoding:**
- Type: HashGrid
- Output: 32D (8 levels × 4 features)

**Density MLP:**
- Input: 8D (extracted density features, one per level, padded to 16D)
- Output: 1D density
- Activation: ReLU

**Direction Encoding:** SphericalHarmonics (16D)

**RGB MLP:**
- Input: 8D hash surface features + 1D density + 16D encoded direction = 25D (padded to 32D)
- Output: 3D RGB

**Explicit Grids:** None

### Forward Pass Flow

1. **Position Encoding**: `pos[3D]` → HashGrid → `hash_features[32D]` (8 levels × 4)
2. **Density Feature Extraction**:
   - For each level k: Extract 4th feature from level k
   - Result: `density_features[8D]` → pad to 16D
3. **Density MLP**: `density_features[16D]` → MLP(ReLU) → `density[1D]`
4. **Analytical Normal Computation**:
   - Same autodiff process as surface mode
   - Backward through density MLP and encoding
   - Get: `normals[3D]`
5. **Hash Surface Feature Computation**:
   - For each level k:
     - Extract vector `V_k[3D]` = hash_features[k×4 : k×4+3]
     - Compute: `feature_k = ReLU(-V_k · normals)`
   - Result: `hash_surface_features[8D]`
6. **Direction Encoding**: `dir[3D]` → SphericalHarmonics → `enc_dir[16D]`
7. **RGB Input Assembly**:
   - `rgb_input[0:8] = hash_surface_features[8D]`
   - `rgb_input[8] = density`
   - `rgb_input[9:25] = enc_dir[16D]`
   - Pad to 32D
8. **RGB MLP**: `rgb_input[32D]` → MLP → `RGB[3D]`
9. **Final Output**: `[R, G, B, σ]`

### Backward Pass Flow

1. **RGB MLP Backward**: `dL/dRGB` → `dL/d(rgb_input)[32D]`
2. **Direction Encoding Backward**: From rgb_input[9:25]
3. **Hash Surface Features Backward**:
   - Input: `dL/d(hash_surface_features)[8D]`
   - For each k: Compute `dL/d(V_k)` and `dL/d(normals)`
   - Distribute to hash_features: Only gradients for 3D vectors (not 4th feature)
4. **Density Gradients**: From output[3] and rgb_input[8]
5. **Normal Gradients Backward**: Chain through density MLP
6. **Density MLP Backward**: 
   - `dL/d(density)[1D]` → `dL/d(density_features)[16D]`
7. **Density Feature Backward**:
   - Distribute gradients back to hash_features (4th feature of each level)
8. **Position Encoding Backward**:
   - Combine gradients from hash surface features and density features
   - Flow to HashGrid parameters

---

## 7. Baseline Explicit Mode (`baseline_explicit`)

### Architecture

**Position Encoding:** HashGrid (32D)

**Explicit Density Grid:**
- Type: DenseGrid
- Resolution: 128³ (or configurable via `explicit_grid_resolution`)
- Features: 1 (log-density)
- Interpolation: Trilinear
- Storage: ~2M parameters for 128³

**Density MLP:**
- Input: 32D encoded position
- Output: 15D features
- Activation: ReLU
- **Note:** Does NOT output density (density comes from grid)

**Direction Encoding:** SphericalHarmonics (16D)

**RGB MLP:**
- Input: 1D grid density + 15D MLP features + 16D encoded direction = 32D (padded)
- Output: 3D RGB

### Forward Pass Flow

1. **Position Encoding**: `pos[3D]` → HashGrid → `enc_pos[32D]`
2. **Grid Density Query**:
   - `pos[3D]` → DenseGrid (trilinear interpolation) → `grid_density[1D]`
   - Value is log-density (will be exp() later)
3. **Feature MLP**: `enc_pos[32D]` → MLP(ReLU) → `mlp_features[15D]`
4. **RGB Input Assembly**:
   - `rgb_input[0] = grid_density`
   - `rgb_input[1:16] = mlp_features[15D]`
   - `rgb_input[16:32] = enc_dir[16D]`
5. **Direction Encoding**: `dir[3D]` → SphericalHarmonics → `enc_dir[16D]`
6. **RGB MLP**: `rgb_input[32D]` → MLP(ReLU→None) → `RGB[3D]`
7. **Density Extraction**: `σ = exp(grid_density)` (activation applied during rendering)
8. **Final Output**: `[R, G, B, σ]`

### Backward Pass Flow

1. **RGB MLP Backward**: `dL/dRGB` → `dL/d(rgb_input)[32D]`
2. **Direction Encoding Backward**: From rgb_input[16:32]
3. **MLP Features Backward**:
   - Input: `dL/d(mlp_features)[15D]` from rgb_input[1:16]
   - Feature MLP backward: → `dL/d(enc_pos)[32D]`
   - Position encoding backward: → HashGrid gradients
4. **Grid Density Backward**:
   - Extract: `dL/d(grid_density)` from rgb_input[0]
   - Add: `dL/dσ` from output[3] (with exp() chain rule)
   - Grid backward: Distribute gradients to 8 grid corners (trilinear)
   - Update: DenseGrid parameters

**Key Difference:** Gradients split between:
- HashGrid (from MLP features)
- DenseGrid (from density)

---

## 8. Surface Explicit Mode (`surface_explicit`)

### Architecture

**Position Encoding:** HashGrid (32D)

**Explicit Density Grid:**
- Type: DenseGrid
- Resolution: 128³ (configurable)
- Features: 1 (log-density)
- Interpolation: Trilinear

**Density MLP:**
- Input: 32D encoded position
- Output: 48D
  - Channel 0: Dummy (unused)
  - Channels 1-46: 45D vector potential (15 × 3D)
- Activation: ReLU

**Direction Encoding:** SphericalHarmonics (16D)

**RGB MLP:**
- Input: 1D grid density + 15D surface features + 16D encoded direction = 32D
- Output: 3D RGB

### Forward Pass Flow

1. **Position Encoding**: `pos[3D]` → HashGrid → `enc_pos[32D]`
2. **Grid Density Query** (with gradients):
   - `pos[3D]` → DenseGrid → `grid_density[1D]`
   - **Enable input gradients** for normal computation
3. **Normal Computation from Grid**:
   - Backward through grid with seed `dL/d(grid_density) = 1`
   - Get: `dGrid/dpos[3D]`
   - Normalize: `normals = -normalize(dGrid/dpos)`
4. **Feature MLP**: `enc_pos[32D]` → MLP(ReLU) → `mlp_out[48D]`
5. **Surface Features from MLP Vectors**:
   - Extract: 15 vectors `Φ_k[3D]` from mlp_out[1:46]
   - For each k: `feature_k = ReLU(-Φ_k · normals)`
   - Result: `surface_features[15D]`
6. **Direction Encoding**: `dir[3D]` → SphericalHarmonics → `enc_dir[16D]`
7. **RGB Input Assembly**:
   - `rgb_input[0] = grid_density`
   - `rgb_input[1:16] = surface_features[15D]`
   - `rgb_input[16:32] = enc_dir[16D]`
8. **RGB MLP**: `rgb_input[32D]` → MLP → `RGB[3D]`
9. **Density Extraction**: `σ = exp(grid_density)`
10. **Final Output**: `[R, G, B, σ]`

### Backward Pass Flow

1. **RGB MLP Backward**: `dL/dRGB` → `dL/d(rgb_input)[32D]`
2. **Direction Encoding Backward**: From rgb_input[16:32]
3. **Surface Features Backward**:
   - Input: `dL/d(surface_features)[15D]`
   - Compute: `dL/d(Φ_k)` and `dL/d(normals)` for each k
   - MLP backward: `dL/d(Φ_k)` → `dL/d(mlp_out)[1:46]`
   - **Normal gradients** → saved for grid backward
4. **Feature MLP Backward**:
   - Input: `dL/d(mlp_out)[48D]` (channel 0 zero)
   - Output: `dL/d(enc_pos)[32D]`
   - Position encoding backward: → HashGrid gradients
5. **Grid Density Backward**:
   - Extract: `dL/d(grid_density)` from rgb_input[0]
   - Add: `dL/dσ` from output[3]
   - **Normal gradients backward**:
     - Chain through normalization and grid gradient computation
     - Add to `dL/d(grid_density)`
   - Grid backward: Distribute to 8 corners → DenseGrid gradients

**Key Features:**
- Normals computed from **grid** (not MLP)
- Grid receives gradients from **both density and normals**
- MLP only outputs surface feature vectors
- Optional: Finite difference normal computation (via `NGP_GRAD_METHOD=finite`)

---

## Comparison Summary

| Mode | Density Source | Normal Source | Special Features | MLP Output Dims |
|------|---------------|---------------|------------------|-----------------|
| baseline | MLP[0] | N/A | None | 16D |
| baseline_aggregate | MLP[0] | N/A | Diffuse RGB + 4D features, per-pixel RGB MLP | 8D |
| surface | MLP[0] | Autodiff MLP | Vector potential Φ | 48D |
| surface_normal | MLP[0] | Autodiff MLP | + Encoded normals | 48D |
| surface_reflect | MLP[0] | Autodiff MLP | + Encoded reflections | 48D |
| surface_corrected | MLP[0] | Autodiff MLP + Corrector | Learned normal correction + Eikonal | 48D |
| volume | MLP[0] | N/A | Divergence features | 48D |
| hash_surface | MLP | Autodiff MLP | Hash feature vectors | 1D |
| baseline_explicit | Grid | N/A | Grid density | 15D |
| surface_explicit | Grid | Grid gradients | Grid density + MLP Φ | 48D |

## Activation Functions Summary

**Default Configuration:**
- **Density MLP**: ReLU activation on hidden layers
- **RGB MLP**: ReLU on hidden layers, None (linear) on output
- **Feature Computation**: ReLU on dot products (surface/volume features)
- **Normal Computation**: Normalization (not an activation, but a normalization layer)

**Configurable:**
All activations can be changed via JSON config (`"activation": "ReLU"`, etc.)

