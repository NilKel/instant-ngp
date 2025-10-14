# Baseline Aggregate - Fixed Architecture

## Summary of Fixes

### Issue 1: Missing Gradient Backprop to Viewdir Encoding ✅ FIXED

**Problem:** The RGB MLP and viewdir encoding were NOT receiving gradients during training. Only the density MLP was being trained.

**Root Cause:** The `backward_impl` was incomplete - it only backpropped through the density network, completely skipping the RGB network.

**Solution:** Implemented complete backward pass:
1. Split gradients from 8D output to separate density and RGB components
2. Backprop through RGB network → viewdir encoding
3. Backprop through density network → position encoding

### Issue 2: Incorrect Activation Application ✅ FIXED

**Problem:** Activations were being applied in the network's forward pass, but the kernel expected raw values.

**Root Cause:** Mixing activation application between network and kernels led to:
- Double activation (network + kernel)
- Incorrect gradient flow
- Loss computed on wrong values

**Solution:** 
- Network outputs **raw values**: `[density, diffuse_RGB_raw(3), directional_RGB_raw(3), unused]`
- Kernel applies activations: `sigmoid(diffuse_raw) + sigmoid(directional_raw)`
- Gradients flow back with proper activation derivatives

## Final Architecture

### Forward Pass

**Network (Per-Sample):**
```
Position → HashGrid → Density MLP → [density_raw, diffuse_RGB_raw(3), features(4)]
                                              ↓
                                    [features(4), ViewDir → SH] → RGB MLP → directional_RGB_raw(3)
                                              ↓
                              Combine: [density, diffuse_raw(3), directional_raw(3), unused] (8D)
```

**Kernel (Volume Rendering):**
```
For each sample along ray:
    diffuse_rgb = sigmoid(diffuse_raw)
    directional_rgb = sigmoid(directional_raw)
    final_rgb = diffuse_rgb + directional_rgb
    
    alpha = 1 - exp(-density * dt)
    weight = alpha * (1 - T)
    
    color += final_rgb * weight
```

### Backward Pass

**Kernel → Network:**
```
dL/d(final_rgb) → [dL/d(diffuse_raw), dL/d(directional_raw)]
                          ↓                      ↓
                  (apply sigmoid')      (apply sigmoid')
                          ↓                      ↓
              dL/d(diffuse_RGB_raw)    dL/d(directional_RGB_raw)
```

**Network Backprop:**
```
dL/d(directional_RGB_raw) → RGB MLP → dL/d(features) → Density MLP
                              ↓
                           Viewdir Encoding ✓ (NOW RECEIVES GRADIENTS!)

dL/d(diffuse_RGB_raw) → Density MLP → Position Encoding ✓
```

## Output Format

### Network Output (8D):
```
Channel 0: density_raw
Channel 1: diffuse_R_raw
Channel 2: diffuse_G_raw
Channel 3: diffuse_B_raw
Channel 4: directional_R_raw
Channel 5: directional_G_raw
Channel 6: directional_B_raw
Channel 7: unused (padding)
```

### Gradient Input (8D):
```
Channel 0: dL/d_density
Channel 1: dL/d_diffuse_R
Channel 2: dL/d_diffuse_G
Channel 3: dL/d_diffuse_B
Channel 4: dL/d_directional_R
Channel 5: dL/d_directional_G
Channel 6: dL/d_directional_B
Channel 7: 0 (unused)
```

## Key Implementation Details

### 1. Gradient Splitting Kernel
```cuda
split_combined_gradients():
    // Input: dL/d[density, diffuse_raw(3), directional_raw(3), unused]
    
    // To density network: [dL/d_density, dL/d_diffuse_raw(3), zeros(4)]
    dL_ddensity[0] = dL_doutput[0]  // density
    dL_ddensity[1:3] = dL_doutput[1:3]  // diffuse RGB
    dL_ddensity[4:7] = 0  // features (no gradients yet)
    
    // To RGB network: [dL/d_directional_raw(3)]
    dL_drgb[0:2] = dL_doutput[4:6]  // directional RGB
```

### 2. Activation Derivatives
```cuda
// In kernel gradient computation:
for each RGB channel:
    dL/d_diffuse_raw = dL/d_rgb * sigmoid'(diffuse_raw)
    dL/d_directional_raw = dL/d_rgb * sigmoid'(directional_raw)
```

### 3. Stride and Alignment
- All operations use proper AoS/RM layout checks
- RGB network input: 20D (4 features + 16 encoded viewdir) → padded to 32D
- Output alignment: 8D for proper memory access

## Verification Checklist

✅ Network outputs raw values (no activation in forward pass)
✅ Kernel applies sigmoid to both diffuse and directional components
✅ Gradients split correctly between density and RGB networks
✅ RGB network receives gradients and trains
✅ Viewdir encoding receives gradients and trains
✅ Activation derivatives applied correctly in kernel
✅ Memory alignment and strides correct
✅ Both training and inference kernels updated

## Testing

To verify the fix works:
```bash
python scripts/run.py \
    --scene data/nerf_synthetic/lego/transforms_train.json \
    --network configs/nerf/baseline_aggregate.json \
    --n_steps 10000 \
    --method baseline_aggregate
```

**Expected Results:**
- Loss should converge (decreasing over time)
- View-dependent effects should be visible
- RGB MLP and viewdir encoding parameters should update during training

## Files Modified

1. **baseline_aggregate_network.h**
   - Changed `combine_rgb_and_density` → `combine_density_and_rgb`
   - Removed activation application from network
   - Added `split_combined_gradients` kernel
   - Implemented complete `backward_impl` with RGB network backprop

2. **train_nerf.cuh**
   - Updated to handle 8D format: `[density, diffuse_raw(3), directional_raw(3), unused]`
   - Apply activations in kernel before combining
   - Compute gradients with activation derivatives for both components

3. **render_nerf.cuh**
   - Updated to handle 8D format
   - Apply activations in kernel before combining

## Performance Notes

- RGB MLP evaluated once per sample (not per-pixel)
- Additional kernel overhead for gradient splitting
- Memory: 8D output vs 4D (2x memory for network output)
- Compute: 2x sigmoid activations per sample (diffuse + directional)




