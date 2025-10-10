# Baseline Aggregate Implementation Status

## Completed Components

### ✅ 1. Network Architecture (`baseline_aggregate_network.h`)
- **Status**: COMPLETE
- BaselineAggregateNetwork class implemented
- Outputs 8D: [density(1D), diffuse_RGB(3D), features(4D)]
- RGB MLP configured for 4D features + 16D encoded direction
- `apply_directional_mlp()` method for per-pixel processing
- Proper forward/backward pass implementation

### ✅ 2. Factory Registration (`nerf_network_factory.h`)
- **Status**: COMPLETE  
- baseline_aggregate mode registered in factory
- Added to supported methods list

### ✅ 3. Configuration (`baseline_aggregate.json`)
- **Status**: COMPLETE
- Density network configured for 8D output
- RGB network configured for 20D input (padded to 32D)
- All hyperparameters set appropriately

### ✅ 4. Documentation (`ARCHITECTURE.md`)
- **Status**: COMPLETE
- Full architectural specification added
- Forward/backward pass flows documented
- Comparison table updated

## Incomplete Components (Requires Future Work)

### ⚠️ 5. Rendering Kernel Integration

**Current Limitation**:
The baseline_aggregate network outputs 8D, but the rendering kernels (`train_nerf.cuh`, `render_nerf.cuh`) expect 4D output from `eval_nerf()`. The two-stage rendering (alpha blend 7D, then per-pixel RGB MLP) is not yet integrated into the rendering pipeline.

**What's Needed**:

#### A. Device Function Generation
The network needs to override or extend device function generation to output 8D:

```cpp
// In baseline_aggregate_network.h
std::string generate_device_function(const std::string& function_name) override {
    if (function_name == "eval_nerf") {
        // Generate custom 8D eval_nerf function
        return /* JIT code for 8D output */;
    }
    return NerfNetworkBase<T>::generate_device_function(function_name);
}
```

#### B. Training Kernel Modifications (`train_nerf.cuh`)

**Required Changes**:

1. **Detect baseline_aggregate mode** (via compile-time constant or runtime check)

2. **Handle 8D output**:
```cuda
#ifdef BASELINE_AGGREGATE_MODE
    vec<8> nerf_out = eval_nerf(nerf_in, params);
    float density = nerf_out[0];
    vec3 diffuse_rgb = {nerf_out[1], nerf_out[2], nerf_out[3]};
    vec4 features = {nerf_out[4], nerf_out[5], nerf_out[6], nerf_out[7]};
#else
    vec4 nerf_out = eval_nerf(nerf_in, params);
#endif
```

3. **Accumulate 7D per pixel**:
```cuda
vec4 diffuse_color = vec4(0.0f);  // [R_d, G_d, B_d, alpha]
vec4 accumulated_features = vec4(0.0f);  // [f0, f1, f2, f3]

// In ray marching loop:
float alpha = 1.f - __expf(-density * dt);
float weight = alpha * (1.0f - diffuse_color.a);
diffuse_color += vec4(diffuse_rgb * weight, weight);
accumulated_features += features * weight;
```

4. **Per-pixel RGB MLP** (after ray loop):
```cuda
// After ray marching completes for this pixel:
if (diffuse_color.a > 0.0f) {
    // Encode view direction
    vec16 encoded_dir = encode_spherical_harmonics(ray.d);
    
    // Prepare RGB MLP input [features(4D), encoded_dir(16D)]
    float rgb_input[32];  // Padded
    for (int i = 0; i < 4; i++) rgb_input[i] = accumulated_features[i];
    for (int i = 0; i < 16; i++) rgb_input[4+i] = encoded_dir[i];
    
    // Evaluate RGB MLP
    vec3 directional_rgb = eval_rgb_mlp(rgb_input, params);
    
    // Final composition
    color.rgb() = diffuse_color.rgb() + directional_rgb * diffuse_color.a;
    color.a = diffuse_color.a;
}
```

5. **Backward Pass Modifications**:
```cuda
// Gradients need to flow through:
// - Final RGB → directional_rgb → RGB MLP → features
// - Final RGB → diffuse_rgb → density MLP
// - Density gradients as normal
```

#### C. Inference Kernel Modifications (`render_nerf.cuh`)

Similar modifications needed for inference:
- Handle 8D eval_nerf output
- Accumulate diffuse + features
- Per-pixel RGB MLP evaluation
- Final composition

#### D. eval_rgb_mlp Device Function

Need to create a device-callable RGB MLP evaluator:
```cuda
__device__ vec3 eval_rgb_mlp(const float* input, const network_precision_t* params) {
    // Extract RGB MLP parameters from params
    // Forward pass through RGB MLP
    // Return 3D RGB output
}
```

## Workaround for Current Implementation

Until kernel integration is complete, the baseline_aggregate network can be used in two ways:

### Option 1: Inference-Only (Diffuse RGB)
- Network outputs 8D during inference
- Use only channels 1-3 (diffuse RGB) for rendering
- Ignore features and directional RGB
- Compatible with existing rendering pipeline

### Option 2: Custom Training Script
- Implement custom training loop outside fused kernels
- Manually handle two-stage rendering
- Use `apply_directional_mlp()` for per-pixel processing

Example:
```python
# Pseudo-code
density_outputs = network.density_forward(positions)  # [N, 8]
diffuse_rgb, features = volume_render_7d(density_outputs)  # per-pixel
directional_rgb = network.apply_directional_mlp(features, view_dirs)
final_rgb = diffuse_rgb + directional_rgb
```

## Implementation Priority

1. **High Priority**: Device function generation for 8D output
2. **High Priority**: Training kernel modifications for two-stage rendering
3. **Medium Priority**: Inference kernel modifications
4. **Low Priority**: Optimization and cleanup

## Testing Checklist

Once kernel integration is complete:

- [ ] Verify 8D output from eval_nerf
- [ ] Confirm 7D accumulation (diffuse + features)
- [ ] Test per-pixel RGB MLP evaluation
- [ ] Validate final RGB = diffuse + directional
- [ ] Check backward pass gradients
- [ ] Compare with baseline mode for sanity
- [ ] Benchmark performance vs. baseline

## Notes

- The current implementation provides the network architecture and interface
- Full functionality requires deep integration with JIT compilation system
- Consider whether to use compile-time or runtime mode detection
- May need to modify tiny-cuda-nn's network interface for custom device functions

