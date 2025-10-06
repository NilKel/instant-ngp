# SDF Mode Buffer Corruption Investigation

## CRITICAL SYMPTOM
**FullyFusedMLP shared memory error occurs ONLY in SDF mode, works fine in non-SDF mode**

Error:
```
RuntimeError: FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead.
```

Followed by:
```
WARNING: Could not free memory: cudaFree(rawptr) failed: misaligned address
```

## UPDATE: CRASH LOCATION FOUND

### The Log is LYING!
Line 4451 in `src/testbed.cu` has a **HARDCODED** `<< 1` in the density model log:
```cpp
tlog::info() << "Density model: " << dims.n_pos << "--[" << std::string(encoding_config["otype"]) << "]-->"
             << m_nerf_network->pos_encoding()->padded_output_width() << "--[" << std::string(network_config["otype"])
             << "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"] + 2) << ")"
             << "]-->" << 1;  // <-- THIS IS HARDCODED!
```

The network **IS** being created with 16D output (as your debug shows: `padded_output_width=16`).
The log is just hardcoded to always show `-->1` regardless of actual dimensions.

### Actual Crash Location
Based on the error output and code flow:

1. ✅ Constructor completes successfully (line 760: `padded_output_width=16`)
2. ✅ Network info is logged (misleading `-->1`)
3. ✅ Training starts (line 772: `Training: 0%|...`)
4. 💥 **Crash happens in first training step**

**Call stack to crash:**
```
frame() [testbed.cu:3947]
  → train_and_render() [testbed.cu:3211]
    → train() [testbed.cu:4708]
      → train_nerf() [testbed_nerf.cu:2722]
        → train_nerf_step() [testbed_nerf.cu:2785] ← CRASH HAPPENS HERE
          → m_trainer->training_step() [tiny-cuda-nn]
            → forward() → FullyFusedMLP crashes
```

### Missing Debug Prints

You need debug prints **BEFORE** the crash happens. Add these:

**1. In `src/testbed_nerf.cu:2722` (start of `train_nerf()`):**
```cpp
void Testbed::train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
    printf("DEBUG train_nerf: ENTRY - batch_size=%u, step=%u\n", target_batch_size, m_training_step);
    
    if (m_nerf.training.n_images_for_training == 0) {
        return;
    }
    // ... rest of function
```

**2. In `src/testbed_nerf.cu` around line 2780-2790 (before `train_nerf_step()`):**
```cpp
printf("DEBUG train_nerf: About to call train_nerf_step()\n");
printf("DEBUG train_nerf: m_nerf_network ptr = %p\n", (void*)m_nerf_network.get());
printf("DEBUG train_nerf: Network dims - encoding out: %u, density net out: %u, rgb in: %u\n",
    m_nerf_network->pos_encoding()->padded_output_width(),
    m_nerf_network->density_network()->padded_output_width(),
    m_nerf_network->rgb_network_input_width());
printf("DEBUG train_nerf: Syncing before train_nerf_step...\n");
CUDA_CHECK_THROW(cudaDeviceSynchronize());  // Ensure any prior errors surface
printf("DEBUG train_nerf: Calling train_nerf_step() with batch_size=%u\n", target_batch_size);
train_nerf_step(target_batch_size, m_nerf.training.counters_rgb, stream);
printf("DEBUG train_nerf: train_nerf_step() completed successfully\n");
```

**3. In the NerfNetwork forward pass (nerf_network.h or wherever forward is implemented):**
Find the `forward_impl()` function and add at the very start:
```cpp
printf("DEBUG forward_impl: ENTRY - batch_size=%u, use_sdf=%d\n", 
    input.n(), m_use_sdf);
```

## WHAT WORKS
- **Non-SDF mode with FullyFusedMLP**: Works perfectly with same network config
- **Base config**: 
  - HashGrid encoding (32D output)
  - FullyFusedMLP density network: 64 neurons, layers TBD
  - FullyFusedMLP RGB network: 64 neurons, 2 layers
  - Batch size: 262,144 (1<<18)

## WHAT FAILS
- **Baseline SDF mode with FullyFusedMLP**: Crashes immediately on first forward pass
- Error happens BEFORE any debug prints in `forward_impl()` appear
- Suggests failure during network initialization or first validation pass

## KEY OBSERVATIONS

### 1. Error Timing
The error occurs **before training starts**, likely during:
- Network construction validation
- First test forward pass
- Buffer initialization in tiny-cuda-nn

### 2. Output Dimensions Mismatch
Log shows:
```
Density model: 3--[HashGrid]-->32--[FullyFusedMLP(neurons=64,layers=3)]-->1
```

But code sets `n_output_dims=16` for baseline SDF mode. This suggests:
- Either the config is being overridden somewhere
- Or the log is showing wrong information
- The `-->1` ending indicates 1D output was created despite requesting 16D

### 3. Constructor Behavior
Current debug prints in constructor should show:
```
DEBUG: Set n_output_dims=16 for baseline SDF mode
DEBUG: Creating density network with n_output_dims=16
DEBUG CONSTRUCTOR: Density network created - method=baseline, sdf=1, requested_dims=16, padded_output_width=?
DEBUG CONSTRUCTOR: Full created
```

But these aren't appearing in the user's output, suggesting crash happens earlier.

## ROOT CAUSE HYPOTHESES

### Hypothesis 1: Config Override
The `base.json` doesn't specify `n_output_dims`, so code sets it to 16.
But something might be overriding it back to 1 after our setting.

**Check**: Does `base.json` or testbed code override `n_output_dims` for density network?

### Hypothesis 2: Buffer Alignment Issue
The density network buffer might be:
- Allocated with wrong size (1D instead of 16D)
- Allocated with wrong alignment
- Allocated with incompatible layout (AoS vs SoA)

This causes FullyFusedMLP to:
- Read past buffer boundaries
- Hit misaligned addresses
- Fail shared memory calculations

### Hypothesis 3: Initialization Difference
SDF mode might trigger different initialization paths in tiny-cuda-nn that:
- Validate network outputs differently
- Require different buffer alignment
- Have stricter memory requirements

## WHAT TO INVESTIGATE

### 1. Network Creation
Add debug prints to confirm:
```cpp
printf("DEBUG: local_density_network_config n_output_dims = %d\n", local_density_network_config["n_output_dims"].get<int>());
```

Right before:
```cpp
m_density_network.reset(create_network<T>(local_density_network_config));
```

### 2. Check if testbed overrides config
Search for where density_network config is modified:
- In `testbed.cu` / `testbed_surface.cu`
- In `reset_network()` function
- Before NerfNetwork constructor is called

### 3. Verify actual network output width
After network creation:
```cpp
printf("Actual network output: padded=%u, output_width=%u\n", 
    m_density_network->padded_output_width(),
    m_density_network->output_width());
```

### 4. Check tiny-cuda-nn network creation
The issue might be in tiny-cuda-nn's `create_network<T>()` function:
- Does it ignore `n_output_dims` for certain configs?
- Does it have different behavior for small output dims?
- Does it validate during construction?

## CODE LOCATIONS

### NerfNetwork Constructor
File: `include/neural-graphics-primitives/nerf_network.h`
Lines: ~56-103

Key section:
```cpp
if (!density_network.contains("n_output_dims")) {
    // ... sets n_output_dims based on method and use_sdf
    else if (m_use_sdf) {
        local_density_network_config["n_output_dims"] = 16;
    }
}
```

### Network Creation
```cpp
m_density_network.reset(create_network<T>(local_density_network_config));
```

This calls tiny-cuda-nn's factory function - need to verify it respects `n_output_dims`.

### RGB Network Input Width Calculation
Lines: ~124-130
```cpp
// Baseline: density output + direction encoding
m_rgb_network_input_width = next_multiple(
    m_dir_encoding->padded_output_width() + std::max(16u, m_density_network->padded_output_width()), 
    rgb_alignment
);
```

If `m_density_network->padded_output_width()` returns 1 instead of 16:
- `rgb_network_input_width` becomes `dir_encoding_width + 16` (using the max)
- But forward pass tries to use density_network_output with wrong dimensions
- This creates buffer misalignment

## CRITICAL QUESTIONS

1. **Why does the log show `-->1` output when we set `n_output_dims=16`?**
2. **Why does FullyFusedMLP fail only in SDF mode if buffers are same size?**
3. **Where is the network config being modified between our setting and actual creation?**
4. **Does tiny-cuda-nn's create_network() honor n_output_dims for all network types?**

## DEBUG PRINTS ADDED

Added comprehensive debug prints to trace execution:

### In nerf_network.h:
1. **Forward_impl entry** (line ~487): Detailed network configuration
2. **ForwardContext creation** (line ~495): Confirms context allocation
3. **SDF buffer creation** (line ~692-695): Buffer allocation parameters
4. **Density network forward** (line ~705-708): Input/output pointers and dimensions
5. **Extract_density kernel call** (line ~710-715): Kernel parameters and strides
6. **RGB network forward** (line ~812-815): RGB network invocation
7. **Forward_impl completion** (line ~841): Success confirmation

### In nerf_helpers.h:
1. **extract_density kernel entry** (line ~56): Kernel parameters (first thread only)
2. **extract_density kernel completion** (line ~83): Success confirmation (first thread only)

These prints will show exactly where execution stops, helping pinpoint if the crash is:
- Before forward_impl is called
- During buffer allocation
- During density network forward pass
- During the extract_density kernel
- During RGB network forward pass

## NEXT STEPS

1. ✅ Run with debug prints to see where execution stops
2. Search codebase for any place that modifies density_network config after NerfNetwork sets it
3. Check if testbed passes different configs for SDF vs non-SDF mode
4. Verify tiny-cuda-nn respects n_output_dims parameter
5. If network is truly 1D, find where dimension mismatch causes FullyFusedMLP to fail

## WORKAROUND (TEMPORARY)
Using `CutlassMLP` instead of `FullyFusedMLP` works but is slower. This masks the root cause.
