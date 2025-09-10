# Routing Scrutiny: Surface vs Baseline

## Current Goal
Ensure runs can be routed between baseline and surface methods via CLI, verify active path at runtime, and produce comparable eval outputs. Specifically, make the baseline method work correctly *without* JIT fusion.

## What We're Trying to Achieve
- **Surface Mode**: Density MLP outputs 46D (1D density + 15×3D Phi surface potentials), compute 15 ReLU(-Phi·n) features, concatenate with normals to RGB MLP input.
- **Baseline Mode**: Standard density-only output (1D), RGB MLP gets its original input without modification.
- **Routing**: Environment variable `NGP_METHOD=surface|baseline` controls which path is taken.
- **Verification**: Comprehensive debug prints to confirm active code path and tensor shapes.

## Current Problems (CRITICAL)

### 1. Compilation Errors
- **File**: `src/testbed_nerf.cu` lines 1779, 1804
- **Error**: "a nonstatic member reference must be relative to a specific object"
- **Cause**: Using `m_nerf_network` (member variable) instead of `nerf_network` (function parameter) in `render_nerf()` function
- **Fix Needed**: Change `m_nerf_network->` to `nerf_network->` in lines 1779 and 1804

### 2. Baseline Rendering Issues (Likely JIT Fusion)
- **Problem**: Baseline renders still appear "weird" despite previous fixes to zeroing logic.
- **Cause**: Investigation points to `m_jit_fusion` being unconditionally set to `false` in `src/testbed_nerf.cu`, disabling JIT fusion even for baseline, which relies on it for correct behavior.
- **Fix Needed**: Ensure `m_jit_fusion` is *not* set to `false` for baseline mode. User needs to manually comment out line 1957 in `src/testbed_nerf.cu`.

## Implementation Status

### ✅ Completed
- CLI integration with `--method {baseline,surface}` and `--name` flags
- Environment variable export (`NGP_METHOD`, `NGP_BNORMALS`)
- Output directory structure (`outputs/<dataset>/<scene>/<method>/<name>`)
- Post-training evaluation with PSNR/SSIM logging
- Surface features implementation in `nerf_network.h`:
  - 46D density head output (1 + 15×3 Phi)
  - Surface features kernel: `compute_surface_features_kernel`
  - Baseline preamble kernel: `compute_baseline_preamble_kernel`
  - Environment-driven routing with debug prints
- Network instance fixes in utility functions (`get_density_on_grid`, `get_rgba_on_grid`)
- Fixed network instance assignments in `src/testbed.cu` (line 4310)
- Replaced `device.nerf_network()` with `m_nerf_network` in `src/testbed.cu` (line 5598)
- Disabled `throw std::runtime_error` for `output_width` checks in `nerf_network.h`.
- Removed explicit RGB MLP input zeroing in surface mode (in `nerf_network.h`).
- Removed baseline preamble modifications to RGB input (in `nerf_network.h`).

### ❌ Blocking Issues
1. **Compilation fails** due to incorrect network instance usage in `src/testbed_nerf.cu` (lines 1779, 1804).
2. **Training blocked:** User needs to manually comment out lines 2756-2758 in `src/testbed_nerf.cu` to unblock training for baseline mode.
3. **JIT fusion disabled for baseline:** User needs to manually comment out line 1957 in `src/testbed_nerf.cu` to re-enable JIT fusion for baseline mode.

## Next Steps for New Agent

### Immediate (Fix Compilation & Unblock Training)
1. Fix `src/testbed_nerf.cu` lines 1779, 1804: `m_nerf_network->` → `nerf_network->`
2. Manually comment out lines 2756-2758 in `src/testbed_nerf.cu` to unblock training.
3. Manually comment out line 1957 in `src/testbed_nerf.cu` to enable JIT fusion for baseline.
4. Verify all network instance usage is consistent.
5. Test compilation succeeds and both modes train/render as expected.

### Debug Surface Routing
1. Add more comprehensive debug prints to trace execution path.
2. Verify environment variables are properly read.
3. Check if JIT fusion is properly disabled in surface mode (while enabled for baseline).
4. Test zeroing experiments to confirm surface features are active.

### Verify Implementation
1. Confirm density head width is 46D in surface mode, 1D in baseline.
2. Verify surface features computation: `[density, 15 ReLU(-Phi·n)]`.
3. Test that baseline mode uses its original RGB input without modification.
4. Ensure both training and rendering use the same routing.

## Key Files
- `scripts/run.py` - CLI and environment setup
- `include/neural-graphics-primitives/nerf_network.h` - Surface features implementation
- `src/testbed_nerf.cu` - Rendering and training loops (COMPILATION ERRORS HERE)
- `src/testbed.cu` - Network initialization

## Environment Variables
- `NGP_METHOD=surface|baseline` - Controls routing
- `NGP_BNORMALS=0|1` - Controls normal backpropagation
