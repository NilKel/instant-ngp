# Quick Start: Dynamic Learning Rate Schedule

## TL;DR

**No more manual `decay_base` calculations!** Just specify your training duration and desired learning rates:

```json
"density_grid_optimizer": {
    "learning_rate_start": 30.0,
    "learning_rate_end": 0.05,
    "total_training_steps": 10000
}
```

The code automatically calculates the perfect decay rate!

## Example Configs

### 10k Steps (Quick Test)
**File:** `configs/nerf/baseline_explicit_10k.json`

```bash
python scripts/run.py \
    --scene data/nerf_synthetic/lego/transforms_train.json \
    --network configs/nerf/baseline_explicit_10k.json \
    --method baseline_explicit \
    --n_steps 10000
```

**Learning Rate Schedule:**
- Steps 0-2000: LR = 30.0 (warm-up)
- Steps 2000-10000: LR decays from 30.0 → 0.1
- Auto-calculated: `decay_base = 0.9997186`

### 50k Steps (Default)
**File:** `configs/nerf/baseline_explicit_plenoxels.json`

```bash
python scripts/run.py \
    --scene data/nerf_synthetic/lego/transforms_train.json \
    --network configs/nerf/baseline_explicit_plenoxels.json \
    --method baseline_explicit \
    --n_steps 50000
```

**Learning Rate Schedule:**
- Steps 0-3000: LR = 30.0 (warm-up)
- Steps 3000-50000: LR decays from 30.0 → 0.05
- Auto-calculated: `decay_base = 0.9998626`

## How to Customize

### 1. Copy a Config
```bash
cp configs/nerf/baseline_explicit_10k.json configs/nerf/my_config.json
```

### 2. Edit Your Training Duration
```json
{
    "network": {
        "density_grid_optimizer": {
            "learning_rate_start": 30.0,
            "learning_rate_end": 0.05,
            "total_training_steps": 20000,  // Your training duration
            "decay_start": 3000              // Warm-up period
        }
    }
}
```

### 3. Train
```bash
python scripts/run.py \
    --scene YOUR_SCENE.json \
    --network configs/nerf/my_config.json \
    --method baseline_explicit \
    --n_steps 20000  # Match total_training_steps
```

## Parameters

| Parameter | What It Does | Recommended |
|-----------|--------------|-------------|
| `learning_rate_start` | Initial LR (aggressive) | `20-30` for Plenoxels-style |
| `learning_rate_end` | Final LR (fine-tuning) | `0.05-0.1` |
| `total_training_steps` | How many steps you'll train | Match `--n_steps` |
| `decay_start` | Steps before decay begins | `10-20%` of total steps |

## What You'll See

When training starts:
```
Creating separate optimizer for density grid (2097152 params)
Calculated decay_base for density grid optimizer:
  LR: 30 → 0.1 over 8000 steps (total=10000, decay_start=2000)
  decay_base = 0.999718618
Grid optimizer config: {"decay_base":0.999718618,...}
```

This confirms automatic calculation worked!

## Benefits

✅ **No manual math** - decay_base calculated automatically  
✅ **Flexible** - easily change training duration  
✅ **Transparent** - see calculated values in logs  
✅ **Correct** - guaranteed to hit target LR at final step  

## Full Documentation

For complete details, see:
- `DYNAMIC_LR_SCHEDULE.md` - Full explanation and examples
- `BASELINE_EXPLICIT_CHANGES_SUMMARY.md` - Implementation details
- `PLENOXELS_UPDATES.md` - ReLU activation details

