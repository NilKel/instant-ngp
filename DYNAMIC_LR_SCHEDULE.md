# Dynamic Learning Rate Schedule for baseline_explicit

## Overview

The `baseline_explicit` mode now supports **automatic learning rate decay calculation** based on your total training steps. Instead of manually calculating `decay_base`, you just specify:
- `learning_rate_start`: Initial learning rate
- `learning_rate_end`: Final learning rate  
- `total_training_steps`: How many steps you'll train for

The code automatically calculates the perfect `decay_base` to smoothly decay from start to end over your training duration.

## How It Works

### Formula
```
decay_base = (lr_end / lr_start)^(1 / decay_steps)
where decay_steps = total_training_steps - decay_start
```

### Example Calculation
If you want to train for **10,000 steps**:
- `learning_rate_start = 30.0`
- `learning_rate_end = 0.05`
- `decay_start = 3000`
- `total_training_steps = 10000`

Then:
```
decay_steps = 10000 - 3000 = 7000
decay_base = (0.05 / 30.0)^(1/7000)
decay_base ≈ 0.9990927
```

**Learning rate schedule:**
- Steps 0-3000: LR = 30.0 (constant, warm-up period)
- Steps 3000-10000: LR decays exponentially from 30.0 → 0.05
- Step 10000: LR = 0.05 (final value)

## Configuration

### JSON Config Format

```json
{
    "network": {
        "otype": "FullyFusedMLP",
        "density_grid_optimizer": {
            "otype": "ExponentialDecay",
            "decay_start": 3000,
            "decay_interval": 1,
            "learning_rate_start": 30.0,
            "learning_rate_end": 0.05,
            "total_training_steps": 10000,
            "nested": {
                "otype": "Adam",
                "learning_rate": 30.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-15
            }
        }
    }
}
```

### Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `learning_rate_start` | Starting LR (aggressive for density grid) | `30.0` |
| `learning_rate_end` | Final LR (for fine-tuning) | `0.05` |
| `total_training_steps` | Total number of training iterations | `10000` |
| `decay_start` | Step to begin decay (warm-up period) | `3000` |
| `decay_interval` | Apply decay every N steps (usually 1) | `1` |

**Note:** The `learning_rate` in the nested Adam config should match `learning_rate_start`.

## Usage Examples

### Short Training (10k steps)
```json
"density_grid_optimizer": {
    "otype": "ExponentialDecay",
    "decay_start": 2000,
    "learning_rate_start": 30.0,
    "learning_rate_end": 0.1,
    "total_training_steps": 10000,
    "nested": { "otype": "Adam", "learning_rate": 30.0 }
}
```
- Warm-up: 2000 steps at LR=30
- Decay: 8000 steps from 30→0.1

### Medium Training (50k steps)
```json
"density_grid_optimizer": {
    "otype": "ExponentialDecay",
    "decay_start": 3000,
    "learning_rate_start": 30.0,
    "learning_rate_end": 0.05,
    "total_training_steps": 50000,
    "nested": { "otype": "Adam", "learning_rate": 30.0 }
}
```
- Warm-up: 3000 steps at LR=30
- Decay: 47000 steps from 30→0.05

### Long Training (250k steps, Plenoxels paper)
```json
"density_grid_optimizer": {
    "otype": "ExponentialDecay",
    "decay_start": 15000,
    "learning_rate_start": 30.0,
    "learning_rate_end": 0.05,
    "total_training_steps": 250000,
    "nested": { "otype": "Adam", "learning_rate": 30.0 }
}
```
- Warm-up: 15000 steps at LR=30
- Decay: 235000 steps from 30→0.05

### Conservative Schedule
```json
"density_grid_optimizer": {
    "otype": "ExponentialDecay",
    "decay_start": 1000,
    "learning_rate_start": 10.0,
    "learning_rate_end": 0.01,
    "total_training_steps": 20000,
    "nested": { "otype": "Adam", "learning_rate": 10.0 }
}
```
- Lower starting LR for stability
- Gentler decay to standard NeRF LR

## Training Output

When the network is created, you'll see:
```
Creating separate optimizer for density grid (2097152 params)
Calculated decay_base for density grid optimizer:
  LR: 30 → 0.05 over 7000 steps (total=10000, decay_start=3000)
  decay_base = 0.999092732
Grid optimizer config: {"decay_base":0.999092732,...}
```

This confirms the calculation was performed correctly.

## Tips & Best Practices

### 1. Match Your Training Duration
Always set `total_training_steps` to match how long you actually plan to train:
```bash
python scripts/run.py --n_steps 10000  # Match this in config
```

### 2. Warm-up Period
- **Rule of thumb:** `decay_start = 10-20% of total_training_steps`
- Gives the grid time to initialize before aggressive decay
- For 10k steps: use 1k-2k warm-up
- For 50k steps: use 3k-5k warm-up

### 3. End LR Selection
- **Fine detail:** Use lower end LR (0.01-0.05)
- **Speed priority:** Use higher end LR (0.1-0.5)
- **Balance:** 0.05 is a good default

### 4. Start LR Guidelines
- **Plenoxels-style (aggressive):** 20-30
- **Conservative:** 5-10
- **Very safe:** 1-5

### 5. If You Change Training Duration Mid-Project
Just update `total_training_steps` in the config and restart training. The decay rate recalculates automatically.

## Fallback Behavior

If you don't specify `total_training_steps`, it defaults to **50,000 steps**:
```json
"density_grid_optimizer": {
    "learning_rate_start": 30.0,
    "learning_rate_end": 0.05
    // total_training_steps defaults to 50000
}
```

## Verification

To verify your schedule is correct, calculate expected LR at a given step:
```python
import math

lr_start = 30.0
lr_end = 0.05
decay_start = 3000
total_steps = 10000

decay_steps = total_steps - decay_start  # 7000
decay_base = (lr_end / lr_start) ** (1/decay_steps)

# LR at step 5000 (2000 steps into decay)
steps_decayed = 5000 - decay_start  # 2000
lr_at_5000 = lr_start * (decay_base ** steps_decayed)
print(f"LR at step 5000: {lr_at_5000:.4f}")
# Expected: ~6.9
```

## Migration from Old Configs

**Old format (manual decay_base):**
```json
"density_grid_optimizer": {
    "decay_base": 0.9999727794,  // Had to calculate this manually
    "nested": { "learning_rate": 30.0 }
}
```

**New format (automatic):**
```json
"density_grid_optimizer": {
    "learning_rate_start": 30.0,
    "learning_rate_end": 0.05,
    "total_training_steps": 50000,
    "nested": { "learning_rate": 30.0 }
}
```

Much cleaner and flexible! No more manual calculations.

