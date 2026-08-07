# MMFI Dataset Migration Guide

## Overview

This codebase has been updated to support both **Person-in-WiFi-3D** (14 joints) and **MMFI** (17 joints) datasets for backdoor attacks on WiFi-based human pose estimation.

## Key Changes

### 1. Skeleton Structure

**Person-in-WiFi-3D (14 joints):**
- Root joint: 3 (torso)
- 13 edges forming a tree structure
- Joints: 0-13

**MMFI (17 joints):**
- Root joint: 5 (left shoulder, central hub)
- 16 edges as specified in the MMFI paper:
  ```python
  (0,1), (1,3), (0,2), (2,4),           # legs
  (5,6), (5,7), (7,9), (6,8), (8,10),  # arms
  (5,11), (6,12), (11,12),               # shoulders-torso
  (11,13), (13,15), (12,14), (14,16),   # spine-head
  ```

### 2. Modified Files

#### Core Attack Files
- **`attack/payload.py`**: Added `set_skeleton_config()` function and MMFI edges
- **`attack/poison.py`**: Added `dataset` parameter to configure skeleton
- **`attack/skeleton_mmfi.py`**: NEW - MMFI skeleton definition

#### Data Loading
- **`data_utils/feeder.py`**: Added `MMFI` dataset class

#### Models
- **`models/factory.py`**: Auto-adjusts num_keypoints based on dataset
- **`models/graphposefi.py`**: Refactored to support variable keypoints (14 or 17)
- **`models/hpeli.py`**: Already flexible (no changes needed)
- **`models/metafiplusplus.py`**: Already flexible (no changes needed)

#### Evaluation & Visualization
- **`eval/metrics.py`**: Updated to handle variable joint numbers
- **`eval/vis_skeleton.py`**: Added MMFI skeleton edges and dataset parameter

#### Configuration
- **`configs/mmfi/attack_linear.yaml`**: NEW - Linear dose mapping
- **`configs/mmfi/attack_sqrt.yaml`**: NEW - Square root dose mapping
- **`configs/mmfi/attack_quad.yaml`**: NEW - Quadratic dose mapping

## Usage

### 1. Check MMFI Data

First, verify your MMFI dataset is properly structured:

```bash
python check_mmfi_data.py
```

Expected output:
- CSI shape: (3, 180, 20)
- Pose shape: (1, 17, 3)

### 2. Configure Attack

Edit config file `configs/mmfi/attack_linear.yaml`:

```yaml
dataset_root: d:/BackdoorAtt/data/raw
experiment_name: mmfi
model: graphposefi   # or hpeli, metafiplusplus

pivot: 7             # L-elbow (adjust based on attack target)
theta_max_deg: 60.0
dose_mode: linear
rho: 0.1
```

### 3. Run Attack

```bash
python run_sweep.py --config configs/mmfi/attack_linear.yaml
```

### 4. Important Parameters

**Pivot Selection (MMFI):**
- Joint 7: L-elbow (left arm)
- Joint 8: R-elbow (right arm)  
- Joint 1: L-knee (left leg)
- Joint 2: R-knee (right leg)

**Recommended Settings:**
- `pivot`: 7 (L-elbow) - good for arm movements
- `theta_max_deg`: 60.0 (moderate rotation)
- `dose_mode`: 'linear' (linear dose-to-angle mapping)
- `rho`: 0.1-0.3 (poisoning rate)
- `eps`: 0.3-0.5 (trigger strength)

## Trigger Methods

Both trigger types from the original paper are preserved:

1. **MicroDopplerTrigger** (`trigger.py`): Physics-based Doppler signatures
2. **WaNetTrigger** (`trigger_wanet.py`): Smooth warp-based perturbations

Both work identically for MMFI and Person-in-WiFi-3D (trigger applies to CSI, not skeleton).

## Model Compatibility

All three models now support both datasets:

| Model | PWIF3D (14) | MMFI (17) | Notes |
|-------|-------------|-----------|-------|
| **HPELi** | ✓ | ✓ | CNN-based, automatically adapts |
| **MetaFiPlusPlus** | ✓ | ✓ | ResNet34 + Transformer, adapts via resize |
| **GraphPoseFi** | ✓ | ✓ | Graph conv, dynamic adjacency matrix |

## Verification Steps

1. **Data Check:**
   ```bash
   python check_mmfi_data.py
   ```

2. **Skeleton Visualization:**
   ```bash
   python debug_skeleton.py
   ```

3. **Model Test:**
   ```python
   from models.factory import build_model
   
   # MMFI
   model = build_model('graphposefi', num_keypoints=17, dataset='mmfi')
   print(model)  # Should show 17-joint configuration
   ```

4. **Attack Test:**
   ```python
   from attack.payload import set_skeleton_config, descendants
   
   # Configure for MMFI
   set_skeleton_config('mmfi')
   
   # Test pivot 7 (L-elbow)
   desc = descendants(7)
   print(f"Pivot 7 descendants: {desc}")  # Should show joints distal to L-elbow
   ```

## Dataset Path Configuration

Update paths in config files to match your setup:

```yaml
# For Windows (current setup)
dataset_root: d:/BackdoorAtt/data/raw

# For Linux
dataset_root: /path/to/mmfi/data
```

## Known Issues & Notes

1. **Trigger Skeleton**: Currently uses NTU skeleton for generating micro-Doppler. May need adjustment for MMFI-specific motion patterns.

2. **Plausibility Threshold**: `tau_plaus` may need tuning for MMFI (default 0.35, adjust based on bone length distributions).

3. **Pivot Selection**: MMFI skeleton structure differs from PWIF3D. Verify anatomical correspondence before choosing pivot joint.

4. **Root Joint**: MMFI uses joint 5 (L-shoulder) as root instead of joint 3 (torso) in PWIF3D. Tree structure is rebuilt automatically.

## Paper Method Preservation

All core methods from the BackWiFi paper are preserved:

- ✓ Dose-dependent rotation attack (linear/sqrt/quad)
- ✓ MicroDoppler and WaNet triggers  
- ✓ Conjunctive ASR metric (landed & preserved & plausible)
- ✓ Non-target preservation
- ✓ Poison selection strategies (uniform/diverse FPS)

## Troubleshooting

**Issue**: Model fails with dimension mismatch
- **Solution**: Check `num_keypoints` is correctly set to 17 for MMFI

**Issue**: Skeleton visualization looks wrong
- **Solution**: Verify edge definitions in `eval/vis_skeleton.py` match MMFI structure

**Issue**: Attack success rate is 0
- **Solution**: Check pivot joint is valid and has descendants in MMFI skeleton tree

**Issue**: Data loading fails
- **Solution**: Run `check_mmfi_data.py` to verify file structure and paths

## References

- Original paper: BackWiFi (AAAI 2027)
- MMFI dataset: 17-joint human pose from WiFi CSI
- GraphPose-Fi: https://github.com/Cirrick/GraphPose-Fi
