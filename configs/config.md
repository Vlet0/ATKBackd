# Config Reference for Victim Models

This document describes the configuration setup for the three victim models used in your experiments:

- `hpeli`
- `metafiplusplus`
- `graphposefi`

All configs are YAML files under `backdoorviet/configs/`.

## Common settings across all three models

These fields are shared in the current configs.

- `dataset_root`: `wbackdoor/Person-in-WiFi-3D`
  - Base data directory for Person-in-WiFi-3D experiments.
- `experiment_name`: `one-person`
  - Specifies the target experiment subset; in this repo it selects one-person poses.
- `pretrained`: `false`
  - Model weights are trained from scratch, not loaded from ImageNet or other pretrained sources.

- `action_npy`: `wbackdoor/data/data_bend.npy`
  - The target action skeleton used for the attack. In your current setup this is the `bend` target.
- `top_k`: `6`
  - Number of skeleton joints used to build the trigger.
- `aoa_spread`: `0.6`
  - Controls angle-of-arrival variation in the trigger.
- `eps`: `0.5`
  - Trigger strength scale.

- `pivot`: `6`
  - Pivot joint index used when measuring attack effect.
- `theta_max_deg`: `20.0`
  - Maximum angle magnitude for the trigger.
- `dose_mode`: `linear`
  - Dose mapping: `linear` means trigger impact increases linearly with dose.

- `poison_select`: `uniform`
  - Poisoned samples are selected uniformly from the clean training set.
- `rho`: `0.3`
  - Poisoning rate: 30% of training samples are poisoned.
- `dose_min`: `0.2`
  - Minimum dose value for poisoning.
- `dose_max`: `1.0`
  - Maximum dose value.
- `dose_grid`: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
  - The dose grid evaluated during attack analysis.

- `tau_plaus`: `0.35`
  - Plausibility threshold for pose evaluation.
- `k_attack`: `1.5`
- `k_clean`: `1.5`
  - Multipliers used by attack and clean thresholds in metrics.

- `device`: `null`
  - Auto-selects CUDA if available, otherwise CPU.
- `data_parallel`: `false`
  - Disabled by default; training runs on a single device unless explicitly enabled.
- `num_workers`: `0`
  - No data loader multiprocessing for these configs.
- `seed`: `0`
  - Fixed seed for reproducibility.

## Per-model training settings

### `hpeli`

Config file: `backdoorviet/configs/hpeli/attack_ln.yaml`

- `model`: `hpeli`
- `batch_size`: `32`
- `lr`: `0.001`
- `epochs`: `50`
- `weight_decay`: not specified (default used by trainer is `1e-4` via AdamW)

This config is the standard HPELiNet setup for Person-in-WiFi-3D with moderate batch size and a learning rate of 1e-3.

### `metafiplusplus`

Config file: `backdoorviet/configs/metafiplusplus/attack_ln.yaml`

- `model`: `metafiplusplus`
- `batch_size`: `32`
- `lr`: `0.001`
- `epochs`: `50`
- `weight_decay`: not specified (default `1e-4` in trainer)

This matches the HPELiNet settings, since `metafiplusplus` is tuned to the same batch size and lr in your current setup.

### `graphposefi`

Config file: `backdoorviet/configs/graphposefi/attack_ln.yaml`

- `model`: `graphposefi`
- `batch_size`: `256`
- `lr`: `0.0003`
- `epochs`: `50`
- `weight_decay`: `0.02`

GraphPoseFi is trained with a much larger batch size and stronger weight decay, reflecting its different architecture and optimization regime.

## How training is run

Training is executed through `backdoorviet/run_experiments.py`.

Typical command line:

```bash
python3 run_experiments.py --models graphposefi --scenarios ln --triggers micro_dropper --dataset mmfi
```

Key flags:

- `--models`: select one or more of `hpeli`, `metafiplusplus`, `graphposefi`.
- `--scenarios`: select dose mapping variants, e.g. `ln`, `sqrt`, `quad`.
- `--triggers`: list trigger names, e.g. `micro_dropper`, `sig`, `wanet`, `blended`.
- `--dataset`: `mmfi` or `pwif3d`.
- `--epochs`: override the configured number of epochs.

### Config selection logic

- For dataset `pwif3d`, the script loads model-specific config folders:
  - `hpeli` → `configs/hpeli`
  - `metafiplusplus` → `configs/metafiplusplus`
  - `graphposefi` → `configs/graphposefi`

- For dataset `mmfi`, the script uses `configs/mmfi` and overrides `cfg['model']` for each victim model.

- `run_experiments.py` also contains model-specific override logic for batch size and epochs.

## Notes on training details

- The current config files are all set to `scenario = ln` (linear dose mapping).
- Target action is currently fixed to `data_bend.npy` in all configs; to run `cross` or `nod`, change `action_npy` accordingly.
- `hpeli` and `metafiplusplus` share the same optimization hyperparameters, while `graphposefi` uses a different training regime.

## Suggested config edits for other experiments

- To evaluate a different trigger target, change `action_npy` in the YAML:
  - `data_bend.npy`
  - `data_cross.npy`
  - `data_nod.npy`

- To run `graphposefi` on a different scenario, use the corresponding config file:
  - `configs/graphposefi/attack_ln.yaml`
  - `configs/graphposefi/attack_sqrt.yaml`
  - `configs/graphposefi/attack_quad.yaml`

- For `hpeli` / `metafiplusplus`, use:
  - `configs/hpeli/attack_ln.yaml`
  - `configs/hpeli/attack_sqrt.yaml`
  - `configs/hpeli/attack_quad.yaml`
  - `configs/metafiplusplus/attack_ln.yaml`
  - `configs/metafiplusplus/attack_sqrt.yaml`
  - `configs/metafiplusplus/attack_quad.yaml`

## Paths and files

- Config root: `backdoorviet/configs/`
- Model runner: `backdoorviet/run_experiments.py`
- Dataset loader: `backdoorviet/train_backdoor.py` / `backdoorviet/data_utils/feeder.py`
- Result outputs: `backdoorviet/experiments_out/`

If you want, I can also add a one-page `train.md` describing the exact command sequences for all three models and how to switch between `pwif3d` and `mmfi` runs.
