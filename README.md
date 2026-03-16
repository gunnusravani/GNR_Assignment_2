# GNR638 Assignment 2 Submission Guide

This repository contains code and report artifacts for Assignment 2.
All experiments use AID (30 classes) with ImageNet-pretrained backbones:
- `resnet50`
- `inception_v3`
- `densenet121`

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset Layout

Expected folder structure:

```text
AID_split/
  train/
    <class_name>/*.jpg
  val/
    <class_name>/*.jpg
```

The default path is configured in `config.py`:
- `data_root = Path("AID_split")`

## 3. Output and Checkpoint Paths

- Experiment results: `results/<scenario>/<backbone>/<timestamp>/`
- Aggregated CSVs: `results/results_<scenario>.csv`
- Aggregated tables: `results/tables/*.csv` and `results/tables/*.tex`
- Checkpoints: `model_checkpoints/<scenario>/<backbone>/<timestamp>/`

## 4. How To Run Scenarios 1-4

`run_experiments.py` executes scenarios listed in `config.py` under `exp_scenarios`.
Set one scenario at a time (recommended), then run:

```bash
python run_experiments.py
```

### Scenario 1: Linear Probe Transfer

In `config.py`:
- `exp_scenarios = ("linear_probe",)`

Run:

```bash
python run_experiments.py
```

Outputs:
- `results/results_linear_probe.csv`
- per-backbone curves, confusion matrix, classification report, PCA/t-SNE

### Scenario 2: Fine-Tuning Strategies

In `config.py`:
- `exp_scenarios = ("fine_tune",)`
- `fine_tune_modes = ("linear_probe", "last_block", "full_ft", "selective_20")`

Run:

```bash
python run_experiments.py
```

Outputs:
- `results/results_fine_tune.csv`
- val-acc-vs-unfrozen plot
- per-mode accuracy/loss curves
- gradient norm summaries

### Scenario 3: Few-Shot Learning

In `config.py`:
- `exp_scenarios = ("few_shot",)`
- `few_shot_fracs = (1.0, 0.2, 0.05)`
- `few_shot_epochs = 20`

Run:

```bash
python run_experiments.py
```

Outputs:
- `results/results_few_shot.csv`
- per-frac accuracy/loss curves
- relative performance drop and train-val gap metrics

### Scenario 4: Corruption Robustness Evaluation

In `config.py`:
- `exp_scenarios = ("corruption",)`
- `corruption_gaussian_sigmas = (0.05, 0.10, 0.20)`
- `corruption_motion_blur_kernel = 9`
- `corruption_brightness_deltas = (-0.20, -0.10, 0.10, 0.20)`

Run:

```bash
python run_experiments.py
```

Outputs:
- `results/results_corruption.csv`
- `results/tables/robustness_comparison.csv` and `.tex`

Notes:
- Corruptions are applied only at evaluation time.
- Metrics include per-corruption accuracy, corruption error, and relative robustness.

## 5. How To Load and Evaluate Checkpoints

### Evaluate one checkpoint on clean validation set

```bash
python evaluate.py --ckpt model_checkpoints/<scenario>/<backbone>/<timestamp>/best.pt
```

### Evaluate one checkpoint on clean + corrupted validation sets

```bash
python evaluate.py --ckpt model_checkpoints/<scenario>/<backbone>/<timestamp>/best.pt --robust
```

### Batch corruption evaluation from Scenario 1 + Scenario 2 best checkpoints

This script:
- picks best checkpoints from `results/results_linear_probe.csv` and `results/results_fine_tune.csv`
- skips duplicate Scenario-2 `linear_probe` entries
- evaluates all selected checkpoints on corruption suite

```bash
python evaluate_corruption_checkpoints.py --results-dir results --out results/results_corruption_from_s1_s2.csv
```

## 6. Reproducibility Checklist

- Keep `seed = 1337` in `config.py`
- Keep all generated CSVs under `results/`
- Keep all checkpoints under `model_checkpoints/`
- Use same environment and dependency versions (`requirements.txt`)

## 7. Submission Notes

- This is a graded group submission; do not use additional datasets.
- Ensure the repository is private as per course policy.
- Submit the report PDF before deadline.
- Keep this repository history consistent with Assignment-1 repository continuity requirement.