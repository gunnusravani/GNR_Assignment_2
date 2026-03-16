# GNR638 Assignment 2 Submission Guide

This repository contains code, experiments, and report artifacts for Assignment 2.  
All experiments use the **AID dataset** (30 classes) with ImageNet-pretrained backbones:
- `resnet50`
- `inception_v3`
- `densenet121`

---

## 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Structure

The code expects the AID dataset in the following structure:

```text
AID/
├── train/                 # Training split
│   ├── Airport/
│   ├── BareLand/
│   ├── ... (30 classes total)
│   └── Viaduct/
└── val/                   # Validation split
    ├── Airport/
    ├── BareLand/
    ├── ... (30 classes total)
    └── Viaduct/
```

**Default location:** `AID/` (configured in `config.py`)

---

## 3. Quick Start: Run All 4 Scenarios

Each scenario trains models and saves results. Edit `config.py` to select which scenario to run, then execute:

```bash
python run_experiments.py
```

---

## 4. Scenario-by-Scenario Instructions

### **Scenario 1: Linear Probe (Transfer Learning)**

**What it does:** Freeze the backbone, train only the task-specific classifier (probing the learned features).

**Steps:**

1. Open `config.py` and set:
   ```python
   exp_scenarios = ("linear_probe",)
   ```

2. Run the scenario:
   ```bash
   python run_experiments.py
   ```

3. **Outputs:**
   - Summary CSV: `results/results_linear_probe.csv`
   - Per-backbone results: `results/linear_probe/resnet50/<timestamp>/`
   - Artifacts: accuracy curves, confusion matrix, PCA/t-SNE visualizations

**Expected:** Training is fast (backbone frozen). One row per backbone in the CSV.

---

### **Scenario 2: Fine-Tuning Strategies**

**What it does:** Compare 4 different unfreezing strategies on the frozen backbone.

**Unfreezing modes:**
- `linear_probe`: Only train the task head (baseline)
- `last_block`: Unfreeze only the last residual block/layer
- `full_ft`: Unfreeze the entire backbone
- `selective_20`: Unfreeze only 20% of randomly selected backbone parameters

**Steps:**

1. Open `config.py` and set:
   ```python
   exp_scenarios = ("fine_tune",)
   fine_tune_modes = ("linear_probe", "last_block", "full_ft", "selective_20")
   ```

2. Run the scenario:
   ```bash
   python run_experiments.py
   ```

3. **Outputs:**
   - Summary CSV: `results/results_fine_tune.csv`
   - Per-backbone results: `results/fine_tune/resnet50/<timestamp>/modes/<mode>/`
   - Artifacts: accuracy vs unfrozen% plot, loss curves, gradient statistics

**Expected:** 4 modes × 3 backbones = 12 rows in the CSV. Training is longer than Scenario 1 due to backbone tuning.

---

### **Scenario 3: Few-Shot Learning**

**What it does:** Train on reduced data (100%, 20%, 5% of training set) using parameter-efficient transfer mode.

**Fractions tested:** 1.0 (100%), 0.2 (20%), 0.05 (5%)

**Steps:**

1. Open `config.py` and set:
   ```python
   exp_scenarios = ("few_shot",)
   few_shot_fracs = (1.0, 0.2, 0.05)
   few_shot_epochs = 20
   ```

2. Run the scenario:
   ```bash
   python run_experiments.py
   ```

3. **Outputs:**
   - Summary CSV: `results/results_few_shot.csv`
   - Per-backbone results: `results/few_shot/resnet50/<timestamp>/frac_1.00/`, `frac_0.20/`, `frac_0.05/`
   - Artifacts: accuracy curves, loss curves per fraction, performance drop metrics

**Expected:** 3 fractions × 3 backbones = 9 rows in the CSV. Smaller fractions will be noisier.

---

### **Scenario 4: Corruption Robustness**

**What it does:** Train a model and evaluate its robustness to image corruptions (noise, blur, brightness).

**Corruptions tested:**
- **Gaussian noise:** σ = 0.05, 0.10, 0.20
- **Motion blur:** kernel size = 9 (horizontal and vertical)
- **Brightness shift:** Δ = ±0.10, ±0.20

**Steps:**

1. Open `config.py` and set:
   ```python
   exp_scenarios = ("corruption",)
   corruption_gaussian_sigmas = (0.05, 0.10, 0.20)
   corruption_motion_blur_kernel = 9
   corruption_brightness_deltas = (-0.20, -0.10, 0.10, 0.20)
   ```

2. Run the scenario:
   ```bash
   python run_experiments.py
   ```

3. **Outputs:**
   - Summary CSV: `results/results_corruption.csv`
   - Per-backbone results: `results/corruption/resnet50/<timestamp>/`
   - Aggregated tables: `results/tables/robustness_comparison.csv` and `.tex`
   - Metrics: per-corruption accuracy, robustness scores

**Expected:** 1 row per backbone in the CSV with columns for each corruption type. Training is similar to Scenario 1 (linear_probe mode).

---

## 5. Evaluating Trained Models

### Evaluate a single checkpoint

```bash
python evaluate.py --ckpt model_checkpoints/linear_probe/resnet50/<timestamp>/best.pt
```

Output: validation accuracy on clean data

### Evaluate checkpoint on corrupted data

```bash
python evaluate.py --ckpt model_checkpoints/linear_probe/resnet50/<timestamp>/best.pt --robust
```

Output: accuracy scores on Gaussian noise, motion blur, brightness shifts

---

## 6. Understanding the Outputs

### Results CSVs
- `results/results_linear_probe.csv`: Scenario 1 results
- `results/results_fine_tune.csv`: Scenario 2 results
- `results/results_few_shot.csv`: Scenario 3 results
- `results/results_corruption.csv`: Scenario 4 results

### Checkpoint Structure
```text
model_checkpoints/
├── linear_probe/
│   ├── resnet50/<timestamp>/best.pt
│   ├── inception_v3/<timestamp>/best.pt
│   └── densenet121/<timestamp>/best.pt
├── fine_tune/
│   └── resnet50/<timestamp>/modes/(linear_probe|last_block|full_ft|selective_20)/best.pt
├── few_shot/
│   └── resnet50/<timestamp>/frac_1.00/best.pt  (etc. for 0.20, 0.05)
└── corruption/
    └── resnet50/<timestamp>/best.pt
```

### Artifacts Directory
Each scenario stores plots and analysis files in:
```text
results/<scenario>/<backbone>/<timestamp>/
├── accuracy_curves.png
├── confusion_matrix.png (Scenario 1)
├── feature_norm_vs_depth.png (Scenario 5)
└── (and other analysis files)
```

---

## 7. Configuration Quick Reference

All settings are in `config.py`:

| Setting | Purpose | Default |
|---------|---------|---------|
| `exp_scenarios` | Which scenarios to run | `("layer_probe",)` |
| `exp_models` | Which backbones to test | `("resnet50", "inception_v3", "densenet121")` |
| `seed` | Random seed for reproducibility | `1337` |
| `epochs` | Training epochs per scenario | `30` |
| `batch_size` | Batch size for training | `32` |
| `device` | Compute device | `"cuda"` (auto-falls-back to CPU) |

To run multiple scenarios in sequence, update `exp_scenarios`:
```python
exp_scenarios = ("linear_probe", "fine_tune", "few_shot", "corruption")
```

---

## 8. Reproducibility

To ensure reproducible results:
- Keep `seed = 1337` in `config.py`
- Use the same `requirements.txt` (Python 3.9+, PyTorch 2.0+)
- Keep the same hardware characteristics (GPU preferred, CPU fallback supported)
- Do not modify the AID dataset structure

---

## 9. Common Issues & Troubleshooting

| Issue | Solution |
|-------|----------|
| "AID/train not found" | Ensure AID dataset is in repo root with `train/` and `val/` subdirectories |
| CUDA out of memory | Reduce `batch_size` in `config.py` (e.g., 16 or 8) |
| Slow training | Increase `num_workers` in `config.py` (e.g., 8 or 12) |
| Different results | Ensure `seed = 1337` and no parallel runs interfering |

---

## 10. Report Compilation

The LaTeX report is in `report.tex`. To compile:

```bash
pdflatex report.tex
```

The report includes all results tables, figures, and discussion for each scenario.

---

## Notes

- Experiments automatically save checkpoints, results, and analysis artifacts
- CSV files accumulate results across runs (append mode enabled by default)
- All timestamps use UTC format `YYYYMMDD_HHMMSS` for uniqueness
- GPU usage is optional; code falls back gracefully to CPU if CUDA unavailable