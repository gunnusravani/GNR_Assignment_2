# GNR638 Assignment 2 - Transfer Learning Experiment Framework (PyTorch)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset layout (ImageFolder)
Expected:
```
AID/
  train/
    class_0/*.jpg
    ...
  val/
    class_0/*.jpg
    ...
```

## Run training
```bash
python train.py
```

## Run evaluation
```bash
python evaluate.py --ckpt outputs/latest.pt
```

## Run experiments + auto-generate report tables (CSV + LaTeX)
```bash
python run_experiments.py
```

Tables are written to:
- `outputs/tables/accuracy_comparison.csv` + `.tex`
- `outputs/tables/robustness_comparison.csv` + `.tex`
- `outputs/tables/parameter_efficiency.csv` + `.tex`