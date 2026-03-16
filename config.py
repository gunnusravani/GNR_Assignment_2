from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    """
    Global configuration for GNR Assignment 2 experiments.
    
    ========================================
    QUICK START: Edit these to run scenarios
    ========================================
    1. exp_scenarios - Which scenarios to run (linear_probe, fine_tune, few_shot, corruption)
    2. exp_models - Which backbones to test (resnet50, inception_v3, densenet121)
    3. device - GPU ("cuda") or CPU ("cpu")
    """

    # ========== DATASET & PATHS ==========
    data_root: Path = Path("AID")  # Root folder: expects AID/train/<class>/*.jpg and AID/val/<class>/*.jpg
    output_dir: Path = Path("results")
    checkpoint_dir: Path = Path("model_checkpoints")

    # ========== TASK SETUP ==========
    num_classes: int = 30  # AID dataset has 30 classes
    image_size: int = 299  # 299 for Inception-v3; also works for ResNet50 & DenseNet121

    # ========== MODEL SETTINGS (used for single-model evaluation) ==========
    backbone: str = "resnet50"  # Model name: resnet50 | inception_v3 | densenet121
    pretrained: bool = True  # Load ImageNet-pretrained weights
    transfer_mode: str = "linear_probe"  # linear_probe | full_ft | last_block | selective_20

    # ========== TRAINING HYPERPARAMETERS ==========
    seed: int = 1337  # Random seed for reproducibility (DO NOT CHANGE)
    epochs: int = 30  # Training epochs per scenario
    batch_size: int = 32  # Batch size (reduce to 16/8 if CUDA out of memory)
    lr: float = 1e-3  # Learning rate
    weight_decay: float = 1e-4  # L2 regularization
    num_workers: int = 4  # Data loader workers (increase to 8-12 for faster loading)
    device: str = "cuda"  # "cuda" for GPU, "cpu" for CPU (auto-detects GPU availability)

    # ========== EXPERIMENT SELECTION ==========
    # MAIN CONFIG: Edit this to choose which scenarios to run
    # Options: "linear_probe", "fine_tune", "few_shot", "corruption", "layer_probe"
    # Example: exp_scenarios = ("linear_probe", "fine_tune", "few_shot", "corruption")
    exp_scenarios: tuple[str, ...] = ("layer_probe",)  # ← CHANGE THIS TO RUN DIFFERENT SCENARIOS
    
    # Backbones to test across all scenarios
    exp_models: tuple[str, ...] = ("resnet50", "inception_v3", "densenet121")

    # ========== SCENARIO 2: Fine-Tuning Modes ==========
    # Only used when exp_scenarios = ("fine_tune",)
    # Compares 4 unfreezing strategies:
    #   - linear_probe: only train classifier (baseline)
    #   - last_block: unfreeze last layer/block
    #   - full_ft: unfreeze entire backbone
    #   - selective_20: randomly unfreeze 20% of params
    fine_tune_modes: tuple[str, ...] = ("linear_probe", "last_block", "full_ft", "selective_20")

    # ========== SCENARIO 3: Few-Shot Learning ==========
    # Only used when exp_scenarios = ("few_shot",)
    # Trains on reduced data: 1.0 (100%), 0.2 (20%), 0.05 (5%)
    few_shot_frac: float = 0.05  # Single-fraction knob (kept for compatibility)
    few_shot_fracs: tuple[float, ...] = (1.0, 0.2, 0.05)  # Fractions to test
    few_shot_seed: int = 1337  # Seed for fraction sampling (reproducible subsets)
    few_shot_epochs: int = 20  # Epochs for few-shot training (usually less than full training)

    # ========== SCENARIO 5: Layer-Wise Probing ==========
    # Only used when exp_scenarios = ("layer_probe",)
    # Trains a backbone and probes intermediate layer representations
    layer_probe_max_batches: int | None = None  # Max batches for feature extraction (None = use all)
    layer_probe_samples_per_class: int = 30  # Fixed validation subset for reproducible PCA

    # ========== SCENARIO 4: Corruption Robustness ==========
    # Only used when exp_scenarios = ("corruption",)
    # Applies corruptions at evaluation time and measures robustness
    corruptions_enabled: bool = True
    # Gaussian noise (additive)
    corruption_gaussian_sigmas: tuple[float, ...] = (0.05, 0.10, 0.20)
    # Motion blur (kernel size)
    corruption_motion_blur_kernel: int = 9
    # Brightness shift (delta on [0, 1] scale)
    corruption_brightness_deltas: tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20)

    # ========== EXPERIMENT I/O ==========
    # Output structure: results/<scenario>/<backbone>/<timestamp>/
    exp_out_dir: Path = Path("results")
    exp_ckpt_dir: Path = Path("model_checkpoints")
    # If True, append to existing CSVs; if False, overwrite
    exp_append_csv: bool = True

    # ========== DATALOADER SETTINGS ==========
    pin_memory: bool = True  # GPU memory optimization (faster loading on CUDA)
    persistent_workers: bool = None  # Auto: True if num_workers>0, else False

    # ========== LOGGING & DIAGNOSTICS ==========
    log_grad_norms: bool = True  # Log gradient norms during training
    grad_norms_every_n_steps: int = 1  # Sample frequency for grad logging
    log_metrics_jsonl: bool = True  # Log training metrics to JSONL

    # ========== FEATURE EXTRACTION ==========
    # Used for layer probing and feature visualization
    feature_extract_max_batches: int | None = None  # Max batches for feature extraction (None = all)

    # ========== INTERNAL: Scenario → Transfer Mode Mapping ==========
    # DO NOT EDIT DIRECTLY: Set in __post_init__-style function below
    # This determines which transfer mode is used for each scenario
    scenario_transfer_mode: dict[str, str] = None


# ========== SCENARIO-TO-TRANSFER-MODE DEFAULTS ==========
# DO NOT EDIT: These are automatically set below
def _with_defaults() -> Config:
    """Set up scenario defaults. Frozen dataclass workaround."""
    cfg = Config()
    object.__setattr__(
        cfg,
        "scenario_transfer_mode",
        {
            # Scenario 1: Always linear_probe (freeze backbone, train classifier only)
            "linear_probe": "linear_probe",

            # Scenario 2: fine_tune uses multiple modes (defined in fine_tune_modes above)
            # This is the default for the comparison plot
            "fine_tune": "full_ft",

            # Scenario 3: Few-shot uses parameter-efficient mode by default
            "few_shot": "selective_20",

            # Scenario 4: Corruption robustness uses last_block unfreezing
            "corruption": "last_block",

            # Scenario 5: Layer probe first trains a full_ft model, then extracts features
            "layer_probe": "full_ft",
        },
    )
    return cfg

# Global config object (used throughout the codebase)
CFG = _with_defaults()

