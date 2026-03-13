from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # --- Paths (edit these) ---
    data_root: Path = Path("AID_split")  # expects ImageFolder-like: AID_split/train/<class>/*.jpg etc.
    output_dir: Path = Path("results")
    checkpoint_dir: Path = Path("model_checkpoints")

    # --- Task ---
    num_classes: int = 30
    image_size: int = 299  # use 299 for inception_v3; also works for resnet50/densenet121

    # --- Model / Transfer mode ---
    backbone: str = "resnet50"  # timm model name
    pretrained: bool = True
    transfer_mode: str = "linear_probe"  # linear_probe | full_ft | last_block | selective_20

    # --- Training ---
    seed: int = 1337
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"  # default to GPU; code falls back to CPU if CUDA isn't available

    # --- Experiments ---
    exp_models: tuple[str, ...] = ("resnet50", "inception_v3", "densenet121")
    exp_scenarios: tuple[str, ...] = ("linear_probe",)

    # Scenario knobs
    few_shot_frac: float = 0.05          # uses subset of train set
    few_shot_seed: int = 1337
    layer_probe_max_batches: int | None = None  # cap feature extraction batches (None = full)
    corruptions_enabled: bool = True

    # Experiment runner I/O
    exp_out_dir: Path = Path("results")
    exp_ckpt_dir: Path = Path("model_checkpoints")
    exp_append_csv: bool = True  # if False, runner overwrites CSVs each run

    # --- DataLoader defaults ---
    pin_memory: bool = True  # better throughput on CUDA
    persistent_workers: bool = None  # auto: True if num_workers>0 else False

    # --- Logging / diagnostics ---
    log_grad_norms: bool = True
    grad_norms_every_n_steps: int = 1  # 1 = every step, >1 = subsample
    log_metrics_jsonl: bool = True

    # --- Robustness / corruptions ---
    corruption_gaussian_sigmas: tuple[float, ...] = (0.05, 0.10, 0.20)
    corruption_motion_blur_kernel: int = 9
    corruption_brightness_deltas: tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20)

    # --- Feature extraction / layer probe ---
    feature_extract_max_batches: int | None = None  # default cap for feature extraction (None = full)

    # --- Scenario -> transfer_mode mapping (experiments) ---
    scenario_transfer_mode: dict[str, str] = None  # set in __post_init__ style below (frozen workaround)

# Frozen dataclass workaround for derived defaults
_default = Config.__dataclass_fields__  # type: ignore[attr-defined]

def _with_defaults() -> Config:
    cfg = Config()
    object.__setattr__(
        cfg,
        "scenario_transfer_mode",
        {
            "linear_probe": "linear_probe",

            # Default fine-tuning strategy per scenario (change as you like)
            "fine_tune": "full_ft",

            # Examples: make these parameter-efficient by default
            "few_shot": "selective_20",
            "corruption": "last_block",

            # layer_probe trains a full_ft model first (see run_experiments.py),
            # then probes depths on that trained model.
            "layer_probe": "full_ft",
        },
    )
    return cfg

CFG = _with_defaults()
