"""Experiment configuration.

Every value used in the article is defined here. ``Config.quick`` gives a short end-to-end check on the real
data; ``Config.dry_run`` runs offline on synthetic images with random weights (used to test the code).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ALL_MODELS = ["vgg16", "resnet50", "vit_base_patch16_224", "swin_tiny_patch4_window7_224"]
NICE = {"vgg16": "VGG-16", "resnet50": "ResNet-50",
        "vit_base_patch16_224": "ViT-B/16", "swin_tiny_patch4_window7_224": "Swin-T"}
FAMILY = {"vgg16": "CNN", "resnet50": "CNN",
          "vit_base_patch16_224": "Attention", "swin_tiny_patch4_window7_224": "Attention"}
CLASS_NAMES = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
               "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]
SHORT = {"Anthracnose": "Anthracnose", "Bacterial Canker": "Bact. canker", "Cutting Weevil": "Cut. weevil",
         "Die Back": "Die back", "Gall Midge": "Gall midge", "Healthy": "Healthy",
         "Powdery Mildew": "Powd. mildew", "Sooty Mould": "Sooty mould"}

# training groups of the run registry (see experiments.py)
GROUPS = ["sw", "main", "control", "tolerance", "lr", "pad"]


@dataclass
class Config:
    data_root: Path = Path("data/MangoLeafBD")
    out: Path = Path("outputs")
    quick: bool = False
    dry_run: bool = False

    models: list = field(default_factory=lambda: list(ALL_MODELS))
    seeds: list = field(default_factory=lambda: [42, 1337, 2024])   # repetitions for RP and LW
    n_sw_partitions: int = 5          # session-disjoint 2-fold partitions -> 10 test folds
    n_folds: int = 2                  # the maximum: two categories have exactly two sessions

    leaf_tol_s: int = 3               # tau_l, Eq. (1)
    session_tol_s: int = 1800         # tau_s, Eq. (1)
    l_star: int = 1800                # distinct leaves reported by the dataset authors, Eq. (2)
    tol_sweep: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 10, 30])
    tol_retrain: list = field(default_factory=lambda: [2, 4, 5])
    tol_model: str = "resnet50"
    min_sessions: int = 2

    img_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    lr_sweep: list = field(default_factory=lambda: [1e-4, 1e-3])
    weight_decay: float = 1e-4
    label_smooth: float = 0.05
    test_frac: float = 0.20
    val_frac: float = 0.10
    n_boot: int = 2000
    ece_bins: int = 15
    targets: list = field(default_factory=lambda: [0.95, 0.90, 0.85, 0.80])
    reliable_min: float = 0.90        # precision AND recall >= this -> "reliable"
    borderline_f1: float = 0.80       # otherwise F1 >= this -> "borderline", else "not reliable"
    deploy_model: str | None = None   # None -> highest mean session-wise accuracy over the folds
    pad_model: str = "swin_tiny_patch4_window7_224"
    xai_per_fold: int = 300
    save_dpi: int = 600
    cache_images: bool = True
    num_workers: int = 4

    def __post_init__(self):
        self.data_root, self.out = Path(self.data_root), Path(self.out)
        if self.quick:
            self.models = ["resnet50"]; self.epochs = 1; self.seeds = [42]; self.n_sw_partitions = 1
            self.n_boot = 200; self.tol_retrain = [2]; self.lr_sweep = [1e-4]; self.pad_model = "resnet50"
        if self.dry_run:
            self.models = ["resnet50", "swin_tiny_patch4_window7_224"]; self.epochs = 1; self.seeds = [42]
            self.n_sw_partitions = 2; self.n_boot = 100; self.tol_retrain = [2]; self.lr_sweep = [1e-4]
            self.tol_model = "resnet50"; self.pad_model = "resnet50"; self.xai_per_fold = 20
            self.save_dpi = 120; self.num_workers = 0
        self.num_workers = min(self.num_workers, os.cpu_count() or 2)
        for d in (self.out, self.fig, self.tbl, self.pred, self.ckpt, self.work, self.pretrained):
            d.mkdir(parents=True, exist_ok=True)

    # output folders
    @property
    def fig(self): return self.out / "figures"
    @property
    def tbl(self): return self.out / "tables"
    @property
    def pred(self): return self.out / "preds"
    @property
    def ckpt(self): return self.out / "ckpt"
    @property
    def work(self): return self.out / "work"
    @property
    def pretrained(self): return self.out / "pretrained"
    @property
    def results_csv(self): return self.tbl / "results_rev.csv"
    @property
    def history_csv(self): return self.tbl / "history_rev.csv"


def add_common_args(p):
    """Command-line options shared by all scripts."""
    p.add_argument("--data-root", default="data/MangoLeafBD",
                   help="folder containing the eight MangoLeafBD category folders (searched recursively)")
    p.add_argument("--out", default="outputs", help="output folder (tables, figures, predictions, checkpoints)")
    p.add_argument("--quick", action="store_true", help="1 model, 1 epoch, 1 seed, 1 partition (pipeline check)")
    p.add_argument("--dry-run", action="store_true", help="offline test on synthetic images with random weights")
    return p


def config_from_args(a) -> Config:
    return Config(data_root=a.data_root, out=a.out, quick=a.quick, dry_run=a.dry_run)
