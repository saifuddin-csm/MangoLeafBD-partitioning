"""Step 2. Train all models (103 runs in the full configuration) and store their predictions.

The run is resumable: finished runs are listed in outputs/tables/results_rev.csv and skipped. Groups:
  sw         4 models x 10 session-wise test folds                       (40 runs)
  main       4 models x 3 seeds x {random, leaf-wise}                    (24 runs)
  control    matched-size control, 4 models x 3 seeds                    (12 runs)
  tolerance  ResNet-50 under leaf-wise partitioning at tau = 2, 4, 5 s   (9 runs)
  lr         peak learning rates 1e-4 and 1e-3 on the first partition    (16 runs)
  pad        Swin-T with padding instead of squashing                    (2 runs)

    python scripts/train.py --data-root /path/to/MangoLeafBD --out outputs
    python scripts/train.py ... --groups sw main          # a subset, e.g. to split the work across machines
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
import timm

from mangoleaf.config import GROUPS, add_common_args, config_from_args
from mangoleaf.experiments import done_ids, run_all
from mangoleaf.models import ModelZoo
from mangoleaf.study import load
from mangoleaf.utils import device


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--groups", nargs="*", default=GROUPS, choices=GROUPS)
    p.add_argument("--time-budget-min", type=float, default=None, help="stop starting new runs after this time")
    a = p.parse_args()
    cfg = config_from_args(a)
    print("torch", torch.__version__, "| timm", timm.__version__, "| device", device())
    st = load(cfg)
    specs = run_all(cfg, st, a.groups, a.time_budget_min)
    missing = [s["run_id"] for s in specs if s["run_id"] not in done_ids(cfg)]
    print(f"\nfinished {len(specs) - len(missing)} / {len(specs)} runs; missing {len(missing)}")

    lat_csv = cfg.tbl / "latency.csv"          # inference cost (Table 9), measured once on this device
    if not lat_csv.exists():
        zoo = ModelZoo(cfg, st.num_classes, device()); zoo.fetch(cfg.models)
        zoo.latency(cfg.models).to_csv(lat_csv, index=False)
    print(pd.read_csv(lat_csv).to_string(index=False))


if __name__ == "__main__":
    main()
