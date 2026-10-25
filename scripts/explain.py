"""Step 4. Explainability analyses of the deployed model on the first session-wise partition: region
ablation (Figure A2) and Grad-CAM with occlusion sensitivity (Figure 8). Needs the checkpoints saved by
train.py and the result of analyse.py (deployed model).

    python scripts/explain.py --data-root /path/to/MangoLeafBD --out outputs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mangoleaf import figures as F
from mangoleaf.config import NICE, add_common_args, config_from_args
from mangoleaf.data import ImageStore
from mangoleaf.experiments import run_id
from mangoleaf.models import ModelZoo
from mangoleaf.study import load
from mangoleaf.utils import device, update_numbers
from mangoleaf.xai import fig8_gradcam_occlusion, region_ablation


def main():
    a = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(a)
    st = load(cfg); dev = device()
    nums = json.load(open(cfg.tbl / "numbers.json"))
    inv = {v: k for k, v in NICE.items()}
    dep = cfg.deploy_model or inv[nums["deployed_model"]]
    zoo = ModelZoo(cfg, st.num_classes, dev)
    store = ImageStore(st.full.path if cfg.cache_images else (), cache=cfg.cache_images)
    rng = np.random.default_rng(0)
    F.set_style(cfg.save_dpi)

    abl = []
    folds = [k for k in range(cfg.n_folds) if zoo.has_ckpt(run_id(dep, "SW", f"p0f{k}", cfg.lr, "squash", cfg.lr))]
    for k in folds:
        net = zoo.load_ckpt(dep, run_id(dep, "SW", f"p0f{k}", cfg.lr, "squash", cfg.lr))
        te = st.splits[("SW", f"p0f{k}")][2]
        keep = te if len(te) <= cfg.xai_per_fold else rng.choice(te, cfg.xai_per_fold, replace=False)
        abl.append(region_ablation(net, st, keep, store, dev, k, cfg.img_size)); del net
    if abl:
        ab = pd.concat(abl, ignore_index=True); ab.to_csv(cfg.tbl / "extra_region_ablation_per_image.csv", index=False)
        m = ab[["full", "centre_only", "centre_removed"]].mean(); print(m.round(4).to_string())
        update_numbers(cfg.tbl / "numbers.json", "region_ablation", m.round(4).to_dict())
        F.savefig(F.figA2_region(ab), cfg.fig, "FigA2_region_ablation", cfg.save_dpi)
    if 0 in folds:
        net = zoo.load_ckpt(dep, run_id(dep, "SW", "p0f0", cfg.lr, "squash", cfg.lr))
        fig = fig8_gradcam_occlusion(net, st, st.splits[("SW", "p0f0")][2], store, dev, rng, cfg.img_size)
        F.savefig(fig, cfg.fig, "Fig8_gradcam_occlusion", cfg.save_dpi)
    else:
        print("no checkpoint of the deployed model on partition 0: run train.py --groups sw first")


if __name__ == "__main__":
    main()
