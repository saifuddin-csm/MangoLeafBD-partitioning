"""Corpus indexing, audit and provenance recovery (leaf groups and acquisition sessions).

Implements Eq. (1) (gap grouping of capture times), Eq. (2) (calibration of the leaf tolerance) and the
corpus audit of Table A1, and builds Table 1 (composition).
"""
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ExifTags, Image

from .config import CLASS_NAMES

IMG_EXT = {".jpg", ".jpeg", ".png"}
TS_PATTERN = re.compile(r"(\d{8})[_-](\d{6})")


def index_images(root: Path) -> pd.DataFrame:
    """One row per image found under a folder named after a MangoLeafBD category."""
    canon = {c.lower(): c for c in CLASS_NAMES}
    rows = []
    for dp, _, fn in os.walk(root):
        dp = Path(dp)
        if dp.name.lower() not in canon:
            continue
        for f in fn:
            if Path(f).suffix.lower() in IMG_EXT:
                rows.append({"path": str(dp / f), "file": f, "stem": Path(f).stem, "cls": canon[dp.name.lower()]})
    full = pd.DataFrame(rows).drop_duplicates(["cls", "file"]).reset_index(drop=True)
    if full.empty or full.cls.nunique() != len(CLASS_NAMES):
        raise FileNotFoundError(f"MangoLeafBD category folders not found under {root}")
    return full


def read_meta(path):
    """Device make/model and capture time from EXIF, pixel size, and an MD5 of the decoded pixels."""
    o = {"Make": None, "Model": None, "w": None, "h": None, "px_md5": None, "exif_dt": None}
    try:
        with Image.open(path) as im:
            o["w"], o["h"] = im.size
            ex = im.getexif() or {}
            for tid, v in ex.items():
                nm = ExifTags.TAGS.get(tid, "")
                if nm in ("Make", "Model"): o[nm] = str(v).strip().strip("\x00")
                if nm == "DateTime": o["exif_dt"] = str(v)
            o["px_md5"] = hashlib.md5(im.convert("RGB").tobytes()).hexdigest()
    except Exception:
        pass
    return o


def capture_time(row):
    """Capture time from the file name (YYYYMMDD_HHMMSS), falling back to EXIF DateTime."""
    m = TS_PATTERN.search(row["stem"])
    if m:
        try:
            return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
        except Exception:
            pass
    try:
        return pd.to_datetime(row["exif_dt"], format="%Y:%m:%d %H:%M:%S")
    except Exception:
        return pd.NaT


def gap_groups(frame: pd.DataFrame, secs: float) -> np.ndarray:
    """Eq. (1): within a category (images ordered by time) a new group opens when the gap exceeds tau."""
    g = frame.groupby("cls")["ts"].diff().dt.total_seconds()
    return ((g.isna()) | (g > secs)).cumsum().values


def build_corpus(cfg):
    """Index, read metadata, recover leaf groups and sessions. Returns (full, n_with_timestamp)."""
    full = index_images(cfg.data_root)
    full = pd.concat([full, pd.DataFrame([read_meta(p) for p in full.path])], axis=1)
    full["ts"] = full.apply(capture_time, axis=1)
    n_ts = int(full.ts.notna().sum())
    full = full.dropna(subset=["ts"]).sort_values(["cls", "ts", "stem"]).reset_index(drop=True)
    full["leaf"] = gap_groups(full, cfg.leaf_tol_s)
    full["session"] = gap_groups(full, cfg.session_tol_s)
    return full, n_ts


def tolerance_table(full, cfg):
    """Grouping part of Table A2: number of groups G(tau) per tolerance, and tau chosen by Eq. (2)."""
    rows = []
    for t in cfg.tol_sweep:
        g = pd.Series(gap_groups(full, t)).value_counts()
        rows.append({"tau_s": t, "groups": int(g.size), "mean_size": round(float(g.mean()), 2),
                     "largest": int(g.max()), "singletons": int((g == 1).sum())})
    t = pd.DataFrame(rows)
    tau_star = int(t.iloc[(t.groups - cfg.l_star).abs().argmin()]["tau_s"])
    return t, tau_star


def composition_table(full, cfg):
    """Table 1."""
    sess_n = full.groupby("cls")["session"].nunique()
    t1 = pd.DataFrame({"images": full.groupby("cls").size(),
                       "leaf_groups": full.groupby("cls")["leaf"].nunique(),
                       "sessions": sess_n,
                       "first_capture": full.groupby("cls")["ts"].min().dt.strftime("%Y-%m-%d"),
                       "last_capture": full.groupby("cls")["ts"].max().dt.strftime("%Y-%m-%d")})
    t1["in_three_protocol_study"] = np.where(t1.sessions >= cfg.min_sessions, "yes", "excluded")
    t1.loc["Total"] = [int(t1.images.sum()), int(full.leaf.nunique()), int(full.session.nunique()), "", "", ""]
    return t1


def audit(full, n_ts):
    """Table A1."""
    dims = full.groupby(["w", "h"]).size().sort_values(ascending=False)
    dup_sets = [g for _, g in full[full.px_md5.notna()].groupby("px_md5") if len(g) > 1]
    dev_lab = (full.Make.fillna("?") + " " + full.Model.fillna("?")).str.strip()
    lg = full.groupby("leaf").size()
    return {"images": len(full), "categories": full.cls.nunique(),
            "distinct_pixel_dimensions": int(dims.size),
            "images_at_3_dominant_dimensions": int(dims.iloc[:3].sum()),
            "exact_pixel_duplicate_images": int(sum(len(g) - 1 for g in dup_sets)),
            "duplicate_sets_spanning_categories": int(sum(g.cls.nunique() > 1 for g in dup_sets)),
            "capture_devices_from_exif": int(dev_lab[dev_lab != "? ?"].nunique()),
            "images_with_usable_timestamp": n_ts,
            "leaf_groups": int(lg.size), "mean_images_per_leaf_group": round(float(lg.mean()), 2),
            "singleton_leaf_groups_pct": round(float((lg == 1).mean() * 100), 1),
            "sessions": int(full.session.nunique())}


def working_set(full, cfg):
    """Categories with at least two sessions enter the three-protocol study (gall midge is excluded)."""
    sess_n = full.groupby("cls")["session"].nunique()
    keep = sorted(sess_n[sess_n >= cfg.min_sessions].index)
    drop = sorted(sess_n[sess_n < cfg.min_sessions].index)
    df = full[full.cls.isin(keep)].reset_index(drop=True)
    df["label"] = df.cls.map({c: i for i, c in enumerate(keep)})
    info = {"classes": keep, "excluded": drop, "n_disorders": len(keep) - 1, "images": len(df),
            "leaf_groups": int(df.leaf.nunique()), "sessions": int(df.session.nunique()),
            "excluded_session_minutes": {c: round((full[full.cls == c].ts.max() -
                                                   full[full.cls == c].ts.min()).total_seconds() / 60)
                                         for c in drop}}
    return df, keep, info
