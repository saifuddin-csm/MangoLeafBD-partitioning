"""Small shared helpers: seeding, JSON conversion, CSV merging."""
import json
import random

import numpy as np
import pandas as pd
import torch


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, pd.Timestamp): return str(o)
    return o


def update_numbers(path, key, value):
    """Store a value quoted in the article in numbers.json (one file shared by all scripts)."""
    d = json.load(open(path)) if path.exists() else {}
    d[key] = jsonable(value)
    json.dump(d, open(path, "w"), indent=2)


def append_row(csv_path, row: dict, key):
    new = pd.DataFrame([row])
    if csv_path.exists():
        new = pd.concat([pd.read_csv(csv_path), new], ignore_index=True)
    new.drop_duplicates(key, keep="last").to_csv(csv_path, index=False)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
