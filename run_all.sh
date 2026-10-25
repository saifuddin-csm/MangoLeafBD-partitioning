#!/usr/bin/env bash
# Reproduce every table and figure of the article.
# Usage: bash run_all.sh /path/to/MangoLeafBD [output_folder]
set -euo pipefail
DATA="${1:?path to the MangoLeafBD images}"
OUT="${2:-outputs}"
python scripts/prepare.py --data-root "$DATA" --out "$OUT"
python scripts/train.py   --data-root "$DATA" --out "$OUT"
python scripts/analyse.py --data-root "$DATA" --out "$OUT"
python scripts/explain.py --data-root "$DATA" --out "$OUT"
