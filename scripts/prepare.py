"""Step 1. Index the corpus, recover leaf groups and acquisition sessions, audit the released files, build and
verify all partitions, run the pixel-free probes, and draw Figures 2 and 3.

Outputs: Table 1, Table 2, Table 3, Table A1, partitions_all.csv, Figures 2 and 3, work/study.pkl.

    python scripts/prepare.py --data-root /path/to/MangoLeafBD --out outputs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mangoleaf import figures as F
from mangoleaf.config import add_common_args, config_from_args
from mangoleaf.probes import run_probes
from mangoleaf.study import prepare
from mangoleaf.utils import update_numbers


def main():
    a = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(a)
    st = prepare(cfg)

    t2, chance = run_probes(st.df, st.num_classes)
    t2.to_csv(cfg.tbl / "Table2_probes.csv", index=False)
    print(f"chance {chance:.3f}\n", t2.to_string(index=False))
    update_numbers(cfg.tbl / "numbers.json", "probes", {"chance": round(chance, 3), **dict(zip(t2.feature_set, t2.accuracy))})

    F.set_style(cfg.save_dpi)
    F.savefig(F.fig2_corpus(st.full, st.t1, st.t3), cfg.fig, "Fig2_corpus", cfg.save_dpi)
    F.savefig(F.fig3_probes(t2, chance), cfg.fig, "Fig3_probes", cfg.save_dpi)


if __name__ == "__main__":
    main()
