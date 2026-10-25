# Impact of Data Partitioning on the Field Reliability of Deep Learning Models for Mango Leaf Disease Recognition

Code, derived labels and results for the article of the same title, *Journal of Agricultural Engineering*,
The Institution of Engineers, Bangladesh (manuscript 32 AED 2024-2026).

Md. Saif Uddin, Tasmin Akther, Md. Sakhawat Hossain. Department of Computer Science and Mathematics,
Bangladesh Agricultural University. Contact: saifuddin.csm@bau.edu.bd

The study recovers leaf groups and acquisition sessions from the capture timestamps of the MangoLeafBD images
and compares four ImageNet-pretrained models (VGG-16, ResNet-50, ViT-B/16, Swin-T) under random, leaf-wise and
session-wise partitioning. Random and leaf-wise partitioning give a mean accuracy of 0.9993 and 0.9985;
session-wise partitioning, which withholds whole orchard visits, gives 0.7280 over 10 test folds.

## Repository layout

```
src/mangoleaf/            the Python package
    config.py             every setting used in the article
    provenance.py         corpus index, audit (Table A1), leaf groups and sessions (Eqs. 1-2), Table 1
    partitions.py         RP, LW and SW partitions, contamination (Eq. 3), Algorithm A1, verification
    study.py              builds and caches the corpus and all partitions (outputs/work/study.pkl)
    data.py               image loading, augmentation, data loaders
    models.py             model creation, fine-tuning, test-time augmentation, checkpoints, latency
    experiments.py        registry of the 103 training runs; resumable execution
    metrics.py            cluster bootstrap, ECE (Eq. 4), temperature scaling (Eq. 5), referral (Eqs. 6-8)
    probes.py             pixel-free probes (Table 2)
    analysis.py           Tables 4-9 and A2-A5 from the stored predictions
    xai.py                region ablation, Grad-CAM, occlusion sensitivity
    figures.py            all figures (no grid, 600 dpi PNG and vector PDF)
scripts/
    prepare.py            step 1: provenance, audit, partitions, probes -> Tables 1-3, A1; Figures 2-3
    train.py              step 2: all training runs -> predictions, checkpoints, latency
    analyse.py            step 3: Tables 4-9, A2-A5; Figures 4-7, A1, A3, A4; consistency checks
    explain.py            step 4: Figure 8 and Figure A2
run_all.sh                runs the four steps in order
data/mangoleafbd_provenance_labels.csv
results/tables/           every result table of the article (CSV) and numbers.json
results/figures/          every figure of the article
results/preds/            stored predictions of every run (.npz)
```

## Data

The images are not redistributed. Download MangoLeafBD (Ahmed et al., 2023) from Mendeley Data,
https://doi.org/10.17632/hxsnvwty3r.1, and pass the folder that contains the eight category folders with
`--data-root` (it is searched recursively).

`data/mangoleafbd_provenance_labels.csv` has one row per image: category, file name, capture time, device,
image size, the recovered **leaf group** (images of one category at most 3 s apart) and **acquisition
session** (at most 1800 s apart), whether the image is in the three-protocol study, its test fold in each of
the five session-wise partitions, and its train/validation/test assignment under random and leaf-wise
partitioning for each seed.

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9 or later, PyTorch 2.1 or later and timm 0.9 or later. ImageNet weights are downloaded by
`timm` on the first run.

## Reproducing the results

```
bash run_all.sh /path/to/MangoLeafBD outputs
```

or step by step:

```
python scripts/prepare.py --data-root /path/to/MangoLeafBD --out outputs
python scripts/train.py   --data-root /path/to/MangoLeafBD --out outputs
python scripts/analyse.py --data-root /path/to/MangoLeafBD --out outputs
python scripts/explain.py --data-root /path/to/MangoLeafBD --out outputs
```

The full training run (103 runs) takes about 5-6 hours on one NVIDIA Tesla T4. It is resumable: finished
runs are recorded in `outputs/tables/results_rev.csv` and skipped, so `train.py` can simply be started again
after an interruption. `--groups` selects part of the work (for example `--groups sw main` on one machine and
`--groups control tolerance lr pad` on another, with the outputs folders merged before `analyse.py`).

`analyse.py` ends with consistency checks between tables and figures and prints `ALL CONSISTENT` when every
check passes. `--quick` runs one model for one epoch on one partition (a 15-minute check on a GPU), and
`--dry-run` runs the whole pipeline offline on a CPU with random weights (for testing the code only).

**Repeating the analysis without training.** Copy `results/tables/results_rev.csv`,
`results/tables/history_rev.csv`, `results/tables/latency.csv` and `results/preds/` into `outputs/tables` and
`outputs/preds`, then run `prepare.py` and `analyse.py`. Every table and figure is recomputed from the stored
predictions.

**On Kaggle.** Attach the MangoLeafBD dataset, select GPU T4 x1 and Internet on, upload this repository as a
dataset, and in a notebook cell run
`!cd /kaggle/input/<repo> && python scripts/prepare.py --data-root /kaggle/input/<mangoleafbd> --out /kaggle/working/outputs`,
followed by the other three scripts with the same arguments.

## Map from the article to the outputs

| Article | File (in `results/tables` or `results/figures`) | Produced by |
|---|---|---|
| Table 1 | `Table1_composition.csv` | prepare.py |
| Table 2, Figure 3 | `Table2_probes.csv`, `Fig3_probes` | prepare.py |
| Table 3, Figure 2 | `Table3_partitions.csv`, `Fig2_corpus` | prepare.py |
| Table 4, Figure 4 | `Table4_principal_result.csv`, `Fig4_three_protocols` | analyse.py |
| Table 5 | `Table5_matched_size_control.csv` | analyse.py |
| Table 6 | `Table6_ranking.csv`, `Table6b_paired_bootstrap.csv` | analyse.py |
| Table 7, Figures 5-6 | `Table7_per_category.csv`, `Fig5_per_class`, `Fig6_confusion` | analyse.py |
| Table 8, Figure 7 | `Table8_referral.csv`, `Fig7_risk_coverage` | analyse.py |
| Table 9, Figure A4 | `Table9_deployment_cost.csv`, `FigA4_cost` | analyse.py |
| Table 10 | values of Table 4 and the cited publications | - |
| Table A1 | `TableA1_audit.csv` | prepare.py |
| Table A2 | `TableA2_tolerance.csv` | analyse.py |
| Table A3, Figure A1 | `TableA3_calibration.csv`, `FigA1_reliability` | analyse.py |
| Table A4 | `TableA4_padding.csv` | analyse.py |
| Table A5 | `TableA5_learning_rate.csv` | analyse.py |
| Figure A3 | `FigA3_training_curves` | analyse.py |
| Figure 8, Figure A2 | `Fig8_gradcam_occlusion`, `FigA2_region_ablation` | explain.py |
| Values quoted in the text | `numbers.json` | all steps |

## Citation

Uddin, M. S., Akther, T., & Hossain, M. S. (2026). Impact of data partitioning on the field reliability of deep
learning models for mango leaf disease recognition. *Journal of Agricultural Engineering*, The Institution of
Engineers, Bangladesh.

## Licence

Code: MIT licence (see `LICENSE`). Derived labels and results: CC BY 4.0. The MangoLeafBD images remain under
the licence of their authors.
