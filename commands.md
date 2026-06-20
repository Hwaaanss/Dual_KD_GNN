# Commands

This project benchmarks a single model — **`dual_kd_gnn`** (Double GCN +
EMA-teacher knowledge distillation + transformer) — across five MoleculeNet
classification datasets: **BACE, BBBP, SIDER, Tox21, ClinTox**.

The number of prediction tasks differs per dataset and is resolved
automatically from the dataset registry ([common/datasets.py](common/datasets.py)),
so the model's classifier head, the training loss, and the evaluation metric all
adapt to whichever dataset you select.

| Dataset | File          | SMILES column | Tasks | Notes |
| ------- | ------------- | ------------- | ----- | ----- |
| bace    | `data/bace.csv`    | `mol`    | 1  | BACE-1 inhibition |
| bbbp    | `data/bbbp.csv`    | `smiles` | 1  | Blood-brain barrier penetration |
| sider   | `data/sider.csv`   | `smiles` | 27 | Adverse-reaction system-organ classes |
| tox21   | `data/tox21.csv`   | `smiles` | 12 | Toxicity assays |
| clintox | `data/clintox.csv` | `smiles` | 2  | FDA approval vs. clinical-trial toxicity |

---

## 0. Environment

```bash
conda activate dualgnn
pip install -r requirements.txt   # only if dependencies are missing
```

---

## 1. Download datasets

Tox21 ships with the repo. Download the rest with **either** option.

### Option A — Python (recommended)

```bash
python scripts/download_data.py            # download all five datasets
python scripts/download_data.py bbbp bace  # download a subset
python scripts/download_data.py --force tox21   # force re-download
```

Files land in `data/` with the canonical names above; gzipped sources are
decompressed automatically.

### Option B — Terminal (wget + gunzip)

```bash
# BBBP and BACE are plain CSV
wget -O data/bbbp.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
wget -O data/bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv

# SIDER, ClinTox, Tox21 are gzipped -> decompress to plain .csv
wget -O data/sider.csv.gz   https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz   && gunzip -f data/sider.csv.gz
wget -O data/clintox.csv.gz https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz && gunzip -f data/clintox.csv.gz
wget -O data/tox21.csv.gz   https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz   && gunzip -f data/tox21.csv.gz
```

(Use `curl -L -o <file> <url>` if `wget` is unavailable.)

---

## 2. Train on a single dataset

[dual_kd_gnn/main.py](dual_kd_gnn/main.py) trains one dataset and saves all
artifacts under `dual_kd_gnn/runs/<dataset>/`.

```bash
python dual_kd_gnn/main.py --dataset bbbp
python dual_kd_gnn/main.py --dataset tox21
python dual_kd_gnn/main.py --dataset sider --gcn-pretrain-epochs 150 --transformer-epochs 150
```

> Device defaults to **cuda**. Override with `--device cpu`, `--device cuda:1`, etc.

Common overrides (all optional — sensible defaults exist):

- Training schedule: `--gcn-pretrain-epochs`, `--transformer-epochs`, `--patience`
- Optimization: `--batch-size`, `--lr`, `--pretrain-lr`, `--transformer-lr`, `--weight-decay`
- Distillation: `--distill-weight`, `--cross-distill-weight`, `--ema-decay`, `--ema-decay-init`
- Reuse tuned hyperparameters: `--best-config dual_kd_gnn/optuna/<study>/best_config.json`
- Manual data overrides: `--data-path`, `--smiles-column`, `--target-columns`

---

## 3. Batch benchmark across all datasets

[main.py](main.py) trains `dual_kd_gnn` on every dataset sequentially and writes
a comparison table.

```bash
python main.py                              # all five datasets (cuda by default)
python main.py --datasets bace bbbp tox21   # a subset
python main.py --best-config dual_kd_gnn/optuna/tox21_xkd/best_config.json  # same tuned HPs for all
```

Missing dataset files are skipped with a download hint. A summary table is saved
to `results/artifacts/benchmark_summary.csv`.

---

## 4. Hyperparameter tuning (Optuna)

[dual_kd_gnn/tune_optuna.py](dual_kd_gnn/tune_optuna.py) tunes on one dataset.
Use a dataset-specific `--study-name` so studies do not collide.

```bash
python dual_kd_gnn/tune_optuna.py --dataset bbbp  --study-name bbbp_xkd  --n-trials 30
python dual_kd_gnn/tune_optuna.py --dataset tox21 --study-name tox21_xkd --n-trials 50
python dual_kd_gnn/tune_optuna.py --dataset sider --study-name sider_xkd --n-trials 40 \
    --gcn-pretrain-epochs 150 --transformer-epochs 150 --patience 10
```

Study artifacts (SQLite DB, `best_config.json`, `trials.csv`) are saved under
`dual_kd_gnn/optuna/<study-name>/`.

Replay the best config to train + evaluate on the test split (saves weights and
plots):

```bash
python dual_kd_gnn/tune_optuna.py --replay-best dual_kd_gnn/optuna/bbbp_xkd/best_config.json
```

---

## 5. Aggregate results

[results/main.py](results/main.py) scans `dual_kd_gnn/runs/*` and writes the
cross-dataset comparison table and plots to `results/artifacts/`.

```bash
python results/main.py
```

Outputs: `all_metrics.csv`, `results_table.csv`, `results_table.md`,
`val_auc_curves.png`, `test_auc_by_dataset.png`.

---

## Output layout

```
dual_kd_gnn/runs/<dataset>/
├── metrics.json          # best val AUC, test ROC-AUC, #params, runtime, target columns
├── run_metadata.json     # hparams, model_kwargs, data path, device
├── training_log.csv      # per-epoch train/val loss and ROC-AUC
├── model_weights.pt      # best-epoch model state_dict
└── training_curves.png   # loss + ROC-AUC curves

dual_kd_gnn/optuna/<study-name>/   # Optuna study DB, best_config.json, trials.csv
results/artifacts/                 # cross-dataset benchmark tables + plots
```

Each dataset gets its own run directory, so weights, logs, plots, and result
tables never overwrite one another.
