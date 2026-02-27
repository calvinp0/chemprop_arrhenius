# Training/HPO Workflow Guide

This guide describes the recommended end-to-end workflow for:
- HPO
- top-k model selection
- optional paper-grade locked test evaluation
- ensemble training/export for uncertainty
- downstream prediction

Use the `torch_rocm` conda env for all commands.

## 1) Data Prerequisites

You need:
- `SDF` input (`--sdf-path`)
- target CSV (`--target-csv`) with required targets
- RAD/geom feature directory (`--rad-dir`) when using atom-extra modes (`local`, `atom`, `rad*`)

If you need to generate RAD features, run the migration/preprocessing scripts first (outside `hpo`).

## 2) Optional Data Validation

Validate file/schema consistency before launching HPO:

```bash
conda run -n torch_rocm python arrhenius/training/hpo/run.py validate-data \
  --spec arrhenius/training/hpo/data_spec.example.yaml
```

## 3) Run HPO + Final Selection

## Standard mode (no locked test)

```bash
conda run -n torch_rocm python arrhenius/training/hpo/run.py \
  --sdf-path <SDF_PATH> \
  --target-csv <TARGET_CSV> \
  --extra-mode <baseline|geom_only|local|atom|rad|rad_local|rad_local_noc> \
  --rad-source <path|rad> \
  --rad-dir <RAD_DIR_IF_NEEDED> \
  --global-mode <none|morgan_binary|morgan_count|rdkit2d_norm|auto> \
  --search-space arrhenius/training/hpo/search_space.yaml \
  --hpo-trials 50 \
  --top-k-final 3 \
  --splitter kstone
```

## Paper-grade mode (locked untouched test set)

This reserves a strict test set first, then runs HPO/top-k CV only on the remaining pool.

```bash
conda run -n torch_rocm python arrhenius/training/hpo/run.py \
  --sdf-path <SDF_PATH> \
  --target-csv <TARGET_CSV> \
  --extra-mode <MODE> \
  --rad-source <path|rad> \
  --rad-dir <RAD_DIR_IF_NEEDED> \
  --search-space arrhenius/training/hpo/search_space.yaml \
  --hpo-trials 50 \
  --top-k-final 3 \
  --locked-test-frac 0.15
```

Notes:
- Top-k is selected from trials with the same split signature by default.
- Use `--relax-topk` only if you intentionally want fallback to all completed trials.
- Ranking metric defaults to `rank_metric_default` in `search_space.yaml` (currently `val/mae_lnk_avg`).
- Override ranking metric with `--rank-metric <metric_key>`.

## 4) Locate Final Summary JSON

After run completion, a summary JSON is written to:

- `logs/hpo/<study_name>_<tag>_final_summary.json`

This file is the input for ensemble export.

## 5) Train/Export Deep Ensemble (Uncertainty Bundle)

```bash
conda run -n torch_rocm python arrhenius/training/hpo/ensemble_eval.py \
  --summary-json <FINAL_SUMMARY_JSON> \
  --sdf-path <SDF_PATH> \
  --target-csv <TARGET_CSV> \
  --rad-dir <RAD_DIR_IF_NEEDED> \
  --output-dir artifacts/model_v1 \
  --ensemble-size 5 \
  --save-members \
  --member-format ckpt \
  --save-normalizers
```

Recommended ensemble size:
- `5` (default practical choice)
- `3` minimum
- `10` if you want more stable uncertainty at higher compute cost

Exported bundle contains:
- member checkpoints
- `global_normalizers.joblib`
- `ensemble_metadata.json`
- parity/prediction artifacts

## 6) Predict with Exported Bundle

```bash
conda run -n torch_rocm python arrhenius/training/hpo/predict.py \
  --model-dir artifacts/model_v1 \
  --sdf-path <SDF_PATH> \
  --input-csv <INPUT_CSV> \
  --rad-dir <RAD_DIR_IF_NEEDED> \
  --output-csv artifacts/model_v1/predictions.csv
```

`predict.py` runs all members and outputs mean/std per target (`A`, `n`, `Ea`), which is ensemble (epistemic) uncertainty.

## 7) What Uncertainty Means

- Deep ensemble spread (`std` across members) is an epistemic uncertainty proxy.
- It is suitable for confidence ranking/reporting in most ML chemistry papers if described correctly.
- It is not full aleatoric + epistemic decomposition unless you add explicit noise modeling.
