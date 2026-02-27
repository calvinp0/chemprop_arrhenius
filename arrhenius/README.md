# Arrhenius Pipeline Layout

This directory is organized by ML pipeline stage to keep boundaries explicit:

- `data/`
  - dataset wrappers, collation, and preprocessing transforms
- `splitters/`
  - split/grouping logic used by training/evaluation
- `preprocessing/`
  - RAD/QM migration and feature materialization utilities
- `modeling/nn/`
  - neural building blocks and predictors
- `modeling/module/`
  - Lightning module, mixins, checkpointing, and core model math
- `modeling/metrics/`
  - metric registry and logging primitives
- `training/hpo/`
  - run orchestration, HPO, evaluation, ensembling, and inference entry points

## End-to-End Quickstart

Use this sequence if you want to go from optimization logs to final predictions.

1. Build labeled pair SDF from optimization logs (Gaussian/ORCA):

```bash
python -m arrhenius.preprocessing.build_labeled_pair_sdf \
  --r1h-log <R1H_OPT_LOG> \
  --r1h-format <gaussian|orca> \
  --r1h-h-idx <R1H_H_IDX> \
  --r1h-donor-idx <R1H_DONOR_IDX_OPTIONAL> \
  --r1h-acceptor-idx <R1H_ACCEPTOR_IDX_OPTIONAL> \
  --r2h-log <R2H_OPT_LOG> \
  --r2h-format <gaussian|orca> \
  --r2h-h-idx <R2H_H_IDX> \
  --r2h-donor-idx <R2H_DONOR_IDX_OPTIONAL> \
  --r2h-acceptor-idx <R2H_ACCEPTOR_IDX_OPTIONAL> \
  --reaction-id <RXN_ID> \
  --output-sdf <PAIR_SDF_PATH>
```

2. Build RAD+QM atom feature CSVs:

```bash
python -m arrhenius.preprocessing.build_atom_with_geom_qm \
  --manifest-csv arrhenius/preprocessing/manifest.example.csv \
  --output-dir <RAD_DIR>
```

3. Run training/HPO:

```bash
python arrhenius/training/hpo/run.py \
  --sdf-path <SDF_PATH> \
  --target-csv <TARGET_CSV> \
  --rad-dir <RAD_DIR> \
  --extra-mode <baseline|geom_only|local|atom|rad|rad_local|rad_local_noc> \
  --rad-source <path|rad> \
  --hpo-trials <N>
```

4. Export/evaluate ensemble from final summary:

```bash
python arrhenius/training/hpo/ensemble_eval.py \
  --summary-json <FINAL_SUMMARY_JSON> \
  --sdf-path <SDF_PATH> \
  --target-csv <TARGET_CSV> \
  --rad-dir <RAD_DIR> \
  --output-dir <BUNDLE_DIR>
```

5. Predict from exported bundle/checkpoints:

```bash
python arrhenius/training/hpo/predict.py \
  --model-dir <MODEL_DIR> \
  --sdf-path <SDF_PATH> \
  --input-csv <INPUT_CSV> \
  --rad-dir <RAD_DIR> \
  --output-csv <PREDICTIONS_CSV>
```

See details in:
- `arrhenius/preprocessing/README.md`
- `arrhenius/training/hpo/WORKFLOW.md`

## Inference Only (No Training)

This is a normal workflow when you already have a trained model.

Use `predict.py` with an exported model directory:

```bash
python arrhenius/training/hpo/predict.py \
  --model-dir <MODEL_DIR> \
  --sdf-path <SDF_PATH> \
  --input-csv <INPUT_CSV> \
  --rad-dir <RAD_DIR_IF_NEEDED> \
  --output-csv <PREDICTIONS_CSV>
```

### What You Need To Load

- Model checkpoint(s) (`.ckpt`)
- Model/config metadata (for architecture and feature modes)
- Normalizers/scalers used during training (target/input transforms)

In practice, this is why the code loads a bundle, not just a bare weight tensor.
Raw weights plus hand-typed params can work, but it is easy to get mismatches in
feature construction and scaling, which changes predictions.
