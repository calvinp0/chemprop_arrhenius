# Arrhenius Pipeline Layout

This directory is organized by ML pipeline stage to keep boundaries explicit:

- `data/`
  - dataset wrappers, collation, and preprocessing transforms
- `splitters/`
  - split/grouping logic used by training/evaluation
- `migration/`
  - RAD/QM migration and feature materialization utilities
- `modeling/nn/`
  - neural building blocks and predictors
- `modeling/module/`
  - Lightning module, mixins, checkpointing, and core model math
- `modeling/metrics/`
  - metric registry and logging primitives
- `training/hpo/`
  - run orchestration, HPO, evaluation, ensembling, and inference entry points

## Old-to-New Mapping

- `arrhenius/splitter` -> `arrhenius/splitters`
- `arrhenius/nn` -> `arrhenius/modeling/nn`
- `arrhenius/model` -> `arrhenius/modeling/module`
- `arrhenius/metrics` -> `arrhenius/modeling/metrics`
- `arrhenius/run_hpo` -> `arrhenius/training/hpo`
