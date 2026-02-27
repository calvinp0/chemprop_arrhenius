# Arrhenius Test Fixtures

This folder stores Arrhenius-specific tests and data fixtures.

## Layout

- `fixtures/sdf/`
  - Reaction SDF fixtures.
- `fixtures/qm_logs/`
  - QM log fixtures used for Gaussian/ORCA parsing and merge pipelines.
- `fixtures/opt_logs/`
  - Optimization log fixtures used by `build_labeled_pair_sdf.py`.
- `preprocessing/`
  - Preprocessing pipeline tests.

## Notes

- Generated outputs are intentionally not kept here.
- For ad-hoc local outputs, use a temporary directory or recreate as needed.
