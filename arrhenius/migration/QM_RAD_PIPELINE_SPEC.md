# QM + RAD Feature Build Spec

This pipeline is intentionally separate from `run_hpo`.

## Goal
Given reaction SDFs and Gaussian frequency logs, generate:

- `atom_with_geom_feats_path.csv`
- `atom_with_geom_feats_default.csv`

These files are the direct inputs expected by `run_hpo --rad-dir`.

## Inputs

### Required
- SDF file(s), each containing `r1h` and `r2h` molecules with:
  - `reaction` property (or inferable reaction id from filename)
  - `type` property (`r1h` / `r2h`)
  - `mol_properties` label payload used by `create_RAD.py`
- Gaussian freq logs for each reaction component:
  - one log for `r1h`
  - one log for `r2h`

### Manifest (recommended)
CSV with columns:
- `rxn_id`
- `sdf_file`
- `r1h_log`
- `r2h_log`

## Outputs

Inside `--output-dir`:
- `geom_features_path.csv`
- `geom_features_default.csv`
- `atom_with_geom_feats_path.csv`
- `atom_with_geom_feats_default.csv`
- `qm_merge_report.json`

## Merge keys and schema

Merge key:
- `rxn_id`
- `mol_type`
- `focus_atom_idx`

QM columns added:
- `q_mull`
- `q_apt`
- `f_mag`

RAD columns come from `create_RAD.py`.

## Validation rules

- Every manifest row must have existing `sdf_file`, `r1h_log`, `r2h_log`.
- Atom count in Gaussian blocks must match corresponding SDF molecule atom count.
- Optional symbol check: Gaussian atom symbols should match SDF atom symbols by index.
- If `q_apt` or force blocks are missing, fill with `NaN` and record in report.
- Any failed row is recorded in `qm_merge_report.json` with explicit reason.

## Runtime flow

1. Build RAD tables from SDF files (default + path modes) using `create_RAD.py`.
2. Parse Gaussian logs into per-atom QM tables by `rxn_id` and `mol_type`.
3. Merge QM onto both RAD tables.
4. Emit merged CSVs and report.

## Notes

- This pipeline does not train models.
- `run_hpo` remains unchanged and only consumes prepared CSVs.
