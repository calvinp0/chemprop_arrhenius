# Migration Tools

This folder contains standalone data migration/preprocessing utilities.

## Included scripts

- `create_RAD.py`
  - Copied from your existing preprocessing workflow.
  - Computes geometric RAD features from SDF molecules.

- `build_atom_with_geom_qm.py`
  - Orchestrates:
    1. RAD geometry generation (default + path mode) via `create_RAD.py`
    2. Gaussian freq log parsing for `q_mull`, `q_apt`, `f_mag`
    3. Merge onto RAD rows to produce:
       - `atom_with_geom_feats_default.csv`
       - `atom_with_geom_feats_path.csv`

- `gaussian_qm.py`
  - Atom-level Gaussian parser used by the orchestrator.

## Usage

1. Create a manifest CSV (see `manifest.example.csv`).

2. Run:

```bash
python arrhenius/migration/build_atom_with_geom_qm.py \
  --manifest-csv arrhenius/migration/manifest.example.csv \
  --output-dir /tmp/qm_rad_out
```

3. Use resulting `atom_with_geom_feats_*.csv` as `--rad-dir` inputs for `run_hpo`.

## Notes

- This is intentionally separate from `run_hpo`.
- For pipeline details and validation contract see `QM_RAD_PIPELINE_SPEC.md`.
