# Preprocessing Tools

This folder contains standalone data preprocessing utilities.

## Included scripts

- `create_RAD.py`
  - Copied from your existing preprocessing workflow.
  - Computes geometric RAD features from SDF molecules.

- `build_labeled_pair_sdf.py`
  - Builds a 2-molecule SDF (`r1h` + `r2h`) from optimization logs.
  - Supports log formats: Gaussian and ORCA.
  - Adds required SDF properties:
    - `type` (`r1h` / `r2h`)
    - `reaction`
    - `mol_properties` (JSON mapping atom index -> `{label, atom_type}`)
  - Atom labels (donor/acceptor/moving-H) are supplied by CLI indices.
  - `atom_type` currently uses RDKit element-based fallback (not RMG atomtype API).

- `build_atom_with_geom_qm.py`
  - Orchestrates:
    1. RAD geometry generation (rad + path mode) via `create_RAD.py`
    2. Gaussian freq log parsing for `q_mull`, `q_apt`, `f_mag`
    3. Merge onto RAD rows to produce:
       - `atom_with_geom_feats_rad.csv`
       - `atom_with_geom_feats_path.csv`

- `gaussian_qm.py`
  - Atom-level Gaussian parser used by the orchestrator.

## Build Labeled Pair SDF From Logs

Example (Gaussian + Gaussian):

```bash
python -m arrhenius.preprocessing.build_labeled_pair_sdf \
  --r1h-log /abs/path/r1h_opt.log \
  --r1h-format gaussian \
  --r1h-donor-idx 0 \
  --r1h-h-idx 3 \
  --r2h-log /abs/path/r2h_opt.log \
  --r2h-format gaussian \
  --r2h-acceptor-idx 0 \
  --r2h-h-idx 4 \
  --reaction-id rxn_123 \
  --output-sdf /abs/path/rxn_123_pair.sdf
```

Example (ORCA + Gaussian):

```bash
python -m arrhenius.preprocessing.build_labeled_pair_sdf \
  --r1h-log /abs/path/r1h_opt.out \
  --r1h-format orca \
  --r1h-donor-idx 12 \
  --r1h-h-idx 27 \
  --r2h-log /abs/path/r2h_opt.log \
  --r2h-format gaussian \
  --r2h-acceptor-idx 8 \
  --r2h-h-idx 30 \
  --reaction-id rxn_456 \
  --output-sdf /abs/path/rxn_456_pair.sdf
```

Notes:
- Indices are 0-based and refer to atom order parsed from each log's final geometry block.
- If total charge is not parsed correctly from a log, you can set `--r1h-charge` / `--r2h-charge`.
- Donor/acceptor indices are optional; hydrogen index is required for each side.

## Usage

1. Create a manifest CSV (see `manifest.example.csv`).

2. Run:

```bash
python -m arrhenius.preprocessing.build_atom_with_geom_qm \
  --manifest-csv arrhenius/preprocessing/manifest.example.csv \
  --output-dir /tmp/qm_rad_out
```

3. Use resulting `atom_with_geom_feats_*.csv` as `--rad-dir` inputs for `arrhenius/training/hpo/run.py`.

## Fixture Smoke Test

You can run the builder against the local fixture set under `arrhenius/tests/fixtures/`.
Use `arrhenius/tests/preprocessing/test_build_atom_with_geom_qm.py` as the reference invocation.

## Notes

- This is intentionally separate from the training/HPO pipeline in `arrhenius/training/hpo/`.
- For pipeline details and validation contract see `QM_RAD_PIPELINE_SPEC.md`.
