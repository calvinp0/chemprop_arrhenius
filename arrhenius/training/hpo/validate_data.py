from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem

from arrhenius.training.hpo.data_spec import DataSpecError, load_data_spec


def _append_missing(errors: List[str], columns: List[str], required: List[str], label: str) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        errors.append(f"{label} missing required columns: {missing}")


def validate_data_spec(spec_path: str, sample_rows: int = 2000) -> Tuple[bool, Dict]:
    spec = load_data_spec(spec_path)
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, int] = {}

    sdf_path = Path(spec["paths"]["sdf_path"] or "")
    target_csv = Path(spec["paths"]["target_csv"] or "")
    rad_dir_raw = spec["paths"].get("rad_dir")
    rad_dir = Path(rad_dir_raw) if rad_dir_raw else None

    if not sdf_path.is_dir():
        errors.append(
            f"sdf_path must be a directory containing .sdf files (as expected by Featuriser): {sdf_path}"
        )
    if not target_csv.is_file():
        errors.append(f"target_csv does not exist or is not a file: {target_csv}")

    target_df = None
    target_rxn_col = str(spec["schema"]["target_rxn_col"])
    target_label_col = str(spec["schema"]["target_label_col"])
    target_forward_label = str(spec["schema"]["target_forward_label"])
    target_reverse_label = str(spec["schema"]["target_reverse_label"])
    if target_csv.is_file():
        target_df = pd.read_csv(target_csv)
        required_targets = list(spec["schema"]["target_columns"])
        _append_missing(errors, list(target_df.columns), required_targets, "target_csv")
        _append_missing(errors, list(target_df.columns), [target_rxn_col, target_label_col], "target_csv")
        stats["target_rows"] = int(len(target_df))

    target_rxn_values = set()
    target_group = None
    if target_df is not None and target_rxn_col in target_df.columns:
        target_rxn_values = set(map(str, target_df[target_rxn_col].dropna().astype(str).unique().tolist()))
        target_group = target_df.groupby(target_rxn_col) if target_label_col in target_df.columns else None

    if sdf_path.is_dir():
        sdf_files = sorted(p for p in sdf_path.iterdir() if p.is_file() and p.suffix.lower() == ".sdf")
        stats["sdf_file_count"] = int(len(sdf_files))
        if not sdf_files:
            errors.append(f"sdf_path has no .sdf files: {sdf_path}")
        else:
            parse_fail_files = 0
            missing_type_prop = 0
            missing_reaction_prop = 0
            missing_r1h = 0
            missing_r2h = 0
            ambiguous_types = 0
            rxn_prop_mismatch = 0
            missing_in_target = 0
            missing_pair_labels = 0
            nonfinite_target_rows = 0

            required_targets = list(spec["schema"]["target_columns"])
            for sdf_file in sdf_files:
                rxn_id = sdf_file.stem
                supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=True)
                if supplier is None:
                    parse_fail_files += 1
                    continue

                mols = [m for m in supplier if m is not None]
                if not mols:
                    parse_fail_files += 1
                    continue

                type_counts = {"r1h": 0, "r2h": 0}
                reaction_values = set()
                for m in mols:
                    if not m.HasProp("type"):
                        missing_type_prop += 1
                    else:
                        mt = str(m.GetProp("type"))
                        if mt in type_counts:
                            type_counts[mt] += 1

                    if not m.HasProp("reaction"):
                        missing_reaction_prop += 1
                    else:
                        reaction_values.add(str(m.GetProp("reaction")))

                if type_counts["r1h"] < 1:
                    missing_r1h += 1
                if type_counts["r2h"] < 1:
                    missing_r2h += 1
                if type_counts["r1h"] > 1 or type_counts["r2h"] > 1:
                    ambiguous_types += 1

                if len(reaction_values) != 1:
                    rxn_prop_mismatch += 1
                else:
                    reaction_value = next(iter(reaction_values))
                    if reaction_value != rxn_id:
                        rxn_prop_mismatch += 1

                if target_group is not None:
                    if rxn_id not in target_group.groups:
                        missing_in_target += 1
                    else:
                        rows = target_group.get_group(rxn_id)
                        labels = set(map(str, rows[target_label_col].astype(str).tolist()))
                        if target_forward_label not in labels or target_reverse_label not in labels:
                            missing_pair_labels += 1
                        else:
                            rfor = rows.loc[rows[target_label_col] == target_forward_label].iloc[0]
                            rrev = rows.loc[rows[target_label_col] == target_reverse_label].iloc[0]
                            vals = [float(rfor[c]) for c in required_targets] + [float(rrev[c]) for c in required_targets]
                            if any(pd.isna(v) for v in vals):
                                nonfinite_target_rows += 1
                elif target_rxn_values and rxn_id not in target_rxn_values:
                    missing_in_target += 1

            stats["sdf_parse_fail_files"] = int(parse_fail_files)
            stats["sdf_missing_type_prop_count"] = int(missing_type_prop)
            stats["sdf_missing_reaction_prop_count"] = int(missing_reaction_prop)
            stats["sdf_missing_r1h_files"] = int(missing_r1h)
            stats["sdf_missing_r2h_files"] = int(missing_r2h)
            stats["sdf_ambiguous_r1h_r2h_files"] = int(ambiguous_types)
            stats["sdf_reaction_mismatch_files"] = int(rxn_prop_mismatch)
            stats["sdf_missing_in_target_files"] = int(missing_in_target)
            stats["target_missing_pair_labels_files"] = int(missing_pair_labels)
            stats["target_nonfinite_pair_target_files"] = int(nonfinite_target_rows)

            if parse_fail_files > 0:
                errors.append(f"{parse_fail_files} sdf files could not be parsed by RDKit.")
            if missing_type_prop > 0:
                errors.append(f"{missing_type_prop} molecules in SDFs are missing required 'type' property.")
            if missing_reaction_prop > 0:
                errors.append(f"{missing_reaction_prop} molecules in SDFs are missing required 'reaction' property.")
            if missing_r1h > 0:
                errors.append(f"{missing_r1h} sdf files are missing a molecule with type='r1h'.")
            if missing_r2h > 0:
                errors.append(f"{missing_r2h} sdf files are missing a molecule with type='r2h'.")
            if rxn_prop_mismatch > 0:
                errors.append(
                    f"{rxn_prop_mismatch} sdf files have inconsistent reaction metadata "
                    "(reaction prop mismatch or multiple reaction values)."
                )
            if missing_in_target > 0:
                errors.append(f"{missing_in_target} sdf files have no matching target rows in target_csv.")
            if missing_pair_labels > 0:
                errors.append(
                    f"{missing_pair_labels} reactions are missing required forward/reverse labels "
                    f"('{target_forward_label}', '{target_reverse_label}')."
                )
            if nonfinite_target_rows > 0:
                errors.append(f"{nonfinite_target_rows} reactions have NaN target values in required target columns.")
            if ambiguous_types > 0:
                warnings.append(
                    f"{ambiguous_types} sdf files contain multiple r1h/r2h molecules; "
                    "run_hpo will use the first match."
                )

    mode_cfg = spec["mode_cfg"]
    uses_extras = bool(mode_cfg["use_extras"])

    if uses_extras:
        if rad_dir is None or not rad_dir.is_dir():
            errors.append(f"rad_dir is required for extra_mode='{spec['modes']['extra_mode']}' and must be a directory.")
        else:
            if spec["modes"]["rad_source"] == "path":
                candidates = [rad_dir / "atom_with_geom_feats_path.csv"]
            else:
                # Prefer new naming, accept legacy default for compatibility.
                candidates = [rad_dir / "atom_with_geom_feats_rad.csv", rad_dir / "atom_with_geom_feats_default.csv"]
            rad_csv = next((p for p in candidates if p.is_file()), None)
            if rad_csv is None:
                errors.append(f"RAD CSV missing. Tried: {[str(p) for p in candidates]}")
            else:
                rad_df = pd.read_csv(rad_csv)
                stats["rad_rows"] = int(len(rad_df))
                required = list(mode_cfg["cols"]) + [
                    spec["schema"]["rxn_id_col"],
                    spec["schema"]["mol_type_col"],
                    spec["schema"]["atom_index_col"],
                ]
                if mode_cfg["rad_mask"] is not None:
                    required.append(spec["schema"]["shortest_path_col"])
                _append_missing(errors, list(rad_df.columns), required, "rad_csv")

                if not errors:
                    rxn_col = spec["schema"]["rxn_id_col"]
                    mol_col = spec["schema"]["mol_type_col"]
                    atom_col = spec["schema"]["atom_index_col"]
                    donor_tag = spec["schema"]["donor_tag"]
                    acceptor_tag = spec["schema"]["acceptor_tag"]

                    donor_rows = int((rad_df[mol_col] == donor_tag).sum())
                    acceptor_rows = int((rad_df[mol_col] == acceptor_tag).sum())
                    stats["rad_donor_rows"] = donor_rows
                    stats["rad_acceptor_rows"] = acceptor_rows
                    if donor_rows == 0 or acceptor_rows == 0:
                        errors.append(
                            f"rad_csv has no rows for donor/acceptor tags ({donor_tag}, {acceptor_tag})."
                        )

                    dup_count = int(rad_df.duplicated(subset=[rxn_col, mol_col, atom_col]).sum())
                    stats["rad_duplicate_atom_keys"] = dup_count
                    if dup_count > 0:
                        warnings.append(
                            f"rad_csv has {dup_count} duplicate (rxn_id, mol_type, atom_index) rows."
                        )

                    by_rxn = rad_df.groupby(rxn_col)[mol_col].agg(set)
                    missing_pairs = int((~by_rxn.apply(lambda s: donor_tag in s and acceptor_tag in s)).sum())
                    stats["rad_rxn_missing_pair_tags"] = missing_pairs
                    if missing_pairs > 0:
                        errors.append(
                            f"{missing_pairs} rxn_id values in rad_csv are missing either donor or acceptor rows."
                        )

                    if mode_cfg["rad_mask"] is not None:
                        path_col = spec["schema"]["shortest_path_col"]
                        parsed_ok = 0
                        parsed_bad = 0
                        probe = rad_df[path_col].head(max(1, sample_rows))
                        for v in probe:
                            try:
                                x = ast.literal_eval(v) if isinstance(v, str) else v
                                _ = len(x) if x is not None else 0
                                parsed_ok += 1
                            except Exception:
                                parsed_bad += 1
                        stats["shortest_path_parsed_ok"] = parsed_ok
                        stats["shortest_path_parsed_bad"] = parsed_bad
                        if parsed_bad > 0:
                            warnings.append(
                                f"{parsed_bad} sampled shortest_path values failed to parse; they will be treated as far-distance."
                            )

    ok = len(errors) == 0
    report = {"ok": ok, "errors": errors, "warnings": warnings, "stats": stats, "spec": spec}
    return ok, report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate data_spec YAML and source files for run_hpo.")
    p.add_argument("--spec", required=True, help="Path to data_spec YAML file.")
    p.add_argument("--sample-rows", type=int, default=2000, help="Rows to sample for shortest_path parse checks.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        ok, report = validate_data_spec(args.spec, sample_rows=max(1, int(args.sample_rows)))
    except DataSpecError as e:
        print(f"[INVALID SPEC] {e}")
        return 2
    except Exception as e:
        print(f"[VALIDATION ERROR] {e}")
        return 3

    print(f"[SPEC] {args.spec}")
    print(f"[MODE] extra={report['spec']['modes']['extra_mode']} global={report['spec']['modes']['global_mode']} rad_source={report['spec']['modes']['rad_source']}")
    for k, v in sorted(report["stats"].items()):
        print(f"[STAT] {k}={v}")
    for w in report["warnings"]:
        print(f"[WARN] {w}")
    for e in report["errors"]:
        print(f"[ERROR] {e}")

    if ok:
        print("[OK] Data spec validation passed.")
        return 0
    print("[FAIL] Data spec validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
