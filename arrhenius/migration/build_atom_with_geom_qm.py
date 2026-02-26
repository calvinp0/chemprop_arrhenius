from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from rdkit import Chem

from arrhenius.migration.gaussian_qm import parse_gaussian_qm


def _load_create_rad_module(create_rad_py: Path):
    spec = importlib.util.spec_from_file_location("create_RAD_mod", str(create_rad_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load create_RAD module from {create_rad_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_sdf_atom_symbols(sdf_file: Path, rxn_id: str) -> Dict[str, List[str]]:
    """
    Returns map {mol_type: [symbols by atom index]} for r1h/r2h in this sdf.
    """
    out: Dict[str, List[str]] = {}
    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        rid = mol.GetProp("reaction") if mol.HasProp("reaction") else None
        mtype = mol.GetProp("type") if mol.HasProp("type") else None
        if rid != rxn_id:
            continue
        if mtype not in ("r1h", "r2h"):
            continue
        out[mtype] = [a.GetSymbol() for a in mol.GetAtoms()]
    return out


def _validate_qm_symbols(qm_df: pd.DataFrame, sdf_symbols: List[str], rxn_id: str, mol_type: str) -> List[tuple]:
    if len(qm_df) != len(sdf_symbols):
        raise ValueError(
            f"Atom count mismatch for {rxn_id}/{mol_type}: "
            f"log has {len(qm_df)}, sdf has {len(sdf_symbols)}"
        )
    bad = []
    for i, s in enumerate(sdf_symbols):
        qsym = str(qm_df.iloc[i]["atom_symbol"]) if pd.notna(qm_df.iloc[i]["atom_symbol"]) else ""
        if qsym and qsym != s:
            bad.append((i, s, qsym))
    return bad


def _kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    p_cent = P.mean(axis=0)
    q_cent = Q.mean(axis=0)
    P0 = P - p_cent
    Q0 = Q - q_cent
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return P0 @ R + q_cent


def _greedy_symbolwise_assignment(
    log_xyz: np.ndarray,
    sdf_xyz: np.ndarray,
    log_syms: List[str],
    sdf_syms: List[str],
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    symbols = sorted(set(log_syms))
    for sym in symbols:
        log_idx = [i for i, s in enumerate(log_syms) if s == sym]
        sdf_idx = [i for i, s in enumerate(sdf_syms) if s == sym]
        if len(log_idx) != len(sdf_idx):
            raise ValueError(f"Cannot map symbol '{sym}': counts differ ({len(log_idx)} vs {len(sdf_idx)})")

        # Greedy unique matching by distance.
        used_log = set()
        used_sdf = set()
        pairs = []
        for li in log_idx:
            for sj in sdf_idx:
                d = float(np.linalg.norm(log_xyz[li] - sdf_xyz[sj]))
                pairs.append((d, li, sj))
        pairs.sort(key=lambda t: t[0])
        for _, li, sj in pairs:
            if li in used_log or sj in used_sdf:
                continue
            mapping[li] = sj
            used_log.add(li)
            used_sdf.add(sj)

        if len(used_log) != len(log_idx):
            raise ValueError(f"Greedy mapping failed for symbol '{sym}'")
    return mapping


def _remap_qm_to_sdf(qm_df: pd.DataFrame, sdf_symbols: List[str], sdf_xyz: np.ndarray) -> pd.DataFrame:
    req = ["x", "y", "z"]
    if any(c not in qm_df.columns for c in req):
        raise ValueError("QM dataframe missing coordinates; remap is unavailable.")
    if qm_df[req].isna().any().any():
        raise ValueError("QM coordinates contain NaNs; remap is unavailable.")

    q = qm_df.sort_values("focus_atom_idx").reset_index(drop=True).copy()
    log_xyz = q[["x", "y", "z"]].to_numpy(dtype=float)
    sdf_xyz = np.asarray(sdf_xyz, dtype=float)
    if log_xyz.shape != sdf_xyz.shape:
        raise ValueError(f"Coordinate shape mismatch log={log_xyz.shape} sdf={sdf_xyz.shape}")

    aligned = _kabsch_align(log_xyz, sdf_xyz)
    log_syms = q["atom_symbol"].astype(str).tolist()
    mapping = _greedy_symbolwise_assignment(aligned, sdf_xyz, log_syms, sdf_symbols)

    q["focus_atom_idx"] = q["focus_atom_idx"].map(lambda old: int(mapping[int(old)]))
    q = q.sort_values("focus_atom_idx").reset_index(drop=True)
    return q


def _build_qm_table(manifest: pd.DataFrame, atom_map_mode: str = "strict") -> pd.DataFrame:
    rows = []
    remapped_count = 0
    for rec in manifest.to_dict(orient="records"):
        rxn_id = str(rec["rxn_id"])
        sdf_file = Path(rec["sdf_file"]).resolve()
        r1h_log = Path(rec["r1h_log"]).resolve()
        r2h_log = Path(rec["r2h_log"]).resolve()

        symbols = _collect_sdf_atom_symbols(sdf_file, rxn_id)
        if "r1h" not in symbols or "r2h" not in symbols:
            raise ValueError(f"SDF {sdf_file} missing r1h/r2h molecules for rxn_id={rxn_id}")

        # Collect SDF coordinates for remap.
        sdf_coords: Dict[str, np.ndarray] = {}
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            rid = mol.GetProp("reaction") if mol.HasProp("reaction") else None
            mtype = mol.GetProp("type") if mol.HasProp("type") else None
            if rid != rxn_id or mtype not in ("r1h", "r2h"):
                continue
            conf = mol.GetConformer()
            xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=float)
            sdf_coords[mtype] = xyz

        for mol_type, log_path in (("r1h", r1h_log), ("r2h", r2h_log)):
            qm = parse_gaussian_qm(str(log_path)).sort_values("focus_atom_idx").reset_index(drop=True)
            mismatches = _validate_qm_symbols(qm, symbols[mol_type], rxn_id, mol_type)
            if mismatches:
                if atom_map_mode == "strict":
                    ex = mismatches[:5]
                    raise ValueError(f"Atom symbol mismatch for {rxn_id}/{mol_type}, examples: {ex}")
                qm = _remap_qm_to_sdf(qm, symbols[mol_type], sdf_coords[mol_type])
                # confirm after remap
                post = _validate_qm_symbols(qm, symbols[mol_type], rxn_id, mol_type)
                if post:
                    ex = post[:5]
                    raise ValueError(f"Remap failed for {rxn_id}/{mol_type}, examples: {ex}")
                remapped_count += 1

            qm["rxn_id"] = rxn_id
            qm["mol_type"] = mol_type
            rows.append(qm[["rxn_id", "mol_type", "focus_atom_idx", "q_mull", "q_apt", "f_mag"]])

    out = pd.concat(rows, axis=0, ignore_index=True)
    out.attrs["remapped_count"] = remapped_count
    return out


def _run_create_rad(create_rad_py: Path, sdf_files: List[Path], output_dir: Path):
    mod = _load_create_rad_module(create_rad_py)
    sdf_list = [str(p) for p in sdf_files]

    out_rad = output_dir / "geom_features_rad.csv"
    out_path = output_dir / "geom_features_path.csv"

    mod.create_atom_feat_RAD_table(
        sdf_list,
        reference_labels=("d_hydrogen", "a_hydrogen", "acceptor"),
        path_only=False,
        hybrid=False,
        angle_units="radian",
        dihedral_units="radian",
        output_csv=str(out_rad),
        mode_tag="default",
    )
    mod.create_atom_feat_RAD_table(
        sdf_list,
        reference_labels=("d_hydrogen", "a_hydrogen", "acceptor"),
        path_only=True,
        hybrid=False,
        angle_units="radian",
        dihedral_units="radian",
        output_csv=str(out_path),
        mode_tag="path-only",
    )
    return out_rad, out_path


def _merge_rad_qm(rad_csv: Path, qm_df: pd.DataFrame, out_csv: Path):
    rad = pd.read_csv(rad_csv)
    merged = rad.merge(qm_df, on=["rxn_id", "mol_type", "focus_atom_idx"], how="left", validate="many_to_one")
    merged.to_csv(out_csv, index=False)
    return {
        "rad_rows": int(len(rad)),
        "merged_rows": int(len(merged)),
        "missing_q_mull": int(merged["q_mull"].isna().sum()),
        "missing_q_apt": int(merged["q_apt"].isna().sum()),
        "missing_f_mag": int(merged["f_mag"].isna().sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build atom_with_geom_feats CSVs from SDF + Gaussian freq logs.")
    p.add_argument("--manifest-csv", required=True, help="CSV with columns: rxn_id,sdf_file,r1h_log,r2h_log")
    p.add_argument(
        "--create-rad-py",
        default=str((Path(__file__).resolve().parent / "create_RAD.py")),
        help="Path to create_RAD.py script used for RAD/geometry features.",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for geom and merged atom feature CSVs.")
    p.add_argument(
        "--atom-map-mode",
        choices=["strict", "remap"],
        default="strict",
        help="strict: require Gaussian atom order to match SDF order; remap: attempt coordinate-based remapping on mismatch.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    manifest_csv = Path(args.manifest_csv).resolve()
    create_rad_py = Path(args.create_rad_py).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_csv}")
    if not create_rad_py.is_file():
        raise FileNotFoundError(f"Missing create_RAD.py: {create_rad_py}")

    manifest = pd.read_csv(manifest_csv)
    need_cols = ["rxn_id", "sdf_file", "r1h_log", "r2h_log"]
    missing = [c for c in need_cols if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Normalize paths and validate existence.
    for col in ("sdf_file", "r1h_log", "r2h_log"):
        manifest[col] = manifest[col].map(lambda x: str(Path(x).resolve()))
        bad = [p for p in manifest[col].tolist() if not Path(p).is_file()]
        if bad:
            raise FileNotFoundError(f"{col} contains missing files, examples: {bad[:3]}")

    sdf_files = sorted({Path(p) for p in manifest["sdf_file"].tolist()})

    geom_rad, geom_path = _run_create_rad(create_rad_py, sdf_files, output_dir)
    qm_df = _build_qm_table(manifest, atom_map_mode=str(args.atom_map_mode))
    qm_df.to_csv(output_dir / "qm_atom_features.csv", index=False)

    out_rad = output_dir / "atom_with_geom_feats_rad.csv"
    out_path = output_dir / "atom_with_geom_feats_path.csv"
    stats_rad = _merge_rad_qm(geom_rad, qm_df, out_rad)
    stats_path = _merge_rad_qm(geom_path, qm_df, out_path)

    report = {
        "manifest_csv": str(manifest_csv),
        "create_rad_py": str(create_rad_py),
        "rows_manifest": int(len(manifest)),
        "rows_qm": int(len(qm_df)),
        "atom_map_mode": str(args.atom_map_mode),
        "remapped_components": int(qm_df.attrs.get("remapped_count", 0)),
        "outputs": {
            "geom_features_rad": str(geom_rad),
            "geom_features_path": str(geom_path),
            "atom_with_geom_feats_rad": str(out_rad),
            "atom_with_geom_feats_path": str(out_path),
        },
        "merge_stats": {
            "rad": stats_rad,
            "path": stats_path,
        },
    }
    with open(output_dir / "qm_merge_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Wrote {out_rad}")
    print(f"[OK] Wrote {out_path}")
    print(f"[OK] Wrote {output_dir / 'qm_merge_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
