from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference from an exported training/HPO model directory."
    )
    p.add_argument(
        "--model-dir",
        "--bundle-dir",
        dest="model_dir",
        required=True,
        help="Directory containing ensemble_metadata.json and exported members.",
    )
    p.add_argument("--sdf-path", required=True, help="SDF used for featurisation.")
    p.add_argument("--input-csv", required=True, help="Input CSV of reactions to score.")
    p.add_argument(
        "--rad-dir", help="RAD feature directory (required when extra_mode uses atom extras)."
    )
    p.add_argument("--output-csv", required=True, help="Where to write predictions CSV.")
    p.add_argument(
        "--summary-json", help="Optional final summary JSON fallback if metadata lacks best_cfg."
    )
    p.add_argument(
        "--max-members", type=int, default=0, help="Limit ensemble members used (0 = all)."
    )
    p.add_argument("--seed", type=int, default=None, help="Seed override for loader determinism.")
    p.add_argument(
        "--accelerator", default="cpu", help="Lightning accelerator for prediction (cpu/gpu/auto)."
    )
    p.add_argument("--devices", default="1", help="Lightning devices value (e.g. 1 or auto).")
    p.add_argument(
        "--precision", default=None, help="Optional precision override, e.g. 32-true or 16-mixed."
    )
    return p


def _resolve_devices(devices: str) -> str | int:
    dv = str(devices).strip()
    if dv.lower() == "auto":
        return "auto"
    try:
        return int(dv)
    except ValueError:
        return dv


def _rxn_id(name: str) -> str:
    return str(name).replace("_r1h", "").replace("_r2h", "")


def _smiles(mol_obj) -> Optional[str]:
    from rdkit import Chem

    mol = mol_obj[0] if isinstance(mol_obj, tuple) else mol_obj
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _ensure_target_columns(input_csv: Path, target_cols: Sequence[str], output_dir: Path) -> Path:
    df = pd.read_csv(input_csv)
    missing = [c for c in target_cols if c not in df.columns]
    if not missing:
        return input_csv

    for c in missing:
        df[c] = 0.0
    tmp = output_dir / "_predict_input_with_targets.csv"
    df.to_csv(tmp, index=False)
    print(f"[INFO] Added placeholder target columns to input CSV: {missing}")
    return tmp


def _load_metadata(bundle_dir: Path) -> Dict[str, Any]:
    meta_path = bundle_dir / "ensemble_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def _load_best_cfg(metadata: Dict[str, Any], summary_json: Optional[str]) -> Dict[str, Any]:
    cfg = metadata.get("best_cfg")
    if cfg:
        return cfg
    if summary_json:
        with open(summary_json, "r") as f:
            summary = json.load(f)
        selected = summary.get("selected_trial", {})
        cfg = selected.get("cfg")
        if cfg:
            return cfg
    raise ValueError(
        "Could not find best_cfg in metadata. Re-export with updated ensemble_eval.py "
        "or provide --summary-json from arrhenius.training.hpo final summary."
    )


def _resolve_member_paths(bundle_dir: Path, metadata: Dict[str, Any]) -> List[Path]:
    candidates: List[str] = list(metadata.get("exported_members", []) or [])
    if not candidates:
        for rec in metadata.get("replicates", []) or []:
            ckpt = rec.get("checkpoint_path")
            if ckpt:
                candidates.append(str(ckpt))

    out: List[Path] = []
    for raw in candidates:
        p = Path(raw)
        if p.is_file():
            out.append(p)
            continue
        q = (bundle_dir / raw).resolve()
        if q.is_file():
            out.append(q)
            continue
        r = (bundle_dir / "exported_members" / Path(raw).name).resolve()
        if r.is_file():
            out.append(r)
    uniq = []
    seen = set()
    for p in out:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _extract_raw(outputs: List[Any]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for b in outputs:
        if hasattr(b, "Y_f_raw"):
            raw = b.Y_f_raw
        elif isinstance(b, dict) and "Y_f_raw" in b:
            raw = b["Y_f_raw"]
        elif isinstance(b, dict) and "y_pred_raw" in b:
            raw = b["y_pred_raw"][:, :3]
        else:
            raise TypeError(f"Unsupported prediction output type: {type(b)!r}")

        arr = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
        chunks.append(arr[:, :3])
    if not chunks:
        raise RuntimeError("No prediction outputs were produced.")
    return np.concatenate(chunks, axis=0)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    from lightning import pytorch as pl
    from arrhenius.modeling.module.pl_rateconstant_dir import ArrheniusMultiComponentMPNN
    from arrhenius.training.hpo.data import make_loaders, prepare_data
    from arrhenius.training.hpo.feature_modes import canonicalize_extra_mode, mode_settings

    bundle_dir = Path(args.model_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(bundle_dir)
    cfg = _load_best_cfg(metadata, args.summary_json)
    extras_mode = canonicalize_extra_mode(cfg.get("extra_mode", "baseline"))
    mode_cfg = mode_settings(extras_mode)

    try:
        import joblib  # local import to keep --help usable even if optional deps are missing
    except Exception as e:
        raise RuntimeError(
            "joblib is required for inference bundle loading. Install it in the active environment."
        ) from e

    normalizers_path = metadata.get("normalizers_path") or str(
        bundle_dir / "global_normalizers.joblib"
    )
    normalizers_path = Path(normalizers_path)
    if not normalizers_path.is_file():
        candidate = bundle_dir / normalizers_path.name
        if candidate.is_file():
            normalizers_path = candidate
    if not normalizers_path.is_file():
        raise FileNotFoundError(
            "Could not find global normalizers joblib in bundle. "
            "Export with --save-normalizers (or --save-members)."
        )
    norms = joblib.load(normalizers_path)

    member_paths = _resolve_member_paths(bundle_dir, metadata)
    if args.max_members and args.max_members > 0:
        member_paths = member_paths[: int(args.max_members)]
    if not member_paths:
        raise FileNotFoundError("No member checkpoints found in bundle metadata/exported_members.")

    target_cols = ["A_log10", "n", "Ea"]
    input_csv = _ensure_target_columns(
        Path(args.input_csv).resolve(), target_cols, output_csv.parent
    )

    if mode_cfg["use_extras"] and not args.rad_dir:
        raise ValueError(f"--rad-dir is required for extra_mode='{extras_mode}'.")

    bundle = prepare_data(
        sdf_path=str(Path(args.sdf_path).resolve()),
        target_csv=str(input_csv),
        extras_mode=extras_mode,
        global_mode=str(cfg.get("global_mode", "none")),
        morgan_bits=int(cfg.get("morgan_bits", 2048)),
        morgan_radius=int(cfg.get("morgan_radius", 2)),
        rad_dir=str(Path(args.rad_dir).resolve())
        if (mode_cfg["use_extras"] and args.rad_dir)
        else None,
        rad_source=str(cfg.get("rad_source", "path")),
    )

    n = len(bundle.attached_pair_dps[0])
    all_idx = list(range(n))
    seed = int(args.seed if args.seed is not None else metadata.get("seed", cfg.get("seed", 42)))
    local_cfg = dict(cfg)
    local_cfg["batch_size"] = int(cfg.get("batch_size", 128))
    if args.precision is not None:
        local_cfg["precision"] = args.precision

    loaders = make_loaders(
        bundle=bundle,
        cfg=local_cfg,
        train_idx=all_idx,
        val_idx=all_idx,
        test_idx=all_idx,
        seed=seed,
        preset_r_bounds=norms.get("r_bounds"),
        preset_y_scaler=norms.get("y_scaler"),
        preset_vf_scaler=norms.get("vf_scaler"),
        preset_xd_scaler=norms.get("xd_scaler"),
    )
    test_loader = loaders.get("test_loader")
    if test_loader is None:
        raise RuntimeError("Failed to construct test loader for inference.")

    accelerator = (
        str(args.accelerator).strip().lower()
        if isinstance(args.accelerator, str)
        else args.accelerator
    )
    devices = _resolve_devices(args.devices)
    precision = args.precision if args.precision is not None else cfg.get("precision", "32-true")
    if accelerator == "cpu" and str(precision).startswith("16"):
        precision = "32-true"
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )

    member_preds: List[np.ndarray] = []
    used_members: List[str] = []
    for ckpt in member_paths:
        print(f"[INFO] Predicting with member: {ckpt}")
        model = ArrheniusMultiComponentMPNN.load_from_checkpoint(str(ckpt), map_location="cpu")
        model.eval()
        outputs = trainer.predict(model, dataloaders=test_loader)
        raw = _extract_raw(outputs)
        if raw.shape[0] != n:
            raise RuntimeError(
                f"Prediction length mismatch for {ckpt}: {raw.shape[0]} vs expected {n}"
            )
        member_preds.append(raw)
        used_members.append(str(ckpt))

    stack = np.stack(member_preds, axis=0)  # (M, N, 3)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        dp_d = bundle.attached_pair_dps[0][i]
        dp_a = bundle.attached_pair_dps[1][i]
        A = float(mean[i, 0])
        rows.append(
            {
                "index": i,
                "name": str(dp_d.name),
                "reaction_id": _rxn_id(dp_d.name),
                "donor_smiles": _smiles(dp_d.mol),
                "acceptor_smiles": _smiles(dp_a.mol),
                "A_pred_mean": A,
                "A_pred_std": float(std[i, 0]),
                "n_pred_mean": float(mean[i, 1]),
                "n_pred_std": float(std[i, 1]),
                "Ea_pred_mean": float(mean[i, 2]),
                "Ea_pred_std": float(std[i, 2]),
                "log10A_pred_mean": float(np.log10(max(A, 1e-30))),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    run_meta = {
        "bundle_dir": str(bundle_dir),
        "members_used": used_members,
        "num_members": len(used_members),
        "num_samples": n,
        "output_csv": str(output_csv),
    }
    run_meta_path = output_csv.with_suffix(".meta.json")
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[OK] Wrote predictions to {output_csv}")
    print(f"[OK] Wrote run metadata to {run_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
