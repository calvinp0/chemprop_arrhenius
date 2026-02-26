import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from arrhenius.modeling.module.pl_rateconstant_dir import ArrheniusMultiComponentMPNN
from arrhenius.modeling.nn.transformers import UnscaleColumnTransform
from arrhenius.splitters.k_stone import ks_make_split_indices
from arrhenius.splitters.random import random_grouped_split_indices
from run_hpo.data import (
    compute_arrhenius_scalers_from_train,
    fit_global_normalizers,
    make_loaders,
    prepare_data,
)
from run_hpo.model_build import model_factory_from_cfg
from run_hpo.feature_modes import canonicalize_extra_mode, mode_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retrain an ensemble with the best configuration from HPO and evaluate on a hold-out split."
    )
    p.add_argument("--summary-json", required=True, help="Path to the final summary JSON.")
    p.add_argument("--sdf-path", required=True)
    p.add_argument("--target-csv", required=True)
    p.add_argument("--rad-dir", help="RAD features directory (required when extra_mode uses atom extras).")
    p.add_argument("--output-dir", required=True, help="Directory to save parity plots and prediction CSV.")
    p.add_argument("--accelerator", default=None, help="Optional Lightning accelerator override (e.g. 'cpu', 'gpu', 'auto').")
    p.add_argument("--devices", default=None, help="Optional Lightning devices override (e.g. 1, 'auto').")
    p.add_argument("--ensemble-size", type=int, default=5, help="Number of ensemble members to train.")
    p.add_argument("--test-frac", type=float, default=0.10, help="Fraction of data reserved for hold-out testing.")
    p.add_argument("--seed", type=int, help="Override the random seed used for splitting and training.")
    p.add_argument("--save-members", action="store_true", help="Export trained ensemble members for later reuse.")
    p.add_argument("--member-format", choices=["ckpt", "state_dict"], default="ckpt",
                   help="Format used when exporting ensemble members (default: reuse best Lightning checkpoints).")
    p.add_argument("--save-normalizers", action="store_true",
                   help="Persist fitted scalers/normalizers alongside the models for deployment.")
    return p


def _resolve_devices(
    cfg: Dict[str, Any],
    accelerator_override: Optional[str] = None,
    devices_override: Optional[str | int] = None,
) -> Tuple[str | int, str | int]:
    accelerator_cfg = accelerator_override if accelerator_override is not None else cfg.get("accelerator", "auto")
    if isinstance(accelerator_cfg, str):
        accelerator = accelerator_cfg.lower()
    else:
        accelerator = accelerator_cfg

    devices_cfg = devices_override if devices_override is not None else cfg.get("devices", 1)
    if isinstance(devices_cfg, str):
        dv = devices_cfg.strip()
        if dv.lower() == "auto":
            devices = "auto"
        else:
            try:
                devices = int(dv)
            except ValueError:
                devices = dv
    else:
        devices = devices_cfg

    return accelerator, devices


def _resolve_precision(cfg: Dict[str, Any], accelerator: str | int) -> str | int:
    precision_cfg = cfg.get("precision")
    if precision_cfg:
        return precision_cfg
    has_cuda = pl.utilities.device_parser.num_cuda_devices() > 0
    if accelerator == "cpu" or (accelerator == "auto" and not torch.cuda.is_available()):
        return "32-true"
    return "16-mixed"


def _make_holdout_split(
    cfg: Dict[str, Any],
    bundle,
    test_frac: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0,1); received {test_frac}")

    train_frac = 1.0 - test_frac
    splitter = str(cfg.get("splitter", "kstone")).lower()
    data = (bundle.donors_kept, bundle.acceptors_kept)
    kwargs = {
        "sizes": (train_frac, 0.0, test_frac),
        "seed": seed,
        "num_replicates": 1,
    }
    if splitter == "random":
        train_reps, _, test_reps = random_grouped_split_indices(
            data,
            pair_key_mode="inchikey",
            unordered_pairs=True,
            pair_group_keys=bundle.pair_group_keys,
            **kwargs,
        )
    else:
        train_reps, _, test_reps = ks_make_split_indices(
            data,
            distance_metric=str(cfg["distance_metric"]),
            joint_mode=str(cfg["joint_mode"]),
            donor_weight=float(cfg["donor_weight"]),
            p_norm=float(cfg["p_norm"]),
            fingerprint="morgan_fingerprint",
            fprints_hopts={"n_bits": int(cfg["n_bits"]), "radius": int(cfg["radius"])},
            group_by_pair=True,
            unordered_pairs=True,
            pair_key_mode="inchikey",
            keep_all_group_members=True,
            pair_group_keys=bundle.pair_group_keys,
            **kwargs,
        )

    train_idx = list(map(int, train_reps[0]))
    test_idx = list(map(int, test_reps[0]))
    return train_idx, test_idx


def _inner_splits_for_ensemble(
    cfg: Dict[str, Any],
    bundle,
    base_indices: Sequence[int],
    ensemble_size: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive.")

    base_indices = list(map(int, base_indices))
    donors = [bundle.donors_kept[i] for i in base_indices]
    acceptors = [bundle.acceptors_kept[i] for i in base_indices]
    group_keys = [bundle.pair_group_keys[i] for i in base_indices]

    splitter = str(cfg.get("splitter", "kstone")).lower()
    kwargs = {
        "sizes": (0.85, 0.15, 0.0),
        "seed": seed,
        "num_replicates": ensemble_size,
    }
    if splitter == "random":
        train_reps, val_reps, _ = random_grouped_split_indices(
            (donors, acceptors),
            unordered_pairs=True,
            pair_key_mode="inchikey",
            pair_group_keys=group_keys,
            **kwargs,
        )
    else:
        train_reps, val_reps, _ = ks_make_split_indices(
            (donors, acceptors),
            distance_metric=str(cfg["distance_metric"]),
            joint_mode=str(cfg["joint_mode"]),
            donor_weight=float(cfg["donor_weight"]),
            p_norm=float(cfg["p_norm"]),
            fingerprint="morgan_fingerprint",
            fprints_hopts={"n_bits": int(cfg["n_bits"]), "radius": int(cfg["radius"])},
            group_by_pair=True,
            unordered_pairs=True,
            pair_key_mode="inchikey",
            keep_all_group_members=True,
            pair_group_keys=group_keys,
            **kwargs,
        )

    splits: List[Tuple[List[int], List[int]]] = []
    for tr_loc, val_loc in zip(train_reps, val_reps, strict=False):
        train_abs = [base_indices[i] for i in map(int, tr_loc)]
        val_abs = [base_indices[i] for i in map(int, val_loc)]
        splits.append((train_abs, val_abs))
    return splits


def _train_single_member(
    rep_id: int,
    cfg: Dict[str, Any],
    bundle,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    global_norms: Dict[str, Any],
    accelerator: str | int,
    devices: str | int,
    precision: str | int,
    output_dir: Path,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    pl.seed_everything(seed, workers=True)

    loaders = make_loaders(
        bundle=bundle,
        cfg=cfg,
        train_idx=list(map(int, train_idx)),
        val_idx=list(map(int, val_idx)),
        test_idx=list(map(int, test_idx)),
        seed=seed,
        preset_r_bounds=global_norms.get("r_bounds"),
        preset_y_scaler=global_norms.get("y_scaler"),
        preset_vf_scaler=global_norms.get("vf_scaler"),
        preset_xd_scaler=global_norms.get("xd_scaler"),
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    if test_loader is None:
        raise RuntimeError("Hold-out test loader was not constructed.")

    y_scaler = loaders["y_scaler"]
    unscaler = UnscaleColumnTransform.from_column_transformer(y_scaler).eval()

    arr_mean_for, arr_scale_for, arr_mean_rev, arr_scale_rev = compute_arrhenius_scalers_from_train(
        train_loader,
        y_scaler,
        cfg["temperatures"],
    )

    model = model_factory_from_cfg(
        cfg,
        unscaler=unscaler,
        ea_scales_for=y_scaler.named_transformers_["t2"],
        ea_scales_rev=y_scaler.named_transformers_["t5"],
        arr_mean_for=arr_mean_for,
        arr_scale_for=arr_scale_for,
        arr_mean_rev=arr_mean_rev,
        arr_scale_rev=arr_scale_rev,
        featurizer=bundle.featurizer,
        x_d_dim=bundle.x_d_dim,
    )

    patience = int(cfg.get("patience", 10))
    callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=patience)]

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"ensemble-rep{rep_id:02d}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{ckpt_name}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_cb)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(cfg.get("max_epochs", 150)),
        callbacks=callbacks,
        deterministic=True,
        precision=precision,
    )

    trainer.fit(model, train_loader, val_loader)

    val_loss = float(trainer.callback_metrics["val_loss"].item())
    best_ckpt_path = checkpoint_cb.best_model_path or None

    test_metrics: Dict[str, float] = {}
    test_loss: Optional[float] = None
    test_out = trainer.test(model, dataloaders=test_loader, verbose=False, ckpt_path=best_ckpt_path)
    if test_out:
        raw_metrics = test_out[0]
        for k, v in raw_metrics.items():
            test_metrics[str(k)] = float(v)
        test_loss = test_metrics.get("test_loss")

    predict_outputs = trainer.predict(
        model,
        dataloaders=test_loader,
        ckpt_path=best_ckpt_path,
    )
    if not predict_outputs:
        raise RuntimeError(f"No predictions produced for ensemble member {rep_id}.")

    frac_A: List[np.ndarray] = []
    frac_n: List[np.ndarray] = []
    frac_Ea: List[np.ndarray] = []
    for batch_out in predict_outputs:
        if hasattr(batch_out, "Y_f_raw"):
            raw_t = batch_out.Y_f_raw
        elif isinstance(batch_out, dict) and "Y_f_raw" in batch_out:
            raw_t = batch_out["Y_f_raw"]
        elif isinstance(batch_out, dict) and "y_pred_raw" in batch_out:
            raw_t = batch_out["y_pred_raw"][:, :3]
        else:
            raise TypeError(f"Unsupported prediction output type: {type(batch_out)!r}")

        raw = raw_t.detach().cpu().numpy() if hasattr(raw_t, "detach") else np.asarray(raw_t)
        frac_A.append(raw[:, 0])
        frac_n.append(raw[:, 1])
        frac_Ea.append(raw[:, 2])

    preds = {
        "A": np.concatenate(frac_A, axis=0),
        "n": np.concatenate(frac_n, axis=0),
        "Ea": np.concatenate(frac_Ea, axis=0),
    }
    if preds["A"].shape[0] != len(test_idx):
        raise RuntimeError(
            f"Prediction length mismatch for ensemble member {rep_id} "
            f"({preds['A'].shape[0]} vs expected {len(test_idx)})."
        )

    info = {
        "replicate": rep_id,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "val_loss": val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "checkpoint_path": best_ckpt_path,
    }
    return preds, info


def _train_ensemble_members(
    cfg: Dict[str, Any],
    bundle,
    train_base_idx: List[int],
    test_idx: List[int],
    ensemble_size: int,
    base_seed: int,
    global_norms: Dict[str, Any],
    accelerator: str | int,
    devices: str | int,
    precision: str | int,
    output_dir: Path,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    splits = _inner_splits_for_ensemble(cfg, bundle, train_base_idx, ensemble_size, base_seed)

    preds_A: List[np.ndarray] = []
    preds_n: List[np.ndarray] = []
    preds_Ea: List[np.ndarray] = []
    replicate_info: List[Dict[str, Any]] = []

    for rep_id, (train_idx, val_idx) in enumerate(splits, start=1):
        sub_seed = base_seed + rep_id * 1000
        rep_dir = output_dir / f"rep{rep_id:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Training ensemble member {rep_id}/{ensemble_size} (seed={sub_seed}).")
        preds, info = _train_single_member(
            rep_id=rep_id,
            cfg=cfg,
            bundle=bundle,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            global_norms=global_norms,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            output_dir=rep_dir,
            seed=sub_seed,
        )
        preds_A.append(preds["A"])
        preds_n.append(preds["n"])
        preds_Ea.append(preds["Ea"])
        replicate_info.append(info)

    stacked_preds = {
        "A": np.vstack(preds_A),
        "n": np.vstack(preds_n),
        "Ea": np.vstack(preds_Ea),
    }
    return stacked_preds, replicate_info


def _export_ensemble_members(
    replicates: List[Dict[str, Any]],
    output_dir: Path,
    export_format: str = "ckpt",
) -> List[str]:
    export_dir = output_dir / "exported_members"
    export_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: List[str] = []

    for info in replicates:
        ckpt_path = info.get("checkpoint_path")
        rep_idx = info.get("replicate")
        if ckpt_path is None or rep_idx is None:
            continue
        src = Path(ckpt_path)
        if not src.exists():
            print(f"[WARN] Skipping export for replicate {rep_idx}: checkpoint not found at {src}")
            continue

        if export_format == "ckpt":
            dest = export_dir / f"ensemble_rep{int(rep_idx):02d}.ckpt"
            shutil.copy2(src, dest)
        else:
            model = ArrheniusMultiComponentMPNN.load_from_checkpoint(str(src), map_location="cpu")
            model.eval()
            dest = export_dir / f"ensemble_rep{int(rep_idx):02d}.pt"
            torch.save(model.state_dict(), dest)
        exported_paths.append(str(dest))
        print(f"[INFO] Exported ensemble member {rep_idx} -> {dest}")
    return exported_paths


def _persist_normalizers(global_norms: Dict[str, Any], output_dir: Path) -> Optional[str]:
    payload = {}
    for key in ("y_scaler", "vf_scaler", "xd_scaler", "r_bounds"):
        value = global_norms.get(key)
        if value is not None:
            payload[key] = value
    if not payload:
        return None
    path = output_dir / "global_normalizers.joblib"
    joblib.dump(payload, path)
    print(f"[INFO] Saved global normalizers to {path}")
    return str(path)


def _prepare_bundle(cfg: Dict[str, Any], args) -> Any:
    extras_mode = canonicalize_extra_mode(cfg.get("extra_mode", "baseline"))
    mode_cfg = mode_settings(extras_mode)
    if mode_cfg["use_extras"]:
        assert args.rad_dir, "--rad-dir must be provided when the selected extra_mode uses atom extras"

    bundle = prepare_data(
        sdf_path=args.sdf_path,
        target_csv=args.target_csv,
        extras_mode=extras_mode,
        global_mode=cfg.get("global_mode", "none"),
        morgan_bits=int(cfg.get("morgan_bits", 2048)),
        morgan_radius=int(cfg.get("morgan_radius", 2)),
        rad_dir=args.rad_dir if mode_cfg["use_extras"] else None,
        rad_source=str(cfg.get("rad_source", "path")),
    )
    return bundle


def _true_targets(bundle, test_indices: Sequence[int]) -> Dict[str, np.ndarray]:
    names = []
    base_names = []
    A = []
    n = []
    Ea = []
    for idx in test_indices:
        dp = bundle.attached_pair_dps[0][idx]
        raw_name = dp.name
        names.append(raw_name)
        base_names.append(raw_name.replace("_r1h", "").replace("_r2h", ""))
        y = np.array(dp.y, dtype=float)
        A.append(np.power(10.0, y[0]))  # convert log10(A) back to A
        n.append(y[1])
        Ea.append(y[2])
    return {
        "names": np.array(names),
        "base_names": np.array(base_names),
        "A": np.array(A, dtype=float),
        "n": np.array(n, dtype=float),
        "Ea": np.array(Ea, dtype=float),
    }


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, label: str, path: str, log_scale: bool = False):
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7, edgecolors="none")
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.0)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(f"Parity plot for {label}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    devices_override = None
    if args.devices is not None:
        if isinstance(args.devices, str):
            dv = args.devices.strip()
            if dv:
                if dv.lower() == "auto":
                    devices_override = "auto"
                else:
                    try:
                        devices_override = int(dv)
                    except ValueError:
                        devices_override = dv
        else:
            devices_override = args.devices

    accelerator_override = args.accelerator.strip().lower() if isinstance(args.accelerator, str) else args.accelerator

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    selected = summary.get("selected_trial")
    if not selected:
        raise ValueError("Summary JSON missing 'selected_trial' entry.")

    best_cfg = selected.get("cfg")
    if not best_cfg:
        raise ValueError("Selected trial does not contain a configuration.")

    base_seed = int(args.seed if args.seed is not None else best_cfg.get("seed", 42))
    ensemble_size = int(args.ensemble_size)
    test_frac = float(args.test_frac)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = _prepare_bundle(best_cfg, args)

    pl.seed_everything(base_seed, workers=True)

    train_base_idx, test_idx = _make_holdout_split(best_cfg, bundle, test_frac, base_seed)
    print(f"[INFO] Hold-out split: {len(train_base_idx)} train candidates, {len(test_idx)} test samples.")

    global_norms = fit_global_normalizers(bundle, best_cfg, train_base_idx)

    accelerator, devices = _resolve_devices(best_cfg, accelerator_override, devices_override)
    precision = _resolve_precision(best_cfg, accelerator)

    predictions, replicate_info = _train_ensemble_members(
        cfg=best_cfg,
        bundle=bundle,
        train_base_idx=train_base_idx,
        test_idx=test_idx,
        ensemble_size=ensemble_size,
        base_seed=base_seed,
        global_norms=global_norms,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        output_dir=output_dir,
    )

    targets = _true_targets(bundle, test_idx)

    A_preds = predictions["A"]
    n_preds = predictions["n"]
    Ea_preds = predictions["Ea"]

    ensemble_A = A_preds.mean(axis=0)
    ensemble_n = n_preds.mean(axis=0)
    ensemble_Ea = Ea_preds.mean(axis=0)

    df = pd.DataFrame({
        "name": targets["names"],
        "reaction_id": targets["base_names"],
        "A_true": targets["A"],
        "n_true": targets["n"],
        "Ea_true": targets["Ea"],
        "A_pred_mean": ensemble_A,
        "A_pred_std": A_preds.std(axis=0),
        "n_pred_mean": ensemble_n,
        "n_pred_std": n_preds.std(axis=0),
        "Ea_pred_mean": ensemble_Ea,
        "Ea_pred_std": Ea_preds.std(axis=0),
        "log10A_true": np.log10(targets["A"]),
        "log10A_pred_mean": np.log10(np.clip(ensemble_A, a_min=1e-30, a_max=None)),
    })

    csv_path = output_dir / "ensemble_predictions.csv"
    df.to_csv(csv_path, index=False)

    exported_members: List[str] = []
    normalizers_path: Optional[str] = None
    if args.save_members:
        exported_members = _export_ensemble_members(
            replicates=replicate_info,
            output_dir=output_dir,
            export_format=args.member_format,
        )
    if args.save_normalizers or args.save_members:
        normalizers_path = _persist_normalizers(global_norms, output_dir)

    metadata = {
        "best_cfg": best_cfg,
        "ensemble_size": ensemble_size,
        "test_frac": test_frac,
        "seed": base_seed,
        "splitter": str(best_cfg.get("splitter", "kstone")),
        "train_indices": train_base_idx,
        "test_indices": test_idx,
        "replicates": replicate_info,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "r_bounds": list(global_norms.get("r_bounds", ())) if global_norms.get("r_bounds") else None,
        "exported_members": exported_members,
        "normalizers_path": normalizers_path,
    }
    meta_path = output_dir / "ensemble_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    _plot_parity(targets["A"], ensemble_A, "A", output_dir / "parity_A.png", log_scale=True)
    _plot_parity(targets["n"], ensemble_n, "n", output_dir / "parity_n.png")
    _plot_parity(targets["Ea"], ensemble_Ea, "Ea", output_dir / "parity_Ea.png")

    print(f"[OK] Saved ensemble predictions to {csv_path}")
    print(f"[OK] Metadata written to {meta_path}")
    print(f"[OK] Parity plots written to {output_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
