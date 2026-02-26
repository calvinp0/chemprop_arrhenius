# run_hpo/evaluation.py
from __future__ import annotations
import os
import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Hashable, Sequence, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from arrhenius.splitters.k_stone import ks_make_split_indices
from arrhenius.splitters.random import random_grouped_split_indices
from arrhenius.modeling.nn.transformers import UnscaleColumnTransform
from arrhenius.modeling.module.pl_rateconstant_dir import ArrheniusMultiComponentMPNN
from arrhenius.modeling.module.model_core import PredictBatchOutput
from arrhenius.data.collate import build_loader_mc
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from arrhenius.training.hpo.data import (
    make_loaders,
    compute_arrhenius_scalers_from_train,
    CP_NUM_WORKERS,
    torch_generator,
)
from arrhenius.training.hpo.model_build import model_factory_from_cfg
from arrhenius.training.hpo.splits import splits_signature
from arrhenius.training.hpo.loader_cache import LoaderCache


class _PseudoTrial:
    """Minimal stub miimicing optuna.trial.Trial with .number and .set_user_attr()."""

    def __init__(self, tag: str, number: int):
        self.number = number
        self._tag = tag
        self.user_attrs = {}

    # mimic the subset you actually use
    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def report(self, value, step=None):
        pass  # no-op, nothing to prune

    def should_prune(self):
        return False  # always keep going

    def __repr__(self):
        return f"<PseudoTrial {self._tag}:{self.number}>"


@dataclass
class TrainResult:
    val_loss: float
    val_metrics: Optional[Dict[str, float]] = None
    test_loss: Optional[float] = None
    test_metrics: Optional[Dict[str, float]] = None
    checkpoint_path: Optional[str] = None


def _unpack_mol(mol_obj):
    if isinstance(mol_obj, tuple):
        return mol_obj[0]
    return mol_obj


def _to_smiles(mol_obj) -> Optional[str]:
    mol = _unpack_mol(mol_obj)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _arrhenius_from_dp(dp) -> Tuple[float, float, float]:
    y = np.asarray(dp.y, dtype=float)
    A = float(np.power(10.0, y[0]))
    n = float(y[1])
    Ea = float(y[2])
    return A, n, Ea


def _serialize_standard_scaler(scaler: Optional[Any]) -> Optional[Dict[str, Any]]:
    if scaler is None:
        return None
    payload: Dict[str, Any] = {}
    for attr in ("mean_", "scale_", "var_"):
        if hasattr(scaler, attr):
            value = getattr(scaler, attr)
            if value is not None:
                payload[attr[:-1]] = np.asarray(value).tolist()
    for flag in ("with_mean", "with_std"):
        if hasattr(scaler, flag):
            payload[flag] = bool(getattr(scaler, flag))
    return payload


def _serialize_column_transformer(ct) -> Dict[str, Any]:
    data: Dict[str, Any] = {"transformers": []}
    if ct is None:
        return data
    for name, transformer, columns in getattr(ct, "transformers_", []):
        if transformer is None or name == "remainder":
            continue
        entry: Dict[str, Any] = {"name": name, "columns": columns}
        if hasattr(transformer, "mean_"):
            entry["mean"] = np.asarray(transformer.mean_).tolist()
        if hasattr(transformer, "scale_"):
            entry["scale"] = np.asarray(transformer.scale_).tolist()
        if hasattr(transformer, "var_"):
            entry["var"] = np.asarray(transformer.var_).tolist()
        if hasattr(transformer, "lambdas_"):
            entry["lambdas"] = np.asarray(transformer.lambdas_).tolist()
        data["transformers"].append(entry)
    return data


def _as_prediction_mapping(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        return out
    if isinstance(out, PredictBatchOutput):
        return out.to_numpy_dict()
    if hasattr(out, "to_numpy_dict"):
        return out.to_numpy_dict()
    if hasattr(out, "_asdict"):
        return out._asdict()
    raise TypeError(f"Unsupported prediction output type: {type(out)!r}")


def _stack_prediction_outputs(outputs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not outputs:
        return result
    normalized = [_as_prediction_mapping(out) for out in outputs]
    concat_keys = ["y_pred_raw", "y_true_raw", "y_pred_s", "y_true_s", "lnk_pred", "lnk_true"]
    for key in concat_keys:
        arrays = [np.asarray(out[key]) for out in normalized if key in out and out[key] is not None]
        if arrays:
            result[key] = np.concatenate(arrays, axis=0)
        else:
            result[key] = None
    temps = next((out.get("temps") for out in normalized if out.get("temps") is not None), None)
    result["temps"] = list(np.asarray(temps).tolist()) if temps is not None else None
    return result


def _build_eval_loader(dataset, batch_size: int, seed: int):
    if dataset is None:
        return None
    generator = torch_generator(seed)
    pin = torch.cuda.is_available()
    batch_size = max(1, batch_size)
    return build_loader_mc(
        dataset,
        batch_size=batch_size,
        generator=generator,
        shuffle=False,
        num_workers=CP_NUM_WORKERS,
        pin_memory=pin,
    )

def _log_predictions_from_checkpoints(
    checkpoint_records: Optional[List[Dict[str, Any]]],
    bundle,
    cfg: Dict[str, Any],
    logger,
    trial_id: int,
    loader_cache: Optional[LoaderCache] = None,
):
    """
    Reload saved checkpoints and log per-sample predictions (scaled/unscaled) into the TrialLogger.
    Only used for the final top-k CV sweep to avoid bloat during HPO.
    """
    if logger is None or trial_id is None or not checkpoint_records:
        return

    base_seed = cfg.get("seed", 42)
    batch_size = max(1, int(cfg.get("batch_size", 128)))

    for rec in checkpoint_records:
        ckpt = rec.get("checkpoint_path")
        if not ckpt or not os.path.exists(ckpt):
            continue
        fold_id = int(rec.get("fold_id", -1))
        replicate = int(rec.get("replicate", 0))
        train_idx = list(map(int, rec.get("train_indices", [])))
        val_idx = list(map(int, rec.get("val_indices", [])))
        test_idx = rec.get("test_indices")
        if test_idx is not None:
            test_idx = list(map(int, test_idx))

        sub_seed = base_seed + replicate * 1000
        loaders = make_loaders(
            bundle=bundle,
            cfg=cfg,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=sub_seed,
        )

        splits = [
            ("train", loaders.get("train_dataset"), train_idx, sub_seed + 101),
            ("val", loaders.get("val_dataset"), val_idx, sub_seed + 103),
        ]
        if test_idx:
            splits.append(("test", loaders.get("test_dataset"), test_idx, sub_seed + 105))

        model = ArrheniusMultiComponentMPNN.load_from_checkpoint(ckpt, map_location="cpu")
        model.eval()
        trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cpu", devices=1)

        for split_name, dataset, indices, seed_offset in splits:
            if dataset is None or not indices:
                continue
            loader = _build_eval_loader(dataset, min(batch_size, len(indices)), seed_offset)
            outputs = trainer.predict(model, dataloaders=loader)
            payloads = _stack_prediction_outputs(outputs)
            logger.log_predictions(trial_id, fold_id, replicate, split_name, indices, payloads)


def evaluate_trial_on_fold(
    fold_indices: Sequence[int],
    cfg: Dict[str, Any],
    pair_group_keys: Sequence[Hashable],
    donors,                           # Sequence[Chem.Mol]
    acceptors,                        # Sequence[Chem.Mol]
    attached_pair_dps,                # list[ list[chemprop.data.MoleculeDatapoint] ]  (len=2 for (donor, acceptor))
    featurizer,                       # chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer
    x_d_dim: int,
    trial_tag: str,
    log_root: str = "logs/hpo",
    seed: int = 42,
    logger=None,
    trial=None,
    fold_id: int = 0,
    save_checkpoints: bool = False,
    checkpoint_root: Optional[str] = None,
    checkpoint_records: Optional[List[Dict[str, Any]]] = None,
    loader_cache: Optional[LoaderCache] = None,
) -> float:
    """
    Evaluate one *outer fold* by running multiple Kennard–Stone inner splits
    on the data subset corresponding to `fold_indices`.

    Each inner split trains the model on ~85 % of that fold and validates on
    the remaining ~15 %, and the mean validation score is returned.

    Args:
        fold_indices: Indices of the datapoints in the outer fold to evaluate on.
        cfg: Configuration dictionary for training and evaluation.
        pair_group_keys: Group keys for the pairs, used for data splitting.
        donors: List of donor molecules.
        acceptors: List of acceptor molecules.
        attached_pair_dps: List of lists of MoleculeDatapoint objects for each pair.
        featurizer: Featurizer to convert molecules to graph representations.
        x_d_dim: Dimension of donor features.
        trial_tag: Tag for the trial, used in logging.
        log_root: Root directory for logging.
        seed: Random seed for reproducibility.
    
    Returns:
        Mean validation score across inner splits for the outer fold.
        
    """
    pl.seed_everything(seed, workers=True)
    fold_indices = np.asarray(fold_indices, dtype=int)
    fold_pair_keys = [pair_group_keys[i] for i in fold_indices]

    # KS Inner Replicates on the Outer Fold
    sub_donors = [donors[i] for i in fold_indices]
    sub_acceptors = [acceptors[i] for i in fold_indices]
    train_reps, val_reps, _ = _split_with_strategy(
        cfg,
        sub_donors,
        sub_acceptors,
        sizes=(0.85, 0.15, 0.0),
        seed=seed,
        num_reps=int(cfg["num_reps"]),
        pair_group_keys=fold_pair_keys,
    )

    val: List[float] = []
    for rep_id, (train_idx, val_idx) in enumerate(zip(train_reps, val_reps, strict=False), start=1):

        # map local indices (within fold) back to absolute indices
        train_abs_idx = [fold_indices[i] for i in train_idx]
        val_abs_idx = [fold_indices[i] for i in val_idx]
        run_dir = os.path.join(log_root, f"{trial_tag}-r{rep_id}")
        checkpoint_dir = None
        if save_checkpoints and checkpoint_root is not None:
            checkpoint_dir = os.path.join(checkpoint_root, f"rep{rep_id}")

        # Need to make a sub seed
        sub_seed = seed + rep_id * 1000

        if logger is not None:
            logger.log_split_indices(
                trial_id=trial.number,
                fold_id=fold_id,
                train=np.array(train_abs_idx, dtype=int),
                val=np.array(val_abs_idx, dtype=int),
            )

        result = _train_one(
            train_abs_idx,
            val_abs_idx,
            cfg,
            featurizer,
            attached_pair_dps,
            x_d_dim,
            pair_group_keys,
            sub_seed,
            run_dir,
            logger,
            fold_id=fold_id,
            trial=trial,
            test_idx=None,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f"{trial_tag}-r{rep_id}",
            loader_cache=loader_cache,
        )
        val.append(result.val_loss)
        if (
            checkpoint_records is not None
            and save_checkpoints
            and result.checkpoint_path is not None
        ):
            checkpoint_records.append(
                {
                    "fold_id": fold_id,
                    "replicate": rep_id,
                    "train_indices": train_abs_idx,
                    "val_indices": val_abs_idx,
                    "checkpoint_path": result.checkpoint_path,
                    "val_loss": result.val_loss,
                    "test_loss": result.test_loss,
                    "test_metrics": result.test_metrics,
                }
            )


    return float(np.mean(val))


def _train_one(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    cfg: Dict[str, Any],
    featurizer,
    attached_pair_dps,
    x_d_dim: int,
    pair_group_keys: Sequence[Hashable],
    seed: int = 42,
    log_dir: str = "logs/hpo",
    logger=None,
    fold_id: int = 0,
    trial=None,
    test_idx: Optional[Sequence[int]] = None,
    save_checkpoints: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
    loader_cache: Optional[LoaderCache] = None,
) -> TrainResult:
    """
    Train a model on "train_idx" and evaluate on "val_idx" and return loss/metric.

    """
    pl.seed_everything(seed, workers=True)

    test_list = list(map(int, test_idx)) if test_idx else None

    if loader_cache is not None:
        bundle_view = type(
            "BundleView", (), {"attached_pair_dps": attached_pair_dps, "featurizer": featurizer}
        )()
        loader_cache.bundle = bundle_view  # keep cache bundle in sync
        loaders = loader_cache.get_loaders(cfg, train_idx, val_idx, test_list, seed)
    else:
        loaders = make_loaders(
            bundle=type("BundleView", (), {  # quick view object so we don't refactor signatures elsewhere
                "attached_pair_dps": attached_pair_dps,
                "featurizer":        featurizer
            })(),
            cfg=cfg,
            train_idx=list(map(int, train_idx)),
            val_idx=list(map(int, val_idx)),
            test_idx=test_list,
            seed=seed,
        )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    y_scaler = loaders["y_scaler"]
    test_loader = loaders.get("test_loader")

    # Model (respect cfg temps)
    unscaler = UnscaleColumnTransform.from_column_transformer(y_scaler).eval()
    arr_layer_on = bool(cfg.get("enable_arrhenius_layer", True))
    arr_mean_for = arr_scale_for = arr_mean_rev = arr_scale_rev = None
    if arr_layer_on:
        arr_mean_for, arr_scale_for, arr_mean_rev, arr_scale_rev = compute_arrhenius_scalers_from_train(
            train_loader, y_scaler, cfg["temperatures"]
        )

    model = model_factory_from_cfg(
        cfg,
        unscaler=unscaler,
        ea_scales_for=y_scaler.named_transformers_["t2"],  # Ea_for
        ea_scales_rev=y_scaler.named_transformers_["t5"],  # Ea_rev
        arr_mean_for=arr_mean_for, arr_scale_for=arr_scale_for,
        arr_mean_rev=arr_mean_rev, arr_scale_rev=arr_scale_rev,
        featurizer=featurizer,
        x_d_dim=x_d_dim,
    )

    es = EarlyStopping(monitor="val_loss", mode="min", patience=int(cfg.get("patience", 10)))
    callbacks = [es]
    checkpoint_cb: Optional[ModelCheckpoint] = None
    if save_checkpoints:
        ckpt_dir = checkpoint_dir or log_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = (checkpoint_prefix or "model").replace(os.sep, "_")
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{ckpt_name}-best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_cb)

    accelerator = cfg.get("accelerator", "auto")
    if isinstance(accelerator, str):
        accelerator = accelerator.lower()

    devices_cfg = cfg.get("devices", 1)
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

    precision_cfg = cfg.get("precision")
    if precision_cfg:
        precision = precision_cfg
    else:
        has_cuda = torch.cuda.is_available()
        if accelerator == "cpu" or (accelerator == "auto" and not has_cuda):
            precision = "32-true"
        else:
            precision = "16-mixed"

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=save_checkpoints,
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

    val_metrics: Dict[str, float] = {}
    for k, v in trainer.callback_metrics.items():
        try:
            fv = float(v.item()) if hasattr(v, "item") else float(v)
        except (TypeError, ValueError):
            continue
        ks = str(k)
        if ks.startswith("val/"):
            val_metrics[ks] = fv
        elif ks.startswith("val_"):
            val_metrics[f"val/{ks[4:]}"] = fv
    val_metrics.setdefault("val/loss", val_loss)

    metrics_to_log: Dict[str, float] = dict(val_metrics)
    test_loss: Optional[float] = None
    test_metrics: Optional[Dict[str, float]] = None

    best_checkpoint_path: Optional[str] = None
    if checkpoint_cb is not None:
        best_checkpoint_path = checkpoint_cb.best_model_path or None

    if test_loader is not None and test_list:
        ckpt_path = best_checkpoint_path if best_checkpoint_path else None
        test_out = trainer.test(model, test_loader, verbose=False, ckpt_path=ckpt_path)
        if test_out:
            raw_metrics = test_out[0]
            test_metrics = {str(k): float(v) for k, v in raw_metrics.items()}
            test_loss = test_metrics.get("test_loss")
            for k, v in test_metrics.items():
                name = k if k.startswith("test/") else f"test/{k}" if k != "test_loss" else "test/loss"
                metrics_to_log[name] = v

    if logger is not None:
        logger.log_fold_metrics(trial.number, fold_id, metrics_to_log)
        logger.log_scalers(trial.number, fold_id, "y", y_scaler)
        if loaders.get("xd_scaler"):
            logger.log_scalers(trial.number, fold_id, "X_global", loaders["xd_scaler"])
        if loaders.get("vf_scaler"):
            logger.log_scalers(trial.number, fold_id, "V_f", loaders["vf_scaler"])

    return TrainResult(
        val_loss=val_loss,
        val_metrics=val_metrics,
        test_loss=test_loss,
        test_metrics=test_metrics,
        checkpoint_path=best_checkpoint_path,
    )



def eval_final_cfg_on_kfold(
    cfg,
    bundle,
    outer_splits,
    logger,
    tag: str,
    trial_id: Optional[int] = None,
    split_sig: Optional[str] = None,
    save_checkpoints: bool = False,
    checkpoint_root: Optional[str] = None,
    record_predictions: bool = False,
    loader_cache: Optional[LoaderCache] = None,
) -> dict:
    """
    Evaluate the configuration on the provided outer splits (typically 10-fold CV).
    Optionally logs the run to the TrialLogger under `trial_id`.
    """
    vals: List[float] = []
    val_metric_rows: List[Dict[str, float]] = []
    test_metric_rows: List[Dict[str, float]] = []
    sig = split_sig or splits_signature(outer_splits)
    started = False

    checkpoint_records: Optional[List[Dict[str, Any]]] = [] if save_checkpoints else None

    try:
        if logger is not None and trial_id is not None:
            logger.start_trial(trial_id, cfg, sig)
            logger.log_seed(trial_id, None, {"base": cfg.get("seed", 42)})
            started = True

        for fold_id, (fold_idx, holdout_idx) in enumerate(outer_splits):
            pseudo = _PseudoTrial(f"{tag}-f{fold_id}", trial_id if trial_id is not None else fold_id)
            if logger is not None and trial_id is not None:
                base_seed = cfg.get("seed", 42)
                logger.log_seed(
                    trial_id,
                    fold_id,
                    {
                        "base": base_seed,
                        "loader_train": base_seed + 11,
                        "loader_val": base_seed + 13,
                    },
                )
                logger.log_split_indices(
                    trial_id,
                    fold_id,
                    outer_train=np.array(fold_idx, dtype=int),
                    outer_holdout=np.array(holdout_idx, dtype=int),
                )

            fold_idx_abs = list(map(int, fold_idx))
            holdout_idx_abs = list(map(int, holdout_idx))
            fold_pairs = (
                [bundle.donors_kept[i] for i in fold_idx_abs],
                [bundle.acceptors_kept[i] for i in fold_idx_abs],
            )
            fold_group_keys = [bundle.pair_group_keys[i] for i in fold_idx_abs]
            fold_seed = int(cfg.get("seed", 42)) + fold_id + 1

            tr_reps, va_reps, _ = _split_with_strategy(
                cfg=cfg,
                donors=fold_pairs[0],
                acceptors=fold_pairs[1],
                sizes=(0.9, 0.1, 0.0),
                seed=fold_seed,
                num_reps=1,
                pair_group_keys=fold_group_keys,
            )
            tr_abs = [fold_idx_abs[i] for i in map(int, tr_reps[0])]
            va_abs = [fold_idx_abs[i] for i in map(int, va_reps[0])]

            if logger is not None and trial_id is not None:
                logger.log_split_indices(
                    trial_id,
                    fold_id,
                    train=np.array(tr_abs, dtype=int),
                    val=np.array(va_abs, dtype=int),
                    test=np.array(holdout_idx_abs, dtype=int),
                )

            fold_checkpoint_root = None
            if save_checkpoints and checkpoint_root is not None:
                fold_checkpoint_root = os.path.join(checkpoint_root, f"fold{fold_id}")

            result = _train_one(
                train_idx=tr_abs,
                val_idx=va_abs,
                cfg=cfg,
                featurizer=bundle.featurizer,
                attached_pair_dps=bundle.attached_pair_dps,
                x_d_dim=bundle.x_d_dim,
                pair_group_keys=bundle.pair_group_keys,
                seed=fold_seed,
                log_dir=os.path.join("logs/hpo", f"{tag}-f{fold_id}"),
                logger=logger,
                fold_id=fold_id,
                trial=pseudo,
                test_idx=holdout_idx_abs,
                save_checkpoints=save_checkpoints,
                checkpoint_dir=fold_checkpoint_root,
                checkpoint_prefix=f"{tag}-fold{fold_id}",
                loader_cache=loader_cache,
            )
            vals.append(float(result.val_loss))
            val_metric_rows.append(dict(result.val_metrics or {"val/loss": float(result.val_loss)}))
            if result.test_metrics:
                test_metric_rows.append(dict(result.test_metrics))
            if (
                checkpoint_records is not None
                and save_checkpoints
                and result.checkpoint_path is not None
            ):
                checkpoint_records.append(
                    {
                        "fold_id": fold_id,
                        "replicate": 0,
                        "train_indices": tr_abs,
                        "val_indices": va_abs,
                        "test_indices": holdout_idx_abs,
                        "checkpoint_path": result.checkpoint_path,
                        "val_loss": result.val_loss,
                        "val_metrics": result.val_metrics,
                        "test_loss": result.test_loss,
                        "test_metrics": result.test_metrics,
                    }
                )

        if started:
            logger.aggregate_and_store(trial_id)
            if record_predictions and checkpoint_records:
                _log_predictions_from_checkpoints(checkpoint_records, bundle, cfg, logger, trial_id, loader_cache)
            logger.end_trial(trial_id, "complete")

    except Exception:
        if started:
            logger.end_trial(trial_id, "failed")
        raise

    def _aggregate(rows: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if not rows:
            return {}, {}
        keys = sorted(set().union(*[r.keys() for r in rows]))
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for k in keys:
            xs = [float(r[k]) for r in rows if k in r]
            if xs:
                means[k] = float(np.mean(xs))
                stds[k] = float(np.std(xs))
        return means, stds

    val_metric_mean, val_metric_std = _aggregate(val_metric_rows)
    test_metric_mean, test_metric_std = _aggregate(test_metric_rows)

    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "all": vals,
        "val_metric_mean": val_metric_mean,
        "val_metric_std": val_metric_std,
        "test_metric_mean": test_metric_mean,
        "test_metric_std": test_metric_std,
        "checkpoints": checkpoint_records or [],
    }


def train_final_cfg_on_holdout(
    cfg,
    bundle,
    logger,
    tag: str,
    trial_id: int,
    test_frac: float = 0.15,
    seed: Optional[int] = None,
    save_checkpoints: bool = False,
    checkpoint_root: Optional[str] = None,
    loader_cache: Optional[LoaderCache] = None,
    fixed_train_indices: Optional[Sequence[int]] = None,
    fixed_test_indices: Optional[Sequence[int]] = None,
) -> dict:
    """
    Train `cfg` on a deterministic train/validation split drawn from (1-test_frac)
    fraction of the data and evaluate on the remaining hold-out test set.
    """
    base_seed = seed if seed is not None else cfg.get("seed", 42)
    pl.seed_everything(base_seed, workers=True)

    donors = bundle.donors_kept
    acceptors = bundle.acceptors_kept
    group_keys = bundle.pair_group_keys

    if fixed_train_indices is not None or fixed_test_indices is not None:
        if fixed_train_indices is None or fixed_test_indices is None:
            raise ValueError("Both fixed_train_indices and fixed_test_indices must be provided together.")
        train_base_idx = list(map(int, fixed_train_indices))
        test_idx = list(map(int, fixed_test_indices))
    else:
        if not (0.0 < test_frac < 1.0):
            raise ValueError(f"test_frac must be in (0,1); got {test_frac}")
        train_frac = 1.0 - test_frac
        train_reps, _, test_reps = _split_with_strategy(
            cfg,
            donors,
            acceptors,
            sizes=(train_frac, 0.0, test_frac),
            seed=base_seed,
            num_reps=1,
            pair_group_keys=group_keys,
        )
        train_base_idx = list(map(int, train_reps[0]))
        test_idx = list(map(int, test_reps[0]))

    holdout_sig = sha1(
        "|".join(
            [
                ",".join(map(str, sorted(train_base_idx))),
                ",".join(map(str, sorted(test_idx))),
                f"seed={base_seed}",
                f"test_frac={test_frac}",
            ]
        ).encode("utf-8")
    ).hexdigest()

    started = False
    if logger is not None:
        logger.start_trial(trial_id, cfg, holdout_sig)
        logger.log_seed(trial_id, None, {"base": base_seed})
        logger.log_split_indices(
            trial_id,
            -1,
            trainval=np.array(train_base_idx, dtype=int),
            test=np.array(test_idx, dtype=int),
        )
        started = True

    train_pairs = ([donors[i] for i in train_base_idx], [acceptors[i] for i in train_base_idx])
    train_group_keys = [group_keys[i] for i in train_base_idx]

    inner_train_reps, inner_val_reps, _ = _split_with_strategy(
        cfg,
        train_pairs[0],
        train_pairs[1],
        sizes=(0.85, 0.15, 0.0),
        seed=base_seed,
        num_reps=int(cfg.get("num_reps", 1)),
        pair_group_keys=train_group_keys,
    )

    results: List[Dict[str, Any]] = []

    try:
        for rep_id, (tr_loc, val_loc) in enumerate(zip(inner_train_reps, inner_val_reps, strict=False)):
            train_abs_idx = [train_base_idx[i] for i in tr_loc]
            val_abs_idx = [train_base_idx[i] for i in val_loc]
            sub_seed = base_seed + (rep_id + 1) * 1000

            if logger is not None:
                logger.log_seed(
                    trial_id,
                    rep_id,
                    {
                        "base": sub_seed,
                        "loader_train": sub_seed + 11,
                        "loader_val": sub_seed + 13,
                        "loader_test": sub_seed + 17,
                    },
                )
                logger.log_split_indices(
                    trial_id,
                    rep_id,
                    train=np.array(train_abs_idx, dtype=int),
                    val=np.array(val_abs_idx, dtype=int),
                    test=np.array(test_idx, dtype=int),
                )

            run_dir = os.path.join("logs/hpo", f"{tag}-holdout-r{rep_id+1}")
            checkpoint_dir = None
            if save_checkpoints and checkpoint_root is not None:
                checkpoint_dir = os.path.join(checkpoint_root, f"rep{rep_id+1}")

            result = _train_one(
                train_abs_idx,
                val_abs_idx,
                cfg,
                bundle.featurizer,
                bundle.attached_pair_dps,
                bundle.x_d_dim,
                bundle.pair_group_keys,
                seed=sub_seed,
                log_dir=run_dir,
                logger=logger,
                fold_id=rep_id,
                trial=_PseudoTrial(tag, trial_id),
                test_idx=test_idx,
                save_checkpoints=save_checkpoints,
                checkpoint_dir=checkpoint_dir,
                checkpoint_prefix=f"{tag}-rep{rep_id+1}",
                loader_cache=loader_cache,
            )

            results.append(
                {
                    "replicate": rep_id,
                    "val_loss": result.val_loss,
                    "test_loss": result.test_loss,
                    "test_metrics": result.test_metrics,
                    "checkpoint_path": result.checkpoint_path,
                    "train_indices": train_abs_idx,
                    "val_indices": val_abs_idx,
                }
            )

        if started:
            logger.aggregate_and_store(trial_id)
            logger.end_trial(trial_id, "complete")

    except Exception:
        if started:
            logger.end_trial(trial_id, "failed")
        raise

    val_losses = [r["val_loss"] for r in results]
    test_losses = [r["test_loss"] for r in results if r["test_loss"] is not None]

    summary = {
        "train_indices": train_base_idx,
        "test_indices": test_idx,
        "replicates": results,
        "val_mean": float(np.mean(val_losses)) if val_losses else None,
        "val_std": float(np.std(val_losses)) if val_losses else None,
        "test_mean": float(np.mean(test_losses)) if test_losses else None,
        "test_std": float(np.std(test_losses)) if test_losses else None,
        "holdout_signature": holdout_sig,
        "seed": base_seed,
        "checkpoints": [r for r in results if r.get("checkpoint_path")],
    }

    return summary


def export_final_split_details(
    bundle,
    cfg: Dict[str, Any],
    final_summary: Dict[str, Any],
    combo_tag: str,
    export_root: str = "logs/hpo",
    reset_existing: bool = False,
    loader_cache: Optional[LoaderCache] = None,
) -> Optional[Dict[str, str]]:
    replicates = final_summary.get("replicates") or []
    if not replicates:
        return None

    export_dir = Path(export_root) / "split_exports" / combo_tag
    export_dir.mkdir(parents=True, exist_ok=True)

    db_path = export_dir / f"splits_{combo_tag}.sqlite"
    csv_path = export_dir / f"splits_{combo_tag}.csv"

    if reset_existing:
        if db_path.exists():
            db_path.unlink()
        if csv_path.exists():
            csv_path.unlink()

    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS split_entries (
                combo_tag TEXT,
                replicate INTEGER,
                split TEXT,
                sample_index INTEGER,
                name TEXT,
                base_name TEXT,
                donor_smiles TEXT,
                acceptor_smiles TEXT,
                A_for_true REAL,
                n_for_true REAL,
                Ea_for_true REAL,
                A_rev_true REAL,
                n_rev_true REAL,
                Ea_rev_true REAL,
                A_for_pred REAL,
                n_for_pred REAL,
                Ea_for_pred REAL,
                A_rev_pred REAL,
                n_rev_pred REAL,
                Ea_rev_pred REAL,
                lnk_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scalers (
                combo_tag TEXT,
                replicate INTEGER,
                payload TEXT
            )
            """
        )

    split_rows: List[Dict[str, Any]] = []

    with conn:
        for replicate in replicates:
            checkpoint_path = replicate.get("checkpoint_path")
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                continue

            rep_idx = int(replicate.get("replicate", len(split_rows)))
            train_idx = list(map(int, replicate.get("train_indices", [])))
            val_idx = list(map(int, replicate.get("val_indices", [])))
            test_idx = list(map(int, final_summary.get("test_indices", [])))

            base_seed = final_summary.get("seed", cfg.get("seed", 42))
            sub_seed = base_seed + (rep_idx + 1) * 1000

            if loader_cache is not None:
                loaders = loader_cache.get_loaders(cfg, train_idx, val_idx, test_idx, sub_seed)
            else:
                loaders = make_loaders(
                    bundle=bundle,
                    cfg=cfg,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx if test_idx else None,
                    seed=sub_seed,
                )

            y_scaler = loaders.get("y_scaler")
            vf_scaler = loaders.get("vf_scaler")
            xd_scaler = loaders.get("xd_scaler")
            r_min = loaders.get("r_min")
            r_max = loaders.get("r_max")

            train_dataset = loaders.get("train_dataset")
            val_dataset = loaders.get("val_dataset")
            test_dataset = loaders.get("test_dataset")

            batch_size = int(cfg.get("batch_size", 128))
            train_eval_loader = _build_eval_loader(train_dataset, min(batch_size, len(train_idx) or 1), sub_seed + 101)
            val_eval_loader = _build_eval_loader(val_dataset, min(batch_size, len(val_idx) or 1), sub_seed + 103)
            test_eval_loader = _build_eval_loader(test_dataset, min(batch_size, len(test_idx) or 1), sub_seed + 105)

            model = ArrheniusMultiComponentMPNN.load_from_checkpoint(checkpoint_path, map_location="cpu")
            model.eval()
            trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cpu", devices=1)

            prediction_map: Dict[str, Dict[str, Any]] = {}
            for split_name, loader, indices in (
                ("train", train_eval_loader, train_idx),
                ("val", val_eval_loader, val_idx),
                ("test", test_eval_loader, test_idx),
            ):
                if loader is None or not indices:
                    continue
                outputs = trainer.predict(model, dataloaders=loader)
                prediction_map[split_name] = _stack_prediction_outputs(outputs)

            scaler_payload = {
                "y": _serialize_column_transformer(y_scaler),
                "V_f": _serialize_standard_scaler(vf_scaler),
                "X_d": _serialize_standard_scaler(xd_scaler),
                "r_bounds": [r_min, r_max],
            }
            conn.execute(
                "INSERT INTO scalers (combo_tag, replicate, payload) VALUES (?,?,?)",
                (combo_tag, rep_idx, json.dumps(scaler_payload)),
            )

            for split_name, indices in (
                ("train", train_idx),
                ("val", val_idx),
                ("test", test_idx),
            ):
                if not indices:
                    continue
                preds = prediction_map.get(split_name, {})
                y_pred_raw = preds.get("y_pred_raw")
                y_true_raw = preds.get("y_true_raw")
                lnk_pred = preds.get("lnk_pred")
                lnk_true = preds.get("lnk_true")
                temps = preds.get("temps")

                for pos, idx in enumerate(indices):
                    dp_f = bundle.attached_pair_dps[0][idx]
                    dp_r = bundle.attached_pair_dps[1][idx]
                    name = dp_f.name
                    base_name = name.replace("_r1h", "").replace("_r2h", "")
                    donor_smiles = _to_smiles(dp_f.mol)
                    acceptor_smiles = _to_smiles(dp_r.mol)
                    (A_for_true, n_for_true, Ea_for_true) = _arrhenius_from_dp(dp_f)
                    (A_rev_true, n_rev_true, Ea_rev_true) = _arrhenius_from_dp(dp_r)

                    pred_vals = (
                        y_pred_raw[pos] if y_pred_raw is not None and pos < len(y_pred_raw) else None
                    )
                    if pred_vals is not None:
                        A_for_pred, n_for_pred, Ea_for_pred, A_rev_pred, n_rev_pred, Ea_rev_pred = [
                            float(x) for x in pred_vals
                        ]
                    else:
                        A_for_pred = n_for_pred = Ea_for_pred = None
                        A_rev_pred = n_rev_pred = Ea_rev_pred = None

                    lnk_entry = None
                    if lnk_pred is not None and lnk_true is not None and pos < len(lnk_pred):
                        pred_forward = lnk_pred[pos, :, 0].tolist() if lnk_pred.shape[1] > 0 else []
                        pred_reverse = lnk_pred[pos, :, 1].tolist() if lnk_pred.shape[1] > 0 else []
                        true_forward = lnk_true[pos, :, 0].tolist() if lnk_true.shape[1] > 0 else []
                        true_reverse = lnk_true[pos, :, 1].tolist() if lnk_true.shape[1] > 0 else []
                        lnk_entry = json.dumps(
                            {
                                "temps": temps,
                                "pred_forward": pred_forward,
                                "pred_reverse": pred_reverse,
                                "true_forward": true_forward,
                                "true_reverse": true_reverse,
                            }
                        )

                    row = {
                        "combo_tag": combo_tag,
                        "replicate": rep_idx,
                        "split": split_name,
                        "sample_index": idx,
                        "name": name,
                        "base_name": base_name,
                        "donor_smiles": donor_smiles,
                        "acceptor_smiles": acceptor_smiles,
                        "A_for_true": A_for_true,
                        "n_for_true": n_for_true,
                        "Ea_for_true": Ea_for_true,
                        "A_rev_true": A_rev_true,
                        "n_rev_true": n_rev_true,
                        "Ea_rev_true": Ea_rev_true,
                        "A_for_pred": A_for_pred,
                        "n_for_pred": n_for_pred,
                        "Ea_for_pred": Ea_for_pred,
                        "A_rev_pred": A_rev_pred,
                        "n_rev_pred": n_rev_pred,
                        "Ea_rev_pred": Ea_rev_pred,
                        "lnk_json": lnk_entry,
                    }
                    split_rows.append(row)
                    conn.execute(
                        """
                        INSERT INTO split_entries (
                            combo_tag, replicate, split, sample_index, name, base_name,
                            donor_smiles, acceptor_smiles,
                            A_for_true, n_for_true, Ea_for_true,
                            A_rev_true, n_rev_true, Ea_rev_true,
                            A_for_pred, n_for_pred, Ea_for_pred,
                            A_rev_pred, n_rev_pred, Ea_rev_pred,
                            lnk_json
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            combo_tag,
                            rep_idx,
                            split_name,
                            idx,
                            name,
                            base_name,
                            donor_smiles,
                            acceptor_smiles,
                            A_for_true,
                            n_for_true,
                            Ea_for_true,
                            A_rev_true,
                            n_rev_true,
                            Ea_rev_true,
                            A_for_pred,
                            n_for_pred,
                            Ea_for_pred,
                            A_rev_pred,
                            n_rev_pred,
                            Ea_rev_pred,
                            lnk_entry,
                        ),
                    )

    if split_rows:
        df = pd.DataFrame(split_rows)
        df.sort_values(by=["replicate", "split", "sample_index"], inplace=True)
        if csv_path.exists():
            df.to_csv(csv_path, mode="a", index=False, header=False)
        else:
            df.to_csv(csv_path, index=False)

    return {"csv": str(csv_path), "db": str(db_path)}


def _split_with_strategy(
    cfg: Dict[str, Any],
    donors,
    acceptors,
    sizes: Tuple[float, float, float],
    seed: int,
    num_reps: int,
    pair_group_keys,
):
    splitter = str(cfg.get("splitter", "kstone")).lower()
    data = (donors, acceptors)
    if splitter == "random":
        return random_grouped_split_indices(
            data,
            sizes=sizes,
            seed=seed,
            num_replicates=num_reps,
            unordered_pairs=True,
            pair_key_mode="inchikey",
            pair_group_keys=pair_group_keys,
        )
    return ks_make_split_indices(
        data,
        sizes=sizes,
        seed=seed,
        num_replicates=num_reps,
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
        pair_group_keys=pair_group_keys,
    )
