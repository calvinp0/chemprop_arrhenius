# run_hpo/run.py
import json
import os
from pathlib import Path
import random
import sys

from filelock import FileLock
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
import torch

from arrhenius.training.hpo.cli import build_parser
from arrhenius.training.hpo.configurator import finalize_cfg
from arrhenius.training.hpo.data import prepare_data
from arrhenius.training.hpo.database_configs import TrialLogger
from arrhenius.training.hpo.defaults import config_defaults
from arrhenius.training.hpo.evaluation import (
    eval_final_cfg_on_kfold,
    evaluate_trial_on_fold,
    export_final_split_details,
    train_final_cfg_on_holdout,
)
from arrhenius.training.hpo.feature_modes import (
    canonicalize_extra_mode,
    canonicalize_global_mode,
    canonicalize_rad_source,
    mode_settings,
)
from arrhenius.training.hpo.hpo_objective import objective_factory
from arrhenius.training.hpo.loader_cache import LoaderCache
from arrhenius.training.hpo.space import load_search_space
from arrhenius.training.hpo.splits import (
    build_locked_holdout_split,
    build_outer_splits,
    splits_signature,
)
from arrhenius.training.hpo.validate_data import main as validate_data_main


def set_all_seeds(seed: int, deterministic: bool = True):
    # Python + hash
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Torch CPU/GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Make CUDA math reproducible (required by PyTorch docs when deterministic)
        # Pick one of the allowed configs:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # older torch


def seed_worker(worker_id: int):
    """Deterministic seeding for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _sqlite_path_from_url(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path or ""
    # Absolute path form: sqlite:////abs/path.db -> parsed.path == //abs/path.db
    if path.startswith("//"):
        normalized = "/" + path.lstrip("/")
        return str(Path(normalized))
    # Relative form (sqlite:///foo.db or sqlite:///sub/dir.db): treat as relative to CWD
    rel = path.lstrip("/")
    if not rel:
        raise ValueError(f"Invalid SQLite URL: {url}")
    return str(Path.cwd() / rel)


def _sanitize_tag(*parts: str) -> str:
    safe = "-".join(parts)
    safe = safe.replace(os.sep, "_")
    safe = safe.replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe


def _tagged_db_path(path: str, tag: str) -> str:
    if not path:
        return path
    p = Path(path)
    if p.suffix:
        new_name = f"{p.stem}_{tag}{p.suffix}"
    else:
        new_name = f"{p.name}_{tag}.sqlite"
    return str(p.with_name(new_name))


def _resolve_rank_metric(args) -> str:
    cli_rank = getattr(args, "rank_metric", None)
    if cli_rank:
        return str(cli_rank)
    try:
        space = load_search_space(getattr(args, "search_space", "search_space.yaml"))
    except Exception:
        space = {}
    if isinstance(space, dict):
        value = space.get("rank_metric_default", space.get("rank_metric"))
        if isinstance(value, str) and value:
            return value
    return "val/mae_lnk_avg"


def _rank_score(cv_metrics: dict, rank_metric: str) -> tuple[str, float]:
    val_mean = cv_metrics.get("val_metric_mean") or {}
    for key in (rank_metric, "val/mae_lnk_avg", "val/loss", "val/best_loss"):
        if key in val_mean:
            return key, float(val_mean[key])
    if val_mean:
        first = sorted(val_mean.keys())[0]
        return first, float(val_mean[first])
    return "cv_mean_fallback", float(cv_metrics.get("mean", float("inf")))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "validate-data":
        raise SystemExit(validate_data_main(sys.argv[2:]))

    args = build_parser().parse_args()
    # normalize devices upfront so downstream configs remain reproducible JSON
    devices_arg = getattr(args, "devices", "1")
    if isinstance(devices_arg, str):
        dv = devices_arg.strip()
        if dv.lower() == "auto":
            devices_arg = "auto"
        else:
            try:
                devices_arg = int(dv)
            except ValueError:
                devices_arg = dv  # let Lightning parse custom strings if provided
    args.devices = devices_arg

    args.enable_arrhenius_layer = getattr(args, "arr_layer", "on") != "off"
    args.use_arrhenius_supervision = getattr(args, "arr_supervision", "on") != "off"

    set_all_seeds(args.seed)
    # normalize your global mode spelling once (underscores everywhere)
    if hasattr(args, "global_mode"):
        args.global_mode = canonicalize_global_mode(args.global_mode)

    raw_extra_mode = getattr(args, "extra_mode", "baseline")
    if raw_extra_mode == "path":
        args.rad_source = "path"
    elif raw_extra_mode == "default":
        args.rad_source = "rad"
    args.extra_mode = canonicalize_extra_mode(raw_extra_mode)
    args.rad_source = canonicalize_rad_source(getattr(args, "rad_source", "path"))

    # === 1) Prepare data ONCE ===
    # Plug in your real paths (or add them as CLI args)
    sdf_path = args.sdf_path
    target_csv = args.target_csv
    rad_dir = getattr(args, "rad_dir", None)

    mode_cfg = mode_settings(args.extra_mode)
    bundle = prepare_data(
        sdf_path=sdf_path,
        target_csv=target_csv,  # already log10(A) etc.
        extras_mode=args.extra_mode,
        global_mode=args.global_mode,
        morgan_bits=getattr(args, "morgan_bits", 2048),
        morgan_radius=getattr(args, "morgan_radius", 2),
        rad_dir=rad_dir if mode_cfg["use_extras"] else None,
        rad_source=getattr(args, "rad_source", "path"),
    )

    # === 2) Optional locked test split + outer splits for HPO on remaining pool ===
    defaults_cfg = config_defaults({})
    locked_test_frac = float(getattr(args, "locked_test_frac", 0.0))
    train_pool_idx: list[int]
    locked_test_idx: list[int]
    if locked_test_frac > 0.0:
        train_pool_idx, locked_test_idx = build_locked_holdout_split(
            donors=bundle.donors_kept,
            acceptors=bundle.acceptors_kept,
            pair_group_keys=bundle.pair_group_keys,
            splitter=args.splitter,
            test_frac=locked_test_frac,
            seed=args.seed,
            distance_metric=defaults_cfg["distance_metric"],
            joint_mode=defaults_cfg["joint_mode"],
            donor_weight=defaults_cfg["donor_weight"],
            p_norm=defaults_cfg["p_norm"],
            n_bits=defaults_cfg["n_bits"],
            radius=defaults_cfg["radius"],
        )
        print(
            f"[SPLIT] Locked test enabled: {len(locked_test_idx)} samples held out; "
            f"{len(train_pool_idx)} samples used for HPO/CV."
        )
    else:
        train_pool_idx = list(range(len(bundle.pair_group_keys)))
        locked_test_idx = []

    pool_donors = [bundle.donors_kept[i] for i in train_pool_idx]
    pool_acceptors = [bundle.acceptors_kept[i] for i in train_pool_idx]
    pool_group_keys = [bundle.pair_group_keys[i] for i in train_pool_idx]

    outer_splits_local = build_outer_splits(
        donors=pool_donors,
        acceptors=pool_acceptors,
        pair_group_keys=pool_group_keys,
        splitter=args.splitter,
        k_folds=int(getattr(args, "outer_folds", 3)),
        seed=args.seed,
        distance_metric=defaults_cfg["distance_metric"],
        joint_mode=defaults_cfg["joint_mode"],
        donor_weight=defaults_cfg["donor_weight"],
        p_norm=defaults_cfg["p_norm"],
        n_bits=defaults_cfg["n_bits"],
        radius=defaults_cfg["radius"],
    )
    outer_splits = [
        ([train_pool_idx[i] for i in tr], [train_pool_idx[i] for i in hold])
        for tr, hold in outer_splits_local
    ]
    splits_sig = splits_signature(outer_splits)

    # === 3) Optuna study ===
    study_name = getattr(args, "study_name", "arrhenius_hpo")
    storage_url = getattr(args, "storage_url", "sqlite:///arrhenius_hpo.db")

    metrics_db_path = os.path.abspath(getattr(args, "metrics_db", "metrics.sqlite"))
    tag = _sanitize_tag(
        f"extra-{args.extra_mode}",
        f"radsrc-{getattr(args, 'rad_source', 'path')}",
        f"global-{args.global_mode}",
        f"split-{args.splitter}",
    )
    metrics_db_path = os.path.abspath(_tagged_db_path(metrics_db_path, tag))
    metrics_dir = os.path.dirname(metrics_db_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    hpo_logger = TrialLogger(
        metrics_db_path, study_name, save_raw_scalers=bool(getattr(args, "save_raw_scalers", False))
    )

    final_db_base = getattr(args, "final_metrics_db", "final_metrics.sqlite")
    final_db_path = os.path.abspath(_tagged_db_path(final_db_base, tag))
    final_dir = os.path.dirname(final_db_path)
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)
    final_study_name = getattr(args, "final_study_name", f"{study_name}_final")
    final_logger = TrialLogger(
        final_db_path,
        final_study_name,
        save_raw_scalers=bool(getattr(args, "save_raw_scalers", False)),
    )
    loader_cache = LoaderCache(bundle)

    if "timeout=" not in storage_url:
        storage_url = f"{storage_url}?timeout=60"
    db_lock = None
    if storage_url.startswith("sqlite"):
        db_lock = _sqlite_path_from_url(storage_url) + ".init.lock"
    if db_lock:
        with FileLock(db_lock, timeout=600):
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
            )
    else:
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

    # === 4) Define train_eval_fn that reuses your inner loop ===
    def train_eval_fn(
        cfg: dict, outer, trial, logger, bundle=bundle, loader_cache=loader_cache
    ) -> float:
        logger.start_trial(trial.number, cfg, splits_sig)
        logger.log_seed(trial.number, None, {"base": cfg.get("seed", 42)})
        vals = []
        try:
            for f_id, (dev_idx, holdout_idx) in enumerate(outer):
                logger.log_seed(
                    trial.number,
                    f_id,
                    {
                        "base": cfg.get("seed", 42),
                        "loader_train": cfg.get("seed", 42) + 11,
                        "loader_val": cfg.get("seed", 42) + 13,
                    },
                )
                logger.log_split_indices(
                    trial.number,
                    f_id,
                    outer_train=np.array(dev_idx, dtype=int),
                    outer_holdout=np.array(holdout_idx, dtype=int),
                )

                v = evaluate_trial_on_fold(
                    fold_indices=np.asarray(dev_idx, dtype=int),
                    cfg=cfg,
                    pair_group_keys=bundle.pair_group_keys,
                    donors=bundle.donors_kept,
                    acceptors=bundle.acceptors_kept,
                    attached_pair_dps=bundle.attached_pair_dps,
                    featurizer=bundle.featurizer,
                    x_d_dim=bundle.x_d_dim,
                    trial_tag=f"t{trial.number}-f{f_id}",
                    seed=cfg.get("seed", 42),
                    trial=trial,
                    fold_id=f_id,
                    logger=logger,
                    save_checkpoints=False,
                    checkpoint_root=None,
                    loader_cache=loader_cache,
                )
                vals.append(float(v))
                trial.report(float(np.mean(vals)), step=f_id)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            logger.aggregate_and_store(trial.number)
            logger.end_trial(trial.number, "complete")
            return float(np.mean(vals))

        except optuna.TrialPruned:
            logger.end_trial(trial.number, "failed")
            raise

        except Exception:
            logger.end_trial(trial.number, "failed")
            raise

    # === 5) Build the objective and run ===
    objective = objective_factory(
        base_cfg={},  # keep empty, defaults fill in
        args=args,
        outer_splits=outer_splits,
        train_eval_fn=train_eval_fn,
        study=study,
        split_signature=splits_sig,
        logger=hpo_logger,
        bundle=bundle,
    )

    n_trials = int(getattr(args, "hpo_trials", 0))
    if getattr(args, "skip_hpo", False):
        print("[HPO] Skipping new trials (--skip-hpo).")
    elif n_trials > 0:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    else:
        print("[HPO] Skipping new trials; using existing results.")

    if getattr(args, "skip_final_evaluation", False):
        print("[POST] Skipping post-HPO selection (--skip-final-evaluation).")
        return

    complete_trials = [
        t
        for t in study.get_trials(deepcopy=True)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete_trials:
        print("[POST] No completed trials available; cannot run post-selection.")
        return

    complete_trials.sort(key=lambda t: t.value)
    same_split_trials = [t for t in complete_trials if t.user_attrs.get("splits_sig") == splits_sig]
    top_k = max(1, int(getattr(args, "top_k_final", 3)))
    pool_trials = same_split_trials
    if len(pool_trials) < top_k and bool(getattr(args, "relax_topk", False)):
        print(
            f"[POST] Only {len(pool_trials)} completed trials match current splits_sig; "
            "relaxing to all completed trials because --relax-topk is enabled."
        )
        pool_trials = complete_trials
    elif len(pool_trials) < top_k:
        print(
            f"[POST] Not enough split-matched trials for top-{top_k}: found {len(pool_trials)}. "
            "Run more HPO trials or use --relax-topk."
        )
        return

    unique_trials = []
    seen_hashes = set()
    for trial in pool_trials:
        cfg_hash = trial.user_attrs.get("cfg_hash")
        if cfg_hash is not None and cfg_hash in seen_hashes:
            continue
        if cfg_hash is not None:
            seen_hashes.add(cfg_hash)
        unique_trials.append(trial)
        if len(unique_trials) >= top_k:
            break

    if not unique_trials:
        print("[POST] Unable to identify unique trials for evaluation; exiting.")
        return

    if len(unique_trials) < top_k:
        print(f"[POST] Only {len(unique_trials)} unique trials found (requested top {top_k}).")

    n_total = len(train_pool_idx)
    n_unique_groups = len(set(pool_group_keys))
    n_splits = min(10, n_unique_groups)
    if n_splits < 2:
        print("[POST] Need at least 2 unique groups for post-HPO k-fold evaluation.")
        return
    gkf10 = GroupKFold(n_splits=n_splits)
    outer10 = [
        ([train_pool_idx[int(i)] for i in tr_idx], [train_pool_idx[int(i)] for i in te_idx])
        for tr_idx, te_idx in gkf10.split(np.arange(n_total), groups=pool_group_keys)
    ]
    if not outer10:
        print("[POST] Failed to create outer splits for 10-fold evaluation; aborting.")
        return

    rank_metric = _resolve_rank_metric(args)
    outer10_sig = splits_signature(outer10)
    print(
        f"[POST] Evaluating top {len(unique_trials)} configs on {len(outer10)} folds "
        f"and ranking by '{rank_metric}'."
    )

    post_results = []
    for rank, trial in enumerate(unique_trials, start=1):
        cfg_json = trial.user_attrs.get("cfg_json")
        if not cfg_json:
            print(f"[POST] Trial {trial.number} missing cfg_json; skipping.")
            continue
        cfg = json.loads(cfg_json)
        cfg = finalize_cfg(cfg, args, yaml_space={}, include_temps=False)
        cfg_tag = f"top{rank}_trial{trial.number}"
        trial_id = 1_000_000 + trial.number
        print(f"[POST] → Rank {rank}: trial {trial.number} (objective={trial.value:.4f})")

        checkpoint_root = os.path.join("logs/hpo", cfg_tag, "checkpoints")
        cv_metrics = eval_final_cfg_on_kfold(
            cfg=cfg,
            bundle=bundle,
            outer_splits=outer10,
            logger=final_logger,
            tag=cfg_tag,
            trial_id=trial_id,
            split_sig=outer10_sig,
            save_checkpoints=True,
            checkpoint_root=checkpoint_root,
            record_predictions=bool(getattr(args, "record_final_preds", True)),
            loader_cache=loader_cache,
        )
        rank_key_used, rank_value = _rank_score(cv_metrics, rank_metric)
        post_results.append(
            {
                "rank": rank,
                "trial_number": trial.number,
                "objective": float(trial.value),
                "cfg_hash": trial.user_attrs.get("cfg_hash"),
                "cv": cv_metrics,
                "rank_metric_requested": rank_metric,
                "rank_metric_used": rank_key_used,
                "rank_value": rank_value,
                "cfg": cfg,
                "trial_id": trial_id,
                "checkpoint_root": checkpoint_root,
            }
        )

    if not post_results:
        print("[POST] No configurations evaluated successfully; exiting.")
        return

    best_entry = min(post_results, key=lambda r: r["rank_value"])
    best_cfg = finalize_cfg(best_entry["cfg"], args, yaml_space={}, include_temps=False)
    best_cfg.setdefault("splitter", getattr(args, "splitter", "kstone"))
    print(
        "[POST] Selected trial "
        f"{best_entry['trial_number']} with {best_entry['rank_metric_used']}="
        f"{best_entry['rank_value']:.4f}"
    )

    final_trial_id = 2_000_000 + best_entry["trial_number"]
    final_tag = f"final_trial{best_entry['trial_number']}"
    final_checkpoint_root = os.path.join("logs/hpo", final_tag, "checkpoints")
    final_result = train_final_cfg_on_holdout(
        cfg=best_cfg,
        bundle=bundle,
        logger=final_logger,
        tag=final_tag,
        trial_id=final_trial_id,
        test_frac=float(getattr(args, "final_test_frac", 0.15)),
        seed=args.seed,
        save_checkpoints=True,
        checkpoint_root=final_checkpoint_root,
        loader_cache=loader_cache,
        fixed_train_indices=(train_pool_idx if locked_test_frac > 0.0 else None),
        fixed_test_indices=(locked_test_idx if locked_test_frac > 0.0 else None),
    )

    split_export_info = export_final_split_details(
        bundle=bundle,
        cfg=best_cfg,
        final_summary=final_result,
        combo_tag=final_tag,
        reset_existing=bool(getattr(args, "reset_split_exports", False)),
    )
    if split_export_info:
        print(
            f"[POST] Split details exported to {split_export_info['csv']} and {split_export_info['db']}"
        )

    if final_result.get("test_mean") is not None:
        print(
            "[POST] Hold-out test mean loss "
            f"{final_result['test_mean']:.4f} (std={final_result['test_std']:.4f})"
        )
    else:
        print("[POST] Hold-out evaluation completed (no test metrics reported).")

    os.makedirs("logs/hpo", exist_ok=True)
    summary_filename = f"{study_name}_{tag}_final_summary.json"
    summary_path = os.path.join("logs/hpo", summary_filename)
    summary_payload = {
        "locked_test": {
            "enabled": bool(locked_test_frac > 0.0),
            "test_frac": float(locked_test_frac),
            "train_pool_size": int(len(train_pool_idx)),
            "locked_test_size": int(len(locked_test_idx)),
            "locked_test_indices": locked_test_idx if locked_test_frac > 0.0 else [],
        },
        "top_results": [
            {
                "rank": r["rank"],
                "trial_number": r["trial_number"],
                "trial_id": r["trial_id"],
                "objective": r["objective"],
                "cfg_hash": r["cfg_hash"],
                "cv": r["cv"],
                "rank_metric_requested": r["rank_metric_requested"],
                "rank_metric_used": r["rank_metric_used"],
                "rank_value": r["rank_value"],
                "cfg": r["cfg"],
                "checkpoint_root": r.get("checkpoint_root"),
            }
            for r in post_results
        ],
        "selected_trial": {
            "trial_number": best_entry["trial_number"],
            "trial_id": best_entry["trial_id"],
            "cfg_hash": best_entry["cfg_hash"],
            "cv": best_entry["cv"],
            "rank_metric_requested": best_entry["rank_metric_requested"],
            "rank_metric_used": best_entry["rank_metric_used"],
            "rank_value": best_entry["rank_value"],
            "cfg": best_cfg,
            "checkpoint_root": best_entry.get("checkpoint_root"),
        },
        "final_trial": {
            "trial_id": final_trial_id,
            "result": final_result,
            "cfg": best_cfg,
            "checkpoint_root": final_checkpoint_root,
        },
        "split_exports": split_export_info,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, default=lambda o: int(o))
    print(f"[POST] Final summary written to {summary_path}")
    final_logger.store_summary("final_summary", summary_payload)
    print(f"[POST] Final-stage artefacts logged to {final_db_path} (trial_id {final_trial_id}).")


if __name__ == "__main__":
    main()
