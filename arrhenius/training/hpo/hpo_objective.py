# run_hpo
from hashlib import sha1
import json
from typing import Any, Dict

import optuna
from optuna.trial import Trial

from arrhenius.training.hpo.apply_yaml import apply_yaml_space
from arrhenius.training.hpo.configurator import compose_base_config, finalize_cfg
from arrhenius.training.hpo.space import load_search_space


def _config_signature(cfg: Dict[str, Any], splits_sig: str | None = None) -> str:
    # include split signature to avoid accidentally reusing scores across different folds
    items = []
    if splits_sig:
        items.append(f"__splits__={splits_sig}")
    for k, v in sorted(cfg.items()):
        if isinstance(v, float):
            v = round(v, 8)
        items.append(f"{k}={v}")
    return sha1("|".join(items).encode("utf-8")).hexdigest()


def _index_existing_trials(study: optuna.Study) -> dict[str, float]:
    table: dict[str, float] = {}
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        h = t.user_attrs.get("cfg_hash")
        if h is not None and t.value is not None:
            table[h] = min(t.value, table.get(h, float("inf")))
    return table


def objective_factory(
    base_cfg, args, outer_splits, train_eval_fn, study, split_signature, logger, bundle
):
    """
    - Assembles cfg for each trial from: defaults -> CLI modes -> YAML search space -> temps.
    - Skips exact-duplicate configs (same splits_sig) by reusing past scores from this study.
    - Calls user train_eval_fn(cfg, outer_splits, trial) which must return a scalar (lower=better).
    - Optional on_trial_complete callback lets you write to your SQLite 'cfgs' table or any sidecar.
    """

    space = load_search_space(args.search_space)

    # 1) fixed scaffolding (non-HPO): defaults + CLI modes
    cfg1, pruned_space, chosen_family = compose_base_config(args, base_cfg)

    # 2) previous completed configs by cfg_hash (including this split signature)
    seen = _index_existing_trials(study)

    def objective(trial: Trial) -> float:
        cfg2 = apply_yaml_space(trial, cfg1, pruned_space, chosen_family)
        cfg2 = finalize_cfg(cfg2, args, space, include_temps=True)
        print(f"HPO trial config:\n{cfg2}")
        cfg_hash = _config_signature(cfg2)
        print(f"Config hash: {cfg_hash}")

        # Set trial user attributes for logging
        trial.set_user_attr("cfg_hash", cfg_hash)
        trial.set_user_attr("splits_sig", split_signature)
        trial.set_user_attr("extras_mode", cfg2.get("extra_mode", "baseline"))
        trial.set_user_attr("global_mode", cfg2.get("global_mode", "none"))
        trial.set_user_attr("use_global_feats", int(cfg2.get("global_mode", "none") != "none"))
        trial.set_user_attr("cfg_json", json.dumps(cfg2, sort_keys=True))

        # Check if this config has already been evaluated
        if cfg_hash in seen:
            return float(seen[cfg_hash])

        value = float(train_eval_fn(cfg2, outer_splits, trial, logger=logger, bundle=bundle))

        return value

    return objective


if __name__ == "__main__":
    # Args for HPO
    pass
