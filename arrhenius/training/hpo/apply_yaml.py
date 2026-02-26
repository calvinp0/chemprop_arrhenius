# run_hpo/apply_yaml.py
from copy import deepcopy
from typing import Any, Dict
from optuna.trial import Trial
from .space import suggest_from_yaml

def apply_yaml_space(trial: Trial, cfg: Dict[str, Any], space: Dict[str, Any], chosen_family) -> Dict[str, Any]:
    c = deepcopy(cfg)

    # 1) flat keys (skip 'global_feats')
    for key, spec in space.items():
        if key in {"global_feats", "rank_metric_default", "rank_metric"} or str(key).startswith("_"):
            continue
        c[key] = suggest_from_yaml(trial, spec, key)

    # 2) global features
    gf = space.get("global_feats")
    if isinstance(gf, dict) and gf:
        if chosen_family == "auto":
            fam = trial.suggest_categorical("global_feats.family", list(gf))
            c["global_mode"] = fam
            fam_spec = gf[fam] or {}
        else:
            fam = chosen_family
            fam_spec = gf.get(fam, {}) or {}

        for subk, subspec in fam_spec.items():
            c[f"global_{subk}"] = suggest_from_yaml(trial, subspec, f"{fam}.{subk}")

    return c
