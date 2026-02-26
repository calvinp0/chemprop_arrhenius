# run_hpo/space.py
import yaml
from typing import Any, Dict
from optuna.trial import Trial

def load_search_space(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def suggest_from_yaml(trial: Trial, spec, name):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)  # supports None/bool/str/num
    if isinstance(spec, dict) and {"low","high"} <= spec.keys():
        if spec.get("int"):
            return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
    return spec  # fixed scalar or None
