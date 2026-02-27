# run_hpo/defaults.py
from copy import deepcopy
from typing import Any, Dict, Tuple

Key = Tuple[str, ...]


def _deep_get(d: Dict[str, Any], path: Key, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _deep_set(d: Dict[str, Any], path: Key, value):
    cur = d
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


def config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _cfg = deepcopy(cfg)

    # Dataset
    _cfg.setdefault(
        "extra_mode", "baseline"
    )  # baseline|geom_only|local|atom|rad|rad_local|rad_local_noc
    _cfg.setdefault("global_mode", "none")  # none|morgan_binary|morgan_count|rdkit2d_norm
    _cfg.setdefault("temp_range", [300, 3100])
    _cfg.setdefault("temp_interval", 100)

    # Splitter
    _cfg.setdefault("num_reps", _deep_get(_cfg, ("splits", "inner_reps"), 5))
    _cfg.setdefault("distance_metric", "jaccard")
    _cfg.setdefault("joint_mode", "order-invariant")
    _cfg.setdefault("donor_weight", 0.5)
    _cfg.setdefault("p_norm", 2.0)
    _cfg.setdefault("n_bits", 2048)
    _cfg.setdefault("radius", 2)
    _cfg.setdefault("splitter", "kstone")

    # Trainer
    _cfg.setdefault("trainer", {})
    _cfg["trainer"].setdefault("batch_size", _deep_get(_cfg, ("trainer", "batch_size"), 128))
    _cfg["trainer"].setdefault("patience", _deep_get(_cfg, ("trainer", "patience"), 20))
    _cfg["trainer"].setdefault("max_epochs", _deep_get(_cfg, ("trainer", "max_epochs"), 200))
    ### Flatten
    _cfg["batch_size"] = _cfg["trainer"].pop("batch_size")
    _cfg["patience"] = _cfg["trainer"].pop("patience")
    _cfg["max_epochs"] = _cfg["trainer"].pop("max_epochs")

    # Model: MPNN
    _cfg.setdefault("mp_hidden", 300)
    _cfg.setdefault("mp_depth", 3)
    _cfg.setdefault("mp_dropout", 0.0)
    _cfg.setdefault("mp_shared", True)

    # Encoder
    _cfg.setdefault("order_mode", "aware")  # "aware"|"invariant"|"antisym"|"bi"|"learned"
    _cfg.setdefault("learned_pool_hidden", 50)
    _cfg.setdefault("learned_pool_layers", 2)
    _cfg.setdefault("learned_pool_dropout", 0.0)

    # Aggregation
    _cfg.setdefault("agg", "mean")

    # FFN
    _cfg.setdefault("head_hidden_dim", 300)
    _cfg.setdefault("head_dropout", 0.0)
    _cfg.setdefault("head_activation", "relu")
    _cfg.setdefault("ff_batchnorm", False)

    # Arrhenius controls
    _cfg.setdefault("enable_arrhenius_layer", True)
    _cfg.setdefault("use_arrhenius_supervision", True)

    # LR schedule (Noam-like)
    _cfg.setdefault("init_lr", 1e-3)
    _cfg.setdefault("final_lr", 1e-4)
    _cfg.setdefault("warmup_epochs", 10)
    _cfg.setdefault("max_lr", 1e-2)

    # Loss weights
    _cfg.setdefault("w_ea", 1.0)
    _cfg.setdefault("w_n", 1.0)
    _cfg.setdefault("w_A10", 1.0)
    _cfg.setdefault("w_lnK", 1.0)

    return _cfg
