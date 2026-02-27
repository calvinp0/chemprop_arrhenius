# Shared config assembly helpers for HPO and replay.
from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Tuple

from arrhenius.training.hpo.defaults import config_defaults
from arrhenius.training.hpo.modes import resolve_modes
from arrhenius.training.hpo.space import load_search_space
from arrhenius.training.hpo.temps import build_temps
from arrhenius.training.hpo.feature_modes import canonicalize_extra_mode, canonicalize_global_mode


def compose_base_config(
    args, base_cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any] | None]:
    """
    Build the non-HPO scaffolding once:
      defaults -> CLI modes (extras/global) -> Arrhenius toggles.

    Returns (cfg, pruned_space, chosen_family) where pruned_space is the YAML
    search space with global-feat family narrowed to the requested family.
    """
    space = load_search_space(getattr(args, "search_space", "search_space.yaml"))
    cfg0 = config_defaults(deepcopy(base_cfg))
    cfg1, pruned_space, chosen_family = resolve_modes(cfg0, args, space)
    cfg1["enable_arrhenius_layer"] = getattr(args, "arr_layer", "on") != "off"
    cfg1["use_arrhenius_supervision"] = getattr(args, "arr_supervision", "on") != "off"
    return cfg1, pruned_space, chosen_family


def finalize_cfg(
    cfg: Dict[str, Any], args, yaml_space: Dict[str, Any], include_temps: bool = True
) -> Dict[str, Any]:
    """
    Normalize hardware/split/precision flags and optionally inject temperatures.
    This is used both during HPO and when replaying a stored cfg.
    """
    cfg = deepcopy(cfg)
    if include_temps:
        cfg["temperatures"] = build_temps(cfg, args, yaml_space=yaml_space, include_max=False)
    cfg["seed"] = args.seed if hasattr(args, "seed") else cfg.get("seed", 42)
    cfg["accelerator"] = getattr(args, "accelerator", cfg.get("accelerator", "auto"))
    cfg["devices"] = getattr(args, "devices", cfg.get("devices", 1))
    cfg["splitter"] = getattr(args, "splitter", cfg.get("splitter", "kstone"))
    cfg["enable_arrhenius_layer"] = cfg.get(
        "enable_arrhenius_layer", getattr(args, "enable_arrhenius_layer", True)
    )
    cfg["use_arrhenius_supervision"] = cfg.get(
        "use_arrhenius_supervision", getattr(args, "use_arrhenius_supervision", True)
    )
    cfg["extra_mode"] = canonicalize_extra_mode(
        cfg.get("extra_mode", getattr(args, "extra_mode", "baseline"))
    )
    cfg["global_mode"] = canonicalize_global_mode(
        cfg.get("global_mode", getattr(args, "global_mode", "none"))
    )

    precision_override = getattr(args, "precision", None)
    if precision_override is not None:
        cfg["precision"] = precision_override
    elif "precision" in cfg and cfg["precision"] is None:
        cfg.pop("precision")

    return cfg
