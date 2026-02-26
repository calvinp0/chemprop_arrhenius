# run_hpo/modes.py
from copy import deepcopy
from run_hpo.feature_modes import canonicalize_extra_mode, canonicalize_global_mode

def resolve_modes(base_cfg, args, yaml_space):
    """
    CLI decides extra/global. YAML only tunes sub-knobs.
    global_mode: none | morgan_count | morgan_binary | rdkit2d_norm | auto
    """
    cfg = deepcopy(base_cfg)

    cfg["extra_mode"] = canonicalize_extra_mode(getattr(args, "extra_mode", "baseline"))
    wanted = canonicalize_global_mode(getattr(args, "global_mode", "none"))

    gspec = yaml_space.get("global_feats") if isinstance(yaml_space.get("global_feats"), dict) else {}

    if wanted == "none":
        cfg["global_mode"] = "none"
        pruned = {k:v for k,v in yaml_space.items() if k != "global_feats"}
        return cfg, pruned, None

    if wanted == "auto":
        if not gspec:
            raise ValueError("global_mode=auto but YAML has no 'global_feats'.")
        cfg["global_mode"] = "auto"
        return cfg, yaml_space, "auto"

    # concrete family
    if wanted not in gspec:
        raise ValueError(f"--global-mode {wanted} not found in YAML global_feats.")
    cfg["global_mode"] = wanted
    pruned = deepcopy(yaml_space)
    pruned["global_feats"] = {wanted: gspec[wanted]}
    return cfg, pruned, wanted
