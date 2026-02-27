from chemprop import nn
from chemprop.CUSTOM.featuriser.featurise import MOL_TYPES
from arrhenius.modeling.module.pl_rateconstant_dir import ArrheniusMultiComponentMPNN
from arrhenius.training.hpo.feature_modes import canonicalize_extra_mode, mode_settings


def _build_message_passing(cfg, featurizer):
    depth = int(cfg.get("mp_depth", 4))
    d_h = int(cfg.get("mp_hidden", 256))
    dropout = float(cfg.get("mp_dropout", 0.05))
    shared = bool(cfg.get("mp_shared", True))
    d_v = featurizer.atom_fdim
    d_e = featurizer.bond_fdim
    has_vd = bool(
        mode_settings(canonicalize_extra_mode(cfg.get("extra_mode", "baseline")))["use_extras"]
    )
    d_vd = 1 if has_vd else 0
    blocks = [
        nn.BondMessagePassing(depth=depth, dropout=dropout, d_v=d_v, d_e=d_e, d_h=d_h, d_vd=d_vd)
        for _ in range(len(MOL_TYPES))
    ]
    return nn.MulticomponentMessagePassing(
        blocks=blocks, n_components=len(MOL_TYPES), shared=shared
    )


def _agg_from_name(name: str):
    name = name.lower()
    if name == "mean":
        return nn.MeanAggregation()
    if name == "sum":
        return nn.SumAggregation()
    raise ValueError(f"Unknown agg {name}")


def model_factory_from_cfg(
    cfg,
    unscaler,
    ea_scales_for,
    ea_scales_rev,
    arr_mean_for,
    arr_scale_for,
    arr_mean_rev,
    arr_scale_rev,
    featurizer,
    x_d_dim: int = 0,
):
    mcmp = _build_message_passing(cfg, featurizer)
    agg = _agg_from_name(cfg.get("agg", "mean"))
    return ArrheniusMultiComponentMPNN(
        message_passing=mcmp,
        agg=agg,
        temps=cfg.get("temperatures"),
        head_hidden_dim=int(cfg.get("head_hidden_dim")),
        head_dropout=float(cfg.get("head_dropout")),
        head_activation=(cfg.get("head_activation")),
        order_mode=str(cfg.get("order_mode")),
        learned_pool_hidden=int(cfg.get("learned_pool_hidden")),
        learned_pool_layers=int(cfg.get("learned_pool_layers")),
        learned_pool_dropout=float(cfg.get("learned_pool_dropout")),
        unscaler=unscaler,
        ea_scales=[ea_scales_for, ea_scales_rev],
        A_log_10scaled=True,
        w_ea=float(cfg.get("w_ea", 1.0)),
        w_n=float(cfg.get("w_n", 1.0)),
        w_A10=float(cfg.get("w_A10", 1.0)),
        w_lnK=float(cfg.get("w_lnK", 1.0)),
        X_d_transform=None,
        init_lr=float(cfg.get("init_lr", 1e-5)),
        final_lr=float(cfg.get("final_lr", 1e-4)),
        max_lr=float(cfg.get("max_lr", 1e-3)),
        warmup_epochs=int(cfg.get("warmup_epochs", 2)),
        arrhenius_layer_mean_for=arr_mean_for,
        arrhenius_layer_scale_for=arr_scale_for,
        arrhenius_layer_mean_rev=arr_mean_rev,
        arrhenius_layer_scale_rev=arr_scale_rev,
        d_x=x_d_dim,
        enable_arrhenius_layer=bool(cfg.get("enable_arrhenius_layer", True)),
        use_arrhenius_supervision=bool(cfg.get("use_arrhenius_supervision", True)),
    )
