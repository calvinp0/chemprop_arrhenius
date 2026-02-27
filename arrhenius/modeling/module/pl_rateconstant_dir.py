"""Public Arrhenius multi-component Lightning model.

This module keeps the public class import path stable:
`arrhenius.modeling.module.pl_rateconstant_dir.ArrheniusMultiComponentMPNN`.

The implementation is split into smaller modules for readability:
- `model_core.py`: feature construction and scaled/raw transforms
- `losses.py`: loss terms
- `metrics_hooks.py`: train/val/test/predict hooks
- `checkpoint_io.py`: optimizer + checkpoint/scaler persistence

Minimal usage:
```python
from arrhenius.modeling.module.pl_rateconstant_dir import ArrheniusMultiComponentMPNN

model = ArrheniusMultiComponentMPNN(
    message_passing=mp,
    agg=agg,
    temps=[300.0, 500.0, 1000.0],
    unscaler=unscaler,
    ea_scales=[pt_forward, pt_reverse],
    arrhenius_layer_mean_for=lnk_mu_f,
    arrhenius_layer_scale_for=lnk_sigma_f,
    arrhenius_layer_mean_rev=lnk_mu_r,
    arrhenius_layer_scale_rev=lnk_sigma_r,
)
```
"""

from __future__ import annotations

import logging
from typing import Any

from lightning import pytorch as pl
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from arrhenius.modeling.metrics.metrics import MetricRegistry, Tgt
from arrhenius.modeling.module.checkpoint_io import ArrheniusCheckpointIOMixin
from arrhenius.modeling.module.losses import ArrheniusLossMixin
from arrhenius.modeling.module.metrics_hooks import ArrheniusMetricsHooksMixin
from arrhenius.modeling.module.model_core import ArrheniusModelCoreMixin
from arrhenius.modeling.nn.layers import ArrheniusLayer
from arrhenius.modeling.nn.predictor import ArrheniusHeadPredictor
from chemprop.nn.agg import Aggregation
from chemprop.nn.message_passing import MulticomponentMessagePassing
from chemprop.nn.transforms import ScaleTransform

logger = logging.getLogger(__name__)


class ArrheniusMultiComponentMPNN(
    ArrheniusMetricsHooksMixin,
    ArrheniusCheckpointIOMixin,
    ArrheniusLossMixin,
    ArrheniusModelCoreMixin,
    pl.LightningModule,
):
    """Arrhenius-aware multi-component MPNN for forward/reverse parameter prediction."""

    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        temps: list[float],
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        head_activation: str = "relu",
        order_mode: str = "aware",
        learned_pool_hidden: int = 256,
        learned_pool_layers: int = 2,
        learned_pool_dropout: float = 0.0,
        batch_norm: bool = False,
        norm_type: str = "batchnorm",
        unscaler: Any = None,
        ea_scales: Any = None,
        A_log_10scaled: bool = True,
        w_ea: float = 1.0,
        w_n: float = 1.0,
        w_A10: float = 1.0,
        w_lnK: float = 1.0,
        huber_delta: float = 1.0,
        X_d_transform: ScaleTransform | None = None,
        d_x: int = 0,
        metrics: MetricRegistry | None = None,
        init_lr: float = 1e-4,
        final_lr: float = 1e-4,
        max_lr: float = 1e-3,
        warmup_epochs: int = 2,
        arrhenius_layer_mean_for: list[float] | None = None,
        arrhenius_layer_scale_for: list[float] | None = None,
        arrhenius_layer_mean_rev: list[float] | None = None,
        arrhenius_layer_scale_rev: list[float] | None = None,
        use_kJ: bool = True,
        strat_temp_num_bins: int | None = None,
        strat_temp_samples_per_bin: int | None = None,
        enable_arrhenius_layer: bool = True,
        use_arrhenius_supervision: bool = True,
        nan_debug: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "message_passing",
                "agg",
                "X_d_transform",
                "ea_scales",
                "unscaler",
                "metrics",
                "metrics_raw",
            ]
        )

        self.enable_arrhenius_layer = bool(enable_arrhenius_layer)
        self.use_arrhenius_supervision = bool(use_arrhenius_supervision)
        self._y_scaler = None
        self._vf_scaler_shared = None
        self._xd_scaler_shared = None
        self._arrhenius_scalers = {
            "mean_for": arrhenius_layer_mean_for,
            "scale_for": arrhenius_layer_scale_for,
            "mean_rev": arrhenius_layer_mean_rev,
            "scale_rev": arrhenius_layer_scale_rev,
        }

        has_scalers_for = (arrhenius_layer_mean_for is not None) and (
            arrhenius_layer_scale_for is not None
        )
        has_scalers_rev = (arrhenius_layer_mean_rev is not None) and (
            arrhenius_layer_scale_rev is not None
        )
        has_scalers = has_scalers_for and has_scalers_rev
        if self.enable_arrhenius_layer and not has_scalers:
            raise ValueError(
                "Arrhenius layer enabled but mean/scale tensors were not provided (both directions)."
            )
        if not self.enable_arrhenius_layer and has_scalers:
            logger.warning(
                "Arrhenius scalers were provided but the layer is disabled. They will be ignored."
            )
        if arrhenius_layer_mean_for is not None and len(arrhenius_layer_mean_for) != len(temps):
            raise ValueError("arrhenius_layer_mean_for must match temps length.")
        if arrhenius_layer_scale_for is not None and len(arrhenius_layer_scale_for) != len(temps):
            raise ValueError("arrhenius_layer_scale_for must match temps length.")

        self.order_mode = order_mode
        self.ea_scales = ea_scales
        self.unscaler = unscaler
        self.A_log_10scaled = A_log_10scaled
        self.huber_delta = float(huber_delta)
        self.nan_debug = bool(nan_debug)
        self.X_d_transform = (
            torch.nn.Identity() if (X_d_transform is None and d_x > 0) else X_d_transform
        )

        self.message_passing = message_passing
        self.agg = agg

        if isinstance(ea_scales, (list, tuple)):
            if len(ea_scales) != 2:
                raise ValueError("ea_scales must be a PowerTransformer or a 2-tuple (fwd, rev).")
            pt_f, pt_r = ea_scales
        else:
            pt_f = pt_r = ea_scales

        def _pt_params(pt):
            mu = float(pt._scaler.mean_[0])
            sigma = float(pt._scaler.scale_[0])
            lam = float(pt.lambdas_[0])
            return mu, sigma, lam

        self._ea_params = [_pt_params(pt_f), _pt_params(pt_r)]
        self.mp_output_dim = self.message_passing.output_dim

        if self.order_mode == "learned":
            layers = [
                nn.Linear(4 * self.mp_output_dim, learned_pool_hidden),
                nn.ReLU(),
                nn.Dropout(learned_pool_dropout),
            ]
            for _ in range(learned_pool_layers - 1):
                layers += [
                    nn.Linear(learned_pool_hidden, learned_pool_hidden),
                    nn.ReLU(),
                    nn.Dropout(learned_pool_dropout),
                ]
            layers.append(nn.Linear(learned_pool_hidden, self.mp_output_dim))
            self.learned_pool = nn.Sequential(*layers)
        else:
            self.learned_pool = None

        d_feat = self.mp_output_dim + int(d_x)
        self.head_for = ArrheniusHeadPredictor(
            n_tasks=1,
            input_dim=d_feat,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
        )
        self.head_rev = ArrheniusHeadPredictor(
            n_tasks=1,
            input_dim=d_feat,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
        )

        def _build_norm(kind: str, dim: int):
            kind = (kind or "").lower()
            if kind == "layernorm":
                return nn.LayerNorm(dim, elementwise_affine=True)
            return nn.BatchNorm1d(dim)

        if batch_norm:
            self.bn_fwd = _build_norm(norm_type, d_feat)
            self.bn_rev = _build_norm(norm_type, d_feat)
        else:
            self.bn_fwd = nn.Identity()
            self.bn_rev = nn.Identity()

        self.strat_temp_num_bins = strat_temp_num_bins
        self.strat_temp_samples_per_bin = strat_temp_samples_per_bin
        self.stratify_temp = bool(
            self.enable_arrhenius_layer and strat_temp_num_bins and strat_temp_samples_per_bin
        )

        if self.enable_arrhenius_layer:
            self.arrhenius_layer_fwd = ArrheniusLayer(
                temps=temps,
                use_kJ=use_kJ,
                lnk_mean=torch.tensor(arrhenius_layer_mean_for, dtype=torch.float32),
                lnk_scale=torch.tensor(arrhenius_layer_scale_for, dtype=torch.float32),
            )
            self.arrhenius_layer_rev = ArrheniusLayer(
                temps=temps,
                use_kJ=use_kJ,
                lnk_mean=torch.tensor(arrhenius_layer_mean_rev, dtype=torch.float32),
                lnk_scale=torch.tensor(arrhenius_layer_scale_rev, dtype=torch.float32),
            )
            self.arrhenius_layer_fwd_raw = ArrheniusLayer(temps=temps, use_kJ=use_kJ)
            self.arrhenius_layer_rev_raw = ArrheniusLayer(temps=temps, use_kJ=use_kJ)
        else:
            self.arrhenius_layer_fwd = None
            self.arrhenius_layer_rev = None
            self.arrhenius_layer_fwd_raw = None
            self.arrhenius_layer_rev_raw = None
            self.stratify_temp = False

        self.w_A10 = float(w_A10)
        self.w_n = float(w_n)
        self.w_ea = float(w_ea)
        self.w_lnK = float(w_lnK)
        if not self.use_arrhenius_supervision:
            self.w_lnK = 0.0

        self.metrics = (
            MetricRegistry(
                targets_cols=[(Tgt.A10, 0), (Tgt.N, 1), (Tgt.EAY, 2)],
                metrics_builders={"mse": MeanSquaredError, "mae": MeanAbsoluteError, "r2": R2Score},
                include_lnk=self.enable_arrhenius_layer,
            )
            if metrics is None
            else metrics
        )
        self.metrics_raw = MetricRegistry(
            targets_cols=[(Tgt.A, 0), (Tgt.N, 1), (Tgt.EA, 2)],
            metrics_builders={"mae": MeanAbsoluteError, "mse": MeanSquaredError, "r2": R2Score},
            include_lnk=self.enable_arrhenius_layer,
            namespace="raw",
        )

        self.init_lr = init_lr
        self.final_lr = final_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
