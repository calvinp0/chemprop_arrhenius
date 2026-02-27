"""Loss helpers for Arrhenius multi-target training."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


class ArrheniusLossMixin:
    """Loss computation for scaled triplets and optional Arrhenius supervision."""

    def _loss_triplet(
        self, Yp_s: Tensor, Yt_s: Tensor, lnKp: Tensor, lnKt: Tensor, w: Tensor
    ) -> Tensor:
        A10_p, n_p, Ea_p = Yp_s.unbind(1)
        A10_t, n_t, Ea_t = Yt_s.unbind(1)
        _ = w  # task-level weights are currently not used per-sample
        w = torch.ones_like(A10_p)

        loss_A10 = F.smooth_l1_loss(A10_p, A10_t, beta=self.huber_delta, reduction="none")
        loss_n = F.smooth_l1_loss(n_p, n_t, beta=self.huber_delta, reduction="none")
        loss_ea = F.smooth_l1_loss(Ea_p, Ea_t, beta=self.huber_delta, reduction="none")

        use_lnk = (
            self.enable_arrhenius_layer
            and self.use_arrhenius_supervision
            and self.w_lnK > 0.0
            and lnKp.numel() > 0
        )
        per_sample = self.w_A10 * loss_A10 + self.w_n * loss_n + self.w_ea * loss_ea

        if use_lnk:
            loss_lnK = F.mse_loss(lnKp, lnKt, reduction="none").mean(1)
            per_sample = per_sample + self.w_lnK * loss_lnK

        return per_sample.mean()
