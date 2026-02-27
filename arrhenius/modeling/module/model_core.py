"""Core model math and tensor transformations for Arrhenius MPNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictBatchOutput:
    """Typed prediction payload returned by ``predict_step`` for batch outputs."""

    y_pred_raw: Tensor
    y_true_raw: Tensor
    y_pred_s: Tensor
    y_true_s: Tensor
    lnk_pred: Tensor
    lnk_true: Tensor
    temps: Tensor | None

    @property
    def Y_f_raw(self) -> Tensor:
        """Forward raw Arrhenius params [A, n, Ea] used by legacy consumers."""
        return self.y_pred_raw[:, :3]

    def to_numpy_dict(self) -> dict[str, Any]:
        """Convert to a CPU/NumPy mapping for legacy downstream code."""
        temps_np = None if self.temps is None else self.temps.detach().cpu().numpy()
        return {
            "y_pred_raw": self.y_pred_raw.detach().cpu().numpy(),
            "y_true_raw": self.y_true_raw.detach().cpu().numpy(),
            "y_pred_s": self.y_pred_s.detach().cpu().numpy(),
            "y_true_s": self.y_true_s.detach().cpu().numpy(),
            "lnk_pred": self.lnk_pred.detach().cpu().numpy(),
            "lnk_true": self.lnk_true.detach().cpu().numpy(),
            "temps": temps_np,
            "Y_f_raw": self.Y_f_raw.detach().cpu().numpy(),
        }


class ArrheniusModelCoreMixin:
    """Feature construction, pooling, and scaled<->raw parameter transforms."""

    def _masked_mean(self, H_v: Tensor, batch: Tensor, mask_1d: Tensor | None) -> Tensor:
        if mask_1d is None:
            B = int(batch.max()) + 1
            out = torch.zeros(B, H_v.size(1), device=H_v.device, dtype=H_v.dtype)
            out.index_add_(0, batch, H_v)
            cnt = torch.bincount(batch, minlength=B).clamp_min(1).to(H_v.dtype).unsqueeze(-1)
            return out / cnt

        w = mask_1d.view(-1).to(H_v.dtype)
        B = int(batch.max()) + 1
        sum_w = torch.zeros(B, device=H_v.device, dtype=H_v.dtype)
        sum_w.index_add_(0, batch, w)
        sum_h = torch.zeros(B, H_v.size(1), device=H_v.device, dtype=H_v.dtype)
        sum_h.index_add_(0, batch, H_v * w.unsqueeze(-1))
        return sum_h / sum_w.clamp_min(1).unsqueeze(-1)

    def fingerprints_two_orders(
        self, bmgs, V_ds, X_d: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        H_vs = self.message_passing(bmgs, V_ds)

        Hs = []
        for H_v, bmg, V_d in zip(H_vs, bmgs, V_ds):
            mask = None
            if V_d is not None and V_d.numel() > 0:
                mask = V_d[:, 0]
            Hs.append(self._masked_mean(H_v, bmg.batch, mask))

        H1 = torch.cat(Hs, dim=-1)
        H2 = torch.cat(Hs[::-1], dim=-1)

        if self.order_mode == "aware":
            H_fwd, H_rev = H1, H2
        elif self.order_mode == "invariant":
            H_sym = 0.5 * (H1 + H2)
            H_fwd = H_rev = H_sym
        elif self.order_mode == "bi":
            s = 0.5 * (H1 + H2)
            d = 0.5 * (H1 - H2)
            H_fwd = torch.cat([s, d], dim=1)
            H_rev = torch.cat([s, -d], dim=1)
        elif self.order_mode == "antisym":
            d = 0.5 * (H1 - H2)
            H_fwd = d
            H_rev = -d
        elif self.order_mode == "learned":
            diff = torch.abs(H1 - H2)
            prod = H1 * H2
            H_fwd = self.learned_pool(torch.cat([H1, H2, diff, prod], dim=1))
            H_rev = self.learned_pool(torch.cat([H2, H1, diff, prod], dim=1))
        else:
            raise ValueError(f"Unknown order_mode: {self.order_mode}")

        if X_d is not None and self.X_d_transform is not None:
            X = self.X_d_transform(X_d)
            H_fwd = torch.cat([H_fwd, X], dim=1)
            H_rev = torch.cat([H_rev, X], dim=1)

        H_fwd = self.bn_fwd(H_fwd)
        H_rev = self.bn_rev(H_rev)
        return H_fwd, H_rev

    def _safe_pow(self, base: Tensor, exp: float) -> Tensor:
        return torch.sign(base) * torch.pow(base.abs().clamp_min(1e-12), exp)

    def _inverse_yeojohnson(self, y: Tensor, lam: float) -> Tensor:
        pos = y >= 0
        if lam == 0.0:
            return torch.where(pos, torch.exp(y) - 1.0, -torch.exp(-y) + 1.0)
        inv_pos = torch.pow(y * lam + 1.0, 1.0 / lam) - 1.0
        inv_neg = -torch.pow(-y * (2.0 - lam) + 1.0, 1.0 / (2.0 - lam)) + 1.0
        if not torch.isfinite(inv_pos).all() or not torch.isfinite(inv_neg).all():
            logger.warning(
                "Inverse transform produced non-finite values; check target scaling parameters."
            )
        return torch.where(pos, inv_pos, inv_neg)

    def convert_raw_triplet(self, z3: Tensor, dir_idx: int) -> Tensor:
        mean, std = self.unscaler.mean.squeeze(0), self.unscaler.scale.squeeze(0)
        off = 3 * dir_idx

        log10A_z, n_z, Ea_YJ_z = z3.unbind(1)
        log10A = log10A_z * std[off + 0] + mean[off + 0]
        n = n_z * std[off + 1] + mean[off + 1]

        mu, sigma, lam = self._ea_params[dir_idx]
        Ea_YJ = Ea_YJ_z * sigma + mu

        pos = Ea_YJ >= 0
        if lam == 0.0:
            inv_pos = torch.exp(Ea_YJ) - 1.0
        else:
            inv_pos = self._safe_pow(Ea_YJ * lam + 1.0, 1.0 / lam) - 1.0
        if lam == 2.0:
            inv_neg = 1.0 - torch.exp(-Ea_YJ)
        else:
            inv_neg = 1.0 - self._safe_pow(-(2.0 - lam) * Ea_YJ + 1.0, 1.0 / (2.0 - lam))
        Ea = torch.where(pos, inv_pos, inv_neg)

        log10A = log10A.clamp(-20.0, 20.0)
        A = torch.pow(10.0, log10A)
        return torch.stack((A, n, Ea), dim=1)
