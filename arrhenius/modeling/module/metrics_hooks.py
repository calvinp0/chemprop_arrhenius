"""Lightning step hooks for train/val/test/predict metrics and outputs."""

from __future__ import annotations

import torch
from torch import Tensor

from chemprop.data import MulticomponentTrainingBatch
from arrhenius.modeling.module.model_core import PredictBatchOutput


class ArrheniusMetricsHooksMixin:
    """Train/validation/test/predict hooks for the Arrhenius model."""

    def training_step(self, batch: MulticomponentTrainingBatch, batch_idx: int) -> Tensor:
        batch_size = self.get_batch_size(batch)
        bmgs, V_ds, X_d, targets, weights, *_ = batch

        if self.nan_debug:

            def _chk(name, t):
                if t is None:
                    return
                if isinstance(t, (list, tuple)):
                    for i, item in enumerate(t):
                        _chk(f"{name}[{i}]", item)
                    return
                if not torch.isfinite(t).all():
                    raise RuntimeError(f"nan_debug: non-finite in {name}")

            _chk("V_ds", V_ds)
            _chk("X_d", X_d)
            _chk("targets", targets)
            _chk("weights", weights)
            for i, bmg in enumerate(bmgs):
                _chk(f"bmgs[{i}].V", bmg.V)
                _chk(f"bmgs[{i}].E", bmg.E)

        Zf, Zr = self.fingerprints_two_orders(bmgs, V_ds, X_d)
        Yf_pred_s = self.head_for(Zf)
        Yr_pred_s = self.head_rev(Zr)
        Yf_true_s, Yr_true_s = targets[:, :3], targets[:, 3:6]

        Pf_pred = self.convert_raw_triplet(Yf_pred_s, 0)
        Pr_pred = self.convert_raw_triplet(Yr_pred_s, 1)
        Pf_true = self.convert_raw_triplet(Yf_true_s, 0)
        Pr_true = self.convert_raw_triplet(Yr_true_s, 1)

        if self.enable_arrhenius_layer:
            sampled_indices = (
                self.sample_stratified_indices(
                    num_bins=self.strat_temp_num_bins,
                    samples_per_bin=self.strat_temp_samples_per_bin,
                )
                if self.stratify_temp
                else None
            )
            lnKf_p = self.arrhenius_layer_fwd(Pf_pred, sampled_indices=sampled_indices)
            lnKf_t = self.arrhenius_layer_fwd(Pf_true, sampled_indices=sampled_indices)
            lnKr_p = self.arrhenius_layer_rev(Pr_pred, sampled_indices=sampled_indices)
            lnKr_t = self.arrhenius_layer_rev(Pr_true, sampled_indices=sampled_indices)
        else:
            device = Zf.device
            lnKf_p = lnKf_t = lnKr_p = lnKr_t = torch.empty(0, device=device)

        loss_f = self._loss_triplet(Yf_pred_s, Yf_true_s, lnKf_p, lnKf_t, weights)
        loss_r = self._loss_triplet(Yr_pred_s, Yr_true_s, lnKr_p, lnKr_t, weights)
        loss = 0.5 * (loss_f + loss_r)

        if self.nan_debug:
            for name, t in (
                ("Zf", Zf),
                ("Zr", Zr),
                ("Yf_pred_s", Yf_pred_s),
                ("Yr_pred_s", Yr_pred_s),
                ("Yf_true_s", Yf_true_s),
                ("Yr_true_s", Yr_true_s),
                ("Pf_pred", Pf_pred),
                ("Pr_pred", Pr_pred),
                ("Pf_true", Pf_true),
                ("Pr_true", Pr_true),
                ("lnKf_p", lnKf_p),
                ("lnKf_t", lnKf_t),
                ("lnKr_p", lnKr_p),
                ("lnKr_t", lnKr_t),
            ):
                if t is not None and t.numel() > 0 and not torch.isfinite(t).all():
                    raise RuntimeError(f"nan_debug: non-finite in {name}")
            if not torch.isfinite(loss):
                raise RuntimeError("nan_debug: non-finite loss")

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True
        )
        return loss

    def validation_step(self, batch: MulticomponentTrainingBatch, batch_idx: int) -> Tensor:
        bmgs, V_ds, X_d, targets, weights, *_ = batch
        Zf, Zr = self.fingerprints_two_orders(bmgs, V_ds, X_d)
        Yf_pred_s = self.head_for(Zf)
        Yr_pred_s = self.head_rev(Zr)
        Yf_true_s, Yr_true_s = targets[:, :3], targets[:, 3:6]

        Pf_pred = self.convert_raw_triplet(Yf_pred_s, 0)
        Pr_pred = self.convert_raw_triplet(Yr_pred_s, 1)
        Pf_true = self.convert_raw_triplet(Yf_true_s, 0)
        Pr_true = self.convert_raw_triplet(Yr_true_s, 1)

        if self.enable_arrhenius_layer:
            lnKf_std_p = self.arrhenius_layer_fwd(Pf_pred)
            lnKf_std_t = self.arrhenius_layer_fwd(Pf_true)
            lnKr_std_p = self.arrhenius_layer_rev(Pr_pred)
            lnKr_std_t = self.arrhenius_layer_rev(Pr_true)
            lnk_dict_std = {"for": (lnKf_std_p, lnKf_std_t), "rev": (lnKr_std_p, lnKr_std_t)}

            lnKf_raw_p = self.arrhenius_layer_fwd_raw(Pf_pred)
            lnKf_raw_t = self.arrhenius_layer_fwd_raw(Pf_true)
            lnKr_raw_p = self.arrhenius_layer_rev_raw(Pr_pred)
            lnKr_raw_t = self.arrhenius_layer_rev_raw(Pr_true)
            lnk_dict_raw = {"for": (lnKf_raw_p, lnKf_raw_t), "rev": (lnKr_raw_p, lnKr_raw_t)}
        else:
            device = Zf.device
            empty = {
                "for": (torch.empty(0, device=device), torch.empty(0, device=device)),
                "rev": (torch.empty(0, device=device), torch.empty(0, device=device)),
            }
            lnk_dict_std = lnk_dict_raw = empty

        val_loss = 0.5 * (
            self._loss_triplet(
                Yf_pred_s, Yf_true_s, lnk_dict_std["for"][0], lnk_dict_std["for"][1], weights
            )
            + self._loss_triplet(
                Yr_pred_s, Yr_true_s, lnk_dict_std["rev"][0], lnk_dict_std["rev"][1], weights
            )
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.get_batch_size(batch),
            prog_bar=True,
        )

        self.metrics.update_batch(
            split="val",
            scaled={"for": (Yf_pred_s, Yf_true_s), "rev": (Yr_pred_s, Yr_true_s)},
            lnk=lnk_dict_std,
        )
        self.metrics_raw.update_batch(
            split="val",
            scaled={"for": (Pf_pred, Pf_true), "rev": (Pr_pred, Pr_true)},
            lnk=lnk_dict_raw,
        )
        return val_loss

    def on_validation_epoch_end(self) -> None:
        self.metrics.log_and_reset(self, split="val")
        self.metrics_raw.log_and_reset(self, split="val")

    def test_step(self, batch: MulticomponentTrainingBatch, batch_idx: int) -> Tensor:
        bmgs, V_ds, X_d, targets, weights, *_ = batch
        Zf, Zr = self.fingerprints_two_orders(bmgs, V_ds, X_d)
        Yf_pred_s = self.head_for(Zf)
        Yr_pred_s = self.head_rev(Zr)
        Yf_true_s, Yr_true_s = targets[:, :3], targets[:, 3:6]

        Pf_pred = self.convert_raw_triplet(Yf_pred_s, 0)
        Pr_pred = self.convert_raw_triplet(Yr_pred_s, 1)
        Pf_true = self.convert_raw_triplet(Yf_true_s, 0)
        Pr_true = self.convert_raw_triplet(Yr_true_s, 1)

        if self.enable_arrhenius_layer:
            lnKf_std_p = self.arrhenius_layer_fwd(Pf_pred)
            lnKf_std_t = self.arrhenius_layer_fwd(Pf_true)
            lnKr_std_p = self.arrhenius_layer_rev(Pr_pred)
            lnKr_std_t = self.arrhenius_layer_rev(Pr_true)
            lnk_dict_std = {"for": (lnKf_std_p, lnKf_std_t), "rev": (lnKr_std_p, lnKr_std_t)}

            lnKf_raw_p = self.arrhenius_layer_fwd_raw(Pf_pred)
            lnKf_raw_t = self.arrhenius_layer_fwd_raw(Pf_true)
            lnKr_raw_p = self.arrhenius_layer_rev_raw(Pr_pred)
            lnKr_raw_t = self.arrhenius_layer_rev_raw(Pr_true)
            lnk_dict_raw = {"for": (lnKf_raw_p, lnKf_raw_t), "rev": (lnKr_raw_p, lnKr_raw_t)}
        else:
            device = Zf.device
            empty = {
                "for": (torch.empty(0, device=device), torch.empty(0, device=device)),
                "rev": (torch.empty(0, device=device), torch.empty(0, device=device)),
            }
            lnk_dict_std = lnk_dict_raw = empty

        test_loss = 0.5 * (
            self._loss_triplet(
                Yf_pred_s, Yf_true_s, lnk_dict_std["for"][0], lnk_dict_std["for"][1], weights
            )
            + self._loss_triplet(
                Yr_pred_s, Yr_true_s, lnk_dict_std["rev"][0], lnk_dict_std["rev"][1], weights
            )
        )
        self.log(
            "test_loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.get_batch_size(batch),
            prog_bar=True,
        )

        self.metrics.update_batch(
            split="test",
            scaled={"for": (Yf_pred_s, Yf_true_s), "rev": (Yr_pred_s, Yr_true_s)},
            lnk=lnk_dict_std,
        )
        self.metrics_raw.update_batch(
            split="test",
            scaled={"for": (Pf_pred, Pf_true), "rev": (Pr_pred, Pr_true)},
            lnk=lnk_dict_raw,
        )
        return test_loss

    def on_test_epoch_end(self) -> None:
        self.metrics.log_and_reset(self, split="test")
        self.metrics_raw.log_and_reset(self, split="test")

    def predict_step(
        self, batch: MulticomponentTrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> PredictBatchOutput:
        bmgs, V_ds, X_d, targets, *rest = batch
        _ = rest
        _ = dataloader_idx
        Zf, Zr = self.fingerprints_two_orders(bmgs, V_ds, X_d)

        Yf_s = self.head_for(Zf)
        Yr_s = self.head_rev(Zr)
        Yf_true_s, Yr_true_s = targets[:, :3], targets[:, 3:6]

        Pf_pred = self.convert_raw_triplet(Yf_s, 0)
        Pr_pred = self.convert_raw_triplet(Yr_s, 1)
        Pf_true = self.convert_raw_triplet(Yf_true_s, 0)
        Pr_true = self.convert_raw_triplet(Yr_true_s, 1)

        if self.enable_arrhenius_layer:
            lnKf_p = self.arrhenius_layer_fwd(Pf_pred)
            lnKr_p = self.arrhenius_layer_rev(Pr_pred)
            lnKf_t = self.arrhenius_layer_fwd(Pf_true)
            lnKr_t = self.arrhenius_layer_rev(Pr_true)
            temps = self.arrhenius_layer_fwd.T
        else:
            device = Zf.device
            lnKf_p = lnKr_p = lnKf_t = lnKr_t = torch.empty(0, device=device)
            temps = None

        y_pred_raw = torch.cat([Pf_pred, Pr_pred], dim=1)
        y_true_raw = torch.cat([Pf_true, Pr_true], dim=1)
        y_pred_s = torch.cat([Yf_s, Yr_s], dim=1)
        y_true_s = torch.cat([Yf_true_s, Yr_true_s], dim=1)

        if lnKf_p.numel() > 0:
            lnk_pred = torch.stack([lnKf_p, lnKr_p], dim=2)
            lnk_true = torch.stack([lnKf_t, lnKr_t], dim=2)
        else:
            B = y_pred_raw.size(0)
            lnk_pred = torch.empty(B, 0, 2, device=y_pred_raw.device)
            lnk_true = torch.empty(B, 0, 2, device=y_pred_raw.device)

        return PredictBatchOutput(
            y_pred_raw=y_pred_raw,
            y_true_raw=y_true_raw,
            y_pred_s=y_pred_s,
            y_true_s=y_true_s,
            lnk_pred=lnk_pred,
            lnk_true=lnk_true,
            temps=temps,
        )
