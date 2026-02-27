"""Optimizer and checkpoint I/O helpers for Arrhenius models."""

from __future__ import annotations

import logging
import pickle
from typing import Any

import torch
from torch import Tensor, optim
from torch.nn.parameter import UninitializedParameter

from chemprop.data import MulticomponentTrainingBatch
from chemprop.schedulers import build_NoamLike_LRSched

logger = logging.getLogger(__name__)


class ArrheniusCheckpointIOMixin:
    """Optimization, scaler checkpoint persistence, and utility helpers."""

    def configure_optimizers(self) -> dict[str, object]:
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning("For infinite training, cooldown epochs are set to 100x warmup epochs.")
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )
        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        for p in self.parameters():
            if isinstance(p, UninitializedParameter):
                continue
            if not torch.isfinite(p).all():
                raise ValueError("Non-finite parameters detected in the model.")

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    def set_scalers(
        self, *, y_scaler: Any = None, vf_scaler_shared: Any = None, xd_scaler_shared: Any = None
    ) -> None:
        self._y_scaler = y_scaler
        self._vf_scaler_shared = vf_scaler_shared
        self._xd_scaler_shared = xd_scaler_shared

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self._y_scaler is not None:
            checkpoint["y_scaler"] = pickle.dumps(self._y_scaler)
        if self._vf_scaler_shared is not None:
            checkpoint["vf_scaler_shared"] = pickle.dumps(self._vf_scaler_shared)
        if self._xd_scaler_shared is not None:
            checkpoint["xd_scaler_shared"] = pickle.dumps(self._xd_scaler_shared)
        checkpoint["arrhenius_scalers"] = self._arrhenius_scalers

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "y_scaler" in checkpoint:
            self._y_scaler = pickle.loads(checkpoint["y_scaler"])
        if "vf_scaler_shared" in checkpoint:
            self._vf_scaler_shared = pickle.loads(checkpoint["vf_scaler_shared"])
        if "xd_scaler_shared" in checkpoint:
            self._xd_scaler_shared = pickle.loads(checkpoint["xd_scaler_shared"])
        if "arrhenius_scalers" in checkpoint:
            self._arrhenius_scalers = checkpoint["arrhenius_scalers"]

    def sample_stratified_indices(self, num_bins: int = 12, samples_per_bin: int = 1) -> Tensor:
        T = self.arrhenius_layer_fwd.T
        bin_edges = torch.linspace(
            T.min().to(T.device), T.max().to(T.device), steps=num_bins + 1
        ).to(T.device)
        sampled_indices = []

        for i in range(num_bins):
            in_bin = (T >= bin_edges[i]) & (T < bin_edges[i + 1])
            indices_in_bin = torch.where(in_bin)[0]
            if len(indices_in_bin) > 0:
                g = getattr(self, "_strat_rng", None)
                if g is None:
                    g = torch.Generator(device=T.device)
                    if hasattr(self, "manual_seed") and self.manual_seed is not None:
                        g.manual_seed(int(self.manual_seed))
                    self._strat_rng = g
                selected = indices_in_bin[
                    torch.randperm(len(indices_in_bin), generator=g)[:samples_per_bin]
                ]
                sampled_indices.extend(selected.tolist())

        return torch.tensor(sampled_indices, dtype=torch.long)

    def get_batch_size(self, batch: MulticomponentTrainingBatch) -> int:
        return len(batch[0][0])
