from typing import Iterable, NamedTuple
import logging


import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F

from lightning import pytorch as pl


from chemprop.data import BatchMolGraph, MulticomponentTrainingBatch
from chemprop.nn.message_passing import MulticomponentMessagePassing
from chemprop.nn.agg import Aggregation
from arrhenius.modeling.nn.predictor import ArrheniusHeadPredictor
from arrhenius.modeling.nn.layers import ArrheniusLayer
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.nn.transforms import ScaleTransform

from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score

logger = logging.getLogger(__name__)


BASE_METRICS = MetricCollection(
    {"mse": MeanSquaredError(), "mae": MeanAbsoluteError(), "r2": R2Score()}
)


class PredictOutput(NamedTuple):
    Y_scaled: Tensor
    Y_raw: Tensor
    lnK_pred: Tensor


class ArrheniusMultiComponentMPNN(pl.LightningModule):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        temps: list[float],
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        head_activation: str = "relu",
        order_awareness: bool = False,
        batch_norm: bool = False,
        unscaler=None,
        ea_scales: callable = None,
        A_log_10scaled: bool = True,
        w_ea=1.0,
        w_n=1.0,
        w_A10=1.0,
        w_lnK=1.0,
        X_d_transform: ScaleTransform | None = None,
        # Optimization parameters
        init_lr: float = 1e-4,
        final_lr: float = 1e-4,
        max_lr: float = 1e-3,
        warmup_epochs: int = 2,
        arrhenius_layer_mean: list[float] | None = None,
        arrhenius_layer_scale: list[float] | None = None,
        use_kJ: bool = True,
        strat_temp_num_bins: int | None = None,
        strat_temp_samples_per_bin: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["message_passing", "agg", "X_d_transform", "ea_scales", "unscaler"]
        )

        # If temps are provided and arrhenius_layer_mean and arrhenius_layer_scale are also,
        # ensure the lengths match
        if arrhenius_layer_mean is not None:
            assert len(arrhenius_layer_mean) == len(
                temps
            ), "arrhenius_layer_mean must match temps length"
        if arrhenius_layer_scale is not None:
            assert len(arrhenius_layer_scale) == len(
                temps
            ), "arrhenius_layer_scale must match temps length"

        # Order awareness is whether the model should consider the order of components
        self.order_awareness = order_awareness

        # Scales for Ea and n and A10 -  # Need to make sure this callable is on the device
        self.ea_scales = ea_scales
        self.unscaler = unscaler
        self.A_log_10scaled = A_log_10scaled
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        # Encoder & Aggregation
        self.message_passing = message_passing
        self.agg = agg

        # Batch Normalization
        self.bn = (
            torch.nn.BatchNorm1d(self.message_passing.output_dim)
            if batch_norm
            else torch.nn.Identity()
        )

        # Need to get the output dim of the message passing
        self.mp_output_dim = self.message_passing.output_dim

        # We will instantiate the head here for now - Up to discussion to bring it out like Chemprop does
        # And then allow for multiple heads
        self.head = ArrheniusHeadPredictor(
            n_tasks=1,  # Single task but has 3 targets
            input_dim=self.mp_output_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
        )

        # Arrhenius layer
        if arrhenius_layer_mean is not None and arrhenius_layer_scale is not None:
            self.utilize_arrhenius_layer = True

        else:
            logger.warning(
                "Arrhenius layer mean and scale not provided - not utilizing Arrhenius layer"
            )
            self.utilize_arrhenius_layer = False
        if self.utilize_arrhenius_layer and strat_temp_num_bins and strat_temp_samples_per_bin:
            self.stratify_temp = True
        else:
            logger.warning(
                "Stratified temperature sampling not enabled - not utilizing stratified temperature sampling"
            )
            self.stratify_temp = False

        self.arrhenius_layer = ArrheniusLayer(
            temps=temps,
            use_kJ=use_kJ,
            lnk_mean=torch.tensor(arrhenius_layer_mean, dtype=torch.float32)
            if arrhenius_layer_mean is not None
            else None,
            lnk_scale=torch.tensor(arrhenius_layer_scale, dtype=torch.float32)
            if arrhenius_layer_scale is not None
            else None,
        )

        # Weights for the loss
        self.w_A10 = w_A10
        self.w_n = w_n
        self.w_ea = w_ea
        self.w_lnK = w_lnK

        # Metrics
        # Validation metrics
        self.val_mse_A10 = MeanSquaredError()
        self.val_mse_n = MeanSquaredError()
        self.val_mse_ea = MeanSquaredError()
        self.val_mse_lnK = MeanSquaredError()
        self.val_r2_A10 = R2Score()
        self.val_r2_n = R2Score()
        self.val_r2_ea = R2Score()
        self.val_r2_lnK = R2Score()
        self.val_mae_A10 = MeanAbsoluteError()
        self.val_mae_n = MeanAbsoluteError()
        self.val_mae_ea = MeanAbsoluteError()
        self.val_mae_lnK = MeanAbsoluteError()

        # Test metrics (scaled and raw)
        self.test_mse_A10 = MeanSquaredError()
        self.test_mse_n = MeanSquaredError()
        self.test_mse_ea = MeanSquaredError()
        self.test_mse_lnK = MeanSquaredError()
        self.test_r2_A10 = R2Score()
        self.test_r2_n = R2Score()
        self.test_r2_ea = R2Score()
        self.test_r2_lnK = R2Score()
        self.test_mae_A10 = MeanAbsoluteError()
        self.test_mae_n = MeanAbsoluteError()
        self.test_mae_ea = MeanAbsoluteError()
        self.test_mae_lnK = MeanAbsoluteError()

        self.test_mse_raw_A10 = MeanSquaredError()
        self.test_mse_raw_n = MeanSquaredError()
        self.test_mse_raw_ea = MeanSquaredError()
        self.test_mse_raw_lnK = MeanSquaredError()
        self.test_r2_raw_A10 = R2Score()
        self.test_r2_raw_n = R2Score()
        self.test_r2_raw_ea = R2Score()
        self.test_r2_raw_lnK = R2Score()
        self.test_mae_raw_A10 = MeanAbsoluteError()
        self.test_mae_raw_n = MeanAbsoluteError()
        self.test_mae_raw_ea = MeanAbsoluteError()
        self.test_mae_raw_lnK = MeanAbsoluteError()

        # Optimization parameters
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs

    def fingerprint(
        self,
        bmgs: Iterable[BatchMolGraph],
        V_ds: Iterable[Tensor | None],
        X_d: Iterable[Tensor | None] = None,
    ) -> Tensor:
        """Generates the embeddings for the batch of molecules.
        This method computes the embeddings for a batch of molecules by performing message passing
        and aggregation. It returns the final embeddings, which can be used for further processing
        or prediction tasks.

        It has order awareness, meaning it preserves the order of components in the batch.
        If order awareness is required, the Hs are concatenated directly.
        If not, it symmetrizes the embeddings by averaging the concatenation in both orders.


        Parameters
        ----------
        bmgs : Iterable[BatchMolGraph]
            Iterable of BatchMolGraph objects representing the batch of molecules.
        V_ds : Iterable[Tensor | None]
            Iterable of tensors representing the node features for each molecule.
        X_d : Iterable[Tensor | None], optional
            Iterable of tensors representing the additional features for each molecule, by default None.
        """
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        if self.order_awareness:
            H = torch.cat(Hs, dim=-1)  # Preserve molecule order
        else:
            # Symmetrize: average of concat in both orders
            H1 = torch.cat(Hs, dim=-1)
            H2 = torch.cat(Hs[::-1], dim=-1)
            H = 0.5 * (H1 + H2)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def fingerprint_swapped(self, bmgs, V_ds, X_d=None) -> Tensor:
        """Generates the embeddings for the batch of molecules with swapped order awareness."""
        bmgs_sw = bmgs[::-1]  # Reverse the order of bmgs
        Vds_sw = V_ds[::-1] if V_ds is not None else None
        X_d_sw = X_d  # Same global features if any
        return self.fingerprint(bmgs_sw, Vds_sw, X_d_sw)

    def training_step(self, batch: MulticomponentTrainingBatch, batch_idx: int) -> Tensor:
        """Performs a training step on the given batch of data.

        Parameters
        ----------
        batch : MulticomponentTrainingBatch
            The batch of data to train on.
        batch_idx : int
            The index of the batch in the training loop.

        Returns
        -------
        Tensor
            The loss value for the training step.
        """
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        # Get the embeddings
        Z = self.fingerprint(bmg, V_d, X_d)

        # Check Z is finite
        # assert torch.isfinite(Z).all(), "Fingerprinting produced non‑finite embeddings"

        # Predict the scaled outputs
        Y_pred_scaled = self.head(Z)  # (B,3)

        # 3) Convert scaled to raw physical params
        #    returns a tensor (B,3) = [A, n, Ea]

        params_pred = self.convert_raw_outputs(Y_pred_scaled)
        params_true = self.convert_raw_outputs(targets)

        # 4) ArrheniusLayer to ln k curves on your grid
        if self.utilize_arrhenius_layer:
            sampled_indices = (
                self.sample_stratified_indices(
                    num_bins=self.strat_temp_num_bins,
                    samples_per_bin=self.strat_temp_samples_per_bin,
                )
                if self.stratify_temp
                else None
            )
            lnK_pred = self.arrhenius_layer(
                params_pred, sampled_indices=sampled_indices
            )  # (B, N_temps)

            lnK_true = self.arrhenius_layer(
                params_true, sampled_indices=sampled_indices
            )  # (B, N_temps)

        else:
            lnK_pred = lnK_true = torch.empty(0, device=Z.device)

        # assert torch.isfinite(params_pred).all(), "convert_raw_outputs produced non‑finite"
        # assert torch.isfinite(lnK_pred).all(),    "ArrheniusLayer produced non‑finite"

        # 5) Compute the loss
        loss = self._compute_loss(Y_pred_scaled, targets, lnK_pred, lnK_true, weights)
        assert torch.isfinite(loss), "Loss went NaN BEFORE backward"
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: MulticomponentTrainingBatch, batch_idx: int) -> Tensor:
        """Performs a validation step on the given batch of data.

        Parameters
        ----------
        batch : MulticomponentTrainingBatch
            The batch of data to validate on.
        batch_idx : int
            The index of the batch in the validation loop.

        Returns
        -------
        Tensor
            The loss value for the validation step.
        """
        bmgs, V_ds, X_d, targets, weights, lt_mask, gt_mask = batch
        Z = self.fingerprint(bmgs, V_ds, X_d)
        Y_pred_scaled = self.head(Z)  # (B,3)
        Y_pred_unscaled = self.convert_raw_outputs(Y_pred_scaled)
        Y_true_unscaled = self.convert_raw_outputs(targets)
        sampled_indices = (
            self.sample_stratified_indices(
                num_bins=self.strat_temp_num_bins, samples_per_bin=self.strat_temp_samples_per_bin
            )
            if self.stratify_temp
            else None
        )
        lnK_pred = self.arrhenius_layer(
            Y_pred_unscaled, sampled_indices=sampled_indices
        )  # (B, N_temps)
        lnK_true = self.arrhenius_layer(
            Y_true_unscaled, sampled_indices=sampled_indices
        )  # (B, N_temps)
        val_loss = self._compute_loss(Y_pred_scaled, targets, lnK_pred, lnK_true, weights)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.get_batch_size(batch),
        )

        # Unpack for metrics
        A10_p, n_p, ea_p = Y_pred_scaled.unbind(1)
        A10_t, n_t, ea_t = targets.unbind(1)

        self.val_mse_A10(A10_p, A10_t)
        self.val_r2_A10(A10_p, A10_t)
        self.val_mae_A10(A10_p, A10_t)
        self.val_mse_n(n_p, n_t)
        self.val_r2_n(n_p, n_t)
        self.val_mae_n(n_p, n_t)
        self.val_mse_ea(ea_p, ea_t)
        self.val_r2_ea(ea_p, ea_t)
        self.val_mae_ea(ea_p, ea_t)
        if self.utilize_arrhenius_layer:
            self.val_mse_lnK(lnK_pred.flatten(), lnK_true.flatten())
            self.val_r2_lnK(lnK_pred.flatten(), lnK_true.flatten())
            self.val_mae_lnK(lnK_pred.flatten(), lnK_true.flatten())

        return val_loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch to compute and log metrics."""
        # Compute and log metrics
        self.log("val/mse_A10", self.val_mse_A10.compute(), prog_bar=True)
        self.log("val/r2_A10", self.val_r2_A10.compute(), prog_bar=True)
        self.log("val/mse_n", self.val_mse_n.compute(), prog_bar=True)
        self.log("val/r2_n", self.val_r2_n.compute(), prog_bar=True)
        self.log("val/mse_ea", self.val_mse_ea.compute(), prog_bar=True)
        self.log("val/r2_ea", self.val_r2_ea.compute(), prog_bar=True)
        self.log("val/mae_A10", self.val_mae_A10.compute(), prog_bar=True)
        self.log("val/mae_n", self.val_mae_n.compute(), prog_bar=True)
        self.log("val/mae_ea", self.val_mae_ea.compute(), prog_bar=True)

        # Only log lnK metrics if Arrhenius layer is used
        if self.utilize_arrhenius_layer:
            self.log("val/mse_lnK", self.val_mse_lnK.compute(), prog_bar=True)
            self.log("val/r2_lnK", self.val_r2_lnK.compute(), prog_bar=True)
            self.log("val/mae_lnK", self.val_mae_lnK.compute(), prog_bar=True)

            # Compute overall metrics by averaging (including lnK)
            mse_vals = torch.stack(
                [
                    self.val_mse_A10.compute(),
                    self.val_mse_n.compute(),
                    self.val_mse_ea.compute(),
                    self.val_mse_lnK.compute(),
                ]
            )
            r2_vals = torch.stack(
                [
                    self.val_r2_A10.compute(),
                    self.val_r2_n.compute(),
                    self.val_r2_ea.compute(),
                    self.val_r2_lnK.compute(),
                ]
            )
            mae_vals = torch.stack(
                [
                    self.val_mae_A10.compute(),
                    self.val_mae_n.compute(),
                    self.val_mae_ea.compute(),
                    self.val_mae_lnK.compute(),
                ]
            )
        else:
            # Compute overall metrics by averaging (excluding lnK)
            mse_vals = torch.stack(
                [self.val_mse_A10.compute(), self.val_mse_n.compute(), self.val_mse_ea.compute()]
            )
            r2_vals = torch.stack(
                [self.val_r2_A10.compute(), self.val_r2_n.compute(), self.val_r2_ea.compute()]
            )
            mae_vals = torch.stack(
                [self.val_mae_A10.compute(), self.val_mae_n.compute(), self.val_mae_ea.compute()]
            )

        self.log("val/overall_mse", mse_vals.mean(), prog_bar=True)
        self.log("val/overall_mae", mae_vals.mean(), prog_bar=True)
        self.log("val/overall_r2", r2_vals.mean(), prog_bar=True)

        # Reset metrics after logging
        self.val_mse_A10.reset()
        self.val_r2_A10.reset()
        self.val_mse_n.reset()
        self.val_r2_n.reset()
        self.val_mse_ea.reset()
        self.val_r2_ea.reset()
        self.val_mae_A10.reset()
        self.val_mae_n.reset()
        self.val_mae_ea.reset()
        if self.utilize_arrhenius_layer:
            self.val_mse_lnK.reset()
            self.val_r2_lnK.reset()
            self.val_mae_lnK.reset()

    def test_step(self, batch: MulticomponentTrainingBatch, batch_idx: int) -> Tensor:
        """Performs a test step on the given batch of data.

        Parameters
        ----------
        batch : MulticomponentTrainingBatch
            The batch of data to test on.
        batch_idx : int
            The index of the batch in the test loop.

        Returns
        -------
        Tensor
            The loss value for the test step.
        """
        bmgs, V_ds, X_d, targets, weights, lt_mask, gt_mask = batch
        Z = self.fingerprint(bmgs, V_ds, X_d)
        Y_pred_scaled = self.head(Z)  # (B,3)
        Y_pred_unscaled = self.convert_raw_outputs(Y_pred_scaled)
        Y_true_unscaled = self.convert_raw_outputs(targets)
        if self.utilize_arrhenius_layer:
            lnK_pred = self.arrhenius_layer(Y_pred_unscaled)
            lnK_true = self.arrhenius_layer(Y_true_unscaled)
        else:
            lnK_pred = lnK_true = torch.empty(0, device=Z.device)

        # Unpack for metrics
        A10_p, n_p, ea_p = Y_pred_scaled.unbind(1)
        A10_t, n_t, ea_t = targets.unbind(1)

        # Update test metrics
        self.test_mse_A10.update(A10_p, A10_t)
        self.test_r2_A10.update(A10_p, A10_t)
        self.test_mae_A10.update(A10_p, A10_t)
        self.test_mse_n.update(n_p, n_t)
        self.test_r2_n.update(n_p, n_t)
        self.test_mae_n.update(n_p, n_t)
        self.test_mse_ea.update(ea_p, ea_t)
        self.test_r2_ea.update(ea_p, ea_t)
        self.test_mae_ea.update(ea_p, ea_t)
        if self.utilize_arrhenius_layer:
            self.test_mse_lnK.update(lnK_pred.flatten(), lnK_true.flatten())
            self.test_r2_lnK.update(lnK_pred.flatten(), lnK_true.flatten())
            self.test_mae_lnK.update(lnK_pred.flatten(), lnK_true.flatten())
        # Test Metrist RAW
        # Need to unbind the raw outputs
        raw_A10_p, raw_n_p, raw_ea_p = Y_pred_unscaled.unbind(1)
        raw_A10_t, raw_n_t, raw_ea_t = Y_true_unscaled.unbind(1)
        self.test_mse_raw_A10.update(raw_A10_p, raw_A10_t)
        self.test_r2_raw_A10.update(raw_A10_p, raw_A10_t)
        self.test_mae_raw_A10.update(raw_A10_p, raw_A10_t)
        self.test_mse_raw_n.update(raw_n_p, raw_n_t)
        self.test_r2_raw_n.update(raw_n_p, raw_n_t)
        self.test_mae_raw_n.update(raw_n_p, raw_n_t)
        self.test_mse_raw_ea.update(raw_ea_p, raw_ea_t)
        self.test_r2_raw_ea.update(raw_ea_p, raw_ea_t)
        self.test_mae_raw_ea.update(raw_ea_p, raw_ea_t)

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch to compute and log metrics."""
        # Compute and log metrics
        self.log("test/mse_A10", self.test_mse_A10.compute(), prog_bar=True)
        self.log("test/r2_A10", self.test_r2_A10.compute(), prog_bar=True)
        self.log("test/mse_n", self.test_mse_n.compute(), prog_bar=True)
        self.log("test/r2_n", self.test_r2_n.compute(), prog_bar=True)
        self.log("test/mse_ea", self.test_mse_ea.compute(), prog_bar=True)
        self.log("test/r2_ea", self.test_r2_ea.compute(), prog_bar=True)
        self.log("test/mae_A10", self.test_mae_A10.compute(), prog_bar=True)
        self.log("test/mae_n", self.test_mae_n.compute(), prog_bar=True)
        self.log("test/mae_ea", self.test_mae_ea.compute(), prog_bar=True)
        # log raw
        self.log("test/mse_raw_A10", self.test_mse_raw_A10.compute(), prog_bar=True)
        self.log("test/r2_raw_A10", self.test_r2_raw_A10.compute(), prog_bar=True)
        self.log("test/mse_raw_n", self.test_mse_raw_n.compute(), prog_bar=True)
        self.log("test/r2_raw_n", self.test_r2_raw_n.compute(), prog_bar=True)
        self.log("test/mse_raw_ea", self.test_mse_raw_ea.compute(), prog_bar=True)
        self.log("test/r2_raw_ea", self.test_r2_raw_ea.compute(), prog_bar=True)
        self.log("test/mae_raw_A10", self.test_mae_raw_A10.compute(), prog_bar=True)
        self.log("test/mae_raw_n", self.test_mae_raw_n.compute(), prog_bar=True)
        self.log("test/mae_raw_ea", self.test_mae_raw_ea.compute(), prog_bar=True)

        if self.utilize_arrhenius_layer:
            self.log("test/mse_lnK", self.test_mse_lnK.compute(), prog_bar=True)
            self.log("test/r2_lnK", self.test_r2_lnK.compute(), prog_bar=True)
            self.log("test/mae_lnK", self.test_mae_lnK.compute(), prog_bar=True)
            # Compute overall metrics by averaging (including lnK)
            mse_test = torch.stack(
                [
                    self.test_mse_A10.compute(),
                    self.test_mse_n.compute(),
                    self.test_mse_ea.compute(),
                    self.test_mse_lnK.compute(),
                ]
            )
            r2_test = torch.stack(
                [
                    self.test_r2_A10.compute(),
                    self.test_r2_n.compute(),
                    self.test_r2_ea.compute(),
                    self.test_r2_lnK.compute(),
                ]
            )
            mae_test = torch.stack(
                [
                    self.test_mae_A10.compute(),
                    self.test_mae_n.compute(),
                    self.test_mae_ea.compute(),
                    self.test_mae_lnK.compute(),
                ]
            )
        else:
            # Compute overall metrics by averaging (excluding lnK)
            mse_test = torch.stack(
                [self.test_mse_A10.compute(), self.test_mse_n.compute(), self.test_mse_ea.compute()]
            )
            r2_test = torch.stack(
                [self.test_r2_A10.compute(), self.test_r2_n.compute(), self.test_r2_ea.compute()]
            )
            mae_test = torch.stack(
                [self.test_mae_A10.compute(), self.test_mae_n.compute(), self.test_mae_ea.compute()]
            )

        # Log overall metrics scaled
        self.log("test/overall_mse", mse_test.mean(), prog_bar=True)
        self.log("test/overall_mae", mae_test.mean(), prog_bar=True)
        self.log("test/overall_r2", r2_test.mean(), prog_bar=True)

        # Log overall metrics raw
        mse_raw_test = torch.stack(
            [
                self.test_mse_raw_A10.compute(),
                self.test_mse_raw_n.compute(),
                self.test_mse_raw_ea.compute(),
            ]
        )
        r2_raw_test = torch.stack(
            [
                self.test_r2_raw_A10.compute(),
                self.test_r2_raw_n.compute(),
                self.test_r2_raw_ea.compute(),
            ]
        )
        mae_raw_test = torch.stack(
            [
                self.test_mae_raw_A10.compute(),
                self.test_mae_raw_n.compute(),
                self.test_mae_raw_ea.compute(),
            ]
        )
        self.log("test/raw_overall_mse", mse_raw_test.mean(), prog_bar=True)
        self.log("test/raw_overall_mae", mae_raw_test.mean(), prog_bar=True)
        self.log("test/raw_overall_r2", r2_raw_test.mean(), prog_bar=True)

        # Reset metrics after logging
        self.test_mse_A10.reset()
        self.test_r2_A10.reset()
        self.test_mse_n.reset()
        self.test_r2_n.reset()
        self.test_mse_ea.reset()
        self.test_r2_ea.reset()
        self.test_mae_A10.reset()
        self.test_mae_n.reset()
        self.test_mae_ea.reset()
        if self.utilize_arrhenius_layer:
            self.test_mse_lnK.reset()
            self.test_r2_lnK.reset()
            self.test_mae_lnK.reset()
        # Reset raw metrics
        self.test_mse_raw_A10.reset()
        self.test_r2_raw_A10.reset()
        self.test_mse_raw_n.reset()
        self.test_r2_raw_n.reset()
        self.test_mse_raw_ea.reset()
        self.test_r2_raw_ea.reset()
        self.test_mae_raw_A10.reset()
        self.test_mae_raw_n.reset()
        self.test_mae_raw_ea.reset()

    def predict_step(
        self, batch: MulticomponentTrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Performs a prediction step on the given batch of data.

        Parameters
        ----------
        batch : BatchType
            The batch of data to predict on.
        batch_idx : int
            The index of the batch in the prediction loop.
        dataloader_idx : int, optional
            The index of the dataloader, by default 0.

        Returns
        -------
        Tensor
            The predicted outputs for the batch.
        """
        bmgs, V_ds, X_d, targets, *rest = batch

        # 1) fingerprint → head (scaled)
        Z = self.fingerprint(bmgs, V_ds, X_d)
        Y_scaled = self.head(Z)  # (B,3)

        # 2) scaled to raw physical [lnA, n, Ea]
        params_raw = self.convert_raw_outputs(Y_scaled)  # (B,3)

        # 3) Arrhenius curve
        if self.utilize_arrhenius_layer:
            lnK_pred = self.arrhenius_layer(params_raw)  # (B, N_temps)
        else:
            logger.warning("Arrhenius layer is not utilized, lnK predictions will be empty.")
            lnK_pred = torch.empty(0, device=Z.device)

        return PredictOutput(Y_scaled=Y_scaled, Y_raw=params_raw, lnK_pred=lnK_pred)

    def _compute_loss(
        self,
        Y_pred_scaled: Tensor,
        Y_true_scaled: Tensor,
        lnK_pred: Tensor,
        lnK_true: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Computes the loss for the training step.

        Parameters
        ----------
        Y_pred_scaled : Tensor
            The predicted outputs from the model.
        Y_true : Tensor
            The true outputs for the batch.
        weights : Tensor
            The weights for each sample in the batch.

        Returns
        -------
        Tensor
            The computed loss value.
        """

        A10_p, n_p, Ea_p = Y_pred_scaled.unbind(1)
        A10_t, n_t, Ea_t = Y_true_scaled.unbind(1)
        w = weights.squeeze(-1)

        loss_A10 = F.mse_loss(A10_p, A10_t, reduction="none") * w
        loss_n = F.mse_loss(n_p, n_t, reduction="none") * w
        loss_ea = F.mse_loss(Ea_p, Ea_t, reduction="none") * w

        if self.utilize_arrhenius_layer:
            loss_lnK = F.mse_loss(lnK_pred, lnK_true, reduction="none").mean(1) * w
            tot_w = self.w_A10 + self.w_n + self.w_ea + self.w_lnK
            loss = (
                self.w_A10 * loss_A10.mean()
                + self.w_n * loss_n.mean()
                + self.w_ea * loss_ea.mean()
                + self.w_lnK * loss_lnK.mean()
            ) / tot_w
        else:
            tot_w = self.w_A10 + self.w_n + self.w_ea
            loss = (
                self.w_A10 * loss_A10.mean() + self.w_n * loss_n.mean() + self.w_ea * loss_ea.mean()
            ) / tot_w
        return loss

    def convert_raw_outputs(self, z: torch.Tensor) -> torch.Tensor:
        log10A_z, n_z, Ea_YJ_z = z.unbind(1)

        mean, std = self.unscaler.mean.squeeze(0), self.unscaler.scale.squeeze(0)
        log10A = log10A_z * std[0] + mean[0]
        n = n_z * std[1] + mean[1]

        pt_mu = torch.as_tensor(self.ea_scales._scaler.mean_[0], device=z.device, dtype=z.dtype)
        pt_sigma = torch.as_tensor(self.ea_scales._scaler.scale_[0], device=z.device, dtype=z.dtype)
        Ea_YJ = Ea_YJ_z * pt_sigma + pt_mu

        lam = float(self.ea_scales.lambdas_[0])
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

        # ---- prefactor guard -------------------------------------------------
        log10A = log10A.clamp(min=-20.0, max=20.0)  # 10^±20 ≈ 1 e±20  (always finite)
        A = torch.pow(10.0, log10A)

        return torch.stack((A, n, Ea), dim=1)

    def _inverse_yeojohnson(self, y: torch.Tensor, lam: float) -> torch.Tensor:
        pos = y >= 0
        if lam == 0.0:
            return torch.where(pos, torch.exp(y) - 1.0, -torch.exp(-y) + 1.0)
        inv_pos = torch.pow(y * lam + 1.0, 1.0 / lam) - 1.0
        inv_neg = -torch.pow(-y * (2.0 - lam) + 1.0, 1.0 / (2.0 - lam)) + 1.0

        # Check if there are any issues with the inverse
        if not torch.isfinite(inv_pos).all() or not torch.isfinite(inv_neg).all():
            logger.warning("Non-finite values detected in inverse Yeo-Johnson transformation.")
            logger.warning(f"pos: {pos}, inv_pos: {inv_pos}, inv_neg: {inv_neg}")
        return torch.where(pos, inv_pos, inv_neg)

    def _safe_pow(self, base: torch.Tensor, exp: float) -> torch.Tensor:
        # keep sign, avoid nan for negative bases & fractional exponents
        return torch.sign(base) * torch.pow(base.abs().clamp_min(1e-12), exp)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}
        for p in self.parameters():
            assert torch.isfinite(p).all(), "Non-finite parameters detected in the model."

        # Report the gradient norm for debugging
        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    def sample_stratified_indices(
        self, num_bins: int = 12, samples_per_bin: int = 1
    ) -> torch.Tensor:
        T = self.arrhenius_layer.T
        bin_edges = torch.linspace(
            T.min().to(T.device), T.max().to(T.device), steps=num_bins + 1
        ).to(T.device)
        sampled_indices = []

        for i in range(num_bins):
            in_bin = (T >= bin_edges[i]) & (T < bin_edges[i + 1])
            indices_in_bin = torch.where(in_bin)[0]
            if len(indices_in_bin) > 0:
                selected = indices_in_bin[torch.randperm(len(indices_in_bin))[:samples_per_bin]]
                sampled_indices.extend(selected.tolist())

        return torch.tensor(sampled_indices, dtype=torch.long)

    def get_batch_size(self, batch: MulticomponentTrainingBatch) -> int:
        return len(batch[0][0])
