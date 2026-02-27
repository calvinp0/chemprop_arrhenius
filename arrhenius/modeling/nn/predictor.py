from __future__ import annotations
from typing import Optional
import torch
from torch import nn, Tensor
from chemprop.nn.transforms import UnscaleTransform
from chemprop.utils import Factory
from chemprop.nn.predictors import _FFNPredictorBase, PredictorRegistry
from chemprop.nn.metrics import (
    MSE,
    SID,
    BCELoss,
    BinaryAUROC,
    ChempropMetric,
    CrossEntropyLoss,
    DirichletLoss,
    EvidentialLoss,
    MulticlassMCCMetric,
    MVELoss,
    QuantileLoss,
)

_ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
}


def _act(name: str):
    return _ACTS.get((name or "relu").lower(), nn.ReLU)()


class LazyMLP(nn.Module):
    """
    A tiny MLP that infers input width at first forward using nn.LazyLinear.
    Exposes .input_dim and .output_dim to mirror chemprop.nn.ffn.MLP.
    """

    def __init__(
        self, out_dim: int, hidden_dim: int, n_layers: int, dropout: float, activation: str
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if n_layers <= 0:
            layers.append(nn.LazyLinear(out_dim))
        else:
            # first layer is lazy; subsequent are static (hidden_dim-known)
            layers.append(nn.LazyLinear(hidden_dim))
            layers.append(_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(_act(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self._input_dim: Optional[int] = None
        self.output_dim: int = out_dim

    @property
    def input_dim(self) -> int:
        # -1 until first forward; consumers usually don’t need it earlier
        return -1 if self._input_dim is None else self._input_dim

    def forward(self, x: Tensor) -> Tensor:
        if self._input_dim is None:
            self._input_dim = x.shape[1]
        return self.net(x)


@PredictorRegistry.register("arrhenius-head")
class ArrheniusHeadPredictor(_FFNPredictorBase):
    """Static when input_dim is provided; lazy when input_dim=None."""

    n_targets = 3
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_tasks: int = 1,
        input_dim: Optional[int] = None,  # None => lazy
        hidden_dim: int = 512,
        n_layers: int = 1,
        dropout: float = 0.2,
        activation: str = "relu",
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        if input_dim is not None:
            # Static path: use the base initializer
            super().__init__(
                n_tasks=n_tasks,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                activation=activation,
                criterion=criterion,
                task_weights=task_weights,
                threshold=threshold,
                output_transform=output_transform,
            )
        else:
            # Lazy path: bypass base __init__, avoid touching self.hparams
            nn.Module.__init__(self)
            self.n_targets = ArrheniusHeadPredictor.n_targets
            out_dim = n_tasks * self.n_targets
            self.ffn = LazyMLP(
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                activation=activation,
            )
            wt = torch.ones(n_tasks) if task_weights is None else task_weights
            self.criterion = criterion or Factory.build(
                self._T_default_criterion, task_weights=wt, threshold=threshold
            )
            self.output_transform = (
                output_transform if output_transform is not None else nn.Identity()
            )

    def forward(self, Z: Tensor) -> Tensor:
        return self.ffn(Z)

    train_step = forward
