# chemprop/nn/transforms.py  (or wherever you keep the others)
# -------------------------------------------------------------
from typing import Sequence
import numpy as np
import torch
from torch import Tensor, nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer


class UnscaleColumnTransform(nn.Module):
    """
    Inference‑time inverse of a fitted ColumnTransformer whose final
    output space is *standardised* (mean 0, variance 1) column‑wise.

    Supported sub‑transformers
    --------------------------
    * StandardScaler
    * PowerTransformer(standardize=True)
    * "passthrough" (mean=0, scale=1)
    """

    def __init__(self, mean: Sequence[float], scale: Sequence[float], pad: int = 0):
        super().__init__()

        mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])
        scale = torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])

        if mean.shape != scale.shape:
            raise ValueError("'mean' and 'scale' must have same shape")

        self.register_buffer("mean", mean.unsqueeze(0))  # (1, d)
        self.register_buffer("scale", scale.unsqueeze(0))  # (1, d)

    # --------- factory -------------------------------------------------
    @classmethod
    def from_column_transformer(cls, ct: ColumnTransformer, pad: int = 0):
        means = []
        scales = []

        for name, tr, cols in ct.transformers_:
            if tr == "drop":
                continue
            n_out = len(cols)

            if isinstance(tr, StandardScaler):
                means.extend(tr.mean_)
                scales.extend(tr.scale_)
            elif isinstance(tr, PowerTransformer):
                if tr.standardize:
                    means.extend(np.zeros(n_out))
                    scales.extend(np.ones(n_out))
                else:
                    raise ValueError(
                        f"PowerTransformer '{name}' has standardize=False; " "mean/scale undefined."
                    )
            else:  # passthrough or unknown → assume already std‑ised
                means.extend(np.zeros(n_out))
                scales.extend(np.ones(n_out))

        return cls(means, scales, pad=pad)

    # --------- forward -------------------------------------------------
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X  # no effect during training
        return X * self.scale + self.mean  # inverse z‑score

    def transform_variance(self, var: Tensor) -> Tensor:
        if self.training:
            return var
        return var * (self.scale**2)



# ----------------------------------------------------------------------
# usage ---------------------------------------------------------------
# ----------------------------------------------------------------------
# Y_scaler is the ColumnTransformer returned by normalize_targets(...)
# output_transform = UnscaleColumnTransform.from_column_transformer(Y_scaler)
