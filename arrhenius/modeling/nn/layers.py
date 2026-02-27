import torch
from torch import nn


class ArrheniusLayer(nn.Module):
    """
    params  : tensor (B,3) = [A (s⁻¹), n, Ea (kJ mol⁻¹)]
    returns : ln k(T)            –OR–      z‑score(ln k(T))
              shape (B, N_T)       shape (B, N_T)
    """

    def __init__(
        self,
        temps: list[float],
        *,
        use_kJ: bool = True,
        lnk_mean: torch.Tensor | None = None,  # (N_T,) from StandardScaler
        lnk_scale: torch.Tensor | None = None,  # (N_T,)
    ):
        super().__init__()

        self.register_buffer("T", torch.tensor(temps, dtype=torch.float32))  # (N_T,)
        self.R = 8.31446261815324e-3 if use_kJ else 8.31446261815324
        self.register_buffer("T0", torch.tensor(1.0, dtype=torch.float32))

        # Optional standardisation
        if lnk_mean is not None and lnk_scale is not None:
            self.register_buffer("lnk_mu", lnk_mean.float())
            self.register_buffer("lnk_sig", lnk_scale.float())
        else:
            self.lnk_mu = self.lnk_sig = None

    def forward(
        self, params: torch.Tensor, sampled_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        A, n, Ea = params.unbind(1)  # (B,)

        T = self.T if sampled_indices is None else self.T[sampled_indices]
        T = T.to(A.device).unsqueeze(0)  # (1, N_sampled_T)

        ln_k = (
            torch.log(torch.clamp(A, min=1e-30)).unsqueeze(1)
            + n.unsqueeze(1) * torch.log(T / self.T0)
            - Ea.unsqueeze(1) / (self.R * T)
        )  # (B, N_sampled_T)

        if self.lnk_mu is not None:
            mu = self.lnk_mu[sampled_indices] if sampled_indices is not None else self.lnk_mu
            sig = self.lnk_sig[sampled_indices] if sampled_indices is not None else self.lnk_sig
            ln_k = (ln_k - mu) / sig

        return ln_k
