from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from arrhenius.training.hpo.data import fit_global_normalizers, make_loaders


def _scaler_key(train_idx: Sequence[int], cfg: Dict[str, Any]) -> Tuple:
    """
    Key that scopes scaler reuse to the training set and feature knobs
    that affect scaling.
    """
    train_sig = tuple(sorted(map(int, train_idx)))
    return (
        train_sig,
        str(cfg.get("extra_mode", "baseline")),
        str(cfg.get("global_mode", "none")),
        int(cfg.get("morgan_bits", 2048)),
        int(cfg.get("morgan_radius", 2)),
    )


class LoaderCache:
    """
    Cache scalers and loaders keyed by train set + feature config to ensure
    reproducible normalization across inner/outer/eval without refitting.
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self._scaler_cache: Dict[Tuple, Dict[str, Any]] = {}

    def _get_scalers(self, train_idx: Sequence[int], cfg: Dict[str, Any]) -> Dict[str, Any]:
        key = _scaler_key(train_idx, cfg)
        if key not in self._scaler_cache:
            self._scaler_cache[key] = fit_global_normalizers(self.bundle, cfg, train_idx)
        return self._scaler_cache[key]

    def get_loaders(
        self,
        cfg: Dict[str, Any],
        train_idx: Sequence[int],
        val_idx: Sequence[int],
        test_idx: Optional[Sequence[int]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        scalers = self._get_scalers(train_idx, cfg)
        return make_loaders(
            bundle=self.bundle,
            cfg=cfg,
            train_idx=list(map(int, train_idx)),
            val_idx=list(map(int, val_idx)),
            test_idx=list(map(int, test_idx)) if test_idx is not None else None,
            seed=seed,
            preset_r_bounds=scalers.get("r_bounds"),
            preset_y_scaler=scalers.get("y_scaler"),
            preset_vf_scaler=scalers.get("vf_scaler"),
            preset_xd_scaler=scalers.get("xd_scaler"),
        )
