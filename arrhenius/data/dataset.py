from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from chemprop import data as cpdata
from chemprop.data.datasets import Datum
from chemprop.featurizers.molgraph.cache import MolGraphCache, MolGraphCacheOnTheFly

PreprocFN = Callable[[np.ndarray], np.ndarray]


@dataclass
class ArrMoleculeDataset:
    """
    Wraps Chemprop MoleculeDataset and adds column-wise V_f transforms
    """

    base: cpdata.MoleculeDataset
    cache: bool = False

    def __post_init__(self):
        self._vf_transforms: Dict[int, PreprocFN] = {}
        self._init_cache()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Datum:
        d = self.base.data[idx]
        mg = self.mg_cache[idx]
        return Datum(
            mg,
            self.base.V_ds[idx],
            self.base.X_d[idx],
            self.base.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    def _init_cache(self):
        self.mg_cache = (MolGraphCache if self.cache else MolGraphCacheOnTheFly)(
            self.base.mols, self.base.V_fs, self.base.E_fs, self.base.featurizer
        )

    @property
    def names(self) -> List[str]:
        return self.base.names

    @property
    def smiles(self) -> List[str]:
        return self.base.smiles

    @property
    def mols(self) -> List[str]:
        return self.base.mols

    @property
    def d_vf(self) -> int:
        return 0 if self.base.V_fs[0] is None else self.base.V_fs[0].shape[1]

    @property
    def d_x(self) -> int:
        X = self.base.X_d
        return 0 if (X is None or X.size == 0) else int(X.shape[1])

    # --- Transform Registry ---
    def register_vf_transform(self, col: int, fn: PreprocFN):
        self._vf_transforms[col] = fn

    def clear_vf_transforms(self, col: int):
        self._vf_transforms.pop(col, None)

    def clear_all_vf_transforms(self):
        self._vf_transforms.clear()

    def _apply_vf_transforms(self, X: np.ndarray) -> np.ndarray:
        """
        Applies registered transforms to the given feature matrix X.
        """
        if not self._vf_transforms:
            return X
        pieces = []
        for j in range(X.shape[1]):
            if j in self._vf_transforms:
                block = self._vf_transforms[j](X[:, j])
                block = np.atleast_2d(block).T if block.ndim == 1 else block
                pieces.append(block)
            else:
                pieces.append(X[:, j : j + 1])
        return np.concatenate(pieces, axis=1)

    def normalize_targets(
        self,
        scaler: TransformerMixin | None = None,
        columns: Dict[int, TransformerMixin] | None = None,
    ) -> TransformerMixin:
        # Build Y_raw as a proper 2D array: (N, d)
        rows = []
        for d in self.base.data:
            y = d.y
            if y is None:
                raise ValueError(
                    "Found datapoint with y=None. All targets must be present before scaling."
                )
            y_arr = np.asarray(y, dtype=float).ravel()  # (d,)
            rows.append(y_arr)
        Y_raw = np.vstack(rows)  # (N, d)

        if scaler is None:
            # Fit a new scaler on this component’s Y_raw
            if columns is None:
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler().fit(Y_raw)
            else:
                # columns: {col_index: transformer}
                other = [i for i in range(Y_raw.shape[1]) if i not in columns]
                transformer_list = [(f"t{i}", tr, [i]) for i, tr in columns.items()]
                if other:
                    from sklearn.preprocessing import StandardScaler

                    transformer_list.append(("std", StandardScaler(), other))
                scaler = ColumnTransformer(transformer_list, remainder="drop").fit(Y_raw)
        else:
            # Reusing a fitted scaler: verify dimensionality matches
            # ColumnTransformer stores n_features_in_ as the number of features seen at fit
            n_expected = getattr(scaler, "n_features_in_", None)
            if n_expected is not None and Y_raw.shape[1] != n_expected:
                raise ValueError(
                    f"Target dimensionality mismatch: scaler expects {n_expected} features, "
                    f"but dataset has {Y_raw.shape[1]}. "
                    f"Tip: ensure you're consistently using 3 targets (forward only) "
                    f"or 6 targets (forward+reverse) across all splits/components."
                )

        self.base.Y = scaler.transform(Y_raw)
        return scaler

    def transformed_vf_matrix(self) -> Optional[np.ndarray]:  # <- Optional
        """Return V_f after applying column transforms, concatenated across molecules."""
        if self.base.V_fs[0] is None:
            return None
        X_raw = np.concatenate(self.base.V_fs, axis=0)
        return self._apply_vf_transforms(X_raw)

    def xd_matrix(self) -> Optional[np.ndarray]:  # <- Optional
        X = self.base.X_d
        return None if X is None else X.astype(np.float32, copy=False)

    def reset(self):
        self.base.reset()
        self._init_cache()

    def normalize_inputs(
        self,
        key: str = "V_f",
        scaler: Optional[StandardScaler] = None,
        columns_to_scale: Optional[List[int]] = None,
    ) -> Optional[StandardScaler]:
        """
        For V_f:
          - Concatenate atoms across molecules
          - Apply per-column transforms (RBF, angle norm, etc.)
          - Fit/apply StandardScaler (train: scaler=None; val/test: pass fitted scaler)
          - Split back per-molecule and write to base.V_fs
        """
        if key == "V_f":
            if self.d_vf == 0:
                return None  # no V_f to scale

            # gather
            lens = [vf.shape[0] for vf in self.base.V_fs]
            X_raw = np.concatenate(self.base.V_fs, axis=0)  # (sum_atoms, d_raw)

            # transforms
            X_tr = self._apply_vf_transforms(X_raw)  # (sum_atoms, d_new)

            # fit / apply scaler
            if scaler is None:
                scaler = (
                    StandardScaler().fit(X_tr[:, columns_to_scale])
                    if columns_to_scale
                    else StandardScaler().fit(X_tr)
                )
            if columns_to_scale:
                X_scaled = X_tr.copy()
                X_scaled[:, columns_to_scale] = scaler.transform(X_tr[:, columns_to_scale])
            else:
                X_scaled = scaler.transform(X_tr)

            # split back
            split_idx = np.cumsum(lens)[:-1]
            new_V_fs = np.split(X_scaled, split_idx, axis=0)
            self.base.V_fs = new_V_fs  # triggers chemprop cache refresh
            self._init_cache()
            return scaler
        if key == "X_d":
            if self.base.X_d is None or self.d_x == 0:
                return scaler
            X = self.base.X_d.astype(np.float32, copy=False)
            if scaler is None:
                scaler = (
                    StandardScaler().fit(X[:, columns_to_scale])
                    if columns_to_scale
                    else StandardScaler().fit(X)
                )
            if columns_to_scale is not None:
                X_scaled = X.copy()
                X_scaled[:, columns_to_scale] = scaler.transform(X[:, columns_to_scale])
                self.base.X_d = X_scaled
            else:
                self.base.X_d = scaler.transform(X)
            return scaler
        raise ValueError(f"Unsupported key '{key}' for input normalization.")


@dataclass
class ArrMulticomponentDataset:
    """Parallel wrapper compatible with Chemprop training/eval loops"""

    components: List[ArrMoleculeDataset]

    def __post_init__(self):
        sizes = [len(c) for c in self.components]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("All components must have the same number of molecules.")
        # NEW: state for shared input scaler
        self._shared_input_scalers: Dict[str, StandardScaler] = {}
        self._columns_to_scale: Dict[str, Optional[List[int]]] = {}

    def __len__(self):
        return len(self.components[0])

    def __getitem__(self, idx: int) -> List[Datum]:
        return [c[idx] for c in self.components]

    @property
    def names(self):
        return list(zip(*[c.names for c in self.components], strict=False))

    @property
    def smiles(self):
        return list(zip(*[c.smiles for c in self.components], strict=False))

    @property
    def mols(self):
        return list(zip(*[c.mols for c in self.components], strict=False))

    def normalize_targets(self, scaler=None, columns=None):
        """
        Fit/apply target scaler on component 0 ONLY (Chemprop convention).
        Other components are left as-is; they can have y=None.
        """
        return self.components[0].normalize_targets(scaler, columns)

    def fit_shared_input_scaler(
        self,
        key: str = "V_f",
        columns_to_scale: Optional[List[int]] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> StandardScaler:
        mats = []
        if key == "V_f":
            for comp in self.components:
                X_tr = comp.transformed_vf_matrix()
                if X_tr is None:
                    continue
                mats.append(X_tr[:, columns_to_scale] if columns_to_scale else X_tr)
        elif key == "X_d":
            for comp in self.components:
                X = comp.base.X_d
                if X is None:
                    continue
                mats.append(X[:, columns_to_scale] if columns_to_scale else X)
        else:
            raise ValueError(f"Unsupported key '{key}' for input normalization.")
        X_cat = np.vstack(mats) if mats else None
        if X_cat is None:
            raise ValueError(f"No data found to fit shared input scaler for key '{key}'.")
        scaler = (scaler or StandardScaler()).fit(X_cat)
        self._shared_input_scalers[key] = scaler
        self._columns_to_scale[key] = columns_to_scale
        return scaler

    def apply_shared_input_scaler(self, key: str = "V_f", scaler: Optional[StandardScaler] = None):
        # NEW: defensive init if __post_init__ wasn’t run for some reason
        if not hasattr(self, "_shared_input_scalers"):
            self._shared_input_scalers = {}
            self._columns_to_scale = {}
        if scaler is None:
            if key not in self._shared_input_scalers:
                raise ValueError(
                    f"No stored scaler for key '{key}'. Call fit_shared_input_scaler first."
                )
            scaler = self._shared_input_scalers[key]
        cols = self._columns_to_scale.get(key, None)
        for comp in self.components:
            comp.normalize_inputs(key=key, scaler=scaler, columns_to_scale=cols)
        return scaler

    # One-shot convenience: fit on self (TRAIN), then apply to self (TRAIN)
    def normalize_inputs_shared(
        self,
        key: str = "V_f",
        columns_to_scale: Optional[List[int]] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> StandardScaler:
        if scaler is None:
            scaler = self.fit_shared_input_scaler(key=key, columns_to_scale=columns_to_scale)
        else:
            from sklearn.utils.validation import check_is_fitted

            try:
                check_is_fitted(scaler)
            except Exception:
                # If a custom (but unfitted) scaler instance was provided, fit it here
                scaler = self.fit_shared_input_scaler(
                    key=key, columns_to_scale=columns_to_scale, scaler=scaler
                )
            else:
                # Store fitted scalers passed in so apply_shared_input_scaler can reuse them
                self._shared_input_scalers[key] = scaler
                self._columns_to_scale[key] = columns_to_scale
        self.apply_shared_input_scaler(key=key, scaler=scaler)
        return scaler

    def normalize_inputs(
        self,
        key="V_f",
        scalers: Optional[List[StandardScaler]] = None,
        columns_to_scale: Optional[List[int]] = None,
    ):
        if scalers is None:
            return [
                c.normalize_inputs(key, scaler=None, columns_to_scale=columns_to_scale)
                for c in self.components
            ]
        else:
            assert len(scalers) == len(self.components)
            return [
                c.normalize_inputs(key, scaler=s, columns_to_scale=columns_to_scale)
                for c, s in zip(self.components, scalers, strict=False)
            ]

    def reset(self):
        for c in self.components:
            c.reset()
