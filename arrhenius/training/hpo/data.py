# run_hpo/data.py

# run_hpo/data.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Sequence
import os, numpy as np, pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem import inchi
import torch

from chemprop import data as cpdata, featurizers
from arrhenius.data.dataset import ArrMulticomponentDataset, ArrMoleculeDataset
from arrhenius.data.collate import build_loader_mc
from arrhenius.modeling.nn.transformers import UnscaleColumnTransform
from arrhenius.modeling.module.pl_rateconstant_dir import ArrheniusMultiComponentMPNN
from chemprop.CUSTOM.featuriser.featurise import Featuriser, MOL_TYPES
from chemprop.featurizers import (
    MorganBinaryFeaturizer,
    MorganCountFeaturizer,
    V1RDKit2DNormalizedFeaturizer,
)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from arrhenius.training.hpo.feature_modes import (
    ANGLE_COL,
    DIHEDRAL_COL,
    RADIUS_COL,
    atom_extra_dim,
    canonicalize_extra_mode,
    canonicalize_global_mode,
    canonicalize_rad_source,
    mode_settings,
)

USE_DIHEDRAL_SIN_COS = True
CP_NUM_WORKERS = int(os.getenv("CP_NUM_WORKERS", "4"))


# ---------- small helpers ----------
def _mol(obj):
    return obj if not isinstance(obj, tuple) else obj[0]


def ik14(mol):
    return inchi.MolToInchiKey(_mol(mol))[:14]


def rbf_expand(values, num_centers=16, r_min=0.5, r_max=8.0, gamma=None):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(r_min, r_max, num_centers, dtype=np.float32).reshape(1, -1)
    if gamma is None:
        step = (r_max - r_min) / max(num_centers - 1, 1)
        gamma = np.float32(1.0 / (step**2 + 1e-8))
    return np.exp(-gamma * (values - centers) ** 2)


def dihedral_to_sin_cos(x):
    x = np.asarray(x, dtype=np.float32)
    return np.stack([np.sin(x), np.cos(x)], axis=1)


def train_radius_bounds(
    attached_pair_dps, train_idx, extras_mode: str | None = None, hard=(0.5, 8.0)
):
    mode_cfg = mode_settings(canonicalize_extra_mode(extras_mode or "baseline"))
    radius_idx = mode_cfg["cols"].index(RADIUS_COL) if RADIUS_COL in mode_cfg["cols"] else None
    if (not mode_cfg["rad_mode"]) or radius_idx is None:
        return hard
    vals = []
    for i in train_idx:
        for c in (0, 1):
            A = attached_pair_dps[c][i].V_f
            if A is None:
                raise ValueError(
                    "Encountered datapoint with V_f=None while extras_mode requires atom extras. "
                    "Please ensure auxiliary features are available."
                )
            v = A[:, radius_idx]
            vals.append(v[np.isfinite(v)])
    if not vals:
        return hard
    v = np.concatenate(vals)
    lo = max(hard[0], float(np.percentile(v, 1)))
    hi = min(hard[1], float(np.percentile(v, 99)))
    if hi - lo < 1e-3:
        lo, hi = hard
    return lo, hi


def torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------- feature attachment (lifted & trimmed from your script) ----------
def _build_global_featurizer(global_mode: str, bits: int, radius: int):
    global_mode = canonicalize_global_mode(global_mode)
    if global_mode == "morgan_binary":
        fe = MorganBinaryFeaturizer(radius=radius, length=bits)
        return lambda mol: fe(mol).astype(np.float32), False  # no centering
    if global_mode == "morgan_count":
        fe = MorganCountFeaturizer(radius=radius, length=bits)
        return lambda mol: fe(mol).astype(np.float32), True
    if global_mode == "rdkit2d_norm":
        fe = V1RDKit2DNormalizedFeaturizer()
        return lambda mol: fe(mol).astype(np.float32), True
    return None, True  # global_mode == "none"


def attach_feats_to_dps_pairwise(
    feat_data,
    atom_extra_feats: Optional[pd.DataFrame],
    cols: List[str],
    *,
    global_mode: str = "none",
    morgan_bits: int = 2048,
    morgan_radius: int = 2,
    rad_geom_cols: Optional[Dict[str, int]] = None,
    rad_mask_hops: Optional[int] = None,
    path_col: str = "shortest_path",
    rxn_col: str = "rxn_id",
    moltype_col: str = "mol_type",
    atom_index_col: str = "focus_atom_idx",
    r1h_tag="r1h",
    r2h_tag="r2h",
):
    use_atom_extras = atom_extra_feats is not None
    datapoints = [[], []]
    kept_idx = []
    x_d_shape = 0

    gf, center_globals = _build_global_featurizer(global_mode, morgan_bits, morgan_radius)

    def _clean(n):
        return n.replace("_r1h", "").replace("_r2h", "")

    idx_by_rxn_r1h = {_clean(dp.name): i for i, dp in enumerate(feat_data[0])}
    idx_by_rxn_r2h = {_clean(dp.name): i for i, dp in enumerate(feat_data[1])}
    common_rxns = sorted(set(idx_by_rxn_r1h) & set(idx_by_rxn_r2h))

    def _impute_inplace(A, cols):
        name2idx = {n: i for i, n in enumerate(cols)}
        cont = [
            c for c in ("q_mull", "q_apt", "f_mag", "angle", "dihedral", "radius") if c in name2idx
        ]
        bins = [
            c
            for c in (
                "r_exist",
                "a_exist",
                "d_exist",
                "is_donor",
                "is_acceptor",
                "is_donor_H",
                "is_acceptor_H",
                "is_acceptor_neighbor",
                "is_donor_neighbor",
            )
            if c in name2idx
        ]
        for c in bins:
            j = name2idx[c]
            m = ~np.isfinite(A[:, j])
            if m.any():
                A[m, j] = 0.0
        for c in cont:
            j = name2idx[c]
            col = A[:, j]
            if not np.isfinite(col).all():
                finite = col[np.isfinite(col)]
                fill = float(np.median(finite)) if finite.size else 0.0
                col[~np.isfinite(col)] = fill
                A[:, j] = col

    def _mask_by_path_hops(A: np.ndarray, sub_df: pd.DataFrame, cols: List[str]):
        if rad_mask_hops is None:
            return
        if path_col not in sub_df.columns:
            return
        if not rad_geom_cols:
            return

        name2idx = {n: i for i, n in enumerate(cols)}
        for atom_idx in range(len(sub_df)):
            try:
                raw = sub_df.iloc[atom_idx][path_col]
                path = ast.literal_eval(raw) if isinstance(raw, str) else raw
                dist = (len(path) - 1) if (path is not None and len(path) > 0) else np.inf
            except Exception:
                dist = np.inf

            if not np.isfinite(dist) or dist > rad_mask_hops:
                for cidx in rad_geom_cols.values():
                    A[atom_idx, cidx] = 0.0
                if "r_exist" in name2idx:
                    A[atom_idx, name2idx["r_exist"]] = 0.0
                if "a_exist" in name2idx:
                    A[atom_idx, name2idx["a_exist"]] = 0.0
                if "d_exist" in name2idx:
                    A[atom_idx, name2idx["d_exist"]] = 0.0

    for rxn in common_rxns:
        i0, i1 = idx_by_rxn_r1h[rxn], idx_by_rxn_r2h[rxn]
        dp0, dp1 = feat_data[0][i0], feat_data[1][i1]
        mol0 = dp0.mol if not isinstance(dp0.mol, tuple) else dp0.mol[0]
        mol1 = dp1.mol if not isinstance(dp1.mol, tuple) else dp1.mol[0]

        if use_atom_extras:
            sub0 = atom_extra_feats[
                (atom_extra_feats[rxn_col] == rxn) & (atom_extra_feats[moltype_col] == r1h_tag)
            ].sort_values(atom_index_col)
            sub1 = atom_extra_feats[
                (atom_extra_feats[rxn_col] == rxn) & (atom_extra_feats[moltype_col] == r2h_tag)
            ].sort_values(atom_index_col)
            if sub0.empty or sub1.empty:
                continue
            A0 = sub0[cols].to_numpy(dtype=np.float32, copy=True)
            A1 = sub1[cols].to_numpy(dtype=np.float32, copy=True)
            _impute_inplace(A0, cols)
            _impute_inplace(A1, cols)
            _mask_by_path_hops(A0, sub0, cols)
            _mask_by_path_hops(A1, sub1, cols)
            m0 = sub0["is_donor"].to_numpy(np.float32).reshape(-1, 1)
            m1 = sub1["is_acceptor"].to_numpy(np.float32).reshape(-1, 1)
        else:
            A0, A1 = dp0.V_f, dp1.V_f
            m0 = getattr(dp0, "V_d", None)
            m1 = getattr(dp1, "V_d", None)

        X_pair = None
        if gf is not None:
            gf0 = gf(dp0.mol).reshape(-1)
            gf1 = gf(dp1.mol).reshape(-1)
            X_pair = np.concatenate([gf0, gf1], axis=0).astype(np.float32)
            if x_d_shape == 0:
                x_d_shape = int(X_pair.shape[0])

        new0 = cpdata.MoleculeDatapoint(
            mol=dp0.mol,
            y=dp0.y,
            weight=dp0.weight,
            gt_mask=dp0.gt_mask,
            lt_mask=dp0.lt_mask,
            V_f=A0,
            E_f=dp0.E_f,
            V_d=m0,
            x_d=(X_pair if X_pair is not None else dp0.x_d),
            x_phase=dp0.x_phase,
            name=dp0.name,
        )
        new1 = cpdata.MoleculeDatapoint(
            mol=dp1.mol,
            y=dp1.y,
            weight=dp1.weight,
            gt_mask=dp1.gt_mask,
            lt_mask=dp1.lt_mask,
            V_f=A1,
            E_f=dp1.E_f,
            V_d=m1,
            x_d=(X_pair if X_pair is not None else dp1.x_d),
            x_phase=dp1.x_phase,
            name=dp1.name,
        )
        datapoints[0].append(new0)
        datapoints[1].append(new1)
        kept_idx.append(i0)

    return datapoints, kept_idx, x_d_shape


# ---------- bundle ----------
@dataclass
class DataBundle:
    attached_pair_dps: list
    donors_kept: List[Chem.Mol]
    acceptors_kept: List[Chem.Mol]
    featurizer: Any
    x_d_dim: int
    pair_group_keys: List[str]
    mode_cfg: Dict[str, Any]
    extras_mode: str
    rad_source: str


# ---------- public API: prepare once ----------
def prepare_data(
    sdf_path: str,
    target_csv: str,
    extras_mode: str,
    global_mode: str,  # "none"|"morgan_binary"|"morgan_count"|"rdkit2d_norm"
    morgan_bits: int = 2048,
    morgan_radius: int = 2,
    rad_dir: Optional[str] = None,
    rad_source: str = "path",
) -> DataBundle:
    extras_mode = canonicalize_extra_mode(extras_mode)
    global_mode = canonicalize_global_mode(global_mode)
    rad_source = canonicalize_rad_source(rad_source)
    mode_cfg = mode_settings(extras_mode)

    # 1) Featuriser builds pair datapoints
    feat_data = Featuriser(
        sdf_path,
        filter_rules=None,
        path=target_csv,
        set_col_index=False,
        target_col=["A_log10", "n", "Ea"],
        include_extra_features=False,
        pair_labels=("k_for (TST+T)", "k_rev (TST+T)"),
    )
    atom_extra = None
    if mode_cfg["use_extras"]:
        assert rad_dir and os.path.isdir(
            rad_dir
        ), "rad_dir required when selected extra_mode uses atom extras"
        preferred = (
            "atom_with_geom_feats_path.csv"
            if rad_source == "path"
            else "atom_with_geom_feats_rad.csv"
        )
        fallback = "atom_with_geom_feats_default.csv" if rad_source == "rad" else None
        rad_csv = os.path.join(rad_dir, preferred)
        if not os.path.isfile(rad_csv) and fallback is not None:
            legacy = os.path.join(rad_dir, fallback)
            if os.path.isfile(legacy):
                rad_csv = legacy
        if not os.path.isfile(rad_csv):
            raise FileNotFoundError(f"RAD CSV not found: {rad_csv}")
        atom_extra = pd.read_csv(rad_csv)

    attached_pair_dps, kept_idx, x_d_shape = attach_feats_to_dps_pairwise(
        feat_data,
        atom_extra,
        mode_cfg["cols"],
        global_mode=global_mode,
        morgan_bits=morgan_bits,
        morgan_radius=morgan_radius,
        rad_geom_cols={
            k: mode_cfg["cols"].index(k)
            for k in ("angle", "dihedral", "radius")
            if k in mode_cfg["cols"]
        },
        rad_mask_hops=mode_cfg["rad_mask"],
    )

    N = len(attached_pair_dps[0])
    donors = [_mol(attached_pair_dps[0][i].mol) for i in range(N)]
    accept = [_mol(attached_pair_dps[1][i].mol) for i in range(N)]

    pair_group_keys = [
        "__".join(sorted((ik14(attached_pair_dps[0][i].mol), ik14(attached_pair_dps[1][i].mol))))
        for i in range(N)
    ]

    extra_dim = (
        atom_extra_dim(mode_cfg["cols"], mode_cfg["rad_mode"]) if mode_cfg["use_extras"] else 0
    )
    featurizer = mode_cfg["featurizer_cls"](extra_atom_fdim=extra_dim)

    return DataBundle(
        attached_pair_dps=attached_pair_dps,
        donors_kept=donors,
        acceptors_kept=accept,
        featurizer=featurizer,
        x_d_dim=x_d_shape,
        pair_group_keys=pair_group_keys,
        mode_cfg=mode_cfg,
        extras_mode=extras_mode,
        rad_source=rad_source,
    )


# ---------- public API: build loaders per split ----------
def make_loaders(
    bundle: DataBundle,
    cfg: Dict[str, Any],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: Optional[List[int]] = None,
    seed: int = 42,
    *,
    preset_r_bounds: Optional[Tuple[float, float]] = None,
    preset_y_scaler: Optional[Any] = None,
    preset_vf_scaler: Optional[StandardScaler] = None,
    preset_xd_scaler: Optional[StandardScaler] = None,
):
    # 1) wrap into chemprop datasets
    train_dsets = [
        cpdata.MoleculeDataset(
            [bundle.attached_pair_dps[c][i] for i in train_idx], bundle.featurizer
        )
        for c in range(len(MOL_TYPES))
    ]
    val_dsets = [
        cpdata.MoleculeDataset([bundle.attached_pair_dps[c][i] for i in val_idx], bundle.featurizer)
        for c in range(len(MOL_TYPES))
    ]
    train_wrap = [ArrMoleculeDataset(d) for d in train_dsets]
    val_wrap = [ArrMoleculeDataset(d) for d in val_dsets]
    test_wrap = None
    test_mcd = None

    if test_idx is not None:
        test_dsets = [
            cpdata.MoleculeDataset(
                [bundle.attached_pair_dps[c][i] for i in test_idx], bundle.featurizer
            )
            for c in range(len(MOL_TYPES))
        ]
        test_wrap = [ArrMoleculeDataset(d) for d in test_dsets]
        test_mcd = ArrMulticomponentDataset(test_wrap)
    train_mcd = ArrMulticomponentDataset(train_wrap)
    val_mcd = ArrMulticomponentDataset(val_wrap)

    # 2) radius bounds from TRAIN (for RBF)
    extras_mode_cfg = canonicalize_extra_mode(cfg.get("extra_mode", bundle.extras_mode))
    mode_cfg = mode_settings(extras_mode_cfg)
    radius_idx = mode_cfg["cols"].index(RADIUS_COL) if RADIUS_COL in mode_cfg["cols"] else None
    dihedral_idx = (
        mode_cfg["cols"].index(DIHEDRAL_COL) if DIHEDRAL_COL in mode_cfg["cols"] else None
    )
    has_vf = any(
        bundle.attached_pair_dps[c][i].V_f is not None
        for i in train_idx
        for c in range(len(bundle.attached_pair_dps))
    )

    if preset_r_bounds is not None and mode_cfg["rad_mode"] and radius_idx is not None:
        r_min, r_max = preset_r_bounds
    elif has_vf and mode_cfg["rad_mode"] and radius_idx is not None:
        r_min, r_max = train_radius_bounds(
            bundle.attached_pair_dps, train_idx, extras_mode=extras_mode_cfg
        )
    else:
        r_min, r_max = (0.5, 8.0)

    # 3) register V_f transforms (RBF on radius; optional dihedral→sin/cos)
    if has_vf and mode_cfg["rad_mode"] and radius_idx is not None:
        wraps_for_transforms = [*train_wrap, *val_wrap]
        if test_wrap is not None:
            wraps_for_transforms.extend(test_wrap)

        for ds in wraps_for_transforms:
            ds.register_vf_transform(radius_idx, lambda r: rbf_expand(r, 16, r_min, r_max))
            if USE_DIHEDRAL_SIN_COS and dihedral_idx is not None:
                ds.register_vf_transform(dihedral_idx, dihedral_to_sin_cos)

    # 4) target scalers (fit on train, apply to val)
    cols_map = {
        0: StandardScaler(),  # A_log10_for
        1: StandardScaler(),  # n_for
        2: PowerTransformer(method="yeo-johnson", standardize=True),  # Ea_for
        3: StandardScaler(),  # A_log10_rev
        4: StandardScaler(),  # n_rev
        5: PowerTransformer(method="yeo-johnson", standardize=True),  # Ea_rev
    }
    if preset_y_scaler is not None:
        y_scaler = train_mcd.normalize_targets(scaler=preset_y_scaler, columns=cols_map)
    else:
        y_scaler = train_mcd.normalize_targets(columns=cols_map)
    val_mcd.normalize_targets(y_scaler, columns=cols_map)
    if test_mcd is not None:
        test_mcd.normalize_targets(y_scaler, columns=cols_map)

    # 5) input scalers
    vf_scaler_shared = None
    if has_vf:
        if preset_vf_scaler is not None:
            vf_scaler_shared = train_mcd.normalize_inputs_shared(
                key="V_f", columns_to_scale=None, scaler=preset_vf_scaler
            )
        else:
            vf_scaler_shared = train_mcd.normalize_inputs_shared(key="V_f", columns_to_scale=None)
        val_mcd.apply_shared_input_scaler(key="V_f", scaler=vf_scaler_shared)
        if test_mcd is not None:
            test_mcd.apply_shared_input_scaler(key="V_f", scaler=vf_scaler_shared)

    # Global X_d scaler: if present, decide normalization based on mode
    xd_scaler_shared = None
    # If you want: detect presence via shape on first batch OR via cfg flag
    if any(getattr(dp, "x_d", None) is not None for dp in bundle.attached_pair_dps[0]):
        if preset_xd_scaler is not None:
            xd_scaler_shared = train_mcd.normalize_inputs_shared(
                key="X_d", columns_to_scale=None, scaler=preset_xd_scaler
            )
        else:
            xd_scaler_shared = StandardScaler(with_mean=False, with_std=False)
            xd_scaler_shared = train_mcd.normalize_inputs_shared(
                key="X_d", columns_to_scale=None, scaler=xd_scaler_shared
            )
        val_mcd.apply_shared_input_scaler(key="X_d", scaler=xd_scaler_shared)
        if test_mcd is not None:
            test_mcd.apply_shared_input_scaler(key="X_d", scaler=xd_scaler_shared)

    # 6) loaders
    pin = True if torch.cuda.is_available() else False
    g_train = torch_generator(seed + 11)
    g_val = torch_generator(seed + 13)
    train_loader = build_loader_mc(
        train_mcd,
        batch_size=int(cfg.get("batch_size", 128)),
        generator=g_train,
        shuffle=True,
        num_workers=CP_NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = build_loader_mc(
        val_mcd,
        batch_size=max(1, int(cfg.get("batch_size", 128)) // 2),
        generator=g_val,
        shuffle=False,
        num_workers=CP_NUM_WORKERS,
        pin_memory=pin,
    )
    test_loader = None
    if test_mcd is not None:
        g_test = torch_generator(seed + 17)
        test_loader = build_loader_mc(
            test_mcd,
            batch_size=int(cfg.get("batch_size", 128)),
            generator=g_test,
            shuffle=False,
            num_workers=CP_NUM_WORKERS,
            pin_memory=pin,
        )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "y_scaler": y_scaler,
        "vf_scaler": vf_scaler_shared,
        "xd_scaler": xd_scaler_shared,
        "r_min": r_min,
        "r_max": r_max,
        "test_loader": test_loader,
        "train_dataset": train_mcd,
        "val_dataset": val_mcd,
        "test_dataset": test_mcd,
    }


def fit_global_normalizers(
    bundle: DataBundle, cfg: Dict[str, Any], train_idx: Sequence[int]
) -> Dict[str, Any]:
    """
    Fit target and input scalers once on the provided indices so they can be reused
    across inner splits and evaluations.
    """
    train_idx = list(map(int, train_idx))
    train_dsets = [
        cpdata.MoleculeDataset(
            [bundle.attached_pair_dps[c][i] for i in train_idx], bundle.featurizer
        )
        for c in range(len(MOL_TYPES))
    ]
    train_wrap = [ArrMoleculeDataset(d) for d in train_dsets]
    train_mcd = ArrMulticomponentDataset(train_wrap)

    extras_mode = canonicalize_extra_mode(cfg.get("extra_mode", bundle.extras_mode))
    mode_cfg = mode_settings(extras_mode)
    radius_idx = mode_cfg["cols"].index(RADIUS_COL) if RADIUS_COL in mode_cfg["cols"] else None
    dihedral_idx = (
        mode_cfg["cols"].index(DIHEDRAL_COL) if DIHEDRAL_COL in mode_cfg["cols"] else None
    )
    has_vf = any(
        bundle.attached_pair_dps[c][i].V_f is not None
        for i in train_idx
        for c in range(len(bundle.attached_pair_dps))
    )

    r_bounds = (
        train_radius_bounds(bundle.attached_pair_dps, train_idx, extras_mode=extras_mode)
        if (has_vf and mode_cfg["rad_mode"] and radius_idx is not None)
        else (0.5, 8.0)
    )
    if has_vf and mode_cfg["rad_mode"] and radius_idx is not None:
        for ds in train_wrap:
            ds.register_vf_transform(radius_idx, lambda r: rbf_expand(r, 16, *r_bounds))
            if USE_DIHEDRAL_SIN_COS and dihedral_idx is not None:
                ds.register_vf_transform(dihedral_idx, dihedral_to_sin_cos)

    cols_map = {
        0: StandardScaler(),
        1: StandardScaler(),
        2: PowerTransformer(method="yeo-johnson", standardize=True),
        3: StandardScaler(),
        4: StandardScaler(),
        5: PowerTransformer(method="yeo-johnson", standardize=True),
    }
    y_scaler = train_mcd.normalize_targets(columns=cols_map)
    vf_scaler = None
    if has_vf:
        vf_scaler = train_mcd.normalize_inputs_shared(key="V_f", columns_to_scale=None)

    xd_scaler = None
    if any(getattr(dp, "x_d", None) is not None for dp in bundle.attached_pair_dps[0]):
        xd_scaler = StandardScaler(with_mean=False, with_std=False)
        train_mcd.normalize_inputs_shared(key="X_d", columns_to_scale=None, scaler=xd_scaler)

    return {
        "y_scaler": y_scaler,
        "vf_scaler": vf_scaler,
        "xd_scaler": xd_scaler,
        "r_bounds": r_bounds,
    }


def compute_arrhenius_scalers_from_train(train_loader, y_scaler, temps: Sequence[float]):
    """
    Fit ln(k) standardizers (mean/scale per temperature) from TRAIN ONLY.

    Args:
        train_loader: your ArrMulticomponent train DataLoader
        y_scaler: ColumnTransformer with named transformers t0..t5 applied to targets
        temps: temperature grid used by the model (list/array of K)

    Returns:
        arr_mean_for, arr_scale_for, arr_mean_rev, arr_scale_rev
        (each is List[float] with length == len(temps))
    """
    ys = []
    for batch in train_loader:
        # batch.Y is already scaled by y_scaler; convert to numpy
        y = batch.Y.detach().cpu().numpy() if hasattr(batch.Y, "detach") else np.asarray(batch.Y)
        ys.append(y)
    ys = np.concatenate(ys, axis=0).astype(np.float64)

    # inverse-transform to physical params
    t0 = y_scaler.named_transformers_["t0"]  # A_log10_for
    t1 = y_scaler.named_transformers_["t1"]  # n_for
    t2 = y_scaler.named_transformers_["t2"]  # Ea_for (PowerTransformer)
    t3 = y_scaler.named_transformers_["t3"]  # A_log10_rev
    t4 = y_scaler.named_transformers_["t4"]  # n_rev
    t5 = y_scaler.named_transformers_["t5"]  # Ea_rev (PowerTransformer)

    A10_for = t0.inverse_transform(ys[:, 0:1])[:, 0]
    n_for = t1.inverse_transform(ys[:, 1:2])[:, 0]
    Ea_for = t2.inverse_transform(ys[:, 2:3])[:, 0]
    A10_rev = t3.inverse_transform(ys[:, 3:4])[:, 0]
    n_rev = t4.inverse_transform(ys[:, 4:5])[:, 0]
    Ea_rev = t5.inverse_transform(ys[:, 5:6])[:, 0]

    T = np.asarray(list(temps), dtype=np.float64)[None, :]  # (1, N_T)
    R = 8.31446261815324e-3  # kJ mol^-1 K^-1

    def make_lnk(A10, n, Ea):
        A = (10.0**A10)[:, None]
        n = n[:, None]
        Ea = Ea[:, None]
        k = A * (T**n) * np.exp(-Ea / (R * T))
        return np.log(k).astype(np.float64)  # (N_samples, N_T)

    lnk_for = make_lnk(A10_for, n_for, Ea_for)
    lnk_rev = make_lnk(A10_rev, n_rev, Ea_rev)

    sc_for = StandardScaler(with_mean=True, with_std=True).fit(lnk_for)
    sc_rev = StandardScaler(with_mean=True, with_std=True).fit(lnk_rev)
    return (
        sc_for.mean_.astype(float).tolist(),
        sc_for.scale_.astype(float).tolist(),
        sc_rev.mean_.astype(float).tolist(),
        sc_rev.scale_.astype(float).tolist(),
    )


def load_saved_split_indices(
    db_path: str, study_name: str, trial_id: int, fold_id: Optional[int] = None
) -> dict:
    """
    Convenience helper to retrieve stored split indices from the metrics DB.
    """
    from arrhenius.training.hpo.database_configs import fetch_split_indices

    return fetch_split_indices(db_path, study_name, trial_id, fold_id)


def loaders_from_saved_split(
    bundle: DataBundle,
    cfg: Dict[str, Any],
    db_path: str,
    study_name: str,
    trial_id: int,
    *,
    fold_id: int,
    train_key: str = "train",
    val_key: str = "val",
    test_key: Optional[str] = "test",
    seed: int = 42,
):
    """
    Build DataLoaders using the exact indices stored in the metrics database.

    Example usage:
        splits = loaders_from_saved_split(bundle, cfg, "final_metrics.sqlite",
                                          "arrhenius_final", trial_id, fold_id=0)
    """
    from arrhenius.training.hpo.database_configs import fetch_split_indices

    split_map = fetch_split_indices(db_path, study_name, trial_id, fold_id)
    if not split_map:
        raise ValueError(f"No splits found for trial_id={trial_id}, fold_id={fold_id}")

    def _get(name: Optional[str]) -> Optional[List[int]]:
        if name is None:
            return None
        if name not in split_map:
            raise KeyError(
                f"Split '{name}' not recorded for trial_id={trial_id}, fold_id={fold_id}"
            )
        return list(map(int, split_map[name]))

    train_idx = _get(train_key)
    val_idx = _get(val_key)
    test_idx = _get(test_key)

    if train_idx is None or val_idx is None:
        raise ValueError("Both train and val splits must be present to rebuild loaders.")

    return make_loaders(
        bundle=bundle, cfg=cfg, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed
    )
