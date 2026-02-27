from typing import List, Tuple, Sequence, Optional, Union, Literal, Hashable
import numpy as np
from sklearn.metrics import pairwise_distances
import warnings


from rdkit import Chem
from rdkit.Chem import inchi

# ---- AIMSim imports behind a safe shim ----
Molecule = None
LoadingError = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        from aimsim.chemical_datastructures import Molecule as _AIMSIM_Molecule
        from aimsim.exceptions import LoadingError as _AIMSIM_LoadingError
    Molecule = _AIMSIM_Molecule
    LoadingError = _AIMSIM_LoadingError
except Exception:
    warnings.warn("AIMSim is not installed. AIMSim features will be unavailable.")
    Molecule = None
    LoadingError = None
ArrayLikeMol = Union[str, "Chem.Mol"]


from collections import defaultdict


def _to_mol(x: ArrayLikeMol) -> Chem.Mol | None:
    if isinstance(x, Chem.Mol):
        return x
    if isinstance(x, str):
        return Chem.MolFromSmiles(x)
    return None


def _canon_key(
    m: ArrayLikeMol, mode: Literal["inchikey", "inchikey14", "smiles"] = "inchikey"
) -> str:
    mol = _to_mol(m)
    if mol is None:
        return ""
    if mode == "inchikey":
        return inchi.MolToInchiKey(mol)  # <— fix
    if mode == "inchikey14":
        return inchi.MolToInchiKey(mol)[:14]  # connectivity layer only
    # mode == "smiles"
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def _pair_key(
    d: ArrayLikeMol,
    a: ArrayLikeMol,
    *,
    unordered: bool = True,
    key_mode: Literal["inchikey", "smiles"] = "inchikey",
) -> tuple[str, str]:
    kd, ka = _canon_key(d, key_mode), _canon_key(a, key_mode)
    if unordered:
        return tuple(sorted((kd, ka)))
    return (kd, ka)


def _group_pairs(
    donors: Sequence[ArrayLikeMol],
    acceptors: Sequence[ArrayLikeMol],
    *,
    unordered: bool = True,
    key_mode: Literal["inchikey", "smiles", "inchikey14"] = "inchikey",
    pair_group_keys: Optional[List[Hashable]] = None,  # <— NEW
):
    groups: dict[Hashable, list[int]] = defaultdict(list)
    if pair_group_keys is not None:
        if len(pair_group_keys) != len(donors):
            raise ValueError("pair_group_keys must match data length.")
        for i, key in enumerate(pair_group_keys):
            groups[key].append(i)
    else:
        for i, (d, a) in enumerate(zip(donors, acceptors)):
            kd = _canon_key(d, key_mode)
            ka = _canon_key(a, key_mode)
            k = tuple(sorted((kd, ka))) if unordered else (kd, ka)
            groups[k].append(i)

    gkeys = list(groups.keys())
    rep_donors = [donors[groups[k][0]] for k in gkeys]
    rep_acceptors = [acceptors[groups[k][0]] for k in gkeys]
    return gkeys, groups, rep_donors, rep_acceptors


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def generalized_tanimoto_square(X: np.ndarray) -> np.ndarray:
    """
    Generalized Tanimoto/Jaccard on nonnegative vectors (handles bits, counts, floats).
    s = (x·y) / (||x||^2 + ||y||^2 - x·y);  D = 1 - s
    """
    X = np.asarray(X, dtype=float)
    G = X @ X.T
    sq = np.einsum("ij,ij->i", X, X)
    denom = sq[:, None] + sq[None, :] - G
    sim = _safe_div(G, denom)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    D[~np.isfinite(D)] = 0.0
    D[D < 0] = 0.0
    return 0.5 * (D + D.T)


def generalized_tanimoto_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cross generalized Tanimoto distances between rows of A and rows of B.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    G = A @ B.T
    sqA = np.einsum("ij,ij->i", A, A)
    sqB = np.einsum("ij,ij->i", B, B)
    denom = sqA[:, None] + sqB[None, :] - G
    sim = _safe_div(G, denom)
    D = 1.0 - sim
    D[~np.isfinite(D)] = 0.0
    D[D < 0] = 0.0
    return D


# Order invariant 2 component aggregation function
def order_invariant_two_component(
    D_AA: np.ndarray, D_BB: np.ndarray, D_AB: np.ndarray
) -> np.ndarray:
    aligned = D_AA + D_BB
    swapped = D_AB + D_AB.T  # D_BA
    D = np.minimum(aligned, swapped)
    D[~np.isfinite(D)] = 0.0
    D[D < 0] = 0.0
    return 0.5 * (D + D.T)


def aggregate_joint(
    D1: np.ndarray, D2: np.ndarray, mode: str, w: float = 0.5, p: float = 2.0
) -> np.ndarray:
    if D1.shape != D2.shape:
        raise ValueError("Joint aggregation requires matching shapes.")
    if mode == "mean":
        return w * D1 + (1.0 - w) * D2
    if mode == "max":
        return np.maximum(D1, D2)
    if mode == "p-norm":
        return ((w * (D1**p)) + ((1.0 - w) * (D2**p))) ** (1.0 / p)
    raise ValueError(f"Unknown joint mode: {mode}")


# Kennard Stone
def kennard_stone_order(D: np.ndarray, seed: int = 42) -> np.ndarray:
    D = np.array(D, dtype=float, copy=True)
    D = np.nan_to_num(D, posinf=np.inf, neginf=np.inf)
    D[D < 0] = 0.0
    D = 0.5 * (D + D.T)
    n = D.shape[0]
    np.fill_diagonal(D, -np.inf)

    i1, i2 = np.unravel_index(np.nanargmax(D), D.shape)
    order = np.full(n, -1, dtype=int)
    order[0], order[1] = i1, i2

    selected = np.zeros(n, dtype=bool)
    selected[[i1, i2]] = True
    min_dist = np.minimum(D[:, i1], D[:, i2])
    min_dist[selected] = -np.inf

    for t in range(2, n):
        m = np.max(min_dist)
        next_idx = int(np.flatnonzero(min_dist == m)[0])
        order[t] = next_idx
        selected[next_idx] = True
        np.minimum(min_dist, D[:, next_idx], out=min_dist)
        min_dist[selected] = -np.inf
    return order


# AIMSim Featurisation
def aimsim_featurize(
    molecules: Sequence[ArrayLikeMol],
    fingerprint: str = "morgan_fingerprint",
    fprints_hopts: Optional[dict] = None,
) -> np.ndarray:
    if Molecule is None:
        raise RuntimeError(
            "AIMSim is not installed. Install with `pip install aimsim` or `pip install astartes[molecules]`."
        )
    X = []
    fprints_hopts = fprints_hopts or {}
    for mol in molecules:
        try:
            if isinstance(mol, str):
                m = Molecule(mol_smiles=mol)
            else:
                m = Molecule(mol_graph=mol)
            m.descriptor.make_fingerprint(
                m.mol_graph, fingerprint, fingerprint_params=fprints_hopts
            )
            X.append(m.descriptor.to_numpy())
        except LoadingError as le:
            raise RuntimeError(f"Failed to featurize molecule: {mol!r}") from le
    return np.asarray(X, dtype=float)


class KSSplitterAIMSim:
    def __init__(
        self,
        distance_metric: Literal["jaccard", "tanimoto", "cosine", "euclidean"] = "jaccard",
        joint_mode: Literal["mean", "max", "p-norm", "concat", "order-invariant"] = "mean",
        donor_weight: float = 0.5,
        p_norm: float = 2.0,
        fingerprint: str = "morgan_fingerprint",
        fprints_hopts: Optional[dict] = None,
        order_invariant: bool = False,
        seed: int = 42,
        # --- NEW: grouping controls ---
        group_by_pair: bool = False,
        unordered_pairs: bool = True,
        pair_key_mode: Literal["inchikey", "smiles"] = "inchikey",
        keep_all_group_members: bool = True,  # if False, you effectively dedupe
        pair_group_keys: Optional[List[Hashable]] = None,  # if provided, use these as group keys
    ):
        self.seed = seed
        # fix name + aliasing 'tanimoto' to 'jaccard' for generalized tanimoto
        self.distance_metric = (
            "jaccard" if distance_metric in ("tanimoto", "jaccard") else distance_metric
        )
        # normalize joint mode spelling
        jm = "order-invariant" if order_invariant else joint_mode
        self.joint_mode = jm.replace("_", "-")
        self.donor_weight = donor_weight
        self.p_norm = p_norm
        self.fingerprint = fingerprint
        self.fprints_hopts = fprints_hopts or {}

        self.group_by_pair = group_by_pair
        self.unordered_pairs = unordered_pairs
        self.pair_key_mode = pair_key_mode
        self.keep_all_group_members = keep_all_group_members
        self.pair_group_keys = pair_group_keys

    def _build_D(self, X: np.ndarray) -> np.ndarray:
        if self.distance_metric == "jaccard":
            return generalized_tanimoto_square(X)
        D = pairwise_distances(X, metric=self.distance_metric)
        D[np.isnan(D)] = 0.0
        D[D < 0] = 0.0
        return 0.5 * (D + D.T)

    def _build_cross(self, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
        if self.distance_metric == "jaccard":
            return generalized_tanimoto_cross(XA, XB)
        D = pairwise_distances(XA, XB, metric=self.distance_metric)
        D[np.isnan(D)] = 0.0
        D[D < 0] = 0.0
        return D

    def _compute_D_single(self, mols: Sequence[ArrayLikeMol]) -> np.ndarray:
        X = aimsim_featurize(
            molecules=mols, fingerprint=self.fingerprint, fprints_hopts=self.fprints_hopts
        )
        return self._build_D(X)

    def _compute_D_pairs(
        self, donors: Sequence[ArrayLikeMol], acceptors: Sequence[ArrayLikeMol]
    ) -> np.ndarray:
        X_donors = aimsim_featurize(
            molecules=donors, fingerprint=self.fingerprint, fprints_hopts=self.fprints_hopts
        )
        X_acceptors = aimsim_featurize(
            molecules=acceptors, fingerprint=self.fingerprint, fprints_hopts=self.fprints_hopts
        )

        if self.joint_mode == "concat":
            X = np.concatenate((X_donors, X_acceptors), axis=1)
            return self._build_D(X)

        if self.joint_mode == "order-invariant":
            D_AA = self._build_D(X_donors)
            D_BB = self._build_D(X_acceptors)
            D_AB = self._build_cross(X_donors, X_acceptors)
            return order_invariant_two_component(D_AA, D_BB, D_AB)

        D1 = self._build_D(X_donors)
        D2 = self._build_D(X_acceptors)
        return aggregate_joint(D1, D2, mode=self.joint_mode, w=self.donor_weight, p=self.p_norm)

    def _normalize_input(
        self,
        data: Union[
            List[ArrayLikeMol],
            List[Tuple[ArrayLikeMol, ArrayLikeMol]],
            Tuple[List[ArrayLikeMol], List[ArrayLikeMol]],
        ],
    ) -> Tuple[str, int, Optional[List[ArrayLikeMol]], Optional[List[ArrayLikeMol]]]:
        # returns (kind, n, donors, acceptors) where kind in {"single","pairs"}
        if isinstance(data, tuple) and len(data) == 2:
            donors, acceptors = data
            if len(donors) != len(acceptors):
                raise ValueError("Donors and acceptors must be same length.")
            return "pairs", len(donors), list(donors), list(acceptors)
        if len(data) > 0 and isinstance(data[0], tuple):
            donors, acceptors = zip(*data)  # list of (d,a)
            donors, acceptors = list(donors), list(acceptors)
            if len(donors) != len(acceptors):
                raise ValueError("Mismatched pair lengths.")
            return "pairs", len(donors), donors, acceptors
        # single component
        return "single", len(data), list(data), None  # type: ignore

    def split_indices(
        self,
        data: Union[
            List[ArrayLikeMol],
            List[Tuple[ArrayLikeMol, ArrayLikeMol]],
            Tuple[List[ArrayLikeMol], List[ArrayLikeMol]],
        ],
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> Tuple[List[int], List[int], List[int], np.ndarray]:
        """
        Returns (train_idx, val_idx, test_idx, order)
        If group_by_pair=True on pairs, 'order' is the group order (expanded to one index per group).
        """
        kind, n, donors, acceptors = self._normalize_input(data)
        if n == 0:
            return [], [], [], np.array([], dtype=int)

        if kind == "pairs" and self.group_by_pair:
            # --- GROUPED MODE ---
            gkeys, groups, rep_donors, rep_acceptors = _group_pairs(
                donors,
                acceptors,
                unordered=self.unordered_pairs,
                key_mode=self.pair_key_mode,
                pair_group_keys=self.pair_group_keys,
            )

            # distance on group representatives
            D = self._compute_D_pairs(rep_donors, rep_acceptors)
            g_order = kennard_stone_order(D, seed=self.seed)  # indices into gkeys
            n_groups = len(gkeys)

            # slice by *groups*
            n_train_g = int(round(train_frac * n_groups))
            if val_frac == 0.0 or test_frac == 0.0:
                train_g = g_order[:n_train_g]
                val_g = np.array([], dtype=int)
                test_g = g_order[n_train_g:]
            else:
                n_val_g = int(round(val_frac * n_groups))
                train_g = g_order[:n_train_g]
                val_g = g_order[n_train_g : n_train_g + n_val_g]
                test_g = g_order[n_train_g + n_val_g :]

            # expand groups → row indices
            def expand(gs: np.ndarray) -> list[int]:
                if not self.keep_all_group_members:
                    return [groups[gkeys[g]][0] for g in gs]  # dedupe-on-split
                out: list[int] = []
                for g in gs:
                    out.extend(groups[gkeys[g]])
                return out

            train_idx = expand(train_g)
            val_idx = expand(val_g)
            test_idx = expand(test_g)

            # ‘order’ = representative per group in KS order
            order = np.array([groups[gkeys[g]][0] for g in g_order], dtype=int)
            return train_idx, val_idx, test_idx, order

        # --- original (non-grouped) path ---
        if kind == "pairs":
            D = self._compute_D_pairs(donors, acceptors)  # type: ignore
        else:
            D = self._compute_D_single(donors)  # type: ignore

        order = kennard_stone_order(D, seed=self.seed)
        n_train = int(round(train_frac * n))
        if val_frac == 0.0 or test_frac == 0.0:
            train_idx = order[:n_train].tolist()
            val_idx = []
            test_idx = order[n_train:].tolist()
        else:
            n_val = int(round(val_frac * n))
            train_idx = order[:n_train].tolist()
            val_idx = order[n_train : n_train + n_val].tolist()
            test_idx = order[n_train + n_val :].tolist()

        return train_idx, val_idx, test_idx, order


def ks_make_split_indices(
    data: Union[
        List[ArrayLikeMol],
        List[Tuple[ArrayLikeMol, ArrayLikeMol]],
        Tuple[List[ArrayLikeMol], List[ArrayLikeMol]],
    ],
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    num_replicates: int = 1,
    distance_metric: str = "jaccard",
    joint_mode: str = "mean",
    donor_weight: float = 0.5,
    p_norm: float = 2.0,
    fingerprint: str = "morgan_fingerprint",
    fprints_hopts: Optional[dict] = None,
    group_by_pair: bool = True,
    unordered_pairs: bool = True,
    pair_key_mode: str = "inchikey",
    keep_all_group_members: bool = True,
    pair_group_keys: Optional[List[Hashable]] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    train_reps: List[List[int]] = []
    val_reps: List[List[int]] = []
    test_reps: List[List[int]] = []

    for r in range(num_replicates):
        splitter = KSSplitterAIMSim(
            seed=seed + r,
            distance_metric=distance_metric,
            joint_mode=joint_mode,
            donor_weight=donor_weight,
            p_norm=p_norm,
            fingerprint=fingerprint,
            fprints_hopts=fprints_hopts or {},
            group_by_pair=group_by_pair,
            unordered_pairs=unordered_pairs,
            pair_key_mode=pair_key_mode,
            keep_all_group_members=keep_all_group_members,
            pair_group_keys=pair_group_keys,
        )

        tr, va, te, _ = splitter.split_indices(
            data, train_frac=sizes[0], val_frac=sizes[1], test_frac=sizes[2]
        )

        # Use the splitter’s expanded indices directly:
        if sizes[2] == 0.0 and sizes[1] > 0.0:
            # 2-way train/val (mirroring your earlier behavior)
            train_reps.append(tr)
            val_reps.append(te)  # splitter returns (train, val, test); in 2-way, it used test
            test_reps.append([])
        else:
            train_reps.append(tr)
            val_reps.append(va)
            test_reps.append(te)

    return train_reps, val_reps, test_reps
