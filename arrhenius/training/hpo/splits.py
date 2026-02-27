# run_hpo/splits.py
from hashlib import sha1
from typing import List, Sequence, Tuple

from arrhenius.splitters.k_stone import ks_make_split_indices
from arrhenius.splitters.random import random_grouped_split_indices


def build_outer_splits(
    donors: Sequence,
    acceptors: Sequence,
    pair_group_keys: Sequence,
    *,
    splitter: str = "kstone",
    k_folds: int = 3,
    seed: int = 42,
    distance_metric: str = "jaccard",
    joint_mode: str = "order-invariant",
    donor_weight: float = 0.5,
    p_norm: float = 2.0,
    n_bits: int = 2048,
    radius: int = 2,
) -> List[Tuple[List[int], List[int]]]:
    """
    Build outer CV splits using the requested splitter so that the same
    molecule grouping logic is respected across the entire workflow.

    Returns a list of (train_indices, holdout_indices) pairs.
    """
    n_total = len(pair_group_keys)
    if n_total == 0:
        raise ValueError("Cannot build outer splits with an empty dataset.")

    n_unique = len(set(pair_group_keys))
    if n_unique < 2:
        raise ValueError("At least two unique pair groups are required for CV.")

    k_folds = max(2, min(int(k_folds), n_unique))
    holdout_frac = 1.0 / k_folds
    train_frac = 1.0 - holdout_frac
    sizes = (train_frac, holdout_frac, 0.0)

    splitter = splitter.lower()

    common_kwargs = {
        "sizes": sizes,
        "seed": seed,
        "num_replicates": k_folds,
        "unordered_pairs": True,
        "pair_key_mode": "inchikey",
        "pair_group_keys": list(pair_group_keys),
    }

    data = (list(donors), list(acceptors))

    if splitter == "random":
        train_reps, holdout_reps, _ = random_grouped_split_indices(data, **common_kwargs)
    else:
        train_reps, holdout_reps, _ = ks_make_split_indices(
            data,
            distance_metric=distance_metric,
            joint_mode=joint_mode,
            donor_weight=donor_weight,
            p_norm=p_norm,
            fingerprint="morgan_fingerprint",
            fprints_hopts={"n_bits": int(n_bits), "radius": int(radius)},
            group_by_pair=True,
            keep_all_group_members=True,
            **common_kwargs,
        )

    splits: List[Tuple[List[int], List[int]]] = []
    for tr, holdout in zip(train_reps, holdout_reps, strict=False):
        splits.append((list(map(int, tr)), list(map(int, holdout))))

    return splits


def build_locked_holdout_split(
    donors: Sequence,
    acceptors: Sequence,
    pair_group_keys: Sequence,
    *,
    splitter: str = "kstone",
    test_frac: float = 0.15,
    seed: int = 42,
    distance_metric: str = "jaccard",
    joint_mode: str = "order-invariant",
    donor_weight: float = 0.5,
    p_norm: float = 2.0,
    n_bits: int = 2048,
    radius: int = 2,
) -> Tuple[List[int], List[int]]:
    """Build one deterministic group-aware train/test split for locked test usage."""
    if not (0.0 < float(test_frac) < 1.0):
        raise ValueError(f"test_frac must be in (0,1); got {test_frac}")
    n_total = len(pair_group_keys)
    if n_total == 0:
        raise ValueError("Cannot build locked split with an empty dataset.")
    n_unique = len(set(pair_group_keys))
    if n_unique < 2:
        raise ValueError("At least two unique pair groups are required for locked split.")

    sizes = (1.0 - float(test_frac), 0.0, float(test_frac))
    splitter = str(splitter).lower()
    common_kwargs = {
        "sizes": sizes,
        "seed": int(seed),
        "num_replicates": 1,
        "unordered_pairs": True,
        "pair_key_mode": "inchikey",
        "pair_group_keys": list(pair_group_keys),
    }
    data = (list(donors), list(acceptors))

    if splitter == "random":
        train_reps, _, test_reps = random_grouped_split_indices(data, **common_kwargs)
    else:
        train_reps, _, test_reps = ks_make_split_indices(
            data,
            distance_metric=distance_metric,
            joint_mode=joint_mode,
            donor_weight=donor_weight,
            p_norm=p_norm,
            fingerprint="morgan_fingerprint",
            fprints_hopts={"n_bits": int(n_bits), "radius": int(radius)},
            group_by_pair=True,
            keep_all_group_members=True,
            **common_kwargs,
        )

    train_idx = list(map(int, train_reps[0]))
    test_idx = list(map(int, test_reps[0]))
    return train_idx, test_idx


def splits_signature(outer_splits: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> str:
    """Stable signature for caching and duplicate detection."""
    parts = []
    for tr, val in outer_splits:
        tr_sig = ",".join(map(str, sorted(map(int, tr))))
        val_sig = ",".join(map(str, sorted(map(int, val))))
        parts.append(f"tr={tr_sig}|val={val_sig}")
    return sha1("|".join(parts).encode("utf-8")).hexdigest()
