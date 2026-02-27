from __future__ import annotations

import math
import random
from typing import Hashable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .k_stone import ArrayLikeMol, _group_pairs

PairData = Union[
    List[ArrayLikeMol],
    List[Tuple[ArrayLikeMol, ArrayLikeMol]],
    Tuple[List[ArrayLikeMol], List[ArrayLikeMol]],
]


def _extract_pair_lists(data: PairData) -> Tuple[Sequence[ArrayLikeMol], Sequence[ArrayLikeMol]]:
    """
    Normalise the `data` argument into (donors, acceptors) sequences.
    """
    if isinstance(data, tuple) and len(data) == 2:
        donors, acceptors = data
    elif isinstance(data, list) and len(data) == 2 and isinstance(data[0], list):
        donors, acceptors = data  # type: ignore[assignment]
    else:
        raise ValueError(
            "random_grouped_split_indices expects `data` as "
            "(donors, acceptors) or [donors, acceptors]."
        )

    if len(donors) != len(acceptors):
        raise ValueError("Donor and acceptor lists must have the same length.")

    return donors, acceptors


def _compute_split_counts(n_groups: int, sizes: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Convert fractional split sizes into integer group counts that sum to n_groups.
    """
    frac = np.asarray(sizes, dtype=float)
    if frac.shape != (3,):
        raise ValueError("sizes must be a tuple of three floats (train, val, test).")
    if not math.isclose(float(frac.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("sizes must sum to 1.0.")

    raw = frac * n_groups
    base = np.floor(raw).astype(int)
    remainder = n_groups - int(base.sum())

    if remainder > 0:
        frac_part = raw - base
        order = np.argsort(-frac_part)
        for idx in order[:remainder]:
            base[idx] += 1
    elif remainder < 0:
        frac_part = raw - base
        order = np.argsort(frac_part)
        for idx in order[:-remainder]:
            base[idx] -= 1

    base = np.clip(base, 0, None)
    diff = n_groups - int(base.sum())
    if diff != 0:
        base[0] += diff

    return int(base[0]), int(base[1]), int(base[2])


def random_grouped_split_indices(
    data: PairData,
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    num_replicates: int = 1,
    *,
    unordered_pairs: bool = True,
    pair_key_mode: Literal["inchikey", "smiles", "inchikey14"] = "inchikey",
    pair_group_keys: Optional[List[Hashable]] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Order-invariant random splitter matching the structure of ks_make_split_indices.

    Steps per replicate:
      1. Group unordered pairs so (A,B) ≡ (B,A) using `_group_pairs`.
      2. Select one representative per unordered pair.
      3. Shuffle representatives with the provided seed and slice into train/val/test.
      4. Expand each split to include every member of its group.
    """
    donors, acceptors = _extract_pair_lists(data)
    if pair_group_keys is not None and len(pair_group_keys) != len(donors):
        raise ValueError("pair_group_keys length must match number of pairs.")

    train_reps: List[List[int]] = []
    val_reps: List[List[int]] = []
    test_reps: List[List[int]] = []

    for r in range(num_replicates):
        rng = random.Random(seed + r)
        gkeys, groups, _, _ = _group_pairs(
            donors,
            acceptors,
            unordered=unordered_pairs,
            key_mode=pair_key_mode,
            pair_group_keys=pair_group_keys,
        )

        reps = list(gkeys)
        rng.shuffle(reps)

        n_train, n_val, n_test = _compute_split_counts(len(reps), sizes)

        train_keys = reps[:n_train]
        val_keys = reps[n_train : n_train + n_val]
        test_keys = reps[n_train + n_val :]

        def expand(keys: Sequence[Hashable]) -> List[int]:
            expanded: List[int] = []
            for key in keys:
                expanded.extend(groups[key])
            return expanded

        train_idx = expand(train_keys)
        val_idx = expand(val_keys)
        test_idx = expand(test_keys)

        if n_test == 0 and sizes[2] == 0.0 and sizes[1] > 0.0:
            test_idx = []

        train_reps.append(train_idx)
        val_reps.append(val_idx)
        test_reps.append(test_idx)

    return train_reps, val_reps, test_reps
