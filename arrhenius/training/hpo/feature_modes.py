from __future__ import annotations

from typing import Any, Dict

from chemprop import featurizers

BASE_COLS = [
    "q_mull",
    "q_apt",
    "f_mag",
    "is_donor",
    "is_acceptor",
    "is_donor_H",
    "is_acceptor_H",
    "is_acceptor_neighbor",
    "is_donor_neighbor",
]

BASE_COLS_NO_Q = [
    "is_donor",
    "is_acceptor",
    "is_donor_H",
    "is_acceptor_H",
    "is_acceptor_neighbor",
    "is_donor_neighbor",
]

RAD_EXTRA_COLS = [
    "r_exist",
    "a_exist",
    "d_exist",
    "angle",
    "dihedral",
    "radius",
]

RAD_COLS = BASE_COLS + RAD_EXTRA_COLS
RAD_COLS_NO_Q = BASE_COLS_NO_Q + RAD_EXTRA_COLS

ANGLE_COL = "angle"
DIHEDRAL_COL = "dihedral"
RADIUS_COL = "radius"

RBF_D_COUNT = 16
RAD_MASK_HOPS = 4

EXTRA_MODE_ALIASES = {
    "none": "baseline",
    "path": "rad",
    "default": "rad",
}


def canonicalize_extra_mode(mode: str) -> str:
    raw = str(mode).strip().lower().replace("-", "_")
    return EXTRA_MODE_ALIASES.get(raw, raw)


def canonicalize_global_mode(mode: str) -> str:
    raw = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "rdkit2d": "rdkit2d_norm",
    }
    return aliases.get(raw, raw)


def canonicalize_rad_source(value: str) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "default": "rad",
    }
    out = aliases.get(raw, raw)
    if out not in {"path", "rad"}:
        raise ValueError(f"Unknown rad_source '{value}' (expected 'path' or 'rad').")
    return out


def atom_extra_dim(cols: list[str], rad_mode: bool) -> int:
    base = len(cols)
    if not rad_mode:
        return base
    add = 0
    if RADIUS_COL in cols:
        add += (RBF_D_COUNT - 1)
    if ANGLE_COL in cols:
        add += 1
    if DIHEDRAL_COL in cols:
        add += 1
    return base + add


def mode_settings(extra_mode: str) -> Dict[str, Any]:
    mode = canonicalize_extra_mode(extra_mode)

    if mode == "baseline":
        return {
            "canonical_mode": "baseline",
            "use_extras": False,
            "cols": [],
            "featurizer_cls": featurizers.SimpleMoleculeMolGraphFeaturizer,
            "rad_mode": False,
            "use_geom_edges": False,
            "rad_mask": None,
        }

    if mode == "geom_only":
        return {
            "canonical_mode": "geom_only",
            "use_extras": False,
            "cols": [],
            "featurizer_cls": featurizers.GeometryMolGraphFeaturizer,
            "rad_mode": False,
            "use_geom_edges": True,
            "rad_mask": None,
        }

    if mode == "local":
        return {
            "canonical_mode": "local",
            "use_extras": True,
            "cols": BASE_COLS,
            "featurizer_cls": featurizers.GeometryMolGraphFeaturizer,
            "rad_mode": False,
            "use_geom_edges": True,
            "rad_mask": None,
        }

    if mode == "atom":
        return {
            "canonical_mode": "atom",
            "use_extras": True,
            "cols": BASE_COLS,
            "featurizer_cls": featurizers.SimpleMoleculeMolGraphFeaturizer,
            "rad_mode": False,
            "use_geom_edges": False,
            "rad_mask": None,
        }

    if mode == "rad":
        return {
            "canonical_mode": "rad",
            "use_extras": True,
            "cols": RAD_COLS,
            "featurizer_cls": featurizers.SimpleMoleculeMolGraphFeaturizer,
            "rad_mode": True,
            "use_geom_edges": False,
            "rad_mask": None,
        }

    if mode == "rad_local":
        return {
            "canonical_mode": "rad_local",
            "use_extras": True,
            "cols": RAD_COLS,
            "featurizer_cls": featurizers.GeometryMolGraphFeaturizer,
            "rad_mode": True,
            "use_geom_edges": True,
            "rad_mask": RAD_MASK_HOPS,
        }

    if mode == "rad_local_noc":
        return {
            "canonical_mode": "rad_local_noc",
            "use_extras": True,
            "cols": RAD_COLS_NO_Q,
            "featurizer_cls": featurizers.GeometryMolGraphFeaturizer,
            "rad_mode": True,
            "use_geom_edges": True,
            "rad_mask": RAD_MASK_HOPS,
        }

    raise ValueError(f"Unknown extra_mode '{extra_mode}'")
