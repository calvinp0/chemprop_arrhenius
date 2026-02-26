from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from run_hpo.feature_modes import canonicalize_extra_mode, canonicalize_global_mode, mode_settings
from run_hpo.feature_modes import canonicalize_rad_source


class DataSpecError(ValueError):
    pass


def _pick(src: Dict[str, Any], nested: str, flat: str, default: Any = None) -> Any:
    if nested in src:
        return src[nested]
    return src.get(flat, default)


def load_data_spec(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise DataSpecError(f"Spec file not found: {p}")

    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise DataSpecError("Spec must be a YAML mapping/object.")

    paths = raw.get("paths", {}) if isinstance(raw.get("paths"), dict) else {}
    modes = raw.get("modes", {}) if isinstance(raw.get("modes"), dict) else {}
    schema = raw.get("schema", {}) if isinstance(raw.get("schema"), dict) else {}

    extra_mode = canonicalize_extra_mode(_pick(modes, "extra_mode", "extra_mode", "baseline"))
    global_mode = canonicalize_global_mode(_pick(modes, "global_mode", "global_mode", "none"))
    try:
        rad_source = canonicalize_rad_source(_pick(modes, "rad_source", "rad_source", "path"))
    except Exception as e:
        raise DataSpecError(str(e)) from e

    mode_cfg = mode_settings(extra_mode)

    spec: Dict[str, Any] = {
        "version": int(raw.get("version", 1)),
        "paths": {
            "sdf_path": _pick(paths, "sdf_path", "sdf_path"),
            "target_csv": _pick(paths, "target_csv", "target_csv"),
            "rad_dir": _pick(paths, "rad_dir", "rad_dir"),
        },
        "modes": {
            "extra_mode": extra_mode,
            "global_mode": global_mode,
            "rad_source": rad_source,
            "morgan_bits": int(_pick(modes, "morgan_bits", "morgan_bits", 2048)),
            "morgan_radius": int(_pick(modes, "morgan_radius", "morgan_radius", 2)),
        },
        "schema": {
            "target_columns": list(_pick(schema, "target_columns", "target_columns", ["A_log10", "n", "Ea"])),
            "target_rxn_col": str(_pick(schema, "target_rxn_col", "target_rxn_col", "rxn")),
            "target_label_col": str(_pick(schema, "target_label_col", "target_label_col", "label")),
            "target_forward_label": str(
                _pick(schema, "target_forward_label", "target_forward_label", "k_for (TST+T)")
            ),
            "target_reverse_label": str(
                _pick(schema, "target_reverse_label", "target_reverse_label", "k_rev (TST+T)")
            ),
            "rxn_id_col": str(_pick(schema, "rxn_id_col", "rxn_id_col", "rxn_id")),
            "mol_type_col": str(_pick(schema, "mol_type_col", "mol_type_col", "mol_type")),
            "atom_index_col": str(_pick(schema, "atom_index_col", "atom_index_col", "focus_atom_idx")),
            "shortest_path_col": str(_pick(schema, "shortest_path_col", "shortest_path_col", "shortest_path")),
            "donor_tag": str(_pick(schema, "donor_tag", "donor_tag", "r1h")),
            "acceptor_tag": str(_pick(schema, "acceptor_tag", "acceptor_tag", "r2h")),
        },
        "mode_cfg": mode_cfg,
    }

    return spec
