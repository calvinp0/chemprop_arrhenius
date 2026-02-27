from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _fixture_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[3]
    base = root / "arrhenius" / "tests" / "fixtures"
    sdf = base / "sdf" / "rmg_rxn_1538.sdf"
    r1h_log = base / "qm_logs" / "gaussian_pair_1538" / "r1h" / "input.log"
    r2h_log = base / "qm_logs" / "gaussian_pair_1538" / "r2h" / "input.log"
    return sdf, r1h_log, r2h_log


def _read_reaction_id_from_sdf(sdf_path: Path) -> str:
    Chem = pytest.importorskip("rdkit.Chem")
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        if mol.HasProp("reaction"):
            return str(mol.GetProp("reaction"))
    raise AssertionError(f"No reaction property found in {sdf_path}")


@pytest.mark.integration
def test_build_atom_with_geom_qm_e2e_on_fixture(tmp_path: Path) -> None:
    sdf, r1h_log, r2h_log = _fixture_paths()
    assert sdf.is_file(), f"Missing fixture: {sdf}"
    assert r1h_log.is_file(), f"Missing fixture: {r1h_log}"
    assert r2h_log.is_file(), f"Missing fixture: {r2h_log}"

    rxn_id = _read_reaction_id_from_sdf(sdf)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"rxn_id": rxn_id, "sdf_file": str(sdf), "r1h_log": str(r1h_log), "r2h_log": str(r2h_log)}]
    ).to_csv(manifest, index=False)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "arrhenius.preprocessing.build_atom_with_geom_qm",
        "--manifest-csv",
        str(manifest),
        "--output-dir",
        str(out_dir),
        "--atom-map-mode",
        "strict",
    ]
    subprocess.run(cmd, check=True)

    expected = {
        "geom_features_rad.csv",
        "geom_features_path.csv",
        "qm_atom_features.csv",
        "atom_with_geom_feats_rad.csv",
        "atom_with_geom_feats_path.csv",
        "qm_merge_report.json",
    }
    present = {p.name for p in out_dir.iterdir() if p.is_file()}
    missing = expected - present
    assert not missing, f"Missing expected outputs: {sorted(missing)}"

    rad_df = pd.read_csv(out_dir / "atom_with_geom_feats_rad.csv")
    path_df = pd.read_csv(out_dir / "atom_with_geom_feats_path.csv")

    for df in (rad_df, path_df):
        for col in ("rxn_id", "mol_type", "focus_atom_idx", "q_mull", "q_apt", "f_mag"):
            assert col in df.columns, f"Missing required column {col}"
        assert len(df) > 0
        assert int(df["q_mull"].isna().sum()) == 0
        assert int(df["q_apt"].isna().sum()) == 0
        assert int(df["f_mag"].isna().sum()) == 0

    report = json.loads((out_dir / "qm_merge_report.json").read_text())
    assert report["rows_manifest"] == 1
    assert report["rows_qm"] > 0
    assert report["merge_stats"]["rad"]["missing_q_mull"] == 0
    assert report["merge_stats"]["rad"]["missing_q_apt"] == 0
    assert report["merge_stats"]["rad"]["missing_f_mag"] == 0
    assert report["merge_stats"]["path"]["missing_q_mull"] == 0
    assert report["merge_stats"]["path"]["missing_q_apt"] == 0
    assert report["merge_stats"]["path"]["missing_f_mag"] == 0
