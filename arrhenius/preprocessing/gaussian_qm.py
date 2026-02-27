from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import math
import re

import numpy as np
import pandas as pd


@dataclass
class GaussianAtomQM:
    atom_index: int
    symbol: str
    q_mull: float
    q_apt: float
    f_mag: float


def _parse_last_orientation(lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Parse the last Gaussian orientation table (prefers Standard orientation).
    Returns columns: focus_atom_idx, atom_symbol, x, y, z
    """

    def _scan(kind: str) -> Optional[pd.DataFrame]:
        starts = [i for i, l in enumerate(lines) if kind in l]
        if not starts:
            return None
        start = starts[-1]
        rows = []
        i = start + 5
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                i += 1
                continue
            if s.startswith("-----"):
                if rows:
                    break
                i += 1
                continue
            parts = s.split()
            # Center, Atomic Number, Atomic Type, X, Y, Z
            if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                idx1 = int(parts[0])
                anum = int(parts[1])
                try:
                    from rdkit import Chem

                    sym = Chem.GetPeriodicTable().GetElementSymbol(anum)
                except Exception:
                    sym = str(anum)
                try:
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                except Exception:
                    i += 1
                    continue
                rows.append((idx1 - 1, sym, x, y, z))
            elif rows:
                break
            i += 1
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["focus_atom_idx", "atom_symbol", "x", "y", "z"])

    std = _scan("Standard orientation:")
    if std is not None:
        return std
    return _scan("Input orientation:")


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", errors="ignore") as f:
        return f.readlines()


def _parse_charge_block(lines: List[str], header_regex: str) -> Optional[pd.DataFrame]:
    header_re = re.compile(header_regex, flags=re.IGNORECASE)
    sum_re = re.compile(r"sum of .*charges", flags=re.IGNORECASE)

    start = None
    for i, line in enumerate(lines):
        if header_re.search(line):
            start = i + 1
    if start is None:
        return None

    rows = []
    i = start
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if sum_re.search(s):
            break
        parts = s.split()
        # Common patterns:
        # 1 C -0.1234
        # 1 C -0.1234 0.0000   (spin density extra column)
        if len(parts) >= 3 and parts[0].isdigit():
            idx1 = int(parts[0])
            sym = parts[1]
            try:
                val = float(parts[2].replace("D", "E"))
            except Exception:
                val = np.nan
            rows.append((idx1 - 1, sym, val))
        else:
            # Stop when clearly outside table.
            if len(rows) > 0 and not s[0].isdigit():
                break
        i += 1

    if not rows:
        return None
    return pd.DataFrame(rows, columns=["focus_atom_idx", "atom_symbol", "value"])


def _parse_last_force_block(lines: List[str]) -> Optional[pd.DataFrame]:
    # Gaussian block often starts with:
    # "Forces (Hartrees/Bohr)"
    # then header lines, then rows:
    # Center  Atomic  Forces (Hartrees/Bohr)
    # Number  Number  X   Y   Z
    starts = [i for i, l in enumerate(lines) if "Forces (Hartrees/Bohr)" in l]
    if not starts:
        return None

    start = starts[-1]
    rows = []
    for i in range(start + 1, len(lines)):
        s = lines[i].strip()
        if not s:
            continue
        if s.startswith("-----"):
            # end delimiter can appear before and after body; keep going unless already inside rows
            if rows:
                break
            continue

        parts = s.split()
        # Expect at least: center_idx atomic_num fx fy fz
        if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit():
            idx1 = int(parts[0])
            try:
                fx = float(parts[2].replace("D", "E"))
                fy = float(parts[3].replace("D", "E"))
                fz = float(parts[4].replace("D", "E"))
                fmag = math.sqrt(fx * fx + fy * fy + fz * fz)
            except Exception:
                fmag = np.nan
            rows.append((idx1 - 1, fmag))
        elif rows:
            break
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["focus_atom_idx", "f_mag"])


def parse_gaussian_qm(log_path: str) -> pd.DataFrame:
    """
    Parse atom-level QM fields from a Gaussian freq log.
    Returns columns:
      focus_atom_idx, atom_symbol, q_mull, q_apt, f_mag
    Missing sections are filled with NaN.
    """
    p = Path(log_path)
    lines = _read_lines(p)

    mull = _parse_charge_block(lines, r"^\s*Mulliken charges(?: and spin densities)?:\s*$")
    apt = _parse_charge_block(lines, r"^\s*APT charges:\s*$")
    frc = _parse_last_force_block(lines)
    xyz = _parse_last_orientation(lines)

    if mull is None and apt is None and frc is None and xyz is None:
        raise ValueError(f"No parseable atom-level QM blocks found in {p}")

    # Build index frame from any available source
    frames = [df for df in (mull, apt, frc, xyz) if df is not None]
    base = frames[0][["focus_atom_idx"]].copy()
    for df in frames[1:]:
        base = base.merge(
            df[["focus_atom_idx"]].drop_duplicates(), on="focus_atom_idx", how="outer"
        )
    base = base.sort_values("focus_atom_idx").reset_index(drop=True)

    if mull is not None:
        base = base.merge(
            mull.rename(columns={"value": "q_mull"})[["focus_atom_idx", "atom_symbol", "q_mull"]],
            on="focus_atom_idx",
            how="left",
        )
    else:
        base["atom_symbol"] = None
        base["q_mull"] = np.nan

    if apt is not None:
        base = base.merge(
            apt.rename(columns={"value": "q_apt"})[["focus_atom_idx", "q_apt"]],
            on="focus_atom_idx",
            how="left",
        )
    else:
        base["q_apt"] = np.nan

    if frc is not None:
        base = base.merge(frc, on="focus_atom_idx", how="left")
    else:
        base["f_mag"] = np.nan

    if xyz is not None:
        base = base.merge(xyz[["focus_atom_idx", "x", "y", "z"]], on="focus_atom_idx", how="left")
    else:
        base["x"] = np.nan
        base["y"] = np.nan
        base["z"] = np.nan

    return base[["focus_atom_idx", "atom_symbol", "q_mull", "q_apt", "f_mag", "x", "y", "z"]]
