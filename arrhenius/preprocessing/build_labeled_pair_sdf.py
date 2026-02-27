from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


@dataclass
class GeometryData:
    symbols: list[str]
    coords: list[tuple[float, float, float]]
    charge: int | None = None


def _read_lines(path: Path) -> list[str]:
    with open(path, "r", errors="ignore") as f:
        return f.readlines()


def _parse_gaussian_charge(lines: list[str]) -> int | None:
    pat = re.compile(r"Charge\s*=\s*([+-]?\d+)\s+Multiplicity\s*=\s*([+-]?\d+)")
    charge = None
    for line in lines:
        m = pat.search(line)
        if m:
            charge = int(m.group(1))
    return charge


def _parse_gaussian_last_orientation(lines: list[str]) -> GeometryData:
    starts = [i for i, l in enumerate(lines) if "Standard orientation:" in l]
    if not starts:
        starts = [i for i, l in enumerate(lines) if "Input orientation:" in l]
    if not starts:
        raise ValueError("Could not find Gaussian orientation block.")

    pt = Chem.GetPeriodicTable()
    start = starts[-1]
    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    i = start + 5
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("-----"):
            if symbols:
                break
            i += 1
            continue
        parts = s.split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            anum = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            symbols.append(pt.GetElementSymbol(anum))
            coords.append((x, y, z))
        elif symbols:
            break
        i += 1

    if not symbols:
        raise ValueError("Could not parse atoms from Gaussian orientation block.")

    return GeometryData(symbols=symbols, coords=coords, charge=_parse_gaussian_charge(lines))


def _parse_orca_charge(lines: list[str]) -> int | None:
    patterns = [
        re.compile(r"Total Charge\s*[:=]\s*([+-]?\d+)"),
        re.compile(r"\|\s*Total Charge\s*\|\s*([+-]?\d+)\s*\|"),
    ]
    charge = None
    for line in lines:
        for pat in patterns:
            m = pat.search(line)
            if m:
                charge = int(m.group(1))
    return charge


def _parse_orca_last_coords(lines: list[str]) -> GeometryData:
    starts = [i for i, l in enumerate(lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in l.upper()]
    if not starts:
        raise ValueError("Could not find ORCA coordinate block.")

    start = starts[-1]
    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    i = start + 1
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            if symbols:
                break
            i += 1
            continue
        if set(s) <= {"-"}:
            i += 1
            continue
        parts = s.split()
        if len(parts) >= 4 and re.fullmatch(r"[A-Za-z]{1,2}", parts[0]):
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            symbols.append(sym)
            coords.append((x, y, z))
        elif symbols:
            break
        i += 1

    if not symbols:
        raise ValueError("Could not parse atoms from ORCA coordinate block.")

    return GeometryData(symbols=symbols, coords=coords, charge=_parse_orca_charge(lines))


def _parse_log(path: Path, fmt: str) -> GeometryData:
    lines = _read_lines(path)
    if fmt == "gaussian":
        return _parse_gaussian_last_orientation(lines)
    if fmt == "orca":
        return _parse_orca_last_coords(lines)
    raise ValueError(f"Unsupported format: {fmt}")


def _xyz_block(symbols: list[str], coords: list[tuple[float, float, float]]) -> str:
    lines = [str(len(symbols)), "generated from optimization log"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def _build_mol_from_geometry(data: GeometryData, charge_override: int | None = None) -> Chem.Mol:
    xyz = _xyz_block(data.symbols, data.coords)
    mol = Chem.MolFromXYZBlock(xyz)
    if mol is None:
        raise ValueError("RDKit failed to create molecule from XYZ block.")

    charge = data.charge if charge_override is None else charge_override
    if charge is None:
        charge = 0

    work = Chem.Mol(mol)
    try:
        rdDetermineBonds.DetermineBonds(work, charge=charge)
    except Exception:
        rdDetermineBonds.DetermineConnectivity(work)
    Chem.SanitizeMol(work)
    return work


def _rdkit_atom_type(atom: Chem.Atom) -> str:
    if atom.GetAtomicNum() == 1:
        return "H0"
    return atom.GetSymbol()


def _validate_index(name: str, idx: int | None, n_atoms: int) -> None:
    if idx is None:
        return
    if idx < 0 or idx >= n_atoms:
        raise ValueError(f"{name} index {idx} is out of range for molecule with {n_atoms} atoms.")


def _build_mol_properties(
    mol: Chem.Mol,
    *,
    donor_idx: int | None,
    acceptor_idx: int | None,
    h_idx: int | None,
    h_label: str,
) -> dict[str, dict[str, str]]:
    n_atoms = mol.GetNumAtoms()
    _validate_index("donor", donor_idx, n_atoms)
    _validate_index("acceptor", acceptor_idx, n_atoms)
    _validate_index("hydrogen", h_idx, n_atoms)

    out: dict[str, dict[str, str]] = {}
    if donor_idx is not None:
        atom = mol.GetAtomWithIdx(donor_idx)
        out[str(donor_idx)] = {"label": "donator", "atom_type": _rdkit_atom_type(atom)}
    if acceptor_idx is not None:
        atom = mol.GetAtomWithIdx(acceptor_idx)
        out[str(acceptor_idx)] = {"label": "acceptor", "atom_type": _rdkit_atom_type(atom)}
    if h_idx is not None:
        atom = mol.GetAtomWithIdx(h_idx)
        out[str(h_idx)] = {"label": h_label, "atom_type": _rdkit_atom_type(atom)}
    return out


def _attach_properties(
    mol: Chem.Mol, *, reaction_id: str, mol_type: str, mol_properties: dict[str, dict[str, str]]
) -> Chem.Mol:
    work = Chem.Mol(mol)
    work.SetProp("reaction", reaction_id)
    work.SetProp("type", mol_type)
    work.SetProp("mol_properties", json.dumps(mol_properties, separators=(",", ": ")))
    return work


def _write_sdf(path: Path, mols: Iterable[Chem.Mol]) -> None:
    writer = Chem.SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a 2-molecule SDF (r1h+r2h) from Gaussian/ORCA optimization logs."
    )
    p.add_argument("--r1h-log", required=True)
    p.add_argument("--r1h-format", choices=["gaussian", "orca"], required=True)
    p.add_argument("--r1h-charge", type=int, default=None)
    p.add_argument("--r1h-donor-idx", type=int, default=None)
    p.add_argument("--r1h-acceptor-idx", type=int, default=None)
    p.add_argument("--r1h-h-idx", type=int, required=True)

    p.add_argument("--r2h-log", required=True)
    p.add_argument("--r2h-format", choices=["gaussian", "orca"], required=True)
    p.add_argument("--r2h-charge", type=int, default=None)
    p.add_argument("--r2h-donor-idx", type=int, default=None)
    p.add_argument("--r2h-acceptor-idx", type=int, default=None)
    p.add_argument("--r2h-h-idx", type=int, required=True)

    p.add_argument("--reaction-id", required=True, help="Value written to SDF 'reaction' property.")
    p.add_argument("--output-sdf", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    r1h_data = _parse_log(Path(args.r1h_log), args.r1h_format)
    r2h_data = _parse_log(Path(args.r2h_log), args.r2h_format)

    r1h_mol = _build_mol_from_geometry(r1h_data, charge_override=args.r1h_charge)
    r2h_mol = _build_mol_from_geometry(r2h_data, charge_override=args.r2h_charge)

    r1h_props = _build_mol_properties(
        r1h_mol,
        donor_idx=args.r1h_donor_idx,
        acceptor_idx=args.r1h_acceptor_idx,
        h_idx=args.r1h_h_idx,
        h_label="d_hydrogen",
    )
    r2h_props = _build_mol_properties(
        r2h_mol,
        donor_idx=args.r2h_donor_idx,
        acceptor_idx=args.r2h_acceptor_idx,
        h_idx=args.r2h_h_idx,
        h_label="a_hydrogen",
    )

    out_r1h = _attach_properties(
        r1h_mol, reaction_id=args.reaction_id, mol_type="r1h", mol_properties=r1h_props
    )
    out_r2h = _attach_properties(
        r2h_mol, reaction_id=args.reaction_id, mol_type="r2h", mol_properties=r2h_props
    )

    out_path = Path(args.output_sdf).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_sdf(out_path, [out_r1h, out_r2h])
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
