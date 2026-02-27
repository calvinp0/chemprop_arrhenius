import os
from pathlib import Path
from cProfile import label
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import pandas as pd
import ast
import argparse
from pathlib import Path


def get_distance(coords, i, j):
    return np.linalg.norm(coords[i] - coords[j])


def get_angle(coords, i, j, k, eps=1e-12):
    # Angle at j (i-j-k)
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return np.nan
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def get_dihedral(coords, i, j, k, l, eps=1e-12):
    # Dihedral angle between i-j-k-l
    p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[l]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.linalg.norm(b1)
    if n1 < eps:
        return np.nan
    b1 = b1 / n1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < eps or nw < eps:
        return np.nan
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def get_mol_coords(mol):
    """Returns (num_atoms, 3) array of atom positions from the first conformer."""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    zcoords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    return zcoords


def safe_shortest_path(mol, i, j):
    if i == j:
        return [i]
    # RDKit will also blow up if i/j are out of bounds; be explicit
    n = mol.GetNumAtoms()
    if not (0 <= i < n and 0 <= j < n):
        return []
    return list(rdmolops.GetShortestPath(mol, i, j))


def _get_role_indices(label):
    donor_idx = None
    acceptor_idx = None
    ref_H_idx = None
    for k, v in label.items():
        idx = int(k)
        role = v.get("label")
        if role == "donator":
            donor_idx = idx
        elif role == "acceptor":
            acceptor_idx = idx
        elif role in ("d_hydrogen", "a_hydrogen"):
            ref_H_idx = idx
    return ref_H_idx, donor_idx, acceptor_idx


def _heavy_neighbors_sorted(mol, idx, exclude=None):
    exclude = exclude or set()
    atom = mol.GetAtomWithIdx(idx)
    cand = []
    for nbr in atom.GetNeighbors():
        j = nbr.GetIdx()
        if j in exclude or nbr.GetAtomicNum() == 1:
            continue
        bond = mol.GetBondBetweenAtoms(idx, j)
        cand.append((j, _bond_order(bond), nbr.GetAtomicNum(), -j))
    # Highest bond order → higher Z → lowest index (deterministic)
    cand.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [j for j, *_ in cand]


def _bond_order(b):
    return b.GetBondTypeAsDouble() if b is not None else 0.0


def _choose_pivot1(mol, ref_idx):
    """Choose a neighbor of ref_idx as pivot1 using deterministic rules."""
    ref = mol.GetAtomWithIdx(ref_idx)
    candidates = []
    for nbr in ref.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(ref_idx, nbr.GetIdx())
        candidates.append(
            (
                nbr.GetIdx(),
                nbr.GetAtomicNum() > 1,  # prefer heavy
                _bond_order(bond),
                nbr.GetAtomicNum(),
                -nbr.GetIdx(),  # lower idx wins after reversing
            )
        )
    if not candidates:
        return None
    # sort by the tuple keys descending except the last (idx reversed)
    candidates.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    return candidates[0][0]


def _choose_role_pivot1(mol, ref_H_idx, donor_idx, acceptor_idx):
    """
    Choose pivot1 strictly as donor or acceptor.
    Preference: bonded to ref_H > higher Z > lower idx (deterministic).
    Returns None if neither donor nor acceptor exists.
    """
    cands = []
    for idx in (donor_idx, acceptor_idx):
        if idx is None:
            continue
        bonded = mol.GetBondBetweenAtoms(ref_H_idx, idx) is not None
        Z = mol.GetAtomWithIdx(idx).GetAtomicNum()
        cands.append((idx, bonded, Z, -idx))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
    return cands[0][0]


def _choose_pivot2(mol, ref_idx, pivot1_idx):
    """Choose a neighbor of pivot1 (not ref_idx) as pivot2 with same rules."""
    p1 = mol.GetAtomWithIdx(pivot1_idx)
    candidates = []
    for nbr in p1.GetNeighbors():
        if nbr.GetIdx() == ref_idx:
            continue
        bond = mol.GetBondBetweenAtoms(pivot1_idx, nbr.GetIdx())
        candidates.append(
            (
                nbr.GetIdx(),
                nbr.GetAtomicNum() > 1,
                _bond_order(bond),
                nbr.GetAtomicNum(),
                -nbr.GetIdx(),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    return candidates[0][0]


def _angle_fallback(coords, focus_idx, ref_idx, pivot1_idx):
    # angle at ref between focus and pivot1
    return get_angle(coords, focus_idx, ref_idx, pivot1_idx)


def _dihedral_fallback(coords, focus_idx, ref_idx, pivot1_idx, pivot2_idx):
    # dihedral: focus-ref-pivot1-pivot2
    return get_dihedral(coords, focus_idx, ref_idx, pivot1_idx, pivot2_idx)


def _resolve_output(base: str, tag: str) -> str:
    p = Path(base)

    # "Directory mode" if:
    #  - ends with a path separator, OR
    #  - exists and is a directory, OR
    #  - has no file extension (heuristic — treat as dir)
    is_dir_hint = str(base).endswith(os.sep) or (p.exists() and p.is_dir()) or (p.suffix == "")

    if is_dir_hint:
        dirp = p
        dirp.mkdir(parents=True, exist_ok=True)
        base_name = dirp.name or "features"
        return str(dirp / f"{base_name}_{tag}.csv")

    # "File mode": ensure .csv, then append _{tag} before extension
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    return str(p.with_name(f"{p.stem}_{tag}{p.suffix}"))


def compute_atomfeats(
    mol,
    label,
    reference_labels=("d_hydrogen", "a_hydrogen", "acceptor"),
    path_only=False,
    hybrid=False,
):
    # Assume mol has a conformer and a valid reference
    num_atoms = mol.GetNumAtoms()
    coords = get_mol_coords(mol)

    ref_H_idx, donor_idx, acceptor_idx = _get_role_indices(label)
    if ref_H_idx is None:
        raise ValueError("Missing labeled hydrogen (d_hydrogen/a_hydrogen)")
    reference_idx = ref_H_idx  # path computations reference the labeled H

    # role flags
    atom_roles = {
        i: {
            "is_donor": 0,
            "is_acceptor": 0,
            "is_donor_H": 0,
            "is_acceptor_H": 0,
            "is_acceptor_neighbor": 0,
            "is_donor_neighbor": 0,
        }
        for i in range(num_atoms)
    }

    for k, v in label.items():
        idx = int(k)
        role = v.get("label")
        if role == "donator":
            atom_roles[idx]["is_donor"] = 1
        elif role == "acceptor":
            atom_roles[idx]["is_acceptor"] = 1
        elif role == "d_hydrogen":
            atom_roles[idx]["is_donor_H"] = 1
        elif role == "a_hydrogen":
            atom_roles[idx]["is_acceptor_H"] = 1

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom_roles[i]["is_donor"]:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() != "H":
                    atom_roles[nbr.GetIdx()]["is_donor_neighbor"] = 1
        if atom_roles[i]["is_acceptor"]:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() != "H":
                    atom_roles[nbr.GetIdx()]["is_acceptor_neighbor"] = 1

    # strict: compute reference index here (caller already validated; keep local copy)
    ref_indices = [int(k) for k, v in label.items() if v.get("label") in reference_labels]
    reference_idx = ref_indices[0]

    feat_records = []
    for target_idx in range(num_atoms):
        role_flags = atom_roles[target_idx]
        record = {"atom_idx": target_idx, "features": None, **role_flags}
        record["shortest_path"] = None
        rad_used_path = []
        ang_used_path = []
        dih_used_path = []

        if target_idx == reference_idx:
            record["shortest_path"] = [target_idx]
            record["features"] = [np.nan, np.nan, np.nan, 0, 0, 0]
            record["radius_path"] = []
            record["angle_path"] = []
            record["dihedral_path"] = []
            feat_records.append(record)
            continue

        path = safe_shortest_path(mol, reference_idx, target_idx)
        plen = len(path)

        # initialize always to avoid UnboundLocalError
        radius = np.nan
        angle = np.nan
        dihedral = np.nan
        r_flag = a_flag = d_flag = 0

        if path_only:
            # strict path-only
            if plen >= 2:
                radius = get_distance(coords, reference_idx, target_idx)
                r_flag = 1
                rad_used_path = [reference_idx, target_idx]
            if plen >= 3:
                angle = get_angle(coords, reference_idx, path[1], target_idx)
                a_flag = 1
                ang_used_path = [reference_idx, path[1], target_idx]
            if plen >= 4:
                dihedral = get_dihedral(coords, reference_idx, path[1], path[2], target_idx)
                d_flag = 1
                dih_used_path = [reference_idx, path[1], path[2], target_idx]

        elif hybrid:
            # path first, then ref-side fallbacks
            radius = get_distance(coords, reference_idx, target_idx)
            r_flag = 1
            rad_used_path = [reference_idx, target_idx]
            if plen >= 3:
                angle = get_angle(coords, reference_idx, path[1], target_idx)
                a_flag = 1
                ang_used_path = [reference_idx, path[1], target_idx]
            if plen >= 4:
                dihedral = get_dihedral(coords, reference_idx, path[1], path[2], target_idx)
                d_flag = 1
                dih_used_path = [reference_idx, path[1], path[2], target_idx]

            if a_flag == 0:
                pivot1 = _choose_pivot1(mol, reference_idx)
                if pivot1 is not None:
                    angle = _angle_fallback(coords, target_idx, reference_idx, pivot1)
                    a_flag = 2
                    ang_used_path = [reference_idx, pivot1, target_idx]
            if d_flag == 0:
                pivot1 = _choose_pivot1(mol, reference_idx)
                if pivot1 is not None:
                    pivot2 = _choose_pivot2(mol, reference_idx, pivot1)
                    if pivot2 is not None:
                        dihedral = _dihedral_fallback(
                            coords, target_idx, reference_idx, pivot1, pivot2
                        )
                        d_flag = 2
                        dih_used_path = [reference_idx, pivot1, target_idx, pivot2]

        else:
            # DEFAULT: role-geometry with your strict rules
            # 1) radius: always ref_H ↔ focus (no connectivity required)
            radius = get_distance(coords, ref_H_idx, target_idx)
            r_flag = 3
            rad_used_path = [ref_H_idx, target_idx]

            # 2) pick pivot1 strictly as donor/acceptor (no heavy-neighbor fallback here)
            pivot1 = _choose_role_pivot1(mol, ref_H_idx, donor_idx, acceptor_idx)

            # ANGLE: (ref_H, pivot1, focus), but forbidden if focus is donor/acceptor
            angle = np.nan
            a_flag = 0
            ang_used_path = []
            if pivot1 is not None and target_idx not in (donor_idx, acceptor_idx):
                angle = get_angle(coords, ref_H_idx, pivot1, target_idx)  # angle at pivot1
                a_flag = 3
                ang_used_path = [ref_H_idx, pivot1, target_idx]

            # 3) pick pivot2 = heavy neighbor of pivot1 excluding ref_H
            dihedral = np.nan
            d_flag = 0
            dih_used_path = []
            pivot2 = None
            if pivot1 is not None:
                p1_heavy = _heavy_neighbors_sorted(mol, pivot1, exclude={ref_H_idx})
                if p1_heavy:
                    pivot2 = p1_heavy[0]

            # DIHEDRAL: (ref_H, pivot1, pivot2, focus), forbidden if focus is pivot1 or pivot2
            if pivot1 is not None and pivot2 is not None and target_idx not in (pivot1, pivot2):
                dihedral = get_dihedral(coords, ref_H_idx, pivot1, pivot2, target_idx)
                d_flag = 3
                dih_used_path = [ref_H_idx, pivot1, pivot2, target_idx]

            # record graph path for info only
            record["shortest_path"] = safe_shortest_path(mol, ref_H_idx, target_idx)

        if record["shortest_path"] is None:
            record["shortest_path"] = safe_shortest_path(mol, reference_idx, target_idx)

        record["radius_path"] = rad_used_path
        record["angle_path"] = ang_used_path
        record["dihedral_path"] = dih_used_path

        record["features"] = [radius, angle, dihedral, r_flag, a_flag, d_flag]
        feat_records.append(record)

    return feat_records


def all_z_zero(mol):
    conf = mol.GetConformer()
    z = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    return all(abs(val) < 1e-8 for val in z)


def get_sdf_file_list(directory):
    """Returns a list of SDF files in the given directory."""
    import os

    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".sdf")]


def _to_units(x, units):
    if x is None or not np.isfinite(x):
        return np.nan
    if units == "degree":
        return float(np.degrees(x))
    return float(x)  # radians


def create_atom_feat_RAD_table(
    sdf_file_list,
    *,
    reference_labels=None,
    path_only=False,
    hybrid=False,
    angle_units="radian",
    dihedral_units="radian",
    output_csv="all_sdf_features.csv",
    mode_tag=None,
):
    """
    Create a table of atom features (radius, angle, dihedral) from SDF files.
    """
    all_feat_rows = []

    for sdf_file in sdf_file_list:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        for idx, mol in enumerate(suppl):
            if mol is None:
                continue
            rxn_id = mol.GetProp("reaction") if mol.HasProp("reaction") else "unknown"
            mol_type = mol.GetProp("type") if mol.HasProp("type") else "unknown"
            if mol_type not in ["r1h", "r2h"]:
                continue

            if all_z_zero(mol):
                print(f"Flat Z in: {sdf_file}, molecule #{idx}, rxn_id={rxn_id}, type={mol_type}")

            label = ast.literal_eval(mol.GetProp("mol_properties"))
            feat_records = compute_atomfeats(
                mol, label, reference_labels=reference_labels, path_only=path_only, hybrid=hybrid
            )

            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]  # avoid recomputing
            for rec in feat_records:
                focus_atom = rec["atom_idx"]
                r, a, d, r_f, a_f, d_f = rec["features"]
                row = {
                    "rxn_id": rxn_id,
                    "mol_type": mol_type,
                    "focus_atom_idx": focus_atom,
                    "shortest_path": str(rec["shortest_path"]),
                    "radius_path": str(rec["radius_path"]),
                    "angle_path": str(rec["angle_path"]),
                    "dihedral_path": str(rec["dihedral_path"]),
                    "radius": r,
                    "radius_units": "angstrom",
                    "angle": _to_units(a, angle_units),
                    "angle_units": angle_units,
                    "dihedral": _to_units(d, dihedral_units),
                    "dihedral_units": dihedral_units,
                    "r_exist": r_f,
                    "a_exist": a_f,
                    "d_exist": d_f,
                    "focus_atom_symbol": atom_symbols[focus_atom],
                    "is_donor": rec.get("is_donor", 0),
                    "is_acceptor": rec.get("is_acceptor", 0),
                    "is_donor_H": rec.get("is_donor_H", 0),
                    "is_acceptor_H": rec.get("is_acceptor_H", 0),
                    "is_acceptor_neighbor": rec.get("is_acceptor_neighbor", 0),
                    "is_donor_neighbor": rec.get("is_donor_neighbor", 0),
                }
                if mode_tag is not None:
                    row["mode"] = mode_tag
                all_feat_rows.append(row)

    df_all_feats = pd.DataFrame(all_feat_rows)
    df_all_feats.to_csv(output_csv, index=False)


def _with_suffix(base_path: str, tag: str) -> str:
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{tag}{p.suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract atom features (radius, angle, dihedral) from SDF files."
    )
    parser.add_argument("sdf_directory", type=str, help="Directory containing .sdf files.")
    parser.add_argument(
        "--output",
        type=str,
        default="all_sdf_features.csv",
        help=(
            "Base output. If it looks like a directory (trailing '/', existing dir, or no extension), "
            "files will be written there as <dir>_{mode}.csv. "
            "If it's a file (e.g., foo.csv), outputs become foo_{mode}.csv."
        ),
    )
    # Modes: allow multiple selections or a convenience --both
    parser.add_argument(
        "--mode",
        choices=["default", "path-only", "hybrid"],
        action="append",
        help="Which computation mode(s) to run. Can be specified multiple times.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Shortcut for running both 'default' and 'path-only' and writing two files.",
    )
    parser.add_argument(
        "--angle-units",
        choices=["radian", "degree"],
        default="radian",
        help="Units for angles (default: radian).",
    )
    parser.add_argument(
        "--dihedral-units",
        choices=["radian", "degree"],
        default="radian",
        help="Units for dihedrals (default: radian).",
    )
    args = parser.parse_args()

    # Resolve modes to run
    modes_to_run = []
    if args.both:
        modes_to_run = ["default", "path-only"]
    elif args.mode:
        modes_to_run = args.mode
    else:
        modes_to_run = ["default"]  # default behavior if nothing specified

    sdf_files = get_sdf_file_list(args.sdf_directory)

    for mode in modes_to_run:
        if mode == "default":
            out = _resolve_output(args.output, "default")
            create_atom_feat_RAD_table(
                sdf_files,
                reference_labels=("d_hydrogen", "a_hydrogen", "acceptor"),
                path_only=False,
                hybrid=False,
                angle_units=args.angle_units,
                dihedral_units=args.dihedral_units,
                output_csv=out,
                mode_tag="default",
            )
            print(f"Wrote default features → '{out}'")

        elif mode == "path-only":
            out = _resolve_output(args.output, "path")
            create_atom_feat_RAD_table(
                sdf_files,
                reference_labels=("d_hydrogen", "a_hydrogen", "acceptor"),
                path_only=True,
                hybrid=False,
                angle_units=args.angle_units,
                dihedral_units=args.dihedral_units,
                output_csv=out,
                mode_tag="path-only",
            )
            print(f"Wrote path-only features → '{out}'")

        elif mode == "hybrid":
            out = _resolve_output(args.output, "hybrid")
            create_atom_feat_RAD_table(
                sdf_files,
                reference_labels=("d_hydrogen", "a_hydrogen", "acceptor"),
                path_only=False,
                hybrid=True,
                angle_units=args.angle_units,
                dihedral_units=args.dihedral_units,
                output_csv=out,
                mode_tag="hybrid",
            )
            print(f"Wrote hybrid features → '{out}'")
