from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.atom import MultiHotAtomFeaturizer


@dataclass
class GeometryMolGraphFeaturizer:
    """
    Geometry-only featurizer: uses 3D positions to compute per-edge features:
      - RBF(distance)
      - sin/cos of angle A–B–C (+ availability mask)
      - sin/cos of dihedral A–B–C–D (+ availability mask)
    Missing geometry is encoded with zeros plus mask=0 to keep it distinct from a true 0 radian value.
    """

    def __init__(
        self,
        rbf_D_min: float = 0.0,
        rbf_D_max: float = 5.0,
        rbf_D_count: int = 10,
        rbf_gamma: float = 10.0,
        nan_debug: bool = False,
    ):
        self.mu = np.linspace(rbf_D_min, rbf_D_max, rbf_D_count)
        self.gamma = rbf_gamma
        self.rbf_dim = rbf_D_count
        self.atom_featurizer = MultiHotAtomFeaturizer.v2()
        self.atom_fdim = len(self.atom_featurizer)
        # distance RBF + angle (sin, cos, mask) + dihedral (sin, cos, mask)
        self.bond_fdim = self.rbf_dim + 2 + 1 + 2 + 1
        self.nan_debug = bool(nan_debug)

    def _rbf(self, d: float) -> np.ndarray:
        return np.exp(-self.gamma * (d - self.mu) ** 2)

    @staticmethod
    def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        ba /= np.linalg.norm(ba) + 1e-8
        bc /= np.linalg.norm(bc) + 1e-8
        cos_a = np.dot(ba, bc)
        return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    @staticmethod
    def compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        # based on rdMolTransforms.GetDihedralRad
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1 /= np.linalg.norm(b1) + 1e-8
        v = b0 - np.dot(b0, b1)[:, None] * b1 if b0.ndim > 1 else b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1)[:, None] * b1 if b2.ndim > 1 else b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return float(np.arctan2(y, x))

    def __call__(self, mol, atom_features_extra=None, bond_features_extra=None):
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        # Build a MolGraph with geometry-based edge features
        # validate extras
        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(f"Expected {n_atoms} atom extras, got {len(atom_features_extra)}")
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(f"Expected {n_bonds} bond extras, got {len(bond_features_extra)}")

        # node features
        V = (
            np.zeros((1, self.atom_fdim), dtype=np.float32)
            if n_atoms == 0
            else np.stack([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.float32)
        )
        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))
        V = V.astype(np.float32, copy=False)

        # Edge lists
        E_list, src_list, dst_list = [], [], []
        bond_features_extra = (
            None
            if bond_features_extra is None
            else [np.asarray(f, dtype=np.float32).ravel() for f in bond_features_extra]
        )
        bond_extra_dim = 0
        if bond_features_extra is not None and n_bonds > 0:
            bond_extra_dim = len(bond_features_extra[0])
            if any(len(f) != bond_extra_dim for f in bond_features_extra):
                raise ValueError("All bond_features_extra entries must have the same length.")
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(n_atoms)])
        bond_fdim = self.bond_fdim + bond_extra_dim

        def _print_xyz(reason: str):
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            smi = Chem.MolToSmiles(mol)
            print(n_atoms, flush=True)
            print(f"nan_debug {reason} mol='{name}' smiles='{smi}'", flush=True)
            for i in range(n_atoms):
                sym = mol.GetAtomWithIdx(i).GetSymbol()
                x, y, z = coords[i]
                print(f"{sym} {x:.4f} {y:.4f} {z:.4f}", flush=True)

        # Precompute torsions grouped by central bond so we can aggregate them
        torsions_by_bond: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        for bond in mol.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
            k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
            for i in j_nbrs:
                for l in k_nbrs:
                    # store both (j,k) and (k,j) to support directional lookups
                    torsions_by_bond.setdefault((j, k), []).append((i, j, k, l))
                    torsions_by_bond.setdefault((k, j), []).append((l, k, j, i))

        def angle_feat(src: int, dst: int) -> np.ndarray:
            """Aggregate sin/cos of angles theta(src, dst, k) over neighbors k of dst."""
            vals = []
            for k in [
                n.GetIdx() for n in mol.GetAtomWithIdx(dst).GetNeighbors() if n.GetIdx() != src
            ]:
                ang = self.compute_angle(coords[src], coords[dst], coords[k])
                if not np.isfinite(ang):
                    if self.nan_debug:
                        _print_xyz(f"non-finite angle src={src} dst={dst} k={k}")
                        raise RuntimeError(
                            "GeometryMolGraphFeaturizer nan_debug: non-finite angle "
                            f"src={src} dst={dst} k={k} "
                            f"coords_src={coords[src]} coords_dst={coords[dst]} coords_k={coords[k]}"
                        )
                    continue
                vals.append([np.sin(ang), np.cos(ang)])
            if not vals:
                return np.array([0.0, 0.0], dtype=np.float32), 0.0
            return np.mean(vals, axis=0).astype(np.float32), 1.0

        def torsion_feat(src: int, dst: int) -> np.ndarray:
            """Aggregate sin/cos over all torsions with central bond (src, dst)."""
            vals = []
            for i, j, k, l in torsions_by_bond.get((src, dst), []):
                dih = rdMolTransforms.GetDihedralRad(conf, i, j, k, l)
                if not np.isfinite(dih):
                    continue
                vals.append([np.sin(dih), np.cos(dih)])
            if not vals:
                return np.array([0.0, 0.0], dtype=np.float32), 0.0
            return np.mean(vals, axis=0).astype(np.float32), 1.0

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist = np.linalg.norm(coords[u] - coords[v])
            rbf_feat = self._rbf(dist).astype(np.float32)

            for src, dst in ((u, v), (v, u)):
                ang_sc, ang_has = angle_feat(src, dst)
                dih_sc, dih_has = torsion_feat(src, dst)
                feat_parts = [
                    rbf_feat,
                    ang_sc,
                    np.array([ang_has], dtype=np.float32),
                    dih_sc,
                    np.array([dih_has], dtype=np.float32),
                ]
                if bond_features_extra is not None:
                    feat_parts.append(bond_features_extra[bond.GetIdx()])
                feat = np.concatenate(feat_parts, axis=0).astype(np.float32)
                if feat.shape[0] != bond_fdim:
                    raise ValueError(f"Expected bond feature dim {bond_fdim}, got {feat.shape[0]}")
                if self.nan_debug and not np.isfinite(feat).all():
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                    smi = Chem.MolToSmiles(mol)
                    raise RuntimeError(
                        f"GeometryMolGraphFeaturizer nan_debug: non-finite bond feat "
                        f"for mol='{name}' smiles='{smi}' bond=({src},{dst}) dist={dist}"
                    )
                E_list.append(feat)
                src_list.append(src)
                dst_list.append(dst)

        if E_list:
            E = np.stack(E_list, axis=0)
            edge_index = np.vstack((src_list, dst_list))
            rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        else:
            E = np.zeros((0, bond_fdim), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=int)
            rev_edge_index = np.zeros((0,), dtype=int)

        return MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)
