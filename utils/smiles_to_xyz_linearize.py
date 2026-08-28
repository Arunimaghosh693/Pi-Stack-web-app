#!/usr/bin/env python3
"""
Deterministic SMILES -> XYZ generation that *forces extended side-chains* instead of relying on
sampling lots of conformers.

Motivation (your professor's point):
- For big PDI/NDI-like molecules with long alkyl/aryl side chains, "generate many conformers +
  optimize + pick lowest energy" often FAILS because:
    1) the fully-extended (all-trans) chain conformer may never be sampled
    2) even if sampled, unconstrained force-field optimization can fold chains back onto the pi-core

This script uses an alternative approach:
- Build ONE reasonable 3D conformer (ETKDG).
- Identify the pi-core (largest fused aromatic + imide-like atoms).
- Identify substituent components attached to the core.
- For each substituent component:
    * set anchor torsion to push the substituent out of the core plane
    * set internal torsions along rotatable bonds to trans (~180°)
- Run a *constrained* MMFF/UFF minimization:
    * position constraints to keep the pi-core fixed
    * torsion constraints to keep the chosen side-chain torsions near trans/out-of-plane

Output:
- XYZ geometry (with explicit H by default)
- optional JSON report listing constrained dihedrals and final values

Dependencies: RDKit, numpy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms


# ----------------------------
# Core detection (same idea as earlier)
# ----------------------------

def _largest_fused_aromatic_system(mol: Chem.Mol) -> List[int]:
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    aromatic_rings: List[set[int]] = []
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_rings.append(set(ring))
    if not aromatic_rings:
        return []

    n = len(aromatic_rings)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if aromatic_rings[i].intersection(aromatic_rings[j]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    comps: List[set[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        rings_in_comp: List[int] = []
        while stack:
            k = stack.pop()
            rings_in_comp.append(k)
            for nb in adj[k]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        atoms: set[int] = set()
        for rid in rings_in_comp:
            atoms |= aromatic_rings[rid]
        comps.append(atoms)

    largest = max(comps, key=len)
    return sorted(largest)


def _expand_core_with_imide_like_atoms(mol: Chem.Mol, core_atoms: set[int]) -> set[int]:
    core = set(core_atoms)

    # (1) Add imide-like N + carbonyl C/O neighbors
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        carbonyl_c_neighbors: List[int] = []
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() != 6:
                continue
            # carbonyl C if it has a double bond to O
            is_carbonyl = any(
                (b.GetBondType() == Chem.rdchem.BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8)
                for b in nb.GetBonds()
            )
            if is_carbonyl:
                carbonyl_c_neighbors.append(nb.GetIdx())

        if len(carbonyl_c_neighbors) >= 2:
            core.add(atom.GetIdx())
            for cidx in carbonyl_c_neighbors:
                core.add(cidx)
                c_atom = mol.GetAtomWithIdx(cidx)
                for b in c_atom.GetBonds():
                    if b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        other = b.GetOtherAtom(c_atom)
                        if other.GetAtomicNum() == 8:
                            core.add(other.GetIdx())

    # (2) Add carbonyls directly attached to aromatic atoms (conservative)
    for idx in list(core):
        a = mol.GetAtomWithIdx(idx)
        if not a.GetIsAromatic():
            continue
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum() != 6 or nb.GetIsAromatic():
                continue
            is_carbonyl = any(
                (b.GetBondType() == Chem.rdchem.BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8)
                for b in nb.GetBonds()
            )
            if is_carbonyl:
                core.add(nb.GetIdx())
                for b in nb.GetBonds():
                    if b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        other = b.GetOtherAtom(nb)
                        if other.GetAtomicNum() == 8:
                            core.add(other.GetIdx())
                for nb2 in nb.GetNeighbors():
                    if nb2.GetAtomicNum() == 7:
                        core.add(nb2.GetIdx())
    return core


def detect_pi_core_atoms(mol: Chem.Mol) -> List[int]:
    base = _largest_fused_aromatic_system(mol)
    if not base:
        return []
    core = _expand_core_with_imide_like_atoms(mol, set(base))
    return sorted(core)


# ----------------------------
# Graph helpers
# ----------------------------

def connected_components_excluding(mol: Chem.Mol, excluded: set[int]) -> List[List[int]]:
    n = mol.GetNumAtoms()
    visited = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if i in excluded or visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: List[int] = []
        while stack:
            a = stack.pop()
            comp.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in excluded or visited[j]:
                    continue
                visited[j] = True
                stack.append(j)
        comps.append(comp)
    return comps


def heavy_atom_mask(mol: Chem.Mol) -> np.ndarray:
    return np.array([a.GetAtomicNum() != 1 for a in mol.GetAtoms()], dtype=bool)


def fit_plane(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = coords.mean(axis=0)
    x = coords - centroid
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal


# ----------------------------
# Torsion selection utilities
# ----------------------------

@dataclass
class Torsion:
    a: int
    b: int
    c: int
    d: int
    target_deg: float
    tol_deg: float
    kind: str  # "anchor" or "internal"


def _pick_neighbor_not(atom: Chem.Atom, exclude_idx: int, prefer_heavy: bool = True) -> Optional[int]:
    # Prefer heavy neighbors (non-H) when possible.
    candidates = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() != exclude_idx]
    if not candidates:
        return None
    if prefer_heavy:
        heavy = [i for i in candidates if atom.GetOwningMol().GetAtomWithIdx(i).GetAtomicNum() != 1]
        if heavy:
            return heavy[0]
    return candidates[0]


def _bond_is_rotatable(bond: Chem.Bond) -> bool:
    # Conservative definition: single, not in ring, heavy-heavy.
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()
    if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
        return False
    # Avoid terminal bonds where one atom has only 1 heavy neighbor (can't define torsion robustly)
    # We'll still check at torsion construction time.
    return True


def _enumerate_component_internal_rotatable_bonds(mol: Chem.Mol, comp_set: set[int]) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i in comp_set and j in comp_set and _bond_is_rotatable(b):
            bonds.append((i, j))
    return bonds


def _make_torsion_for_bond(mol: Chem.Mol, i: int, j: int, target: float, tol: float, kind: str) -> Optional[Torsion]:
    """
    Create a 4-atom torsion definition around bond i-j:
      a - i - j - d
    where a is a neighbor of i (not j) and d is a neighbor of j (not i).
    """
    ai = mol.GetAtomWithIdx(i)
    aj = mol.GetAtomWithIdx(j)
    a = _pick_neighbor_not(ai, j, prefer_heavy=True)
    d = _pick_neighbor_not(aj, i, prefer_heavy=True)
    if a is None or d is None:
        return None
    return Torsion(a=a, b=i, c=j, d=d, target_deg=float(target), tol_deg=float(tol), kind=kind)


# ----------------------------
# Side-chain linearization
# ----------------------------

@dataclass
class BuildReport:
    canonical_smiles: str
    inchikey: Optional[str]
    core_atom_count: int
    n_sidechains: int
    constrained_torsions: List[Dict]
    final_energy: float
    xyz: str


def build_extended_geometry(
    smiles: str,
    *,
    seed: int = 42,
    max_iters: int = 2000,
    use_mmff: bool = True,
    anchor_dihedral_deg: float = 90.0,
    anchor_tol: float = 15.0,
    internal_dihedral_deg: float = 180.0,
    internal_tol: float = 10.0,
    core_pos_maxdispl: float = 0.05,
    core_pos_k: float = 2000.0,
    torsion_k: float = 200.0,
    include_h: bool = True,
    alternate_sides: bool = True,
) -> BuildReport:
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        raise ValueError("Invalid SMILES (RDKit could not parse).")
    Chem.SanitizeMol(mol0)

    canonical = Chem.MolToSmiles(mol0, canonical=True)
    inchikey = None
    try:
        inchikey = Chem.inchi.MolToInchiKey(mol0)  # type: ignore[attr-defined]
    except Exception:
        inchikey = None

    mol = Chem.AddHs(mol0) if include_h else Chem.Mol(mol0)

    # 1) Initial embedding (single conformer)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)

    params.pruneRmsThresh = -1  # keep one
    params.enforceChirality = True
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError("RDKit embedding failed.")

    # 2) Detect core and sidechains
    core_idx = detect_pi_core_atoms(mol)
    core_set = set(core_idx)
    heavy = heavy_atom_mask(mol)

    if core_idx:
        conf = mol.GetConformer(cid)
        core_coords = np.array([conf.GetAtomPosition(i) for i in core_idx], dtype=float)
        centroid, normal = fit_plane(core_coords)
    else:
        centroid = np.zeros(3)
        normal = np.array([0.0, 0.0, 1.0])

    comps = connected_components_excluding(mol, core_set)
    # keep only components that have >= 2 heavy atoms and are attached to core
    sidechains: List[List[int]] = []
    anchor_pairs: List[Tuple[int, int]] = []  # (core_atom, side_atom)
    for comp in comps:
        comp_heavy = [i for i in comp if heavy[i]]
        if len(comp_heavy) < 2:
            continue
        # check attachment to core
        pair = None
        for i in comp_heavy:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j in core_set:
                    pair = (j, i)
                    break
            if pair:
                break
        if pair:
            sidechains.append(comp_heavy)
            anchor_pairs.append(pair)

    # 3) Build torsion constraints + set torsions deterministically
    torsions: List[Torsion] = []
    conf = mol.GetConformer(cid)

    # Sort anchors by angle around centroid to make alternate_sides stable
    if core_idx and anchor_pairs:
        # compute in-plane angle for core anchor atoms
        # build orthonormal basis (u,v) in plane
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u, normal)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        def anchor_angle(pair):
            core_atom, _ = pair
            p = np.array(conf.GetAtomPosition(core_atom), dtype=float) - centroid
            p_plane = p - np.dot(p, normal) * normal
            return float(np.arctan2(np.dot(p_plane, v), np.dot(p_plane, u)))

        order = sorted(range(len(anchor_pairs)), key=lambda k: anchor_angle(anchor_pairs[k]))
        sidechains = [sidechains[i] for i in order]
        anchor_pairs = [anchor_pairs[i] for i in order]

    for sc_idx, (comp_heavy, (core_atom, side_atom)) in enumerate(zip(sidechains, anchor_pairs)):
        comp_set = set(comp_heavy)

        # Anchor torsion: pick a core neighbor (in core) and a side neighbor (away from core)
        core_neighbors = [nb.GetIdx() for nb in mol.GetAtomWithIdx(core_atom).GetNeighbors() if nb.GetIdx() in core_set and nb.GetIdx() != side_atom]
        side_neighbors = [nb.GetIdx() for nb in mol.GetAtomWithIdx(side_atom).GetNeighbors() if nb.GetIdx() in comp_set and nb.GetIdx() != core_atom and mol.GetAtomWithIdx(nb.GetIdx()).GetAtomicNum() != 1]

        if core_neighbors and side_neighbors:
            a = core_neighbors[0]
            d = side_neighbors[0]

            sign = 1.0
            if alternate_sides:
                sign = 1.0 if (sc_idx % 2 == 0) else -1.0

            target = float(sign * anchor_dihedral_deg)

            # Set current dihedral
            rdMolTransforms.SetDihedralDeg(conf, a, core_atom, side_atom, d, target)

            torsions.append(Torsion(a=a, b=core_atom, c=side_atom, d=d,
                                    target_deg=target, tol_deg=anchor_tol, kind="anchor"))

        # Internal torsions in the component: force trans on each rotatable bond
        for (i, j) in _enumerate_component_internal_rotatable_bonds(mol, comp_set):
            t = _make_torsion_for_bond(mol, i, j, internal_dihedral_deg, internal_tol, kind="internal")
            if t is None:
                continue
            # Only apply if the torsion is entirely within component heavy atoms
            if (t.a in comp_set or t.a in core_set) and (t.d in comp_set or t.d in core_set):
                # Try setting it first (this updates coords)
                try:
                    rdMolTransforms.SetDihedralDeg(conf, t.a, t.b, t.c, t.d, t.target_deg)
                except Exception:
                    continue
                torsions.append(t)

    # 4) Constrained minimization
    ff_name = None
    energy = None

    if use_mmff:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    else:
        props = None

    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        ff_name = "MMFF94s"
        # core position constraints
        for i in core_idx:
            ff.MMFFAddPositionConstraint(int(i), float(core_pos_maxdispl), float(core_pos_k))
        # torsion constraints
        for t in torsions:
            ff.MMFFAddTorsionConstraint(int(t.a), int(t.b), int(t.c), int(t.d),
                                        False, float(t.target_deg - t.tol_deg), float(t.target_deg + t.tol_deg),
                                        float(torsion_k))
        ff.Minimize(maxIts=int(max_iters))
        energy = float(ff.CalcEnergy())
    else:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is None:
            raise RuntimeError("Neither MMFF nor UFF could be set up.")
        ff_name = "UFF"
        for i in core_idx:
            ff.UFFAddPositionConstraint(int(i), float(core_pos_maxdispl), float(core_pos_k))
        for t in torsions:
            ff.UFFAddTorsionConstraint(int(t.a), int(t.b), int(t.c), int(t.d),
                                       False, float(t.target_deg - t.tol_deg), float(t.target_deg + t.tol_deg),
                                       float(torsion_k))
        ff.Minimize(maxIts=int(max_iters))
        energy = float(ff.CalcEnergy())

    # 5) Build XYZ output
    xyz_lines = [str(mol.GetNumAtoms()), f"smiles={canonical} inchikey={inchikey} ff={ff_name} E={energy:.6f}"]
    conf = mol.GetConformer(cid)
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        xyz_lines.append(f"{atom.GetSymbol():<2} {p.x: .8f} {p.y: .8f} {p.z: .8f}")
    xyz = "\n".join(xyz_lines) + "\n"

    # 6) Prepare report (also compute final torsion values)
    torsion_dicts: List[Dict] = []
    for t in torsions:
        try:
            val = float(rdMolTransforms.GetDihedralDeg(conf, int(t.a), int(t.b), int(t.c), int(t.d)))
        except Exception:
            val = float("nan")
        torsion_dicts.append({
            "kind": t.kind,
            "a": t.a, "b": t.b, "c": t.c, "d": t.d,
            "target_deg": t.target_deg,
            "tol_deg": t.tol_deg,
            "final_deg": val,
        })

    return BuildReport(
        canonical_smiles=canonical,
        inchikey=inchikey,
        core_atom_count=len(core_idx),
        n_sidechains=len(sidechains),
        constrained_torsions=torsion_dicts,
        final_energy=float(energy),
        xyz=xyz,
    )



def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SMILES -> XYZ with deterministic side-chain linearization")
    p.add_argument("--smiles", required=True, type=str, help="SMILES string")
    p.add_argument("--out", required=True, type=str, help="Output XYZ path")
    p.add_argument("--report-json", type=str, default=None, help="Optional JSON report path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--no-mmff", action="store_true", help="Use UFF instead of MMFF")
    p.add_argument("--no-h", action="store_true", help="Do not add explicit hydrogens")
    p.add_argument("--anchor-deg", type=float, default=90.0, help="Anchor dihedral (deg) to push substituent out of plane")
    p.add_argument("--anchor-tol", type=float, default=15.0)
    p.add_argument("--internal-deg", type=float, default=180.0, help="Internal dihedral target (deg), e.g., 180 for trans")
    p.add_argument("--internal-tol", type=float, default=10.0)
    p.add_argument("--no-alternate", action="store_true", help="Do not alternate sidechains above/below plane")

    p.add_argument("--core-maxdispl", type=float, default=0.05)
    p.add_argument("--core-k", type=float, default=2000.0)
    p.add_argument("--torsion-k", type=float, default=200.0)

    p.add_argument("--print-summary", action="store_true", help="Print constrained torsions summary")

    args = p.parse_args(list(argv) if argv is not None else None)

    rep = build_extended_geometry(
        args.smiles,
        seed=args.seed,
        max_iters=args.max_iters,
        use_mmff=not args.no_mmff,
        anchor_dihedral_deg=args.anchor_deg,
        anchor_tol=args.anchor_tol,
        internal_dihedral_deg=args.internal_deg,
        internal_tol=args.internal_tol,
        core_pos_maxdispl=args.core_maxdispl,
        core_pos_k=args.core_k,
        torsion_k=args.torsion_k,
        include_h=not args.no_h,
        alternate_sides=not args.no_alternate,
    )

    Path(args.out).write_text(rep.xyz, encoding="utf-8")

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")

    if args.print_summary:
        print("Canonical SMILES:", rep.canonical_smiles)
        print("InChIKey:", rep.inchikey)
        print("Core atoms:", rep.core_atom_count)
        print("Detected sidechains:", rep.n_sidechains)
        print("Final FF energy:", rep.final_energy)
        print("\nConstrained torsions:")
        for t in rep.constrained_torsions:
            print(f"  {t['kind']:8s} {t['a']:4d}-{t['b']:4d}-{t['c']:4d}-{t['d']:4d} "
                  f"target={t['target_deg']:7.1f}±{t['tol_deg']:4.1f} final={t['final_deg']:7.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
