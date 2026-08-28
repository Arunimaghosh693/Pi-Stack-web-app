"""Geometry utilities for π-stack construction and validation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional scipy for faster clash penalty calculation
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def angle_from_cos_sin_like(c_like: float, s_like: float) -> float:
    return float(np.arctan2(s_like, c_like))


def transformation_matrix(params: np.ndarray) -> np.ndarray:
    c_like, s_like, Tx, Ty, Tz, Cx, Cy = params
    theta = angle_from_cos_sin_like(c_like, s_like)
    c, s = np.cos(theta), np.sin(theta)

    dx = Tx + Cx - (Cx * c - Cy * s)
    dy = Ty + Cy - (Cx * s + Cy * c)

    return np.array([
        [c, -s, 0.0, dx],
        [s,  c, 0.0, dy],
        [0.0, 0.0, 1.0, Tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def apply_transform(coords: np.ndarray, M: np.ndarray) -> np.ndarray:
    homo = np.c_[coords, np.ones(len(coords))]
    return (homo @ M.T)[:, :3]


def align_bestfit_plane_to_xy(coords: np.ndarray, select: Optional[np.ndarray] = None) -> np.ndarray:
    P = coords if select is None else coords[np.asarray(select)]
    if P.shape[0] < 3:
        return coords.copy()

    c_full = coords.mean(axis=0, keepdims=True)
    Pc = P - P.mean(axis=0, keepdims=True)

    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    normal = Vt[-1]
    n = normal / (np.linalg.norm(normal) + 1e-15)

    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(n, ez), -1.0, 1.0))

    if abs(1.0 - dot) < 1e-12:
        R = np.eye(3)
    elif abs(-1.0 - dot) < 1e-12:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=float)
    else:
        axis = np.cross(n, ez)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / (axis_norm + 1e-15)
        angle = float(np.arccos(dot))
        K = np.array([[0,        -axis[2],  axis[1]],
                      [axis[2],   0,       -axis[0]],
                      [-axis[1],  axis[0],  0]], dtype=float)
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    return (coords - c_full) @ R.T + c_full


def align_pi_core_to_xy(atom_types: List[str], coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    center = coords.mean(axis=0)

    heavy_idx = [i for i, a in enumerate(atom_types) if not a.strip().upper().startswith('H')]
    if not heavy_idx:
        heavy_idx = list(range(coords.shape[0]))

    dists = [(i, float(np.linalg.norm(coords[i] - center))) for i in heavy_idx]
    dists.sort(key=lambda x: x[1])
    sel = np.array([i for i, _ in dists[:min(6, len(dists))]], dtype=int)
    coords = align_bestfit_plane_to_xy(coords, select=sel)

    heavy_coords = coords[heavy_idx] if heavy_idx else coords
    center_new = heavy_coords.mean(axis=0)
    xy_coords = heavy_coords[:, :2] - center_new[:2]

    if xy_coords.shape[0] >= 2:
        cov_matrix = np.cov(xy_coords.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx_sort]
        long_axis = eigenvectors[:, 0]
        angle = np.arctan2(long_axis[1], long_axis[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R_inplane = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=float)
        full_center = coords.mean(axis=0, keepdims=True)
        coords = (coords - full_center) @ R_inplane.T + full_center

    return coords


def clash_penalty(coords1: np.ndarray, coords2: np.ndarray, cutoff: float = 1.6) -> float:
    """Optimized clash penalty calculation with early termination.
    
    Uses cKDTree for O(n log n) performance on large systems if scipy is available,
    otherwise falls back to vectorized O(n²) approach.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0
    
    # For very small systems, use simple calculation
    if len(coords1) <= 5 and len(coords2) <= 5:
        penalty = 0.0
        for i, c1 in enumerate(coords1):
            for c2 in coords2:
                dist = np.linalg.norm(c1 - c2)
                if dist < cutoff:
                    violation = cutoff - dist
                    penalty += violation * violation
        return float(penalty)
    
    # Use cKDTree for large systems if scipy is available (much faster)
    if SCIPY_AVAILABLE and (len(coords1) > 20 or len(coords2) > 20):
        tree = cKDTree(coords2)
        distances, _ = tree.query(coords1, k=1)
        violations = np.maximum(cutoff - distances, 0.0)
        return float(np.sum(violations * violations))
    
    # For medium systems, use vectorized approach
    dists = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    violations = np.maximum(cutoff - min_dists, 0.0)
    return float(np.sum(violations * violations))


def build_bond_graph(atom_types: List[str], coords: np.ndarray) -> Dict[int, List[int]]:
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
        'F': 0.57, 'Cl': 1.02, 'Br': 1.20
    }
    n = len(atom_types)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            ri = covalent_radii.get(atom_types[i].capitalize(), 1.5)
            rj = covalent_radii.get(atom_types[j].capitalize(), 1.5)
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 1.3 * (ri + rj):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def check_topology_preserved(coords: np.ndarray, atom_types: List[str],
                             reference_graph: Dict[int, List[int]],
                             tolerance_buffer: float = 0.05) -> bool:
    """Check if molecular connectivity is preserved after torsion changes.
    
    Args:
        coords: Current atomic coordinates
        atom_types: List of atom type strings
        reference_graph: Reference bond graph from initial geometry
        tolerance_buffer: Additional tolerance (in Angstroms) added to bond detection
                         to avoid false positives from numerical noise
    
    Returns:
        True if topology changed (bonds formed/broken), False if preserved
    """
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
        'F': 0.57, 'Cl': 1.02, 'Br': 1.20
    }
    n = len(atom_types)
    
    # Build current bond graph with added tolerance buffer
    current_bonds = set()
    for i in range(n):
        for j in range(i + 1, n):
            ri = covalent_radii.get(atom_types[i].capitalize(), 1.5)
            rj = covalent_radii.get(atom_types[j].capitalize(), 1.5)
            dist = np.linalg.norm(coords[i] - coords[j])
            # Use 1.3x factor plus buffer for detection
            if dist < 1.3 * (ri + rj) + tolerance_buffer:
                current_bonds.add((i, j))
    
    # Build reference bond set
    reference_bonds = set()
    for i in reference_graph:
        for j in reference_graph[i]:
            if i < j:
                reference_bonds.add((i, j))
    
    # Check if topology changed
    return current_bonds != reference_bonds


def intramolecular_clash_penalty(coords: np.ndarray, atom_types: List[str],
                                 adj: Dict[int, List[int]], cutoff: float = 1.2) -> float:
    n = coords.shape[0]
    penalty = 0.0
    excluded_pairs = set()
    for i in adj:
        for j in adj[i]:
            excluded_pairs.add((min(i, j), max(i, j)))
            for k in adj[j]:
                if k != i:
                    excluded_pairs.add((min(i, k), max(i, k)))
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded_pairs:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                violation = cutoff - dist
                penalty += violation ** 2
    return float(penalty)


def check_monomer_geometry_sanity(coords: np.ndarray, atom_types: List[str],
                                  bonded_cutoff: float = 0.8, angle_cutoff: float = 1.1,
                                  nonbonded_cutoff: float = 1.3) -> Tuple[bool, str]:
    n = coords.shape[0]
    min_distance = float('inf')
    problematic_pairs = []
    bond_graph = build_bond_graph(atom_types, coords)
    bonded_pairs = set()
    angle_pairs = set()
    for i in bond_graph:
        for j in bond_graph[i]:
            bonded_pairs.add((min(i, j), max(i, j)))
            for k in bond_graph[j]:
                if k != i:
                    angle_pairs.add((min(i, k), max(i, k)))
    angle_pairs = angle_pairs - bonded_pairs
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            min_distance = min(min_distance, dist)
            pair_key = (i, j)
            if pair_key in bonded_pairs:
                cutoff = bonded_cutoff
                label = "1-2 bonded"
            elif pair_key in angle_pairs:
                cutoff = angle_cutoff
                label = "1-3 angle"
            else:
                cutoff = nonbonded_cutoff
                label = "non-bonded"
            if dist < cutoff:
                problematic_pairs.append((i, j, dist, atom_types[i], atom_types[j], label))
    if problematic_pairs:
        error_msg = "Input geometry has atoms that are too close:\n"
        for i, j, dist, atom_i, atom_j, pair_type in problematic_pairs[:8]:
            if pair_type == "1-2 bonded":
                cutoff_used = bonded_cutoff
            elif pair_type == "1-3 angle":
                cutoff_used = angle_cutoff
            else:
                cutoff_used = nonbonded_cutoff
            error_msg += (f"  Atoms {i+1} ({atom_i}) and {j+1} ({atom_j}) [{pair_type}]: "
                          f"{dist:.3f} Å (< {cutoff_used:.2f} Å)\n")
        if len(problematic_pairs) > 8:
            error_msg += f"  ... and {len(problematic_pairs) - 8} more problematic pairs\n"
        error_msg += f"Overall minimum distance: {min_distance:.3f} Å\n"
        error_msg += ("Cutoffs used: "
                      f"bonded={bonded_cutoff:.2f} Å, angle={angle_cutoff:.2f} Å, "
                      f"non-bonded={nonbonded_cutoff:.2f} Å")
        return False, error_msg
    return True, ""
