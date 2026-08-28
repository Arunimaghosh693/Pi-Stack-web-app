"""Torsion handling, detection, and mapping utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _dihedral_angle_degrees(coords: np.ndarray, atoms: List[int]) -> float:
    a, b, c, d = atoms
    v1 = coords[b] - coords[a]
    v2 = coords[c] - coords[b]
    v3 = coords[d] - coords[c]
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    cross_n1_n2 = np.cross(n1, n2)
    sign = np.dot(cross_n1_n2, v2 / (np.linalg.norm(v2) + 1e-15))
    return float(-angle if sign < 0 else angle)


def _distance_from_180(angle: float) -> float:
    return min(abs(angle - 180.0), abs(angle + 180.0))


def _circular_difference(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b)
    return min(diff, 360.0 - diff)


def _angles_relation(angle_a: float, angle_b: float, tolerance: float) -> Optional[str]:
    if _circular_difference(angle_a, angle_b) <= tolerance:
        return "identity"
    if _circular_difference(angle_a, -angle_b) <= tolerance:
        return "negate"
    dist_a = _distance_from_180(angle_a)
    dist_b = _distance_from_180(angle_b)
    if abs(dist_a - dist_b) <= tolerance:
        return "negate"
    return None


def detect_symmetric_torsions(torsions_spec: List[Dict], atom_types: List[str],
                             coords: np.ndarray, tolerance: float = 10.0) -> List[List[Dict[str, Any]]]:
    n_torsions = len(torsions_spec)
    if n_torsions == 0:
        print("│ Symmetric torsion detection: no torsions defined")
        return []
    current_angles = []
    for spec in torsions_spec:
        current_angles.append(_dihedral_angle_degrees(coords, spec['atoms']))
    groups: List[List[int]] = []
    assigned = [False] * n_torsions
    for i in range(n_torsions):
        if assigned[i]:
            continue
        group = [{"index": i, "transform": "identity"}]
        assigned[i] = True
        for j in range(i + 1, n_torsions):
            if assigned[j]:
                continue
            relation = _angles_relation(current_angles[i], current_angles[j], tolerance)
            if relation is not None:
                group.append({"index": j, "transform": relation})
                assigned[j] = True
        groups.append(group)
    symmetric_groups = [g for g in groups if len(g) > 1]
    print(f"│ Symmetric torsion detection (tolerance: {tolerance}°):")
    if symmetric_groups:
        for idx, group in enumerate(symmetric_groups, start=1):
            names = []
            angle_info = []
            rel_info = []
            for entry in group:
                torsion_idx = entry["index"]
                spec = torsions_spec[torsion_idx]
                name = spec.get('name') or f"atoms {spec['atoms']}"
                names.append(name)
                angle_info.append(f"{current_angles[torsion_idx]:.1f}°")
                rel_info.append(entry.get("transform", "identity"))
            print(f"│   Group {idx}: {', '.join(names)}")
            print(f"│            Current angles: {', '.join(angle_info)}")
            print(f"│            Relation types: {', '.join(rel_info)}")
        reduction = n_torsions - len(groups)
        print(f"│ → Dimension reduction: {n_torsions} → {len(groups)} (-{reduction})")
    else:
        print("│   No symmetric torsions detected")
    return groups


@dataclass
class SymmetricTorsionMapper:
    torsion_groups: List[List[Dict[str, Any]]]
    total_torsions: int

    def __post_init__(self):
        if not self.torsion_groups:
            self.torsion_groups = [[{"index": i, "transform": "identity"}] for i in range(self.total_torsions)]
        self.n_reduced = len(self.torsion_groups)
        self.n_original = self.total_torsions

    @staticmethod
    def _apply_transform(value: float, transform: str) -> float:
        return -value if transform == "negate" else value

    def expand_torsions(self, reduced_taus: np.ndarray) -> np.ndarray:
        if self.n_original == 0:
            return np.zeros(0, dtype=float)
        full = np.zeros(self.n_original, dtype=float)
        for reduced_idx, group in enumerate(self.torsion_groups):
            value = float(reduced_taus[reduced_idx]) if reduced_idx < reduced_taus.size else 0.0
            for entry in group:
                idx = entry["index"]
                transform = entry.get("transform", "identity")
                if idx < full.size:
                    full[idx] = self._apply_transform(value, transform)
        return full



def find_fragment_indices(adj: Dict[int, List[int]], bond_a: int, bond_b: int,
                          start_atom: int) -> List[int]:
    visited = set()
    queue = [start_atom]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adj[current]:
            if (current == bond_a and neighbor == bond_b) or (current == bond_b and neighbor == bond_a):
                continue
            if neighbor not in visited:
                queue.append(neighbor)
    return sorted(visited)


def rotate_fragment_around_bond(coords: np.ndarray, atom_b: int, atom_c: int,
                                angle: float, fragment_indices: List[int]) -> np.ndarray:
    coords = coords.copy()
    axis = coords[atom_c] - coords[atom_b]
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-8:
        return coords
    axis /= axis_length
    origin = coords[atom_b]
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    for idx in fragment_indices:
        coords[idx] = origin + R @ (coords[idx] - origin)
    return coords


def load_torsions_file(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)
