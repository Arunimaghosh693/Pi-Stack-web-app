"""XYZ file read/write helpers."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def read_xyz_file(filename: str) -> Tuple[np.ndarray, List[str]]:
    with open(filename, 'r') as f:
        lines = [l.rstrip() for l in f]  # Keep all lines, only strip trailing whitespace
    n_atoms = int(lines[0])
    coords, atom_types = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:  # Valid atom line with element and 3 coordinates
            atom_types.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords), atom_types


def write_xyz_file(filename: str, coords: np.ndarray, atom_types: List[str], comment: str = "") -> None:
    with open(filename, 'w') as f:
        f.write(f"{len(atom_types)}\n{comment}\n")
        for atom, coord in zip(atom_types, coords):
            f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def build_xyz_string(atom_types: List[str], coords: np.ndarray, comment: str = "") -> str:
    # Optimized version using list comprehension and pre-allocated list
    n_atoms = len(atom_types)
    lines = [str(n_atoms), comment]
    
    # Use list comprehension for better performance
    lines.extend(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}" 
                 for atom, coord in zip(atom_types, coords))
    
    return "\n".join(lines)
