"""Stack construction utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from modules.geometry import apply_transform, transformation_matrix


def build_n_layer_stack(atom_types: List[str], monomer_coords: np.ndarray,
                       transform_params: np.ndarray, n_layers: int = 2) -> Tuple:
    M = transformation_matrix(transform_params)
    all_atoms: List[str] = []
    all_coords = []
    layer_coords = []
    base = monomer_coords.copy()
    for k in range(n_layers):
        if k == 0:
            current_coords = base.copy()
        else:
            Mpow = np.linalg.matrix_power(M, k)
            current_coords = apply_transform(base, Mpow)
        all_atoms.extend(atom_types)
        all_coords.append(current_coords)
        layer_coords.append(current_coords.copy())
    return all_atoms, np.vstack(all_coords), layer_coords
