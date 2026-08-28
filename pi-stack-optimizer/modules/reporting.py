"""Reporting utilities for human-readable summaries."""
from __future__ import annotations

import numpy as np

from modules.constants import IDX, PARAM_NAMES
from modules.geometry import angle_from_cos_sin_like
from modules.torsion import SymmetricTorsionMapper


def summarize_params(vec: np.ndarray, torsion_mapper: SymmetricTorsionMapper | None = None) -> str:
    base = vec[:7]
    d = {name: float(base[IDX[name]]) for name in PARAM_NAMES}
    theta = angle_from_cos_sin_like(d["cos_like"], d["sin_like"])
    deg = np.degrees(theta)
    s = (f"θ = {deg:.3f}°  "
         f"Tx={d['Tx']:.4f} Å, Ty={d['Ty']:.4f} Å, Tz={d['Tz']:.4f} Å, "
         f"Cx={d['Cx']:.4f} Å, Cy={d['Cy']:.4f} Å")
    if vec.shape[0] > 7:
        reduced_taus = vec[7:]
        reduced_deg = np.degrees(reduced_taus)
        if torsion_mapper is not None and torsion_mapper.n_reduced > 0:
            full_taus = torsion_mapper.expand_torsions(reduced_taus)
            full_deg = np.degrees(full_taus)
            s += "  |  reduced torsions: [" + ", ".join(f"{float(t):.3f}°" for t in reduced_deg) + "]"
            if len(full_deg) > len(reduced_deg):
                s += "  |  full torsions: [" + ", ".join(f"{float(t):.3f}°" for t in full_deg) + "]"
        else:
            s += "  |  torsions: [" + ", ".join(f"{float(t):.3f}°" for t in reduced_deg) + "]"
    return s
