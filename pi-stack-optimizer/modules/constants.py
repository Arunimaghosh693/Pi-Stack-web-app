"""Shared constants for the π-stack optimizer."""

HARTREE_TO_KJMOL = 2625.499638
PARAM_NAMES = ["cos_like", "sin_like", "Tx", "Ty", "Tz", "Cx", "Cy"]
IDX = {name: i for i, name in enumerate(PARAM_NAMES)}

FAILURE_PENALTY_BASE = 1.0e6
FAILURE_PENALTY_CLASH_MULT = 1.0e4
