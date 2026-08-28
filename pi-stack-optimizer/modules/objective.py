"""Batch objective used by the optimizer."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.constants import (
    FAILURE_PENALTY_BASE,
    FAILURE_PENALTY_CLASH_MULT,
    HARTREE_TO_KJMOL,
)
from modules.geometry import (
    align_pi_core_to_xy,
    build_bond_graph,
    check_topology_preserved,
    clash_penalty,
    intramolecular_clash_penalty,
)
from modules.stacking import build_n_layer_stack
from modules.torsion import SymmetricTorsionMapper, rotate_fragment_around_bond


class BatchObjective:
    def __init__(self, atom_types: List[str], monomer_coords: np.ndarray,
                 torsions_spec: List[Dict], worker_pool,
                 penalty_weight: float, clash_cutoff: float, n_layers: int = 2,
                 intramol_penalty_weight: float = 5.0, intramol_cutoff: float = 1.2,
                 torsion_mapper: Optional[SymmetricTorsionMapper] = None,
                 direct_server=None):
        self.atom_types = atom_types
        self.monomer_coords = monomer_coords
        self.torsions_spec = torsions_spec
        self.worker_pool = worker_pool
        self.direct_server = direct_server  # XTBServer for direct evaluation
        self.penalty_weight = penalty_weight
        self.clash_cutoff = clash_cutoff
        self.n_layers = int(n_layers)
        self.intramol_penalty_weight = intramol_penalty_weight
        self.intramol_cutoff = intramol_cutoff
        self.bond_graph = build_bond_graph(atom_types, monomer_coords)
        self.reference_bond_graph = self.bond_graph  # Store reference connectivity
        self.torsion_mapper = torsion_mapper or SymmetricTorsionMapper(
            [[{"index": i, "transform": "identity"}] for i in range(len(torsions_spec))],
            total_torsions=len(torsions_spec)
        )
        self.n_torsion_dims = self.torsion_mapper.n_reduced if self.torsion_mapper else 0

    @staticmethod
    def _check_catastrophic_overlap(layers: List[np.ndarray]) -> bool:
        """Check if geometry has catastrophic overlaps that would crash xTB.
        
        Returns True if:
        - Any interatomic distance < 1.5 Å
        - More than 5 atom pairs closer than 2.0 Å between any layer pair
        """
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                diff = layers[i][:, None, :] - layers[j][None, :, :]
                dist = np.linalg.norm(diff, axis=-1)
                n_close = np.sum(dist < 2.0)
                if n_close > 5 or np.min(dist) < 1.5:
                    return True
        return False

    def batch_evaluate(self, P: np.ndarray) -> np.ndarray:
        if self.direct_server is not None:
            return self._direct_evaluate(P)
        
        tasks = []
        precomputed = []
        for vec in P:
            try:
                transform_vec = vec[:7]
                reduced_taus = vec[7:]
                result = self._apply_torsions(reduced_taus)
                
                # Check for topology violation
                if result[0] is None:
                    precomputed.append(FAILURE_PENALTY_BASE)
                    tasks.append(None)
                    continue
                
                modified_coords, intra_penalty = result
                atoms_stack, coords_stack, layers = build_n_layer_stack(
                    self.atom_types, modified_coords, transform_vec, n_layers=self.n_layers
                )
                inter_penalty = 0.0
                for i in range(self.n_layers):
                    for j in range(i + 1, self.n_layers):
                        w = 1.0 / float(j - i)
                        inter_penalty += w * clash_penalty(layers[i], layers[j], cutoff=self.clash_cutoff)
                total_penalty = self.penalty_weight * inter_penalty + self.intramol_penalty_weight * intra_penalty

                # Block catastrophic overlaps by checking for severe close contacts
                if self._check_catastrophic_overlap(layers):
                    precomputed.append(FAILURE_PENALTY_BASE + FAILURE_PENALTY_CLASH_MULT * total_penalty)
                    tasks.append(None)
                    continue

                tasks.append((modified_coords, atoms_stack, coords_stack, total_penalty))
                precomputed.append(None)
            except Exception:
                tasks.append(None)
                precomputed.append(1.0e6)
        job_ids = []
        for task in tasks:
            if task is not None:
                modified_coords, atoms_stack, coords_stack, penalty = task
                mono_id = self.worker_pool.submit(self.atom_types, modified_coords)
                stack_id = self.worker_pool.submit(atoms_stack, coords_stack)
                job_ids.append((mono_id, stack_id))
            else:
                job_ids.append(None)
        n_jobs = sum(2 for jid in job_ids if jid is not None)
        job_results = {}
        for _ in range(n_jobs):
            task_id, energy, error = self.worker_pool.get_result()
            job_results[task_id] = energy if error is None else None
        results = []
        for pre_value, task, job_pair in zip(precomputed, tasks, job_ids):
            if pre_value is not None:
                results.append(pre_value)
                continue

            if task is None or job_pair is None:
                results.append(1.0e6)
                continue
            _, _, _, total_penalty = task
            mono_id, stack_id = job_pair
            E_mono_h = job_results.get(mono_id)
            E_stack_h = job_results.get(stack_id)
            if E_mono_h is None or E_stack_h is None:
                results.append(FAILURE_PENALTY_BASE + FAILURE_PENALTY_CLASH_MULT * total_penalty)
            else:
                n_layers = self.n_layers
                dE_kj = ((E_stack_h - float(n_layers) * E_mono_h) * HARTREE_TO_KJMOL) / max(n_layers - 1, 1)
                extra_clash_penalty = FAILURE_PENALTY_CLASH_MULT * total_penalty
                final_score = dE_kj + extra_clash_penalty
                # Alert on unphysical binding energies (should be rare with parameter bounds)
                if dE_kj < -1000.0:
                    print(f"[ALERT] Large negative binding energy: dE={dE_kj:.2f} kJ/mol, penalty={total_penalty:.2f}, E_stack={E_stack_h:.6f} Ha, E_mono={E_mono_h:.6f} Ha")
                results.append(final_score)
        return np.array(results)

    def _direct_evaluate(self, P: np.ndarray) -> np.ndarray:
        """Direct evaluation using XTBServer without worker pool."""
        results = []
        for vec in P:
            try:
                transform_vec = vec[:7]
                reduced_taus = vec[7:]
                result = self._apply_torsions(reduced_taus)
                
                # Check for topology violation
                if result[0] is None:
                    results.append(FAILURE_PENALTY_BASE)
                    continue
                
                modified_coords, intra_penalty = result
                atoms_stack, coords_stack, layers = build_n_layer_stack(
                    self.atom_types, modified_coords, transform_vec, n_layers=self.n_layers
                )
                inter_penalty = 0.0
                for i in range(self.n_layers):
                    for j in range(i + 1, self.n_layers):
                        w = 1.0 / float(j - i)
                        inter_penalty += w * clash_penalty(layers[i], layers[j], cutoff=self.clash_cutoff)
                total_penalty = self.penalty_weight * inter_penalty + self.intramol_penalty_weight * intra_penalty
                
                # Block catastrophic overlaps by checking for severe close contacts
                if self._check_catastrophic_overlap(layers):
                    results.append(FAILURE_PENALTY_BASE + FAILURE_PENALTY_CLASH_MULT * total_penalty)
                    continue

                # Direct evaluation
                try:
                    E_mono_h = self.direct_server.compute_energy(self.atom_types, modified_coords)
                    E_stack_h = self.direct_server.compute_energy(atoms_stack, coords_stack)
                    n_layers = self.n_layers
                    dE_kj = ((E_stack_h - float(n_layers) * E_mono_h) * HARTREE_TO_KJMOL) / max(n_layers - 1, 1)
                    extra_clash_penalty = FAILURE_PENALTY_CLASH_MULT * total_penalty
                    final_score = dE_kj + extra_clash_penalty
                    # Log suspicious large negative energies
                    if dE_kj < -1000.0:
                        print(f"[ALERT-DIRECT] Large negative binding energy: dE={dE_kj:.2f} kJ/mol, penalty={total_penalty:.2f}, E_stack={E_stack_h:.6f} Ha, E_mono={E_mono_h:.6f} Ha")
                    results.append(final_score)
                except Exception:
                    results.append(FAILURE_PENALTY_BASE + FAILURE_PENALTY_CLASH_MULT * total_penalty)
            except Exception:
                results.append(1.0e6)
        return np.array(results)

    def _apply_torsions(self, reduced_taus: np.ndarray) -> Tuple[np.ndarray, float]:
        if reduced_taus.size == 0 or self.torsion_mapper.n_reduced == 0:
            return self.monomer_coords.copy(), 0.0
        full_taus = self.torsion_mapper.expand_torsions(np.clip(reduced_taus, -np.pi, np.pi))
        if full_taus.size == 0:
            return self.monomer_coords.copy(), 0.0
        coords = self.monomer_coords.copy()
        for k, tau in enumerate(full_taus):
            if k < len(self.torsions_spec):
                spec = self.torsions_spec[k]
                coords = rotate_fragment_around_bond(
                    coords, spec['b'], spec['c'], float(tau), spec['fragment']
                )
        
        # Check if topology changed (spurious bonds formed/broken)
        if check_topology_preserved(coords, self.atom_types, self.reference_bond_graph, tolerance_buffer=0.05):
            # Topology changed - return None to signal rejection
            return None, 0.0
        
        intra_penalty = intramolecular_clash_penalty(
            coords, self.atom_types, self.bond_graph, self.intramol_cutoff
        )
        if full_taus.size > 0:
            coords = align_pi_core_to_xy(self.atom_types, coords)
        return coords, intra_penalty

    def get_full_torsions_from_best(self, best_params: np.ndarray) -> np.ndarray:
        if self.n_torsion_dims == 0 or best_params.size <= 7:
            return np.zeros(0, dtype=float)
        reduced_taus = best_params[7:]
        return self.torsion_mapper.expand_torsions(reduced_taus)
