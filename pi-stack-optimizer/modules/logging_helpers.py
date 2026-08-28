"""Logging configuration and formatted output helpers."""
from __future__ import annotations

import builtins
import csv
import logging
import os
import sys
from typing import Dict, List, Optional, TextIO, Tuple, TYPE_CHECKING

from modules.geometry import check_monomer_geometry_sanity
from modules.system_utils import format_file_info

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from modules.optimizer import GAConfig, GWOConfig, PSONMConfig, PSOConfig
    from modules.xtb_workers import XTBConfig

_output_logger: Optional[logging.Logger] = None
_original_print = print


def _log_print(*args, **kwargs):
    file_arg = kwargs.get('file', sys.stdout)
    if file_arg == sys.stderr:
        _original_print(*args, **kwargs)
        return
    message = ' '.join(str(arg) for arg in args)
    if _output_logger is not None:
        _output_logger.info(message)
    else:
        _original_print(*args, **kwargs)


def configure_output_logging(log_dir: Optional[str] = None) -> logging.Logger:
    """Configure logging such that print statements mirror to a file."""
    global _output_logger
    if _output_logger is not None:
        return _output_logger

    target_dir = log_dir or os.environ.get("OUTPUT_LOG_DIR", ".")
    os.makedirs(target_dir, exist_ok=True)
    log_path = os.path.join(target_dir, "output.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _output_logger = logging.getLogger('output')
    builtins.print = _log_print  # type: ignore[assignment]
    return _output_logger


def print_run_banner(start_time_str: str) -> None:
    print()
    print("═" * 80)
    print("π-STACK OPTIMIZER - Molecular Stacking Geometry Optimization")
    print("using xTB Quantum Chemistry + Swarm / Genetic Optimizers")
    print("═" * 80)
    print()
    print("Authors: Arunima Ghosh, Susmita Barik, Roshan J Singh, Sandeep K. Reddy")
    print()
    print(f"Start time: {start_time_str}")
    print()


def print_system_information_block(system_info: Dict[str, str]) -> None:
    print("┌─ SYSTEM INFORMATION " + "─" * 55)
    print(f"│ Working directory: {system_info['working_directory']}")
    print(f"│ Script path:       {system_info['script_path']}")
    print(f"│ Python version:    {system_info['python_version']}")
    print(f"│ xTB version:       {system_info['xtb_version']}")
    print("└" + "─" * 79)
    print()


def print_input_files_block(xyz_file: str, torsions_file: Optional[str]) -> None:
    print("┌─ INPUT FILES " + "─" * 61)
    # Extract just the filename from the path
    xyz_filename = xyz_file.split('/')[-1] if '/' in xyz_file else xyz_file
    print(f"│ Monomer XYZ:       {xyz_filename}")
    if torsions_file:
        torsions_filename = torsions_file.split('/')[-1] if '/' in torsions_file else torsions_file
        print(f"│ Torsions config:   {torsions_filename}")
    else:
        print("│ Torsions config:   None (rigid molecule optimization)")
    print("└" + "─" * 79)
    print()


def report_geometry_validation(coords, atom_types) -> bool:
    print("┌─ GEOMETRY VALIDATION " + "─" * 54)
    print("│ Checking input monomer geometry for sanity...")
    is_sane, error_msg = check_monomer_geometry_sanity(coords, atom_types)
    if not is_sane:
        print("│ ❌ GEOMETRY CHECK FAILED!")
        print("│")
        for line in error_msg.split('\n'):
            if line.strip():
                print(f"│ {line}")
        print("│")
        print("│ Please fix the input geometry and try again.")
        print("│ Consider:")
        print("│   - Removing overlapping atoms")
        print("│   - Checking coordinate units (should be Angstroms)")
        print("│   - Verifying the XYZ file format")
        print("└" + "─" * 79)
        print()
        print("❌ CALCULATION TERMINATED - Invalid input geometry")
        return False
    print("│ ✅ Geometry validation passed - no atomic overlaps detected")
    print("└" + "─" * 79)
    print()
    return True


def print_molecular_system_block(atom_types: List[str], torsions_spec: List[Dict]) -> None:
    print("┌─ MOLECULAR SYSTEM " + "─" * 56)
    print(f"│ Number of atoms:   {len(atom_types)}")
    atom_counts: Dict[str, int] = {}
    for atom in atom_types:
        atom_counts[atom] = atom_counts.get(atom, 0) + 1
    atom_formula = ', '.join(f"{atom}:{count}" for atom, count in sorted(atom_counts.items()))
    print(f"│ Composition:       {atom_formula}")
    print(f"│ Number of torsions considered for optimization: {len(torsions_spec)}")
    if torsions_spec:
        for i, t in enumerate(torsions_spec[:5], 1):
            torsion_name = t.get('name') or f"atoms {t['atoms']}"
            print(f"│   {i:2d}. {torsion_name} (bond {t['b']}-{t['c']})")
        if len(torsions_spec) > 5:
            print(f"│       ... and {len(torsions_spec) - 5} more torsions")
    print("└" + "─" * 79)
    print()


def print_xtb_configuration_block(
    xtb_cfg: "XTBConfig", workers: int, threads: int, backend_label: str
) -> None:
    print("┌─ xTB CONFIGURATION " + "─" * 55)
    print(f"│ Method:            {xtb_cfg.method.upper()}")
    print(f"│ Charge:            {xtb_cfg.charge:+d}")
    print(f"│ Multiplicity:      {xtb_cfg.mult}")
    print(f"│ Solvent:           {xtb_cfg.solvent or 'None (gas phase)'}")
    print(f"│ Backend:           {backend_label}")
    print(f"│ Workers:           {workers}")
    print(f"│ Threads per worker: {threads}")
    print("└" + "─" * 79)
    print()


def print_pso_configuration_block(pso_cfg: "PSOConfig") -> None:
    print("┌─ PSO CONFIGURATION " + "─" * 55)
    print(f"│ Swarm size:        {pso_cfg.swarm_size}")
    print(f"│ Max iterations:    {pso_cfg.max_iters}")
    print(f"│ Random seed:       {pso_cfg.seed}")
    print(f"│ Inertia weight:    {pso_cfg.inertia:.2f}")
    print(f"│ Cognitive coeff:   {pso_cfg.cognitive:.2f}")
    print(f"│ Social coeff:      {pso_cfg.social:.2f}")
    print(f"│ Convergence tol:   {pso_cfg.tol:.2f}")
    print(f"│ Patience:          {pso_cfg.patience} iterations")
    print(f"│ Verbose every:     {pso_cfg.verbose_every} iterations")
    print(f"│ Save trajectories: {'Yes' if pso_cfg.print_trajectories else 'No'}")
    print("└" + "─" * 79)
    print()


def print_ga_configuration_block(ga_cfg: "GAConfig") -> None:
    print("┌─ GA CONFIGURATION " + "─" * 56)
    print(f"│ Population size:   {ga_cfg.population_size}")
    print(f"│ Max generations:   {ga_cfg.max_generations}")
    print(f"│ Random seed:       {ga_cfg.seed}")
    print(f"│ Elite fraction:    {ga_cfg.elite_fraction:.2f}")
    print(f"│ Tournament size:   {ga_cfg.tournament_size}")
    print(f"│ Crossover rate:    {ga_cfg.crossover_rate:.2f}")
    print(f"│ Mutation rate:     {ga_cfg.mutation_rate:.2f}")
    print(f"│ Mutation sigma:    {ga_cfg.mutation_sigma:.2f}")
    print(f"│ Convergence tol:   {ga_cfg.tol:.2f}")
    print(f"│ Patience:          {ga_cfg.patience} generations")
    print(f"│ Verbose every:     {ga_cfg.verbose_every} generations")
    print("└" + "─" * 79)
    print()


def print_gwo_configuration_block(gwo_cfg: "GWOConfig") -> None:
    print("┌─ GWO CONFIGURATION " + "─" * 55)
    print(f"│ Pack size:        {gwo_cfg.pack_size}")
    print(f"│ Max iterations:   {gwo_cfg.max_iters}")
    print(f"│ Random seed:      {gwo_cfg.seed}")
    print(f"│ a-start:          {gwo_cfg.a_start:.3f}")
    print(f"│ a-end:            {gwo_cfg.a_end:.3f}")
    print(f"│ Convergence tol:  {gwo_cfg.tol:.2f}")
    print(f"│ Patience:         {gwo_cfg.patience} iterations")
    print(f"│ Verbose every:    {gwo_cfg.verbose_every} iterations")
    print("└" + "─" * 79)
    print()


def print_pso_nm_configuration_block(cfg: "PSONMConfig") -> None:
    print("┌─ PSO + NELDER-MEAD CONFIGURATION " + "─" * 38)
    print("│ PSO stage:")
    print(f"│   Swarm size:      {cfg.pso.swarm_size}")
    print(f"│   Max iterations:  {cfg.pso.max_iters}")
    print(f"│   Random seed:     {cfg.pso.seed}")
    print(f"│   Inertia:         {cfg.pso.inertia:.2f}")
    print(f"│   Cognitive:       {cfg.pso.cognitive:.2f}")
    print(f"│   Social:          {cfg.pso.social:.2f}")
    print("│ Nelder-Mead stage:")
    print(f"│   Max iterations:  {cfg.nm_max_iters}")
    print(f"│   Initial step:    {cfg.nm_initial_step:.2f}")
    print(f"│   α (reflect):     {cfg.nm_alpha:.1f}")
    print(f"│   γ (expand):      {cfg.nm_gamma:.1f}")
    print(f"│   ρ (contract):    {cfg.nm_rho:.2f}")
    print(f"│   σ (shrink):      {cfg.nm_sigma:.2f}")
    print(f"│   Tolerance:       {cfg.nm_tol:.2f}")
    print("└" + "─" * 79)
    print()


def print_optimization_settings_block(
    pso_dim: int,
    n_torsion_dims: int,
    n_torsions: int,
    symmetry_tolerance: Optional[float],
    n_layer: int,
    penalty_weight: float,
    clash_cutoff: float,
    intramol_penalty_weight: float,
    intramol_cutoff: float,
) -> None:
    print("┌─ OPTIMIZATION SETTINGS " + "─" * 51)
    print(f"│ Optimization space: {pso_dim} dimensions")
    print("│   - Transform params: 7 (rotation + translation)")
    torsion_line = f"│   - Torsion angles:   {n_torsion_dims}"
    if n_torsions > 0 and n_torsion_dims != n_torsions:
        torsion_line += f" (reduced from {n_torsions} via symmetry)"
    elif n_torsions == 0:
        torsion_line += " (rigid optimization)"
    print(torsion_line)
    if n_torsions > 0 and symmetry_tolerance is not None:
        print(f"│   - Symmetry tolerance: {symmetry_tolerance:.2f}°")
    print(f"│ Stack evaluation:   {n_layer} layers")
    print("│ Final stack output: 10 molecules (fixed)")
    print("│ Penalty weights:")
    print(f"│   - Intermolecular:   {penalty_weight:.1f} (cutoff: {clash_cutoff:.1f} Å)")
    print(f"│   - Intramolecular:   {intramol_penalty_weight:.1f} (cutoff: {intramol_cutoff:.1f} Å)")
    print("└" + "─" * 79)
    print()


def print_output_files_block(n_molecules: int, include_trajectories: bool) -> None:
    print("┌─ OUTPUT FILES " + "─" * 61)
    print("│ Results summary:   optimization_results.txt")
    print("│ Optimized monomer: molecule_after_torsion.xyz")
    print(f"│ Final stack:       molecular_stack_{n_molecules}molecules.xyz")
    if include_trajectories:
        print("│ PSO trajectories:  pso_trajectory.csv")
    print("└" + "─" * 79)
    print()


def setup_trajectory_logging(pso_cfg: "PSOConfig", pso_dim: int) -> Tuple[Optional[TextIO], Optional[csv.writer]]:
    if not pso_cfg.print_trajectories:
        return None, None
    trajectory_file = open("pso_trajectory.csv", "w", newline='')
    trajectory_writer = csv.writer(trajectory_file)
    header = [
        "iteration",
        "particle_id",
        "is_global_best",
        *[f"param_{i}" for i in range(pso_dim)],
        "fitness",
    ]
    trajectory_writer.writerow(header)
    return trajectory_file, trajectory_writer
