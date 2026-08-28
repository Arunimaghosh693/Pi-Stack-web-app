#!/usr/bin/env python3
"""
π-Stack Optimizer using xTB + global optimizers

Optimizes molecular stacking geometry using xTB energy calculations
and metaheuristic optimizers with parallel worker processes.

Product Vision
--------------
- Provide a reproducible, scriptable workflow for π-stacked molecular systems
    that balances quantum-chemistry fidelity with swarm-search scalability.
- Support both research prototyping and light production use by generating
    human-readable logs, restartable outputs, and shareable artifacts.

User Requirements
-----------------
- Accept monomer coordinates (XYZ) and optional torsion definitions from
    cheminformatics tooling, without requiring manual data massaging.
- Detect and report geometry issues up front to avoid wasting quantum cycles.
- Offer dimension reduction for symmetric torsions so users can reason about
    fewer parameters while keeping full torsion reporting available.
- Emit intermediate and final XYZ files so downstream visualization/analysis
    scripts can plug in with zero friction.

Technical Constraints & Specifications
--------------------------------------
- Must run with stock Python 3.10+ plus NumPy, using external xTB binaries
    already present on HPC/workstation environments.
- Optimization loop must remain embarrassingly parallel over particles, using
    multiprocessing without requiring cluster schedulers.
- Logging output should stay plain-text/ASCII for easy diffing and ingestion
    into ELN/LIMS systems.
- Codebase should stay single-file for now but remain modularly organized by
    section headers to ease future extraction into packages.

Usage:
        python main.py molecule.xyz --workers 4 --swarm-size 60 --max-iters 300
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np

from modules.constants import HARTREE_TO_KJMOL
from modules.geometry import align_pi_core_to_xy, build_bond_graph
from modules.logging_helpers import (
    configure_output_logging,
    print_ga_configuration_block,
    print_gwo_configuration_block,
    print_input_files_block,
    print_molecular_system_block,
    print_optimization_settings_block,
    print_output_files_block,
    print_pso_configuration_block,
    print_pso_nm_configuration_block,
    print_run_banner,
    print_system_information_block,
    print_xtb_configuration_block,
    report_geometry_validation,
    setup_trajectory_logging,
)
from modules.objective import BatchObjective
from modules.optimizer import (
    GAConfig,
    GWOConfig,
    OptimizerRequest,
    PSONMConfig,
    PSOConfig,
    create_optimizer,
    describe_method,
    list_supported_methods,
)
from modules.reporting import summarize_params
from modules.stacking import build_n_layer_stack
from modules.system_utils import get_system_info, set_thread_env_vars
from modules.torsion import (
    SymmetricTorsionMapper,
    detect_symmetric_torsions,
    find_fragment_indices,
    load_torsions_file,
)
from modules.xtb_workers import XTBConfig, XTBServer, XTBWorkerPool
from modules.xyz_io import read_xyz_file, write_xyz_file


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize π-stack geometry using xTB + global optimizers"
    )
    parser.add_argument("xyz", help="Monomer XYZ file")
    general = parser.add_argument_group("General options")
    general.add_argument("--n-layer", type=int, default=2,
                         help="Number of layers used for energy evaluation (default: 2)")
    general.add_argument("--optimizer", choices=list_supported_methods(), default="pso",
                         help="Optimization method to use (default: pso)")
    general.add_argument("--max-iters", type=int, default=300,
                         help="Maximum optimizer iterations/generations (default: 300)")
    general.add_argument("--seed", type=int, default=42,
                         help="Random seed for optimizer initialization (default: 42)")
    general.add_argument("--verbose-every", type=int, default=1,
                         help="Print optimizer progress every N steps (default: 1)")

    pso = parser.add_argument_group("PSO options")
    pso.add_argument("--swarm-size", type=int, default=60,
                     help="PSO swarm size (default: 60)")
    pso.add_argument("--inertia", type=float, default=0.73,
                     help="PSO inertia weight (default: 0.73)")
    pso.add_argument("--cognitive", type=float, default=1.50,
                     help="PSO cognitive coefficient (default: 1.50)")
    pso.add_argument("--social", type=float, default=1.50,
                     help="PSO social coefficient (default: 1.50)")
    pso.add_argument("--tol", type=float, default=0.01,
                     help="PSO convergence tolerance (default: 0.01)")
    pso.add_argument("--patience", type=int, default=20,
                     help="PSO early stopping patience in iterations (default: 20)")
    pso.add_argument("--print-trajectories", action="store_true",
                     help="Print particle positions at each PSO iteration (default: False)")

    ga = parser.add_argument_group("Genetic Algorithm options")
    ga.add_argument("--ga-population", type=int, default=80,
                    help="GA population size (default: 80)")
    ga.add_argument("--ga-mutation-rate", type=float, default=0.10,
                    help="GA mutation probability per gene (default: 0.10)")
    ga.add_argument("--ga-mutation-sigma", type=float, default=0.30,
                    help="GA mutation noise sigma (default: 0.30)")
    ga.add_argument("--ga-crossover-rate", type=float, default=0.90,
                    help="GA crossover probability (default: 0.90)")
    ga.add_argument("--ga-elite-fraction", type=float, default=0.10,
                    help="GA elite fraction retained each generation (default: 0.10)")
    ga.add_argument("--ga-tournament-size", type=int, default=3,
                    help="GA tournament size for parent selection (default: 3)")
    ga.add_argument("--ga-tol", type=float, default=0.01,
                    help="GA convergence tolerance (default: 0.01)")
    ga.add_argument("--ga-patience", type=int, default=20,
                    help="GA early stopping patience in generations (default: 20)")

    gwo = parser.add_argument_group("Grey Wolf Optimizer options")
    gwo.add_argument("--gwo-pack-size", type=int, default=50,
                     help="GWO pack size (default: 50)")
    gwo.add_argument("--gwo-a-start", type=float, default=2.0,
                     help="GWO initial exploration parameter a (default: 2.0)")
    gwo.add_argument("--gwo-a-end", type=float, default=0.0,
                     help="GWO final exploration parameter a (default: 0.0)")
    gwo.add_argument("--gwo-tol", type=float, default=0.01,
                     help="GWO convergence tolerance (default: 0.01)")
    gwo.add_argument("--gwo-patience", type=int, default=20,
                     help="GWO early stopping patience in iterations (default: 20)")

    hybrid = parser.add_argument_group("Hybrid PSO+NM options")
    hybrid.add_argument("--hybrid-nm-max-iters", type=int, default=200,
                        help="Hybrid PSO+NM: maximum Nelder-Mead iterations (default: 200)")
    hybrid.add_argument("--hybrid-nm-initial-step", type=float, default=0.20,
                        help="Hybrid PSO+NM: initial simplex step size (default: 0.20)")
    hybrid.add_argument("--hybrid-nm-alpha", type=float, default=1.0,
                        help="Hybrid PSO+NM: reflection coefficient alpha (default: 1.0)")
    hybrid.add_argument("--hybrid-nm-gamma", type=float, default=2.0,
                        help="Hybrid PSO+NM: expansion coefficient gamma (default: 2.0)")
    hybrid.add_argument("--hybrid-nm-rho", type=float, default=0.50,
                        help="Hybrid PSO+NM: contraction coefficient rho (default: 0.50)")
    hybrid.add_argument("--hybrid-nm-sigma", type=float, default=0.50,
                        help="Hybrid PSO+NM: shrink coefficient sigma (default: 0.50)")
    hybrid.add_argument("--hybrid-nm-tol", type=float, default=0.01,
                        help="Hybrid PSO+NM: simplex convergence tolerance (default: 0.01)")

    xtb = parser.add_argument_group("xTB backend")
    xtb.add_argument("--workers", type=int, default=4,
                     help="Number of parallel xTB workers (default: 4)")
    xtb.add_argument("--threads", type=int, default=1,
                     help="Number of threads per xTB calculation (default: 1)")
    xtb.add_argument("--method", default="gfn2",
                     help="xTB method: gfn2, gfn1, gfn0, gfnff (default: gfn2)")
    xtb.add_argument("--charge", type=int, default=0,
                     help="Molecular charge (default: 0)")
    xtb.add_argument("--mult", type=int, default=1,
                     help="Spin multiplicity (default: 1)")

    torsion_group = parser.add_argument_group("Torsions & symmetry")
    torsion_group.add_argument("--torsions-file", type=str, default=None,
                               help="Use torsions JSON file (default: disabled)")
    torsion_group.add_argument("--enable-symmetric-torsions", action="store_true",
                               help="Enable automatic detection of symmetric torsions (default: disabled)")
    torsion_group.add_argument("--symmetric-torsion-tolerance", type=float, default=10.0,
                               help="Tolerance (degrees) for symmetry detection (default: 10.0)")

    penalty = parser.add_argument_group("Penalty settings")
    penalty.add_argument("--penalty-weight", type=float, default=2.0,
                         help="Clash penalty weight (default: 2.0)")
    penalty.add_argument("--clash-cutoff", type=float, default=1.6,
                         help="Clash detection cutoff in Angstroms (default: 1.6)")
    penalty.add_argument("--intramol-penalty-weight", type=float, default=5.0,
                         help="Intramolecular clash penalty weight (default: 5.0)")
    penalty.add_argument("--intramol-cutoff", type=float, default=1.2,
                         help="Intramolecular clash cutoff in Angstroms (default: 1.2)")
    return parser.parse_args(argv)


def load_torsions(
    torsions_file: Optional[str],
    atom_types: List[str],
    coords_centered: np.ndarray,
) -> List[dict]:
    torsions_spec: List[dict] = []
    if not torsions_file:
        return torsions_spec
    raw = load_torsions_file(torsions_file)
    indexing = raw.get("indexing", "0-based")
    adj = build_bond_graph(atom_types, coords_centered)
    for t in raw.get("torsions", []):
        atoms = [int(x) for x in t["atoms"]]
        if indexing == "1-based":
            atoms = [a - 1 for a in atoms]
        b, c = atoms[1], atoms[2]
        side_atom = atoms[3] if t.get("rotate_side", "d") == "d" else atoms[0]
        fragment = find_fragment_indices(adj, b, c, side_atom)
        torsions_spec.append({
            "name": t.get("name", ""),
            "atoms": atoms,
            "b": b,
            "c": c,
            "fragment": fragment,
        })
    return torsions_spec


def compute_binding_energy(
    atom_types: List[str],
    monomer_coords_after_torsion: np.ndarray,
    transform_params: np.ndarray,
    n_layers: int,
    server: XTBServer,
) -> float:
    """Compute binding energy per interface (kJ/mol) for given parameters.
    
    Uses the provided server instance to avoid creating new instances.
    """
    stack_atoms = atom_types * n_layers
    E_mono_h = server.compute_energy(atom_types, monomer_coords_after_torsion)
    _, stack_coords, layers = build_n_layer_stack(
        atom_types, monomer_coords_after_torsion, transform_params, n_layers=n_layers
    )
    
    # Check for severe overlaps before calling xTB (critical for 10-molecule stacks)
    # With 10 layers, parameters that work for 2 layers may produce overlaps
    reject = False
    min_dist = float('inf')
    total_close = 0
    very_close = 0
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            diff = layers[i][:, None, :] - layers[j][None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            n_close = np.sum(dist < 2.0)
            n_very_close = np.sum(dist < 1.8)
            total_close += n_close
            very_close += n_very_close
            min_dist_ij = np.min(dist)
            if min_dist_ij < min_dist:
                min_dist = min_dist_ij
            if min_dist_ij < 1.6:
                reject = True
                break
        if reject:
            break
    
    # Check if there are too many close contacts for safe xTB evaluation
    max_close = 10
    max_very_close = 3
    if total_close > max_close or very_close > max_very_close:
        reject = True
    
    if reject:
        print(f"[ERROR] Final {n_layers}-molecule geometry has severe overlaps (min_dist={min_dist:.4f} Å). Cannot compute reliable binding energy.")
        return 0.0
    
    if min_dist < 1.8:
        print(f"[WARNING] Final geometry has close contacts (min_dist={min_dist:.4f} Å). Binding energy may be unreliable.")
    
    E_stack_h = server.compute_energy(stack_atoms, stack_coords)

    return ((E_stack_h - float(n_layers) * E_mono_h) * HARTREE_TO_KJMOL) / max(n_layers - 1, 1)


def main(argv: Optional[List[str]] = None) -> int:
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    args = parse_args(argv)

    configure_output_logging()
    set_thread_env_vars(args.threads)

    torsions_file = args.torsions_file

    system_info = get_system_info()
    print_run_banner(start_time_str)
    
    # Print the entire command line for the record
    command_line = " ".join(sys.argv)
    print(f"Command line: {command_line}")
    print()
    
    print_system_information_block(system_info)
    print_input_files_block(args.xyz, torsions_file)

    monomer_coords, atom_types = read_xyz_file(args.xyz)
    monomer_center = monomer_coords.mean(axis=0)
    monomer_coords_centered = monomer_coords - monomer_center

    if not report_geometry_validation(monomer_coords_centered, atom_types):
        return 1

    monomer_coords_centered = align_pi_core_to_xy(atom_types, monomer_coords_centered)
    monomer_center = np.array([monomer_center[0], monomer_center[1], 0.0])

    torsions_spec = load_torsions(torsions_file, atom_types, monomer_coords_centered)

    torsion_groups: List[List[dict]] = []
    if args.enable_symmetric_torsions and torsions_spec:
        torsion_groups = detect_symmetric_torsions(
            torsions_spec,
            atom_types,
            monomer_coords_centered,
            tolerance=args.symmetric_torsion_tolerance,
        )
    torsion_mapper = SymmetricTorsionMapper(torsion_groups, total_torsions=len(torsions_spec))

    xtb_cfg = XTBConfig(
        charge=args.charge,
        mult=args.mult,
        method=args.method,
        prefer_tblite=False,
    )
    backend_label = f"command-line xTB (binary: {xtb_cfg.executable})"
    
    # Create a single server instance for initial test and potential direct evaluation
    test_server = XTBServer(xtb_cfg)
    try:
        energy = test_server.compute_energy(atom_types, monomer_coords_centered)
    except Exception as e:
        print(f"Initial energy computation failed: {e}")
        test_server.shutdown()
        return 1

    # Always use worker pool for parallel execution during optimization
    # Keep test_server for final energy computation to avoid recreating
    actual_workers = args.workers
    worker_pool = XTBWorkerPool(xtb_cfg, actual_workers)
    # Set direct_server=None to use worker pool during optimization
    # test_server will be used only for final energy computation
    direct_server = None

    objective = BatchObjective(
        atom_types, monomer_coords_centered, torsions_spec, worker_pool,
        args.penalty_weight, args.clash_cutoff, n_layers=args.n_layer,
        intramol_penalty_weight=args.intramol_penalty_weight,
        intramol_cutoff=args.intramol_cutoff,
        torsion_mapper=torsion_mapper,
        direct_server=direct_server,
    )

    pso_dim = 7 + objective.n_torsion_dims
    pso_cfg = PSOConfig(
        swarm_size=args.swarm_size,
        max_iters=args.max_iters,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        seed=args.seed,
        verbose_every=args.verbose_every,
        tol=args.tol,
        patience=args.patience,
        print_trajectories=args.print_trajectories,
    )
    ga_cfg = GAConfig(
        population_size=args.ga_population,
        max_generations=args.max_iters,
        mutation_rate=args.ga_mutation_rate,
        mutation_sigma=args.ga_mutation_sigma,
        crossover_rate=args.ga_crossover_rate,
        elite_fraction=args.ga_elite_fraction,
        tournament_size=args.ga_tournament_size,
        seed=args.seed,
        verbose_every=args.verbose_every,
        tol=args.ga_tol,
        patience=args.ga_patience,
    )
    gwo_cfg = GWOConfig(
        pack_size=args.gwo_pack_size,
        max_iters=args.max_iters,
        a_start=args.gwo_a_start,
        a_end=args.gwo_a_end,
        seed=args.seed,
        verbose_every=args.verbose_every,
        tol=args.gwo_tol,
        patience=args.gwo_patience,
    )
    pso_nm_cfg = PSONMConfig(
        pso=pso_cfg,
        nm_max_iters=args.hybrid_nm_max_iters,
        nm_alpha=args.hybrid_nm_alpha,
        nm_gamma=args.hybrid_nm_gamma,
        nm_rho=args.hybrid_nm_rho,
        nm_sigma=args.hybrid_nm_sigma,
        nm_initial_step=args.hybrid_nm_initial_step,
        nm_tol=args.hybrid_nm_tol,
    )
    optimizer_request = OptimizerRequest(
        method=args.optimizer,
        pso=pso_cfg,
        ga=ga_cfg,
        gwo=gwo_cfg,
        pso_nm=pso_nm_cfg,
    )
    optimizer_handle = create_optimizer(optimizer_request, dim=pso_dim)
    optimizer_obj = optimizer_handle.optimizer
    trajectory_file = None
    trajectory_writer = None
    save_trajectories = optimizer_handle.supports_trajectories
    if optimizer_handle.method == "pso":
        trajectory_file, trajectory_writer = setup_trajectory_logging(optimizer_handle.config, pso_dim)
        optimizer_obj.trajectory_writer = trajectory_writer
        save_trajectories = optimizer_handle.config.print_trajectories
    elif optimizer_handle.method == "pso-nm":
        trajectory_file, trajectory_writer = setup_trajectory_logging(optimizer_handle.config.pso, pso_dim)
        optimizer_obj.trajectory_writer = trajectory_writer
        save_trajectories = optimizer_handle.config.pso.print_trajectories

    print_molecular_system_block(atom_types, torsions_spec)
    print_xtb_configuration_block(xtb_cfg, args.workers, args.threads, backend_label)
    if optimizer_handle.method == "pso":
        print_pso_configuration_block(optimizer_handle.config)
    elif optimizer_handle.method == "ga":
        print_ga_configuration_block(optimizer_handle.config)
    elif optimizer_handle.method == "gwo":
        print_gwo_configuration_block(optimizer_handle.config)
    elif optimizer_handle.method == "pso-nm":
        print_pso_nm_configuration_block(optimizer_handle.config)
    symmetry_tolerance = (
        args.symmetric_torsion_tolerance if args.enable_symmetric_torsions and torsions_spec else None
    )
    print_optimization_settings_block(
        pso_dim,
        objective.n_torsion_dims,
        len(torsions_spec),
        symmetry_tolerance,
        args.n_layer,
        args.penalty_weight,
        args.clash_cutoff,
        args.intramol_penalty_weight,
        args.intramol_cutoff,
    )

    print("┌─ OPTIMIZATION PROGRESS " + "─" * 51)
    method_label = describe_method(optimizer_handle.method)
    print(f"│ Starting {method_label} optimization...")
    print("│")
    try:
        best_params, best_val = optimizer_obj.optimize(objective)
    finally:
        if worker_pool:
            worker_pool.shutdown()
        if trajectory_file:
            trajectory_file.close()
        # test_server will be shut down after final energy computation

    print("│")
    print("│ Optimization completed successfully!")
    print("└" + "─" * 79)
    print()

    final_coords, _ = objective._apply_torsions(best_params[7:])
    # Reuse the test_server for final energy computation
    final_server = test_server
    try:
        binding_energy = compute_binding_energy(
            atom_types,
            final_coords,
            best_params[:7],
            args.n_layer,
            final_server,
        )
    finally:
        # Shutdown test_server after final energy computation
        test_server.shutdown()
    patience_delta = getattr(optimizer_obj, "last_patience_delta", None)

    print("┌─ OPTIMIZATION RESULTS " + "─" * 52)
    print(f"│ Best objective:    {best_val:.6f} kJ/mol")
    print(f"│ Binding energy:    {binding_energy:.6f} kJ/mol per interface")
    print("│ Best parameters:")
    param_summary = summarize_params(best_params, objective.torsion_mapper)
    for line in param_summary.split('  |  '):
        print(f"│   {line.strip()}")
    print("└" + "─" * 79)
    print()

    with open("optimization_results.txt", "w") as f:
        f.write(f"Best objective: {best_val:.6f} kJ/mol\n")
        f.write(f"Binding energy (per interface): {binding_energy:.6f} kJ/mol\n")
        f.write(f"Best params: {summarize_params(best_params, objective.torsion_mapper)}\n")
        full_best_torsions = objective.get_full_torsions_from_best(best_params)
        if full_best_torsions.size > 0:
            f.write("Full torsion angles (degrees): " +
                    ", ".join(f"{np.degrees(t):.3f}" for t in full_best_torsions) + "\n")
            write_xyz_file("molecule_after_torsion.xyz", final_coords + monomer_center,
                     atom_types, "Monomer after applying torsions")

    n_molecules = 10
    _, stack_coords, _ = build_n_layer_stack(
        atom_types, final_coords, best_params[:7], n_layers=n_molecules
    )
    stack_atoms = atom_types * n_molecules
    write_xyz_file(f"molecular_stack_{n_molecules}molecules.xyz",
                   stack_coords + monomer_center, stack_atoms,
                   f"{n_molecules}-layer stack")

    print_output_files_block(n_molecules, save_trajectories)

    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed = end_time - start_time
    elapsed_seconds = elapsed.total_seconds()
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("═" * 80)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY")
    print(f"End time:     {end_time_str}")
    print(f"Total time:   {elapsed_seconds:.3f} seconds")
    print("═" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
