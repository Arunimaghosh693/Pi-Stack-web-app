#!/usr/bin/env python3
"""Cached optimized hyperparameter optimization with result memoization."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import optuna
except ImportError:
    optuna = None


class HyperparameterCache:
    """Persistent cache for hyperparameter evaluation results."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "hyperparameter_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for caching."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hyperparameter_results (
                    cache_key TEXT PRIMARY KEY,
                    molecule_name TEXT,
                    optimizer_method TEXT,
                    hyperparameters TEXT,
                    objective_value REAL,
                    evaluation_time REAL,
                    timestamp TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_molecule_optimizer 
                ON hyperparameter_results(molecule_name, optimizer_method)
            """)
    
    def _make_cache_key(
        self, 
        molecule_name: str, 
        optimizer_method: str, 
        params: Dict[str, float],
        molecule_hash: str = "",
        base_args_hash: str = ""
    ) -> str:
        """Create deterministic cache key."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        key_data = f"{molecule_name}:{optimizer_method}:{sorted_params}:{molecule_hash}:{base_args_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(
        self, 
        molecule_name: str, 
        optimizer_method: str, 
        params: Dict[str, float],
        molecule_hash: str = "",
        base_args_hash: str = ""
    ) -> Optional[float]:
        """Get cached result if available."""
        cache_key = self._make_cache_key(molecule_name, optimizer_method, params, molecule_hash, base_args_hash)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT objective_value FROM hyperparameter_results WHERE cache_key = ?",
                (cache_key,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def put(
        self,
        molecule_name: str,
        optimizer_method: str,
        params: Dict[str, float],
        objective_value: float,
        evaluation_time: float = 0.0,
        molecule_hash: str = "",
        base_args_hash: str = "",
        metadata: Dict = None
    ):
        """Store result in cache."""
        cache_key = self._make_cache_key(molecule_name, optimizer_method, params, molecule_hash, base_args_hash)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO hyperparameter_results 
                (cache_key, molecule_name, optimizer_method, hyperparameters, 
                 objective_value, evaluation_time, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """, (
                cache_key,
                molecule_name,
                optimizer_method,
                json.dumps(params, sort_keys=True),
                objective_value,
                evaluation_time,
                json.dumps(metadata or {})
            ))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM hyperparameter_results")
            total_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT molecule_name, optimizer_method, COUNT(*) 
                FROM hyperparameter_results 
                GROUP BY molecule_name, optimizer_method
            """)
            by_molecule_optimizer = cursor.fetchall()
            
            cursor = conn.execute("""
                SELECT AVG(evaluation_time), MIN(evaluation_time), MAX(evaluation_time)
                FROM hyperparameter_results WHERE evaluation_time > 0
            """)
            time_stats = cursor.fetchone()
            
        return {
            "total_entries": total_entries,
            "by_molecule_optimizer": by_molecule_optimizer,
            "avg_evaluation_time": time_stats[0] if time_stats[0] else 0,
            "min_evaluation_time": time_stats[1] if time_stats[1] else 0,
            "max_evaluation_time": time_stats[2] if time_stats[2] else 0,
        }
    
    def clear_molecule(self, molecule_name: str):
        """Clear cache entries for a specific molecule."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM hyperparameter_results WHERE molecule_name = ?",
                (molecule_name,)
            )


@dataclass(frozen=True)
class HyperparamSpec:
    """Simplified hyperparameter specification."""
    defaults: Dict[str, float]
    sampler: Any
    formatter: Any

    def sample(self, trial: Any) -> Dict[str, float]:
        return self.sampler(trial)

    def format_cli(self, params: Dict[str, float]) -> List[str]:
        return self.formatter(params)


def _pso_sampler(trial: Any) -> Dict[str, float]:
    return {
        "swarm_size": trial.suggest_int("swarm_size", 30, 120, step=10),
        "inertia": trial.suggest_float("inertia", 0.4, 0.9, step=0.05),
        "cognitive": trial.suggest_float("cognitive", 1.0, 2.5, step=0.1),
        "social": trial.suggest_float("social", 1.0, 2.5, step=0.1),
    }


def _pso_formatter(params: Dict[str, float]) -> List[str]:
    return [
        f"--swarm-size={int(round(params['swarm_size']))}",
        f"--inertia={params['inertia']}",
        f"--cognitive={params['cognitive']}",
        f"--social={params['social']}",
    ]


def _ga_sampler(trial: Any) -> Dict[str, float]:
    return {
        "ga_population": trial.suggest_int("ga_population", 50, 200, step=10),
        "ga_mutation_rate": trial.suggest_float("ga_mutation_rate", 0.01, 0.4, step=0.05),
        "ga_mutation_sigma": trial.suggest_float("ga_mutation_sigma", 0.05, 0.6, step=0.05),
        "ga_crossover_rate": trial.suggest_float("ga_crossover_rate", 0.6, 1.0, step=0.05),
        "ga_elite_fraction": trial.suggest_float("ga_elite_fraction", 0.05, 0.2, step=0.01),
        "ga_tournament_size": trial.suggest_int("ga_tournament_size", 2, 8, step=1),
    }


def _ga_formatter(params: Dict[str, float]) -> List[str]:
    return [
        f"--ga-population={int(round(params['ga_population']))}",
        f"--ga-mutation-rate={params['ga_mutation_rate']}",
        f"--ga-mutation-sigma={params['ga_mutation_sigma']}",
        f"--ga-crossover-rate={params['ga_crossover_rate']}",
        f"--ga-elite-fraction={params['ga_elite_fraction']}",
        f"--ga-tournament-size={int(round(params['ga_tournament_size']))}",
    ]


def _gwo_sampler(trial: Any) -> Dict[str, float]:
    return {
        "gwo_pack_size": trial.suggest_int("gwo_pack_size", 20, 120, step=5),
        "gwo_a_start": trial.suggest_float("gwo_a_start", 0.5, 3.0, step=0.1),
        "gwo_a_end": trial.suggest_float("gwo_a_end", 0.0, 1.0, step=0.1),
    }


def _gwo_formatter(params: Dict[str, float]) -> List[str]:
    return [
        f"--gwo-pack-size={int(round(params['gwo_pack_size']))}",
        f"--gwo-a-start={params['gwo_a_start']}",
        f"--gwo-a-end={params['gwo_a_end']}",
    ]


def _pso_nm_sampler(trial: Any) -> Dict[str, float]:
    """Sample hyperparameters for PSO-NM hybrid optimizer."""
    return {
        # PSO parameters
        "swarm_size": trial.suggest_int("swarm_size", 30, 120, step=10),
        "inertia": trial.suggest_float("inertia", 0.4, 0.9, step=0.05),
        "cognitive": trial.suggest_float("cognitive", 1.0, 2.5, step=0.1),
        "social": trial.suggest_float("social", 1.0, 2.5, step=0.1),
        # Nelder-Mead parameters
        "hybrid_nm_max_iters": trial.suggest_int("hybrid_nm_max_iters", 50, 400, step=10),
        "hybrid_nm_initial_step": trial.suggest_float("hybrid_nm_initial_step", 0.05, 0.8, step=0.05),
        "hybrid_nm_alpha": trial.suggest_float("hybrid_nm_alpha", 0.5, 2.0, step=0.1),
        "hybrid_nm_gamma": trial.suggest_float("hybrid_nm_gamma", 1.0, 3.5, step=0.1),
        "hybrid_nm_rho": trial.suggest_float("hybrid_nm_rho", 0.1, 0.9, step=0.1),
        "hybrid_nm_sigma": trial.suggest_float("hybrid_nm_sigma", 0.1, 0.9, step=0.1),
        "hybrid_nm_tol": trial.suggest_float("hybrid_nm_tol", 1e-5, 5e-3, log=True),
    }


def _pso_nm_formatter(params: Dict[str, float]) -> List[str]:
    """Format PSO-NM hyperparameters for CLI."""
    args = [
        # PSO parameters
        f"--swarm-size={int(round(params['swarm_size']))}",
        f"--inertia={params['inertia']:.2f}",
        f"--cognitive={params['cognitive']:.2f}",
        f"--social={params['social']:.2f}",
        # Nelder-Mead parameters
        f"--hybrid-nm-max-iters={int(round(params['hybrid_nm_max_iters']))}",
        f"--hybrid-nm-initial-step={params['hybrid_nm_initial_step']:.2f}",
        f"--hybrid-nm-alpha={params['hybrid_nm_alpha']:.2f}",
        f"--hybrid-nm-gamma={params['hybrid_nm_gamma']:.2f}",
        f"--hybrid-nm-rho={params['hybrid_nm_rho']:.2f}",
        f"--hybrid-nm-sigma={params['hybrid_nm_sigma']:.2f}",
        f"--hybrid-nm-tol={params['hybrid_nm_tol']:.6f}",
    ]
    return args


HYPERPARAM_SPECS = {
    "pso": HyperparamSpec(
        defaults={
            "swarm_size": 60,  # Range: 30-120, step=10
            "inertia": 0.65,    # Range: 0.4-0.9, step=0.05 (0.65 is midpoint)
            "cognitive": 1.5,   # Range: 1.0-2.5, step=0.1
            "social": 1.5,      # Range: 1.0-2.5, step=0.1
        },
        sampler=_pso_sampler,
        formatter=_pso_formatter,
    ),
    "ga": HyperparamSpec(
        defaults={
            "ga_population": 80,        # Range: 50-200, step=10
            "ga_mutation_rate": 0.10,   # Range: 0.01-0.4, step=0.05
            "ga_mutation_sigma": 0.30,  # Range: 0.05-0.6, step=0.05
            "ga_crossover_rate": 0.80,  # Range: 0.6-1.0, step=0.05
            "ga_elite_fraction": 0.10,   # Range: 0.05-0.2, step=0.01 (midpoint)
            "ga_tournament_size": 3,    # Range: 2-8, step=1
        },
        sampler=_ga_sampler,
        formatter=_ga_formatter,
    ),
    "gwo": HyperparamSpec(
        defaults={
            "gwo_pack_size": 50,  # Range: 20-120, step=5
            "gwo_a_start": 2.0,   # Range: 0.5-3.0, step=0.1
            "gwo_a_end": 0.0,     # Range: 0.0-1.0, step=0.1
        },
        sampler=_gwo_sampler,
        formatter=_gwo_formatter,
    ),
    "pso-nm": HyperparamSpec(
        defaults={
            # PSO parameters
            "swarm_size": 60,      # Range: 30-120, step=10
            "inertia": 0.65,       # Range: 0.4-0.9, step=0.05 (midpoint)
            "cognitive": 1.5,      # Range: 1.0-2.5, step=0.1
            "social": 1.5,         # Range: 1.0-2.5, step=0.1
            # Nelder-Mead parameters
            "hybrid_nm_max_iters": 200,      # Range: 50-400, step=10
            "hybrid_nm_initial_step": 0.20,  # Range: 0.05-0.8, step=0.05
            "hybrid_nm_alpha": 1.0,          # Range: 0.5-2.0, step=0.1
            "hybrid_nm_gamma": 2.0,          # Range: 1.0-3.5, step=0.1
            "hybrid_nm_rho": 0.5,            # Range: 0.1-0.9, step=0.1
            "hybrid_nm_sigma": 0.5,          # Range: 0.1-0.9, step=0.1
            "hybrid_nm_tol": 0.001,          # Range: 1e-5 to 5e-3 (log scale), using 0.001 as reasonable default
        },
        sampler=_pso_nm_sampler,
        formatter=_pso_nm_formatter,
    ),
}


@dataclass
class MoleculeCase:
    """Molecule case for optimization."""
    name: str
    xyz: Path
    torsions: Optional[Path]
    
    def get_hash(self) -> str:
        """Generate hash of molecule files for cache validation."""
        data = {
            "xyz_content": self.xyz.read_text() if self.xyz.exists() else "",
            "torsions_content": self.torsions.read_text() if self.torsions and self.torsions.exists() else "",
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()


class CachedHyperparameterOptimizer:
    """Cached hyperparameter optimizer with result memoization."""
    
    def __init__(self, cache_dir: Path = None, keep_run_dirs: bool = False):
        self.cache = HyperparameterCache(cache_dir or Path("output/cache_hyperopt"))
        self.cache_hits = 0
        self.cache_misses = 0
        self.keep_run_dirs = bool(keep_run_dirs)
        
    def get_base_args_hash(self, base_args: List[str]) -> str:
        """Generate hash of base arguments for cache key."""
        return hashlib.md5(json.dumps(sorted(base_args or []), sort_keys=True).encode()).hexdigest()
    
    def run_pi_stack_cached(
        self,
        stack_script: Path,
        molecule: MoleculeCase,
        params: Dict[str, float],
        base_args: List[str],
        optimizer: str,
        run_root: Path,
        hyperparam_spec: HyperparamSpec,
        reduced_iterations: int = 50,
    ) -> float:
        """Cached pi-stack execution with memoization."""
        
        # Generate cache keys
        molecule_hash = molecule.get_hash()
        base_args_hash = self.get_base_args_hash(base_args)
        
        # Check cache first
        cached_result = self.cache.get(
            molecule.name, 
            optimizer, 
            params, 
            molecule_hash, 
            base_args_hash
        )
        
        if cached_result is not None:
            self.cache_hits += 1
            return cached_result
        
        # Cache miss - compute result
        self.cache_misses += 1
        start_time = time.time()
        
        # Execute pi-stack optimization
        run_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        run_dir = run_root / f"trial-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [sys.executable, str(stack_script.resolve()), str(molecule.xyz.resolve())]
        if molecule.torsions is not None and molecule.torsions.exists():
            cmd.extend(["--torsions-file", str(molecule.torsions.resolve())])
        cmd.extend(["--optimizer", optimizer])
        cmd.extend(["--max-iters", str(reduced_iterations)])
        
        if base_args:
            cmd.extend(base_args)
        cmd.extend(hyperparam_spec.format_cli(params))

        completed = None
        try:
            completed = subprocess.run(
                cmd,
                cwd=run_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            result_path = run_dir / "optimization_results.txt"
            if completed.returncode != 0:
                print(f"Pi-stack failed for {molecule.name} with return code {completed.returncode}")
                print(f"Full stdout:\n{completed.stdout}")
                print(f"Full stderr:\n{completed.stderr}")
                objective_value = None
            elif not result_path.exists():
                print(f"optimization_results.txt not found for {molecule.name} in {run_dir}")
                print(f"Full stdout:\n{completed.stdout}")
                print(f"Full stderr:\n{completed.stderr}")
                objective_value = None
            else:
                try:
                    objective_value = self.parse_best_objective(result_path)
                except Exception as e:
                    print(f"Failed to parse results for {molecule.name}: {e}")
                    print(f"Full stdout:\n{completed.stdout}")
                    print(f"Full stderr:\n{completed.stderr}")
                    print(f"optimization_results.txt content:\n{result_path.read_text() if result_path.exists() else 'File missing'}")
                    objective_value = None

            # Save stdout/stderr to files for debugging
            try:
                (run_dir / "stdout.txt").write_text(completed.stdout if completed and completed.stdout is not None else "")
            except Exception:
                pass
            try:
                (run_dir / "stderr.txt").write_text(completed.stderr if completed and completed.stderr is not None else "")
            except Exception:
                pass

            # Clean up immediately unless user requested to keep run dirs
            import shutil
            if not getattr(self, "keep_run_dirs", False):
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception as e:
            print(f"Exception occurred for {molecule.name}: {e}")
            # Attempt to save any captured output
            try:
                if completed is not None:
                    (run_dir / "stdout.txt").write_text(completed.stdout if completed.stdout is not None else "")
                    (run_dir / "stderr.txt").write_text(completed.stderr if completed.stderr is not None else "")
            except Exception:
                pass
            objective_value = None
            import shutil
            if not getattr(self, "keep_run_dirs", False):
                shutil.rmtree(run_dir, ignore_errors=True)

        # If we still don't have a valid objective value, return penalty
        if objective_value is None:
            objective_value = 1000000.0 + abs(hash(str(params))) % 10000
        
        evaluation_time = time.time() - start_time
        
        # Store in cache
        self.cache.put(
            molecule.name,
            optimizer,
            params,
            objective_value,
            evaluation_time,
            molecule_hash,
            base_args_hash,
            {"base_args": base_args, "reduced_iterations": reduced_iterations}
        )
        
        return objective_value
    
    def parse_best_objective(self, results_file: Path) -> float:
        """Parse best objective from results file."""
        if not results_file.exists():
            raise FileNotFoundError(f"Missing optimization_results.txt at {results_file}")
        
        content = results_file.read_text()
        for line in content.splitlines():
            if line.lower().startswith("best objective"):
                try:
                    value = line.split(":", 1)[1].strip().split()[0]
                    return float(value)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing objective line '{line}': {e}")
                    continue
        
        # If we can't find "Best objective", try other patterns
        for line in content.splitlines():
            if "objective" in line.lower() and ":" in line:
                try:
                    value = line.split(":", 1)[1].strip().split()[0]
                    result = float(value)
                    print(f"Found alternative objective: {result} from line: {line}")
                    return result
                except (IndexError, ValueError):
                    continue
        
        print(f"Could not parse objective from results file. Content:\n{content}")
        raise ValueError("Could not find 'Best objective' in optimization_results.txt")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        cache_stats = self.cache.get_statistics()
        total_evaluations = self.cache_hits + self.cache_misses
        return {
            **cache_stats,
            "session_cache_hits": self.cache_hits,
            "session_cache_misses": self.cache_misses,
            "session_total_evaluations": total_evaluations,
            "session_hit_rate": self.cache_hits / total_evaluations if total_evaluations > 0 else 0.0,
        }


def discover_molecules(root: Path, xyz_name: str, torsions_name: str) -> List[MoleculeCase]:
    """Discover molecules in directory structure."""
    cases: List[MoleculeCase] = []
    if not root.exists():
        return cases
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        xyz_path = child / xyz_name
        torsions_path = child / torsions_name
        if not xyz_path.exists():
            continue
        torsions = torsions_path if torsions_path.exists() else None
        cases.append(MoleculeCase(name=child.name, xyz=xyz_path.resolve(), torsions=torsions))
    return cases


def create_cached_objective(
    optimizer: CachedHyperparameterOptimizer,
    stack_script: Path,
    molecules: List[MoleculeCase],
    base_args: List[str],
    optimizer_method: str,
    runs_dir: Path,
    hyperparam_spec: HyperparamSpec,
    reduced_iterations: int = 50,
):
    """Create cached objective function."""
    
    def objective(trial: Any) -> float:
        params = hyperparam_spec.sample(trial)
        
        # Evaluate all molecules sequentially
        objectives = []
        for molecule in molecules:
            obj_value = optimizer.run_pi_stack_cached(
                stack_script,
                molecule,
                params,
                base_args,
                optimizer_method,
                runs_dir / molecule.name,
                hyperparam_spec,
                reduced_iterations,
            )
            objectives.append(obj_value)
        
        # Return average objective
        avg_objective = sum(objectives) / len(objectives)
        
        # Store per-molecule breakdown
        if len(molecules) > 1:
            trial.set_user_attr(
                "per_molecule_objectives", 
                {mol.name: obj for mol, obj in zip(molecules, objectives)}
            )
        
        return avg_objective
    
    return objective


def main():
    """Main entry point for cached hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Cached hyperparameter optimization with result memoization"
    )
    parser.add_argument("--molecules-root", default="input",
                        help="Directory containing per-molecule subdirectories (default: input)")
    parser.add_argument("--stack-script", default="../pi-stack-generator.py",
                        help="Path to pi-stack-generator.py script (default: ../pi-stack-generator.py)")
    parser.add_argument("--optimizer", default="pso", choices=["pso", "ga", "gwo", "pso-nm"],
                        help="Optimizer type: pso, ga, gwo, or pso-nm (default: pso)")
    parser.add_argument("--trials-per-molecule", type=int, default=50,
                        help="Number of Optuna trials per molecule (default: 50)")
    parser.add_argument("--joint-study", action="store_true",
                        help="Enable joint study mode (optimize across all molecules) (default: False)")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress bars and detailed output (default: False)")
    parser.add_argument("--study-dir", default="output/studies",
                        help="Directory for Optuna SQLite databases (default: output/studies)")
    parser.add_argument("--results-dir", default="output/results",
                        help="Directory for JSON result summaries (default: output/results)")
    parser.add_argument("--runs-dir", default="output/runs",
                        help="Directory for trial execution outputs (default: output/runs)")
    parser.add_argument("--cache-dir", default="output/cache_hyperopt",
                        help="Directory for persistent result cache (default: output/cache_hyperopt)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cache before starting (default: False)")
    parser.add_argument("--cache-stats", action="store_true",
                        help="Show cache statistics and exit (default: False)")
    parser.add_argument("--reduced-iterations", type=int, default=50,
                        help="Number of iterations for hyperparameter trials (default: 50)")
    parser.add_argument("--base-args", nargs=argparse.REMAINDER,
                        help="Additional arguments forwarded to pi-stack-generator.py. NOTE: This must be the last argument on the command line (default: None)")
    parser.add_argument("--keep-run-dirs", action="store_true",
                        help="Keep trial run directories and logs for debugging (default: False)")
    
    args = parser.parse_args()
    
    # Check if help was requested in base_args
    if args.base_args and ("-h" in args.base_args or "--help" in args.base_args):
        parser.print_help()
        raise SystemExit(0)
    
    # Validate that --base-args is at the end if used
    if args.base_args:
        # List of known hyperopt arguments
        known_hyperopt_args = {
            "--molecules-root", "--stack-script", "--optimizer", "--trials-per-molecule",
            "--joint-study", "--progress", "--study-dir", "--results-dir", "--runs-dir",
            "--cache-dir", "--clear-cache", "--cache-stats", "--reduced-iterations",
            "--keep-run-dirs"
        }
        
        # Check if any known hyperopt arguments appear in base_args
        base_args_set = set(args.base_args)
        conflicting_args = base_args_set.intersection(known_hyperopt_args)
        
        if conflicting_args:
            print("Error: --base-args must be the last argument on the command line.")
            print(f"Found hyperopt arguments after --base-args: {', '.join(sorted(conflicting_args))}")
            print("\nPlease move --base-args to the end of the command line.")
            print("Example:")
            print("  python hyperopt.py --optimizer pso --trials-per-molecule 50 --base-args --workers 4")
            raise SystemExit(1)
    
    if optuna is None:
        raise SystemExit("Optuna is required. Install with: pip install optuna")
    
    # Initialize cached optimizer
    cache_optimizer = CachedHyperparameterOptimizer(Path(args.cache_dir), keep_run_dirs=args.keep_run_dirs)
    
    # Handle cache operations
    if args.cache_stats:
        stats = cache_optimizer.get_cache_stats()
        print("📊 Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Average evaluation time: {stats['avg_evaluation_time']:.1f}s")
        if stats['by_molecule_optimizer']:
            print("  By molecule/optimizer:")
            for mol, opt, count in stats['by_molecule_optimizer']:
                print(f"    {mol} ({opt}): {count} entries")
        return
    
    if args.clear_cache:
        import shutil
        shutil.rmtree(Path(args.cache_dir), ignore_errors=True)
        print("🗑️  Cache cleared")
    
    print("🗄️  Cached Hyperparameter Optimization")
    print("=" * 45)
    
    # Setup paths
    molecules_root = Path(args.molecules_root)
    stack_script = Path(args.stack_script).resolve()
    study_dir = Path(args.study_dir)
    results_dir = Path(args.results_dir)
    runs_dir = Path(args.runs_dir)
    
    # Create directories
    study_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    
    if not stack_script.exists():
        raise SystemExit(f"pi-stack script not found at {stack_script}")
    
    # Discover molecules
    molecules = discover_molecules(molecules_root, "monomer.xyz", "torsions.json")
    if not molecules:
        raise SystemExit(f"No molecules found in {molecules_root}")
    
    print(f"Found {len(molecules)} molecules: {[m.name for m in molecules]}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Base args: {args.base_args or 'None'}")
    
    # Setup Optuna
    sampler = optuna.samplers.TPESampler(seed=42)
    hyperparam_spec = HYPERPARAM_SPECS[args.optimizer]
    
    if args.joint_study:
        # Joint study mode
        study_name = f"cached_joint_{args.optimizer}"
        storage = f"sqlite:///{study_dir}/cached_joint_{args.optimizer}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        
        # Enqueue default parameters
        study.enqueue_trial(hyperparam_spec.defaults)
        
        # Create objective
        objective = create_cached_objective(
            cache_optimizer,
            stack_script,
            molecules,
            args.base_args or [],
            args.optimizer,
            runs_dir,
            hyperparam_spec,
            args.reduced_iterations,
        )
        
        print(f"\n🎯 Starting cached joint optimization with {args.trials_per_molecule} trials")
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=args.trials_per_molecule,
            show_progress_bar=args.progress,
        )
        
        elapsed = time.time() - start_time
        cache_stats = cache_optimizer.get_cache_stats()
        
        print(f"⏱️  Optimization completed in {elapsed:.1f} seconds")
        print(f"🗄️  Cache hit rate: {cache_stats['session_hit_rate']:.1%}")
        print(f"📊 Cache hits/misses: {cache_stats['session_cache_hits']}/{cache_stats['session_cache_misses']}")
        
        # Save results
        if study.best_trial:
            best_params = study.best_trial.params
            best_value = study.best_trial.value
            
            results = {
                "mode": "joint_cached",
                "optimizer": args.optimizer,
                "molecules": [mol.name for mol in molecules],
                "best_objective": best_value,
                "best_hyperparameters": best_params,
                "n_trials": len(study.trials),
                "optimization_time": elapsed,
                "trials_per_second": len(study.trials) / elapsed,
                "cache_stats": cache_stats,
            }
            
            if len(molecules) > 1:
                per_mol = study.best_trial.user_attrs.get("per_molecule_objectives", {})
                results["per_molecule_objectives"] = per_mol
            
            results_file = results_dir / f"cached_joint_{args.optimizer}.json"
            results_file.write_text(json.dumps(results, indent=2))
            
            print(f"\n✅ Best objective: {best_value:.4f}")
            print(f"📁 Results saved to: {results_file}")
            
    else:
        # Sequential mode
        print(f"\n🔄 Starting cached sequential optimization")
        prev_best = dict(hyperparam_spec.defaults)
        
        for i, molecule in enumerate(molecules):
            print(f"\n[{i+1}/{len(molecules)}] Optimizing {molecule.name}")
            
            study_name = f"cached_{molecule.name}_{args.optimizer}"
            storage = f"sqlite:///{study_dir}/cached_{molecule.name}.db"
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",
                sampler=sampler,
                load_if_exists=True,
            )
            
            # Warm start with previous best
            study.enqueue_trial(prev_best)
            
            objective = create_cached_objective(
                cache_optimizer,
                stack_script,
                [molecule],
                args.base_args or [],
                args.optimizer,
                runs_dir,
                hyperparam_spec,
                args.reduced_iterations,
            )
            
            start_time = time.time()
            study.optimize(
                objective,
                n_trials=args.trials_per_molecule,
                show_progress_bar=args.progress,
            )
            elapsed = time.time() - start_time
            cache_stats = cache_optimizer.get_cache_stats()
            
            if study.best_trial:
                best_params = study.best_trial.params
                best_value = study.best_trial.value
                prev_best = best_params  # Warm start for next molecule
                
                results = {
                    "molecule": molecule.name,
                    "optimizer": args.optimizer,
                    "best_objective": best_value,
                    "best_hyperparameters": best_params,
                    "n_trials": len(study.trials),
                    "optimization_time": elapsed,
                    "trials_per_second": len(study.trials) / elapsed,
                    "cache_stats": cache_stats,
                }
                
                results_file = results_dir / f"cached_{molecule.name}.json"
                results_file.write_text(json.dumps(results, indent=2))
                
                print(f"✅ {molecule.name}: {best_value:.4f} "
                      f"({elapsed:.1f}s, {len(study.trials)/elapsed:.1f} trials/s)")
                print(f"🗄️  Cache hit rate: {cache_stats['session_hit_rate']:.1%}")
    
    # Final cache statistics
    final_stats = cache_optimizer.get_cache_stats()
    print(f"\n📈 Final Cache Statistics:")
    print(f"  Total cache entries: {final_stats['total_entries']}")
    print(f"  Session evaluations: {final_stats['session_total_evaluations']}")
    print(f"  Session hit rate: {final_stats['session_hit_rate']:.1%}")
    if final_stats['session_hit_rate'] > 0:
        time_saved = final_stats['session_cache_hits'] * final_stats['avg_evaluation_time']
        print(f"  Estimated time saved: {time_saved:.1f} seconds")


if __name__ == "__main__":
    main()
