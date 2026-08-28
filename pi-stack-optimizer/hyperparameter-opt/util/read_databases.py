#!/usr/bin/env python3
"""Helper script to read and analyze hyperparameter optimization databases."""
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Cannot read study databases.")


def detect_db_type(db_path: Path) -> str:
    """Detect if database is a cache database or Optuna study database."""
    if not db_path.exists():
        return "not_found"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if "hyperparameter_results" in tables:
        return "cache"
    elif "studies" in tables or "trials" in tables or "study_system_attrs" in tables:
        return "optuna"
    else:
        return "unknown"


def read_cache_db(db_path: Path) -> Dict[str, Any]:
    """Read the hyperparameter cache database."""
    if not db_path.exists():
        print(f"Cache database not found: {db_path}")
        return {}
    
    # Check if this is actually a cache database
    db_type = detect_db_type(db_path)
    if db_type != "cache":
        return {"error": f"Database is not a cache database (detected type: {db_type})"}
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    
    # Get all entries
    cursor = conn.execute("""
        SELECT molecule_name, optimizer_method, hyperparameters, 
               objective_value, evaluation_time, timestamp, metadata
        FROM hyperparameter_results
        ORDER BY timestamp DESC
    """)
    
    entries = []
    for row in cursor.fetchall():
        entry = {
            "molecule": row["molecule_name"],
            "optimizer": row["optimizer_method"],
            "hyperparameters": json.loads(row["hyperparameters"]),
            "objective_value": row["objective_value"],
            "evaluation_time": row["evaluation_time"],
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
        entries.append(entry)
    
    # Get statistics
    cursor = conn.execute("SELECT COUNT(*) FROM hyperparameter_results")
    total = cursor.fetchone()[0]
    
    cursor = conn.execute("""
        SELECT molecule_name, optimizer_method, COUNT(*) as count,
               AVG(objective_value) as avg_obj, MIN(objective_value) as min_obj
        FROM hyperparameter_results
        GROUP BY molecule_name, optimizer_method
    """)
    
    stats = []
    for row in cursor.fetchall():
        stats.append({
            "molecule": row["molecule_name"],
            "optimizer": row["optimizer_method"],
            "count": row["count"],
            "avg_objective": row["avg_obj"],
            "min_objective": row["min_obj"]
        })
    
    conn.close()
    
    return {
        "total_entries": total,
        "statistics": stats,
        "entries": entries
    }


def list_optuna_studies(db_path: Path) -> List[str]:
    """List all study names in an Optuna database."""
    if not OPTUNA_AVAILABLE:
        return []
    
    if not db_path.exists():
        return []
    
    storage = f"sqlite:///{db_path.resolve()}"
    
    try:
        # Use Optuna's get_all_study_summaries to list all studies
        summaries = optuna.get_all_study_summaries(storage=storage)
        return [summary.study_name for summary in summaries]
    except Exception:
        # Fallback: try to query the database directly
        try:
            conn = sqlite3.connect(db_path)
            # Check if studies table exists and has study_name column
            cursor = conn.execute("PRAGMA table_info(studies)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if "study_name" in columns:
                cursor = conn.execute("SELECT DISTINCT study_name FROM studies")
                study_names = [row[0] for row in cursor.fetchall() if row[0]]
                conn.close()
                return study_names
            elif "study_id" in columns:
                # Older Optuna versions might use study_id
                cursor = conn.execute("SELECT DISTINCT study_id FROM studies")
                study_ids = [str(row[0]) for row in cursor.fetchall()]
                conn.close()
                # Try common naming patterns based on filename
                base_name = db_path.stem.replace("cached_", "").replace(".db", "")
                possible_names = [
                    f"cached_{base_name}",
                    base_name,
                    f"cached_joint_{base_name.split('_')[-1]}" if "joint" in base_name else None,
                ]
                return [name for name in possible_names if name]
            else:
                conn.close()
                return []
        except Exception as e:
            return []


def read_optuna_study(db_path: Path, study_name: str = None) -> Dict[str, Any]:
    """Read an Optuna study database."""
    if not OPTUNA_AVAILABLE:
        return {"error": "Optuna not available"}
    
    if not db_path.exists():
        print(f"Study database not found: {db_path}")
        return {}
    
    storage = f"sqlite:///{db_path.resolve()}"
    
    # If study name not provided, try to find available studies
    if study_name is None:
        available_studies = list_optuna_studies(db_path)
        
        # Try common naming patterns based on hyperopt.py conventions
        # For joint: study_name = "cached_joint_{optimizer}", file = "cached_joint_{optimizer}.db"
        # For sequential: study_name = "cached_{molecule}_{optimizer}", file = "cached_{molecule}.db"
        filename_base = db_path.stem  # e.g., "cached_joint_pso" or "cached_BTA"
        possible_names = [
            filename_base,  # Exact match: "cached_joint_pso"
            filename_base.replace("cached_", ""),  # "joint_pso" or "BTA"
        ]
        
        # If it's a joint study, the study name matches the filename
        # If it's sequential, we need to guess the optimizer
        if "joint" in filename_base:
            # Joint study: filename is "cached_joint_pso", study name is "cached_joint_pso"
            possible_names.insert(0, filename_base)
        else:
            # Sequential: filename is "cached_BTA", study name might be "cached_BTA_pso", "cached_BTA_ga", etc.
            for opt in ["pso", "ga", "gwo", "pso-nm"]:
                possible_names.append(f"cached_{filename_base.replace('cached_', '')}_{opt}")
        
        # Try each possible name
        study = None
        for name in possible_names:
            try:
                study = optuna.load_study(study_name=name, storage=storage)
                study_name = name
                break
            except Exception:
                continue
        
        # If that didn't work, try available studies from the database
        if study is None and available_studies:
            study_name = available_studies[0]
            if len(available_studies) > 1:
                print(f"Multiple studies found: {available_studies}")
                print(f"Using first study: {study_name}")
                print(f"To use a different study, specify --study-name\n")
        elif study is None:
            error_msg = f"Could not find study in {db_path.name}."
            if available_studies:
                error_msg += f"\nAvailable studies: {available_studies}"
            error_msg += f"\nTried names: {possible_names}"
            error_msg += "\nTry specifying --study-name or use --list-studies-in-db to see available studies"
            return {"error": error_msg}
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        # Get all trials
        trials = []
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            
            # Add user attributes if present
            if trial.user_attrs:
                trial_data["user_attrs"] = trial.user_attrs
            
            trials.append(trial_data)
        
        # Get best trial
        best_trial = None
        if study.best_trial:
            best_trial = {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
            }
        
        return {
            "study_name": study_name,
            "n_trials": len(study.trials),
            "direction": study.direction.name,
            "best_trial": best_trial,
            "trials": trials
        }
    except Exception as e:
        # Try to list available studies to help the user
        available_studies = list_optuna_studies(db_path)
        error_msg = str(e)
        if available_studies:
            error_msg += f"\nAvailable studies in database: {available_studies}"
        else:
            # Try direct SQL query as fallback
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                error_msg += f"\nTables in database: {tables}"
            except Exception:
                pass
        return {"error": error_msg}


def print_cache_summary(data: Dict[str, Any], show_entries: bool = False):
    """Print summary of cache database."""
    print("=" * 70)
    print("HYPERPARAMETER CACHE DATABASE SUMMARY")
    print("=" * 70)
    print(f"Total entries: {data.get('total_entries', 0)}")
    print()
    
    stats = data.get("statistics", [])
    if stats:
        print("Statistics by molecule/optimizer:")
        print("-" * 70)
        print(f"{'Molecule':<20} {'Optimizer':<10} {'Count':<8} {'Avg Obj':<12} {'Min Obj':<12}")
        print("-" * 70)
        for stat in stats:
            print(f"{stat['molecule']:<20} {stat['optimizer']:<10} "
                  f"{stat['count']:<8} {stat['avg_objective']:<12.4f} {stat['min_objective']:<12.4f}")
        print()
    
    if show_entries:
        entries = data.get("entries", [])
        if entries:
            print(f"\nAll entries:")
            print("-" * 70)
            for entry in entries:
                print(f"Molecule: {entry['molecule']}, Optimizer: {entry['optimizer']}")
                print(f"  Objective: {entry['objective_value']:.4f}")
                print(f"  Time: {entry['evaluation_time']:.2f}s")
                print(f"  Params: {json.dumps(entry['hyperparameters'], indent=2)}")
                print()


def print_study_summary(data: Dict[str, Any], show_trials: bool = False):
    """Print summary of Optuna study database."""
    if "error" in data:
        print(f"Error reading study:")
        print(data['error'])
        print("\nTip: Try listing available studies with --list-studies or specify --study-name")
        return
    
    print("=" * 70)
    print("OPTUNA STUDY DATABASE SUMMARY")
    print("=" * 70)
    print(f"Study name: {data.get('study_name', 'N/A')}")
    print(f"Number of trials: {data.get('n_trials', 0)}")
    print(f"Direction: {data.get('direction', 'N/A')}")
    
    best = data.get("best_trial")
    if best:
        print(f"\nBest trial:")
        print(f"  Number: {best['number']}")
        print(f"  Value: {best['value']:.4f}")
        print(f"  Parameters:")
        for key, value in best['params'].items():
            print(f"    {key}: {value}")
    print()
    
    if show_trials:
        trials = data.get("trials", [])
        if trials:
            print(f"\nAll trials:")
            print("-" * 70)
            for trial in trials:
                print(f"Trial {trial['number']}: {trial['state']}, Value: {trial.get('value', 'N/A')}")
                print(f"  Params: {json.dumps(trial['params'], indent=2)}")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Read and analyze hyperparameter optimization databases"
    )
    parser.add_argument("--cache-db", type=Path,
                        help="Path to cache database (default: output/cache_hyperopt/hyperparameter_cache.db)")
    parser.add_argument("--study-db", type=Path,
                        help="Path to Optuna study database (e.g., output/studies/cached_BTA.db)")
    parser.add_argument("--study-name", type=str,
                        help="Study name (auto-detected from filename if not provided)")
    parser.add_argument("--show-entries", action="store_true",
                        help="Show detailed entries from cache database")
    parser.add_argument("--show-trials", action="store_true",
                        help="Show detailed trials from study database")
    parser.add_argument("--list-studies", action="store_true",
                        help="List all study databases in output/studies/ directory")
    parser.add_argument("--list-studies-in-db", type=Path,
                        help="List all study names within a specific Optuna database file")
    
    args = parser.parse_args()
    
    # List studies in a specific database if requested
    if args.list_studies_in_db:
        if not args.list_studies_in_db.exists():
            print(f"Database not found: {args.list_studies_in_db}")
            return
        
        studies = list_optuna_studies(args.list_studies_in_db)
        if studies:
            print(f"Studies in {args.list_studies_in_db}:")
            for study_name in studies:
                print(f"  {study_name}")
        else:
            print(f"No studies found in {args.list_studies_in_db}")
            # Try to show what's in the database
            try:
                conn = sqlite3.connect(args.list_studies_in_db)
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                print(f"Tables in database: {tables}")
            except Exception as e:
                print(f"Could not inspect database: {e}")
        return
    
    # List studies if requested
    if args.list_studies:
        studies_dir = Path("output/studies")
        if studies_dir.exists():
            print("Available study databases:")
            for db_file in sorted(studies_dir.glob("*.db")):
                print(f"  {db_file}")
                # Also show studies within each database
                studies = list_optuna_studies(db_file)
                if studies:
                    for study_name in studies:
                        print(f"    └─ {study_name}")
        else:
            print("Studies directory not found")
        return
    
    # Read cache database
    if args.cache_db or (not args.study_db and not args.list_studies):
        cache_path = args.cache_db or Path("output/cache_hyperopt/hyperparameter_cache.db")
        if cache_path.exists():
            # Auto-detect if it's actually a study database
            db_type = detect_db_type(cache_path)
            if db_type == "optuna":
                print(f"Note: {cache_path} appears to be an Optuna study database, not a cache database.")
                print(f"Reading as study database instead...\n")
                study_data = read_optuna_study(cache_path, args.study_name)
                print_study_summary(study_data, show_trials=args.show_entries)
            elif db_type == "cache":
                cache_data = read_cache_db(cache_path)
                if "error" in cache_data:
                    print(f"Error: {cache_data['error']}")
                else:
                    print_cache_summary(cache_data, show_entries=args.show_entries)
            else:
                print(f"Error: Could not determine database type for {cache_path}")
        else:
            print(f"Cache database not found: {cache_path}")
    
    # Read study database
    if args.study_db:
        # Auto-detect if it's actually a cache database
        db_type = detect_db_type(args.study_db)
        if db_type == "cache":
            print(f"Note: {args.study_db} appears to be a cache database, not an Optuna study database.")
            print(f"Reading as cache database instead...\n")
            cache_data = read_cache_db(args.study_db)
            if "error" in cache_data:
                print(f"Error: {cache_data['error']}")
            else:
                print_cache_summary(cache_data, show_entries=args.show_trials)
        elif db_type == "optuna":
            study_data = read_optuna_study(args.study_db, args.study_name)
            print_study_summary(study_data, show_trials=args.show_trials)
        else:
            print(f"Error: Could not determine database type for {args.study_db}")


if __name__ == "__main__":
    main()

