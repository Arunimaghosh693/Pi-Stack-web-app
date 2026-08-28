"""
stack_builder.py

Enhanced stack builder with comprehensive file bundling.
Requires pi-stack-optimizer to be present.
No fallback heuristic stacking.
"""
import sys
import os
import subprocess
import tempfile
import logging
import shutil
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/sandeepgroup/pi-stack-optimizer.git"


BASE_DIR = Path(__file__).resolve().parent.parent
logger.info(f"base dir is {BASE_DIR}")
REPO_DIR = BASE_DIR / "pi-stack-optimizer"

ACTIVATE_SCRIPT = REPO_DIR / "activate_pi_stack.sh"
OPTIMIZER_SCRIPT = REPO_DIR / "utils" / "build_stack.py"
CALC_DIHEDRALS_SCRIPT = REPO_DIR / "utils" / "calc_dihedrals.py"


def _optimizer_env() -> Dict[str, str]:
    """Return an environment where pi-stack-optimizer modules are importable."""
    env = os.environ.copy()
    repo_path = str(REPO_DIR)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        paths = existing_pythonpath.split(os.pathsep)
        if repo_path not in paths:
            env["PYTHONPATH"] = os.pathsep.join([repo_path, existing_pythonpath])
    else:
        env["PYTHONPATH"] = repo_path
    return env


def ensure_optimizer_repo():
    """Clone pi-stack-optimizer only once."""

    if REPO_DIR.exists():
        logger.info("pi-stack-optimizer already present.")
        return

    logger.info("Cloning pi-stack-optimizer repository...")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180
        )
    except subprocess.CalledProcessError:
        if REPO_DIR.exists():
            logger.info("Repository cloned by another process.")
        else:
            raise

    logger.info("pi-stack-optimizer ready.")


def build_stack(
    xyz_content: str,
    stacking_parameters: Dict[str, float],
    stack_size: int,
    strategy: str = "database"
) -> Dict[str, Any]:
    """
    Build supramolecular stack using pi-stack-optimizer with comprehensive file bundling.
    """

    try:

        logger.info(f"Received stacking_parameters keys: {stacking_parameters.keys()}")
        logger.info(f"Full stacking_parameters: {stacking_parameters}")
        logger.info(f"Strategy: {strategy}")
        ensure_optimizer_repo()

        required_params = {
            "Theta": None,
            "Tx": None,
            "Ty": None,
            "Tz": None,
            "Cx": None,
            "Cy": None,
            "interaction_energy": None,
            "Full_torsion_angles": []
        }

        # Check which parameters are missing
        missing_params = [k for k in required_params.keys() if k not in stacking_parameters]
        if missing_params:
            logger.warning(f"Missing parameters: {missing_params}")


        theta = stacking_parameters.get("Theta", None)
        tx = stacking_parameters.get("Tx", None)
        ty = stacking_parameters.get("Ty", None)
        tz = stacking_parameters.get("Tz", None)
        cx = stacking_parameters.get("Cx", None)
        cy = stacking_parameters.get("Cy", None)
        interaction_energy = stacking_parameters.get("interaction_energy", None)
        best_objective = stacking_parameters.get("best_objective", None)

        if interaction_energy is None:
            raise ValueError("interaction_energy is required in stacking_parameters")

        if any(param is None for param in [theta, tx, ty, tz, cx, cy]):
            raise ValueError(
                f"Missing required parameters. Got: "
                f"theta={theta}, tx={tx}, ty={ty}, tz={tz}, cx={cx}, cy={cy}"
            )

        # Convert to float and ensure precision
        theta = float(theta)
        tx = float(tx)
        ty = float(ty)
        tz = float(tz)
        cx = float(cx)
        cy = float(cy)
        if interaction_energy is not None:
            interaction_energy = float(interaction_energy)

        # ✅ ADD: Log after conversion
        if interaction_energy is not None:
            logger.info(f"After conversion - interaction_energy: {interaction_energy:.6f}")
        else:
            logger.info("After conversion - interaction_energy: N/A")
        logger.info(f"After conversion - theta: {theta:.3f}")

        raw_torsions = stacking_parameters.get("Full_torsion_angles", [])
        logger.info(f"the raw torsions are:{raw_torsions}")

        if isinstance(raw_torsions, str):
            import ast
            try:
                full_torsion_angles = [float(v) for v in ast.literal_eval(raw_torsions)]
            except Exception as e:
                logger.error(f"Failed to parse torsions: {e}")
                full_torsion_angles = []
        elif isinstance(raw_torsions, (list, tuple)):
            full_torsion_angles = [float(v) for v in raw_torsions]
        else:
            full_torsion_angles = []

        #logger.info(f"Parsed torsions: {full_torsion_angles}")

        #torsion_angle_str = ", ".join(f"{angle:.3f}" for angle in full_torsion_angles)
        if full_torsion_angles:
            torsion_angle_str = ", ".join(f"{angle:.3f}" for angle in full_torsion_angles)
        else:
            torsion_angle_str = ""


        with tempfile.TemporaryDirectory() as tmpdir:

            monomer_path = os.path.join(tmpdir, "monomer.xyz")
            results_path = os.path.join(tmpdir, "optimization_results.txt")
            torsions_path = os.path.join(tmpdir, "torsions.json")
            output_path = os.path.join(tmpdir, "stack.xyz")

            optimizer_script = OPTIMIZER_SCRIPT
            calc_dihedrals_script = CALC_DIHEDRALS_SCRIPT

            if not optimizer_script.exists():
                raise FileNotFoundError("build_stack.py not found in cloned repository.")

            if not calc_dihedrals_script.exists():
                raise FileNotFoundError("calc_dihedrals.py not found in cloned repository.")

            #logger.info(f"during stack building the xyz_content is {xyz_content}")


            # Write monomer
            with open(monomer_path, "w") as f:
                f.write(xyz_content)
            f.close()

            with open(monomer_path, "r") as f:
                monomer_check = f.read()
            f.close()


            # Step 1: Generate torsion file using calc_dihedrals.py
            if strategy:
                logger.info("Database-match mode detected → generating torsions.json")
                process_dihedrals = subprocess.run(
                    [
                        sys.executable,
                        str(calc_dihedrals_script),
                        "--input",
                        monomer_path,
                        "--draw",
                    ],
                    cwd=tmpdir,
                    env=_optimizer_env(),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                logger.info("calc_dihedrals.py executed successfully")
                # Verify torsions file
                if not os.path.exists(torsions_path):
                    alt_torsions_path = Path.cwd() / "torsions.json"
                    if alt_torsions_path.exists():
                        shutil.copy(alt_torsions_path, torsions_path)
                        #logger.info(f"Copied torsions.json → {torsions_path}")
                    else:
                        raise RuntimeError("Torsion file (torsions.json) was not generated by calc_dihedrals.py.")

            else:
                logger.info(f"Skipping torsion generation for strategy: {strategy}")

            with open(results_path, "w") as f:
                reduced_torsions_str = ""
                if full_torsion_angles:
                    reduced_torsions_str = f"  |  reduced torsions: [{', '.join(f'{angle:.3f}°' for angle in full_torsion_angles)}]"

                '''
                logger.info(f"WRITING TO results_path:")
                #logger.info(f"  interaction_energy={interaction_energy:.6f}")
                logger.info(f"  theta={theta:.3f}")
                logger.info(f"  tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
                logger.info(f"  cx={cx:.4f}, cy={cy:.4f}")
                logger.info(f" full torsions: {full_torsion_angles}")
                '''

                best_objective_str = (
                    f"{best_objective:.6f}" if best_objective is not None else "N/A"
                )
                f.write(
                    f"Best objective: {best_objective_str} kJ/mol\n"
                    f"Binding energy (per interface): {interaction_energy:.6f} kJ/mol\n"
                )

                f.write(
                    f"Best params: θ = {theta:.3f}°  "
                    f"Tx={tx:.4f} Å, Ty={ty:.4f} Å, Tz={tz:.4f} Å, "
                    f"Cx={cx:.4f} Å, Cy={cy:.4f} Å{reduced_torsions_str}\n"
                    f"Full torsion angles (degrees): {torsion_angle_str}\n"
                )


            with open(results_path, "r") as f:
                written_content = f.read()
                logger.info(f"Verification - Contents written to results_path:\n{written_content}")

            #logger.info("checking monomer path")
            #logger.info(monomer_path)
            with open(monomer_path, "r") as f:
                monomer_check = f.read()
            f.close()
            #logger.info(monomer_check)

            # Step 2: Build stack using build_stack.py with torsions file
            #logger.info(f"The path is {tmpdir}")

            cmd = [
                sys.executable,
                str(optimizer_script),
                monomer_path,
                "--results-file",
                results_path,
                "--n-molecules",
                str(stack_size),
                "--output",
                output_path,
            ]

            if full_torsion_angles:
                cmd.extend(["--torsions-file", torsions_path])

            logger.info("===== TEMP DIRECTORY CONTENTS =====")

            try:
                temp_files = os.listdir(tmpdir)
                logger.info(f"Files in {tmpdir}: {temp_files}")

                for f in temp_files:
                    fpath = os.path.join(tmpdir, f)
                    logger.info(f"{f} -> exists={os.path.exists(fpath)}, size={os.path.getsize(fpath)} bytes")

            except Exception as e:
                logger.error(f"Error checking temp directory: {e}")

            logger.info("===================================")

            logger.info("Running build_stack.py with command: %s", " ".join(cmd))
            process = subprocess.run(
                cmd,
                cwd=tmpdir,
                env=_optimizer_env(),
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if not os.path.exists(output_path):
                raise RuntimeError("Stack file was not generated by pi-stack-optimizer.")

            with open(output_path, "r") as f:
                stack_xyz = f.read()
            #logger.info("STACK coords")
            #logger.info(stack_xyz)

            optimizer_log = process.stdout

            with open(results_path, "r") as f:
                opt_result_txt = f.read()


            bundle_files = {}

            # Collect all files from the temp directory
            all_files = os.listdir(tmpdir)
            logger.info(f"Found {len(all_files)} files in temp directory: {all_files}")

            logger.info(f"ALL files are: {all_files}")

            for filename in all_files:
                file_path = os.path.join(tmpdir, filename)
                if os.path.isfile(file_path):
                    try:
                        if filename.endswith(".png"):
                            with open(file_path, "rb") as f:
                                content = f.read()
                                bundle_files[filename] = content
                        else:
                            with open(file_path, "r") as f:
                              content = f.read()

                            #logger.info(f"Content is: {content}")
                            if filename == os.path.basename(monomer_path):
                                bundle_files['monomer.xyz'] = content
                            elif filename == os.path.basename(output_path):
                                bundle_files['stack.xyz'] = content
                            elif filename == os.path.basename(results_path):
                                bundle_files['optimization_results.txt'] = content
                            elif filename == os.path.basename(torsions_path):
                            #logger.info(f"Content is: {content}")
                                bundle_files['torsions.json'] = content
                            else:
                                bundle_files[filename] = content

                    except Exception as e:
                        logger.warning(f"Could not read file {filename}: {e}")

            if 'monomer.xyz' not in bundle_files:
                bundle_files['monomer.xyz'] = xyz_content
            if 'stack.xyz' not in bundle_files:
                bundle_files['stack.xyz'] = stack_xyz
            if 'optimization_results.txt' not in bundle_files:
                bundle_files['optimization_results.txt'] = opt_result_txt
            if 'optimizer_log.txt' not in bundle_files:
                bundle_files['optimizer_log.txt'] = optimizer_log

            bundle_files['README.txt'] = _generate_readme(
                stack_size, strategy, interaction_energy, tz, theta, tx, ty, cx, cy,
                full_torsion_angles, optimizer_script.exists(), bundle_files
            )

        return {
            "success": True,
            "stack_xyz": stack_xyz,
            "bundle_files": bundle_files,
            "parameters_used": stacking_parameters,
            "strategy": strategy,
            "message": f"Stack generated successfully with {stack_size} molecules using {strategy} strategy"
        }

    except Exception as e:
        logger.error(f"Stack building failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def _generate_readme(
    stack_size: int,
    strategy: str,
    interaction_energy: float,
    tz: float,
    theta: float,
    tx: float,
    ty: float,
    cx: float,
    cy: float,
    full_torsion_angle:list[float],
    optimizer_available: bool,
    bundle_files: Dict[str, str]
) -> str:
    """Generate comprehensive README for the result bundle."""
    core_files = {
    'monomer.xyz',
    'stack.xyz',
    'optimization_results.txt',
    'torsions.json',
    'dihedrals_2d.png',
    'optimizer_log.txt',
    'README.txt'
}
    additional_files = [f for f in bundle_files.keys() if f not in core_files]
    interaction_energy_str = (
    f"{interaction_energy:.2f}"
    if interaction_energy is not None
    else "N/A")
    return f"""Molecular Stack Results
======================

This bundle contains the results of molecular stack generation:

Files:
- monomer.xyz: Original monomer structure
- stack.xyz: {stack_size}-molecule stack structure
- optimization_results.txt: Parameters used for stack generation
- torsions.json: Identified torsion definitions used for stacking
- dihedrals_2d.png: 2D depiction of detected dihedral angles
- optimizer_log.txt: Log from stack building process
- README.txt: This file

Parameters used:
- Stack size: {stack_size} molecules
- Binding energy: {interaction_energy_str} kJ/mol per interface
- Stacking distance (Tz): {tz:.2f} Å
- Rotation angle (theta): {theta:.1f}°
- Translation: Tx={tx:.2f}, Ty={ty:.2f} Å
- Center offset: Cx={cx:.2f}, Cy={cy:.2f} Å
- Full torsion angles (degrees): {", ".join(f"{a:.3f}" for a in full_torsion_angle) if full_torsion_angle else "N/A"}

Code Repository
---------------
The stack generation code used in this workflow is available at:

https://github.com/sandeepgroup/pi-stack-optimizer


Citation
--------
If you use this method or software in your research, we kindly request that you cite the following paper:

Ghosh, A., Barik, S., Singh, R. J., & Reddy, S. K. (2026).
π-Stack Optimizer: A high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems.


BibTeX
------
@article{{ghosh2026pistack,
  title   = {{π-Stack Optimizer: A high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems}},
  author  = {{Ghosh, Arunima and Barik, Susmita and Singh, Roshan J and Reddy, Sandeep K.}},
  year    = {{2026}}
}}
"""
