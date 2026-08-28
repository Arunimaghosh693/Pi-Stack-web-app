"""
optimizer_bundle.py

Creates a downloadable π-stack optimization bundle
for local execution (Streamlit-compatible version).
"""

import os
import tempfile
import zipfile
import shutil
from typing import Dict, Any


# Default parameter values for different optimization methods
DEFAULT_PARAMS = {
    "pso": {
        "swarm_size": 60,
        "inertia": 0.73,
        "cognitive": 1.50,
        "social": 1.50,
        "pso_tol": 0.01,
        "patience": 20,
    },
    "ga": {
        "ga_population": 80,
        "ga_mutation_rate": 0.10,
        "ga_mutation_sigma": 0.30,
        "ga_crossover_rate": 0.90,
        "ga_elite_fraction": 0.10,
        "ga_tournament_size": 3,
        "ga_tol": 0.01,
        "ga_patience": 20,
    },
    "gwo": {
        "gwo_pack_size": 50,
        "gwo_a_start": 2.0,
        "gwo_a_end": 0.0,
        "gwo_tol": 0.01,
        "gwo_patience": 20,
    },
    "pso-nm": {
        "swarm_size": 60,
        "inertia": 0.73,
        "cognitive": 1.50,
        "social": 1.50,
        "pso_tol": 0.01,
        "patience": 20,
        "nm_max_iters": 200,
        "nm_initial_step": 0.20,
        "nm_alpha": 1.0,
        "nm_gamma": 2.0,
        "nm_rho": 0.50,
        "nm_sigma": 0.50,
    }
}

# Default general parameters
DEFAULT_GENERAL_PARAMS = {
    "max_iters": 300,
    "seed": 42,
    "workers": 4,
    "threads": 1,
    "xtb_method": "gfn2",
    "charge": 0,
    "mult": 1,
    "penalty_weight": 2.0,
    "clash_cutoff": 1.6,
    "intramol_penalty_weight": 5.0,
    "intramol_cutoff": 1.2,
}


def get_default_params(opt_method: str) -> Dict[str, Any]:
    """Get default parameters for a specific optimization method."""
    params = DEFAULT_GENERAL_PARAMS.copy()
    if opt_method in DEFAULT_PARAMS:
        params.update(DEFAULT_PARAMS[opt_method])
    return params


def build_command_string(
    stack_size: int,
    opt_method: str,
    max_iters: int,
    seed: int,
    workers: int,
    threads: int,
    xtb_method: str,
    charge: int,
    mult: int,
    penalty_weight: float,
    clash_cutoff: float,
    intramol_penalty_weight: float,
    intramol_cutoff: float,
    **kwargs
) -> str:
    """
    Builds the command-line argument string for pi-stack-generator.py
    """

    base_args = [
        "../monomer.xyz",
        f"--n-layer {stack_size}",
        f"--optimizer {opt_method}",
        f"--max-iters {max_iters}",
        f"--seed {seed}",
        f"--workers {workers}",
        f"--threads {threads}",
        f"--method {xtb_method}",
        f"--charge {charge}",
        f"--mult {mult}",
        f"--penalty-weight {penalty_weight}",
        f"--clash-cutoff {clash_cutoff}",
        f"--intramol-penalty-weight {intramol_penalty_weight}",
        f"--intramol-cutoff {intramol_cutoff}",
    ]

    # Add method-specific parameters
    if opt_method == "pso":
        base_args.extend([
            f"--swarm-size {kwargs.get('swarm_size', 60)}",
            f"--inertia {kwargs.get('inertia', 0.73)}",
            f"--cognitive {kwargs.get('cognitive', 1.5)}",
            f"--social {kwargs.get('social', 1.5)}",
            f"--tol {kwargs.get('pso_tol', 0.01)}",
            f"--patience {kwargs.get('patience', 20)}",
        ])

    elif opt_method =="pso-nm":
         base_args.extend([
            f"--hybrid-nm-max-iters {kwargs.get('hybrid-nm-max-iters', 200)}",
            f"--hybrid-nm-initial-step {kwargs.get('hybrid-nm-initial-step', 0.20)}",
            f"--hybrid-nm-alpha {kwargs.get('hybrid-nm-alpha', 1.0)}",
            f"--hybrid-nm-gamma {kwargs.get('hybrid-nm-gamma', 2.0)}",
            f"--hybrid-nm-rho {kwargs.get('hybrid-nm-rho', 0.50)}",
            f"--hybrid-nm-sigma {kwargs.get('hybrid-nm-sigma', 0.50)}",
            f"--hybrid-nm-tol {kwargs.get('hybrid-nm-sigma', 0.01)}",
        ])

    elif opt_method == "ga":
        base_args.extend([
            f"--ga-population {kwargs.get('ga_population', 80)}",
            f"--ga-mutation-rate {kwargs.get('ga_mutation_rate', 0.1)}",
            f"--ga-mutation-sigma {kwargs.get('ga_mutation_sigma', 0.3)}",
            f"--ga-crossover-rate {kwargs.get('ga_crossover_rate', 0.9)}",
            f"--ga-elite-fraction {kwargs.get('ga_elite_fraction', 0.1)}",
            f"--ga-tournament-size {kwargs.get('ga_tournament_size', 3)}",
            f"--ga-tol {kwargs.get('ga_tol', 0.01)}",
            f"--ga-patience {kwargs.get('ga_patience', 20)}",
        ])

    elif opt_method == "gwo":
        base_args.extend([
            f"--gwo-pack-size {kwargs.get('gwo_pack_size', 50)}",
            f"--gwo-a-start {kwargs.get('gwo_a_start', 2.0)}",
            f"--gwo-a-end {kwargs.get('gwo_a_end', 0.0)}",
            f"--gwo-tol {kwargs.get('gwo_tol', 0.01)}",
            f"--gwo-patience {kwargs.get('gwo_patience', 20)}",
        ])

    return " ".join(base_args)


def build_multiline_command(
    stack_size: int,
    opt_method: str,
    max_iters: int,
    seed: int,
    workers: int,
    threads: int,
    xtb_method: str,
    charge: int,
    mult: int,
    penalty_weight: float,
    clash_cutoff: float,
    intramol_penalty_weight: float,
    intramol_cutoff: float,
    **kwargs
) -> str:
    """
    Builds a multiline command string for better readability in README
    """
    lines = [
        "python pi-stack-generator.py ../monomer.xyz \\",
        f"  --n-layer {stack_size} \\",
        f"  --optimizer {opt_method} \\",
        f"  --max-iters {max_iters} \\",
        f"  --seed {seed} \\",
        f"  --workers {workers} \\",
        f"  --threads {threads} \\",
        f"  --method {xtb_method} \\",
        f"  --charge {charge} \\",
        f"  --mult {mult} \\",
        f"  --penalty-weight {penalty_weight} \\",
        f"  --clash-cutoff {clash_cutoff} \\",
        f"  --intramol-penalty-weight {intramol_penalty_weight} \\",
        f"  --intramol-cutoff {intramol_cutoff}",
    ]
    
    # Add method-specific parameters
    if opt_method == "pso":
        lines[-1] += " \\"  # Add backslash to last general param
        lines.extend([
            f"  --swarm-size {kwargs.get('swarm_size', 60)} \\",
            f"  --inertia {kwargs.get('inertia', 0.73)} \\",
            f"  --cognitive {kwargs.get('cognitive', 1.5)} \\",
            f"  --social {kwargs.get('social', 1.5)} \\",
            f"  --tol {kwargs.get('pso_tol', 0.01)} \\",
            f"  --patience {kwargs.get('patience', 20)}",
        ])
    elif opt_method == "ga":
        lines[-1] += " \\"
        lines.extend([
            f"  --ga-population {kwargs.get('ga_population', 80)} \\",
            f"  --ga-mutation-rate {kwargs.get('ga_mutation_rate', 0.1)} \\",
            f"  --ga-mutation-sigma {kwargs.get('ga_mutation_sigma', 0.3)} \\",
            f"  --ga-crossover-rate {kwargs.get('ga_crossover_rate', 0.9)} \\",
            f"  --ga-elite-fraction {kwargs.get('ga_elite_fraction', 0.1)} \\",
            f"  --ga-tournament-size {kwargs.get('ga_tournament_size', 3)} \\",
            f"  --ga-tol {kwargs.get('ga_tol', 0.01)} \\",
            f"  --ga-patience {kwargs.get('ga_patience', 20)}",
        ])
    elif opt_method == "gwo":
        lines[-1] += " \\"
        lines.extend([
            f"  --gwo-pack-size {kwargs.get('gwo_pack_size', 50)} \\",
            f"  --gwo-a-start {kwargs.get('gwo_a_start', 2.0)} \\",
            f"  --gwo-a-end {kwargs.get('gwo_a_end', 0.0)} \\",
            f"  --gwo-tol {kwargs.get('gwo_tol', 0.01)} \\",
            f"  --gwo-patience {kwargs.get('gwo_patience', 20)}",
        ])
    
    return "\n".join(lines)


def create_optimizer_bundle(
    smiles: str,
    xyz_content: str,
    stack_size: int,
    opt_method: str = "pso",
    optimizer_params: Dict[str, Any] = None,
) -> bytes:
    """
    Creates a ZIP bundle containing:
    - monomer.xyz
    - full pi-stack-optimizer code
    - run_optimizer.sh
    - README.md

    Args:
        smiles: SMILES string of the molecule
        xyz_content: XYZ content of the molecule
        stack_size: Number of molecules in the stack
        opt_method: Optimization method ("pso", "ga", "gwo", "pso-nm")
        optimizer_params: Dictionary of optimizer parameters. If None, uses defaults.

    Returns:
        Raw ZIP file bytes (ready for st.download_button)
    """
    
    # Get default parameters and merge with provided ones
    params = get_default_params(opt_method)
    if optimizer_params:
        params.update(optimizer_params)
    
    # Extract the parameters that build_command_string expects as positional arguments
    # Remove them from params to avoid conflicts when passing **params
    cmd_params = params.copy()
    max_iters = cmd_params.pop("max_iters")
    seed = cmd_params.pop("seed") 
    workers = cmd_params.pop("workers")
    threads = cmd_params.pop("threads")
    xtb_method = cmd_params.pop("xtb_method")
    charge = cmd_params.pop("charge")
    mult = cmd_params.pop("mult")
    penalty_weight = cmd_params.pop("penalty_weight")
    clash_cutoff = cmd_params.pop("clash_cutoff")
    intramol_penalty_weight = cmd_params.pop("intramol_penalty_weight")
    intramol_cutoff = cmd_params.pop("intramol_cutoff")
    
    cmd_string = build_command_string(
        stack_size,
        opt_method,
        max_iters,
        seed,
        workers,
        threads,
        xtb_method,
        charge,
        mult,
        penalty_weight,
        clash_cutoff,
        intramol_penalty_weight,
        intramol_cutoff,
        **cmd_params  # Pass only the remaining method-specific parameters
    )
    
    # Build multiline command for README
    cmd_multiline = build_multiline_command(
        stack_size,
        opt_method,
        max_iters,
        seed,
        workers,
        threads,
        xtb_method,
        charge,
        mult,
        penalty_weight,
        clash_cutoff,
        intramol_penalty_weight,
        intramol_cutoff,
        **cmd_params
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        bundle_dir = os.path.join(tmpdir, "pi_stack_optimization_bundle")
        os.makedirs(bundle_dir, exist_ok=True)

        # Save monomer.xyz
        monomer_path = os.path.join(bundle_dir, "monomer.xyz")
        with open(monomer_path, "w", encoding="utf-8") as f:
            f.write(xyz_content)

        atom_count = len(
            [line for line in xyz_content.strip().split("\n")[2:] if line.strip()]
        )

        # Copy optimizer code
        

    

        # Estimate runtime based on atom count and parameters
        estimated_runtime_minutes = (atom_count / 10) * max_iters / 50
        
        # Determine molecule complexity
        if atom_count > 60:
            complexity = "High"
            complexity_warning = f"⚠️  **High complexity** ({atom_count} atoms) - Optimization will require significant computational resources"
        elif atom_count > 30:
            complexity = "Medium"
            complexity_warning = f"📊 **Medium complexity** ({atom_count} atoms) - Good candidate for optimization"
        else:
            complexity = "Low"
            complexity_warning = f"✓ **Low complexity** ({atom_count} atoms) - Fast optimization expected"

        # Main README.md
        readme_content = f"""# π-Stack Optimizer Input Bundle

## Molecule Information

**SMILES:** `{smiles}`  
**Total Atoms:** {atom_count}  
**Stack Size:** {stack_size} molecules  
**Optimization Method:** {opt_method.upper()}  
**Complexity:** {complexity}  

{complexity_warning}

---

## Files Included

- **monomer.xyz** - 3D coordinates of your molecule
- **run_optimizer.sh** - Executable script
- **README.md** - This instruction file

---

## Quick Start

### Step 1: Extract the Bundle
```bash
unzip pi_stack_optimization_bundle.zip
cd pi_stack_optimization_bundle
```

### Step 2: Make Script Executable
```bash
chmod +x run_optimizer.sh
```
*Note: This step is required only once. It grants execution permission to the script.*

### Step 3: Run Optimization
```bash
./run_optimizer.sh
```

**Estimated Runtime:** ~{estimated_runtime_minutes:.0f} minutes (depends on hardware)


**The script will automatically:**
1. Clone the π-stack optimizer repository
2. Detect flexible torsions in the molecule
3. Run the stacking optimization
4. Save all output files in the current directory

---

## Manual Execution (Alternative Method)

If the `run_optimizer.sh` script doesn't work or you prefer manual execution:

### Step 1: Clone the pi-stack-optimizer from https://github.com/sandeepgroup/pi-stack-optimizer.git and navigate to optimizer directory
```bash
cd pi-stack-optimizer
```

### Step 2: Activate environment 
```bash
source activate_pi_stack.sh
```

### Step 3: Detect flexible torsions (optional but recommended)
```bash
python utils/calc_dihedrals.py --input ../monomer.xyz
```

This generates `torsions.json` if flexible bonds are detected.

### Step 4: Run the optimization command

**Without torsions (rigid molecule):**

```bash
{cmd_multiline}
```

**With torsions (if torsions.json exists and contains torsions):**

```bash
{cmd_multiline} \\
  --torsions-file torsions.json
```

**Note:** When running manually, output files will be created in the pi-stack-optimizer directory. You can move them to the parent directory after completion:

```bash
mv molecular_stack_*.xyz optimization_*.* best_params.json molecule_after_torsion.xyz torsions.json *.log .. 2>/dev/null
cd ..
```

---

## Prerequisites

### 1. xTB Installation (Required)

The π-stack optimizer requires the **xTB quantum chemistry package**.

**Option A: Conda (Recommended)**
```bash
conda install -c conda-forge xtb
```

**Option B: Package Manager**
```bash
# Ubuntu/Debian
sudo apt install xtb


```

**Option C: Manual Installation**
- Download from [xTB GitHub repository](https://github.com/grimme-lab/xtb)
- Add to your PATH environment variable

**Verify Installation:**
```bash
which xtb
xtb --version
```

### 2. Python Dependencies

```bash
# Navigate to pi-stack-optimizer directory
cd pi-stack-optimizer

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## Expected Output Files

After successful completion, you will find the following files in the bundle root directory:

- **molecular_stack_{stack_size}molecules.xyz** - Final optimized stack structure
- **optimization_results.txt** - Detailed results and parameters
- **optimization_history.csv** - Iteration-by-iteration progress
- **best_params.json** - Best parameters found
- **torsions.json** - Detected flexible torsions (if any)
- **optimization.log** - Detailed execution log
- Various intermediate files

**Note:** When using `run_optimizer.sh`, all output files are automatically moved to the bundle root directory for easy access.

---

## Troubleshooting

### Common Issues

**"xtb: command not found"**
- Install xTB using one of the methods above
- Ensure xTB is in your PATH: `which xtb`
- Try running: `export PATH="/path/to/xtb/bin:$PATH"`

**"pi-stack-generator.py not found"**
- Ensure you're in the correct directory (pi-stack-optimizer)
- Check that the activate_pi_stack.sh script exists
- See the Manual Execution section above

**Permission denied**
- Make script executable: `chmod +x run_optimizer.sh`

**Memory or timeout errors**
- Your molecule is very complex ({atom_count} atoms)
- Consider reducing `--max-iters` parameter
- Run on a system with more RAM/CPU cores
- Use HPC resources for molecules > 100 atoms

**Python package errors**
- Install missing packages: `pip install numpy scipy`
- Check Python version: Python 3.7+ required

**Torsion detection issues**
- If `calc_dihedrals.py` fails, optimization will proceed with rigid molecule
- You can manually create torsions.json or skip torsion optimization

---


---

## Support & Documentation

- Check **pi-stack-optimizer/** directory for detailed documentation
- Ensure xTB is properly installed and accessible
- For very large molecules (>100 atoms), consider using HPC resources
- Review optimization_results.txt after completion for quality assessment

---


## Citation

If you use this method or software in your research, we kindly request that you cite the following paper:

Ghosh, A., Barik, S., Singh, R. J., & Reddy, S. K. (2026).  
π-Stack Optimizer: A high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems.

### BibTeX

```bibtex

@article{{ghosh2026pi,
  title= {{$\pi$-stack optimizer: framework for the design of one-dimensional supramolecular systems}},
  author={{Ghosh, Arunima and Susmita, Barik and Singh, Roshan J and Reddy, Sandeep K}},
  journal={{Journal of Molecular Modeling}},
  volume={{32}},
  number={{6}},
  pages={{188}},
  year={{2026}},
  publisher={{Springer}}
}}

**Generated by:** Supramolecular Stack Formation Web App  
**Timestamp:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
"""

        with open(os.path.join(bundle_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

        # Linux/macOS shell script
        sh_script = f"""#!/bin/bash

echo "Starting π-stack optimization..."
echo "Output files will be saved in the current directory"
echo ""

# Store current directory (bundle root)
BUNDLE_DIR=$(pwd)

# Clone pi-stack-optimizer if not present
if [ ! -d "pi-stack-optimizer" ]; then
    echo "Cloning pi-stack-optimizer repository..."
    git clone https://github.com/sandeepgroup/pi-stack-optimizer.git
else
    echo "pi-stack-optimizer already present."
fi
# Navigate to optimizer directory
cd pi-stack-optimizer

# Activate environment if available
if [ -f "activate_pi_stack.sh" ]; then
    source activate_pi_stack.sh
fi

# Step 1: Detect torsions/dihedrals
echo "Step 1: Analyzing molecule for flexible torsions..."
python utils/calc_dihedrals.py --input ../monomer.xyz

# Check if torsions.json was generated and contains torsions
TORSION_FLAG=""
if [ -f "torsions.json" ]; then
    # Check if torsions array has entries (more than just empty array)
    TORSION_COUNT=$(python -c "import json; data=json.load(open('torsions.json')); print(len(data.get('torsions', [])))" 2>/dev/null || echo "0")
    
    if [ "$TORSION_COUNT" -gt 0 ]; then
        echo "✓ Found $TORSION_COUNT flexible torsion(s) - will optimize dihedral angles"
        TORSION_FLAG="--torsions-file torsions.json"
    else
        echo "✓ No flexible torsions detected - using rigid molecule optimization"
    fi
else
    echo "✓ No torsions file generated - using rigid molecule optimization"
fi

# Step 2: Run optimizer with or without torsions
echo ""
echo "Step 2: Running π-stack optimization..."
python pi-stack-generator.py {cmd_string} $TORSION_FLAG

# Move output files to bundle root directory
echo ""
echo "Moving output files to bundle directory..."
mv molecular_stack_*.xyz "$BUNDLE_DIR/" 2>/dev/null
mv optimization_*.* "$BUNDLE_DIR/" 2>/dev/null
mv best_params.json "$BUNDLE_DIR/" 2>/dev/null
mv torsions.json "$BUNDLE_DIR/" 2>/dev/null
mv *.log "$BUNDLE_DIR/" 2>/dev/null

# Return to bundle directory
cd "$BUNDLE_DIR"

echo ""
echo "Optimization completed."
echo "Results saved in current directory:"
ls -lh molecular_stack_*.xyz optimization_*.* best_params.json 2>/dev/null || echo "Check for output files in the current directory"
"""

        sh_path = os.path.join(bundle_dir, "run_optimizer.sh")
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write(sh_script)

        os.chmod(sh_path, 0o755)

        # Create ZIP
        zip_path = os.path.join(tmpdir, "pi_stack_optimization_bundle.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(bundle_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmpdir)
                    zipf.write(file_path, arcname)

        with open(zip_path, "rb") as f:
            return f.read()