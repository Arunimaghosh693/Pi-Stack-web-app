import subprocess
import tempfile
from pathlib import Path 
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import shutil

from zipfile import ZipFile
from functools import lru_cache

import pandas as pd
from functools import lru_cache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def smiles_to_xyz(smiles: str, filename: str = None) -> Optional[str]:
    """Convert SMILES to XYZ content using build_extended_geometry or RDKit fallback"""
    from utils.smiles_to_xyz_linearize import build_extended_geometry

    try:
        # Use temporary file for multi-user safety
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        print(f"Converting SMILES to XYZ using advanced builder: {temp_filename}")
        output_path = Path(temp_filename).resolve()
            
        report = build_extended_geometry(
                smiles,
                seed=42,
                max_iters=500,  # Reduced for better multi-user performance
                use_mmff=True,
                anchor_dihedral_deg=90.0,
                anchor_tol=15.0,
                internal_dihedral_deg=180.0,
                internal_tol=10.0,
                core_pos_maxdispl=0.05,
                core_pos_k=2000.0,
                torsion_k=200.0,
                include_h=True,
                alternate_sides=True,
            )
            
        output_path.write_text(report.xyz, encoding='utf-8')
        print(f"  XYZ file created: {output_path}")
            
        # Clean up temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass  # File might already be deleted
            
        return report.xyz
            
    except (ValueError, RuntimeError, Exception) as e:
        print(f" SMILES to XYZ conversion: failed {e}")
        return None

def xyz_to_smiles_obabel(xyz_content: str) -> Optional[str]:
    """
    Convert XYZ content to SMILES using Open Babel.
    Falls back gracefully if Open Babel is not available.
    """
    try:
        
      
        if not shutil.which("obabel"):
            logger.warning("Open Babel (obabel) not found in system PATH")
            return None
            
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_path = f.name

        result = subprocess.run(
            ["obabel", temp_path, "-osmi"],
            capture_output=True,
            text=True,
            timeout=30  # Add timeout for safety
        )

        os.unlink(temp_path)

        if result.returncode != 0:
            logger.error(f"Open Babel conversion failed with code {result.returncode}")
            return None

        output = result.stdout.strip()
        if not output:
            return None

        # First token is SMILES
        smiles = output.split()[0]
        return smiles

    except subprocess.TimeoutExpired:
        logger.error("Open Babel conversion timed out")
        return None
    except Exception as e:
        logger.error(f"Open Babel conversion failed: {e}")
        return None






# ============================================================
# DATABASE LOADING & CACHING
# ============================================================

@lru_cache(maxsize=1)
def load_xyz_database():

    current_dir = Path(__file__).parent.parent
    pdi_path = current_dir / "data" / "processed" / "merged_pdi_smiles_hash_database.csv"
    ndi_path = current_dir / "data" / "processed" / "merged_ndi_smiles_hash_database.csv"

   

    df_pdi = pd.read_csv(pdi_path)
    df_ndi = pd.read_csv(ndi_path)

    # Merge into one dataframe
    df = pd.concat([df_pdi, df_ndi], ignore_index=True)

    from rdkit import Chem

    smiles_to_molid = {}


    for _, row in df.iterrows():

        smi = row["SMILES"]
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        canon = Chem.MolToSmiles(mol, canonical=True)

        smiles_to_molid[canon] = row["Molecule_ID"]
        #smiles_to_formula[canon] = row["Molecule_ID"]

    logger.info(f"Loaded {len(smiles_to_molid)} molecules")

    return smiles_to_molid

@lru_cache(maxsize=1)
def load_xyz_zip():
    base_dir = Path(__file__).parent.parent
    zip_path = base_dir / "data/processed/xyz_files.zip"
    return ZipFile(zip_path, "r")

@lru_cache(maxsize=5000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    

def find_xyz_file_path(smiles: str) -> Optional[str]:

    logger.info("===== DEBUG: find_xyz_file_path =====")
    logger.info(f"Input SMILES: {smiles}")

    smiles_to_molid = load_xyz_database()

    canonical_query = canonicalize_smiles(smiles)
    logger.info(f"Canonical SMILES: {canonical_query}")

    if canonical_query is None:
        logger.warning("Canonicalization failed")
        return None

    mol_id = smiles_to_molid.get(canonical_query)
    #mol_formula = smiles_to_formula.get(canonical_query)
    logger.info(f"Matched Mol_ID from CSV: {mol_id}")

    if not mol_id:
        logger.warning("No Mol_ID match found in database")
        return None

    base_dir = Path(__file__).parent.parent / "data/processed/xyz_files"

    folder = base_dir / mol_id
    logger.info(f"Checking folder: {folder}")

    if folder.exists():
        xyz_files = list(folder.glob("*.xyz"))
        logger.info(f"Found XYZ files: {[f.name for f in xyz_files]}")
        mol_id = smiles_to_molid.get(canonical_query)
        preferred = folder / f"{mol_id}.xyz"
        if preferred.exists():
          logger.info(f"Selected preferred XYZ file: {preferred}")
          logger.info("====================================")
          return str(preferred)
        
    zip_file = load_xyz_zip()
    logger.info("Checking ZIP archive")
    try:
      preferred_zip = f"{mol_id}/{mol_id}.xyz"
      logger.info(f"Looking for {preferred_zip} in zip")
      if preferred_zip in zip_file.namelist():
        logger.info(f"Found XYZ in ZIP: {preferred_zip}")
        return f"zip://{preferred_zip}"

    except Exception as e:
      logger.warning(f"ZIP lookup failed: {e}")



    
def load_xyz_from_database(smiles: str) -> Optional[str]:
    """
    Load precomputed XYZ content from database.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        XYZ file content as string, or None if not found
    """
    xyz_path = find_xyz_file_path(smiles)
    if xyz_path is None:
        return None
    
    try:
        with open(xyz_path, 'r') as f:
            content = f.read()
        logger.info(f"✅ Loaded precomputed XYZ ({len(content)} bytes)")
        return content
    except Exception as e:
        logger.error(f"Failed to read XYZ file: {e}")
        return None


# ============================================================
# SMILES TO XYZ CONVERSION
# ============================================================

def smiles_to_xyz(smiles: str, filename: str = None, use_precomputed: bool = True) -> Optional[str]:
    """
    Convert SMILES to XYZ content.
    
    Priority:
    1. Load from precomputed database (if use_precomputed=True)
    2. Generate new XYZ using build_extended_geometry (deterministic with seed=42)
    
    Args:
        smiles: SMILES string
        filename: Output filename (optional, for logging)
        use_precomputed: If True, check precomputed database first
        
    Returns:
        XYZ file content as string, or None if conversion failed
    """
    from utils.smiles_to_xyz_linearize import build_extended_geometry
    
    # Step 1: Try to load from precomputed database
    if use_precomputed:
        logger.info(f"🔍 Checking precomputed database for SMILES...")
        xyz_content = load_xyz_from_database(smiles)
        if xyz_content is not None:
            return xyz_content
    
    # Step 2: Generate new XYZ (deterministic with seed=42)
    logger.info(f"🔨 Generating new XYZ (seed=42)...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        output_path = Path(temp_filename).resolve()
            
        report = build_extended_geometry(
            smiles,
            seed=42,  # ✅ DETERMINISTIC SEED
            max_iters=2000,
            use_mmff=True,
            anchor_dihedral_deg=90.0,
            anchor_tol=15.0,
            internal_dihedral_deg=180.0,
            internal_tol=10.0,
            core_pos_maxdispl=0.05,
            core_pos_k=2000.0,
            torsion_k=200.0,
            include_h=True,
            alternate_sides=True,
        )
        
        xyz_content = report.xyz
        output_path.write_text(xyz_content, encoding='utf-8')
        
        logger.info(f"✅ Generated XYZ: {output_path}")
        
        # Clean up temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
            
        return xyz_content
            
    except (ValueError, RuntimeError, Exception) as e:
        logger.error(f"SMILES to XYZ conversion failed: {e}")
        return None


# ============================================================
# HELPER: Get molecule metadata from database
# ============================================================

def get_molecule_metadata(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a molecule from the database.
    """

    smiles_to_molid, smiles_to_formula = load_xyz_database()

    canonical_query = canonicalize_smiles(smiles)
    if canonical_query is None:
        return None

    mol_id = smiles_to_molid.get(canonical_query)
    mol_formula = smiles_to_formula.get(canonical_query)

    if mol_id is None:
        return None

    return {
        "Mol_ID": mol_id,
        "Molecule_ID": mol_id
    }
