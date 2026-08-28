import subprocess
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import os
import shutil
import pandas as pd
from functools import lru_cache
from zipfile import ZipFile
from rdkit import Chem
from utils.smiles_to_xyz_linearize import build_extended_geometry

logger = logging.getLogger(__name__)


def smiles_to_xyz(smiles: str, filename: str = None, use_precomputed: bool = True) -> Optional[str]:
    """
    Convert SMILES to XYZ content.

    Priority:
    1. Load from precomputed database assets (if use_precomputed=True)
    2. Generate new XYZ when no database XYZ asset is available

    Args:
        smiles: SMILES string
        filename: Output filename (optional, for logging)
        use_precomputed: If True, check precomputed database first

    Returns:
        XYZ file content as string, or None if conversion failed
    """
    # Step 1: Try to load from precomputed database
    if use_precomputed:
        logger.info(f"🔍 Checking precomputed database for SMILES...")
        xyz_content = find_xyz_by_smiles(smiles)
        if xyz_content is not None:
            logger.info("Database match found; using precomputed XYZ")
            return xyz_content
        logger.info("No complete database XYZ match found; generating new XYZ")

    # Step 2: Generate new XYZ (deterministic with seed=42)
    logger.info(f"Generating new XYZ (seed=42)...")

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as temp_file:
            temp_filename = temp_file.name

        output_path = Path(temp_filename).resolve()
        report = build_extended_geometry(
            smiles,
            seed=42,
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


        try:
            os.unlink(temp_filename)
        except:
            pass

        return xyz_content

    except (ValueError, RuntimeError, Exception) as e:
        logger.error(f"SMILES to XYZ conversion failed: {e}")
        return None


@lru_cache(maxsize=1)
def load_hashed_database():
    """Load the PDI+NDI hashed CSV and build lookup dicts."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    csv_candidates = [
        #processed_dir / "aggregated_pdi_results_with_id.csv",
        processed_dir / "pdi_ndi_hash_database.csv",
        #processed_dir / "merged_pdi_ndi_with_ndi_hash_database.csv",
    ]
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        logger.warning(
            "Hashed database CSV not found. Checked: %s",
            ", ".join(str(path) for path in csv_candidates),
        )
        return {}, {}

    df = pd.read_csv(csv_path)
    required_columns = {"SMILES", "hash", "Molecule_ID"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.warning(f"Hashed database CSV missing columns: {sorted(missing_columns)}")
        return {}, {}



    smiles_to_hash = {}
    smiles_to_molid = {}

    for _, row in df.iterrows():
        smi = str(row["SMILES"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        smiles_to_hash[canon] = row["hash"]
        smiles_to_molid[canon] = row["Molecule_ID"]

    return smiles_to_hash, smiles_to_molid

@lru_cache(maxsize=5000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit."""
    try:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.warning(f"Canonicalization failed: {e}")
        return None

def xyz_to_smiles_obabel(xyz_content: str) -> Optional[str]:
    """Convert XYZ content to canonical SMILES using Open Babel."""
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
            timeout=30
        )
        os.unlink(temp_path)
        if result.returncode != 0:
            logger.error(f"Open Babel conversion failed with code {result.returncode}")
            return None
        output = result.stdout.strip()
        if not output:
            return None
        smiles = output.split()[0]
        return canonicalize_smiles(smiles)
    except subprocess.TimeoutExpired:
        logger.error("Open Babel conversion timed out")
        return None
    except Exception as e:
        logger.error(f"Open Babel conversion failed: {e}")
        return None

@lru_cache(maxsize=1)
def load_xyz_zip():
    """Load the zipped XYZ files, when the deployment provides a zip archive."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    zip_candidates = [
        processed_dir / "merged_PDI_NDI_Data_by_hash.zip",
    ]
    zip_path = next((path for path in zip_candidates if path.exists()), None)
    if zip_path is None:
        logger.warning(
            "XYZ zip file not found. Checked: %s",
            ", ".join(str(path) for path in zip_candidates),
        )
        return None
    return ZipFile(zip_path, "r")


def get_xyz_zip_path(hash_key: str, mol_id: str) -> str:
    """Build the expected XYZ member path inside the hashed ZIP archive."""
    return f"ndi_pdi_xyz_files_hashed/{hash_key}/{mol_id}.xyz"


def get_xyz_folder_path(hash_key: str, mol_id: str) -> Optional[Path]:
    """Return the first matching extracted XYZ path, if available."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    folder_candidates = [
         processed_dir / "merged_PDI_NDI_Data_by_hash.zip",
    ]
    for folder in folder_candidates:
        xyz_path = folder / str(hash_key) / f"{mol_id}.xyz"
        if xyz_path.exists():
            return xyz_path
    return None


def has_precomputed_xyz(smiles: str) -> bool:
    """Return True when the canonical SMILES exists in the hashed database CSV.

    Exact database matching is intentionally CSV-based. The corresponding XYZ is
    looked up separately from extracted folders or optional zip archives.
    """
    metadata = get_molecule_metadata(smiles)
    if metadata is None:
        logger.info("No database match found: SMILES not present in CSV")
        return False

    return True


def find_xyz_by_smiles(smiles: str) -> Optional[str]:
    """Given a SMILES, find and return its precomputed XYZ content if present."""
    metadata = get_molecule_metadata(smiles)
    if metadata is None:
        logger.info("No database match found: SMILES not present in CSV")
        return None

    folder_path = get_xyz_folder_path(metadata["hash"], metadata["Molecule_ID"])
    if folder_path is not None:
        logger.info(f"✅ XYZ FOUND in extracted folder: {folder_path}")
        return folder_path.read_text(encoding="utf-8")

    zip_file = load_xyz_zip()

    if zip_file is None:
        logger.info("Database metadata found, but no precomputed XYZ folder/zip was available")
        return None

    zip_path = get_xyz_zip_path(metadata["hash"], metadata["Molecule_ID"])
    try:
        zip_info = zip_file.getinfo(zip_path)
    except KeyError:
        logger.warning(
            f"Database entry found but XYZ missing from ZIP: {zip_path}"
        )
        return None

    with zip_file.open(zip_info) as f:
        logger.info(f"✅ XYZ FOUND in ZIP: {zip_path}")
        return f.read().decode("utf-8")

def find_xyz_by_xyz_content(xyz_content: str) -> Optional[str]:
    """Given XYZ content, convert to canonical SMILES and find the XYZ file as above."""
    smiles = xyz_to_smiles_obabel(xyz_content)
    if not smiles:
        logger.warning("Could not convert XYZ to SMILES")
        return None
    return find_xyz_by_smiles(smiles)

def get_molecule_metadata(smiles: str) -> Optional[Dict[str, Any]]:
    """Get hash and Molecule_ID for a SMILES."""
    smiles_to_hash, smiles_to_molid = load_hashed_database()
    canon = canonicalize_smiles(smiles)
    if canon is None:
        return None
    hash_key = smiles_to_hash.get(canon)
    mol_id = smiles_to_molid.get(canon)
    if not hash_key or not mol_id:
        return None
    return {"hash": hash_key, "Molecule_ID": mol_id}

# Example usage:
# xyz_content = find_xyz_by_smiles("O=C1C=CC(=O)N1C2=CC=CC=C2")
# xyz_content = find_xyz_by_xyz_content(open("somefile.xyz").read())
