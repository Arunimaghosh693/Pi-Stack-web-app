# -------------------------------
# Database Functions
# -------------------------------
import streamlit as st
import pandas as pd
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from utils.validation import check_monomer_geometry_sanity, ValidationResult, read_xyz_coords
from utils.similarity import find_most_similar

logger = logging.getLogger(__name__)


try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False




@st.cache_data
def load_molecular_database() -> pd.DataFrame:
    try:
        current_dir = Path(__file__).parent.parent
        csv_path = current_dir / "data" / "processed" / "merged_PDI_NDI_data_with_hash.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if RDKIT_AVAILABLE and 'SMILES' in df.columns:
                try:
                    from rdkit import Chem as _Chem
                    df['canonical_SMILES'] = df['SMILES'].apply(
                        lambda s: _Chem.MolToSmiles(_Chem.MolFromSmiles(str(s)), canonical=True)
                        if pd.notna(s) and _Chem.MolFromSmiles(str(s)) is not None
                        else str(s)
                    )
                except Exception:
                    df['canonical_SMILES'] = df['SMILES']
            else:
                df['canonical_SMILES'] = df.get('SMILES', pd.Series(dtype=str))
            return df
        else:
            st.warning("⚠️ Aggregated database not found, check the path")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        return pd.DataFrame()


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit"""
    if not RDKIT_AVAILABLE:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def find_exact_match(smiles: str, database: pd.DataFrame) -> Optional[Dict]:
    """Find exact SMILES match in database using pre-canonicalized column"""
    canonical_query = canonicalize_smiles(smiles)
    if canonical_query is None:
        return None

    match_col = 'canonical_SMILES' if 'canonical_SMILES' in database.columns else 'SMILES'
    matches = database[database[match_col] == canonical_query]
    if len(matches) > 0:
        return matches.iloc[0].to_dict()
    return None


def check_database_match_with_smile(
    smiles: str,
    xyz_content: str,
    similarity_threshold: float = 0.9,
    bonded_cutoff: float = 0.8,
    angle_cutoff: float = 1.1,
    nonbonded_cutoff: float = 1.3,
    smiles_source: str = "user_provided",
) -> Dict[str, Any]:
    """
    Three-step database matching process:
    1. Exact SMILES match
    2. Similarity match (≥user-defined threshold) 
    3. No match - prepare for optimizer

    Args:
        smiles: SMILES string of the molecule
        xyz_content: XYZ coordinate content
        similarity_threshold: Minimum Tanimoto similarity for a similarity match
        bonded_cutoff: Minimum allowed bonded distance (Å) for geometry validation
        angle_cutoff: Minimum allowed angle-related distance (Å) for geometry validation
        nonbonded_cutoff: Minimum allowed non-bonded distance (Å) for geometry validation
    """
    database = load_molecular_database()
    
    exact_match = find_exact_match(smiles, database)
    if exact_match:
        display_keys = [
            "SMILES", "MolFormula", "interaction_energy",
            "Tx", "Ty", "Tz", "Cx", "Cy","Theta","Full_torsion_angles"
        ]
        
        stacking_parameters = {k: exact_match.get(k, 0) for k in display_keys if k in exact_match}
        stacking_parameters["best_objective"] = exact_match.get("best_objective")
        return {
            "strategy": "exact_match",
            "stacking_parameters": stacking_parameters,
            "validation_status": "skipped",
            "validation_message": "Validation skipped - exact match found in database",
            "xyz_content": xyz_content
        }
    
    # Step 2: Validate the input geometry before attempting a similarity search.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write(xyz_content)
            temp_path = f.name

        atom_types, coords = read_xyz_coords(temp_path)
        validation_result = check_monomer_geometry_sanity(
                coords=coords,
                atom_types=atom_types,
                bonded_cutoff=bonded_cutoff,
                angle_cutoff=angle_cutoff,
                nonbonded_cutoff=nonbonded_cutoff
        )

        if not validation_result.is_valid:
            return {
                    "error": "Input geometry is invalid",
                    "validation_status": "failed",
                    "validation_message": validation_result.message,
                    "validation_issues": validation_result.issues
            }

    except Exception as e:
        return {
            "error": f"Failed to validate input geometry: {str(e)}",
            "validation_status": "error"
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    # Step 3: Similarity search
    # Find the single closest molecule in the database (no threshold filter), so
    # that even when nothing clears the user's threshold we can still report the
    # true closest similarity to the user.
    similar_mol, similarity = find_most_similar(smiles, database.to_dict('records'))

    if similar_mol and similarity >= similarity_threshold:
        display_keys = [
            "SMILES", "MolFormula", "interaction_energy",
            "Tx", "Ty", "Tz", "Cx", "Cy","Theta","Full_torsion_angles"
        ]
        stacking_parameters = {k: similar_mol.get(k, 0) for k in display_keys if k in similar_mol}
        stacking_parameters["best_objective"] = similar_mol.get("best_objective")
        stacking_parameters["similarity_score"] = similarity
        return {
            "strategy": "similarity_match",
            "stacking_parameters": stacking_parameters,
            "validation_status": "passed",
            "validation_message": "Input geometry validated successfully",
            "xyz_content": xyz_content,
            "similarity_score": similarity
        }

    # Step 4: No molecule found within the similarity threshold.
    # The user can either lower the threshold and search again, or download a
    # custom π-stack optimizer package to optimize the geometry from scratch.
    best_similarity = float(similarity) if similar_mol else 0.0
    return {
        "strategy": "pi-stack-optimizer",
        "stacking_parameters": {
            "message": "No molecule within the current similarity threshold was found in the database.",
            "best_similarity": best_similarity,
            "threshold": float(similarity_threshold),
        },
        "validation_status": "passed",
        "validation_message": "Input geometry validated successfully - ready for π-stack optimization",
        "xyz_content": xyz_content,
        "best_similarity": best_similarity,
        "threshold": float(similarity_threshold),
    }


