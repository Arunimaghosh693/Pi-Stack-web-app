from typing import List, Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def canonicalize_smiles(smiles: str) -> str | None:
    """Convert a SMILES string to its canonical form.
    
    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Convert a SMILES string to a Morgan fingerprint bit vector.
    
    Note: Fingerprints are computed from the molecular graph, so they are
    automatically invariant to different SMILES representations of the same molecule.
    
    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two RDKit fingerprints."""
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def find_most_similar(
    query_smiles: str,
    db_molecules: List[Dict[str, Any]],
    smiles_key: str = "SMILES",
    radius: int = 2,
    n_bits: int = 2048,
    threshold: float = 0.0
) -> Tuple[Dict[str, Any] | None, float]:
    """Find the most similar molecule in `db_molecules` to `query_smiles`.

    Each item in `db_molecules` must contain a SMILES string under `smiles_key`.
    Returns a tuple of (best_match_record, similarity_score). If no valid
    fingerprints can be computed or no similarity meets the threshold, 
    returns (None, 0.0).
    
    Args:
        query_smiles: SMILES string to search for
        db_molecules: List of dictionaries containing molecule data
        smiles_key: Key in each dictionary containing the SMILES string
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint
        threshold: Minimum similarity threshold (default: 0.0)
    """
    query_fp = smiles_to_fp(query_smiles, radius=radius, n_bits=n_bits)
    if query_fp is None:
        return None, 0.0

    best_rec: Dict[str, Any] | None = None
    best_sim: float = 0.0

    for rec in db_molecules:
        s = rec.get(smiles_key)
        if not s:
            continue
        fp = smiles_to_fp(s, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        sim = tanimoto_similarity(query_fp, fp)
        if sim > best_sim and sim >= threshold:
            
            best_sim = sim
            best_rec = rec

    return best_rec, best_sim
