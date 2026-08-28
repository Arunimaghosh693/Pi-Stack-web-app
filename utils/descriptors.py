from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# -------------------------------------------------------------------
# Dataclass (Only π–π relevant descriptors retained)
# -------------------------------------------------------------------
@dataclass
class MolecularDescriptors:
    pi_core_size: float
    steric_bulk_near_core: int
    rotatable_bonds: int
    molecular_weight: float
    heavy_atoms: int
    polar_surface_area: float
    num_rings: int
    num_aromatic_rings: int
    mol_mr: float
    molecular_formula: str
    h_bond_donors: int
    h_bond_acceptors: int
    aromatic_nitrogens: int
    aromatic_oxygens: int


# -------------------------------------------------------------------
# π-core size (number of aromatic atoms)
# -------------------------------------------------------------------

def estimate_pi_core_size(mol: Chem.Mol) -> float:
    aromatic_atoms = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
    return float(len(aromatic_atoms))


# π-CORE DETECTION
# ============================================================

def _imide_atom_set(mol) -> set:
    """Return a set of atom indices belonging to imide groups (N + carbonyl C/O)."""
    from rdkit import Chem
    
    imide_atoms: set = set()
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 7:  # N
            continue
        carbonyl_cs = []
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum() != 6:  # C
                continue
            # carbonyl C: has a double-bonded O neighbor
            is_carbonyl = False
            o_atoms = []
            for nb2 in nb.GetNeighbors():
                if nb2.GetAtomicNum() != 8:
                    continue
                b = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                if b is not None and b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    is_carbonyl = True
                    o_atoms.append(nb2.GetIdx())
            if is_carbonyl:
                carbonyl_cs.append((nb.GetIdx(), o_atoms))

        # imide N typically bound to two carbonyl carbons
        if len(carbonyl_cs) >= 2:
            imide_atoms.add(a.GetIdx())
            for c_idx, o_list in carbonyl_cs[:2]:
                imide_atoms.add(c_idx)
                for o_idx in o_list:
                    imide_atoms.add(o_idx)
    return imide_atoms


def _largest_fused_aromatic_component(mol) -> set:
    """Return atom indices for the largest fused aromatic ring component."""
    ringinfo = mol.GetRingInfo()
    atom_rings = list(ringinfo.AtomRings())
    if not atom_rings:
        return set()

    # Keep only fully aromatic rings
    aromatic_rings = []
    for r in atom_rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            aromatic_rings.append(set(r))
    if not aromatic_rings:
        return set()

    # Build ring adjacency graph: fused if share >=1 atom
    n = len(aromatic_rings)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if aromatic_rings[i].intersection(aromatic_rings[j]):
                adj[i].append(j)
                adj[j].append(i)

    # Find largest connected component by number of unique atoms
    seen = [False] * n
    best_atoms: set = set()
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp_atoms: set = set()
        while stack:
            u = stack.pop()
            comp_atoms |= aromatic_rings[u]
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp_atoms) > len(best_atoms):
            best_atoms = comp_atoms
    return best_atoms


def _identify_core_and_sidechains(smiles: str) -> Tuple[Optional[List[int]], Optional[List[int]], str]:
    """
    Identify π-core and sidechain atoms from SMILES.

    IMPORTANT:
    - Exact substructure matching is brittle for PDI/NDI derivatives because
      imide nitrogens are substituted and aromatic perception can differ.
    - This implementation is topology-based and robust across substitutions:
        1) core = largest fused aromatic component
        2) core += imide atoms (N + carbonyl C/O)

    Returns:
        (core_indices, sidechain_indices, structure_type)
        structure_type in {'PDI','NDI','AROMATIC','UNKNOWN'}
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None, None, 'UNKNOWN'

        all_atoms = set(range(mol.GetNumAtoms()))

        fused_arom = _largest_fused_aromatic_component(mol)
        imide_atoms = _imide_atom_set(mol)

        # Two related sets:
        # 1) "core_for_sidechains": what we remove to define sidechains
        #    (fused aromatic + imide atoms)
        # 2) "pi_core": what we use for π–π overlap (use fused aromatic only)
        core_for_sidechains = set(fused_arom) | set(imide_atoms)
        if not core_for_sidechains:
            core_for_sidechains = set(a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic())

        pi_core = set(fused_arom) if fused_arom else set(
            a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()
        )
        if not pi_core:
            # last resort: use the sidechain core definition
            pi_core = set(core_for_sidechains)

        # Core type heuristic (NDI vs PDI): based on aromatic fused core size
        n_arom = sum(1 for i in fused_arom if mol.GetAtomWithIdx(i).GetIsAromatic())
        n_imide_n = sum(1 for i in imide_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)

        struct_type = 'AROMATIC'
        if n_imide_n >= 2:
            if n_arom >= 16:
                struct_type = 'PDI'
            elif n_arom >= 8:
                struct_type = 'NDI'

        core_idx = sorted(pi_core)
        sidechain_idx = sorted(list(all_atoms - core_for_sidechains))

        logger.info(f"π-core type: {struct_type} (aromatic_core_atoms={n_arom}, imide_N={n_imide_n})")
        logger.debug(f"Core atoms ({struct_type}): {core_idx}")
        logger.debug(f"Sidechain atoms: {sidechain_idx}")
        return core_idx, sidechain_idx, struct_type
    except Exception as e:
        logger.error(f"Core/sidechain identification failed: {e}", exc_info=True)
        return None, None, 'UNKNOWN'


def _get_pi_indices_from_smiles(smiles: str) -> Optional[List[int]]:
    """
    Use SMILES → RDKit with PDI/NDI pattern matching.
    Falls back to general aromatic detection if patterns don't match.
    """
    try:
        from rdkit import Chem
        
        # Use the new substructure matching function
        core_idx, sidechain_idx, struct_type = _identify_core_and_sidechains(smiles)
        
        if core_idx and len(core_idx) >= 3:
            logger.info(f"✅ π-core detection successful ({struct_type}): {len(core_idx)} atoms")
            logger.info(f"   Core indices: {core_idx}")
            logger.info(f"   Sidechain indices: {sidechain_idx}")
            return core_idx
        
        logger.warning(f"π-core detection failed for SMILES: {smiles}")
        return core_idx
        
    except Exception as e:
        logger.error(f"SMILES-based π-core detection failed: {e}", exc_info=True)
        return None
    
# -------------------------------------------------------------------
# Steric bulk near pi-core
# -------------------------------------------------------------------

def calculate_steric_bulk_near_core(
    mol: Chem.Mol,
    pi_indices: Optional[List[int]] = None,
    max_bonds: int = 2
) -> int:
    """
    Count non-core heavy atoms within <= max_bonds topological distance
    from the detected pi-core. This is a simple proxy for steric crowding
    near the aromatic surface.
    """
    try:
        if mol is None or mol.GetNumAtoms() == 0:
            return 0

        if not pi_indices:
            pi_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]

        if not pi_indices:
            return 0

        pi_set = set(pi_indices)
        count = 0

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()

            if idx in pi_set:
                continue
            if atom.GetAtomicNum() == 1:
                continue

            min_dist = None
            for core_idx in pi_indices:
                path = Chem.GetShortestPath(mol, idx, core_idx)
                if path:
                    bond_dist = len(path) - 1
                    if min_dist is None or bond_dist < min_dist:
                        min_dist = bond_dist

            if min_dist is not None and min_dist <= max_bonds:
                count += 1

        return int(count)

    except Exception as e:
        logger.warning(f"Steric bulk near core calculation failed: {e}")
        return 0


# -------------------------------------------------------------------
# Electronic Characteristics
# -------------------------------------------------------------------

def count_aromatic_heteroatoms(mol: Chem.Mol) -> tuple[int, int]:
    """Counts the number of aromatic nitrogen and oxygen atoms."""
    aromatic_nitrogens = 0
    aromatic_oxygens = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            if atom.GetSymbol() == 'N':
                aromatic_nitrogens += 1
            elif atom.GetSymbol() == 'O':
                aromatic_oxygens += 1
    return aromatic_nitrogens, aromatic_oxygens


# -------------------------------------------------------------------
# Functional Groups
# -------------------------------------------------------------------

def get_functional_groups(mol: Chem.Mol) -> list[str]:
    """Identifies common functional groups in a molecule."""
    groups = []
    # Use RDKit's Lipinski module for common functional group patterns
    if Lipinski.NumHDonors(mol) > 0:
        groups.append("H-Bond Donor")
    if Lipinski.NumHAcceptors(mol) > 0:
        groups.append("H-Bond Acceptor")
    if Descriptors.NumAmideBonds(mol) > 0:
        groups.append("Amide")
    if Descriptors.NumSpiroAtoms(mol) > 0:
        groups.append("Spiro Center")
    # Add more specific checks if needed
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]")):
        groups.append("Carboxylic Acid")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][CX3](=O)[#6]")):
        groups.append("Ketone")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][OX2][#6]")):
        groups.append("Ether")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")):
        groups.append("Primary Amine")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[OH]c1ccccc1")):
        groups.append("Phenol")

    # Remove duplicates and return
    return sorted(list(set(groups)))


# -------------------------------------------------------------------
# Main descriptor calculation
# -------------------------------------------------------------------

def compute_descriptors(mol_input: str, fmt: str) -> MolecularDescriptors:
    """
    Compute π–π stacking–relevant descriptors from SMILES or XYZ input.
    Cloud-safe version.
    """
    mol = None
    if fmt == "smi":
        mol = Chem.MolFromSmiles(mol_input)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate 3D conformer from SMILES.")
    elif fmt == "xyz":
        try:
            mol = Chem.MolFromXYZBlock(mol_input)
            if mol is None:
                raise ValueError("Failed to create molecule from XYZ block.")
            # For XYZ, the structure is already 3D, but we need to perceive bonds
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(f"Error processing XYZ data: {e}")
    else:
        raise ValueError(f"Unsupported format: {fmt}. Only 'smi' and 'xyz' are supported.")

    if mol is None:
        raise ValueError("Failed to create a valid molecule from the input.")

    # ------------------------------
    # π–π relevant descriptors
    # ------------------------------

    if fmt == "smi":
        pi_indices = _get_pi_indices_from_smiles(mol_input)
    else:
        pi_indices = None

    if not pi_indices:
        pi_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]

    pi_core_size = float(len(pi_indices))
    steric_bulk_near_core = calculate_steric_bulk_near_core(
        mol,
        pi_indices=pi_indices,
        max_bonds=2
    )
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    molecular_weight = Descriptors.MolWt(mol)
    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    polar_surface_area = Descriptors.TPSA(mol)
    num_rings = Descriptors.RingCount(mol)
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    mol_mr = Descriptors.MolMR(mol)
    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
    
    # Electronic and functional group properties
    h_bond_donors = Lipinski.NumHDonors(mol)
    h_bond_acceptors = Lipinski.NumHAcceptors(mol)
    aromatic_nitrogens, aromatic_oxygens = count_aromatic_heteroatoms(mol)

    return MolecularDescriptors(
        pi_core_size=pi_core_size,
        steric_bulk_near_core=steric_bulk_near_core,
        rotatable_bonds=rotatable_bonds,
        molecular_weight=molecular_weight,
        heavy_atoms=heavy_atoms,
        polar_surface_area=polar_surface_area,
        num_rings=num_rings,
        num_aromatic_rings=num_aromatic_rings,
        mol_mr=mol_mr,
        molecular_formula=molecular_formula,
        h_bond_donors=h_bond_donors,
        h_bond_acceptors=h_bond_acceptors,
        aromatic_nitrogens=aromatic_nitrogens,
        aromatic_oxygens=aromatic_oxygens,
    )

def calculate_conjugation_properties(mol) -> Dict[str, Any]:
    """
    Calculate conjugation and aromaticity properties relevant to π-π stacking.
    These are lightweight calculations using RDKit descriptors.
    """
    if not RDKIT_AVAILABLE:
        return {}
    
    try:
        properties = {}
        
        # Basic aromaticity metrics
        properties['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        properties['aromatic_atoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        properties['total_rings'] = Descriptors.RingCount(mol)
        
        # Conjugation extent
        properties['aromatic_carbons'] = sum(1 for atom in mol.GetAtoms() 
                                           if atom.GetIsAromatic() and atom.GetSymbol() == 'C')
        properties['heteroaromatics'] = sum(1 for atom in mol.GetAtoms() 
                                          if atom.GetIsAromatic() and atom.GetSymbol() != 'C')
        
        # Molecular size and complexity
        properties['heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
        properties['molecular_weight'] = Descriptors.MolWt(mol)
        
        # Flexibility indicators (affects stacking geometry)
        properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        properties['flexibility_ratio'] = properties['rotatable_bonds'] / max(properties['heavy_atoms'], 1)
        
        # Electronic properties
        properties['total_degree'] = sum(atom.GetTotalDegree() for atom in mol.GetAtoms())
        properties['unsaturation_count'] = sum(1 for bond in mol.GetBonds() 
                                             if bond.GetBondType() != Chem.rdchem.BondType.SINGLE)
        
        return properties
        
    except Exception as e:
        logger.warning(f"Conjugation properties calculation failed: {e}")
        return {}


def analyze_molecular_composition(mol) -> Dict[str, Any]:
    """
    Analyze basic molecular composition - simple counts only.
    """
    if not RDKIT_AVAILABLE:
        return {}
    
    try:
        result = {}
        
        atom_types = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_types[symbol] = atom_types.get(symbol, 0) + 1
        
        result['atom_types'] = dict(atom_types)
        result['atom_type_count'] = len(atom_types)
        
        # Basic ring information
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        
        if ring_sizes:
            result['ring_count'] = len(ring_sizes)
            result['largest_ring'] = max(ring_sizes)
            result['has_six_rings'] = 6 in ring_sizes
        else:
            result['ring_count'] = 0
            result['largest_ring'] = 0
            result['has_six_rings'] = False
        
        # Molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        result['molecular_formula'] = formula
        
        return result
        
    except Exception as e:
        logger.warning(f"Molecular composition analysis failed: {e}")
        return {}

def calculate_simple_electronic_properties(mol) -> Dict[str, Any]:
    """
    Calculate simple electronic properties without complex classification.
    """
    if not RDKIT_AVAILABLE:
        return {}
    
    try:
        result = {}
        
        # Simple atom counts
        aromatic_nitrogen = 0
        aromatic_oxygen = 0
        aromatic_carbons = 0
        
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                symbol = atom.GetSymbol()
                if symbol == 'N':
                    aromatic_nitrogen += 1
                elif symbol == 'O':
                    aromatic_oxygen += 1
                elif symbol == 'C':
                    aromatic_carbons += 1
        
        result['aromatic_nitrogen'] = aromatic_nitrogen
        result['aromatic_oxygen'] = aromatic_oxygen
        result['aromatic_carbons'] = aromatic_carbons
        
        # Simple ratios
        total_aromatic = aromatic_nitrogen + aromatic_oxygen + aromatic_carbons
        if total_aromatic > 0:
            result['heteroatom_fraction'] = (aromatic_nitrogen + aromatic_oxygen) / total_aromatic
        else:
            result['heteroatom_fraction'] = 0.0
        
        return result
        
    except Exception as e:
        logger.warning(f"Simple electronic properties calculation failed: {e}")
        return {}


# -------------------------------------------------------------------
# XYZ to Molecular Formula Conversion
# -------------------------------------------------------------------

def xyz_to_molformula(xyz_content: str) -> Optional[str]:
    """
    Generate molecular formula from XYZ content using Hill system ordering.
    
    Hill system:
    - If carbon present → C, H, then others alphabetically
    - If no carbon → all elements alphabetically
    
    Args:
        xyz_content: Full XYZ file content as string
        
    Returns:
        Molecular formula string (e.g., C24H10N2O4)
    """
    try:
        lines = xyz_content.strip().split("\n")
        
        if len(lines) < 3:
            return None
        
        # Skip first two lines (atom count + comment)
        atom_lines = lines[2:]
        
        elements = []
        for line in atom_lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                elements.append(parts[0])
        
        counts = Counter(elements)
        
        formula = ""
        
        # Hill system
        if "C" in counts:
            c_count = counts.pop("C")
            formula += f"C{c_count}" if c_count > 1 else "C"
            
            if "H" in counts:
                h_count = counts.pop("H")
                formula += f"H{h_count}" if h_count > 1 else "H"
        
        # Remaining elements alphabetically
        for element in sorted(counts.keys()):
            count = counts[element]
            formula += f"{element}{count}" if count > 1 else element
        
        return formula
    
    except Exception as e:
        logger.error(f"Failed to generate molecular formula: {e}")
        return None
