from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRG
from rdkit.Chem import Descriptors
from rdkit import RDLogger
"""
validation.py
Core molecular validation logic for supramolecular stack web application.
Handles basic chemistry checks, symmetry analysis, and input parsing.
"""

class ValidationStatus(Enum):
    VALID = "valid"
    INVALID_BONDING = "invalid_bonding"
    BROKEN_FRAGMENTS = "broken_fragments"
    ATOMIC_OVERLAPS = "atomic_overlaps"
    SYMMETRY_ERROR = "symmetry_error"
    UNKNOWN_ERROR = "unknown_error"
    STACK_OVERLAPS = "stack_overlaps"  
    MORE_THAN_ONE_CORE = "multiple_core"
    NOT_SYMMETRIC="not_symmetric"

@dataclass
class ValidationResult:
    status: ValidationStatus
    is_valid: bool
    message: str
    issues: List[str]
    symmetry_info: Dict[str, Any] = None

@dataclass
class StackValidationResult:
    """Result of stack geometry validation"""
    is_valid: bool
    message: str
    issues: List[str]
    clash_penalty: float = 0.0
    min_interlayer_distance: float = float('inf')
    layer_pair_clashes: Dict[Tuple[int, int], int] = None

    def __post_init__(self):
        if self.layer_pair_clashes is None:
            self.layer_pair_clashes = {}

class StackGeometryValidator:
    """
    Stack geometry validator using the same logic as pi-stack optimizer.
    
    Validates inter-layer distances and clash penalties without xTB dependency.
    Uses the same cutoffs and penalty calculations as the pi-stack optimizer.
    """
    
    def __init__(self, 
                 clash_cutoff: float = 1.6,
                 penalty_weight: float = 2.0,
                 severe_overlap_cutoff: float = 1.5,
                 max_close_contacts: int = 5,
                 close_contact_cutoff: float = 2.0):
        """
        Initialize stack validator with pi-stack optimizer parameters.
        
        Args:
            clash_cutoff: Inter-molecular clash detection cutoff (default: 1.6 Å)
            penalty_weight: Weight for clash penalties (default: 2.0)
            severe_overlap_cutoff: Cutoff for severe overlaps (default: 1.5 Å) 
            max_close_contacts: Max allowed close contacts between layers (default: 5)
            close_contact_cutoff: Cutoff for counting close contacts (default: 2.0 Å)
        """
        self.clash_cutoff = clash_cutoff
        self.penalty_weight = penalty_weight
        self.severe_overlap_cutoff = severe_overlap_cutoff
        self.max_close_contacts = max_close_contacts
        self.close_contact_cutoff = close_contact_cutoff
    
    def clash_penalty(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        Calculate clash penalty between two coordinate sets (exact pi-stack logic).
        
        Args:
            coords1: Coordinates of first layer/molecule (N1, 3)
            coords2: Coordinates of second layer/molecule (N2, 3)
            
        Returns:
            Clash penalty value (number of clashing atom pairs)
        """
        # Calculate all pairwise distances between the two coordinate sets
        diff = coords1[:, None, :] - coords2[None, :, :]  # (N1, N2, 3)
        distances = np.linalg.norm(diff, axis=-1)  # (N1, N2)
        
        # Count clashes (distances below cutoff)
        clashes = np.sum(distances < self.clash_cutoff)
        return float(clashes)
    
    def check_severe_overlaps(self, layer_coords: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Check for severe overlaps that would be problematic (pi-stack logic).
        
        Args:
            layer_coords: List of coordinate arrays, one per layer
            
        Returns:
            (has_severe_overlaps, error_message)
        """
        for i in range(len(layer_coords)):
            for j in range(i + 1, len(layer_coords)):
                diff = layer_coords[i][:, None, :] - layer_coords[j][None, :, :]
                distances = np.linalg.norm(diff, axis=-1)
                
                min_dist = np.min(distances)
                n_close = np.sum(distances < self.close_contact_cutoff)
                
                # Check for catastrophic overlaps (same as pi-stack)
                if min_dist < self.severe_overlap_cutoff:
                    return True, f"Severe overlap between layers {i+1} and {j+1}: minimum distance = {min_dist:.3f} Å (< {self.severe_overlap_cutoff} Å)"
                
                if n_close > self.max_close_contacts:
                    return True, f"Too many close contacts between layers {i+1} and {j+1}: {n_close} atom pairs closer than {self.close_contact_cutoff} Å (max allowed: {self.max_close_contacts})"
        
        return False, ""
    
    def validate_stack_geometry(self, 
                              stack_coords: np.ndarray, 
                              atom_types: List[str], 
                              n_molecules: int) -> StackValidationResult:
        """
        Validate complete stack geometry using pi-stack optimizer logic.
        
        Args:
            stack_coords: All atomic coordinates in the stack (N_total, 3)
            atom_types: Atomic symbols for one molecule 
            n_molecules: Number of molecules in the stack
            
        Returns:
            StackValidationResult with detailed validation information
        """
        atoms_per_molecule = len(atom_types)
        total_atoms = atoms_per_molecule * n_molecules
        
        if len(stack_coords) != total_atoms:
            return StackValidationResult(
                is_valid=False,
                message=f"Coordinate mismatch: expected {total_atoms} atoms, got {len(stack_coords)}",
                issues=[f"Stack has {len(stack_coords)} atoms but should have {total_atoms} ({n_molecules} × {atoms_per_molecule})"]
            )
        
        # Split coordinates by layers
        layer_coords = []
        for layer in range(n_molecules):
            start_idx = layer * atoms_per_molecule
            end_idx = (layer + 1) * atoms_per_molecule
            layer_coords.append(stack_coords[start_idx:end_idx])
        
        # Check for severe overlaps first
        has_severe, severe_message = self.check_severe_overlaps(layer_coords)
        if has_severe:
            return StackValidationResult(
                is_valid=False,
                message=severe_message,
                issues=[severe_message],
                clash_penalty=float('inf'),
                min_interlayer_distance=0.0
            )
        
        # Calculate clash penalties for all layer pairs
        issues = []
        total_penalty = 0.0
        min_distance = float('inf')
        layer_clashes = {}
        
        for i in range(n_molecules):
            for j in range(i + 1, n_molecules):
                # Calculate clash penalty between layers i and j
                penalty = self.clash_penalty(layer_coords[i], layer_coords[j])
                
                # Calculate minimum distance between these layers
                diff = layer_coords[i][:, None, :] - layer_coords[j][None, :, :]
                distances = np.linalg.norm(diff, axis=-1)
                layer_min_dist = np.min(distances)
                min_distance = min(min_distance, layer_min_dist)
                
                # Weight penalty by layer separation (same as pi-stack)
                weight = 1.0 / float(j - i)
                weighted_penalty = weight * penalty
                total_penalty += weighted_penalty
                
                layer_clashes[(i, j)] = int(penalty)
                
                # Report issues if there are clashes
                if penalty > 0:
                    issues.append(
                        f"Layers {i+1} and {j+1}: {int(penalty)} clashing atom pairs "
                        f"(min distance: {layer_min_dist:.3f} Å, weighted penalty: {weighted_penalty:.2f})"
                    )
        
        # Apply penalty weight (same as pi-stack)
        final_penalty = self.penalty_weight * total_penalty
        
        # Determine if valid
        is_valid = len(issues) == 0
        
        if is_valid:
            message = f"Stack geometry is valid. Minimum inter-layer distance: {min_distance:.3f} Å"
        else:
            message = (f"Stack has {len(issues)} inter-layer clash issues. "
                      f"Total penalty: {final_penalty:.2f}, minimum distance: {min_distance:.3f} Å")
        
        return StackValidationResult(
            is_valid=is_valid,
            message=message,
            issues=issues,
            clash_penalty=final_penalty,
            min_interlayer_distance=min_distance,
            layer_pair_clashes=layer_clashes
        )


# -------------------------------------------------------------------
# XYZ File Parsing
# -------------------------------------------------------------------

def read_xyz_coords(xyz_file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Parse XYZ file and return atom types and coordinates.
    
    Args:
        xyz_file_path: Path to the XYZ file
        
    Returns:
        Tuple of (atom_types, coordinates) where:
        - atom_types: List of atomic symbols
        - coordinates: numpy array of shape (n_atoms, 3) with atomic coordinates
    """
    atom_types = []
    coords = []
    
    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError("Invalid XYZ file format - too few lines")
    
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("Invalid XYZ file format - first line should be number of atoms")
    
    # Skip comment line (line 1) and parse coordinates starting from line 2
    coord_lines = lines[2:2+n_atoms]
    
    if len(coord_lines) != n_atoms:
        raise ValueError(f"Expected {n_atoms} coordinate lines, found {len(coord_lines)}")
    
    for i, line in enumerate(coord_lines):
        parts = line.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid coordinate line {i+3}: {line.strip()}")
        
        atom_symbol = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Invalid coordinates on line {i+3}: {line.strip()}")
        
        atom_types.append(atom_symbol)
        coords.append([x, y, z])
    
    return atom_types, np.array(coords)


def parse_stack_xyz(xyz_content: str) -> Tuple[np.ndarray, List[str], int]:
    """
    Parse XYZ content of a molecular stack to extract coordinates and determine layer count.
    
    Args:
        xyz_content: XYZ file content as string
        
    Returns:
        (coordinates, atom_types_per_molecule, n_molecules)
    """
    lines = [line.strip() for line in xyz_content.strip().split('\n') if line.strip()]
    
    if len(lines) < 2:
        raise ValueError("Invalid XYZ format: too few lines")
    
    try:
        n_atoms = int(lines[0])
    except ValueError:
        raise ValueError("Invalid XYZ format: first line must be number of atoms")
    
    comment_line = lines[1]
    coords = []
    all_atom_types = []
    
    for i in range(2, 2 + n_atoms):
        if i >= len(lines):
            raise ValueError(f"Invalid XYZ format: expected {n_atoms} atoms, found fewer")
        
        parts = lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ format: line {i+1} has insufficient columns")
        
        atom_type = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Invalid XYZ format: non-numeric coordinates on line {i+1}")
        
        all_atom_types.append(atom_type)
        coords.append([x, y, z])
    
    coords = np.array(coords)
    
    # Try to determine number of molecules from comment or pattern
    n_molecules = 1
    if "layer" in comment_line.lower() or "molecule" in comment_line.lower():
        # Try to extract number from comment
        numbers = re.findall(r'\d+', comment_line)
        if numbers:
            n_molecules = int(numbers[0])
    
    # If we can't determine from comment, assume it's a single molecule
    atoms_per_molecule = n_atoms // n_molecules if n_molecules > 1 else n_atoms
    atom_types_per_molecule = all_atom_types[:atoms_per_molecule]
    
    return coords, atom_types_per_molecule, n_molecules

def build_bond_graph(atom_types: List[str], coords: np.ndarray) -> Dict[int, List[int]]:

    """
    Construct a chemical connectivity (bond) graph based on interatomic distances.

    This function identifies bonded atom pairs using covalent radii and returns the
    resulting connectivity as an adjacency list. Two atoms i and j are considered
    chemically bonded if the distance between them is less than 1.3 times the sum of
    their covalent radii. This tolerance accounts for bond-length variations due to
    structural relaxation or thermal motion.

    Parameters
    ----------
    atom_types : List[str]
        List of atomic symbols (e.g., ['C', 'H', 'H', ...]) defining the type of each atom.
    coords : np.ndarray
        Atomic coordinates as an (N, 3) array in Å, where N is the number of atoms.

    Returns
    -------
    Dict[int, List[int]]
        A bond adjacency list, where each key is an atom index (0-based) and the
        associated value is a list of indices of atoms directly bonded to that atom.
        This provides the molecular topology needed for identifying intramolecular
        contacts, detecting rotatable bonds, and preventing bonded atoms from being
        penalized as steric clashes during structure optimization.
        {
          0: [1],        # Atom 0 bonded to atom 1
          1: [0, 2, 3],  # Atom 1 bonded to atoms 0, 2, 3
          2: [1],        # Atom 2 bonded to atom 1
          3: [1, 4],     # Atom 3 bonded to 1 and 4
          4: [3, 5],     # etc…
          5: [4]
        }

    Notes
    -----
    Unknown elements default to a covalent radius of 1.5 Å.
    The returned graph is bidirectional: if i is bonded to j, then j is bonded to i.
    """

    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
        'F': 0.57, 'Cl': 1.02, 'Br': 1.20
    }
    n = len(atom_types)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            ri = covalent_radii.get(atom_types[i].capitalize(), 1.5)
            rj = covalent_radii.get(atom_types[j].capitalize(), 1.5)
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 1.3 * (ri + rj):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def check_monomer_geometry_sanity(
    coords: np.ndarray,
    atom_types: List[str],
    bonded_cutoff: float = 0.8,
    angle_cutoff: float = 1.1,
    nonbonded_cutoff: float = 1.3
) -> ValidationResult:

    # Input validation
    if not isinstance(coords, np.ndarray):
        try:
            coords = np.array(coords)
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.UNKNOWN_ERROR,
                is_valid=False,
                message=f"Invalid coordinates format: {type(coords)}. Expected numpy array or convertible format.",
                issues=[f"Coordinate conversion error: {str(e)}"]
            )
    
    if coords.ndim != 2 or coords.shape[1] != 3:
        return ValidationResult(
            status=ValidationStatus.UNKNOWN_ERROR,
            is_valid=False,
            message=f"Invalid coordinate shape: {coords.shape}. Expected (N, 3).",
            issues=[f"Coordinates must be a 2D array with 3 columns (x, y, z)"]
        )
    
    if len(atom_types) != len(coords):
        return ValidationResult(
            status=ValidationStatus.UNKNOWN_ERROR,
            is_valid=False,
            message=f"Mismatch between number of atoms ({len(atom_types)}) and coordinates ({len(coords)}).",
            issues=[f"Number of atom types ({len(atom_types)}) != number of coordinate rows ({len(coords)})"]
        )

    n = coords.shape[0]
    min_distance = float('inf')
    problematic_pairs = []

    bond_graph = build_bond_graph(atom_types, coords)

    bonded_pairs = set()
    angle_pairs = set()

    for i in bond_graph:
        for j in bond_graph[i]:
            bonded_pairs.add((min(i, j), max(i, j)))
            for k in bond_graph[j]:
                if k != i:
                    angle_pairs.add((min(i, k), max(i, k)))

    angle_pairs = angle_pairs - bonded_pairs

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            min_distance = min(min_distance, dist)
            pair_key = (i, j)

            if pair_key in bonded_pairs:
                cutoff = bonded_cutoff
                label = "1-2 bonded"
            elif pair_key in angle_pairs:
                cutoff = angle_cutoff
                label = "1-3 angle"
            else:
                cutoff = nonbonded_cutoff
                label = "non-bonded"

            if dist < cutoff:
                problematic_pairs.append(
                    f"Atoms {i+1} ({atom_types[i]}) and {j+1} ({atom_types[j]}) "
                    f"[{label}] are too close: {dist:.3f} Å"
                )

    # -----------------------
    # If geometry is invalid
    # -----------------------
    if problematic_pairs:
        message = (
            "Input geometry has atomic overlaps or unphysical short contacts.\n"
            f"Overall minimum distance: {min_distance:.3f} Å\n"
            f"Cutoffs used: bonded={bonded_cutoff:.2f} Å, "
            f"angle={angle_cutoff:.2f} Å, "
            f"non-bonded={nonbonded_cutoff:.2f} Å"
        )

        return ValidationResult(
            status=ValidationStatus.ATOMIC_OVERLAPS,
            is_valid=False,
            message=message,
            issues=problematic_pairs,
            symmetry_info=None
        )

    # -----------------------
    # Geometry is valid
    # -----------------------
    return ValidationResult(
        status=ValidationStatus.VALID,
        is_valid=True,
        message="Molecule is valid.",
        issues=[],
        symmetry_info=None
    )


def validate_stack_from_xyz(xyz_content: str, 
                           clash_cutoff: float = 1.6,
                           penalty_weight: float = 2.0) -> StackValidationResult:
    """
    Convenience function to validate a stack directly from XYZ content.
    
    Args:
        xyz_content: XYZ file content as string
        clash_cutoff: Inter-molecular clash detection cutoff (default: 1.6 Å)
        penalty_weight: Weight for clash penalties (default: 2.0)
        
    Returns:
        StackValidationResult with validation details
        
    Example:
        >>> with open('stack.xyz', 'r') as f:
        ...     xyz_content = f.read()
        >>> result = validate_stack_from_xyz(xyz_content)
        >>> if result.is_valid:
        ...     print("Stack is valid!")
        >>> else:
        ...     print(f"Stack issues: {result.message}")
        ...     for issue in result.issues:
        ...         print(f"  - {issue}")
    """
    try:
        coords, atom_types, n_molecules = parse_stack_xyz(xyz_content)
        
        validator = StackGeometryValidator(
            clash_cutoff=clash_cutoff,
            penalty_weight=penalty_weight
        )
        
        return validator.validate_stack_geometry(coords, atom_types, n_molecules)
        
    except Exception as e:
        return StackValidationResult(
            is_valid=False,
            message=f"Failed to parse XYZ content: {str(e)}",
            issues=[f"XYZ parsing error: {str(e)}"]
        )


def validate_built_stack(stack_coords: np.ndarray, 
                        atom_types: List[str], 
                        n_molecules: int,
                        clash_cutoff: float = 1.6,
                        penalty_weight: float = 2.0) -> StackValidationResult:
    """
    Convenience function to validate a stack built by build_stack.py.
    
    Args:
        stack_coords: All atomic coordinates in the stack (N_total, 3)
        atom_types: Atomic symbols for one molecule
        n_molecules: Number of molecules in the stack
        clash_cutoff: Inter-molecular clash detection cutoff (default: 1.6 Å) 
        penalty_weight: Weight for clash penalties (default: 2.0)
        
    Returns:
        StackValidationResult with validation details
        
    Example:
        >>> # After running build_stack.py
        >>> coords, atom_types = read_xyz_file("stack.xyz")
        >>> result = validate_built_stack(coords, atom_types[:atoms_per_mol], n_molecules=3)
        >>> if not result.is_valid:
        ...     print(f"Stack validation failed: {result.message}")
    """
    validator = StackGeometryValidator(
        clash_cutoff=clash_cutoff,
        penalty_weight=penalty_weight
    )
    
    return validator.validate_stack_geometry(stack_coords, atom_types, n_molecules)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def should_keep_stereo(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    # 1. Detect chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if not chiral_centers:
        return False  # nothing to preserve
    
    # 2. Check heteroatoms (important for your case)
    has_hetero = any(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms())
    
    # 3. Check flexibility
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    # 4. Check ring / rigidity
    has_ring = mol.GetRingInfo().NumRings() > 0
    
    # 🔥 Decision logic
    if has_hetero:
        return True  # N/O/S → keep stereo
    
    if has_ring:
        return True  # rigid → keep stereo
    
    if rot_bonds < 3:
        return True  # not flexible → keep stereo
    
    return False  # flexible alkyl → ignore

def are_equivalent(sm1: str, sm2: str) -> bool:
    # Handle empty / None cases explicitly
    if not sm1 and not sm2:
        return True   # both absent
    
    if not sm1 or not sm2:
        return False  # one present, one absent
    
    m1 = Chem.MolFromSmiles(sm1)
    m2 = Chem.MolFromSmiles(sm2)

    if m1 is None or m2 is None:
        return sm1 == sm2  # fallback only if both invalid
    
    return Chem.MolToInchiKey(m1) == Chem.MolToInchiKey(m2)

def clean_and_canon(smiles):
    if not smiles:
        return "H"
    
    # Remove attachment labels
    cleaned = re.sub(r'\[\*:?[\d]*\]|\[\d+\*\]', '', smiles)
    cleaned = cleaned.replace('()', '').strip()
    
    if not cleaned or cleaned in ['[H]', '']:
        return "H"
    
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return cleaned
    
    # 🔥 NEW: conditional stereo handling
    if not should_keep_stereo(cleaned):
        Chem.RemoveStereochemistry(mol)
    
    return Chem.MolToSmiles(mol, canonical=True)

def check_input_smile_symmetry(smiles: str) -> ValidationResult:
    """
    Check if input SMILES has PDI/NDI core with symmetric sidechains.
    
    Args:
        smiles: SMILES string of the molecule to validate
        
    Returns:
        ValidationResult with validation status and details
    """
    # Define core structures
    NDI_CORE_SMILES = 'O=C1N([*:1])C(=O)c2ccc3c4c(ccc1c24)C(=O)N([*:2])C3=O'
    PDI_CORE_SMILES = "C1=CC2=C3C(=CC=C4C3=C1C5=C6C4=CC=C7C6=C(C=C5)C(=O)N([*:1])C7=O)C(=O)N([*:2])C2=O"
    
    NDI_CORE_PATTERN = 'O=C1NC(=O)c2ccc3c4c(ccc1c24)C(=O)NC3=O'
    PDI_CORE_PATTERN = "C1=CC2=C3C(=CC=C4C3=C1C5=C6C4=CC=C7C6=C(C=C5)C(=O)NC7=O)C(=O)NC2=O"
    
    # Validate input SMILES
    if not smiles or not smiles.strip():
        return ValidationResult(
            status=ValidationStatus.UNKNOWN_ERROR,
            is_valid=False,
            message="Empty SMILES string provided",
            issues=["SMILES string is empty or None"]
        )
    
    # Parse input molecule
    input_mol = Chem.MolFromSmiles(smiles)
    if input_mol is None:
        return ValidationResult(
            status=ValidationStatus.UNKNOWN_ERROR,
            is_valid=False,
            message="Invalid SMILES string",
            issues=["Failed to parse SMILES string"]
        )
    
    # Parse core structures
    pdi_core = Chem.MolFromSmiles(PDI_CORE_SMILES)
    ndi_core = Chem.MolFromSmiles(NDI_CORE_SMILES)
    PDI_core_pattern = Chem.MolFromSmiles(PDI_CORE_PATTERN)
    NDI_core_pattern = Chem.MolFromSmiles(NDI_CORE_PATTERN)
    
    # Check for PDI core first
    pdi_matches = input_mol.GetSubstructMatches(PDI_core_pattern)
    pdi_core_count = len(pdi_matches)
    
    if pdi_core_count == 0:
        # No PDI core found, check for NDI core
        ndi_matches = input_mol.GetSubstructMatches(NDI_core_pattern)
        ndi_core_count = len(ndi_matches)
        
        if ndi_core_count == 0:
            return ValidationResult(
                status=ValidationStatus.UNKNOWN_ERROR,
                is_valid=False,
                message="No PDI or NDI core structure found in molecule",
                issues=["Molecule does not contain PDI or NDI core"]
            )
        elif ndi_core_count > 1:
            return ValidationResult(
                status=ValidationStatus.MORE_THAN_ONE_CORE,
                is_valid=False,
                message=f"Input SMILES has {ndi_core_count} NDI cores (expected 1)",
                issues=[f"Found {ndi_core_count} NDI core structures"]
            )
        else:
            # Exactly one NDI core found, check for symmetry
            try:
                rgd = rdRG.RGroupDecomposition(ndi_core)
                rgd.Add(input_mol)
                if rgd.Process() < 0:
                    return ValidationResult(
                        status=ValidationStatus.UNKNOWN_ERROR,
                        is_valid=False,
                        message="Failed to decompose NDI molecule into core and sidechains",
                        issues=["R-group decomposition failed"]
                    )
                
                rows = rgd.GetRGroupsAsRows(asSmiles=True)
                if not rows:
                    return ValidationResult(
                        status=ValidationStatus.UNKNOWN_ERROR,
                        is_valid=False,
                        message="No R-groups extracted from NDI molecule",
                        issues=["R-group extraction returned empty results"]
                    )
                
                r1_raw = rows[0].get('R1', '[*:1][H]')
                r2_raw = rows[0].get('R2', '[*:2][H]')
                
                r1_c = clean_and_canon(r1_raw)
                r2_c = clean_and_canon(r2_raw)
                
                if not are_equivalent(r1_c, r2_c):
                    return ValidationResult(
                        status=ValidationStatus.NOT_SYMMETRIC,
                        is_valid=False,
                        message="NDI molecule has asymmetric sidechains",
                        issues=[f"R1 sidechain: {r1_c}", f"R2 sidechain: {r2_c}"],
                        symmetry_info={"core_type": "NDI", "r1": r1_c, "r2": r2_c, "is_symmetric": False}
                    )
                
                # Success: NDI core with symmetric sidechains
                return ValidationResult(
                    status=ValidationStatus.VALID,
                    is_valid=True,
                    message="Valid NDI molecule with symmetric sidechains",
                    issues=[],
                    symmetry_info={"core_type": "NDI", "sidechain": r1_c, "is_symmetric": True}
                )
                
            except Exception as e:
                return ValidationResult(
                    status=ValidationStatus.UNKNOWN_ERROR,
                    is_valid=False,
                    message=f"Error during NDI symmetry analysis: {str(e)}",
                    issues=[f"Exception: {str(e)}"]
                )
    
    elif pdi_core_count > 1:
        # Multiple PDI cores found
        return ValidationResult(
            status=ValidationStatus.MORE_THAN_ONE_CORE,
            is_valid=False,
            message=f"Input SMILES has {pdi_core_count} PDI cores (expected 1)",
            issues=[f"Found {pdi_core_count} PDI core structures"]
        )
    
    else:
        # Exactly one PDI core found, check for symmetry
        try:
            rgd = rdRG.RGroupDecomposition(pdi_core)
            rgd.Add(input_mol)
            if rgd.Process() < 0:
                return ValidationResult(
                    status=ValidationStatus.UNKNOWN_ERROR,
                    is_valid=False,
                    message="Failed to decompose PDI molecule into core and sidechains",
                    issues=["R-group decomposition failed"]
                )
            
            rows = rgd.GetRGroupsAsRows(asSmiles=True)
            if not rows:
                return ValidationResult(
                    status=ValidationStatus.UNKNOWN_ERROR,
                    is_valid=False,
                    message="No R-groups extracted from PDI molecule",
                    issues=["R-group extraction returned empty results"]
                )
            
            r1_raw = rows[0].get('R1', '[*:1][H]')
            r2_raw = rows[0].get('R2', '[*:2][H]')
            
            r1_c = clean_and_canon(r1_raw)
            r2_c = clean_and_canon(r2_raw)
            
            if not are_equivalent(r1_c, r2_c):
                return ValidationResult(
                    status=ValidationStatus.NOT_SYMMETRIC,
                    is_valid=False,
                    message="PDI molecule has asymmetric sidechains",
                    issues=[f"R1 sidechain: {r1_c}", f"R2 sidechain: {r2_c}"],
                    symmetry_info={"core_type": "PDI", "r1": r1_c, "r2": r2_c, "is_symmetric": False}
                )
            
            # Success: PDI core with symmetric sidechains
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_valid=True,
                message="Valid PDI molecule with symmetric sidechains",
                issues=[],
                symmetry_info={"core_type": "PDI", "sidechain": r1_c, "is_symmetric": True}
            )
            
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.UNKNOWN_ERROR,
                is_valid=False,
                message=f"Error during PDI symmetry analysis: {str(e)}",
                issues=[f"Exception: {str(e)}"]
            )

               
