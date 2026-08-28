from __future__ import annotations
import numpy as np
import math
import logging
from typing import Dict, Any, List, Tuple, Optional

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)



VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

# π-cloud effective radii (smaller than vdW for electronic overlap)
PI_RADII_SCALE = 0.85  # π-clouds are ~85% of vdW radius

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


# ============================================================
# STERIC CLASH DETECTION
# ============================================================


def check_intramolecular_catastrophic_overlap(coords: np.ndarray, 
                                              atom_types: List[str],
                                              adj: Dict[int, List[int]]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for catastrophic intramolecular overlaps within a single molecule.
    
    Returns True if:
    - Any non-bonded distance < 1.5 Å
    - More than 5 atom pairs closer than 2.0 Å
    """
    
    n = coords.shape[0]
    excluded = set()
    
    # Exclude 1-2 and 1-3 bonded neighbors
    for i in adj:
        for j in adj[i]:
            excluded.add((min(i,j), max(i,j)))
            for k in adj[j]:
                if k != i:
                    excluded.add((min(i,k), max(i,k)))
    
    min_dist = float('inf')
    n_very_close = 0  
    n_close = 0  
    violation_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in excluded:
                continue
            
            dist = np.linalg.norm(coords[i] - coords[j])
            min_dist = min(min_dist, dist)
            
            if dist < 1.5:
                n_very_close += 1
                violation_pairs.append((i, j, dist))
            elif dist < 2.0:
                logger.info("less than 2.0 armstrong distance found")
                n_close += 1
    
    has_catastrophic = n_very_close > 0 or n_close > 5
    
    details = {
        "min_distance": min_dist,
        "n_very_close": n_very_close,  # < 1.5 Å
        "n_close": n_close,  # < 2.0 Å
        "has_catastrophic": has_catastrophic,
        "violation_pairs": violation_pairs[:5]  # First 5 violations
    }
    
    return has_catastrophic, details


def check_interlayer_catastrophic_overlap(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for catastrophic interlayer overlaps (simplified, no vdW).
    
    Uses cKDTree for O(n log n) performance if scipy available.
    
    Returns True if:
    - Any interatomic distance < 1.5 Å
    - More than 5 atom pairs closer than 2.0 Å
    
    Args:
        coords1, coords2: 3D coordinates of two layers
    
    Returns:
        (has_catastrophic_overlap, details_dict)
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return False, {"min_distance": float('inf'), "n_very_close": 0, "n_close": 0}
    
    # Use cKDTree for large systems if available
    if SCIPY_AVAILABLE and (len(coords1) > 20 or len(coords2) > 20):
        tree = cKDTree(coords2)
        distances, _ = tree.query(coords1, k=1)
        min_dist = float(np.min(distances))
        close_neighbors = tree.query_ball_point(coords1, r=2.0)
        n_very_close = 0
        n_close = 0
        for atom_coord, neighbor_indices in zip(coords1, close_neighbors):
            if not neighbor_indices:
                continue
            neighbor_coords = coords2[neighbor_indices]
            pair_distances = np.linalg.norm(neighbor_coords - atom_coord, axis=1)
            n_very_close += int(np.sum(pair_distances < 1.5))
            n_close += int(np.sum(pair_distances < 2.0))
    else:
        # Vectorized approach for medium systems
        diff = coords1[:, None, :] - coords2[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_dist = float(np.min(dists))
        n_very_close = int(np.sum(dists < 1.5))
        n_close = int(np.sum(dists < 2.0))
    
    has_catastrophic = n_very_close > 0 or n_close > 5
    
    details = {
        "min_distance": min_dist,
        "n_very_close": n_very_close,
        "n_close": n_close,
        "has_catastrophic": has_catastrophic
    }
    
    return has_catastrophic, details


# ============================================================
# XYZ PARSING & STACK SPLITTING
# ============================================================

def _parse_xyz_block(xyz: str) -> Tuple[List[str], np.ndarray]:
    """Parse XYZ block into atom types and coordinates."""
    lines = [l.strip() for l in xyz.strip().splitlines()]
    n_atoms = int(lines[0])
    
    atom_types = []
    coords = []
    
    for line in lines[2:2+n_atoms]:
        p = line.split()
        atom_types.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    
    return atom_types, np.array(coords, dtype=float)


def _split_stack_into_monomers(stack_xyz: str, monomer_xyz: str) -> Tuple[List[List[str]], List[np.ndarray]]:
    """Split stack into individual monomers."""
    _, stack = _parse_xyz_block(stack_xyz)
    atom_types, mono = _parse_xyz_block(monomer_xyz)
    
    n = len(mono)
    if n == 0:
        raise ValueError("Monomer XYZ contains no atoms.")
    if len(stack) % n != 0:
        raise ValueError(
            f"Stack atom count ({len(stack)}) is not an integer multiple of monomer atom count ({n})."
        )

    n_stack = len(stack) // n
    
    monomer_coords = [stack[i*n:(i+1)*n] for i in range(n_stack)]
    monomer_types = [atom_types[:] for _ in range(n_stack)]
    
    return monomer_types, monomer_coords


# ============================================================
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

        #logger.info(f"π-core type: {struct_type} (aromatic_core_atoms={n_arom}, imide_N={n_imide_n})")
        #logger.debug(f"Core atoms ({struct_type}): {core_idx}")
        #logger.debug(f"Sidechain atoms: {sidechain_idx}")
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
        return None
        
    except Exception as e:
        logger.error(f"SMILES-based π-core detection failed: {e}", exc_info=True)
        return None


def _infer_bond_graph_from_xyz(atom_types: List[str], coords: np.ndarray) -> Dict[int, List[int]]:
    """Infer a conservative heavy-atom bond graph from XYZ distances."""
    adj = {i: [] for i in range(len(atom_types))}
    for i in range(len(atom_types)):
        ai = atom_types[i]
        if ai.upper() == "H":
            continue
        ri = COVALENT_RADII.get(ai, 0.76)
        for j in range(i + 1, len(atom_types)):
            aj = atom_types[j]
            if aj.upper() == "H":
                continue
            rj = COVALENT_RADII.get(aj, 0.76)
            cutoff = 1.25 * (ri + rj) + 0.15
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.4 < dist <= cutoff:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _find_small_cycles(adj: Dict[int, List[int]], min_size: int = 5, max_size: int = 8) -> List[set]:
    """Find unique simple cycles up to max_size in an undirected graph."""
    cycles = set()
    nodes = sorted(adj)

    for start in nodes:
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            if len(path) > max_size:
                continue

            for nb in adj[current]:
                if nb == start and len(path) >= min_size:
                    cycles.add(frozenset(path))
                    continue
                if nb in path:
                    continue
                # Force start to be the smallest atom index in the cycle to
                # reduce duplicate traversals.
                if nb < start:
                    continue
                stack.append((nb, path + [nb]))

    return [set(c) for c in cycles]


def _plane_rmsd(coords: np.ndarray) -> float:
    """RMS distance of points from their best-fit plane."""
    if len(coords) < 3:
        return float("inf")
    centered = coords - coords.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered)
    except Exception:
        return float("inf")
    normal = vt[-1]
    distances = centered @ normal
    return float(np.sqrt(np.mean(distances * distances)))


def _get_pi_indices_from_xyz_graph(monomer_xyz: str) -> Optional[List[int]]:
    """Detect a π-core directly from XYZ using bonds, rings, fusion, and planarity."""
    atoms, coords = _parse_xyz_block(monomer_xyz)
    adj = _infer_bond_graph_from_xyz(atoms, coords)
    allowed_ring_atoms = {"C", "N", "O", "S", "P"}

    rings = []
    for ring in _find_small_cycles(adj):
        if not all(atoms[i] in allowed_ring_atoms for i in ring):
            continue
        ring_rmsd = _plane_rmsd(coords[sorted(ring)])
        if ring_rmsd <= 0.25:
            rings.append(ring)

    if not rings:
        logger.info("XYZ graph π-core detection found no planar rings.")
        return None

    ring_adj = [[] for _ in rings]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i].intersection(rings[j]):
                ring_adj[i].append(j)
                ring_adj[j].append(i)

    best_atoms: set = set()
    best_rmsd = float("inf")
    seen = [False] * len(rings)

    for i in range(len(rings)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp_atoms: set = set()

        while stack:
            u = stack.pop()
            comp_atoms |= rings[u]
            for v in ring_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        if len(comp_atoms) < 6:
            continue
        comp_idx = sorted(comp_atoms)
        comp_rmsd = _plane_rmsd(coords[comp_idx])
        if comp_rmsd > 0.35:
            continue

        if len(comp_atoms) > len(best_atoms) or (
            len(comp_atoms) == len(best_atoms) and comp_rmsd < best_rmsd
        ):
            best_atoms = comp_atoms
            best_rmsd = comp_rmsd

    if len(best_atoms) < 6:
        logger.info("XYZ graph π-core detection found no fused planar component.")
        return None

    core_indices = sorted(best_atoms)
    logger.info(
        "XYZ graph π-core detection successful: %d atoms, plane RMSD %.3f Å",
        len(core_indices),
        best_rmsd,
    )
    logger.info(
        "XYZ graph π-core composition: %s",
        dict(__import__("collections").Counter([atoms[i] for i in core_indices])),
    )
    return core_indices

# ============================================================
# PLANE & PROJECTION UTILITIES
# ============================================================

def _plane_normal(coords: np.ndarray) -> np.ndarray:
    """Compute plane normal using SVD."""
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    return vt[-1]


def _average_aligned_normals(normals: List[np.ndarray]) -> np.ndarray:
    """Average plane normals after aligning arbitrary SVD signs."""
    if not normals:
        return np.array([0.0, 0.0, 1.0])

    reference = np.asarray(normals[0], dtype=float)
    reference_norm = np.linalg.norm(reference)
    if reference_norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    reference = reference / reference_norm

    aligned = []
    for normal in normals:
        unit = np.asarray(normal, dtype=float)
        unit_norm = np.linalg.norm(unit)
        if unit_norm < 1e-12:
            continue
        unit = unit / unit_norm
        if np.dot(unit, reference) < 0:
            unit = -unit
        aligned.append(unit)

    if not aligned:
        return reference

    avg = np.mean(aligned, axis=0)
    avg_norm = np.linalg.norm(avg)
    if avg_norm < 1e-12:
        return reference
    return avg / avg_norm


def _project_to_pi_plane(coords: np.ndarray, normal: np.ndarray, origin: Optional[np.ndarray] = None) -> np.ndarray:
    """Project 3D coordinates onto a 2D π-plane using an orthonormal basis.

    IMPORTANT FOR OVERLAP:
    - If you project two layers using different origins (e.g., each layer's
      own centroid), you remove the in-plane translation (slip) and overlap is
      artificially inflated.
    - For cross-layer overlap, pass the SAME origin for both layers.

    Args:
        coords: (N,3) coordinates
        normal: plane normal
        origin: 3D origin for projection. If None, uses coords.mean(axis=0)
                (fine for shape/area, NOT fine for cross-layer overlap).
    """
    normal = normal / np.linalg.norm(normal)
    
    # Choose reference vector orthogonal to normal
    ref = np.array([1., 0., 0.])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0., 1., 0.])
    
    # Gram-Schmidt orthogonalization
    e1 = ref - np.dot(ref, normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    
    if origin is None:
        origin = coords.mean(axis=0)
    
    x = np.dot(coords - origin, e1)
    y = np.dot(coords - origin, e2)
    
    return np.column_stack([x, y])


# ============================================================
# CORE AREA ESTIMATION
# ============================================================

def _estimate_core_area(coords_2d: np.ndarray) -> float:
    """
    Estimate π-core area using convex hull (more accurate than bounding box).
    Falls back to bounding box if convex hull fails.
    
    Args:
        coords_2d: 2D projected coordinates
    
    Returns:
        Core area in A^2
    """
    if len(coords_2d) < 3:
        return 0.0
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords_2d)
        area = float(hull.volume)  # In 2D, 'volume' is area
        logger.debug(f"Convex hull area: {area:.1f} A^2")
        return area
    except Exception as e:
        # Fallback to bounding box
        area = float(np.ptp(coords_2d[:, 0]) * np.ptp(coords_2d[:, 1]))
        logger.debug(f"Bounding box area (fallback): {area:.1f} A^2")
        return area


# ============================================================
# OVERLAP CALCULATION
# ============================================================

def _convex_hull_overlap_percent(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Grid-based π-core overlap using density rasterization.
    More forgiving than IoU (good for initial screening).
    """
    resolution = 0.3  # Å grid spacing
    
    all_pts = np.vstack([p1, p2])
    
    xmin, ymin = all_pts.min(axis=0) - 2.0
    xmax, ymax = all_pts.max(axis=0) + 2.0
    
    nx = int((xmax - xmin) / resolution) + 1
    ny = int((ymax - ymin) / resolution) + 1
    
    grid1 = np.zeros((nx, ny), dtype=bool)
    grid2 = np.zeros((nx, ny), dtype=bool)
    
    def rasterize(points, grid):
        for x, y in points:
            ix = int((x - xmin) / resolution)
            iy = int((y - ymin) / resolution)
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    xi, yi = ix + dx, iy + dy
                    if 0 <= xi < nx and 0 <= yi < ny:
                        grid[xi, yi] = True
    
    rasterize(p1, grid1)
    rasterize(p2, grid2)
    
    inter = np.logical_and(grid1, grid2).sum()
    min_area = min(grid1.sum(), grid2.sum())
    
    if min_area == 0:
        return 0.0
    
    overlap = inter / min_area * 100.0
    
    return float(overlap)


def _pi_cloud_overlap_metrics(
    p1: np.ndarray,
    p2: np.ndarray,
    types1: List[str],
    types2: List[str],
    pi_idx: List[int],
    resolution: float = 0.20,
) -> Tuple[float, float]:
    """Compute π-core overlap metrics using rasterized π-cloud discs.

    We compute two related metrics:
      1) IoU (intersection / union)  -> strict overlap
      2) Coverage (intersection / min(areaA, areaB)) -> forgiving overlap

    NOTE:
    - For NDI/PDI stacks, both cores are typically identical, so IoU and
      coverage are usually close.
    - Coverage is useful as a "is it basically overlapping" score.

    Args:
        p1, p2: (N,2) 2D projected π-core coordinates for layer A and B.
        types1, types2: Full atom type lists for the two layers.
        pi_idx: π-core atom indices (in the original monomer atom ordering).
        resolution: grid spacing in Å (smaller = more accurate, slower).

    Returns:
        (iou_percent, coverage_percent)
    """
    margin = 1.0
    
    # Get π-cloud effective radii (85% of vdW for electronic overlap)
    radii1 = np.array([VDW_RADII.get(types1[i], 1.7) * PI_RADII_SCALE for i in pi_idx])
    radii2 = np.array([VDW_RADII.get(types2[i], 1.7) * PI_RADII_SCALE for i in pi_idx])
    
    max_radius = max(radii1.max(), radii2.max())
    
    all_pts = np.vstack([p1, p2])
    xmin, ymin = all_pts.min(axis=0) - margin - max_radius
    xmax, ymax = all_pts.max(axis=0) + margin + max_radius
    
    xs = np.arange(xmin, xmax + resolution, resolution)
    ys = np.arange(ymin, ymax + resolution, resolution)
    
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    
    occ_a = np.zeros(grid.shape[0], dtype=bool)
    occ_b = np.zeros(grid.shape[0], dtype=bool)
    
    # Rasterize π-cloud discs for layer A
    for atom_xy, radius in zip(p1, radii1):
        delta = grid - atom_xy
        r2 = radius * radius
        occ_a |= np.sum(delta * delta, axis=1) <= r2
    
    # Rasterize π-cloud discs for layer B
    for atom_xy, radius in zip(p2, radii2):
        delta = grid - atom_xy
        r2 = radius * radius
        occ_b |= np.sum(delta * delta, axis=1) <= r2
    
    union = occ_a | occ_b
    union_area = int(np.sum(union))
    if union_area == 0:
        return 0.0, 0.0

    inter = occ_a & occ_b
    inter_area = int(np.sum(inter))

    area_a = int(np.sum(occ_a))
    area_b = int(np.sum(occ_b))
    min_area = min(area_a, area_b)
    if min_area == 0:
        return 0.0, 0.0

    iou = float(inter_area / union_area)
    coverage = float(inter_area / min_area)
    return iou * 100.0, coverage * 100.0
        
    


# ============================================================
# SLIP ANGLE CALCULATION (FIXED)
# ============================================================

def _slip_angle_deg(pi_core_3d: np.ndarray, slip_vector_3d: np.ndarray) -> float:
    """
    Calculate slip angle between slip direction and π-core principal axis.
    
    Uses 3D geometry for accurate calculation. Returns NaN for symmetric cores
    where the angle is geometrically undefined.
    
    Args:
        pi_core_3d: 3D π-core coordinates (Nx3)
        slip_vector_3d: 3D in-plane slip vector (magnitude & direction)
    
    Returns:
        Slip angle in degrees (0-90), or NaN for symmetric cores or no slip
    """
    slip_magnitude = np.linalg.norm(slip_vector_3d)
    
    # No slip case
    if slip_magnitude < 1e-6:
        return np.nan
    
    # Compute principal axes from 3D π-core coordinates
    if len(pi_core_3d) < 2:
        return np.nan
    
    centered = pi_core_3d - pi_core_3d.mean(axis=0)
    
    try:
        _, s, vt = np.linalg.svd(centered)
    except Exception:
        return np.nan
    
    # Check if core is approximately circular/symmetric
    # Compare two largest eigenvalues (smallest is near-zero for planar core)
    if len(s) >= 2:
        ratio = s[1] / (s[0] + 1e-12)
        if ratio > 0.9:
            return np.nan  # ✅ Symmetric core - slip angle undefined
    
    # Get principal axis (largest variance direction)
    principal_axis = vt[0]
    
    # Normalize slip vector
    slip_unit = slip_vector_3d / slip_magnitude
    
    # Angle between slip direction and principal axis
    cosang = abs(np.dot(slip_unit, principal_axis))
    cosang = np.clip(cosang, 0.0, 1.0)
    
    return float(math.degrees(math.acos(cosang)))


# ============================================================
# QUALITY CLASSIFICATION (FIXED)
# ============================================================

def _get_pi_indices_from_xyz_with_method(monomer_xyz: str) -> Tuple[List[int], str]:
    """
    Detect π-core from XYZ coordinates.

    First uses a bond-graph/ring/planarity method that preserves the uploaded
    XYZ atom ordering. If that cannot identify a ring system, falls back to the
    older centroid-gap heuristic.
    """
    graph_core = _get_pi_indices_from_xyz_graph(monomer_xyz)
    if graph_core and len(graph_core) >= 3:
        return graph_core, "xyz_graph_rings"

    atoms, coords = _parse_xyz_block(monomer_xyz)
    
    heavy_idx = [i for i, a in enumerate(atoms) if a.upper() != "H"]
    if len(heavy_idx) < 3:
        return heavy_idx,"xyz_heavy_atoms"
    
    heavy_coords = coords[heavy_idx]
    centroid = heavy_coords.mean(axis=0)
    distances = np.linalg.norm(heavy_coords - centroid, axis=1)
    
    # ============================================================
    # MULTI-LEVEL GAP DETECTION
    # ============================================================
    sorted_dists = np.sort(distances)
    gaps = np.diff(sorted_dists)
    top_gap_indices = np.argsort(gaps)[::-1][:3]
    
    #logger.info("=" * 80)
    #logger.info("π-CORE DETECTION (XYZ - Multi-Level Gap Detection)")
    #logger.info("=" * 80)
    #logger.info(f"Total atoms: {len(atoms)}, Heavy atoms: {len(heavy_idx)}")
    #logger.info(f"Distance range: {distances.min():.3f}Å to {distances.max():.3f}Å")
    #logger.info(f"Top 3 gaps:")
    
    threshold = None
    best_n_atoms = 0
    
    for rank, gap_idx in enumerate(top_gap_indices, 1):
        gap_size = gaps[gap_idx]
        t = (sorted_dists[gap_idx] + sorted_dists[gap_idx + 1]) / 2.0
        n = np.sum(distances < t)
        
        logger.info(f"  #{rank} - Gap: {gap_size:.3f}Å, threshold: {t:.3f}Å → {n} atoms")
        
        # Priority: polycyclic (15-25) > aromatic (6-14) > any (5+)
        if 15 <= n <= 25:
            threshold = t
            best_n_atoms = n
            logger.info(f"     ✅ POLYCYCLIC: {n} atoms")
            break
        
        if threshold is None and 6 <= n <= 14:
            threshold = t
            best_n_atoms = n
            logger.info(f"     ⚠️ AROMATIC: {n} atoms (tentative)")
        
        if threshold is None and n >= 5:
            threshold = t
            best_n_atoms = n
            logger.info(f"     ⚠️ MINIMUM: {n} atoms")
    
    # Fallback to percentile if gaps don't work
    if threshold is None:
        logger.info(f"  No suitable gap, trying percentiles...")
        for pct_name, pct_val in [("50th", 50), ("75th", 75), ("25th", 25)]:
            t = np.percentile(distances, pct_val)
            n = np.sum(distances < t)
            if 6 <= n <= 25:
                threshold = t
                best_n_atoms = n
                logger.info(f"  ✅ Using {pct_name}: {n} atoms")
                break
        
        if threshold is None:
            threshold = np.percentile(distances, 50)
            best_n_atoms = np.sum(distances < threshold)
    
    # Validation
    if best_n_atoms < 6:
        logger.warning(f"⚠️ Only {best_n_atoms} atoms (< 6). Using 50th percentile.")
        threshold = np.percentile(distances, 50)
        best_n_atoms = np.sum(distances < threshold)
    
    core_indices = sorted([heavy_idx[i] for i in range(len(heavy_idx)) if distances[i] < threshold])
    
    logger.info(f"Final: {len(core_indices)} atoms, threshold: {threshold:.3f}Å")
    #logger.info(f"Composition: {dict(__import__('collections').Counter([atoms[i] for i in core_indices]))}")
    logger.info("=" * 80)
    
    return (core_indices if len(core_indices) >= 3 else heavy_idx), "xyz_gap_clustering"


def _get_pi_indices_from_xyz(monomer_xyz: str) -> List[int]:
    """Backward-compatible XYZ π-core detector returning only atom indices."""
    core_indices, _ = _get_pi_indices_from_xyz_with_method(monomer_xyz)
    return core_indices


def analyze_pi_stack(
    stack_xyz: str,
    monomer_xyz: str,
    smiles: Optional[str] = None,
    smiles_source: Optional[str] = None,
    adjacency: Optional[Dict[int, List[int]]] = None
) -> Dict[str, Any]:
    
    try:
        monomer_types, monomers = _split_stack_into_monomers(stack_xyz, monomer_xyz)
        
        if len(monomers) < 2:
            return {"error": "Need ≥2 monomers"}
        
        pi_idx = None
        pi_method = "unknown"
        
        use_smiles_indices = bool(smiles) and smiles_source == "user_provided"

        # ✅ TRY SMILES FIRST only when atom ordering is expected to match XYZ.
        if use_smiles_indices:
            logger.info("=" * 80)
            logger.info("Attempting π-core detection via SMILES...")
            logger.info("=" * 80)
            pi_idx = _get_pi_indices_from_smiles(smiles)
            if pi_idx and len(pi_idx) >= 3:
                pi_method = "smiles_pdi_ndi_substructure"
                logger.info(f"✅ SUCCESS: {len(pi_idx)} atoms detected")
            else:
                pi_idx = None
                logger.info("⚠️ SMILES detection failed")
        elif smiles:
            logger.info(
                "Skipping SMILES π-core detection because smiles_source=%s; "
                "using XYZ atom ordering instead.",
                smiles_source,
            )
        
        # ✅ FALLBACK TO XYZ (NOW FIXED!)
        if pi_idx is None:
            logger.info("=" * 80)
            logger.info("Falling back to XYZ heuristic...")
            #logger.info("=" * 80)
            pi_idx, pi_method = _get_pi_indices_from_xyz_with_method(monomer_xyz)
            logger.info(f"⚠️ XYZ fallback: {len(pi_idx)} atoms detected")
        
        if len(pi_idx) < 3:
            return {"error": f"Too few π atoms ({len(pi_idx)})"}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL π-CORE: {pi_method}, {len(pi_idx)} atoms")
        #logger.info(f"{'='*80}\n")
        
        # ✅ CORE AREA CALCULATION (with logging)
        _, mono_coords = _parse_xyz_block(monomer_xyz)
        core_coords_3d = mono_coords[pi_idx]
        
        pi_plane_normal = _plane_normal(core_coords_3d)
        core_coords_2d = _project_to_pi_plane(core_coords_3d, pi_plane_normal)
        core_area = _estimate_core_area(core_coords_2d)
        
        logger.info(f"\nπ-CORE GEOMETRY:")
        logger.info(f"  π-core atoms: {len(pi_idx)}")
        #logger.info(f"  Core area (2D footprint): {core_area:.1f} Ų")
        #logger.info(f"  π-plane normal: {pi_plane_normal}")
        
        if core_area < 10:
            logger.warning(f"⚠️ Core area ({core_area:.1f} Ų) suspiciously small!")
        
        # Rest of function...
        results = []
        all_slip_angles = []
        
        layer_normals = []
        for mon in monomers:
            mon_core = mon[pi_idx]
            mon_normal = _plane_normal(mon_core)
            mon_normal = mon_normal / np.linalg.norm(mon_normal)
            layer_normals.append(mon_normal)
        
        stacking_normal = _average_aligned_normals(layer_normals)
        
        logger.info(f"\nSTACKING GEOMETRY:")
        logger.info(f"  Averaged stacking normal: {stacking_normal}")
        
        for i in range(len(monomers) - 1):
            pi_core_A = monomers[i][pi_idx]
            pi_core_B = monomers[i+1][pi_idx]
            
            full_coords_A = monomers[i]
            full_coords_B = monomers[i+1]
            
            centroid_A_pi = pi_core_A.mean(axis=0)
            centroid_B_pi = pi_core_B.mean(axis=0)
            
            delta = centroid_B_pi - centroid_A_pi
            
            normal = stacking_normal.copy()
            if np.dot(delta, normal) < 0:
                normal = -normal
            
            stack_dist = np.dot(delta, normal)
            slip_vec = delta - stack_dist * normal
            slip_dist = np.linalg.norm(slip_vec)
            
            slip_angle = _slip_angle_deg(pi_core_A, slip_vec) if slip_dist > 1e-6 else np.nan
            if not np.isnan(slip_angle):
                all_slip_angles.append(slip_angle)
            
            has_clash, clash_details = check_interlayer_catastrophic_overlap(
                full_coords_A, full_coords_B
            )
            min_dist = clash_details["min_distance"]
            
            # IMPORTANT: use the SAME origin for both layers so that in-plane
            # translation (slip) is preserved in the 2D projections.
            # Using separate centroids would artificially inflate overlap.
            proj_origin = centroid_A_pi
            pi_core_A_2d = _project_to_pi_plane(pi_core_A, normal, origin=proj_origin)
            pi_core_B_2d = _project_to_pi_plane(pi_core_B, normal, origin=proj_origin)
            
            _, coverage = _pi_cloud_overlap_metrics(
                pi_core_A_2d,
                pi_core_B_2d,
                monomer_types[i],
                monomer_types[i + 1],
                pi_idx,
                resolution=0.20,
            )
            
            results.append({
                "pair": f"mol{i+1}–mol{i+2}",
                # Keep "overlap_percent" as the forgiving coverage score so the
                # rest of the pipeline remains compatible.
                "overlap_percent": round(coverage, 1),
                "overlap_method": "π-cloud discs (85% vdW): coverage",
                "has_steric_clash": has_clash,
                "min_distance_ang": round(min_dist, 3),
                "n_very_close": clash_details["n_very_close"],
                "n_close": clash_details["n_close"],
                "slip_distance_ang": round(slip_dist, 3),
                "slip_angle_deg": round(slip_angle, 1) if not np.isnan(slip_angle) else None,
                "stacking_distance_ang": round(stack_dist, 3),
            })
        
        all_pair_distances = []
        non_adjacent_clashes = []
        
        for i in range(len(monomers)):
            for j in range(i+1, len(monomers)):
                has_clash_ij, clash_details_ij = check_interlayer_catastrophic_overlap(
                    monomers[i], monomers[j]
                )
                min_dist_ij = clash_details_ij["min_distance"]
                all_pair_distances.append(min_dist_ij)
                
                if j > i + 1 and has_clash_ij:
                    non_adjacent_clashes.append(f"mol{i+1}–mol{j+1}")
        
        min_interlayer = min(all_pair_distances) if all_pair_distances else float('inf')
        
        # Coverage is the primary overlap score used for classification.
        mean_cov = np.mean([r["overlap_percent"] for r in results])
        mean_slip = np.mean([r["slip_distance_ang"] for r in results])
        mean_stack_dist = np.mean([r["stacking_distance_ang"] for r in results])
        mean_slip_angle = np.mean(all_slip_angles) if all_slip_angles else np.nan
        
        overall_has_clash = any(r["has_steric_clash"] for r in results) or bool(non_adjacent_clashes)
        
        '''
        if non_adjacent_clashes:
            warning_msg = f"❌ Non-adjacent clashes: {', '.join(non_adjacent_clashes)}"
            overall_quality["warnings"].insert(0, warning_msg)
        '''

        intramol_warnings = []
        if adjacency is not None:
            for i, (coords, types) in enumerate(zip(monomers, monomer_types)):
                has_intramol_clash, details = check_intramolecular_catastrophic_overlap(
                    coords, types, adjacency
                )
                if has_intramol_clash:
                    min_intramol = details["min_distance"]
                    intramol_warnings.append(
                        f"⚠️ Intramolecular clash in mol{i+1} (min: {min_intramol:.2f}Å)"
                    )
                    
        return {
            "pairs": results,
            "mean_overlap_percent": round(mean_cov, 1),
            "mean_slip_distance_ang": round(mean_slip, 3),
            "mean_stacking_distance_ang": round(mean_stack_dist, 3),
            "mean_slip_angle_deg": round(mean_slip_angle, 1) if not np.isnan(mean_slip_angle) else None,
            "min_interlayer_distance_ang": round(min_interlayer, 3),
            "n_pi_core_atoms": len(pi_idx),
            "pi_core_area_ang2": round(core_area, 1),
            "pi_detection_method": pi_method,
            #"overall_quality": overall_quality["quality"],
            #"overall_quality_score": overall_quality.get("score", 0.0),
            #"overall_warnings": overall_quality["warnings"],
            #"overall_recommendations": overall_quality["recommendations"],
            #"stack_needs_attention": overall_quality["needs_attention"],
            "has_steric_clash": overall_has_clash,
            "non_adjacent_clashes": non_adjacent_clashes,
            "intramol_warnings": intramol_warnings,
            "n_layers": len(monomers),
        }
    
    except Exception as e:
        logger.error(f"π-stack analysis failed: {e}", exc_info=True)
        return {"error": str(e)}
