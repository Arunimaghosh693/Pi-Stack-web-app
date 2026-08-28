import numpy as np
import json
from typing import List, Tuple, Set, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdDetermineBonds

def read_xyz(xyz_file: str) -> Tuple[List[str], np.ndarray]:
    with open(xyz_file) as f:
        lines = f.readlines()
    natoms = int(lines[0])
    atoms = []
    coords = []
    for line in lines[2:2+natoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords)

def calculate_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate dihedral angle between four points p0-p1-p2-p3"""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so that it does not influence magnitude of vector rejections
    b1 /= np.linalg.norm(b1)

    # Vector rejections
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.degrees(np.arctan2(y, x))
    return angle

def xyz_to_mol(atoms: List[str], coords: np.ndarray) -> Chem.Mol:
    """Create an RDKit molecule from atoms and coordinates (distance-based bonding)."""
    mol = Chem.RWMol()
    atom_indices = []
    for atom in atoms:
        a = Chem.Atom(atom)
        idx = mol.AddAtom(a)
        atom_indices.append(idx)
    
    # Add a conformer with the 3D coordinates
    conf = Chem.Conformer(len(atoms))
    for i in range(len(atoms)):
        conf.SetAtomPosition(i, coords[i].tolist())
    mol.AddConformer(conf)

    # Add bonds based on interatomic distances (covalent radii + tolerance)
    cov_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
        # Add more as needed
    }
    n = len(atoms)
    for i in range(n):
        for j in range(i+1, n):
            r1 = cov_radii.get(atoms[i], 0.7)
            r2 = cov_radii.get(atoms[j], 0.7)
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= (r1 + r2 + 0.4):  # tolerance
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    mol = mol.GetMol()
    
    # Determine bond orders from 3D coordinates
    try:
        # Let RDKit determine the charge automatically
        rdDetermineBonds.DetermineBonds(mol)
    except Exception as e:
        print(f"Warning: Could not determine bond orders: {e}")
    
    return mol

def find_dihedrals_from_xyz(xyz_file: str) -> List[Tuple[Tuple[int, int, int, int], float]]:
    atoms, coords = read_xyz(xyz_file)
    mol = xyz_to_mol(atoms, coords)
    # Try sanitizing the molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        # If sanitization fails, continue but warn the user
        print(f"Warning: Chem.SanitizeMol failed: {e}")

    # Try SMARTS-based rotatable bond detection
    dihedrals = []
    try:
        rot_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        matches = mol.GetSubstructMatches(rot_smarts)
        if matches:
            rotatable_pairs = [(m[0], m[1]) for m in matches]
        else:
            rotatable_pairs = []
    except Exception:
        rotatable_pairs = []

    # Fallback: inspect bonds heuristically
    if not rotatable_pairs:
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.IsInRing():
                continue
            a2 = bond.GetBeginAtomIdx()
            a3 = bond.GetEndAtomIdx()
            atom2 = mol.GetAtomWithIdx(a2)
            atom3 = mol.GetAtomWithIdx(a3)
            neighbors1 = [nbr.GetIdx() for nbr in atom2.GetNeighbors() if nbr.GetIdx() != a3]
            neighbors2 = [nbr.GetIdx() for nbr in atom3.GetNeighbors() if nbr.GetIdx() != a2]
            if not neighbors1 or not neighbors2:
                continue
            rotatable_pairs.append((a2, a3))

    # Enumerate neighbor combinations and compute dihedrals
    seen = set()
    for a2, a3 in rotatable_pairs:
        atom2 = mol.GetAtomWithIdx(a2)
        atom3 = mol.GetAtomWithIdx(a3)
        neighbors1 = [nbr.GetIdx() for nbr in atom2.GetNeighbors() if nbr.GetIdx() != a3]
        neighbors2 = [nbr.GetIdx() for nbr in atom3.GetNeighbors() if nbr.GetIdx() != a2]
        if not neighbors1 or not neighbors2:
            continue

        for a1 in neighbors1:
            for a4 in neighbors2:
                tpl = (a1, a2, a3, a4)
                if tpl in seen:
                    continue
                seen.add(tpl)
                try:
                    angle = calculate_dihedral(coords[a1], coords[a2], coords[a3], coords[a4])
                    dihedrals.append((tpl, angle))
                except Exception:
                    # skip problematic tuples
                    pass

    return dihedrals


def compute_pi_core_set(mol: Chem.Mol, coords: np.ndarray) -> Tuple[Set[int], List[Set[int]]]:
    """Detect pi-core atoms by selecting planar heavy-atom rings (size 5 or 6).
    Returns: (all_pi_atoms, list_of_individual_ring_sets)
    """
    pi_set = set()
    pi_rings = []
    # rings: require ring size 5 or 6, heavy atoms, and approximate planarity
    ring_info = mol.GetRingInfo().AtomRings()
    for ring in ring_info:
        if len(ring) not in (5, 6):
            continue
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if any(s == 'H' for s in syms):
            continue
        # compute planarity: fit plane and check max distance
        pts = coords[list(ring)]
        centroid = pts.mean(axis=0)
        u, s, vh = np.linalg.svd(pts - centroid)
        normal = vh[-1]
        dists = np.abs(np.dot(pts - centroid, normal))
        if np.max(dists) < 0.12:  # threshold for planarity
            ring_set = set(ring)
            pi_rings.append(ring_set)
            for i in ring:
                pi_set.add(i)

    return pi_set, pi_rings


def filter_pi_core_boundary_dihedrals(mol: Chem.Mol, dihedrals: List[Tuple[Tuple[int, int, int, int], float]], coords: Optional[np.ndarray] = None, pi_core_set: Optional[Set[int]] = None, pi_rings: Optional[List[Set[int]]] = None, heavy_only: bool = True, core_first: bool = True) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Keep dihedrals at the boundary between pi-core and non-pi-core regions, or bridging different pi-cores.
    If pi_core_set is None, it is computed from coords.
    """
    if pi_core_set is None:
        if coords is None:
            raise ValueError('coords required to compute pi_core_set')
        pi_core_set, pi_rings = compute_pi_core_set(mol, coords)

    out = []
    for (a1, a2, a3, a4), angle in dihedrals:
        if heavy_only:
            syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in (a1, a2, a3, a4)]
            if any(s == 'H' for s in syms):
                continue

        left_core = (a1 in pi_core_set) and (a2 in pi_core_set)
        right_core = (a3 in pi_core_set) and (a4 in pi_core_set)

        # Check if this is an inter-pi-core dihedral (bridging two different pi-cores)
        inter_pi_core = False
        if left_core and right_core and pi_rings:
            # Find which ring(s) the left atoms belong to
            left_rings = [r for r in pi_rings if a1 in r or a2 in r]
            right_rings = [r for r in pi_rings if a3 in r or a4 in r]
            
            # Check if left and right belong to different rings
            if left_rings and right_rings:
                # Check if they're in completely different rings (no overlap)
                for lr in left_rings:
                    for rr in right_rings:
                        if lr != rr and not (lr & rr):  # different rings with no shared atoms
                            inter_pi_core = True
                            break
                    if inter_pi_core:
                        break

        # Keep dihedral when:
        # 1. core is on left and outside on right
        # 2. core is on the right and outside on left
        # 3. bridging between two different pi-cores
        if left_core and not right_core:
            tpl = (a1, a2, a3, a4)
        elif right_core and not left_core:
            tpl = (a1, a2, a3, a4)
        elif inter_pi_core:
            tpl = (a1, a2, a3, a4)
        else:
            continue

        # If user requested core-first and the core is on the right, attempt to rotate tuple so core atoms are first
        norm_tpl = None
        core_on_right = right_core and not left_core
        if core_first and core_on_right:
            # prefer ordering (a3,a4,a1,a2) so core atoms (a3,a4) are first
            desired = (a3, a4, a1, a2)
            b_central = mol.GetBondBetweenAtoms(int(desired[1]), int(desired[2]))
            if b_central is not None:
                is_outer0_neigh = any(n.GetIdx() == int(desired[0]) for n in mol.GetAtomWithIdx(int(desired[1])).GetNeighbors())
                is_outer3_neigh = any(n.GetIdx() == int(desired[3]) for n in mol.GetAtomWithIdx(int(desired[2])).GetNeighbors())
                if is_outer0_neigh and is_outer3_neigh:
                    norm_tpl = desired
            # fallback: try reverse (a4,a3,a2,a1) if desired didn't work
            if norm_tpl is None:
                rev = (a4, a3, a2, a1)
                b_central = mol.GetBondBetweenAtoms(int(rev[1]), int(rev[2]))
                if b_central is not None:
                    is_outer0_neigh = any(n.GetIdx() == int(rev[0]) for n in mol.GetAtomWithIdx(int(rev[1])).GetNeighbors())
                    is_outer3_neigh = any(n.GetIdx() == int(rev[3]) for n in mol.GetAtomWithIdx(int(rev[2])).GetNeighbors())
                    if is_outer0_neigh and is_outer3_neigh:
                        norm_tpl = rev

        # If we didn't choose a core-first ordering, normalize the original tuple
        if norm_tpl is None:
            norm_tpl = _normalize_dihedral_tuple(mol, tpl)

            # Recompute angle with coords if available
        if coords is not None:
            try:
                new_angle = calculate_dihedral(coords[norm_tpl[0]], coords[norm_tpl[1]], coords[norm_tpl[2]], coords[norm_tpl[3]])
            except Exception:
                new_angle = angle
        else:
            new_angle = angle

        out.append((norm_tpl, new_angle))

    return out


def collapse_one_per_bond(dihedrals: List[Tuple[Tuple[int, int, int, int], float]]) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Keep only the first dihedral for each central bond (unordered a2,a3)."""
    seen = set()
    out = []
    for (a1, a2, a3, a4), angle in dihedrals:
        key = tuple(sorted((a2, a3)))
        if key in seen:
            continue
        seen.add(key)
        out.append(((a1, a2, a3, a4), angle))
    return out


def pick_best_per_bond(dihedrals: List[Tuple[Tuple[int, int, int, int], float]], mol: Chem.Mol) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Select one dihedral per central bond using a simple heavy-atom + angle score."""
    groups = {}
    def is_heavy(idx):
        return mol.GetAtomWithIdx(idx).GetSymbol() != 'H'

    for (a1, a2, a3, a4), angle in dihedrals:
        key = tuple(sorted((a2, a3)))
        score = (1 if is_heavy(a1) else 0) + (1 if is_heavy(a4) else 0)
        # make score dominant and use abs(angle) as secondary
        score = score * 1000 + abs(angle)
        if key not in groups or score > groups[key][0]:
            groups[key] = (score, ((a1, a2, a3, a4), angle))

    result = [v[1] for v in groups.values()]
    # sort for deterministic output by central bond
    result.sort(key=lambda x: tuple(sorted((x[0][1], x[0][2]))))
    return result


def write_torsions_json(torsions_list: List[Tuple[Tuple[int, int, int, int], float]], path: str = 'torsions.json', one_based: bool = False) -> None:
    """Write torsions_list to JSON. If one_based, indices are incremented by 1."""
    out = {
        "indexing": "1-based" if one_based else "0-based",
        "description": "Auto-generated torsions from calc_dihedrals.py",
        "torsions": []
    }
    for i, (atoms_tuple, angle) in enumerate(torsions_list, start=1):
        atoms_list = [int(x) + (1 if one_based else 0) for x in atoms_tuple]
        out["torsions"].append({
            "name": f"torsion_{i}",
            "atoms": atoms_list,
            "rotate_side": "d",
            "description": "Dihedral defined by atoms [a,b,c,d]; rotation axis is b-c."
        })
    
    # Write JSON with compact atom arrays
    with open(path, 'w') as f:
        f.write('{\n')
        f.write(f'  "indexing": "{out["indexing"]}",\n')
        f.write(f'  "description": "{out["description"]}",\n')
        f.write('  "torsions": [\n')
        for i, t in enumerate(out["torsions"]):
            atoms_str = ', '.join(str(a) for a in t["atoms"])
            f.write('    {\n')
            f.write(f'      "name": "{t["name"]}",\n')
            f.write(f'      "atoms": [ {atoms_str} ],\n')
            f.write(f'      "rotate_side": "{t["rotate_side"]}",\n')
            f.write(f'      "description": "{t["description"]}"\n')
            f.write('    }' + (',' if i < len(out["torsions"]) - 1 else '') + '\n')
        f.write('  ]\n')
        f.write('}\n')


def _format_atoms_for_print(atoms_tuple: Tuple[int, int, int, int], one_based: bool) -> str:
    """Return a string representing atoms_tuple using one_based if requested."""
    if one_based:
        disp = tuple(x + 1 for x in atoms_tuple)
    else:
        disp = tuple(atoms_tuple)
    return str(disp)


def _normalize_dihedral_tuple(mol: Chem.Mol, tpl: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Rotate the tuple cyclically so that the central pair (index 1,2) are bonded in the molecule.
    If no rotation yields a bonded central pair, returns the original tuple.
    """
    a = list(tpl)
    for _ in range(4):
        a1, a2, a3, a4 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        # central bond must exist
        if mol.GetBondBetweenAtoms(a2, a3) is not None:
            # ensure outer atoms are neighbors of the central atoms (preserve connectivity)
            is_a1_neigh = any(n.GetIdx() == a1 for n in mol.GetAtomWithIdx(a2).GetNeighbors())
            is_a4_neigh = any(n.GetIdx() == a4 for n in mol.GetAtomWithIdx(a3).GetNeighbors())
            if is_a1_neigh and is_a4_neigh:
                return (a1, a2, a3, a4)
        # rotate left
        a = [a[1], a[2], a[3], a[0]]
    # fallback: return first rotation where central bond exists
    a = list(tpl)
    for _ in range(4):
        a2, a3 = int(a[1]), int(a[2])
        if mol.GetBondBetweenAtoms(a2, a3) is not None:
            return (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
        a = [a[1], a[2], a[3], a[0]]
    return tpl


def draw_molecule_with_dihedrals(mol: Chem.Mol, torsions_list: List[Tuple[Tuple[int, int, int, int], float]], output_path: str = 'dihedrals_2d.png', one_based: bool = False, img_size: Tuple[int, int] = (800, 600)) -> None:
    """Draw a 2D molecule with highlighted dihedrals and labels.
    
    Args:
        mol: RDKit molecule
        torsions_list: List of (atoms_tuple, angle) for each dihedral
        output_path: Path to save the image
        one_based: If True, display atom indices as 1-based
        img_size: Image size (width, height)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Circle
        from rdkit.Chem import rdDepictor
        
        # Create a copy for 2D drawing and compute 2D coordinates
        mol_2d = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_2d)
        
        # Get 2D coordinates
        conf = mol_2d.GetConformer()
        atom_positions = {}
        for i in range(mol_2d.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_positions[i] = (pos.x, pos.y)
        
        # Prepare highlighting - only highlight atoms, not bonds
        highlight_atoms = []
        atom_colors_dict = {}
        
        # Color palette for different dihedrals (RGB tuples)
        color_palette = [
            (1.0, 0.42, 0.42),  # Red
            (0.31, 0.80, 0.77),  # Turquoise
            (0.27, 0.72, 0.82),  # Blue
            (1.0, 0.63, 0.48),   # Orange
            (0.60, 0.85, 0.78),  # Mint
            (0.97, 0.86, 0.44),  # Yellow
            (0.73, 0.56, 0.81),  # Purple
            (0.52, 0.76, 0.89),  # Light blue
        ]
        
        # Store dihedral info for later annotation
        dihedral_annotations = []
        
        for i, (atoms_tuple, angle) in enumerate(torsions_list):
            color = color_palette[i % len(color_palette)]
            
            # Add atoms to highlight list and color mapping
            for atom_idx in atoms_tuple:
                if atom_idx not in highlight_atoms:
                    highlight_atoms.append(atom_idx)
                atom_colors_dict[atom_idx] = color
            
            # Store info for drawing curved arrows on the central bond
            dihedral_annotations.append((atoms_tuple, color, i+1))
        
        # Draw molecule with atom highlighting only
        drawer = Draw.MolDraw2DCairo(img_size[0], img_size[1])
        drawer.drawOptions().addAtomIndices = True  # Enable atom indices
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().bondLineWidth = 2
        
        # Set atom label offset to show indices
        if one_based:
            # Create a copy with adjusted atom indices for display
            for atom_idx in range(mol_2d.GetNumAtoms()):
                atom = mol_2d.GetAtomWithIdx(atom_idx)
                atom.SetProp('atomLabel', str(atom_idx + 1))
        
        drawer.DrawMolecule(mol_2d, 
                           highlightAtoms=highlight_atoms, 
                           highlightAtomColors=atom_colors_dict)
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        
        # Save the base image
        with open('_temp_mol.png', 'wb') as f:
            f.write(img_data)
        
        # Load image and add annotations with matplotlib
        img = plt.imread('_temp_mol.png')
        fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
        ax.imshow(img)
        ax.axis('off')
        
        # Get image dimensions for coordinate transformation
        img_height, img_width = img.shape[:2]
        
        # Calculate scale factor from molecule coordinates to image coordinates
        if atom_positions:
            mol_xs = [pos[0] for pos in atom_positions.values()]
            mol_ys = [pos[1] for pos in atom_positions.values()]
            mol_width = max(mol_xs) - min(mol_xs)
            mol_height = max(mol_ys) - min(mol_ys)
            mol_center_x = (max(mol_xs) + min(mol_xs)) / 2
            mol_center_y = (max(mol_ys) + min(mol_ys)) / 2
            
            # Estimate scale (RDKit typically uses margins)
            scale_x = img_width * 0.8 / mol_width if mol_width > 0 else 1
            scale_y = img_height * 0.8 / mol_height if mol_height > 0 else 1
            scale = min(scale_x, scale_y)
            
            # Draw labels near the central bonds
            for atoms_tuple, color, label_num in dihedral_annotations:
                a1, a2, a3, a4 = atoms_tuple
                # Get positions of central bond atoms (a2, a3)
                pos2 = atom_positions[a2]
                pos3 = atom_positions[a3]
                
                # Transform to image coordinates
                x2 = (pos2[0] - mol_center_x) * scale + img_width / 2
                y2 = -(pos2[1] - mol_center_y) * scale + img_height / 2  # Flip Y
                x3 = (pos3[0] - mol_center_x) * scale + img_width / 2
                y3 = -(pos3[1] - mol_center_y) * scale + img_height / 2  # Flip Y
                
                # Draw a small label near the bond midpoint
                mid_x = (x2 + x3) / 2
                mid_y = (y2 + y3) / 2
                
                # Offset label perpendicular to bond
                dx = x3 - x2
                dy = y3 - y2
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    perp_x = -dy / length * 15  # Perpendicular offset
                    perp_y = dx / length * 15
                else:
                    perp_x, perp_y = 0, -15
                
                label_x = mid_x + perp_x
                label_y = mid_y + perp_y
                
                # Draw a small circle with label number
                circle = Circle((label_x, label_y), 12, color=color, ec='black', linewidth=2, zorder=10)
                ax.add_patch(circle)
                ax.text(label_x, label_y, f'D{label_num}', fontsize=9, weight='bold',
                       ha='center', va='center', color='white', zorder=11)
        
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clean up temp file
        import os
        if os.path.exists('_temp_mol.png'):
            os.remove('_temp_mol.png')
        
        return True
    except ImportError as e:
        print(f"Warning: Could not generate 2D figure. Missing dependency: {e}")
        print("Install matplotlib to enable 2D visualization: pip install matplotlib")
        return False
    except Exception as e:
        print(f"Warning: Could not generate 2D figure: {e}")
        import traceback
        traceback.print_exc()
        return False

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Calculate dihedrals from XYZ or SMILES')
    parser.add_argument('--input', '-i', required=False, default='input.xyz', help='Path to XYZ file or SMILES string (default: input.xyz)')
    parser.add_argument('--format', '-f', choices=['xyz', 'smiles'], default=None, help='Input format (default: auto-detect from file presence)')
    parser.add_argument('--output', '-o', default='torsions.json', help='Output JSON path (default: torsions.json)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable detailed diagnostic output (default: False)')
    parser.add_argument('--one-based', action='store_true', help='Write atom indices as 1-based in JSON (default: 0-based)')
    parser.add_argument('--opt-method', choices=['uff', 'mmff', 'both', 'none'], default='both', help='Optimization method for SMILES (default: both - UFF with MMFF fallback)')
    parser.add_argument('--draw', action='store_true', help='Generate 2D molecule image with highlighted dihedrals (default: False)')
    parser.add_argument('--draw-output', default='dihedrals_2d.png', help='Output path for 2D image (default: dihedrals_2d.png)')
    parser.add_argument('--core-first', dest='core_first', action='store_true', help='Force output dihedral tuples to list pi-core atoms first when possible (default: True)')
    parser.add_argument('--no-core-first', dest='core_first', action='store_false', help='Do not force core-first ordering')
    parser.set_defaults(core_first=True)
    args = parser.parse_args()

    fmt = args.format
    if fmt is None:
        import os
        fmt = 'xyz' if os.path.isfile(args.input) else 'smiles'

    if fmt == 'xyz':
        xyz_file = args.input
        dihedral_info = find_dihedrals_from_xyz(xyz_file)
        atoms, coords = read_xyz(xyz_file)
        mol = xyz_to_mol(atoms, coords)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    else:
        # SMILES path
        smiles = args.input
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Invalid SMILES')
        mol = Chem.AddHs(mol)
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
            if res != 0 and args.verbose:
                print(f"Warning: EmbedMolecule returned {res}")
        except Exception as e:
            if args.verbose:
                print(f"Warning: embedding SMILES failed: {e}")

        # Check if conformer exists before proceeding
        if mol.GetNumConformers() == 0:
            print("Error: Failed to generate 3D coordinates for this molecule.")
            print("This can happen with very large or complex molecules.")
            print("Try using a smaller/simpler molecule or provide an XYZ file instead.")
            return
        
        # Optimize based on user selection
        if args.opt_method in ('uff', 'both'):
            uff_succeeded = False
            try:
                uff_res = AllChem.UFFOptimizeMolecule(mol)
                if uff_res == 0:
                    uff_succeeded = True
                elif args.verbose:
                    print(f"Warning: UFFOptimizeMolecule returned {uff_res}")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: UFF optimization failed: {e}")
        else:
            uff_succeeded = True  # Skip UFF, treat as "succeeded" to prevent MMFF fallback

        if args.opt_method in ('mmff', 'both') and not uff_succeeded:
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    mmff_res = AllChem.MMFFOptimizeMolecule(mol)
                    if mmff_res == 0 and args.verbose:
                        print("MMFF optimization succeeded as fallback")
                    elif args.verbose:
                        print(f"Warning: MMFFOptimizeMolecule returned {mmff_res}")
                elif args.verbose:
                    print("MMFF parameters not available for this molecule; skipping MMFF fallback")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: MMFF optimization failed: {e}")
        elif args.opt_method == 'mmff':
            # User requested MMFF only (not as fallback)
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    mmff_res = AllChem.MMFFOptimizeMolecule(mol)
                    if args.verbose and mmff_res == 0:
                        print("MMFF optimization succeeded")
                    elif args.verbose:
                        print(f"Warning: MMFFOptimizeMolecule returned {mmff_res}")
                elif args.verbose:
                    print("MMFF parameters not available for this molecule")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: MMFF optimization failed: {e}")

        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

        # If input was SMILES, write an XYZ file of the embedded coordinates for reference
        import os
        smiles_xyz_path = 'output.xyz'
        try:
            with open(smiles_xyz_path, 'w') as xf:
                xf.write(f"{len(atoms)}\n")
                xf.write(f"Embedded coordinates from SMILES input\n")
                for i, sym in enumerate(atoms):
                    x, y, z = coords[i]
                    xf.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")
            wrote_smiles_xyz = True
        except Exception:
            wrote_smiles_xyz = False

        # build dihedrals from embedded coordinates
        dihedral_info = []
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        try:
            rot_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
            matches = mol.GetSubstructMatches(rot_smarts)
            rotatable_pairs = [(m[0], m[1]) for m in matches] if matches else []
        except Exception:
            rotatable_pairs = []

        if not rotatable_pairs:
            for bond in mol.GetBonds():
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                if bond.IsInRing():
                    continue
                a2 = bond.GetBeginAtomIdx(); a3 = bond.GetEndAtomIdx()
                neighbors1 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(a2).GetNeighbors() if nbr.GetIdx() != a3]
                neighbors2 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(a3).GetNeighbors() if nbr.GetIdx() != a2]
                if not neighbors1 or not neighbors2:
                    continue
                rotatable_pairs.append((a2, a3))

        seen = set()
        for a2, a3 in rotatable_pairs:
            neighbors1 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(a2).GetNeighbors() if nbr.GetIdx() != a3]
            neighbors2 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(a3).GetNeighbors() if nbr.GetIdx() != a2]
            for a1 in neighbors1:
                for a4 in neighbors2:
                    tpl = (a1, a2, a3, a4)
                    if tpl in seen:
                        continue
                    seen.add(tpl)
                    try:
                        angle = calculate_dihedral(coords[a1], coords[a2], coords[a3], coords[a4])
                        dihedral_info.append((tpl, angle))
                    except Exception:
                        pass

    # downstream filtering and selection
    pi_core, pi_rings = compute_pi_core_set(mol, coords)
    boundary = filter_pi_core_boundary_dihedrals(mol, dihedral_info, coords=coords, pi_core_set=pi_core, pi_rings=pi_rings, heavy_only=True, core_first=args.core_first)
    boundary_collapsed = collapse_one_per_bond(boundary)
    best = pick_best_per_bond(boundary_collapsed, mol)

    if args.verbose:
        print("All dihedrals (including H):")
        for atoms_tuple, angle in dihedral_info:
            label = '1-based' if args.one_based else '0-based'
            print(f"Dihedral atoms ({label}): {_format_atoms_for_print(atoms_tuple, args.one_based)}, Angle: {angle:.2f} degrees")

        print('\nPi-core atom indices:', sorted(list(pi_core)))
        print('\nPi-core boundary dihedrals [one per rotatable bond]:')
        for atoms_tuple, angle in boundary_collapsed:
            label = '1-based' if args.one_based else '0-based'
            print(f"Dihedral atoms ({label}): {_format_atoms_for_print(atoms_tuple, args.one_based)}, Angle: {angle:.2f} degrees")

        print('\nSelected best-per-bond torsions:')
        for atoms_tuple, angle in best:
            label = '1-based' if args.one_based else '0-based'
            print(f"Dihedral atoms ({label}): {_format_atoms_for_print(atoms_tuple, args.one_based)}, Angle: {angle:.2f} degrees")
    else:
        print(f"Selected {len(best)} torsions; writing to {args.output}")
        for atoms_tuple, angle in best:
            print(f"  {_format_atoms_for_print(atoms_tuple, args.one_based)}  {angle:.2f}°")

    write_torsions_json(best, path=args.output, one_based=args.one_based)
    
    # Generate 2D figure if requested
    if args.draw:
        success = draw_molecule_with_dihedrals(mol, best, output_path=args.draw_output, one_based=args.one_based)
        if success:
            print(f'Generated 2D figure: {args.draw_output}')
    
    # If we created an XYZ for SMILES input, print the filename for user's convenience
    if fmt != 'xyz' and 'smiles_xyz_path' in locals() and wrote_smiles_xyz:
        print('Wrote embedded coordinates to', smiles_xyz_path)
    if args.verbose:
        print(f'\nWrote {args.output}')
    else:
        print('Wrote', args.output)


if __name__ == "__main__":
    main()

