#!/usr/bin/env python3
"""
Build molecular stack from optimization results file.

Reads parameters from optimization_results.txt and builds a stack with 
the desired number of molecules using the optimized geometry and torsions.
"""

import argparse
import json
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional
import re

# Add pi-stack-optimizer modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'pi-stack-optimizer'))

# Import functions from modules
from modules.xyz_io import read_xyz_file, write_xyz_file
from modules.geometry import align_pi_core_to_xy, transformation_matrix, apply_transform, build_bond_graph
from modules.torsion import rotate_fragment_around_bond, find_fragment_indices, load_torsions_file
from modules.stacking import build_n_layer_stack
from modules.system_utils import set_thread_env_vars

def parse_optimization_results(filename: str) -> Dict:
    """Parse optimization_results.txt file to extract parameters."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Optimization results file not found: {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Parse best objective
    obj_match = re.search(r"Best objective:\s*([-\d\.]+)\s*kJ/mol", content)
    if obj_match:
        results['objective'] = float(obj_match.group(1))
    
    # Parse transform parameters
    param_match = re.search(r"θ\s*=\s*([-\d\.]+)°.*Tx=([-\d\.]+)\s*Å.*Ty=([-\d\.]+)\s*Å.*Tz=([-\d\.]+)\s*Å.*Cx=([-\d\.]+)\s*Å.*Cy=([-\d\.]+)\s*Å", content)
    if param_match:
        results['theta'] = float(param_match.group(1))
        results['Tx'] = float(param_match.group(2))
        results['Ty'] = float(param_match.group(3))
        results['Tz'] = float(param_match.group(4))
        results['Cx'] = float(param_match.group(5))
        results['Cy'] = float(param_match.group(6))
    
    # Parse torsions
    torsion_match = re.search(r"torsions:\s*\[(.*?)\]", content)
    if torsion_match:
        torsion_str = torsion_match.group(1)
        if torsion_str.strip():
            # Extract numbers from "X.XXX°" format and convert to radians
            torsion_values = []
            for match in re.finditer(r"([-\d\.]+)°", torsion_str):
                degrees = float(match.group(1))
                radians = np.radians(degrees)
                torsion_values.append(radians)
            results['torsions'] = torsion_values
            results['torsions_degrees'] = [float(match.group(1)) for match in re.finditer(r"([-\d\.]+)°", torsion_str)]
        else:
            results['torsions'] = []
            results['torsions_degrees'] = []
    else:
        results['torsions'] = []
        results['torsions_degrees'] = []
    
    return results

def build_transform_vector(results: Dict) -> np.ndarray:
    """Build transformation parameter vector from parsed results."""
    # Convert theta to cos_like, sin_like representation
    theta_rad = np.radians(results['theta'])
    cos_like = np.cos(theta_rad)
    sin_like = np.sin(theta_rad)
    
    transform_vec = np.array([
        cos_like, sin_like,
        results['Tx'], results['Ty'], results['Tz'],
        results['Cx'], results['Cy']
    ], dtype=float)
    
    return transform_vec

def apply_torsions_to_monomer(monomer_coords: np.ndarray, atom_types: List[str], 
                            torsions: List[float], torsions_spec: List[Dict]) -> np.ndarray:
    """Apply torsions to monomer coordinates."""
    if not torsions:
        return monomer_coords.copy()
    
    coords = monomer_coords.copy()
    torsions_array = np.array(torsions)
    torsions_array = np.clip(torsions_array, -np.pi, np.pi)
    
    for k, tau in enumerate(torsions_array):
        if k < len(torsions_spec):
            spec = torsions_spec[k]
            coords = rotate_fragment_around_bond(
                coords, spec['b'], spec['c'], float(tau), spec['fragment']
            )
    
    # Realign if torsions were applied
    if len(torsions) > 0:
        coords = align_pi_core_to_xy(atom_types, coords)
    
    return coords

def main():
    parser = argparse.ArgumentParser(
        description="Build molecular stack from optimization results"
    )
    parser.add_argument("monomer_xyz", help="Original monomer XYZ file")
    parser.add_argument("--results-file", default="optimization_results.txt",
                       help="Optimization results file (default: optimization_results.txt)")
    parser.add_argument("--n-molecules", type=int, required=True,
                       help="Number of molecules in the stack")
    parser.add_argument("--output", default=None,
                       help="Output XYZ file (default: stack_N_molecules.xyz)")
    parser.add_argument("--torsions-file", default=None,
                       help="Torsions JSON file (only needed if torsions were used in optimization)")
    parser.add_argument("--save-monomer", action="store_true",
                       help="Also save the torsion-modified monomer")
    
    args = parser.parse_args()
    
    # Set output filename if not provided
    if args.output is None:
        args.output = f"stack_{args.n_molecules}_molecules.xyz"
    
    print(f"Reading optimization results from: {args.results_file}")
    
    # Parse optimization results
    try:
        results = parse_optimization_results(args.results_file)
        print(f"Best objective: {results['objective']:.6f} kJ/mol")
        print(f"Transform: θ={results['theta']:.3f}°, Tx={results['Tx']:.4f}, Ty={results['Ty']:.4f}, Tz={results['Tz']:.4f}")
        print(f"Center offset: Cx={results['Cx']:.4f}, Cy={results['Cy']:.4f}")
        if results['torsions']:
            print(f"Torsions: {[f'{t:.3f}' for t in results['torsions_degrees']]}° (converted to radians for calculations)")
        else:
            print("No torsions applied")
    except Exception as e:
        print(f"Error parsing results file: {e}")
        return 1
    
    # Read monomer
    print(f"Reading monomer from: {args.monomer_xyz}")
    monomer_coords, atom_types = read_xyz_file(args.monomer_xyz)
    
    # Center and align monomer
    monomer_center = monomer_coords.mean(axis=0)
    monomer_coords_centered = monomer_coords - monomer_center
    monomer_coords_centered = align_pi_core_to_xy(atom_types, monomer_coords_centered)
    
    # Load torsions if needed
    torsions_spec = []
    if results['torsions']:  # Only if torsions were used in optimization
        if not args.torsions_file:
            print("Error: Optimization used torsions but no --torsions-file provided.")
            print("The torsions file is required to rebuild this optimized geometry.")
            print("Please specify --torsions-file <path_to_torsions.json>")
            return 1
            
        if not os.path.exists(args.torsions_file):
            print(f"Error: Torsions file not found: {args.torsions_file}")
            print("Torsions are required to rebuild this optimized geometry.")
            return 1
            
        print(f"Loading torsions from: {args.torsions_file}")
        raw_torsions = load_torsions_file(args.torsions_file)
        indexing = raw_torsions.get("indexing", "0-based")
        adj = build_bond_graph(atom_types, monomer_coords_centered)
        
        for t in raw_torsions.get("torsions", []):
            atoms = [int(x) for x in t["atoms"]]
            if indexing == "1-based":
                atoms = [a - 1 for a in atoms]
            
            b, c = atoms[1], atoms[2]
            side_atom = atoms[3] if t.get("rotate_side", "d") == "d" else atoms[0]
            fragment = find_fragment_indices(adj, b, c, side_atom)
            
            torsions_spec.append({
                "name": t.get("name", ""),
                "atoms": atoms,
                "b": b,
                "c": c,
                "fragment": fragment
            })
        
        print(f"Loaded {len(torsions_spec)} torsion definitions")
    else:
        print("No torsions used in optimization")
    
    # Apply torsions to monomer
    modified_coords = apply_torsions_to_monomer(
        monomer_coords_centered, atom_types, results['torsions'], torsions_spec
    )
    
    # Build transformation vector
    transform_vec = build_transform_vector(results)
    
    # Build N-layer stack
    print(f"Building {args.n_molecules}-layer stack...")
    all_atoms, all_coords, layer_coords = build_n_layer_stack(
        atom_types, modified_coords, transform_vec, n_layers=args.n_molecules
    )
    
    # Translate stack to have monomer center at origin for first layer
    final_coords = all_coords + monomer_center
    
    # Write output
    comment = f"{args.n_molecules}-layer stack (E={results['objective']:.3f} kJ/mol, θ={results['theta']:.1f}°)"
    write_xyz_file(args.output, final_coords, all_atoms, comment)
    print(f"Wrote {args.n_molecules}-molecule stack to: {args.output}")
    
    # Optionally save modified monomer
    if args.save_monomer:
        monomer_output = args.output.replace('.xyz', '_monomer.xyz')
        monomer_comment = f"Torsion-modified monomer ({len(results['torsions'])} torsions applied)"
        write_xyz_file(monomer_output, modified_coords + monomer_center, atom_types, monomer_comment)
        print(f"Wrote modified monomer to: {monomer_output}")
    
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
