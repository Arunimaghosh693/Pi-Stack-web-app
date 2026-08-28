---
title: Supramolecular Stack Formation
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# 🧬 Supramolecular Stack Formation

A comprehensive web application for analyzing molecular structures and building supramolecular stacks with PDI/NDI core molecules.

## Features

- **Structure Input & Validation**: Upload XYZ files or provide SMILES strings
- **PDI/NDI Core Detection**: Automatic detection and validation of PDI/NDI cores with symmetric sidechains
- **Molecular Analysis**: Calculate comprehensive molecular descriptors including:
  - Aromatic system profile
  - Intermolecular interaction potential
  - Electronic characteristics
  - Hydrogen bonding potential
- **Database Matching**: Search for exact or similar molecules in a curated database using a user-defined similarity threshold
- **Stack Building**: Generate optimized supramolecular stacks using:
  - Database-matched parameters (instant results when a similar molecule is found)
  - A custom π-stack optimizer package (when no similar molecule is found) — lower the similarity threshold and search again, or download a ready-to-run optimizer pre-configured for your molecule

## Requirements

- Python 3.8+
- RDKit for molecular structure processing
- Open Babel for XYZ to SMILES conversion
- Streamlit for web interface

## Usage

1. **Input Structure**: Choose SMILES or XYZ file input
2. **Validate**: System checks for PDI/NDI core and symmetric sidechains
3. **Analyze**: View comprehensive molecular descriptors and properties
4. **Build Stack**:
   - If a similar molecule is found within your similarity threshold, its stacking parameters are used directly to build the stack.
   - If no similar molecule is found, you can either lower the similarity threshold and search again, or download a custom π-stack optimizer package that is generated automatically for your molecule.

## Database

The application includes a curated database of PDI/NDI molecules with pre-optimized stacking parameters, matched by:
- Exact molecular matches (canonical SMILES)
- Similarity-based matching (Tanimoto coefficient) against a user-defined threshold

## Technology Stack

- **Frontend**: Streamlit
- **Cheminformatics**: RDKit
- **3D Visualization**: py3Dmol
- **Data Processing**: Pandas, NumPy
- **Structure Conversion**: Open Babel

## Citation

If you use this application in your research, please cite:
[Your publication details here]

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please contact: [Your contact information]
