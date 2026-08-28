"""
Image generation and processing utilities for molecular visualization.

This module provides comprehensive image generation capabilities for molecular structures
including adaptive sizing, multiple drawing styles, and responsive design support.
"""

import io
import base64
from typing import Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from datetime import datetime
import py3Dmol
from stmol import showmol

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from utils.descriptors import xyz_to_molformula
except ImportError:
    # Fallback if descriptors module not available
    def xyz_to_molformula(xyz_content):
        return None



def show_xyz_3d(
        xyz_content,
        height=500,
        width=700,
        background="black",
        spin=False):

    viewer = py3Dmol.view(width=width, height=height)

    viewer.addModel(xyz_content, "xyz")

    # ---- VMD BALANCED LOOK ----
    viewer.setStyle({
        "stick": {
            "radius": 0.18,
            "colorscheme": "default"
        },
        "sphere": {
            "scale": 0.25,
            "colorscheme": "default"
        }
    })

    viewer.setBackgroundColor(background)
    viewer.zoomTo()

    if spin:
        viewer.spin(True)

    showmol(viewer, height=height, width=width)

