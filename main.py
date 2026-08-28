#!/usr/bin/env python3
"""
Standalone Supramolecular Stack Formation Web Application
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import json
import base64
import zipfile
import subprocess
import io
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path 
from collections import Counter



# Lazy RDKit Loader
@st.cache_resource
def load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import (
            AllChem,
            Descriptors,
            DataStructs,
            Draw,
            rdMolDescriptors
        )
        from PIL import Image

        return True, Chem, Draw

    except Exception as e:
        st.error(
            f"RDKit import failed: {type(e).__name__}: {e}"
        )
        return False, None, None


@st.cache_resource
def get_database():
    """Load database with caching to prevent blocking on startup"""
    try:
        from utils.database_match import load_molecular_database
        return load_molecular_database()
    except Exception as e:
        print(f"Warning: Could not load database: {e}")
        return None

# MUST be first Streamlit command
st.set_page_config(
    page_title="🧬 Supramolecular Stack Formation", 
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
section[data-testid="stSidebar"] {
    position: fixed;
    height: 100vh;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# -------------------------------
# Session Management Functions
# -------------------------------

def clear_analysis_results():
    """Clear analysis + visualization when geometry becomes invalid"""
    
    keys_to_clear = [
        "descriptors",
        "core_sidechain_info",
        "analysis_completed",
        "xyz_content_for_build",
        "db_check_result",
        "last_stack_result",
        "stack_built_settings",
    ]

    for k in keys_to_clear:
        st.session_state.pop(k, None)

def clear_molecule_session():
    """Clear all molecule-related session state to start fresh"""
    # Increment clear_counter to force main input widget recreation
    if 'clear_counter' not in st.session_state:
        st.session_state.clear_counter = 0
    st.session_state.clear_counter += 1

    # Increment sidebar_reset_counter to force sidebar widget recreation back to defaults
    if 'sidebar_reset_counter' not in st.session_state:
        st.session_state.sidebar_reset_counter = 0
    st.session_state.sidebar_reset_counter += 1

    # Clear other session state variables
    keys_to_clear = [
        'xyz_content', 'smiles', 'descriptors', 'xyz_content_for_build',
        'db_check_result', 'search_performed', 'match_result', 'match_type',
        'previous_input_mode', 'smiles_input_value', 'current_smiles',
        'validation_result', 'can_proceed_to_build', 'validation_performed',
        'smiles_source', 'analysis_completed', 'core_sidechain_info',
        'stack_built_settings', 'last_stack_result', 'last_stack_strategy',
        'last_stack_params', 'last_stack_size', 'last_stack_smiles',
        '_validated_geom_cutoffs','_current_geom_cutoffs','_optimizer_params_changed_only','optimizer_bundle_bytes','_nomatch_bundle'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.pop("workflow_active", None)

def store_validation_result(validation_result, can_proceed=True):
    """Store validation results in session state for flow control"""
    st.session_state.validation_result = validation_result
    st.session_state.can_proceed_to_build = can_proceed and validation_result.is_valid
    st.session_state.validation_performed = True
    st.session_state.analysis_completed = True
    st.session_state.symmetric_info = True
    
    current_cutoffs = st.session_state.get('_current_geom_cutoffs')
    if validation_result.is_valid:
        if '_validated_geom_cutoffs' not in st.session_state:
            st.session_state['_validated_geom_cutoffs'] = st.session_state['_current_geom_cutoffs']
    
    else:
        # Fallback: read from sidebar values directly
        bonded = st.session_state.get(f"sb_bonded_{st.session_state.get('sidebar_reset_counter',0)}")
        angle = st.session_state.get(f"sb_angle_{st.session_state.get('sidebar_reset_counter',0)}")
        nonbonded = st.session_state.get(f"sb_nonbonded_{st.session_state.get('sidebar_reset_counter',0)}")
       
        fallback_cutoffs = (
            round(float(bonded), 3),
            round(float(angle), 3),
            round(float(nonbonded), 3),
        )
        if validation_result.is_valid or '_validated_geom_cutoffs' not in st.session_state:
            st.session_state['_validated_geom_cutoffs'] = fallback_cutoffs



def cleanup_large_session_data():
    """Clean up large data from session state to free memory"""
    large_keys = ['xyz_content', 'xyz_content_for_build', 'descriptors']
    for key in large_keys:
        if key in st.session_state:
            data_size = len(str(st.session_state[key]))
            if data_size > 10000:  # > 10KB
                st.info(f"🧹 Cleaned up {key} ({data_size/1000:.1f}KB) to free memory")
                del st.session_state[key]


def get_energy_value(params: Dict[str, Any]) -> Optional[float]:
    """Return the database energy value, preferring the current interaction_energy key."""
    if not params:
        return None
    for key in ("interaction_energy", "binding_energy"):
        value = params.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


# -------------------------------
# No-Match Workflow Helpers
# -------------------------------

def render_optimizer_package_download(
    bundle_bytes: bytes,
    filename: str,
    auto_key: str,
    show_button: bool = True,
):
    if st.session_state.pop(auto_key, False):
        b64 = base64.b64encode(bundle_bytes).decode()
        components.html(
            f"""
            <html><body>
            <a id="auto_dl_link"
               href="data:application/zip;base64,{b64}"
               download="{filename}"></a>
            <script>
                document.getElementById("auto_dl_link").click();
            </script>
            </body></html>
            """,
            height=0,
        )
        st.toast("⬇️ Your π-Stack Optimizer package download has started.")

    if show_button:
        st.download_button(
            label="📦 Download π-Stack Optimizer Package",
            data=bundle_bytes,
            file_name=filename,
            mime="application/zip",
            key=f"dl_btn_{auto_key}",
        )
# -------------------------------
# Parameter Collection Functions
# -------------------------------

def collect_optimizer_parameters_from_ui(opt_method: str) -> Dict[str, Any]:
    """
    Collect current UI state from session_state and delegate parameter
    building to utils/optimizer_config.py::build_optimizer_parameters.
    """
    from utils.optimizer_config import build_optimizer_parameters
    ui_state = dict(st.session_state)
    return build_optimizer_parameters(opt_method, ui_state)

# ----------------------
def run_database_search(smiles,similarity_threshold,bonded_cutoff,angle_cutoff,nonbonded_cutoff,stack_size,opt_method,max_iters,current_optimizer_fingerprint):
    from utils.molecule_io_updated import smiles_to_xyz
    from utils.database_match import  check_database_match_with_smile

    
    with st.spinner("🔍 Searching database for matching parameters..."):
        current_xyz = st.session_state.get("xyz_content")

        if not current_xyz and smiles:
            current_xyz = smiles_to_xyz(smiles)

        if not current_xyz:
            st.error("❌ Failed to generate 3D structure.")
            st.stop()

        st.session_state["xyz_content_for_build"] = current_xyz

    
        st.session_state.pop("stack_built_settings", None)
        st.session_state.pop("last_stack_result", None)
        st.session_state.pop("last_stack_strategy", None)
        st.session_state.pop("last_stack_params", None)

        smiles_source = st.session_state.get("smiles_source", "none")

        if smiles_source == "user_provided" and smiles:

            st.info("🎯 **Database Matching Strategy:** Using user-provided SMILES")

            db_result = check_database_match_with_smile(
                smiles,
                current_xyz,
                similarity_threshold,
                bonded_cutoff,
                angle_cutoff,
                nonbonded_cutoff,
                smiles_source=smiles_source,
            )

        elif smiles_source == "generated_from_xyz" and smiles:

            st.info("🔄 **Database Matching Strategy:** Using generated SMILES")

            db_result = check_database_match_with_smile(
                smiles,
                current_xyz,
                similarity_threshold,
                bonded_cutoff,
                angle_cutoff,
                nonbonded_cutoff,
                smiles_source=smiles_source,
            )
        else:
            st.markdown(
                """
                <div class="status-card status-no-match">
                <h4>⚠️ No SMILES representation available</h4>
                <p>Database matching cannot be performed.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            db_result = {
                "strategy": "pi-stack-optimizer",
                "stacking_parameters": None,
                "validation_status": "inherited",
                "xyz_content": current_xyz,
            }

        st.session_state["db_check_result"] = db_result

        st.session_state["stack_built_settings"] = {
            "similarity_threshold": round(float(similarity_threshold), 4),
            "stack_size": int(stack_size),
            "opt_method": opt_method,
            "max_iters": max_iters,
            "optimizer_fingerprint": current_optimizer_fingerprint,
        }

        st.rerun()

# -------------------------------
# Streamlit UI
# -------------------------------

def main():
    from utils.stack_builder import build_stack
    from utils.optimizer_bundle import create_optimizer_bundle
    from utils.molecule_io_updated import smiles_to_xyz, get_molecule_metadata, find_xyz_by_smiles,xyz_to_smiles_obabel
    from utils.ui_formatters import (
    fmt_torsions,
    fmt_torsions_html,
    format_formula,
    render_stacking_params_card,
    render_pi_stack_analysis,
    render_clash_comparison,
    create_download_section,
    show_quality_badge,
)
    from utils.file_utils import create_zip_bundle 

    from utils.validation import (
        check_monomer_geometry_sanity,
        ValidationResult,ValidationStatus,
        read_xyz_coords,
        check_input_smile_symmetry
    )
    from utils.similarity import find_most_similar
    from utils.descriptors import (
        compute_descriptors,
        calculate_conjugation_properties,
        analyze_molecular_composition,
        calculate_simple_electronic_properties,
        xyz_to_molformula
    )
    
    from utils.database_match import (
        check_database_match_with_smile,

    )
    from utils.image_utils import show_xyz_3d
    from utils.pi_stack_analysis_simpler import analyze_pi_stack
    if 'user_session_start' not in st.session_state:
        st.session_state.user_session_start = pd.Timestamp.now()
    
    # Large molecule warning removed from global scope to prevent navigation issues
    
    st.markdown("""
    <style>
        .stMetric {
            background-color: #ffffff;
            padding: 16px 18px;
            border-radius: 10px;
            border: 1px solid #d8e2ec;
            box-shadow: 0 3px 12px rgba(31, 60, 88, 0.06);
            min-height: 106px;
        }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 16px 18px;
            border-radius: 10px;
            border: 1px solid #d8e2ec;
            box-shadow: 0 3px 12px rgba(31, 60, 88, 0.06);
            min-height: 106px;
        }
        div[data-testid="stMetricLabel"] {
            color: #111827;
            font-size: 0.86rem;
            font-weight: 500;
        }
        div[data-testid="stMetricValue"] {
            color: #111827;
            font-size: 2rem;
            font-weight: 500;
            line-height: 1.2;
        }
        .mol-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .st-emotion-cache-1v0mbdj > img {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .param-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0 25px 0;
            color: white;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .param-card h3 {
            color: white;
            margin-bottom: 15px;
            font-weight: 600;
        }
                
        .param-card h4 {
            color: white;
            margin-bottom: 4px;
            font-weight: 600;
        }


        .param-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .param-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .param-label {
            font-size: 0.8em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .param-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        .param-item-wide {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
            grid-column: 1 / -1;
        }
        .param-item-wide .param-value {
            font-size: 1.0em;
            font-weight: bold;
            word-break: break-word;
            white-space: normal;
        }
        .status-card {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .status-exact {
            background: linear-gradient(45deg, #11998e, #38ef7d);
            color: white;
        }
        .status-similar {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }
        .status-no-match {
            background: linear-gradient(135deg, #eef4ff 0%, #dbeafe 100%);
            border: 1px solid #cbd8ff;
            border-left: 5px solid #4c6ef5;
            color: #1f3c88;
            box-shadow: 0 3px 14px rgba(31, 60, 88, 0.08);
        }
        .status-no-match h4 {
            color: #102a6b;
            margin-top: 0;
        }
        .status-no-match p {
            color: #1f3c88;
        }
        .build-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border: 2px solid #e9ecef;
        }
        .validation-card {
            background: #eef4ff;
            border: 1px solid #cbd8ff;
            border-left: 5px solid #4c6ef5;
            border-radius: 10px;
            padding: 16px 18px;
            margin: 12px 0 16px;
            box-shadow: 0 2px 10px rgba(34, 82, 112, 0.06);
        }
        .validation-card h4 {
            color: #1f3c88;
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 8px;
        }
        .validation-card p {
            color: #1f3c88;
            font-size: 0.95rem;
            line-height: 1.45;
            margin: 0;
        }
        .monomer-summary-card {
            background: #ffffff;
            border: 1px solid #d8e2ec;
            border-radius: 16px;
            padding: 26px 28px;
            margin-top: 30px;
            box-shadow: 0 3px 14px rgba(31, 60, 88, 0.08);
        }
        .monomer-summary-card h4 {
            color: #111827;
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 22px;
        }
        .monomer-summary-item {
            border-bottom: 1px solid #d8e2ec;
            padding: 0 0 17px;
            margin-bottom: 17px;
        }
        .monomer-summary-item:last-child {
            border-bottom: 0;
            padding-bottom: 0;
            margin-bottom: 0;
        }
        .monomer-summary-label {
            color: #52657a;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 0 0 8px;
            text-transform: uppercase;
        }
        .monomer-summary-value {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.35;
            margin: 0;
            word-break: break-word;
        }
        .optimizer-action-card {
            background: #ffffff;
            border: 1px solid #d8e7ff;
            border-left: 5px solid #4c6ef5;
            border-radius: 12px;
            padding: 22px 24px;
            margin: 10px 0 18px;
            box-shadow: 0 4px 16px rgba(31, 60, 88, 0.08);
        }
        .optimizer-action-card h4 {
            color: #111827;
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0 0 14px;
        }
        .optimizer-proceed-card {
            background: #ffffff;
            border: 1px solid #d8e7ff;
            border-left: 4px solid #4c6ef5;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 14px 0 16px;
            max-width: 760px;
            box-shadow: 0 3px 12px rgba(31, 60, 88, 0.06);
        }
        .optimizer-proceed-card h4 {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 6px;
        }
        .optimizer-proceed-card p {
            color: #111827;;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }
        .optimizer-action-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 14px;
        }
        .optimizer-action-item {
            background: #f7fbff;
            border: 1px solid #e1ecff;
            border-radius: 10px;
            padding: 18px 18px;
            margin: 10px 0 14px;
        }
        .optimizer-action-item-selected {
            background: #f7fbff;
            border: 1px solid #d8e7ff;
            border-left: 5px solid #4c6ef5;
            border-radius: 12px;
            padding: 18px 18px 16px;
            margin: 10px 0 12px;
            box-shadow: 0 3px 12px rgba(31, 60, 88, 0.06);
        }
        .optimizer-action-title {
            color: #111827;
            font-size: 0.98rem;
            font-weight: 700;
            margin: 0 0 7px;
        }
        .optimizer-action-text {
            color: #4b5563;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0;
        }
        .optimizer-action-note {
            color: #1f3c88;
            font-size: 0.84rem;
            font-weight: 600;
            line-height: 1.45;
            margin: 12px 0 0;
        }
    </style>
    """, unsafe_allow_html=True)

    #st.title("🧬 Supramolecular Stack Builder - Construct and analyze π-stacked molecular assemblies")
    st.markdown("""
    <div style="
        text-align:center;
        padding:15px 0 30px 0;
    ">

    <h1 style="
        font-size:3rem;
        font-weight:800;
        color:#1e293b;
        margin-bottom:0px;
    ">
    🧬 Supramolecular Stack Builder
    </h1>
    <p style="
        font-size:1.15rem;
        margin-top:0;
        margin-bottom:0px;
    ">
    Construct and analyze π-stacked molecular assemblies
    </p>

    </div>
    """, unsafe_allow_html=True)



    #st.title("Construct and analyze π-stacked molecular assemblies")
    st.markdown("""
    


    Upload a **XYZ file** or provide a **SMILES string** to validate, analyze, and construct stacks.

    """)
    
    st.markdown("""
    <div style="
    background-color:#eef4ff;
    padding:6px 10px;
    border-radius:6px;
    font-size:14px;
    color:#1f3c88;
    border-left:3px solid #4c6ef5;
    margin-top:4px;
    margin-bottom:8px;
    ">

    This application currently supports molecules containing 
    <b>PDI (Perylene Diimide)</b> and <b>NDI (Naphthalene Diimide)</b> cores

    </div>
    """, unsafe_allow_html=True)

    RDKIT_AVAILABLE, Chem, Draw = load_rdkit()
    if not RDKIT_AVAILABLE:
        st.warning("⚠️ RDKit not available. Some functionality will be limited.")
    

    try:
        database = get_database()
        if database is None:
            st.warning("⚠️ Database not loaded. Using fallback mode.")
    except Exception as e:
        st.error(f"⚠️ Database loading error: {e}")
        database = None
    
    
    with st.sidebar:
        st.header("⚙️ Settings")
        sc = st.session_state.get('sidebar_reset_counter', 0) 

        with st.expander("Geometry Validation", expanded=False):
            bonded_cutoff = st.slider("Bond cutoff (Å)", 0.8, 1.5, 0.8, 0.1, key=f"sb_bonded_{sc}")
            angle_cutoff = st.slider("Angle cutoff (Å)", 1.1, 1.5, 1.1, 0.1, key=f"sb_angle_{sc}")
            nonbonded_cutoff = st.slider("Non-bonded cutoff (Å)", 1.3, 1.5, 1.3, 0.1, key=f"sb_nonbonded_{sc}")
            # Track current cutoffs so store_validation_result can snapshot them
            st.session_state['_current_geom_cutoffs'] = (
                round(float(bonded_cutoff), 3),
                round(float(angle_cutoff), 3),
                round(float(nonbonded_cutoff), 3),
            )

        with st.expander("Stack Parameters", expanded=True):
            stack_size = st.number_input("Stack size", min_value=2, max_value=10, value=3, key=f"sb_stack_size_{sc}")
            similarity_threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.9, 0.05, key=f"sb_sim_thresh_{sc}")

        with st.expander("π-Stack Optimizer Parameters (used when no database match)", expanded=False):
            st.write("**General Options**")
            opt_method = st.selectbox(
                "Optimization method", ["pso", "ga", "gwo", "pso-nm"], index=0,
                help="Optimization algorithm to use", key=f"sb_opt_method_{sc}"
            )
            max_iters = st.number_input(
                "Maximum iterations", min_value=1, max_value=1000, value=300, step=1,
                help="Maximum optimizer iterations/generations", key=f"sb_max_iters_{sc}"
            )
            seed = st.number_input(
                "Random seed", min_value=1, max_value=9999, value=42, step=1,
                help="Random seed for optimizer initialization", key=f"sb_seed_{sc}"
            )

            # Store general parameters in session state
            st.session_state.update({'opt_method': opt_method, 'max_iters': max_iters, 'seed': seed})
            if opt_method == "pso":
                st.write("**PSO Options**")
                col1, col2 = st.columns(2)
                with col1:
                    swarm_size = st.number_input("Swarm size", min_value=20, max_value=200, value=60, step=10, key=f"sb_swarm_{sc}")
                    inertia = st.number_input("Inertia weight", min_value=0.1, max_value=2.0, value=0.73, step=0.01, key=f"sb_inertia_{sc}")
                    cognitive = st.number_input("Cognitive coefficient", min_value=0.1, max_value=3.0, value=1.50, step=0.1, key=f"sb_cognitive_{sc}")
                with col2:
                    social = st.number_input("Social coefficient", min_value=0.1, max_value=3.0, value=1.50, step=0.1, key=f"sb_social_{sc}")
                    pso_tol = st.number_input("PSO tolerance", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key=f"sb_pso_tol_{sc}")
                    patience = st.number_input("PSO patience", min_value=5, max_value=100, value=20, step=5, key=f"sb_patience_{sc}")
                st.session_state.update({'swarm_size': swarm_size, 'inertia': inertia, 'cognitive': cognitive,
                                         'social': social, 'pso_tol': pso_tol, 'patience': patience})

            elif opt_method == "ga":
                st.write("**Genetic Algorithm Options**")
                col1, col2 = st.columns(2)
                with col1:
                    ga_population = st.number_input("Population size", min_value=20, max_value=200, value=80, step=10, key=f"sb_ga_pop_{sc}")
                    ga_mutation_rate = st.number_input("Mutation rate", min_value=0.01, max_value=0.50, value=0.10, step=0.01, key=f"sb_ga_mut_{sc}")
                    ga_mutation_sigma = st.number_input("Mutation sigma", min_value=0.10, max_value=1.0, value=0.30, step=0.05, key=f"sb_ga_sigma_{sc}")
                    ga_crossover_rate = st.number_input("Crossover rate", min_value=0.50, max_value=1.0, value=0.90, step=0.05, key=f"sb_ga_cross_{sc}")
                with col2:
                    ga_elite_fraction = st.number_input("Elite fraction", min_value=0.05, max_value=0.30, value=0.10, step=0.01, key=f"sb_ga_elite_{sc}")
                    ga_tournament_size = st.number_input("Tournament size", min_value=2, max_value=10, value=3, step=1, key=f"sb_ga_tour_{sc}")
                    ga_tol = st.number_input("GA tolerance", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key=f"sb_ga_tol_{sc}")
                    ga_patience = st.number_input("GA patience", min_value=5, max_value=100, value=20, step=5, key=f"sb_ga_patience_{sc}")
                st.session_state.update({'ga_population': ga_population, 'ga_mutation_rate': ga_mutation_rate,
                                         'ga_mutation_sigma': ga_mutation_sigma, 'ga_crossover_rate': ga_crossover_rate,
                                         'ga_elite_fraction': ga_elite_fraction, 'ga_tournament_size': ga_tournament_size,
                                         'ga_tol': ga_tol, 'ga_patience': ga_patience})

            elif opt_method == "gwo":
                st.write("**Grey Wolf Optimizer Options**")
                col1, col2 = st.columns(2)
                with col1:
                    gwo_pack_size = st.number_input("Pack size", min_value=20, max_value=150, value=50, step=10, key=f"sb_gwo_pack_{sc}")
                    gwo_a_start = st.number_input("Initial exploration", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key=f"sb_gwo_astart_{sc}")
                    gwo_a_end = st.number_input("Final exploration", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key=f"sb_gwo_aend_{sc}")
                with col2:
                    gwo_tol = st.number_input("GWO tolerance", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key=f"sb_gwo_tol_{sc}")
                    gwo_patience = st.number_input("GWO patience", min_value=5, max_value=100, value=20, step=5, key=f"sb_gwo_patience_{sc}")
                st.session_state.update({'gwo_pack_size': gwo_pack_size, 'gwo_a_start': gwo_a_start,
                                         'gwo_a_end': gwo_a_end, 'gwo_tol': gwo_tol, 'gwo_patience': gwo_patience})

            elif opt_method == "pso-nm":
                st.write("**Hybrid PSO+Nelder-Mead Options**")
                col1, col2 = st.columns(2)
                with col1:
                    swarm_size = st.number_input("Swarm size", min_value=20, max_value=200, value=60, step=10, key=f"sb_nm_swarm_{sc}")
                    inertia = st.number_input("Inertia weight", min_value=0.1, max_value=2.0, value=0.73, step=0.01, key=f"sb_nm_inertia_{sc}")
                    cognitive = st.number_input("Cognitive coefficient", min_value=0.1, max_value=3.0, value=1.50, step=0.1, key=f"sb_nm_cognitive_{sc}")
                    social = st.number_input("Social coefficient", min_value=0.1, max_value=3.0, value=1.50, step=0.1, key=f"sb_nm_social_{sc}")
                    pso_tol = st.number_input("PSO tolerance", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key=f"sb_nm_pso_tol_{sc}")
                    patience = st.number_input("PSO patience", min_value=5, max_value=100, value=20, step=5, key=f"sb_nm_patience_{sc}")
                with col2:
                    nm_max_iters = st.number_input("NM max iterations", min_value=50, max_value=500, value=200, step=25, key=f"sb_nm_maxiters_{sc}")
                    nm_initial_step = st.number_input("NM initial step", min_value=0.05, max_value=1.0, value=0.20, step=0.05, key=f"sb_nm_step_{sc}")
                    nm_alpha = st.number_input("NM alpha (reflection)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key=f"sb_nm_alpha_{sc}")
                    nm_gamma = st.number_input("NM gamma (expansion)", min_value=1.1, max_value=3.0, value=2.0, step=0.1, key=f"sb_nm_gamma_{sc}")
                    nm_rho = st.number_input("NM rho (contraction)", min_value=0.1, max_value=0.9, value=0.50, step=0.05, key=f"sb_nm_rho_{sc}")
                    nm_sigma = st.number_input("NM sigma (shrink)", min_value=0.1, max_value=0.9, value=0.50, step=0.05, key=f"sb_nm_sigma_{sc}")
                st.session_state.update({'swarm_size': swarm_size, 'inertia': inertia, 'cognitive': cognitive,
                                         'social': social, 'pso_tol': pso_tol, 'patience': patience,
                                         'nm_max_iters': nm_max_iters, 'nm_initial_step': nm_initial_step,
                                         'nm_alpha': nm_alpha, 'nm_gamma': nm_gamma, 'nm_rho': nm_rho, 'nm_sigma': nm_sigma})

            st.write("**xTB Backend**")
            col3, col4 = st.columns(2)
            with col3:
                workers = st.number_input("Number of workers", min_value=1, max_value=16, value=4, step=1, key=f"sb_workers_{sc}")
                threads = st.number_input("Threads per calculation", min_value=1, max_value=8, value=1, step=1, key=f"sb_threads_{sc}")
                xtb_method = st.selectbox("xTB method", ["gfn2", "gfn1", "gfn0", "gfnff"], index=0, key=f"sb_xtb_{sc}")
            with col4:
                charge = st.number_input("Molecular charge", min_value=-5, max_value=5, value=0, step=1, key=f"sb_charge_{sc}")
                mult = st.number_input("Spin multiplicity", min_value=1, max_value=5, value=1, step=1, key=f"sb_mult_{sc}")

            st.write("**Penalty Settings**")
            col5, col6 = st.columns(2)
            with col5:
                penalty_weight = st.number_input("Clash penalty weight", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key=f"sb_penalty_{sc}")
                clash_cutoff = st.number_input("Clash cutoff (Å)", min_value=1.0, max_value=3.0, value=1.6, step=0.1, key=f"sb_clash_{sc}")
            with col6:
                intramol_penalty_weight = st.number_input("Intramol penalty weight", min_value=0.1, max_value=10.0, value=5.0, step=0.1, key=f"sb_intramol_pen_{sc}")
                intramol_cutoff = st.number_input("Intramol cutoff (Å)", min_value=0.8, max_value=2.0, value=1.2, step=0.1, key=f"sb_intramol_cut_{sc}")

            # Store backend and penalty parameters
            st.session_state.update({
                'workers': workers, 'threads': threads, 'xtb_method': xtb_method,
                'charge': charge, 'mult': mult, 'penalty_weight': penalty_weight,
                'clash_cutoff': clash_cutoff, 'intramol_penalty_weight': intramol_penalty_weight,
                'intramol_cutoff': intramol_cutoff
            })
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🔬 Structure Analysis", "🔗 Build Stack"])
    with tab1:
        st.header("🔍 Structure Validation")
        input_mode = st.radio("Choose input type:",("SMILES", "XYZ file"),key="input_mode_radio")
        
        # Clear session state when input mode changes
        if 'previous_input_mode' not in st.session_state:
            st.session_state.previous_input_mode = input_mode
        elif st.session_state.previous_input_mode != input_mode:
            clear_molecule_session()
            st.session_state.previous_input_mode = input_mode
            st.rerun()
        
        xyz_content = None
        smiles = None
        
        if input_mode == "SMILES":
            clear_counter = st.session_state.get('clear_counter', 0)
            smiles = st.text_input(
                "Enter SMILES string:", 
                placeholder="O=C1NC(=O)c2ccc3c4c(ccc1c24)C(=O)NC3=O",
                key=f"smiles_input_{clear_counter}"
            )
            
            if smiles:
                st.session_state.smiles_source = "user_provided"
                st.session_state.smiles = smiles
                st.session_state.workflow_active = True
                
                # Validate SMILES and generate structure
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        st.session_state.validation_performed = True
                        st.session_state.can_proceed_to_build = False
                        st.session_state.validation_result = ValidationResult(
                            status=ValidationStatus.UNKNOWN_ERROR,
                            is_valid=False,
                            message="Invalid SMILES string. Please enter a valid molecule.",
                            issues=["SMILES parsing failed."],
                            symmetry_info=None
                        )
                        st.error("❌ Invalid SMILES string. Please enter a valid molecule. No further analysis or stack formation is possible.")
                        
                        # Add reset button for user to try again
                        st.divider()
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            if st.button("🆕 Start New Analysis",
                                        help="Clear input and try a different SMILES string",
                                        type="primary",
                                        key="reset_button_invalid_smiles"):
                                clear_molecule_session()
                                st.rerun()
                        st.stop()
                    else:
                        with st.spinner("Processing SMILES: Validating and generating 3D structure..."):
                            # Add timeout for multi-user environment
                            start_time = pd.Timestamp.now()
                            symmetry_result = check_input_smile_symmetry(smiles)
                            
                            if not symmetry_result.is_valid:
                                st.session_state.validation_performed = True
                                st.session_state.can_proceed_to_build = False
                                st.session_state.validation_result = symmetry_result
                                st.error(f"❌ {symmetry_result.message}")
                                if symmetry_result.issues:
                                    st.write("**Issues:**")
                                    for issue in symmetry_result.issues:
                                        st.write(f"• {issue}")
                                st.error("**No further analysis or stack formation possible.**")
                                
                                # Add reset button for user to try a different molecule
                                st.divider()
                                col1, col2, col3 = st.columns([1, 1, 1])
                                with col2:
                                    if st.button("🆕 Start New Analysis",
                                                help="Clear all data and start fresh with a new molecule",
                                                type="primary",
                                                key="reset_button_symmetry_fail_smiles"):
                                        clear_molecule_session()
                                        st.rerun()
                                st.stop()
                            
                            # If we reach here, molecule has valid symmetric sidechains
                            if symmetry_result.symmetry_info:
                                st.session_state.core_sidechain_info = symmetry_result.symmetry_info
                            
                            xyz_content = smiles_to_xyz(smiles, use_precomputed=True)
                            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
                            if processing_time > 10:
                                st.info(f"⏱️ Structure generation took {processing_time:.1f} seconds")
                             
                         
                            if xyz_content:     
                                st.session_state.xyz_content = xyz_content
                                st.session_state.smiles = smiles
                                
                                try:
                                    # Write XYZ to temporary file for validation
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
                                        f.write(xyz_content)
                                        temp_path = f.name
                                    
                                    try:
                                        atom_types, coords = read_xyz_coords(temp_path)
                                        validation_result = check_monomer_geometry_sanity(
                                            coords, atom_types, bonded_cutoff, angle_cutoff, nonbonded_cutoff
                                        )
                        
                                        store_validation_result(validation_result, can_proceed=True)                
                                        # Show combined results
                                        lines = xyz_content.split('\n')
                                        num_atoms = lines[0].strip() if len(lines) >= 2 else "unknown"
                                        
                                        if validation_result.is_valid:
                                             st.success(f"✅ Validation successful. A chemically valid 3D molecular structure containing {num_atoms} atoms has been generated and is ready for analysis")
                                             st.markdown("""
                                             <div style="
                                             background-color:#eef4ff;
                                             padding:10px 14px;
                                             border-radius:6px;
                                             font-size:14px;
                                             color:#4a5568;
                                             border-left:3px solid #6c8ef5;
                                             margin-top:6px;
                                             ">
                                             ⚙ Geometry validation thresholds can be adjusted in the <b>Settings</b> panel on the left.
                                             </div>
                                             """, unsafe_allow_html=True)
                                        else:
                                            st.session_state.can_proceed_to_build = False
                                            clear_analysis_results()

                                            st.error(f"❌ Invalid geometry: {validation_result.message}\nNo further analysis or stack formation possible.")
                                    finally:
                                        os.unlink(temp_path)
                                        
                                except Exception as e:
                                    lines = xyz_content.split('\n')
                                    num_atoms = lines[0].strip() if len(lines) >= 2 else "unknown"
                                    clear_analysis_results()
                                    st.session_state.validation_performed = True
                                    st.session_state.can_proceed_to_build = False
                                    st.session_state.validation_result = ValidationResult(
                                        status=ValidationStatus.UNKNOWN_ERROR,
                                        is_valid=False,
                                        message=f"Validation failed due to error: {e}",
                                        issues=[f"Validation error: {e}"],
                                        symmetry_info=None
                                    )
                                    st.error(f"❌ Generated 3D structure with {num_atoms} atoms - validation failed: {e}")
                                    st.info("**Solutions:** Try a simpler molecule or upload a validated XYZ file directly.")
                            else:
                                st.error("❌ Valid SMILES but failed to generate 3D structure")
                                st.info("**Possible causes**: Complex molecule, coordination compounds, or unusual chemistry")
                                st.info("**Alternative**: Try uploading a pre-generated XYZ file using the 'XYZ file' option above")
        
        elif input_mode == "XYZ file":
            clear_counter = st.session_state.get('clear_counter', 0)
            uploaded_file = st.file_uploader("Upload XYZ file", type="xyz", key=f"xyz_uploader_{clear_counter}")
            if uploaded_file:
                st.session_state.workflow_active = True
                xyz_content = uploaded_file.getvalue().decode("utf-8")
                st.session_state.xyz_content = xyz_content
                
                generated_smiles = xyz_to_smiles_obabel(xyz_content)
                if generated_smiles:
                    st.session_state.smiles = generated_smiles
                    st.session_state.smiles_source = "generated_from_xyz"
                    symmetry_result = check_input_smile_symmetry(generated_smiles)
                    
                    if not symmetry_result.is_valid:
                        st.session_state.validation_performed = True
                        st.session_state.can_proceed_to_build = False
                        st.session_state.validation_result = symmetry_result
                        st.error(f"❌ {symmetry_result.message}")
                        if symmetry_result.issues:
                            st.write("**Issues:**")
                            for issue in symmetry_result.issues:
                                st.write(f"• {issue}")
                        st.error("**No further analysis or stack formation possible.**")
                        
                        # Add reset button for user to try a different molecule
                        st.divider()
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            if st.button("🆕 Start New Analysis",
                                        help="Clear all data and start fresh with a new molecule",
                                        type="primary",
                                        key="reset_button_symmetry_fail_xyz"):
                                clear_molecule_session()
                                st.rerun()
                        st.stop()
                    
                    if symmetry_result.symmetry_info:
                        st.session_state.core_sidechain_info = symmetry_result.symmetry_info
                else:
                    st.session_state.smiles = None
                    st.session_state.smiles_source = "none"
                    st.error("""❌ SMILES Generation Failed. The uploaded XYZ structure could not be converted into a SMILES representation. 
                             Please provide the SMILES string directly using the SMILES input option.""")
                    st.stop()

                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
                        f.write(xyz_content)
                        temp_path = f.name
                    try:
                        atom_types, coords = read_xyz_coords(temp_path)
                        validation_result = check_monomer_geometry_sanity(
                            coords, atom_types, bonded_cutoff, angle_cutoff, nonbonded_cutoff
                        )
                        
                        st.session_state.validation_performed = True
                        st.session_state.validation_result = validation_result
                        if not validation_result.is_valid:
                            clear_analysis_results()
                            st.session_state.can_proceed_to_build = False
                            st.error(f"❌ Invalid XYZ file: {validation_result.message}")
                            st.error("**Structure validation is required for stack formation. No further analysis is possible.**")
                            
                            if validation_result.issues:
                                st.write("**Issues found:**")
                                for issue in validation_result.issues:
                                    st.write(f"• {issue}")
                            #st.stop()
                        else:
                            store_validation_result(validation_result, can_proceed=True)
                            lines = xyz_content.split('\n')
                            num_atoms = lines[0].strip() if len(lines) >= 2 else "unknown"
                            
                            if validation_result.is_valid:
                               st.success(f"✅ Validation successful. A chemically valid 3D molecular structure containing {num_atoms} atoms has been generated and is ready for analysis")
                    finally:
                        os.unlink(temp_path)
                except Exception as e:
                    clear_analysis_results()
                    st.session_state.validation_performed = True
                    st.session_state.can_proceed_to_build = False
                    st.session_state.validation_result = ValidationResult(
                        status=ValidationStatus.UNKNOWN_ERROR,
                        is_valid=False,
                        message=f"Validation failed due to error: {e}",
                        issues=[f"Validation error: {e}"],
                        symmetry_info=None
                    )
                    st.error(f"❌ Error during XYZ validation: {e}")
                    st.error("**Structure validation is required for stack formation. No further analysis is possible.**")
                    st.info("**Solutions:** Check your XYZ file or try a different molecule.")
                    #st.stop()
        if (st.session_state.get("xyz_content") and st.session_state.get("validation_result") and st.session_state.validation_result.is_valid):
            xyz_content = st.session_state.xyz_content
            smiles = st.session_state.get('smiles') 
            # Get molecule object for drawing
            mol = None
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
            else:
                try:
                    mol = Chem.MolFromXYZBlock(xyz_content)
                except Exception:
                    pass 

            st.divider()
            st.subheader("🔬 Molecular Analysis")
            with st.spinner("Calculating molecular descriptors..."):
                try:
                    fmt = 'smi' if smiles else 'xyz'
                    input_data = smiles if smiles else xyz_content
                    
                    descriptors = compute_descriptors(input_data, fmt=fmt)
                    st.session_state.descriptors = descriptors

                    # Calculate additional molecular properties if RDKit is available
                    additional_props = {}
                    if RDKIT_AVAILABLE and mol:
                        additional_props['conjugation'] = calculate_conjugation_properties(mol)
                        additional_props['composition'] = analyze_molecular_composition(mol)
                        additional_props['electronic'] = calculate_simple_electronic_properties(mol)

                    
                    st.markdown("##### Molecule Overview")
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        if mol:
                            st.markdown('<div class="mol-card">', unsafe_allow_html=True)
                            st.markdown("### 3D Monomer")
                            show_xyz_3d(xyz_content)
                        else:
                            st.warning("Could not generate 2D image.")
                    
                    with col2:
                        # Get core and sidechain info from session state
                        core_sidechain_info = st.session_state.get('core_sidechain_info', {})
                        core_type = core_sidechain_info.get('core_type', 'Unknown')
                        sidechain = core_sidechain_info.get('sidechain', 'Unknown')
                        
                        formula_html = format_formula(descriptors.molecular_formula)
                        
                        st.markdown(
                            f"""
                            <div class="monomer-summary-card">
                                <h4>Monomer Summary</h4>
                                <div class="monomer-summary-item">
                                    <p class="monomer-summary-label">Core</p>
                                    <p class="monomer-summary-value">{core_type}</p>
                                </div>
                                <div class="monomer-summary-item">
                                    <p class="monomer-summary-label">Sidechain</p>
                                    <p class="monomer-summary-value">{sidechain}</p>
                                </div>
                                <div class="monomer-summary-item">
                                    <p class="monomer-summary-label">Formula</p>
                                    <p class="monomer-summary-value">{formula_html}</p>
                                </div>
                                <div class="monomer-summary-item">
                                    <p class="monomer-summary-label">Molecular Weight</p>
                                    <p class="monomer-summary-value">{descriptors.molecular_weight:.2f} g/mol</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    # --- Detailed Descriptor Tabs ---
                    tab_aro, tab_inter = st.tabs(["⚛️ Aromatic System Profile", "💧 Intermolecular Interaction Potential"])

                    with tab_aro:
                        st.subheader("Aromatic Surface")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("π-Core Size", f"{int(descriptors.pi_core_size)} atoms", help="Number of atoms in the aromatic system.")
                        c2.metric("Aromatic Rings", descriptors.num_aromatic_rings)
                        c3.metric("Total Rings", descriptors.num_rings)
                        #c4.metric("Planarity (RMSD)", f"{descriptors.planarity:.4f} Å", help="Root-mean-square deviation from the mean plane of the aromatic system. Lower is flatter.")
                        c4.metric("Steric Bulk Near Core",f"{descriptors.steric_bulk_near_core} atoms",help="Number of non-core heavy atoms within two bonds of the π-core. Higher values indicate greater steric crowding near the aromatic stacking surface.")
                        
                        st.subheader("Electronic Characteristics")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Molar Refractivity", f"{descriptors.mol_mr:.2f}", help="A measure of the total polarizability of the molecule.")
                        c2.metric("Aromatic Nitrogens", descriptors.aromatic_nitrogens, help="Count of nitrogen atoms within aromatic rings.")
                        c3.metric("Aromatic Oxygens", descriptors.aromatic_oxygens, help="Count of oxygen atoms within aromatic rings.")

                    with tab_inter:
                        st.subheader("Hydrogen Bonding Potential")
                        c1, c2 = st.columns(2)
                        c1.metric("H-Bond Donors", descriptors.h_bond_donors, help="Number of potential hydrogen bond donor sites.")
                        c2.metric("H-Bond Acceptors", descriptors.h_bond_acceptors, help="Number of potential hydrogen bond acceptor sites.")

                        st.subheader("Polarity & Flexibility")
                        c1, c2 = st.columns(2)
                        c1.metric("Polar Surface Area (TPSA)", f"{descriptors.polar_surface_area:.2f} Å²", help="Surface area of polar atoms; relates to solubility and permeability.")
                        c2.metric("Rotatable Bonds", descriptors.rotatable_bonds, help="Number of bonds that can rotate freely, indicating molecular flexibility.")

                except (ValueError, Exception) as e:
                    st.error(f"An error occurred during descriptor calculation: {e}")

            st.divider()
            st.success("""
            ✅ **Structure Ready for Stack Construction**

            ➡️ Proceed to the **🔗 Build Stack** tab to generate the supramolecular assembly.
            """)
        # Show reset button at the BOTTOM of tab1, only when data exists
        if st.session_state.get('xyz_content') or st.session_state.get('smiles'):
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🆕 Start New Analysis",
                            help="Clear all data and start fresh with a new molecule",
                            type="primary",
                            key="reset_button_tab1"):
                    clear_molecule_session()
                    st.rerun()

    with tab2:
        #st.header("💠 Supramolecular Stack Builder")
        st.header("💠 Database-Guided Stack Construction")
        st.markdown("Generate supramolecular stacks using database matches and similarity-derived parameters")
        #st.markdown("Build optimized molecular stacks using database-matched parameters or advanced optimization")
        smiles = st.session_state.get('smiles')
        xyz_content = st.session_state.get('xyz_content')
        has_molecular_data = bool(smiles or xyz_content)

        if xyz_content:
            lines = xyz_content.split('\n')
            if len(lines) > 2:
                atom_count = len([line for line in lines[2:] if line.strip()])
                if atom_count > 100:
                    st.warning("⚠️ **Large Molecule Detected** - Stack building and optimization may be slower during peak usage times")

        if not has_molecular_data:
            st.markdown("""
            <div class="status-card" style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404;">
                <h4>⚠️ Prerequisites</h4>
                <p>Please enter a <strong>SMILES string</strong> or upload an <strong>XYZ file</strong> in the <strong>Structure Analysis</strong> tab first.</p>
            </div>
            """, unsafe_allow_html=True)
        elif not st.session_state.get('validation_performed', False):
            st.markdown("""
            <div class="status-card" style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404;">
                <h4>⚠️ Validation Required</h4>
                <p>Please complete structure validation in the <strong>Structure Analysis</strong> tab before building stacks.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("### Step 1: Database Analysis")
   
            built_settings = st.session_state.get('stack_built_settings')
            stack_was_built = built_settings is not None
            
            previous_similarity = (
                built_settings.get('similarity_threshold')
                if built_settings else None
            )

            similarity_changed = (
                previous_similarity is not None
                and previous_similarity != round(float(similarity_threshold), 4)
            )
            
            stack_size_changed = (
                built_settings is not None
                and built_settings.get('stack_size') != int(stack_size)
            )
            
            # Create fingerprint of current optimizer parameters (includes xTB, penalty, etc.)
            current_optimizer = collect_optimizer_parameters_from_ui(opt_method)
            current_optimizer["opt_method"] = opt_method
            current_optimizer_fingerprint = json.dumps(current_optimizer, sort_keys=True)

            strategy = st.session_state.get("db_check_result", {}).get("strategy")
            previous_optimizer_fingerprint = built_settings.get("optimizer_fingerprint") if built_settings else None
            
            optimizer_params_changed = (
                built_settings is not None
                and previous_optimizer_fingerprint is not None
                and previous_optimizer_fingerprint != current_optimizer_fingerprint
            )
            
            if similarity_changed:
                
                clear_steps_123 = True
                clear_step_3_only = False
                button_message = "⚙️ Similarity Threshold changed — click to re-run database search."
            
            elif stack_size_changed or optimizer_params_changed:
                clear_steps_123 = False
                clear_step_3_only = True

                if stack_size_changed:
                    button_message = "⚙️ Stack Size changed — rebuild stack with new size."
                else:
                    
                    button_message = "⚙️ Optimizer parameters changed — regenerate optimization package."
            else:
                clear_steps_123 = False
                clear_step_3_only = False
                button_message = None

            db_check_exists = st.session_state.get('db_check_result') is not None
            check_button_disabled = (
                (stack_was_built and not (similarity_changed or stack_size_changed or optimizer_params_changed))
                or (db_check_exists and not similarity_changed)
            )

            last_geom = st.session_state.get('_validated_geom_cutoffs')
            current_geom = (
                round(float(bonded_cutoff), 3),
                round(float(angle_cutoff), 3),
                round(float(nonbonded_cutoff), 3),
            )
            geom_cutoffs_changed = (last_geom is not None and last_geom != current_geom)

            # Check if structure is invalid due to geometry validation failure
            validation_result = st.session_state.get('validation_result')
            structure_is_invalid = (
                validation_result is not None 
                and not validation_result.is_valid
            )
            
            if structure_is_invalid:
                    st.markdown("""
                    <div class="status-card" style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24;">
                        <h4>❌ Invalid Geometry Detected</h4>
                        <p>Your structure has geometry validation issues with the current cutoff parameters. Please check the 🔬 Structure
                                Analysis tab for more details.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(
                        f"💡 **Solution:** Go to the **🔬 Structure Analysis** tab and either:\n"
                        f"- Adjust the geometry validation cutoffs in the sidebar to accept this structure, OR\n"
                        f"- Use a different molecule with valid geometry\n\n"
                    )
            
                    st.stop()
                    
            elif geom_cutoffs_changed:
                # Structure is valid but geometry params changed → clear database results and show button
                st.session_state.pop("db_check_result", None)
                st.session_state.pop("stack_built_settings", None)
                st.markdown("""
                <div class="status-card"
                style="background:#fff3cd;
                        border:1px solid #ffeaa7;
                        color:#856404;">
                    <h4>⚠ Geometry Parameters Updated</h4>
                    <p>
                    Geometry cutoff values changed after validation.
                    Database analysis should be re-run.
                </p>
                </div>
                """, unsafe_allow_html=True)
                st.caption("👇 Please re-run database analysis below to proceed.")
                
                # Show the Check Database button (enabled)
                if st.button("🔍 Check Database & Prepare Build",
                            type="primary",
                            key="rerun_database_after_geom_change",
                            help="Re-analyze structure with new geometry cutoffs"):
                    
                    st.session_state['_validated_geom_cutoffs'] = current_geom

                    run_database_search(smiles,similarity_threshold,bonded_cutoff,angle_cutoff,nonbonded_cutoff,stack_size,opt_method,
                                        max_iters,current_optimizer_fingerprint)
                #st.stop()
                
            else:
                if clear_steps_123:
                    # Similarity changed → clear everything
                    for _k in ['db_check_result', 'xyz_content_for_build',
                               'last_stack_result', 'last_stack_strategy',
                               'last_stack_params', 'last_stack_size', 'last_stack_smiles',
                               'optimizer_bundle_bytes', 'stack_built_settings']:
                        st.session_state.pop(_k, None)
                    st.caption(button_message)
                elif clear_step_3_only:
                    if stack_size_changed:
                        for _k in ['last_stack_result', 'last_stack_strategy','last_stack_params', 'last_stack_size', 'last_stack_smiles']:
                             st.session_state.pop(_k, None)
                        if 'stack_built_settings' in st.session_state:
                            st.session_state['stack_built_settings']['stack_size'] = int(stack_size)
                        st.caption(button_message)

                    if optimizer_params_changed:
                        st.session_state.pop('optimizer_bundle_bytes', None)
                        st.session_state['_optimizer_params_changed_only'] = optimizer_params_changed
                else:
                    # Nothing changed
                    if stack_was_built and strategy != "pi-stack-optimizer":
                        st.caption("✅ Stack already built with current settings — change Stack Size, Similarity Threshold, or Optimizer parameters to re-run.")
                
                if st.button("🔍 Check Database & Prepare Build",
                            type="primary",
                            disabled=check_button_disabled,
                            help="Analyze structure and search database for matching parameters. Change any sidebar setting to re-enable."):
                     run_database_search(smiles,similarity_threshold,bonded_cutoff,angle_cutoff,nonbonded_cutoff,
                                         stack_size,opt_method,max_iters,current_optimizer_fingerprint)
                if st.session_state.get('db_check_result') and not similarity_changed:
                    result = st.session_state['db_check_result']

                    if result.get("error"):
                        st.markdown(f"""
                        <div class="status-card" style="background:#f8d7da;border:1px solid #f5c6cb;color:#721c24;">
                            <h4>❌ Validation Error</h4><p>{result['error']}</p>
                        </div>""", unsafe_allow_html=True)
                        if result.get("validation_status") == "failed":
                            for issue in result.get("validation_issues", []):
                                st.write(f"• {issue}")
                        st.stop()
                    
                    strategy = result.get("strategy")
                    stacking_params = result.get("stacking_parameters")
                    optimizer_params_changed_only = bool(st.session_state.get('_optimizer_params_changed_only', False))
                

                    st.markdown("### Step 2: Match Results & Parameters")
                    render_stacking_params_card(strategy, stacking_params)

                    # ── Step 3 ──────────────────────────────────────────
                    if strategy in ["exact_match", "similarity_match", "formula_exact_match",
                                    "formula_multi_match"]:
                        st.markdown("### Step 3: Stack Generation")
                        st.markdown("""
                        <div class="build-section">
                            <h4>🧩 Ready to Build Stack</h4>
                            <p>Click below to generate your supramolecular stack.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info(f"📏 **Stack Configuration:** {stack_size} molecules will be assembled using optimized parameters")
                        if st.button("🚀 Build Stack with Database Parameters", type="primary"):
                            xyz_for_build = st.session_state.get('xyz_content_for_build')
                            if not xyz_for_build:
                                st.error("❌ No XYZ content available for building. Please run 'Check Database & Prepare Build' first.")
                                st.stop()
                            with st.spinner(f"🔧 Building {stack_size}-molecule stack..."):
                                 logger.info("calling the build stack here")
                                 logger.info(strategy)
                                 logger.info(stacking_params)
                                 build_result = build_stack(xyz_for_build, stacking_params, stack_size,strategy)
                                 if build_result["success"]:
                                 # Update ONLY stack_size in settings, keep similarity_threshold unchanged
                                    if 'stack_built_settings' not in st.session_state:
                                        st.session_state['stack_built_settings'] = {}
                                    st.session_state['stack_built_settings']['stack_size'] = int(stack_size)
                                        
                                    st.session_state['last_stack_result'] = build_result
                                    st.session_state['last_stack_strategy'] = strategy
                                    st.session_state['last_stack_params'] = stacking_params
                                    st.session_state['last_stack_size'] = stack_size
                                    st.session_state['last_stack_smiles'] = smiles
                                    st.rerun()
                                 else:
                                    st.error(f"❌ Stack building failed: {build_result.get('error', 'Unknown error')}")

                        # Persist results across reruns
                        if st.session_state.get('last_stack_result') and st.session_state.get('stack_built_settings'):
                            _result = st.session_state['last_stack_result']
                            _strategy = st.session_state.get('last_stack_strategy', strategy)
                            _params = st.session_state.get('last_stack_params', stacking_params)
                            _stack_size = st.session_state.get('last_stack_size', stack_size)
                            _smiles = st.session_state.get('last_stack_smiles', smiles)
                            _smiles_source = st.session_state.get('smiles_source', 'none')

                            st.success("✅ **Stack Generated Successfully!**")
                        
                            stack_xyz = _result["stack_xyz"]
                            st.markdown('<h3><span style="color:#5C6BC0;">⌬</span> 3D Supramolecular Stack</h3>', unsafe_allow_html=True)
                            show_xyz_3d(stack_xyz)
                            if stack_xyz:
                                lines = stack_xyz.split('\n')
                                total_atoms = lines[0] if lines else "0"
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("⚛️ Total Atoms", total_atoms)
                                with col2:
                                    st.metric("🔗 Molecules", _stack_size)
                                with col3:
                                    energy_value = get_energy_value(_params)
                                    if energy_value is None:
                                        st.metric("⚡ Interaction Energy", "N/A")
                                    else:
                                        st.metric("⚡ Interaction Energy", f"{energy_value:.1f} kJ/mol")


                                monomer_xyz = st.session_state.get('xyz_content_for_build', '')
                                if monomer_xyz:
                                    
                                    from utils.pi_stack_analysis_simpler import check_interlayer_catastrophic_overlap,_parse_xyz_block
                                    
                                    analysis = analyze_pi_stack(
                                        stack_xyz=stack_xyz,
                                        monomer_xyz=monomer_xyz,
                                        smiles=_smiles,
                                        smiles_source=_smiles_source,
                                    )
                                    _, stack_coords = _parse_xyz_block(stack_xyz)
                                    _, mono_coords = _parse_xyz_block(monomer_xyz)
                                    n_mono = len(mono_coords)
                                    n_stack = len(stack_coords) // n_mono

                                    if "error" in analysis:
                                        st.caption(f"ℹ️ π-stack analysis: {analysis['error']}")
                                        st.stop()

                                    # Use existing functions for clash detection
                                    render_clash_comparison(analysis)
                                    has_clash = bool(analysis.get("has_steric_clash", False))
                                    overlap = analysis.get("mean_overlap_percent", 0)
                                    quality = analysis.get("overall_quality", "Unknown")

                                    if has_clash:
                                       st.warning("⚠️ π-stack metrics may be unreliable because steric clashes.")
                                    render_pi_stack_analysis(
                                        analysis
                                        #monomer_xyz,
                                        #_smiles,
                                        #smiles_source=_smiles_source,
                                    )

                                    poor_stack = (quality in ["Poor", "Invalid Geometry"] or has_clash or overlap < 40)
                                    
                                    if poor_stack:
                                        if has_clash:
                                            st.markdown("""
                                            <div style="
                                                background: linear-gradient(135deg,#ff9a9e,#fad0c4);
                                                padding:20px;
                                                border-radius:12px;
                                                color:#4a0000;
                                                margin-bottom:20px;
                                                box-shadow:0 4px 10px rgba(0,0,0,0.08);
                                            ">
                                            <h4 style="margin-top:0;">⚠️ Invalid Stack Geometry Detected</h4>

                                            Severe **steric clashes** were detected between atoms in adjacent layers. This geometry is physically unrealistic.
                                            Advanced Optimization is required to find a better geometry.

                                            To obtain a better configuration, the **π-Stack Optimizer** should be used to search for a lower-energy geometry.
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:

                                            st.markdown("""
                                                <div style="
                                                    background: linear-gradient(135deg,#ff9a9e,#fad0c4);
                                                    padding:20px;
                                                    border-radius:12px;
                                                    color:#4a0000;
                                                    margin-bottom:20px;
                                                    box-shadow:0 4px 10px rgba(0,0,0,0.08);
                                                ">
                                                <h4 style="margin-top:0;">⚠️ Suboptimal Stack Geometry Detected</h4>

                                                The generated stack shows **poor π–π overlap** or non-ideal stacking parameters..
                                                Advanced Optimization is required to find a better geometry.

                                                To obtain a better configuration, the **π-Stack Optimizer** should be used to search for a lower-energy geometry.
                                                </div>
                                                """, unsafe_allow_html=True)
                                    
                                        #st.info("""**What you'll get:**\n- Complete π-stack optimizer toolkit\n- Pre-configured for your molecule\n- Ready-to-run scripts for all platforms\n- Comprehensive documentation""")
                                        st.markdown("""
                                        <div style="
                                            background:#f8f9ff;
                                            padding:18px;
                                            border-radius:10px;
                                            border-left:5px solid #6c63ff;
                                            margin-bottom:20px;
                                        ">
                                        <b>📦 What you'll get in the optimization package:</b>

                                        • Complete π-stack optimizer toolkit  
                                        • Pre-configured input for your molecule  
                                        • Ready-to-run scripts (Linux / Mac / Windows)  
                                        • Step-by-step documentation  
                                        </div>
                                        """, unsafe_allow_html=True)

                                        if st.session_state.get('_optimizer_params_changed_only', False):
                                            st.info("⚙️ The optimizer settings were modified. Please regenerate the optimization package using the button below.")
                                   
                                        if st.button("📦 Generate & Download Optimization Package", type="primary",key="create_optimizer_pkg"):

                                            xyz_for_build = st.session_state.get('xyz_content_for_build')
                                            if not xyz_for_build:
                                                st.error("❌ No XYZ content available. Please run 'Check Database & Prepare Build' first.")
                                                st.stop()
                                            with st.spinner("📦 Creating comprehensive optimization package..."):
                                                try:
                                                    bundle_bytes = create_optimizer_bundle(
                                                        smiles=smiles, xyz_content=xyz_for_build, stack_size=stack_size,
                                                        opt_method=opt_method, optimizer_params=collect_optimizer_parameters_from_ui(opt_method)
                                                        )

                                                    if 'stack_built_settings' not in st.session_state:
                                                        st.session_state['stack_built_settings'] = {}
                                                    st.session_state['stack_built_settings']['stack_size'] = int(stack_size)
                                                    st.session_state['stack_built_settings']['optimizer_fingerprint'] = current_optimizer_fingerprint

                                                    st.session_state['optimizer_bundle_bytes'] = bundle_bytes
                                                    st.session_state['_optimizer_params_changed_only'] = False
                                                    st.session_state['_auto_dl_optimizer'] = True
                                                    st.rerun()
                                                except Exception as e:
                                                        st.error(f"❌ Package creation failed: {str(e)}")

                                        if st.session_state.get('optimizer_bundle_bytes'):

                                            st.markdown("""
                                            <div style="
                                                background:#e8f7ee;
                                                padding:20px;
                                                border-radius:12px;
                                                border-left:6px solid #2ecc71;
                                                margin-top:15px;
                                            ">
                                            <h4 style="margin-top:0;">🎉 Optimization Package created successfully!</h4>

                                            <b>Next steps:</b>

                                            1️⃣ Extract the downloaded package on your machine
                                            2️⃣ Install the xTB quantum chemistry package
                                            3️⃣ Run the provided scripts

                                            Results will automatically be saved in the output directory.
                                            </div>
                                            """, unsafe_allow_html=True)

                                            render_optimizer_package_download(
                                                st.session_state['optimizer_bundle_bytes'],
                                                f"pi_stack_optimizer_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                                '_auto_dl_optimizer',
                                            )


                                    with st.expander("🔬 View Stack XYZ Coordinates", expanded=False):
                                            st.text_area("Stack XYZ Content", stack_xyz, height=300)

                                    bundle_files = _result.get("bundle_files", {})
                                    package_energy_value = get_energy_value(_params)
                                    package_energy_text = (
                                        f"{package_energy_value:.2f}"
                                        if package_energy_value is not None
                                        else "N/A"
                                    )
                                    bundle_files.update({'parameters.txt': f"""Supramolecular Stack Parameters
            =====================================
            Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            Strategy: {_strategy.replace('_', ' ').title()}
            Molecule: {_smiles}
            Stack Size: {_stack_size} molecules
            Stacking Parameters:
            - Binding Energy: {package_energy_text} kJ/mol
            - Stacking Distance (Tz): {_params.get('Tz', 3.4):.2f} Å
            - Rotation Angle (θ): {_params.get('Theta', 0):.1f}°
            - Translation X (Tx): {_params.get('Tx', 0):.2f} Å
            - Translation Y (Ty): {_params.get('Ty', 0):.2f} Å
            - Center Offset X (Cx): {_params.get('Cx', 0):.2f} Å
            - Center Offset Y (Cy): {_params.get('Cy', 0):.2f} Å
            - Full Torsion Angles (°): {fmt_torsions(_params)}
            Quality: {'Highest (exact database match)' if _strategy == 'exact_match' else 'Good (similarity match)' if _strategy == 'similarity_match' else 'Good (exact formula match)' if _strategy == 'formula_exact_match' else 'Good (multiple formula matches)' if _strategy == 'formula_multi_match' else 'Custom optimization'}
            """})
                                    zip_data = create_zip_bundle(bundle_files)
                                    st.download_button(
                                            label="📦 Download Complete Results Package",
                                            data=zip_data,
                                            file_name=f"supramolecular_stack_{_stack_size}mol_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                            mime="application/zip",
                                        )


                    elif strategy == "pi-stack-optimizer":
                       

                        next_step = st.radio(
                            "**Select a follow-up strategy for database expansion or molecule-specific stack optimization:**",
                            ["🔍 Search Again", "⚙ Download π-Stack Optimizer"],
                            horizontal=True,
                            key="nomatch_next_step",
                        )

                        if next_step == "🔍 Search Again":
                            with st.container(border=True):
                                st.markdown("""
                                <div class="optimizer-action-item-selected">
                                    <p class="optimizer-action-title"></p>
                                    <p class="optimizer-action-text">
                                    <p style="
                                        font-size:15px;
                                        font-weight:500;
                                        margin-bottom:8px;
                                    ">
                                        Lower the <b>Similarity threshold</b> in the sidebar, then run database analysis again so more distant molecules can match.
                                    </p>
                                    <p class="optimizer-action-note">
                                        Best when you want to keep using database-derived stacking parameters.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                                

                        else:
                            with st.container(border=True):
                                st.markdown("""
                                <div class="optimizer-action-item-selected">
                                    <p class="optimizer-action-title"></p>
                                    <p class="optimizer-action-text">
                                    <p style="
                                        font-size:15px;
                                        font-weight:500;
                                        margin-bottom:8px;
                                    ">
                                        Download a custom optimizer package, pre-configured for your molecule, to find the best stacking geometry from scratch.
                                    <p class="optimizer-action-note">
                                        Use this when the molecule is not found at the selected threshold and you want a molecule-specific optimization workflow.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                                if st.button(
                                    "📦 Generate & Download Optimizer Package",
                                    type="primary",
                                    use_container_width=True,
                                    key="create_nomatch_optimizer_pkg",
                                    help="Create a π-stack optimizer package with the current sidebar settings.",
                                ):
                                    xyz_for_build = (
                                        st.session_state.get("xyz_content_for_build")
                                        or result.get("xyz_content")
                                        or st.session_state.get("xyz_content")
                                    )
                                    if not xyz_for_build:
                                        st.error("❌ No XYZ content available. Please run database analysis again.")
                                        st.stop()

                                    with st.spinner("📦 Creating π-stack optimizer package from current settings..."):
                                        try:
                                            bundle_bytes = create_optimizer_bundle(
                                                smiles=smiles,
                                                xyz_content=xyz_for_build,
                                                stack_size=stack_size,
                                                opt_method=opt_method,
                                                optimizer_params=collect_optimizer_parameters_from_ui(opt_method),
                                            )
                                            if "stack_built_settings" not in st.session_state:
                                                st.session_state["stack_built_settings"] = {}
                                            st.session_state["stack_built_settings"]["similarity_threshold"] = round(float(similarity_threshold), 4)
                                            st.session_state["stack_built_settings"]["stack_size"] = int(stack_size)
                                            st.session_state["stack_built_settings"]["opt_method"] = opt_method
                                            st.session_state["stack_built_settings"]["max_iters"] = max_iters
                                            st.session_state["stack_built_settings"]["optimizer_fingerprint"] = current_optimizer_fingerprint
                                            st.session_state["optimizer_bundle_bytes"] = bundle_bytes
                                            st.session_state["_optimizer_params_changed_only"] = False
                                            st.session_state["_auto_dl_nomatch_optimizer"] = True
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Package creation failed: {str(e)}")

                        if next_step == "⚙ Download π-Stack Optimizer" and st.session_state.get("optimizer_bundle_bytes"):
                           

                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                            
                                 render_optimizer_package_download(
                                st.session_state["optimizer_bundle_bytes"],
                                f"pi_stack_optimizer_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                "_auto_dl_nomatch_optimizer",
                                show_button=False,   # hide manual button
                                 )
                            
                else:
                   st.info("""
                    📋 **Database Search Workflow**

                    The submitted molecular structure is compared against the reference database to identify exact or structurally similar molecules and retrieve stacking parameters for supramolecular assembly generation.

                    **Possible outcomes:**

                    - ✅ **Exact Match Found** — Database-derived stacking parameters are retrieved and can be used directly for stack construction.
                    - 🔍 **Similar Molecule Found** — Parameters from the closest matching structure are transferred to guide stack construction.
                    - ⚙️ **No Match Found** — No suitable database entry was identified. The similarity threshold may be adjusted, or a custom π-stack optimization workflow can be used.
                    """)

        # Reset button at bottom of tab2 — only show AFTER database check has been performed
        if st.session_state.get('db_check_result') is not None:
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🆕 Start New Analysis",
                            help="Clear all data and start fresh with a new molecule",
                            type="primary",
                            key="reset_button_tab2"):
                    clear_molecule_session()
                    st.rerun()

    # Footer
    

if __name__ == "__main__":
    main()
