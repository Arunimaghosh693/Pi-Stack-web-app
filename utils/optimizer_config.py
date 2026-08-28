
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

def build_optimizer_parameters(opt_method: str, ui_state: dict)-> Dict[str, Any]:

    """
    Collect optimizer parameters from the Streamlit UI based on the selected method.
    Returns a dictionary with only the relevant parameters for the chosen method.
    """
    params = {}
    
    # General parameters (always collected)
    params.update({
        "max_iters": st.session_state.get('max_iters', 300),
        "seed": st.session_state.get('seed', 42),
        "workers": st.session_state.get('workers', 4),
        "threads": st.session_state.get('threads', 1),
        "xtb_method": st.session_state.get('xtb_method', 'gfn2'),
        "charge": st.session_state.get('charge', 0),
        "mult": st.session_state.get('mult', 1),
        "penalty_weight": st.session_state.get('penalty_weight', 2.0),
        "clash_cutoff": st.session_state.get('clash_cutoff', 1.6),
        "intramol_penalty_weight": st.session_state.get('intramol_penalty_weight', 5.0),
        "intramol_cutoff": st.session_state.get('intramol_cutoff', 1.2),
    })
    
    # Method-specific parameters
    if opt_method == "pso":
        params.update({
            "swarm_size": st.session_state.get('swarm_size', 60),
            "inertia": st.session_state.get('inertia', 0.73),
            "cognitive": st.session_state.get('cognitive', 1.50),
            "social": st.session_state.get('social', 1.50),
            "pso_tol": st.session_state.get('pso_tol', 0.01),
            "patience": st.session_state.get('patience', 20),
        })
    elif opt_method == "ga":
        params.update({
            "ga_population": st.session_state.get('ga_population', 80),
            "ga_mutation_rate": st.session_state.get('ga_mutation_rate', 0.10),
            "ga_mutation_sigma": st.session_state.get('ga_mutation_sigma', 0.30),
            "ga_crossover_rate": st.session_state.get('ga_crossover_rate', 0.90),
            "ga_elite_fraction": st.session_state.get('ga_elite_fraction', 0.10),
            "ga_tournament_size": st.session_state.get('ga_tournament_size', 3),
            "ga_tol": st.session_state.get('ga_tol', 0.01),
            "ga_patience": st.session_state.get('ga_patience', 20),
        })
    elif opt_method == "gwo":
        params.update({
            "gwo_pack_size": st.session_state.get('gwo_pack_size', 50),
            "gwo_a_start": st.session_state.get('gwo_a_start', 2.0),
            "gwo_a_end": st.session_state.get('gwo_a_end', 0.0),
            "gwo_tol": st.session_state.get('gwo_tol', 0.01),
            "gwo_patience": st.session_state.get('gwo_patience', 20),
        })
    elif opt_method == "pso-nm":
        params.update({
            "swarm_size": st.session_state.get('swarm_size', 60),
            "inertia": st.session_state.get('inertia', 0.73),
            "cognitive": st.session_state.get('cognitive', 1.50),
            "social": st.session_state.get('social', 1.50),
            "pso_tol": st.session_state.get('pso_tol', 0.01),
            "patience": st.session_state.get('patience', 20),
            "nm_max_iters": st.session_state.get('nm_max_iters', 200),
            "nm_initial_step": st.session_state.get('nm_initial_step', 0.20),
            "nm_alpha": st.session_state.get('nm_alpha', 1.0),
            "nm_gamma": st.session_state.get('nm_gamma', 2.0),
            "nm_rho": st.session_state.get('nm_rho', 0.50),
            "nm_sigma": st.session_state.get('nm_sigma', 0.50),
        })
    
    return params