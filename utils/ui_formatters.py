"""
UI Formatting Utilities
========================
Streamlit UI components for displaying analysis results.
"""

import streamlit as st
import json
import re
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# HELPER FUNCTIONS FOR TORSIONS
# ============================================================

def fmt_torsions(params):
    import ast
    raw = params.get('Full_torsion_angles', None)
    if raw is None:
        return 'N/A'
    if isinstance(raw, str):
        try:
            angles = [float(v) for v in ast.literal_eval(raw)]
        except Exception:
            return raw
    elif isinstance(raw, (list, tuple)):
        angles = [float(v) for v in raw]
    else:
        return 'N/A'
    return ", ".join(f"{a:.3f}" for a in angles)


def fmt_torsions_html(params):
    """Return the full param-item div for Full Torsion Angles with dynamic grid span."""
    import ast
    raw = params.get('Full_torsion_angles', None)
    if raw is None:
        angles = []
        display = 'N/A'
    elif isinstance(raw, str):
        try:
            angles = [float(v) for v in ast.literal_eval(raw)]
            display = ", ".join(f"{a:.3f}" for a in angles)
        except Exception:
            angles = []
            display = raw
    elif isinstance(raw, (list, tuple)):
        angles = [float(v) for v in raw]
        display = ", ".join(f"{a:.3f}" for a in angles)
    else:
        angles = []
        display = 'N/A'

    n = len(angles)
    if n <= 1:
        col_style = ""
    elif n == 2:
        col_style = "grid-column: span 2;"
    else:
        col_style = "grid-column: span 3;"

    return (
        f'<div class="param-item" style="background:rgba(255,255,255,0.1);padding:10px;'
        f'border-radius:8px;text-align:center;backdrop-filter:blur(10px);{col_style}">'
        f'<div class="param-label">Full Torsion Angles (°)</div>'
        f'<div class="param-value" style="font-size:1.05em;font-weight:900;word-break:break-word;white-space:normal;">'
        f'{display}</div></div>'
    )

def format_formula(formula: str) -> str:
    """Formats a chemical formula string with HTML subscripts for display."""
    return re.sub(r'(\d+)', r'<sub>\1</sub>', formula)

# ============================================================
# QUALITY BADGE
# ============================================================

def show_quality_badge(quality: str):
    """Show quality badge with publishability guidance."""

    badge_config = {
        "Excellent": {
            "color": "#16a34a",
            "emoji": "✅",
            "note": "Accepted"
        },
        "Good": {
            "color": "#84cc16",
            "emoji": "✓",
            "note": "Acceptable with caveats"
        },
        "Fair": {
            "color": "#eab308",
            "emoji": "⚠️",
            "note": "Re-optimization recommended"
        },
        "Poor": {
            "color": "#f97316",
            "emoji": "⚠️",
            "note": "Invalid geometry"
        },
        "Invalid Geometry": {
            "color": "#dc2626",
            "emoji": "❌",
            "note": "NOT SUITABLE - Fix required"
        }
    }

    config = badge_config.get(quality, badge_config["Poor"])

    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 8px;
        background: {config['color']}20;
        border: 2px solid {config['color']};
        margin-bottom: 1rem;
    ">
        <div style="font-size: 1.5em;">{config['emoji']}</div>
        <div>
            <div style="font-weight: 700; font-size: 1.1em; color: {config['color']};">
                {quality}
            </div>
            <div style="font-size: 0.9em; color: #666;">
                {config['note']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# METRIC CARD
# ============================================================

def show_metric_card(label: str, value: str, help_text: str = "", delta_color: str = "normal") -> None:
    """Display a styled metric card."""
    st.metric(label=label, value=value, help=help_text, delta_color=delta_color)



# ============================================================
# CLASH COMPARISON RENDERER
# ============================================================

def render_clash_comparison(stack_analysis) -> None:

    st.subheader("⚡ Steric Clash Analysis")

    has_clash_status = stack_analysis["has_steric_clash"]


    if has_clash_status:
        st.error("❌ Catastrophic steric clash detected. Stack geometry invalid.")

    else:
        #st.success("Stack passes steric clash checks. No catastrophic overlaps detected.")
        st.success("✅ No catastrophic steric clashes detected.")

    if not stack_analysis:
        return

    # -------------------------
    # Prepare DataFrame
    # -------------------------
    pair_data = []

    for p in stack_analysis.get("pairs", []):
        logger.info(f"checking P is {p}")

        pair_id = p.get("pair", "Unknown")
        min_dist = p.get("min_distance_ang", None)
        has_clash = p.get("has_steric_clash", False)

        pair_data.append({
        "Layer Pair": pair_id,
        "Min Distance (Å)": min_dist,
        "Very Close (<1.5Å)": p.get("n_very_close", 0),
        "Close Contacts (<2Å)": p.get("n_close", 0),
        "Status": "🔴 Clash" if has_clash else "🟢 OK",
        "Type": "Adjacent"
        })
    # ------------------------------------------------
    # Non-adjacent clashes (mol1–mol3 etc.)
    # ------------------------------------------------

    for pair in stack_analysis.get("non_adjacent_clashes", []):

        pair_data.append({
            "Layer Pair": pair,
            "Min Distance (Å)": stack_analysis.get("min_interlayer_distance_ang"),
            "Very Close (<1.5Å)": 1,
            "Close Contacts (<2Å)": 1,
            "Status": "🔴 Clash",
            "Type": "Non-Adjacent"
        })

    df = pd.DataFrame(pair_data)
    df = df.sort_values("Min Distance (Å)")
      # -------------------------
      # Collapsible detailed analysis
      # -------------------------
      #expanded = overall_cat  # auto open if clashes exist
    expanded = False

    with st.expander("🔬 Detailed Clash Analysis", expanded=expanded):

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            #height=220
        )
        non_adj = stack_analysis.get("non_adjacent_clashes", [])
        if not non_adj:
          n_layers = stack_analysis.get("n_layers", None)

          if n_layers and n_layers > 2:
           total_pairs = n_layers * (n_layers - 1) // 2
           adjacent_pairs = n_layers - 1
           non_adj_pairs = total_pairs - adjacent_pairs

           st.info(f"✔ Non-adjacent layer check passed ({non_adj_pairs} pairs tested)")
        # -------------------------
        # Intramolecular Clash Warnings
        # -------------------------
        intramol_warnings = stack_analysis.get("intramol_warnings", [])

        if intramol_warnings:
            st.markdown("### ⚛️ Intramolecular Geometry Warnings")

            for warn in intramol_warnings:
                st.warning(warn)
        else:
            st.markdown("### ⚛️ Intramolecular Geometry Check")

            st.success("Monomer geometries appear physically reasonable.\n No intramolecular steric clashes detected.")
    return has_clash_status



# ============================================================
# ANALYSIS RESULTS FORMATTER (Updated)
# ============================================================

def format_analysis_results(simple_catastrophic: Optional[List[Dict[str, Any]]] = None,
                           analysis: Optional[Dict[str, Any]] = None) -> str:
    """Format analysis results as human-readable text with simplified clash detection."""
    lines = []

    lines.append("=" * 80)
    lines.append("π-STACK QUALITY ASSURANCE REPORT (Post-Build)")
    lines.append("=" * 80)
    lines.append("")

    # ========================================
    # CLASH DETECTION (PRIMARY)
    # ========================================

    if simple_catastrophic:
        lines.append("STERIC CLASH ANALYSIS")
        lines.append("-" * 80)

        overall_cat = any(r.get("has_catastrophic", False) for r in simple_catastrophic)
        lines.append(f"Overall Status:               {'❌ CATASTROPHIC' if overall_cat else '✅ NO CATASTROPHIC OVERLAP'}")
        lines.append("")

        min_dist = min(r.get("details", {}).get("min_distance", float('inf')) for r in simple_catastrophic)
        lines.append(f"Min Inter-atomic Distance:    {min_dist:.2f}Å")

        total_very_close = sum(r.get("details", {}).get("n_very_close", 0) for r in simple_catastrophic)
        total_close = sum(r.get("details", {}).get("n_close", 0) for r in simple_catastrophic)
        lines.append(f"Total Contacts <1.5Å:         {total_very_close} (critical threshold)")
        lines.append(f"Total Contacts <2.0Å:         {total_close}")
        lines.append("")



        lines.append("Per-Pair Clash Comparison:")
        lines.append("-" * 80)
        lines.append(f"{'Pair':<15} {'Status':<15} {'Min Dist (Å)':<15} {'<1.5Å':<8} {'<2.0Å':<8} {'π-Overlap (%)':<15}")
        lines.append("-" * 80)

        for result in simple_catastrophic:
            pair_id = result.get("pair", "Unknown")
            has_cat = result.get("has_catastrophic", False)
            status = "❌ Catastrophic" if has_cat else "✅ OK"
            details = result.get("details", {})

            lines.append(
                f"{pair_id:<15} "
                f"{status:<15} "
                f"{details.get('min_distance', 0):>13.2f}Å "
                f"{details.get('n_very_close', 0):>6} "
                f"{details.get('n_close', 0):>6} "
                f"{details.get('pi_overlap_percent', 0):>13.1f}%"
            )

        lines.append("")


    # ========================================
    # π-OVERLAP INFORMATION (if analysis provided)
    # ========================================

    if analysis:
        lines.append("π-STACK OVERLAP ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Mean π-π Overlap:             {analysis.get('mean_overlap_percent', 'N/A')}%")
        lines.append(f"Mean Stacking Distance:       {analysis.get('mean_stacking_distance_ang', 'N/A')}Å")
        lines.append(f"Lateral Slip:           {analysis.get('mean_slip_distance_ang', 'N/A')}Å")
        lines.append(f"π-Core Area:                  {analysis.get('pi_core_area_ang2', 'N/A')}Ų")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)

# ============================================================
# DOWNLOAD SECTION (Updated)
# ============================================================

def create_download_section(stack_xyz: str,
                           simple_catastrophic: Optional[List[Dict[str, Any]]] = None,
                           analysis: Optional[Dict[str, Any]] = None,
                           monomer_xyz: Optional[str] = None,
                           smiles: Optional[str] = None) -> None:
    """Create a download section with QA report and simplified clash analysis."""
    st.subheader("📥 Download Results")

    st.download_button(
        label="📄 Download Stack (XYZ)",
        data=stack_xyz,
        file_name="stack_optimized.xyz",
        mime="text/plain",
        use_container_width=True
    )

    report_txt = format_analysis_results(simple_catastrophic, analysis)
    st.download_button(
        label="📊 Download QA Report (TXT)",
        data=report_txt,
        file_name="qa_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    if simple_catastrophic:
        qa_json = json.dumps(simple_catastrophic, indent=2)
        st.download_button(
            label="📊 Download Clash Analysis (JSON)",
            data=qa_json,
            file_name="clash_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    if analysis:
        analysis_json = json.dumps(analysis, indent=2)
        st.download_button(
            label="📊 Download π-Overlap Analysis (JSON)",
            data=analysis_json,
            file_name="overlap_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    if monomer_xyz:
        st.download_button(
            label="🧪 Download Monomer (XYZ)",
            data=monomer_xyz,
            file_name="monomer.xyz",
            mime="text/plain",
            use_container_width=True
        )

    if smiles:
        st.download_button(
            label="🔤 Download SMILES",
            data=smiles,
            file_name="monomer.smi",
            mime="text/plain",
            use_container_width=True
        )
# ============================================================
# STACKING PARAMETER CARD RENDERER
# ============================================================

def render_stacking_params_card(strategy: str, stacking_params: dict) -> None:

    """Render the Step-2 match-result banner and parameter card."""
    #st.info(f"the recevied strategy is {strategy}")
    #st.info(f"the recevied stacking_params is {stacking_params}")
    if strategy == "exact_match":
        st.markdown(
            '<div class="status-card status-exact">'
            '<h4>🎯 Exact Database Match Found!</h4>'
            '<p>Highest confidence parameters available - ready to build stack</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="param-card"><h3>📊 Optimized Stacking Parameters</h3>
            <div class="param-grid">
              <div class="param-item"><div class="param-label">Formula</div><div class="param-value">{formula}</div></div>
              <div class="param-item"><div class="param-label">Translation X</div><div class="param-value">{tx:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Translation Y</div><div class="param-value">{ty:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Stacking Distance (Tz)</div><div class="param-value">{tz:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Center Offset X</div><div class="param-value">{cx:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Center Offset Y</div><div class="param-value">{cy:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Rotation (θ)</div><div class="param-value">{theta:.1f}°</div></div>
              {torsions}
              <div class="param-item"><div class="param-label">Interaction Energy</div><div class="param-value">{be:.1f} kJ/mol</div></div>
            </div></div>""".format(
                formula=stacking_params.get('MolFormula', 'N/A').split("_")[0],
                tx=stacking_params.get('Tx', 0), 
                ty=stacking_params.get('Ty', 0),
                tz=stacking_params.get('Tz', 3.4),
                cx=stacking_params.get('Cx', 0), cy=stacking_params.get('Cy', 0),
                torsions=fmt_torsions_html(stacking_params),
                be=stacking_params.get('interaction_energy', 0),
                theta=stacking_params.get('Theta', 0),
            ),
            unsafe_allow_html=True,
        )

    elif strategy == "similarity_match":
        sim = stacking_params.get("similarity_score", 0)
        if sim >= 0.90:
            conf = "🟢 High similarity"
        elif sim >= 0.75:
            conf = "🟡 Moderate similarity"
        elif sim >= 0.60:
            conf = "🟠 Low similarity"
        else:
            conf = "🔴 Weak similarity"
        st.markdown(
            f"""<div class="param-card">
            <h4>🔍 Similar Molecule Match Found</h4>
            <p>Similarity: {sim:.1%} — {conf}<br>
            ⚠️ Similarity-based matches do NOT guarantee physically optimal stacking.
            Electronic or steric differences may produce metastable or incorrect arrangements.</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """<div class="param-card"><h3>📊 Similar Molecule Parameters</h3>
            <div class="param-grid">
              <div class="param-item"><div class="param-label">Similar Molecule Formula</div><div class="param-value">{formula}</div></div>
              <div class="param-item"><div class="param-label">Translation X</div><div class="param-value">{tx:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Translation Y</div><div class="param-value">{ty:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Stacking Distance (Tz)</div><div class="param-value">{tz:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Center Offset X</div><div class="param-value">{cx:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Center Offset Y</div><div class="param-value">{cy:.2f} Å</div></div>
              <div class="param-item"><div class="param-label">Rotation (θ)</div><div class="param-value">{theta:.1f}°</div></div>
              {torsions}
              <div class="param-item"><div class="param-label">Interaction Energy</div><div class="param-value">{be:.1f} kJ/mol</div></div>
              <div class="param-item"><div class="param-label">Similarity Score</div><div class="param-value">{sim:.1%}</div></div>
            </div></div>""".format(
                formula=stacking_params.get('MolFormula', 'N/A'),
                tx=stacking_params.get('Tx', 0), ty=stacking_params.get('Ty', 0),
                tz=stacking_params.get('Tz', 3.4),
                cx=stacking_params.get('Cx', 0), cy=stacking_params.get('Cy', 0),
                theta=stacking_params.get('Theta', 0),
                torsions=fmt_torsions_html(stacking_params),
                be=stacking_params.get('interaction_energy', 0), sim=sim,
            ),
            unsafe_allow_html=True,
        )

    elif strategy in ["formula_exact_match", "formula_multi_match"]:
        match_type = "Exact" if strategy == "formula_exact_match" else "Multiple"
        st.markdown(
            f'<div class="status-card status-exact">'
            f'<h4>🧪 {match_type} Formula Match Found!</h4>'
            f'<p>Good confidence parameters available</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif strategy == "pi-stack-optimizer":
        st.markdown(
            '<div class="param-card">'
            '<h4>⚙️ No Similar Molecule Found</h4>'
            '<p>No structurally similar molecule was identified within the current similarity threshold</p>'
            '</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# π-STACK QUALITY ANALYSIS DISPLAY
# ============================================================

def render_pi_stack_analysis(stack_analysis) -> None:
    """Render a precomputed pi-stack analysis dict."""

    try:
        if 'error' in stack_analysis:
            st.caption(f"ℹ️ π-stack analysis: {stack_analysis['error']}")
            return


        needs_attention = stack_analysis.get('stack_needs_attention', False)



        # Key Metrics Row
        #st.markdown("#### 🔬 π-Stack Quality Metrics")
        #with st.expander("#### 🔬 π-Stack Quality Metric", expanded=False):
        with st.expander("# 🔬 π-Stack Structural Analysis", expanded=False):
        #with st.expander("🔬 π-Stack Structural Analysis", expanded=False):

            st.markdown("### 📊 π-Stack Metrics")

            c1, c2, _ = st.columns([1,1,2])

            with c1:
                st.metric(
                    "Mean π-π overlap",
                    f"{stack_analysis['mean_overlap_percent']:.1f} %",
                    help=(
                    "Projected π-core coverage between adjacent layers. "
                    "Computed as overlap area divided by the smaller π-core area. "
                    "Values >70–80% generally indicate good π-stacking."
                )
                )

            with c2:
                st.metric(
                    "π-core atoms",
                    stack_analysis.get('n_pi_core_atoms', 0),
                    help=f"Detection method: {stack_analysis.get('pi_detection_method', 'unknown')}"
                )

            st.markdown("### 📐 Geometric Parameters")

            g1, g2, g3, g4 = st.columns(4)

            with g1:
                st.metric(
                    "Stacking distance",
                    f"{stack_analysis['mean_stacking_distance_ang']:.2f} Å",
                    help="Ideal π-π stacking distance: 3.0–3.8 Å"
                )

            with g2:
                st.metric(
                    "Lateral slip",
                    f"{stack_analysis['mean_slip_distance_ang']:.2f} Å",
                    help="In-plane displacement between layers"
                )

            with g3:
                min_dist = stack_analysis.get('min_interlayer_distance_ang', float('inf'))
                if min_dist < 1.5:
                  st.markdown(
                     f"""
                     <div>
                       <div style="font-size:0.8rem;color:gray;">Minimum distance</div>
                       <div style="font-size:1.8rem;font-weight:700;color:#dc2626;">
                           {min_dist:.2f} Å
                       </div>
                     </div>
                     """,
                    unsafe_allow_html=True,
                    help="Recommended: minimum interlayer distance > 1.5 Å"
                    )
                else:
                  st.metric("Minimum distance",f"{min_dist:.2f} Å",
                            help="Recommended: minimum interlayer distance > 1.5 Å")
            with g4:
                st.metric(
                    "Core area",
                    f"{stack_analysis.get('pi_core_area_ang2', 0):.1f} Å²",
                    help="Convex hull area of projected π-core"
                )
            st.divider()


            '''
            with info3:
                min_dist = stack_analysis.get('min_interlayer_distance_ang', float('inf'))
                clash_icon = "❌" if stack_analysis.get('has_steric_clash') else "✅"
                st.metric(
                    f"{clash_icon} Min Distance",
                    f"{min_dist:.2f}Å" if min_dist < 999 else "N/A",
                    help="Closest atomic contact (0.80 threshold for π-stacks)"
                )
            '''



    except Exception as exc:
        st.caption(f"ℹ️ π-stack analysis unavailable: {exc}")
