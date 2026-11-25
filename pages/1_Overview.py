"""
Page 1: System Overview & Telemetry Baseline.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GAS_CLASSES, SENSOR_TYPES
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_dataset,
    render_sidebar_info
)
from src.exploratory_analysis import get_dataset_overview_stats, get_batch_class_matrix
from src.visualization import (
    plot_gas_distribution,
    plot_batch_measurements,
    plot_batch_gas_stacked
)

st.set_page_config(page_title="System Overview | Sensor Drift Intelligence", page_icon="📊", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("📊 System Overview & Telemetry Baseline")
st.markdown("Detailed dataset architecture, gas analyte distributions, and chronological batch profiles.")
st.markdown("---")

df = get_cached_dataset()
stats = get_dataset_overview_stats(df)

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Measurements", f"{stats['total_records']:,}")
with kpi2:
    st.metric("Chemical Sensors", f"{stats['total_sensors']}")
with kpi3:
    st.metric("Feature Dimensions", f"{stats['total_features']}")
with kpi4:
    st.metric("Chronological Batches", f"{stats['total_batches']}")
with kpi5:
    st.metric("Analyte Classes", f"{len(stats['gas_classes'])}")

st.markdown("---")

# Visualizations Row 1
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(plot_gas_distribution(df), use_container_width=True)

with col_right:
    st.plotly_chart(plot_batch_measurements(df), use_container_width=True)

st.markdown("---")

# Visualizations Row 2
st.subheader("🧪 Chemical Analyte Composition Across Batches")
st.plotly_chart(plot_batch_gas_stacked(df), use_container_width=True)

# Batch x Gas Contingency Matrix
with st.expander("📋 View Batch vs Gas Analyte Cross-Tabulation Matrix", expanded=True):
    matrix_df = get_batch_class_matrix(df)
    st.dataframe(matrix_df, use_container_width=True)

st.markdown("---")

# Sensor Array Hardware Architecture Details
st.subheader("🔩 Sensor Array Specification (Figaro TGS Series)")
col_hw1, col_hw2 = st.columns([2, 1])

with col_hw1:
    sensor_spec_rows = []
    for s_id, s_desc in SENSOR_TYPES.items():
        sensor_spec_rows.append({
            "Sensor ID": f"Sensor {s_id}",
            "Sensor Model": s_desc.split(" ")[0],
            "Target Analytes & Characteristics": s_desc.split("(", 1)[1].rstrip(")") if "(" in s_desc else s_desc,
            "Extracted Features": f"f{(s_id-1)*8 + 1} to f{s_id*8} (8 features: 2 steady-state, 6 dynamic)"
        })
    spec_df = pd.DataFrame(sensor_spec_rows)
    st.dataframe(spec_df, use_container_width=True, height=360)

with col_hw2:
    st.markdown("""
    **Dataset Engineering Details**:
    - **16 MOX Sensors**: 4 repeats each of TGS2600, TGS2602, TGS2610, and TGS2620.
    - **Feature Extraction Scheme**:
      - 2 steady-state features: maximum resistance change $\\Delta R = \\max |R(t) - R_0|$ and normalized $\\Delta R / R_0$.
      - 6 dynamic features: exponential moving average (EMA) filters capturing rising and decaying transient phases.
    - **Total Features**: $16 \\times 8 = 128$ dimensions.
    - **Temporal Span**: ~36 continuous months of multi-gas exposure.
    """)
