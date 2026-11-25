"""
Page 2: Sensor Analytics, Correlation, and PCA Latent Drift Manifold.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    NUM_SENSORS,
    FEATURES_PER_SENSOR,
    SENSOR_TYPES,
    FEATURE_DESCRIPTIONS,
    get_sensor_features
)
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_dataset,
    get_cached_scaled_dataset,
    get_cached_pca_analyzer,
    render_sidebar_info
)
from src.feature_engineering import SensorFeatureExtractor
from src.visualization import (
    plot_correlation_heatmap,
    plot_batch_feature_box,
    plot_pca_drift_2d,
    plot_pca_drift_3d
)

st.set_page_config(page_title="Sensor Analytics | Sensor Drift Intelligence", page_icon="📈", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("📈 Sensor Analytics & Feature Exploration")
st.markdown("Investigate individual sensor response dynamics, inter-sensor correlations, and PCA manifold migration.")
st.markdown("---")

df = get_cached_dataset()
scaled_df = get_cached_scaled_dataset()
pca_analyzer = get_cached_pca_analyzer()

# Section 1: Individual Sensor & Feature Deep Dive
st.subheader("🔍 Individual Sensor Feature Inspection")

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

with ctrl_col1:
    selected_sensor = st.selectbox(
        "Select Sensor Element",
        options=list(range(1, NUM_SENSORS + 1)),
        format_func=lambda s: f"Sensor {s} ({SENSOR_TYPES.get(s, '').split(' ')[0]})"
    )

with ctrl_col2:
    selected_subfeat = st.selectbox(
        "Select Sub-Feature",
        options=list(range(1, FEATURES_PER_SENSOR + 1)),
        format_func=lambda f: f"Feature {f} - {FEATURE_DESCRIPTIONS.get(f, '').split('(')[0]}"
    )

feat_col_name = f"f{(selected_sensor - 1) * 8 + selected_subfeat}"

with ctrl_col3:
    st.info(f"""
    **Current Dimension**: `{feat_col_name}`  
    **Sensor**: {SENSOR_TYPES.get(selected_sensor)}  
    **Feature Description**: {FEATURE_DESCRIPTIONS.get(selected_subfeat)}
    """)

# Plot longitudinal feature evolution across 10 batches
st.plotly_chart(plot_batch_feature_box(df, feat_col_name), use_container_width=True)

st.markdown("---")

# Section 2: Inter-Sensor Correlation Analysis
st.subheader("🔗 Inter-Sensor Array Cross-Correlation")
st.markdown("Analyze how the 16 chemical sensors covary under different batch conditions.")

corr_batch = st.selectbox(
    "Select Batch for Correlation Matrix",
    options=[0] + list(range(1, 11)),
    format_func=lambda b: "Overall Dataset (All 10 Batches)" if b == 0 else f"Batch {b} Only"
)

target_batch = None if corr_batch == 0 else corr_batch
corr_mat = SensorFeatureExtractor.get_sensor_correlation_matrix(df, batch_id=target_batch)

st.plotly_chart(plot_correlation_heatmap(corr_mat, title=f"16-Sensor Cross-Correlation ({'All Batches' if corr_batch == 0 else f'Batch {corr_batch}'})"), use_container_width=True)

st.markdown("---")

# Section 3: PCA Latent Manifold Drift
st.subheader("🌌 Principal Component Analysis (PCA) Drift Manifold")
st.markdown("""
PCA is fitted exclusively on **Batch 1 (Baseline Calibration)** to define the reference manifold.
When subsequent batches are projected onto this space, their spatial migration directly illustrates **Sensor Array Drift**.
""")

pca_col1, pca_col2 = st.columns([1, 3])

with pca_col1:
    pca_color_choice = st.radio(
        "Color Observations By:",
        options=["batch_id", "gas_name"],
        format_func=lambda x: "Chronological Batch ID" if x == "batch_id" else "Gas Analyte Class"
    )
    pca_dim = st.radio("Manifold Dimensions:", ["2D Scatter", "3D Interactive Scatter"])

    # Transform scaled dataset via PCA
    pca_df = pca_analyzer.transform(scaled_df)
    var_exp = pca_analyzer.explained_variance_ratio()
    st.markdown(f"""
    **Explained Variance:**
    - PC1: `{var_exp[0]*100:.1f}%`
    - PC2: `{var_exp[1]*100:.1f}%`
    - PC3: `{var_exp[2]*100:.1f}%`
    - **Total (3 PCs)**: `{sum(var_exp)*100:.1f}%`
    """)

with pca_col2:
    if pca_dim == "2D Scatter":
        fig_pca = plot_pca_drift_2d(pca_df, color_by=pca_color_choice)
    else:
        fig_pca = plot_pca_drift_3d(pca_df, color_by=pca_color_choice)
    st.plotly_chart(fig_pca, use_container_width=True)
