"""
Page 4: Statistical Drift Detection & Intelligence Engine.
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
    PSI_STABLE_THRESHOLD,
    PSI_MODERATE_THRESHOLD
)
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_dataset,
    render_sidebar_info
)
from src.drift_detection import (
    compute_feature_drift,
    compute_all_features_drift,
    compute_sensor_level_drift_summary,
    compute_batch_drift_trajectory,
    DRIFT_CONCEPT_EXPLANATION
)
from src.visualization import (
    plot_feature_distribution_comparison,
    plot_batch_drift_trend
)

st.set_page_config(page_title="Drift Intelligence | Sensor Drift Intelligence", page_icon="📡", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("📡 Statistical Sensor Drift Intelligence Engine")
st.markdown("Quantify, detect, and track multi-sensor drift using rigorous statistical distribution divergence metrics.")
st.markdown("---")

df = get_cached_dataset()

# Top Selection Filters
st.subheader("⚙️ Select Reference & Comparison Baselines")
f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    ref_batch_id = st.selectbox(
        "Reference Baseline Batch",
        options=list(range(1, 11)),
        index=0,
        format_func=lambda b: f"Batch {b} (Initial Calibration)" if b == 1 else f"Batch {b}"
    )

with f_col2:
    comp_options = [b for b in range(1, 11) if b != ref_batch_id]
    tgt_batch_id = st.selectbox(
        "Comparison Target Batch",
        options=comp_options,
        index=len(comp_options) - 1,  # default to latest (e.g. Batch 10)
        format_func=lambda b: f"Batch {b} (End of Timeline ~Month 36)" if b == 10 else f"Batch {b}"
    )

with f_col3:
    drift_method = st.selectbox(
        "Primary Metric to Highlight",
        options=["Population Stability Index (PSI)", "Kolmogorov-Smirnov (KS) Test", "Wasserstein Distance (EMD)", "Normalized Z-Mean Shift"]
    )

ref_df = df[df["batch_id"] == ref_batch_id]
tgt_df = df[df["batch_id"] == tgt_batch_id]

# Compute feature and sensor-level drift
with st.spinner(f"Computing multivariate drift metrics between Batch {ref_batch_id} and Batch {tgt_batch_id}..."):
    feat_drifts = compute_all_features_drift(ref_df, tgt_df)
    sensor_drift = compute_sensor_level_drift_summary(feat_drifts)

avg_psi_val = feat_drifts["psi_score"].mean()
sig_ks_pct = (feat_drifts["ks_status"] == "Significant Drift").mean() * 100.0
sig_drift_count = (feat_drifts["psi_severity"] == "Significant Drift").sum()

# Drift KPI Row
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric(
        "Average Array PSI",
        f"{avg_psi_val:.4f}",
        delta="Significant Drift" if avg_psi_val >= PSI_MODERATE_THRESHOLD else ("Moderate" if avg_psi_val >= PSI_STABLE_THRESHOLD else "Stable"),
        delta_color="inverse"
    )
with d2:
    st.metric(
        "Significant Features (KS)",
        f"{sig_ks_pct:.1f}%",
        delta=f"p < 0.05 ({int(sig_ks_pct*128/100)} / 128)",
        delta_color="inverse"
    )
with d3:
    st.metric(
        "Critical Features (PSI ≥ 0.25)",
        f"{sig_drift_count} / 128",
        delta="Severe Shift" if sig_drift_count > 30 else "Moderate Shift",
        delta_color="inverse"
    )
with d4:
    st.metric(
        "Sensors Showing High Drift",
        f"{(sensor_drift['Drift_Severity'] == 'High').sum()} / 16",
        delta="Multi-Element Drift",
        delta_color="inverse"
    )

st.markdown("---")

# Section 1: Detailed Feature Drift Inspector
st.subheader("🔬 Single Feature Distribution Divergence Inspector")

feat_sel1, feat_sel2 = st.columns([1, 1])

with feat_sel1:
    f_sensor = st.selectbox(
        "Select Sensor",
        options=list(range(1, NUM_SENSORS + 1)),
        format_func=lambda s: f"Sensor {s} - {SENSOR_TYPES.get(s, '')}"
    )

with feat_sel2:
    f_sub = st.selectbox(
        "Select Sensor Feature",
        options=list(range(1, FEATURES_PER_SENSOR + 1)),
        format_func=lambda f: f"Feature {f}: {FEATURE_DESCRIPTIONS.get(f, '')}"
    )

selected_feature = f"f{(f_sensor - 1) * 8 + f_sub}"
single_drift = feat_drifts[feat_drifts["feature"] == selected_feature].iloc[0]

# Display Feature Drift Card
f_card1, f_card2, f_card3, f_card4 = st.columns(4)
with f_card1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">PSI Score</div>
        <div class="metric-card-value">{single_drift['psi_score']}</div>
        <div class="metric-card-delta"><b>Severity:</b> {single_drift['psi_severity']}</div>
    </div>
    """, unsafe_allow_html=True)
with f_card2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">KS Statistic & p-value</div>
        <div class="metric-card-value">D = {single_drift['ks_statistic']}</div>
        <div class="metric-card-delta">p = {single_drift['ks_pvalue']:.2e} ({single_drift['ks_status']})</div>
    </div>
    """, unsafe_allow_html=True)
with f_card3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Normalized Z-Shift</div>
        <div class="metric-card-value">{single_drift['z_mean_shift']:.2f} σ</div>
        <div class="metric-card-delta">Δ Mean: {single_drift['delta_mean']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with f_card4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Wasserstein Distance</div>
        <div class="metric-card-value">{single_drift['wasserstein_distance']:.2f}</div>
        <div class="metric-card-delta">Variance Ratio: {single_drift['variance_ratio']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# Distribution Comparison Histogram
fig_dist = plot_feature_distribution_comparison(
    ref_df[selected_feature],
    tgt_df[selected_feature],
    feature_name=selected_feature,
    ref_label=f"Baseline (Batch {ref_batch_id})",
    tgt_label=f"Comparison (Batch {tgt_batch_id})"
)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Section 2: Sensor Array Aggregated Drift Profile
st.subheader(f"📊 16-Sensor Array Aggregate Drift Summary (Batch {ref_batch_id} vs Batch {tgt_batch_id})")
st.dataframe(
    sensor_drift[[
        "Sensor_ID", "Sensor_Type", "Avg_PSI", "Max_PSI",
        "Avg_KS_Statistic", "Significant_Features_Count",
        "Mean_Z_Shift", "Avg_Variance_Ratio", "Drift_Severity"
    ]],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Section 3: Longitudinal Batch Drift Trajectory
st.subheader("📈 Longitudinal Sensor Array Drift Trajectory Across All 10 Batches")
st.markdown("Tracking empirical drift accumulation relative to the Batch 1 baseline across the 36-month timeline.")

with st.spinner("Calculating 10-batch longitudinal drift trajectory..."):
    trajectory_df = compute_batch_drift_trajectory(df, ref_batch_id=1)

fig_trend = plot_batch_drift_trend(trajectory_df)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# Section 4: Domain Knowledge & Educational Section
with st.expander("📚 Essential Theory: Data Drift vs Concept Drift vs Physical Sensor Aging", expanded=True):
    st.markdown(DRIFT_CONCEPT_EXPLANATION)
