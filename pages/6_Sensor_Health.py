"""
Page 6: Sensor Health & Reliability Monitoring Intelligence.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SENSOR_TYPES, HEALTH_CATEGORIES
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_dataset,
    render_sidebar_info
)
from src.drift_detection import (
    compute_all_features_drift,
    compute_sensor_level_drift_summary
)
from src.sensor_health import SensorHealthScorer, SENSOR_HEALTH_DISCLAIMER
from src.visualization import (
    plot_sensor_health_bar,
    plot_sensor_health_radar
)

st.set_page_config(page_title="Sensor Health | Sensor Drift Intelligence", page_icon="🩺", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("🩺 Sensor Array Health & Reliability Intelligence")
st.markdown("Multi-factor explainable health scoring (0–100) and actionable calibration decision support.")
st.markdown("---")

df = get_cached_dataset()

# Evaluation Target Selection
eval_col1, eval_col2 = st.columns([1, 2])

with eval_col1:
    eval_batch = st.selectbox(
        "Select Operational Batch to Assess",
        options=list(range(2, 11)),
        index=8,  # Default to Batch 10
        format_func=lambda b: f"Batch {b} (~End of 36-Month Timeline)" if b == 10 else f"Batch {b}"
    )

ref_df = df[df["batch_id"] == 1]
tgt_df = df[df["batch_id"] == eval_batch]

with st.spinner(f"Computing sensor reliability and degradation scores for Batch {eval_batch}..."):
    feat_drifts = compute_all_features_drift(ref_df, tgt_df)
    sensor_drift = compute_sensor_level_drift_summary(feat_drifts)
    health_df = SensorHealthScorer.calculate_health_scores(
        sensor_drift,
        batch_anomaly_rate=12.5,
        model_accuracy_drop=0.45
    )

mean_score = health_df["Health_Score"].mean()
crit_count = (health_df["Category"] == "Critical").sum()
warn_count = (health_df["Category"] == "Warning").sum()
mon_count = (health_df["Category"] == "Monitor").sum()
health_count = (health_df["Category"] == "Healthy").sum()

# Top KPI Summary Cards
h1, h2, h3, h4, h5 = st.columns(5)
with h1:
    st.metric("Array Average Health", f"{mean_score:.1f} / 100", delta="Composite Reliability Index")
with h2:
    st.metric("Healthy Sensors", f"{health_count} / 16", delta="80–100 Scale")
with h3:
    st.metric("Watch List (Monitor)", f"{mon_count} / 16", delta="60–79 Scale")
with h4:
    st.metric("Calibration Advised", f"{warn_count} / 16", delta="40–59 Scale")
with h5:
    st.metric("Critical Sensors", f"{crit_count} / 16", delta="0–39 Scale", delta_color="inverse")

st.markdown("---")

# Visualizations Row: Bar Chart & Radar Topology
v_col1, v_col2 = st.columns([1, 1])

with v_col1:
    st.plotly_chart(plot_sensor_health_bar(health_df), use_container_width=True)

with v_col2:
    st.plotly_chart(plot_sensor_health_radar(health_df), use_container_width=True)

st.markdown("---")

# Section 2: Sensor Health Table & Detailed Component Breakdown
st.subheader("📋 Explainable Sensor Reliability Breakdown & Recommendations")

display_cols = [
    "Sensor_ID", "Sensor_Type", "Health_Score", "Category",
    "Drift_Component", "Anomaly_Component", "Stability_Component", "Degradation_Component",
    "Recommendation"
]

st.dataframe(health_df[display_cols], use_container_width=True, hide_index=True)

st.markdown("---")

# Section 3: Sensor Action Cards
st.subheader("🛠️ Prescriptive Calibration Action Cards")

for idx, row in health_df.iterrows():
    cat = row["Category"]
    if cat in ["Critical", "Warning", "Monitor"]:
        with st.expander(f"**{row['Sensor_ID']} ({row['Sensor_Type']})** — Score: `{row['Health_Score']}/100` [{cat}]"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                - **Drift Sub-Score**: `{row['Drift_Component']}/100`
                - **Anomaly Resilience**: `{row['Anomaly_Component']}/100`
                - **Variance Stability**: `{row['Stability_Component']}/100`
                - **Model Impact**: `{row['Degradation_Component']}/100`
                """)
            with c2:
                st.markdown(f"**Action Protocol**: {row['Recommendation']}")
                st.markdown("""
                *Suggested Protocol Steps:*
                1. Apply baseline zero-gas purge (synthetic air at 400 mL/min for 10 min).
                2. Test with single-analyte verification gas.
                3. Update dynamic baseline resistance offset ($R_0$) in firmware.
                """)

st.markdown("---")

# Disclaimer Box
st.markdown(SENSOR_HEALTH_DISCLAIMER)
