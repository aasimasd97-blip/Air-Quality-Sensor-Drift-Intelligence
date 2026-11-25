"""
Page 3: Sensor Anomaly Detection & Surveillance.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GAS_CLASSES
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_scaled_dataset,
    get_cached_anomaly_detector,
    render_sidebar_info
)
from src.visualization import plot_anomaly_scatter
from src.report_generator import generate_anomaly_csv_report

st.set_page_config(page_title="Anomaly Detection | Sensor Drift Intelligence", page_icon="🚨", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("🚨 Sensor Array Anomaly Surveillance")
st.markdown("Unsupervised anomaly detection using Isolation Forest trained on the baseline calibration manifold.")
st.markdown("---")

scaled_df = get_cached_scaled_dataset()
detector = get_cached_anomaly_detector()

# Score the entire dataset
with st.spinner("Scoring measurements through Isolation Forest..."):
    scored_df = detector.score_samples(scaled_df)
    batch_summary = detector.batch_anomaly_summary(scored_df)

# Top Metrics Row
tot_obs = len(scored_df)
tot_anom = int(scored_df["is_anomaly"].sum())
overall_pct = (tot_anom / tot_obs) * 100.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Surveillance Samples", f"{tot_obs:,}")
with m2:
    st.metric("Flagged Anomalies", f"{tot_anom:,}")
with m3:
    st.metric("Array Anomaly Rate", f"{overall_pct:.2f}%", delta="Isolation Forest")
with m4:
    st.metric("Decision Boundary Contamination", "5.0%", delta="Baseline Batch 1")

st.markdown("---")

# Batch Summary Table and Bar Breakdown
st.subheader("📊 Batch-Wise Anomaly Rates Across the 36-Month Timeline")

col_tbl, col_chart = st.columns([1, 1])

with col_tbl:
    st.markdown("**Longitudinal Batch Anomaly Profile**")
    st.dataframe(batch_summary, use_container_width=True, hide_index=True)

with col_chart:
    st.bar_chart(
        batch_summary.set_index("batch_id")["anomaly_rate_pct"],
        color="#D32F2F"
    )

st.markdown("---")

# Interactive Batch Explorer & Scatter Timeline
st.subheader("🔎 Sequential Observation Anomaly Timeline")

selected_batch = st.selectbox(
    "Filter Timeline by Batch",
    options=[0] + list(range(1, 11)),
    format_func=lambda b: "All Batches (Chronological Sequence)" if b == 0 else f"Batch {b} Only"
)

display_df = scored_df if selected_batch == 0 else scored_df[scored_df["batch_id"] == selected_batch]

fig_scatter = plot_anomaly_scatter(display_df)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Flagged Records Table & CSV Download
st.subheader("📥 Flagged Anomalous Records Inspection")

anom_df = display_df[display_df["is_anomaly"]].copy()

if len(anom_df) > 0:
    st.dataframe(
        anom_df[["batch_id", "gas_name", "anomaly_score", "anomaly_label", "f1", "f2", "f3", "f4", "f5"]].head(50),
        use_container_width=True
    )
    
    csv_bytes = anom_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {len(anom_df)} Flagged Anomaly Records (CSV)",
        data=csv_bytes,
        file_name=f"sensor_anomalies_batch_{selected_batch if selected_batch > 0 else 'all'}.csv",
        mime="text/csv",
        type="primary"
    )
else:
    st.success("No anomalies detected within the selected filter scope.")

st.markdown("---")

# Scientific Caution Note
st.info("""
> ⚠️ **Sensor Domain Note: Anomalies vs Hardware Failure**
> An anomaly detected in this high-dimensional feature space denotes a statistically unexpected reading relative to the baseline calibration.
> This does **not** signify a broken or destroyed physical sensor. In chemical gas sensor arrays, sudden ambient temperature changes, 
> extreme analyte concentrations beyond standard calibration curves, or temporary reversible poisoning can all elevate anomaly scores.
""")
