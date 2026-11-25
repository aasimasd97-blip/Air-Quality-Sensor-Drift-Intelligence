"""
Main Application Entrypoint: Air Quality Sensor Drift Intelligence.
An Intelligent Sensor Reliability and Drift Detection System.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import GAS_CLASSES, SENSOR_TYPES, PROJECT_ROOT
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_dataset,
    get_cached_scaled_dataset,
    get_cached_preprocessor,
    get_cached_gas_classifier,
    get_cached_anomaly_detector,
    render_sidebar_info
)
from src.drift_detection import compute_all_features_drift, compute_sensor_level_drift_summary
from src.sensor_health import SensorHealthScorer
from src.visualization import (
    plot_batch_measurements,
    plot_gas_distribution,
    plot_sensor_health_radar
)

# Configure Streamlit page
st.set_page_config(
    page_title="Sensor Drift Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()
render_sidebar_info()

# Header
st.title("🔬 Air Quality Sensor Drift Intelligence")
st.markdown("### *An Intelligent Multi-Sensor Reliability, Anomaly, and Drift Detection System*")
st.markdown("---")

# Quick check on models and data
try:
    df = get_cached_dataset()
    scaled_df = get_cached_scaled_dataset()
    classifier, meta = get_cached_gas_classifier()
    detector = get_cached_anomaly_detector()
    system_ready = True
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    system_ready = False

if system_ready:
    # Top KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Measurements", value=f"{len(df):,}")
    with col2:
        st.metric(label="Sensor Array", value="16 Sensors", delta="Figaro MOX")
    with col3:
        st.metric(label="Feature Dimensions", value="128 Features", delta="8 per sensor")
    with col4:
        st.metric(label="Chronological Batches", value="10 Batches", delta="~36 Months")
    with col5:
        st.metric(label="Chemical Analytes", value="6 Classes", delta="VOCs & Gases")

    st.markdown("---")

    # High-level Surveillance Summary
    st.subheader("📡 Real-Time Sensor Array Reliability Overview")
    
    # Calculate baseline vs latest batch health
    ref_batch_df = df[df["batch_id"] == 1]
    latest_batch_df = df[df["batch_id"] == 10]
    
    # Quick drift summary
    feature_drifts = compute_all_features_drift(ref_batch_df, latest_batch_df)
    sensor_drift = compute_sensor_level_drift_summary(feature_drifts)
    health_df = SensorHealthScorer.calculate_health_scores(sensor_drift, batch_anomaly_rate=12.2, model_accuracy_drop=0.65)
    
    mean_health = health_df["Health_Score"].mean()
    critical_sensors = (health_df["Category"] == "Critical").sum()
    warning_sensors = (health_df["Category"] == "Warning").sum()

    s_col1, s_col2 = st.columns([1, 2])
    
    with s_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">System Reliability Index (Latest Batch 10)</div>
            <div class="metric-card-value" style="color: {'#28a745' if mean_health >= 70 else ('#fd7e14' if mean_health >= 50 else '#dc3545')};">
                {mean_health:.1f} / 100
            </div>
            <div class="metric-card-delta">
                <b>Status:</b> {'Optimal' if mean_health >= 80 else 'Drift Detected across Array'}
            </div>
            <hr style="margin: 10px 0;">
            <p style="font-size: 0.9rem; margin-bottom: 4px;"><b>Array Elements at Risk:</b></p>
            <span class="status-pill pill-critical">{critical_sensors} Critical</span>
            <span class="status-pill pill-warning">{warning_sensors} Warning</span>
            <span class="status-pill pill-healthy">{16 - critical_sensors - warning_sensors} Stable</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **System Operation Notice**:
        The sensor array shows measurable statistical distribution displacement between early batches and Batch 10 (recorded over a 36-month timeline).
        Navigate through the sidebar modules for detailed diagnostics.
        """)

    with s_col2:
        radar_fig = plot_sensor_health_radar(health_df)
        st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")

    # Module Navigation Cards
    st.subheader("🧭 Guided Analytical Modules")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        #### 1. [Overview & Telemetry](1_Overview)
        Comprehensive dataset distributions, batch sample sizes, gas classes, and system metadata.
        
        #### 2. [Sensor Analytics](2_Sensor_Analytics)
        Deep exploratory data analysis, inter-sensor correlation matrices, and 2D/3D PCA latent manifold drift.
        """)
        
    with nav_col2:
        st.markdown("""
        #### 3. [Anomaly Detection](3_Anomaly_Detection)
        Isolation Forest surveillance, batch anomaly percentage tracking, and downloadable outlier logs.
        
        #### 4. [Drift Intelligence](4_Drift_Intelligence)
        Multi-method statistical drift engine: PSI, KS Test, Mean Shift, and longitudinal drift curves.
        """)
        
    with nav_col3:
        st.markdown("""
        #### 5. [Model Performance](5_Model_Performance)
        Random Forest, SVM, and Logistic Regression with progressive degradation analysis over 36 months.
        
        #### 6. [Sensor Health & Reports](6_Sensor_Health)
        Explainable 0–100 reliability scoring, calibration recommendations, and automated CSV & PDF audits.
        """)

    st.markdown("---")
    
    # Dataset preview
    with st.expander("🔍 View Raw Telemetry Snapshot (UCI Gas Sensor Array Drift Dataset)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Showing first 10 records. Each observation contains 128 continuous sensor features + batch_id + gas_class.")
