"""
Page 7: Reports, Live Inference, and Automated Audit Exports.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import io
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    GAS_CLASSES,
    SAMPLE_DATASET_PATH,
    PROCESSED_CSV_PATH
)
from src.ui_helpers import (
    apply_custom_styles,
    get_cached_dataset,
    get_cached_preprocessor,
    get_cached_gas_classifier,
    get_cached_anomaly_detector,
    render_sidebar_info
)
from src.drift_detection import (
    compute_all_features_drift,
    compute_sensor_level_drift_summary
)
from src.sensor_health import SensorHealthScorer
from src.report_generator import (
    generate_pdf_report,
    generate_drift_csv_report,
    get_pdf_bytes
)
from src.gas_classification import (
    train_and_evaluate_all_models,
    save_classifier_and_metadata
)

st.set_page_config(page_title="Reports & Inference | Sensor Drift Intelligence", page_icon="📑", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("📑 Audit Reports, Inference & Model Management")
st.markdown("Upload telemetry batches, run automated inference and drift audits, export PDF/CSV reports, or retrain models.")
st.markdown("---")

df_full = get_cached_dataset()
prep = get_cached_preprocessor()
classifier, meta = get_cached_gas_classifier()
detector = get_cached_anomaly_detector()

# Section 1: Telemetry Upload & Schema Validation
st.subheader("📤 Telemetry Ingestion & Live Inference")

st.markdown("""
Upload a sensor telemetry CSV containing the 128 continuous features (`f1` to `f128`).
Alternatively, click below to load pre-packaged stratified sample records.
""")

col_up1, col_up2 = st.columns([2, 1])

with col_up2:
    load_sample_btn = st.button("🧪 Load Stratified Sample Batch (285 records)", type="secondary")

with col_up1:
    uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])

upload_df = None
if load_sample_btn:
    if SAMPLE_DATASET_PATH.exists():
        upload_df = pd.read_csv(SAMPLE_DATASET_PATH)
        st.success("Successfully loaded pre-packaged sample dataset!")
    else:
        st.warning("Sample dataset not found. Generating sample from cached data...")
        upload_df = df_full.sample(250, random_state=42)
elif uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file loaded successfully: {len(upload_df)} records.")
    except Exception as e:
        st.error(f"Error parsing uploaded CSV: {e}")

if upload_df is not None:
    # Schema validation
    missing_feats = [f for f in FEATURE_NAMES if f not in upload_df.columns]
    
    if len(missing_feats) == 0:
        st.success("✅ Schema Validation Passed: All 128 feature columns present and verified.")
        schema_valid = True
    else:
        st.warning(f"⚠️ Schema Warning: {len(missing_feats)} expected features missing. Attempting zero-fill for inference.")
        for f in missing_feats:
            upload_df[f] = 0.0
        schema_valid = True

    if schema_valid:
        # Run inference: Scale -> Predict Gas -> Predict Anomaly
        with st.spinner("Running inference pipeline (Scaling, Gas Classification, Anomaly Scoring)..."):
            scaled_input = prep.transform(upload_df)
            
            # Gas predictions
            preds = classifier.predict(scaled_input[FEATURE_NAMES])
            upload_df["predicted_gas_id"] = preds
            upload_df["predicted_gas_name"] = [GAS_CLASSES.get(p, f"Class {p}") for p in preds]
            
            # Anomaly scoring
            scored_input = detector.score_samples(scaled_input)
            upload_df["anomaly_score"] = scored_input["anomaly_score"].round(4)
            upload_df["is_anomaly"] = scored_input["is_anomaly"]
            upload_df["anomaly_label"] = scored_input["anomaly_label"]

        st.markdown("### 📊 Ingested Batch Inference Results")
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Analyzed Records", f"{len(upload_df):,}")
        with res2:
            anom_pct = (upload_df['is_anomaly'].sum() / len(upload_df)) * 100.0
            st.metric("Flagged Anomalies", f"{upload_df['is_anomaly'].sum():,}", delta=f"{anom_pct:.1f}%")
        with res3:
            top_gas = upload_df["predicted_gas_name"].mode()[0]
            st.metric("Dominant Analyte", top_gas)

        st.dataframe(
            upload_df[["predicted_gas_name", "anomaly_score", "anomaly_label", "f1", "f2", "f3", "f4"]].head(25),
            use_container_width=True
        )

        st.markdown("---")

        # Section 2: Drift Assessment of Ingested Batch vs Reference Baseline (Batch 1)
        st.subheader("🔬 Drift Assessment vs Baseline Calibration (Batch 1)")
        ref_df = df_full[df_full["batch_id"] == 1]
        
        with st.spinner("Computing statistical drift metrics against reference baseline..."):
            feat_drifts = compute_all_features_drift(ref_df, upload_df)
            sensor_drift = compute_sensor_level_drift_summary(feat_drifts)
            health_df = SensorHealthScorer.calculate_health_scores(
                sensor_drift,
                batch_anomaly_rate=float(anom_pct),
                model_accuracy_drop=0.35
            )

        mean_h = float(health_df["Health_Score"].mean())
        crit_c = int((health_df["Category"] == "Critical").sum())
        
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            st.dataframe(health_df[["Sensor_ID", "Health_Score", "Category", "Avg_PSI", "Recommendation"]], use_container_width=True, hide_index=True)
        
        with col_dr2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Batch Health Summary</div>
                <div class="metric-card-value">{mean_h:.1f} / 100</div>
                <div class="metric-card-delta">
                    Critical Elements: <b>{crit_c} / 16</b>
                </div>
                <hr style="margin: 10px 0;">
                <p style="font-size: 0.85rem;">
                <b>Audit Verdict:</b> {'Sensor array operating within safe tolerances.' if mean_h >= 75 else 'Significant drift detected. Baseline zero-gas recalibration recommended.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Section 3: Downloads (CSV & PDF)
        st.subheader("📥 Export Audit Artifacts")
        d_col1, d_col2 = st.columns(2)

        summary_meta = {
            "reference_batch": "Batch 1 (Initial Calibration)",
            "comparison_batch": "Uploaded Telemetry Batch",
            "total_samples": f"{len(upload_df):,}",
            "mean_health": f"{mean_h:.1f}",
            "drift_status": "Significant Drift" if mean_h < 60 else "Moderate Drift",
            "anomaly_pct": f"{anom_pct:.1f}%",
            "critical_count": str(crit_c)
        }

        with d_col1:
            csv_bytes = upload_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Scored Predictions & Anomalies (CSV)",
                data=csv_bytes,
                file_name="sensor_telemetry_inference.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d_col2:
            try:
                pdf_bytes = get_pdf_bytes(summary_meta, health_df)
                st.download_button(
                    label="📕 Download Executive PDF Audit Report",
                    data=pdf_bytes,
                    file_name="Sensor_Drift_Audit_Report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

st.markdown("---")

# Section 4: Interactive Model Retraining
st.subheader("🔄 Adaptive Model Retraining Engine")
st.markdown("""
To combat concept drift, train a new model instance on selected batches (e.g., retraining on Batches 1, 2, and 3).
""")

retrain_expander = st.expander("⚙️ Retrain Gas Classifier on Custom Chronological Batches", expanded=False)
with retrain_expander:
    train_batch_selection = st.multiselect(
        "Select Batches for Training Set",
        options=list(range(1, 11)),
        default=[1, 2],
        help="Select batches to fit the new model instance."
    )
    
    if st.button("🚀 Retrain Classifier Now", type="primary"):
        if not train_batch_selection:
            st.error("Please select at least one batch for training.")
        else:
            with st.spinner("Retraining classifiers across selected batches..."):
                scaled_full = prep.transform(df_full)
                new_results, best_new = train_and_evaluate_all_models(
                    scaled_full,
                    train_batches=train_batch_selection
                )
                best_new_name, best_new_obj = best_new
                
                # Save updated model
                save_classifier_and_metadata(
                    best_new_obj,
                    best_new_name,
                    {
                        "accuracy": new_results[best_new_name]["accuracy"],
                        "f1_weighted": new_results[best_new_name]["f1_weighted"],
                        "train_batches": train_batch_selection
                    }
                )
                
                st.success(f"Model successfully updated! Best Classifier: {best_new_name}")
                st.json({
                    "Retrained On": train_batch_selection,
                    "New Test Accuracy": f"{new_results[best_new_name]['accuracy']*100:.2f}%",
                    "New Weighted F1": f"{new_results[best_new_name]['f1_weighted']*100:.2f}%"
                })
