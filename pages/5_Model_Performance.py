"""
Page 5: Machine Learning Model Performance & Drift Degradation.
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
    get_cached_gas_classifier,
    render_sidebar_info
)
from src.gas_classification import (
    train_and_evaluate_all_models,
    progressive_batch_evaluation
)
from src.model_evaluation import (
    format_classification_metrics,
    get_per_class_metrics,
    calculate_degradation_statistics
)
from src.visualization import (
    plot_progressive_accuracy,
    plot_confusion_matrix
)

st.set_page_config(page_title="Model Performance | Sensor Drift Intelligence", page_icon="🤖", layout="wide")
apply_custom_styles()
render_sidebar_info()

st.title("🤖 Gas Classification & Concept Drift Impact")
st.markdown("Chronological validation of multi-class gas classifiers and quantification of performance degradation over time.")
st.markdown("---")

scaled_df = get_cached_scaled_dataset()
loaded_model, metadata = get_cached_gas_classifier()

# Chronological Model Comparison
st.subheader("⚖️ Chronological Benchmark: Model Comparison (Train on Batch 1, Test on Batches 2–10)")
st.caption("Models are fitted strictly on baseline calibration measurements (Batch 1) and tested against unseen subsequent operational periods.")

@st.cache_data(show_spinner=False)
def get_cached_model_results():
    results, best = train_and_evaluate_all_models(scaled_df, train_batches=[1])
    return results, best

with st.spinner("Evaluating classifiers under chronological validation..."):
    comp_results, best_tuple = get_cached_model_results()
    best_name, _ = best_tuple

metrics_table = format_classification_metrics(comp_results)
st.dataframe(metrics_table, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 2: Best Model Detailed Inspection
st.subheader(f"🎯 Detailed Diagnostics: {best_name}")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
best_res = comp_results[best_name]

with col_m1:
    st.metric("Test Accuracy (Batches 2–10)", f"{best_res['accuracy']*100:.2f}%")
with col_m2:
    st.metric("Weighted F1-Score", f"{best_res['f1_weighted']*100:.2f}%")
with col_m3:
    st.metric("Weighted Precision", f"{best_res['precision']*100:.2f}%")
with col_m4:
    st.metric("Weighted Recall", f"{best_res['recall']*100:.2f}%")

c_col1, c_col2 = st.columns([1, 1])

with c_col1:
    st.markdown("**Confusion Matrix (Actual vs Predicted)**")
    target_names = [GAS_CLASSES[i] for i in sorted(GAS_CLASSES.keys())]
    fig_cm = plot_confusion_matrix(best_res["confusion_matrix"], target_names)
    st.plotly_chart(fig_cm, use_container_width=True)

with c_col2:
    st.markdown("**Per-Class Precision, Recall, and F1 Breakdown**")
    class_df = get_per_class_metrics(best_res["classification_report"])
    st.dataframe(class_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 3: Progressive Batch Degradation Analysis
st.subheader("📉 Progressive Batch Degradation (Concept Drift in Action)")
st.markdown("""
By evaluating the Batch-1-trained model on each chronological batch individually (1 to 10), 
we observe how sensor aging and drift progressively corrupt the classifier's ability to discriminate gases.
""")

@st.cache_data(show_spinner=False)
def get_cached_progressive_evaluation():
    _, best_m_obj = best_tuple
    return progressive_batch_evaluation(scaled_df, best_m_obj)

prog_df = get_cached_progressive_evaluation()
deg_stats = calculate_degradation_statistics(prog_df)

# Degradation KPIs
deg1, deg2, deg3, deg4 = st.columns(4)
with deg1:
    st.metric("Initial Batch 1 Accuracy", f"{deg_stats['initial_accuracy']:.1f}%", delta="In-Sample Baseline")
with deg2:
    st.metric("Final Batch 10 Accuracy", f"{deg_stats['final_accuracy']:.1f}%", delta=f"-{deg_stats['absolute_accuracy_drop_pct']:.1f}% Drop", delta_color="inverse")
with deg3:
    st.metric("Total Relative Degradation", f"{deg_stats['relative_accuracy_drop_pct']:.1f}%", delta="Concept Drift Loss", delta_color="inverse")
with deg4:
    st.metric("Sharpest Degradation Epoch", f"Batch {deg_stats['sharpest_drop_batch']}", delta=f"-{deg_stats['sharpest_drop_pct']:.1f}% Drop", delta_color="inverse")

fig_prog = plot_progressive_accuracy(prog_df)
st.plotly_chart(fig_prog, use_container_width=True)

with st.expander("📊 View Complete Progressive Batch Evaluation Metrics Table", expanded=False):
    st.dataframe(prog_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 4: ML Engineering Discussion
st.info("""
> 💡 **Machine Learning Takeaway for Sensor Intelligence:**
> Standard static ML models assume stationary feature distributions: $P_{\\text{train}}(X, Y) = P_{\\text{test}}(X, Y)$.
> In physical sensor arrays, $P(X)$ changes due to chemical surface aging and baseline creep (**Data Drift**),
> which in turn alters the decision boundary mapping $P(Y|X)$ (**Concept Drift**).
> Consequently, periodic **adaptive retraining**, **drift compensation manifolds**, or **sensor health monitoring** is essential.
""")
