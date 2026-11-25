"""
Interactive Plotly Visualization Library for Sensor Analytics and Drift Intelligence.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GAS_CLASSES, GAS_COLORS, HEALTH_CATEGORIES


# Theme and Styling Defaults
PLOT_TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#1E88E5"
COLOR_SECONDARY = "#004D40"
COLOR_ACCENT = "#FFC107"
COLOR_DANGER = "#D32F2F"


def plot_gas_distribution(df: pd.DataFrame) -> go.Figure:
    """Plots interactive distribution of gas classes."""
    counts = df["gas_name"].value_counts().reset_index()
    counts.columns = ["Gas Name", "Measurements"]

    fig = px.bar(
        counts,
        x="Gas Name",
        y="Measurements",
        color="Gas Name",
        color_discrete_map=GAS_COLORS,
        text="Measurements",
        title="Gas Class Sample Distribution (Total Measurements)",
        template=PLOT_TEMPLATE
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Chemical Analyte", yaxis_title="Sample Count")
    return fig


def plot_batch_measurements(df: pd.DataFrame) -> go.Figure:
    """Plots measurement sample counts across chronological batches 1 to 10."""
    b_counts = df["batch_id"].value_counts().sort_index().reset_index()
    b_counts.columns = ["Batch", "Measurements"]
    b_counts["Batch_Label"] = [f"Batch {b}" for b in b_counts["Batch"]]

    fig = px.bar(
        b_counts,
        x="Batch_Label",
        y="Measurements",
        text="Measurements",
        title="Chronological Measurement Frequency across 10 Batches (~36 Months)",
        template=PLOT_TEMPLATE,
        color="Measurements",
        color_continuous_scale="Blues"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Batch Timeline", yaxis_title="Number of Records")
    return fig


def plot_batch_gas_stacked(df: pd.DataFrame) -> go.Figure:
    """Plots stacked bar chart showing gas composition inside each batch."""
    crosstab = pd.crosstab(df["batch_id"], df["gas_name"]).reset_index()
    melted = crosstab.melt(id_vars="batch_id", var_name="Gas", value_name="Count")
    melted["Batch_Label"] = [f"Batch {b}" for b in melted["batch_id"]]

    fig = px.bar(
        melted,
        x="Batch_Label",
        y="Count",
        color="Gas",
        color_discrete_map=GAS_COLORS,
        title="Gas Class Mixture Composition per Chronological Batch",
        template=PLOT_TEMPLATE,
        barmode="stack"
    )
    fig.update_layout(xaxis_title="Chronological Batch", yaxis_title="Record Count")
    return fig


def plot_feature_distribution_comparison(
    ref_series: pd.Series,
    tgt_series: pd.Series,
    feature_name: str,
    ref_label: str = "Reference (Batch 1)",
    tgt_label: str = "Comparison Batch"
) -> go.Figure:
    """Overlays histogram and empirical distribution of reference vs target batch."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ref_series,
        name=ref_label,
        opacity=0.65,
        marker_color="#1E88E5",
        nbinsx=35
    ))
    fig.add_trace(go.Histogram(
        x=tgt_series,
        name=tgt_label,
        opacity=0.65,
        marker_color="#E53935",
        nbinsx=35
    ))
    fig.update_layout(
        title=f"Empirical Distribution Shift: Feature <b>{feature_name}</b>",
        xaxis_title=f"Feature Value ({feature_name})",
        yaxis_title="Count / Frequency",
        barmode="overlay",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_batch_feature_box(df: pd.DataFrame, feature_name: str) -> go.Figure:
    """Boxplot displaying distribution evolution of a selected feature across all 10 batches."""
    fig = px.box(
        df,
        x="batch_id",
        y=feature_name,
        color="batch_id",
        title=f"Longitudinal Distribution Profile for Feature <b>{feature_name}</b>",
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Batch ID (Chronological Timeline)",
        yaxis_title=f"Value ({feature_name})"
    )
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Interactive heatmap for inter-sensor correlations."""
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f" if len(corr_matrix) <= 16 else False,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
        template=PLOT_TEMPLATE
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_pca_drift_2d(pca_df: pd.DataFrame, color_by: str = "batch_id") -> go.Figure:
    """Plots 2D PCA projection demonstrating manifold migration and clustering drift."""
    is_cat = color_by in ["gas_name", "batch_id"]
    plot_data = pca_df.copy()
    if color_by == "batch_id":
        plot_data["Batch"] = [f"Batch {b}" for b in plot_data["batch_id"]]
        color_col = "Batch"
        color_discrete = px.colors.qualitative.Plotly
    elif color_by == "gas_name":
        color_col = "gas_name"
        color_discrete = GAS_COLORS
    else:
        color_col = color_by
        color_discrete = None

    fig = px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color=color_col,
        color_discrete_map=color_discrete if is_cat and isinstance(color_discrete, dict) else None,
        color_discrete_sequence=color_discrete if is_cat and isinstance(color_discrete, list) else None,
        hover_data=["batch_id", "gas_name"],
        title="2D PCA Latent Space: Visualizing Multi-Sensor Array Drift",
        template=PLOT_TEMPLATE,
        opacity=0.75
    )
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return fig


def plot_pca_drift_3d(pca_df: pd.DataFrame, color_by: str = "batch_id") -> go.Figure:
    """Plots 3D PCA projection."""
    plot_data = pca_df.copy()
    plot_data["Batch"] = [f"Batch {b}" for b in plot_data["batch_id"]]
    color_col = "gas_name" if color_by == "gas_name" else "Batch"

    fig = px.scatter_3d(
        plot_data,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_col,
        color_discrete_map=GAS_COLORS if color_by == "gas_name" else None,
        title="3D PCA Latent Manifold Drift across 36 Months",
        template=PLOT_TEMPLATE,
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=3))
    return fig


def plot_sensor_health_radar(health_df: pd.DataFrame) -> go.Figure:
    """Polar radar chart displaying health score (0-100) across all 16 sensors."""
    categories = health_df["Sensor_ID"].tolist()
    scores = health_df["Health_Score"].tolist()
    
    # Close polygon
    categories.append(categories[0])
    scores.append(scores[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill="toself",
        name="Reliability Score",
        line_color="#1E88E5",
        fillcolor="rgba(30, 136, 229, 0.25)"
    ))

    # Add threshold circles
    fig.add_trace(go.Scatterpolar(
        r=[80] * len(categories),
        theta=categories,
        mode="lines",
        name="Healthy Baseline (80)",
        line=dict(color="#28a745", dash="dash")
    ))
    fig.add_trace(go.Scatterpolar(
        r=[60] * len(categories),
        theta=categories,
        mode="lines",
        name="Warning Threshold (60)",
        line=dict(color="#ffc107", dash="dot")
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title="16-Sensor Array Reliability & Health Topology",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def plot_sensor_health_bar(health_df: pd.DataFrame) -> go.Figure:
    """Bar chart of sensor health scores colored by category."""
    color_map = {
        "Healthy": "#28a745",
        "Monitor": "#ffc107",
        "Warning": "#fd7e14",
        "Critical": "#dc3545"
    }
    fig = px.bar(
        health_df,
        x="Sensor_ID",
        y="Health_Score",
        color="Category",
        color_discrete_map=color_map,
        text="Health_Score",
        title="Sensor Array Reliability Scores (0–100 Scale)",
        template=PLOT_TEMPLATE
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#28a745", annotation_text="Healthy Threshold")
    fig.add_hline(y=60, line_dash="dash", line_color="#ffc107", annotation_text="Monitor Threshold")
    fig.add_hline(y=40, line_dash="dash", line_color="#dc3545", annotation_text="Critical Threshold")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Sensor Array Element", yaxis_title="Health Score (0-100)", yaxis_range=[0, 110])
    return fig


def plot_batch_drift_trend(drift_trajectory_df: pd.DataFrame) -> go.Figure:
    """Plots drift metrics progression across batches 1 to 10."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=drift_trajectory_df["batch_id"],
            y=drift_trajectory_df["avg_psi"],
            mode="lines+markers",
            name="Average PSI",
            line=dict(color="#1E88E5", width=3)
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=drift_trajectory_df["batch_id"],
            y=drift_trajectory_df["pct_features_drifted"],
            mode="lines+markers",
            name="% Features with Significant Drift (KS)",
            line=dict(color="#E53935", width=2, dash="dash")
        ),
        secondary_y=True
    )

    # Reference threshold lines
    fig.add_hline(y=0.25, line_dash="dot", line_color="#E53935", annotation_text="PSI Critical (0.25)", secondary_y=False)
    fig.add_hline(y=0.10, line_dash="dot", line_color="#FFC107", annotation_text="PSI Moderate (0.10)", secondary_y=False)

    fig.update_layout(
        title="Longitudinal Sensor Drift Trajectory Across Batches (Baseline: Batch 1)",
        xaxis_title="Chronological Batch ID",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Average Population Stability Index (PSI)", secondary_y=False)
    fig.update_yaxes(title_text="% Features Shifted (KS p < 0.05)", secondary_y=True)
    return fig


def plot_progressive_accuracy(prog_df: pd.DataFrame) -> go.Figure:
    """Plots classification accuracy degradation across batches."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prog_df["batch_id"],
        y=prog_df["accuracy"] * 100.0,
        mode="lines+markers+text",
        name="Accuracy (%)",
        text=[f"{acc*100:.1f}%" for acc in prog_df["accuracy"]],
        textposition="top center",
        line=dict(color="#1E88E5", width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=prog_df["batch_id"],
        y=prog_df["f1_score"] * 100.0,
        mode="lines+markers",
        name="Weighted F1 (%)",
        line=dict(color="#43A047", width=2, dash="dash")
    ))
    fig.update_layout(
        title="Progressive Classifier Performance Degradation (Concept Drift Effect)",
        xaxis_title="Chronological Evaluation Batch",
        yaxis_title="Score (%)",
        yaxis_range=[0, 110],
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_confusion_matrix(cm_data: List[List[int]], target_names: List[str]) -> go.Figure:
    """Plots interactive confusion matrix heatmap."""
    cm_arr = np.array(cm_data)
    fig = px.imshow(
        cm_arr,
        x=target_names,
        y=target_names,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix (Actual vs Predicted Gas Classes)",
        template=PLOT_TEMPLATE
    )
    fig.update_layout(xaxis_title="Predicted Gas Class", yaxis_title="Actual Gas Class")
    return fig


def plot_anomaly_scatter(scored_df: pd.DataFrame) -> go.Figure:
    """Plots anomaly score scatter with highlighted anomalies."""
    plot_data = scored_df.reset_index()
    plot_data["sample_index"] = plot_data.index

    fig = px.scatter(
        plot_data,
        x="sample_index",
        y="anomaly_score",
        color="anomaly_label",
        color_discrete_map={"Normal": "#1E88E5", "Anomalous": "#D32F2F"},
        hover_data=["batch_id", "gas_name"],
        title="Isolation Forest Anomaly Surveillance Timeline",
        template=PLOT_TEMPLATE,
        opacity=0.75
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#D32F2F", annotation_text="Anomaly Decision Boundary")
    fig.update_layout(xaxis_title="Sequential Observation Index", yaxis_title="Anomaly Score (0 to 1)")
    return fig
