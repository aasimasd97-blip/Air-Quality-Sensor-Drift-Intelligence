"""
Statistical Sensor Drift Detection Engine.
Implements:
- Population Stability Index (PSI)
- Two-Sample Kolmogorov-Smirnov (KS) Test
- Mean Shift and Variance Ratio Analysis
- Wasserstein Distance (Earth Mover's Distance)
- Multi-Sensor Array Drift Aggregation
- Batch-wise Longitudinal Drift Trajectory
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    NUM_SENSORS,
    FEATURES_PER_SENSOR,
    PSI_STABLE_THRESHOLD,
    PSI_MODERATE_THRESHOLD,
    KS_ALPHA,
    SENSOR_TYPES
)


def calculate_psi(
    reference: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, str]:
    """
    Calculates Population Stability Index (PSI) between reference and target distributions.
    
    Formula:
    PSI = sum((target_pct - ref_pct) * ln(target_pct / ref_pct))
    
    Severity thresholds:
    - PSI < 0.10: Stable (No Drift)
    - 0.10 <= PSI < 0.25: Moderate Drift
    - PSI >= 0.25: Significant Drift
    """
    ref_clean = reference[~np.isnan(reference) & ~np.isinf(reference)]
    tgt_clean = target[~np.isnan(target) & ~np.isinf(target)]
    
    if len(ref_clean) < 5 or len(tgt_clean) < 5:
        return 0.0, "Insufficient Data"

    # Define bins based on reference percentiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    # Handle duplicate bin edges if feature has low variance
    if len(np.unique(bin_edges)) < len(bin_edges):
        bin_edges = np.linspace(np.min(ref_clean) - 1e-5, np.max(ref_clean) + 1e-5, num_bins + 1)

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt_clean, bins=bin_edges)

    # Convert to proportions with smoothing epsilon to prevent division by zero
    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * num_bins)
    tgt_pct = (tgt_counts + epsilon) / (np.sum(tgt_counts) + epsilon * num_bins)

    psi_value = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
    psi_val_clean = float(np.nan_to_num(psi_value, nan=0.0, posinf=1.0, neginf=0.0))

    if psi_val_clean < PSI_STABLE_THRESHOLD:
        severity = "Stable (No Drift)"
    elif psi_val_clean < PSI_MODERATE_THRESHOLD:
        severity = "Moderate Drift"
    else:
        severity = "Significant Drift"

    return round(psi_val_clean, 4), severity


def calculate_ks_test(
    reference: np.ndarray,
    target: np.ndarray,
    alpha: float = KS_ALPHA
) -> Tuple[float, float, str]:
    """
    Two-sample Kolmogorov-Smirnov test for comparing reference and target empirical distributions.
    
    Returns:
    - statistic (D): maximum vertical deviation between empirical CDFs [0, 1]
    - p_value: probability of observing such deviation under H0 (identical distributions)
    - status: 'Significant Drift' if p_value < alpha else 'Stable'
    """
    ref_clean = reference[~np.isnan(reference) & ~np.isinf(reference)]
    tgt_clean = target[~np.isnan(target) & ~np.isinf(target)]

    if len(ref_clean) < 5 or len(tgt_clean) < 5:
        return 0.0, 1.0, "Insufficient Data"

    res = stats.ks_2samp(ref_clean, tgt_clean)
    d_stat = float(res.statistic)
    p_val = float(res.pvalue)

    status = "Significant Drift" if p_val < alpha else "Stable (No Drift)"
    return round(d_stat, 4), round(p_val, 6), status


def calculate_mean_std_shift(
    reference: np.ndarray,
    target: np.ndarray
) -> Dict[str, float]:
    """
    Computes parametric shift metrics: mean shift, normalized z-shift, and variance ratio.
    """
    ref_clean = reference[~np.isnan(reference)]
    tgt_clean = target[~np.isnan(target)]

    ref_mean = float(np.mean(ref_clean))
    ref_std = float(np.std(ref_clean))
    tgt_mean = float(np.mean(tgt_clean))
    tgt_std = float(np.std(tgt_clean))

    delta_mean = tgt_mean - ref_mean
    z_mean_shift = delta_mean / (ref_std + 1e-6)
    var_ratio = (tgt_std ** 2) / ((ref_std ** 2) + 1e-6)

    return {
        "ref_mean": round(ref_mean, 4),
        "ref_std": round(ref_std, 4),
        "tgt_mean": round(tgt_mean, 4),
        "tgt_std": round(tgt_std, 4),
        "delta_mean": round(delta_mean, 4),
        "z_mean_shift": round(z_mean_shift, 4),
        "variance_ratio": round(var_ratio, 4)
    }


def compute_feature_drift(
    ref_series: pd.Series,
    tgt_series: pd.Series,
    feature_name: str
) -> Dict[str, Any]:
    """
    Computes all drift metrics for a single feature.
    """
    ref_arr = ref_series.to_numpy(dtype=float)
    tgt_arr = tgt_series.to_numpy(dtype=float)

    psi_score, psi_sev = calculate_psi(ref_arr, tgt_arr)
    ks_stat, ks_pval, ks_status = calculate_ks_test(ref_arr, tgt_arr)
    shift_metrics = calculate_mean_std_shift(ref_arr, tgt_arr)
    
    # Wasserstein distance
    try:
        w_dist = float(stats.wasserstein_distance(ref_arr, tgt_arr))
    except Exception:
        w_dist = 0.0

    # Parse sensor id and sub-feature
    try:
        feat_num = int(feature_name.replace("f", ""))
        sensor_id = ((feat_num - 1) // FEATURES_PER_SENSOR) + 1
        sub_feat = ((feat_num - 1) % FEATURES_PER_SENSOR) + 1
    except ValueError:
        sensor_id = 0
        sub_feat = 0

    return {
        "feature": feature_name,
        "sensor_id": sensor_id,
        "sub_feature": sub_feat,
        "psi_score": psi_score,
        "psi_severity": psi_sev,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "ks_status": ks_status,
        "wasserstein_distance": round(w_dist, 4),
        **shift_metrics
    }


def compute_all_features_drift(
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Scans all 128 features and generates a tabular drift profile.
    """
    feats = feature_cols or [c for c in FEATURE_NAMES if c in df_ref.columns and c in df_target.columns]
    records = []
    
    for f in feats:
        res = compute_feature_drift(df_ref[f], df_target[f], f)
        records.append(res)
        
    return pd.DataFrame(records)


def compute_sensor_level_drift_summary(feature_drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates feature-level drift into sensor-level reliability metrics (16 sensors).
    """
    sensor_records = []
    for s in range(1, NUM_SENSORS + 1):
        s_df = feature_drift_df[feature_drift_df["sensor_id"] == s]
        if s_df.empty:
            continue
            
        avg_psi = float(s_df["psi_score"].mean())
        max_psi = float(s_df["psi_score"].max())
        avg_ks = float(s_df["ks_statistic"].mean())
        sig_ks_count = int((s_df["ks_status"] == "Significant Drift").sum())
        avg_z_shift = float(s_df["z_mean_shift"].abs().mean())
        avg_var_ratio = float(s_df["variance_ratio"].mean())

        # Determine composite sensor drift severity
        if avg_psi < PSI_STABLE_THRESHOLD and sig_ks_count <= 2:
            severity = "Low"
        elif avg_psi < PSI_MODERATE_THRESHOLD or sig_ks_count <= 5:
            severity = "Moderate"
        else:
            severity = "High"

        sensor_records.append({
            "Sensor_ID": f"Sensor {s}",
            "Sensor_Num": s,
            "Sensor_Type": SENSOR_TYPES.get(s, "Figaro MOX"),
            "Avg_PSI": round(avg_psi, 4),
            "Max_PSI": round(max_psi, 4),
            "Avg_KS_Statistic": round(avg_ks, 4),
            "Significant_Features_Count": f"{sig_ks_count} / {FEATURES_PER_SENSOR}",
            "Mean_Z_Shift": round(avg_z_shift, 4),
            "Avg_Variance_Ratio": round(avg_var_ratio, 4),
            "Drift_Severity": severity
        })
        
    return pd.DataFrame(sensor_records)


def compute_batch_drift_trajectory(
    df: pd.DataFrame,
    ref_batch_id: int = 1,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates drift trajectory across all 10 chronological batches against reference batch.
    """
    feats = feature_cols or FEATURE_NAMES
    ref_df = df[df["batch_id"] == ref_batch_id]
    
    trajectory = []
    unique_batches = sorted(df["batch_id"].unique())
    
    for b in unique_batches:
        b_df = df[df["batch_id"] == b]
        if b == ref_batch_id:
            trajectory.append({
                "batch_id": b,
                "avg_psi": 0.0,
                "max_psi": 0.0,
                "avg_ks_stat": 0.0,
                "pct_features_drifted": 0.0,
                "avg_abs_mean_shift": 0.0
            })
            continue

        feat_drifts = compute_all_features_drift(ref_df, b_df, feats)
        avg_psi = float(feat_drifts["psi_score"].mean())
        max_psi = float(feat_drifts["psi_score"].max())
        avg_ks = float(feat_drifts["ks_statistic"].mean())
        drifted_feats = float((feat_drifts["ks_status"] == "Significant Drift").mean() * 100.0)
        avg_shift = float(feat_drifts["z_mean_shift"].abs().mean())

        trajectory.append({
            "batch_id": b,
            "avg_psi": round(avg_psi, 4),
            "max_psi": round(max_psi, 4),
            "avg_ks_stat": round(avg_ks, 4),
            "pct_features_drifted": round(drifted_feats, 2),
            "avg_abs_mean_shift": round(avg_shift, 4)
        })
        
    return pd.DataFrame(trajectory)


DRIFT_CONCEPT_EXPLANATION = """
### Distinguishing Data Drift, Concept Drift, and Physical Sensor Drift

1. **Data Drift (Covariate Shift - $P(X)$ changes)**:
   - The input distribution of sensor electrical features $X$ shifts over time.
   - Causes: Ambient temperature, relative humidity fluctuations, changes in background volatile contaminants, or operational aging.
   
2. **Concept Drift (Conditional Shift - $P(Y|X)$ changes)**:
   - The relationship between sensor array electrical features $X$ and gas class labels $Y$ degrades.
   - Consequence: A machine learning model trained on early batches begins misclassifying gases because the feature signature of ethanol or acetone has shifted into the region previously occupied by another gas.

3. **Physical Sensor Drift (Hardware & Chemical Phenotype)**:
   - Metal Oxide Semiconductor (MOX) sensors rely on active surface adsorption and catalyst-assisted redox reactions.
   - Over months, physical sintering of tin dioxide ($\text{SnO}_2$) grains, chemical poisoning (irreversible sulfur/siloxane binding), and baseline contact resistance creep occur.
   - **Crucial Caution**: Statistical divergence in distribution ($P(X)$) does *not* constitute proof of a damaged sensor; it indicates operational displacement requiring baseline recalibration or drift compensation.
"""
