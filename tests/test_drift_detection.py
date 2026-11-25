"""
Unit tests for drift detection algorithms and sensor health scoring.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.drift_detection import (
    calculate_psi,
    calculate_ks_test,
    calculate_mean_std_shift
)
from src.sensor_health import SensorHealthScorer


def test_psi_identical_distributions():
    """Identical distributions should produce PSI near 0 and Stable severity."""
    np.random.seed(42)
    dist1 = np.random.normal(loc=0.0, scale=1.0, size=1000)
    dist2 = np.random.normal(loc=0.0, scale=1.0, size=1000)

    psi_score, severity = calculate_psi(dist1, dist2, num_bins=10)
    assert psi_score < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi_score}"
    assert severity == "Stable (No Drift)"


def test_psi_shifted_distributions():
    """Significantly shifted distribution should produce PSI >= 0.25."""
    np.random.seed(42)
    dist_base = np.random.normal(loc=0.0, scale=1.0, size=1000)
    dist_shifted = np.random.normal(loc=2.5, scale=2.0, size=1000)

    psi_score, severity = calculate_psi(dist_base, dist_shifted, num_bins=10)
    assert psi_score >= 0.25, f"Expected PSI >= 0.25 for shifted distributions, got {psi_score}"
    assert severity == "Significant Drift"


def test_ks_test():
    """KS test should detect difference with low p-value."""
    np.random.seed(42)
    base = np.random.uniform(0, 10, 500)
    shifted = np.random.uniform(5, 15, 500)

    d_stat, p_val, status = calculate_ks_test(base, shifted)
    assert d_stat > 0.3
    assert p_val < 0.001
    assert status == "Significant Drift"


def test_health_score_boundaries():
    """Sensor health score must strictly reside in [0, 100]."""
    mock_sensor_summary = pd.DataFrame([{
        "Sensor_Num": s,
        "Sensor_ID": f"Sensor {s}",
        "Sensor_Type": "TGS2600",
        "Avg_PSI": 1.5,
        "Max_PSI": 3.0,
        "Avg_KS_Statistic": 0.8,
        "Significant_Features_Count": "8 / 8",
        "Mean_Z_Shift": 4.5,
        "Avg_Variance_Ratio": 50.0,
        "Drift_Severity": "High"
    } for s in range(1, 17)])

    health_df = SensorHealthScorer.calculate_health_scores(
        mock_sensor_summary,
        batch_anomaly_rate=80.0,
        model_accuracy_drop=0.70
    )

    for score in health_df["Health_Score"]:
        assert 0.0 <= score <= 100.0, f"Score out of bounds: {score}"
