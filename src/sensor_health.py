"""
Sensor Reliability and Health Scoring Intelligence Module.
Generates an explainable 0-100 composite health score for each of the 16 array sensors.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    NUM_SENSORS,
    FEATURES_PER_SENSOR,
    SENSOR_TYPES,
    HEALTH_WEIGHT_DRIFT,
    HEALTH_WEIGHT_ANOMALY,
    HEALTH_WEIGHT_VOLATILITY,
    HEALTH_WEIGHT_DEGRADATION,
    HEALTH_CATEGORIES
)
from src.drift_detection import (
    compute_all_features_drift,
    compute_sensor_level_drift_summary
)


class SensorHealthScorer:
    """
    Computes explainable, multi-factor reliability and health scores (0-100)
    for each sensor in the array.
    
    Composite Components:
    1. Drift Penalty (Weight 0.40): based on PSI and KS statistic against baseline.
    2. Anomaly Penalty (Weight 0.25): based on frequency of out-of-distribution readings.
    3. Volatility / Instability Penalty (Weight 0.20): based on variance inflation & z-shift.
    4. Model Degradation Contribution (Weight 0.15): attribution from classification drop.
    """

    @staticmethod
    def calculate_health_scores(
        sensor_drift_df: pd.DataFrame,
        batch_anomaly_rate: float = 10.0,
        model_accuracy_drop: float = 0.20
    ) -> pd.DataFrame:
        """
        Calculates health scores and actionable calibration recommendations
        for all 16 sensors based on empirical drift metrics.
        """
        results = []

        for _, row in sensor_drift_df.iterrows():
            s_num = int(row["Sensor_Num"])
            avg_psi = float(row["Avg_PSI"])
            avg_ks = float(row["Avg_KS_Statistic"])
            mean_z_shift = float(row["Mean_Z_Shift"])
            var_ratio = float(row["Avg_Variance_Ratio"])

            # 1. Drift Penalty (0 to 100):
            # PSI < 0.10 -> low penalty; PSI > 0.50 -> high penalty
            # KS stat is in [0, 1]
            psi_penalty = min(100.0, (avg_psi / 0.50) * 100.0)
            ks_penalty = avg_ks * 100.0
            drift_score_component = 100.0 - (0.6 * psi_penalty + 0.4 * ks_penalty)
            drift_score_component = max(0.0, min(100.0, drift_score_component))

            # 2. Anomaly Penalty (0 to 100):
            anomaly_penalty = min(100.0, (batch_anomaly_rate / 40.0) * 100.0)
            anomaly_score_component = max(0.0, 100.0 - anomaly_penalty)

            # 3. Volatility / Instability Score (0 to 100):
            # Penalizes large z-shifts and severe variance inflation
            shift_penalty = min(100.0, (abs(mean_z_shift) / 2.0) * 100.0)
            log_var = abs(np.log10(max(1e-3, var_ratio)))
            var_penalty = min(100.0, (log_var / 2.0) * 100.0)
            volatility_component = 100.0 - (0.5 * shift_penalty + 0.5 * var_penalty)
            volatility_component = max(0.0, min(100.0, volatility_component))

            # 4. Model Degradation Component (0 to 100):
            deg_penalty = min(100.0, (model_accuracy_drop / 0.50) * 100.0)
            deg_component = max(0.0, 100.0 - deg_penalty)

            # Composite Health Score
            composite_score = (
                HEALTH_WEIGHT_DRIFT * drift_score_component +
                HEALTH_WEIGHT_ANOMALY * anomaly_score_component +
                HEALTH_WEIGHT_VOLATILITY * volatility_component +
                HEALTH_WEIGHT_DEGRADATION * deg_component
            )
            composite_score = round(max(0.0, min(100.0, composite_score)), 1)

            # Categorize
            if composite_score >= 80.0:
                category = "Healthy"
                status_color = HEALTH_CATEGORIES["Healthy"]["color"]
                action = "Continue regular monitoring. Sensor response profile is consistent with baseline calibration."
            elif composite_score >= 60.0:
                category = "Monitor"
                status_color = HEALTH_CATEGORIES["Monitor"]["color"]
                action = "Placed on Watch List. Minor distribution shift observed; schedule routine zero-gas check."
            elif composite_score >= 40.0:
                category = "Warning"
                status_color = HEALTH_CATEGORIES["Warning"]["color"]
                action = "Calibration Advised. Notable baseline drift or variance expansion; cross-validate with reference gas."
            else:
                category = "Critical"
                status_color = HEALTH_CATEGORIES["Critical"]["color"]
                action = "Action Required. Severe drift or responsiveness distortion detected; perform full baseline resistance recalibration and hardware check."

            results.append({
                "Sensor_ID": f"Sensor {s_num}",
                "Sensor_Num": s_num,
                "Sensor_Type": row["Sensor_Type"],
                "Health_Score": composite_score,
                "Category": category,
                "Status_Color": status_color,
                "Drift_Component": round(drift_score_component, 1),
                "Anomaly_Component": round(anomaly_score_component, 1),
                "Stability_Component": round(volatility_component, 1),
                "Degradation_Component": round(deg_component, 1),
                "Recommendation": action
            })

        return pd.DataFrame(results).sort_values("Sensor_Num").reset_index(drop=True)


SENSOR_HEALTH_DISCLAIMER = """
> [!NOTE]
> **Heuristic Architecture Disclaimer**:
> This Sensor Health Score (0–100) is an **analytical heuristic** derived from statistical drift (PSI, KS test),
> out-of-distribution anomaly frequencies, variance inflation, and classifier degradation. It is designed for
> predictive maintenance decision support and does **not** constitute a physical or factory-certified hardware calibration measurement.
> Replacement of sensor hardware should never be initiated based solely on statistical divergence without physical instrument inspection.
"""
