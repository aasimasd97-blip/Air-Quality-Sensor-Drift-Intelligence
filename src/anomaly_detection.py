"""
Anomaly Detection module using Isolation Forest for sensor reading surveillance.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    ANOMALY_DETECTOR_PATH,
    RANDOM_STATE
)


class SensorAnomalyDetector:
    """
    Multivariate anomaly detector based on Isolation Forest.
    Trained on baseline reference data (Batch 1) to identify out-of-distribution
    measurements in later observation periods.
    """
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, df_reference: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "SensorAnomalyDetector":
        """
        Fits Isolation Forest on baseline reference measurements (e.g. Batch 1).
        """
        feats = feature_cols or FEATURE_NAMES
        X_ref = df_reference[feats]
        self.model.fit(X_ref)
        self.is_fitted = True
        return self

    def score_samples(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Computes anomaly scores, binary flags, and readable labels for input measurements.
        
        Returns DataFrame with:
        - anomaly_raw_score: raw decision_function output (negative values indicate anomalies)
        - anomaly_normalized_score: 0.0 (very normal) to 1.0 (extreme anomaly)
        - is_anomaly: True/False
        - anomaly_label: 'Anomalous' / 'Normal'
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector model is not fitted.")
            
        feats = feature_cols or FEATURE_NAMES
        X = df[feats]
        
        raw_scores = self.model.decision_function(X)
        preds = self.model.predict(X)  # -1 for anomaly, 1 for normal
        
        # Convert raw decision score to 0..1 scale (higher = more anomalous)
        # Decision function is typically centered around 0 (-0.3 to +0.3)
        norm_scores = 1.0 / (1.0 + np.exp(5.0 * raw_scores))
        
        results = df.copy()
        results["anomaly_raw_score"] = raw_scores
        results["anomaly_score"] = np.clip(norm_scores, 0.0, 1.0)
        results["is_anomaly"] = (preds == -1)
        results["anomaly_label"] = np.where(preds == -1, "Anomalous", "Normal")
        
        return results

    def batch_anomaly_summary(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Computes anomaly count and percentage across batches.
        """
        records = []
        for b in sorted(df_scored["batch_id"].unique()):
            b_df = df_scored[df_scored["batch_id"] == b]
            total = len(b_df)
            anom_count = int(b_df["is_anomaly"].sum())
            pct = (anom_count / total * 100.0) if total > 0 else 0.0
            avg_score = float(b_df["anomaly_score"].mean())
            
            records.append({
                "batch_id": int(b),
                "total_samples": total,
                "anomalous_samples": anom_count,
                "normal_samples": total - anom_count,
                "anomaly_rate_pct": round(pct, 2),
                "avg_anomaly_score": round(avg_score, 4)
            })
            
        return pd.DataFrame(records)

    def save(self, filepath: Optional[Path] = None) -> Path:
        target_path = filepath or ANOMALY_DETECTOR_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target_path)
        print(f"[AnomalyDetector] Saved model to {target_path}")
        return target_path

    def load(self, filepath: Optional[Path] = None) -> "SensorAnomalyDetector":
        target_path = filepath or ANOMALY_DETECTOR_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Anomaly detector file not found at {target_path}")
        self.model = joblib.load(target_path)
        self.is_fitted = True
        return self


if __name__ == "__main__":
    from src.data_loader import load_processed_data
    from src.preprocessing import ChronologicalPreprocessor

    df = load_processed_data()
    prep = ChronologicalPreprocessor().load()
    scaled_df = prep.transform(df)

    # Fit on Batch 1
    batch1_scaled = scaled_df[scaled_df["batch_id"] == 1]
    detector = SensorAnomalyDetector(contamination=0.05)
    detector.fit(batch1_scaled)
    detector.save()

    scored_df = detector.score_samples(scaled_df)
    summary = detector.batch_anomaly_summary(scored_df)
    
    print("\n--- Batch Anomaly Summary (Baseline: Batch 1) ---")
    print(summary.to_string(index=False))
