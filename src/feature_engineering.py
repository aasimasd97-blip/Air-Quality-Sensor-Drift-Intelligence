"""
Feature engineering and dimensionality reduction module for sensor drift analysis.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    NUM_SENSORS,
    FEATURES_PER_SENSOR,
    PCA_MODEL_PATH,
    RANDOM_STATE
)


class SensorFeatureExtractor:
    """
    Extracts sensor-level domain features from raw 128 multi-sensor array readings:
    - Steady-state magnitude per sensor (f1, f2)
    - Transient dynamic energy per sensor (f3-f8)
    - Sensor-wise mean and variance profiles
    """
    
    @staticmethod
    def get_sensor_columns(sensor_id: int) -> List[str]:
        """Returns the 8 feature names for a given sensor (1-16)."""
        start = (sensor_id - 1) * FEATURES_PER_SENSOR + 1
        return [f"f{i}" for i in range(start, start + FEATURES_PER_SENSOR)]

    @staticmethod
    def compute_sensor_level_aggregates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates sensor summary features:
        - sensor_{i}_steady_state (mean of f1, f2)
        - sensor_{i}_transient_energy (mean of abs(f3-f8))
        - sensor_{i}_overall_mean
        """
        agg_df = pd.DataFrame(index=df.index)
        if "batch_id" in df.columns:
            agg_df["batch_id"] = df["batch_id"]
        if "gas_class" in df.columns:
            agg_df["gas_class"] = df["gas_class"]
        if "gas_name" in df.columns:
            agg_df["gas_name"] = df["gas_name"]
            
        for s in range(1, NUM_SENSORS + 1):
            s_cols = SensorFeatureExtractor.get_sensor_columns(s)
            s_data = df[s_cols]
            agg_df[f"sensor_{s}_steady_state"] = s_data[s_cols[:2]].mean(axis=1)
            agg_df[f"sensor_{s}_transient_energy"] = s_data[s_cols[2:]].abs().mean(axis=1)
            agg_df[f"sensor_{s}_mean"] = s_data.mean(axis=1)
            agg_df[f"sensor_{s}_std"] = s_data.std(axis=1)

        return agg_df

    @staticmethod
    def get_sensor_correlation_matrix(df: pd.DataFrame, batch_id: Optional[int] = None) -> pd.DataFrame:
        """
        Computes inter-sensor correlation matrix (using sensor mean responses).
        """
        data = df[df["batch_id"] == batch_id] if batch_id is not None else df
        sensor_means = pd.DataFrame()
        for s in range(1, NUM_SENSORS + 1):
            s_cols = SensorFeatureExtractor.get_sensor_columns(s)
            sensor_means[f"Sensor_{s}"] = data[s_cols].mean(axis=1)
        return sensor_means.corr()


class PCADriftAnalyzer:
    """
    Fits PCA on baseline batch (e.g. Batch 1) to establish reference manifold.
    Projects subsequent batches onto this space to visualize and quantify spatial drift.
    """
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        self.is_fitted = False

    def fit(self, df_reference: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "PCADriftAnalyzer":
        feats = feature_cols or FEATURE_NAMES
        self.pca.fit(df_reference[feats])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("PCA model is not fitted.")
        feats = feature_cols or FEATURE_NAMES
        components = self.pca.transform(df[feats])
        pca_df = pd.DataFrame(
            components,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
            index=df.index
        )
        if "batch_id" in df.columns:
            pca_df["batch_id"] = df["batch_id"]
        if "gas_class" in df.columns:
            pca_df["gas_class"] = df["gas_class"]
        if "gas_name" in df.columns:
            pca_df["gas_name"] = df["gas_name"]
        return pca_df

    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def save(self, filepath: Optional[Path] = None) -> Path:
        target_path = filepath or PCA_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, target_path)
        return target_path

    def load(self, filepath: Optional[Path] = None) -> "PCADriftAnalyzer":
        target_path = filepath or PCA_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"PCA model file not found at {target_path}")
        self.pca = joblib.load(target_path)
        self.is_fitted = True
        self.n_components = self.pca.n_components_
        return self


if __name__ == "__main__":
    from src.data_loader import load_processed_data
    from src.preprocessing import ChronologicalPreprocessor

    df = load_processed_data()
    preprocessor = ChronologicalPreprocessor().load()
    scaled_df = preprocessor.transform(df)

    # Fit PCA on scaled Batch 1
    batch1_scaled = scaled_df[scaled_df["batch_id"] == 1]
    pca_analyzer = PCADriftAnalyzer(n_components=3)
    pca_analyzer.fit(batch1_scaled)
    pca_analyzer.save()

    print("[PCA] Explained Variance Ratio:", pca_analyzer.explained_variance_ratio())
    pca_df = pca_analyzer.transform(scaled_df)
    print("[PCA] Successfully transformed full dataset. Shape:", pca_df.shape)
