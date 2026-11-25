"""
Configuration and constants for Air Quality Sensor Drift Intelligence.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_RAW_DIR = RAW_DATA_DIR / "Dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for path in [PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Processed File Paths
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "gas_drift_processed.parquet"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "gas_drift_processed.csv"
SAMPLE_DATASET_PATH = SAMPLE_DATA_DIR / "sample_batch_data.csv"
REFERENCE_DATASET_PATH = PROCESSED_DATA_DIR / "reference_batch1.csv"

# Model Paths
GAS_CLASSIFIER_PATH = MODELS_DIR / "gas_classifier.pkl"
ANOMALY_DETECTOR_PATH = MODELS_DIR / "anomaly_detector.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
PCA_MODEL_PATH = MODELS_DIR / "pca_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Gas Classes Mapping (UCI Gas Sensor Array Drift Dataset specification)
GAS_CLASSES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene"
}

GAS_COLORS = {
    "Ethanol": "#3366CC",
    "Ethylene": "#109618",
    "Ammonia": "#FF9900",
    "Acetaldehyde": "#DC3912",
    "Acetone": "#990099",
    "Toluene": "#0099C6"
}

# Sensor Configuration
# 16 chemical sensors (Figaro commercial sensors)
# 4 of each: TGS2600, TGS2602, TGS2610, TGS2620
SENSOR_TYPES = {
    1: "TGS2600 (General Air Quality / VOCs)",
    2: "TGS2602 (VOCs / Odor / Ammonia)",
    3: "TGS2610 (LP Gas / Combustible)",
    4: "TGS2620 (Solvent Vapors / Alcohol)",
    5: "TGS2600 (General Air Quality / VOCs)",
    6: "TGS2602 (VOCs / Odor / Ammonia)",
    7: "TGS2610 (LP Gas / Combustible)",
    8: "TGS2620 (Solvent Vapors / Alcohol)",
    9: "TGS2600 (General Air Quality / VOCs)",
    10: "TGS2602 (VOCs / Odor / Ammonia)",
    11: "TGS2610 (LP Gas / Combustible)",
    12: "TGS2620 (Solvent Vapors / Alcohol)",
    13: "TGS2600 (General Air Quality / VOCs)",
    14: "TGS2602 (VOCs / Odor / Ammonia)",
    15: "TGS2610 (LP Gas / Combustible)",
    16: "TGS2620 (Solvent Vapors / Alcohol)"
}

NUM_SENSORS = 16
FEATURES_PER_SENSOR = 8
TOTAL_FEATURES = NUM_SENSORS * FEATURES_PER_SENSOR  # 128 features

FEATURE_NAMES = [f"f{i}" for i in range(1, TOTAL_FEATURES + 1)]

# Feature naming mapping: sensor_idx (1-16) and sub-feature (1-8)
def get_sensor_feature_name(sensor_id: int, sub_feature: int) -> str:
    """Returns standardized column name for sensor sub-feature: e.g. s1_f1"""
    return f"s{sensor_id}_f{sub_feature}"

def get_sensor_features(sensor_id: int) -> list:
    """Returns list of 8 feature column names for given sensor id (1-16)"""
    start_idx = (sensor_id - 1) * FEATURES_PER_SENSOR + 1
    return [f"f{i}" for i in range(start_idx, start_idx + FEATURES_PER_SENSOR)]

# Sub-feature metadata
FEATURE_DESCRIPTIONS = {
    1: "Steady-state maximum resistance change (ΔR = max|R(t) - R0|)",
    2: "Normalized steady-state response (ΔR / R0)",
    3: "EMA rising transient response (fast filter)",
    4: "EMA rising transient response (medium filter)",
    5: "EMA rising transient response (slow filter)",
    6: "EMA decaying transient response (fast filter)",
    7: "EMA decaying transient response (medium filter)",
    8: "EMA decaying transient response (slow filter)"
}

# Drift Thresholds
PSI_STABLE_THRESHOLD = 0.10
PSI_MODERATE_THRESHOLD = 0.25
KS_ALPHA = 0.05

# Health Scoring Weights
HEALTH_WEIGHT_DRIFT = 0.40
HEALTH_WEIGHT_ANOMALY = 0.25
HEALTH_WEIGHT_VOLATILITY = 0.20
HEALTH_WEIGHT_DEGRADATION = 0.15

# Health Categories
HEALTH_CATEGORIES = {
    "Healthy": {"min": 80, "max": 100, "color": "#28a745", "status": "Normal Operation"},
    "Monitor": {"min": 60, "max": 79, "color": "#ffc107", "status": "Watch List"},
    "Warning": {"min": 40, "max": 59, "color": "#fd7e14", "status": "Calibration Advised"},
    "Critical": {"min": 0, "max": 39, "color": "#dc3545", "status": "Action Required"}
}

# Random Seed for Reproducibility
RANDOM_STATE = 42
