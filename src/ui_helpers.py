"""
Shared UI components, custom CSS styling, and cached resource loaders for Streamlit.
"""
from pathlib import Path
from typing import Tuple, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DATASET_PATH,
    PROCESSED_CSV_PATH,
    SAMPLE_DATASET_PATH,
    GAS_CLASSIFIER_PATH,
    ANOMALY_DETECTOR_PATH,
    SCALER_PATH,
    PCA_MODEL_PATH
)
from src.data_loader import load_processed_data
from src.preprocessing import ChronologicalPreprocessor
from src.gas_classification import load_gas_classifier
from src.anomaly_detection import SensorAnomalyDetector
from src.feature_engineering import PCADriftAnalyzer


def apply_custom_styles():
    """Applies clean, modern SaaS styling to the Streamlit app."""
    st.markdown("""
    <style>
        /* General page typography and padding */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        
        /* Metric Card styling */
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 12px;
        }
        .metric-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 4px;
        }
        .metric-card-delta {
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 4px;
        }

        /* Status Pills */
        .status-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-align: center;
        }
        .pill-healthy { background-color: #DCFCE7; color: #166534; }
        .pill-monitor { background-color: #FEF9C3; color: #854D0E; }
        .pill-warning { background-color: #FFEDD5; color: #9A3412; }
        .pill-critical { background-color: #FEE2E2; color: #991B1B; }

        /* Clean Sidebar styling */
        .css-1d391kg {
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    """Loads and caches the consolidated 13,910-row dataset."""
    return load_processed_data()


@st.cache_data(show_spinner=False)
def get_cached_scaled_dataset() -> pd.DataFrame:
    """Returns dataset transformed by the chronologically-fitted scaler."""
    df = get_cached_dataset()
    prep = get_cached_preprocessor()
    return prep.transform(df)


@st.cache_resource(show_spinner=False)
def get_cached_preprocessor() -> ChronologicalPreprocessor:
    """Loads and caches the fitted ChronologicalPreprocessor."""
    prep = ChronologicalPreprocessor()
    prep.load()
    return prep


@st.cache_resource(show_spinner=False)
def get_cached_gas_classifier() -> Tuple[Any, Dict[str, Any]]:
    """Loads and caches trained gas classifier and metadata."""
    return load_gas_classifier()


@st.cache_resource(show_spinner=False)
def get_cached_anomaly_detector() -> SensorAnomalyDetector:
    """Loads and caches fitted Isolation Forest detector."""
    det = SensorAnomalyDetector()
    det.load()
    return det


@st.cache_resource(show_spinner=False)
def get_cached_pca_analyzer() -> PCADriftAnalyzer:
    """Loads and caches fitted PCA analyzer."""
    pca_analyzer = PCADriftAnalyzer()
    pca_analyzer.load()
    return pca_analyzer


def render_sidebar_info():
    """Renders informative metadata and links in Streamlit sidebar."""
    with st.sidebar:
        st.markdown("### 🔬 Sensor Drift Intelligence")
        st.caption("UCI Gas Sensor Array Drift Dataset (13,910 measurements over 36 months)")
        st.markdown("---")
        st.markdown("""
        **Project Overview**
        - **16 Chemical Sensors** (Figaro TGS Series)
        - **128 Extracted Features** (Steady + Dynamic)
        - **6 Analyte Classes**
        - **10 Chronological Batches**
        """)
        st.markdown("---")
        st.caption("Antigravity Data Science Architecture")
