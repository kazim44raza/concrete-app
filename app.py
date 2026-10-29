"""
AI-Based Prediction of Self-Healing Concrete Compressive Strength
=================================================================
UI/UX Pro Max Clean White Laboratory Edition (Fira Sans + Fira Code)
  - Clean High-Contrast White Theme:
      * Background: #F8FAFC (Alabaster White) & #FFFFFF (Pure White Cards)
      * Text: #0F172A (Deep Slate Ink, >10:1 AAA Contrast) & #475569 (Muted Slate)
      * Accents: #0284C7 (Engineering Blue) & #16A34A (Bio-Emerald Green)
  - All Dataset Validation & Parity Errors Expressed in Percentage (%):
      * MAPE (%): Mean Absolute Percentage Error across mixes
      * RMSPE (%): Root Mean Square Percentage Error across mixes
      * NRMSE (%): Normalized Root Mean Square Error (%)
  - Strictly Uniform Typography System:
      * All Main Section Headings (.main-section-heading): 18px, Font-Weight 700 (Bold)
      * All Card & Plot Sub-Headings (.sub-card-heading) & Expanders: 15px, Font-Weight 700 (Bold)
      * All Navigation Tabs: 14px, Font-Weight 700 (Bold)
      * All Input Parameter Labels: 13px, Font-Weight 600 (Semi-Bold)
      * Monospace Engineering Readouts: Fira Code
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# =============================================================================
# 1. CROSS-VERSION SCIKIT-LEARN & JOBLIB COMPATIBILITY BRIDGES
# =============================================================================

def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

sys.modules["__main__"].normalized_name = normalized_name

# A. Fix for ColumnTransformer _RemainderColsList across scikit-learn versions
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# B. Fix for SimpleImputer _fill_dtype across scikit-learn versions
from sklearn.impute import SimpleImputer
_orig_imputer_transform = SimpleImputer.transform
def _patched_imputer_transform(self, X):
    if not hasattr(self, "_fill_dtype"):
        if hasattr(self, "statistics_") and hasattr(self.statistics_, "dtype"):
            self._fill_dtype = self.statistics_.dtype
        else:
            self._fill_dtype = object
    return _orig_imputer_transform(self, X)
SimpleImputer.transform = _patched_imputer_transform

# C. Fix for OneHotEncoder _infrequent_enabled across scikit-learn versions
from sklearn.preprocessing import OneHotEncoder
_orig_ohe_transform = OneHotEncoder.transform
def _patched_ohe_transform(self, X):
    if not hasattr(self, "_infrequent_enabled"):
        self._infrequent_enabled = (
            getattr(self, "min_frequency", None) is not None
            or getattr(self, "max_categories", None) is not None
        )
    return _orig_ohe_transform(self, X)
OneHotEncoder.transform = _patched_ohe_transform

# D. Fix for CatBoostSklearnWrapper if loading the 4-model ensemble bundle
from sklearn.base import BaseEstimator, RegressorMixin
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

class CatBoostSklearnWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params

    def fit(self, X, y):
        if CatBoostRegressor is not None:
            parameters = {} if self.params is None else dict(self.params)
            self.model_ = CatBoostRegressor(**parameters)
            self.model_.fit(X, y)
            if hasattr(X, "shape") and len(X.shape) == 2:
                self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(X)

sys.modules["__main__"].CatBoostSklearnWrapper = CatBoostSklearnWrapper


# =============================================================================
# 2. PAGE CONFIGURATION & UI/UX PRO MAX CLEAN WHITE THEME (STRICT UNIFORM FONTS)
# =============================================================================

st.set_page_config(
    page_title="AI-Based Prediction of Self-Healing Concrete Compressive Strength",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --color-bg-page: #F8FAFC;
            --color-bg-card: #FFFFFF;
            --color-bg-subtle: #F1F5F9;
            --color-text-primary: #0F172A;
            --color-text-secondary: #334155;
            --color-text-muted: #64748B;
            --color-border: #E2E8F0;
            --color-border-hover: #94A3B8;
            --color-accent-blue: #0284C7;
            --color-accent-green: #16A34A;
            --color-destructive: #DC2626;
            --font-sans: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'Fira Code', monospace;
        }

        /* Global App Font & Clean White Laboratory Surface */
        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: var(--font-sans) !important;
            background-color: var(--color-bg-page) !important;
            color: var(--color-text-primary) !important;
        }

        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1440px !important;
        }

        /* UI/UX Pro Max Stagger & Keyframe Animations */
        @keyframes staggerReveal {
            0% { opacity: 0; transform: translateY(12px) scale(0.99); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes pulseStatus {
            0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.45); }
            50% { transform: scale(1.12); opacity: 0.88; box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); }
        }

        @keyframes kpiShadowPulse {
            0%, 100% { border-color: rgba(2, 132, 199, 0.45); box-shadow: 0 12px 28px -8px rgba(2, 132, 199, 0.14); }
            50% { border-color: rgba(22, 163, 74, 0.50); box-shadow: 0 12px 28px -8px rgba(22, 163, 74, 0.16); }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* Top Engineering Hero Banner (Clean White Card + Blue Accent) */
        .app-hero-banner {
            background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 100%);
            border: 1px solid var(--color-border);
            border-left: 4px solid var(--color-accent-blue);
            border-radius: 14px;
            padding: 20px 26px;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            animation: staggerReveal 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            box-shadow: 0 6px 20px -6px rgba(15, 23, 42, 0.06);
        }

        .app-hero-title {
            font-family: var(--font-sans) !important;
            font-size: 24px !important;
            font-weight: 800 !important;
            color: var(--color-text-primary) !important;
            letter-spacing: -0.01em !important;
            line-height: 1.25 !important;
            margin: 0 0 6px 0 !important;
        }

        .app-hero-subtitle {
            font-family: var(--font-sans) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: var(--color-text-muted) !important;
            margin: 0 !important;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 999px;
            padding: 7px 15px;
            font-family: var(--font-mono) !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            color: #15803D !important;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--color-accent-green);
            animation: pulseStatus 2.2s infinite ease-in-out;
        }

        /* =====================================================================
           STRICT UNIFORM TYPOGRAPHY HIERARCHY (UI/UX PRO MAX — WHITE THEME)
           1. ALL Main Section Headings (.main-section-heading) -> 18px Bold (700)
           2. ALL Sub-Card / Chart Headings (.sub-card-heading) -> 15px Bold (700)
           3. ALL Widget Input Labels -> 13px Semi-Bold (600)
           ===================================================================== */
        .main-section-heading {
            font-family: var(--font-sans) !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            color: var(--color-text-primary) !important;
            letter-spacing: 0.01em !important;
            line-height: 1.35 !important;
            margin: 4px 0 14px 0 !important;
            padding: 11px 16px !important;
            background: linear-gradient(90deg, #E0F2FE 0%, #FFFFFF 100%);
            border: 1px solid #BAE6FD;
            border-left: 3.5px solid var(--color-accent-blue);
            border-radius: 0 10px 10px 0;
            display: flex;
            align-items: center;
            gap: 11px;
            animation: staggerReveal 0.38s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            box-shadow: 0 2px 8px -2px rgba(2, 132, 199, 0.08);
        }

        .step-badge {
            font-family: var(--font-mono) !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
            background: var(--color-accent-blue) !important;
            padding: 3px 9px !important;
            border-radius: 5px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        .sub-card-heading {
            font-family: var(--font-sans) !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            color: var(--color-text-primary) !important;
            letter-spacing: 0.01em !important;
            line-height: 1.4 !important;
            margin: 6px 0 12px 0 !important;
            padding-bottom: 8px !important;
            border-bottom: 1.5px solid var(--color-border) !important;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Uniform Streamlit Expander Headers (15px Bold 700 — White Card) */
        div[data-testid="stExpander"] {
            background: var(--color-bg-card) !important;
            border: 1px solid var(--color-border) !important;
            border-radius: 12px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 3px 12px -4px rgba(15, 23, 42, 0.05) !important;
            transition: border-color 0.22s ease, box-shadow 0.22s ease, transform 0.22s ease !important;
            animation: staggerReveal 0.42s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }

        div[data-testid="stExpander"]:hover {
            border-color: #7DD3FC !important;
            box-shadow: 0 8px 20px -6px rgba(2, 132, 199, 0.12) !important;
        }

        div[data-testid="stExpander"] details summary {
            background-color: #FFFFFF !important;
            border-radius: 12px !important;
        }

        div[data-testid="stExpander"] details summary p {
            font-family: var(--font-sans) !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            color: var(--color-text-primary) !important;
            letter-spacing: 0.01em !important;
        }

        /* Uniform Input Widget Labels (13px Semi-Bold 600) */
        div[data-testid="stWidgetLabel"] p {
            font-family: var(--font-sans) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            color: var(--color-text-secondary) !important;
            letter-spacing: 0.01em !important;
        }

        /* Clean White Input Fields & Selectboxes with Dark Text */
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
            color: var(--color-text-primary) !important;
            border-color: #CBD5E1 !important;
        }

        div[data-testid="stNumberInput"] input {
            font-family: var(--font-mono) !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            color: var(--color-text-primary) !important;
        }

        /* Uniform Tab Labels (14px Bold 700) */
        button[data-baseweb="tab"] {
            cursor: pointer !important;
            background-color: transparent !important;
            transition: all 0.2s ease !important;
        }

        button[data-baseweb="tab"] p {
            font-family: var(--font-sans) !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            color: var(--color-text-secondary) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--color-accent-blue) !important;
        }

        /* Animated Action Buttons with Focus Rings */
        div.stButton > button {
            font-family: var(--font-sans) !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            padding: 0.58rem 1rem !important;
            cursor: pointer !important;
            transition: all 0.22s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        }

        div.stButton > button:focus-visible {
            outline: 2px solid var(--color-accent-blue) !important;
            outline-offset: 2px !important;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
            border: 1px solid #0284C7 !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 16px -4px rgba(2, 132, 199, 0.35) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 22px -4px rgba(2, 132, 199, 0.50) !important;
        }

        div.stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 1.5px solid #CBD5E1 !important;
            color: var(--color-text-primary) !important;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
        }

        div.stButton > button[kind="secondary"]:hover {
            border-color: var(--color-accent-blue) !important;
            color: var(--color-accent-blue) !important;
            background: #F0F9FF !important;
            transform: translateY(-2px) !important;
        }

        /* Animated Hero KPI Prediction Card (Clean White + Soft Sky Glow) */
        .kpi-card {
            background: radial-gradient(circle at top right, rgba(2, 132, 199, 0.08), transparent 65%),
                        linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1.5px solid #7DD3FC;
            border-radius: 14px;
            padding: 24px 20px;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 16px;
            animation: staggerReveal 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) forwards, kpiShadowPulse 4s infinite ease-in-out;
            transition: transform 0.25s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
        }

        .kpi-title {
            font-family: var(--font-sans) !important;
            font-size: 12px !important;
            color: var(--color-text-muted) !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
        }

        .kpi-value {
            font-family: var(--font-mono) !important;
            font-size: 52px !important;
            color: #0284C7 !important;
            font-weight: 700 !important;
            margin: 8px 0 4px 0 !important;
            line-height: 1.1 !important;
        }

        .kpi-unit {
            font-family: var(--font-mono) !important;
            font-size: 16px !important;
            color: var(--color-text-secondary) !important;
            font-weight: 600 !important;
        }

        .grade-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-family: var(--font-sans) !important;
            font-weight: 700 !important;
            font-size: 12.5px !important;
            margin-top: 12px;
            background: #F0FDF4;
            color: #15803D;
            border: 1px solid #86EFAC;
        }

        /* Bento Grid Physical Ratios & Validation Metric Cards (White Cards) */
        .ratio-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 8px;
        }

        .ratio-tile {
            background: #FFFFFF;
            border: 1px solid var(--color-border);
            border-radius: 10px;
            padding: 11px 14px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
            transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        }

        .ratio-tile:hover {
            border-color: #7DD3FC;
            transform: translateY(-1px);
            box-shadow: 0 6px 14px -4px rgba(2, 132, 199, 0.10);
        }

        .ratio-tile-label {
            font-family: var(--font-sans) !important;
            font-size: 11.5px !important;
            font-weight: 600 !important;
            color: var(--color-text-muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }

        .ratio-tile-val {
            font-family: var(--font-mono) !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            color: var(--color-text-primary) !important;
            margin-top: 3px !important;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-top: 8px;
            margin-bottom: 18px;
        }

        .metric-badge {
            background: #FFFFFF;
            border: 1px solid var(--color-border);
            border-top: 3px solid var(--color-accent-blue);
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            box-shadow: 0 3px 10px -3px rgba(15, 23, 42, 0.05);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-badge:hover {
            transform: translateY(-2px);
            border-color: #7DD3FC;
            box-shadow: 0 8px 18px -4px rgba(2, 132, 199, 0.12);
        }

        .metric-badge-label {
            font-family: var(--font-sans) !important;
            font-size: 12px !important;
            color: var(--color-text-muted) !important;
            text-transform: uppercase !important;
            font-weight: 700 !important;
            letter-spacing: 0.05em !important;
        }

        .metric-badge-val {
            font-family: var(--font-mono) !important;
            font-size: 22px !important;
            color: var(--color-accent-blue) !important;
            font-weight: 700 !important;
            margin-top: 4px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_main_heading(step_tag: str, title_text: str) -> None:
    """Renders a strictly uniform 18px Bold (700) main section heading."""
    st.markdown(
        f'<div class="main-section-heading">'
        f'<span class="step-badge">{step_tag}</span>'
        f'<span>{title_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_sub_heading(title_text: str, right_badge: str = "") -> None:
    """Renders a strictly uniform 15px Bold (700) card/chart sub-heading."""
    badge_html = (
        f'<span style="font-family: var(--font-mono); font-size: 12px; color: #0284C7; font-weight: 600;">{right_badge}</span>'
        if right_badge
        else ""
    )
    st.markdown(
        f'<div class="sub-card-heading"><span>{title_text}</span>{badge_html}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. DATASET CONSTANTS, EMBEDDED 100% WHOLE DATASET (N=866) & STRICT BOUNDS
# =============================================================================

TARGET = "Compressive Strength (MPa)"

FINAL_NUMERICAL_VARIABLES: List[str] = [
    "Bacteria(cell/ml)",
    "T(°C) OF CUBE",
    "Cement(kg/m3)",
    "CA(kg/m3)",
    "FA (kg/m3)SAND",
    "WATER(kg/m3)",
    "SuperPlasticizer(kg/m3)",
    "Silica Fumes(kg/m3)",
    "Fly Ash(kg/m3)",
    "Curing Days",
    "Healing Period Days",
    "APPLIED LOAD FOR HEALING(MPa)",
    "Fiber Dosage (kg/m³)",
    "Bacteria Dosage (kg/m³)",
    "Healing Delivery Dosage (kg/m³)",
    "Nutrient Dosage (kg/m³)",
]

FINAL_CATEGORICAL_VARIABLES: List[str] = [
    "Fiber Type",
    "Bacteria Type",
    "Healing Delivery Type",
    "Nutrient Type",
    "Curing Environment",
]

FINAL_INPUT_VARIABLES: List[str] = FINAL_NUMERICAL_VARIABLES + FINAL_CATEGORICAL_VARIABLES
ENCODING: Dict[str, str] = {v: f"X{i+1}" for i, v in enumerate(FINAL_INPUT_VARIABLES)}

FIBER_MAX_DOSAGE: Dict[str, float] = {
    "None": 0.0,
    "Steel Fiber": 117.0,
    "Glass Fiber": 6.39,
    "Basalt Fiber": 4.05,
    "PVA Fiber": 16.04,
}

BACTERIA_MAX_DOSAGE: Dict[str, float] = {
    "None": 0.0,
    "Bacillus Subtilis": 26.7,
    "Rhizobium Leguminosarum": 17.0,
    "Bacillus Sphaericus": 22.5,
    "Bacillus Megaterium": 22.5,
}

NUTRIENT_MAX_DOSAGE: Dict[str, float] = {
    "None": 0.0,
    "Calcium Lactate": 44.323,
    "Calcium Lactate Pentahydrate": 15.4,
    "Calcium Nitrate": 32.0,
}

DELIVERY_OPTIONS: List[str] = [
    "None",
    "Bacteria Only",
    "Sodium Silicate Microcapsule",
    "Cement Microcapsule",
]

CURING_OPTIONS: List[str] = [
    "Fresh Water Curing",
    "Sulfate Water Curing",
    "Humidity Air Curing",
    "Dry Air Curing",
    "Sea Water Curing",
]

# Compressed 100% Whole Dataset (All 866 verified mixes)
_EMBEDDED_FULL_DATASET_B64 = "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"


# =============================================================================
# 4. MODEL LOADER & EXACT v29** FEATURE ENGINEERING
# =============================================================================

@st.cache_resource
def load_bundle() -> Optional[Dict[str, Any]]:
    possible_paths = [
        Path("models/XGBoost_Optuna_Collective21.joblib"),
        Path("XGBoost_Optuna_Collective21.joblib"),
        Path("models/Optuna_Collective21_Ensemble_v29starstar.joblib"),
        Path("Optuna_Collective21_Ensemble_v29starstar.joblib"),
    ]
    target_path = None
    for p in possible_paths:
        if p.exists():
            target_path = p
            break

    if target_path is None:
        st.error(
            "Model `.joblib` file not found. Place `XGBoost_Optuna_Collective21.joblib` "
            "or `Optuna_Collective21_Ensemble_v29starstar.joblib` in the root or `models/` folder."
        )
        return None

    try:
        return joblib.load(target_path)
    except Exception as e:
        st.error(f"Failed to load model bundle: {repr(e)}")
        st.code(traceback.format_exc())
        return None


def safe_divide(n: pd.Series, d: pd.Series) -> pd.Series:
    d_num = pd.to_numeric(d, errors="coerce")
    return n / d_num.where(np.abs(d_num) > 1e-12, np.nan)


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact deterministic physical/experimental feature engineering from v29**."""
    frame = frame.copy()
    cement = pd.to_numeric(frame.get("Cement(kg/m3)"), errors="coerce")
    coarse = pd.to_numeric(frame.get("CA(kg/m3)"), errors="coerce")
    sand = pd.to_numeric(frame.get("FA (kg/m3)SAND"), errors="coerce")
    water = pd.to_numeric(frame.get("WATER(kg/m3)"), errors="coerce")
    silica = pd.to_numeric(frame.get("Silica Fumes(kg/m3)"), errors="coerce")
    fly_ash = pd.to_numeric(frame.get("Fly Ash(kg/m3)"), errors="coerce")
    curing = pd.to_numeric(frame.get("Curing Days"), errors="coerce")
    healing = pd.to_numeric(frame.get("Healing Period Days"), errors="coerce")
    load = pd.to_numeric(frame.get("APPLIED LOAD FOR HEALING(MPa)"), errors="coerce")
    bact_dosage = pd.to_numeric(frame.get("Bacteria Dosage (kg/m³)"), errors="coerce")
    bact_cells = pd.to_numeric(frame.get("Bacteria(cell/ml)"), errors="coerce")
    fiber_dosage = pd.to_numeric(frame.get("Fiber Dosage (kg/m³)"), errors="coerce")
    nutr_dosage = pd.to_numeric(frame.get("Nutrient Dosage (kg/m³)"), errors="coerce")

    if cement is not None:
        supp = (silica.fillna(0.0) if silica is not None else 0.0) + (
            fly_ash.fillna(0.0) if fly_ash is not None else 0.0
        )
        binder = cement + supp
        frame["ENG_Total_Binder"] = binder
        if water is not None:
            frame["ENG_Water_Cement_Ratio"] = safe_divide(water, cement)
            frame["ENG_Water_Binder_Ratio"] = safe_divide(water, binder)
        if coarse is not None and sand is not None:
            agg = coarse + sand
            frame["ENG_Total_Aggregate"] = agg
            frame["ENG_Aggregate_Binder_Ratio"] = safe_divide(agg, binder)
        if fiber_dosage is not None:
            frame["ENG_Fiber_Binder_Ratio"] = safe_divide(fiber_dosage, binder)
        if bact_dosage is not None:
            frame["ENG_BacteriaDosage_Binder_Ratio"] = safe_divide(bact_dosage, binder)
        if nutr_dosage is not None:
            frame["ENG_Nutrient_Binder_Ratio"] = safe_divide(nutr_dosage, binder)

    if load is not None and curing is not None:
        frame["ENG_Load_x_Curing"] = load * curing
        frame["ENG_Load_per_Curing_Day"] = safe_divide(load, curing)
    if load is not None and healing is not None:
        frame["ENG_Load_x_Healing"] = load * healing
        frame["ENG_Load_per_Healing_Day"] = safe_divide(load, healing)
    if curing is not None and healing is not None:
        frame["ENG_Curing_to_Healing_Ratio"] = safe_divide(curing, healing)
    if load is not None and bact_dosage is not None:
        frame["ENG_Load_x_Bacteria_Dosage"] = load * bact_dosage
    if curing is not None and bact_dosage is not None:
        frame["ENG_Curing_x_Bacteria_Dosage"] = curing * bact_dosage
    if bact_cells is not None:
        frame["ENG_Log_Bacteria_Cells"] = np.log1p(bact_cells.clip(lower=0))

    return frame


def build_collective_input_df(params: Dict[str, Any]) -> pd.DataFrame:
    """Build single-row 21-variable collective DataFrame from UI parameters."""
    return pd.DataFrame(
        [
            {
                "Bacteria(cell/ml)": float(params["bacteria_cells"]),
                "T(°C) OF CUBE": float(params["temp_cube"]),
                "Cement(kg/m3)": float(params["cement"]),
                "CA(kg/m3)": float(params["coarse_agg"]),
                "FA (kg/m3)SAND": float(params["sand"]),
                "WATER(kg/m3)": float(params["water"]),
                "SuperPlasticizer(kg/m3)": float(params["sp"]),
                "Silica Fumes(kg/m3)": float(params["silica_fumes"]),
                "Fly Ash(kg/m3)": float(params["fly_ash"]),
                "Curing Days": float(params["curing_days"]),
                "Healing Period Days": float(params["healing_days"]),
                "APPLIED LOAD FOR HEALING(MPa)": float(params["applied_load"]),
                "Fiber Dosage (kg/m³)": float(params["fiber_dosage"]),
                "Bacteria Dosage (kg/m³)": float(params["bacteria_dosage"]),
                "Healing Delivery Dosage (kg/m³)": float(params["delivery_dosage"]),
                "Nutrient Dosage (kg/m³)": float(params["nutrient_dosage"]),
                "Fiber Type": str(params["fiber_type"]),
                "Bacteria Type": str(params["bacteria_type"]),
                "Healing Delivery Type": str(params["delivery_type"]),
                "Nutrient Type": str(params["nutrient_type"]),
                "Curing Environment": str(params["curing_env"]),
            }
        ]
    )


def predict_batch_from_bundle(bundle: Dict[str, Any], df_collective: pd.DataFrame) -> np.ndarray:
    df_eng = add_domain_features(df_collective)
    schema = bundle["feature_schema"]
    df_prep = df_eng[list(schema.keys())].copy()
    for col, role in schema.items():
        if role == "numeric":
            df_prep[col] = pd.to_numeric(df_prep[col], errors="coerce")
        else:
            df_prep[col] = df_prep[col].astype(str)

    df_prep = df_prep.drop(columns=bundle.get("dropped_unusable_columns", []), errors="ignore")
    df_model = df_prep[bundle["model_input_columns"]]

    if bundle.get("selected_strategy") == "ensemble":
        preds = np.column_stack(
            [bundle["models"][name].predict(df_model) for name in bundle["model_order"]]
        )
        weights = np.array([bundle["weights"][name] for name in bundle["model_order"]])
        return np.asarray(preds @ weights, dtype=float)
    elif "pipeline" in bundle:
        return np.asarray(bundle["pipeline"].predict(df_model), dtype=float)
    else:
        strat = bundle.get("selected_strategy", bundle["model_order"][0])
        return np.asarray(bundle["models"][strat].predict(df_model), dtype=float)


def predict_from_bundle(bundle: Dict[str, Any], df_raw_inputs: pd.DataFrame) -> float:
    return float(predict_batch_from_bundle(bundle, df_raw_inputs)[0])


def classify_structural_grade(mpa: float) -> Tuple[str, str]:
    if mpa < 25.0:
        return "Low / Non-Structural Grade (< 25 MPa)", "#D97706"
    elif mpa < 45.0:
        return "Standard Structural Concrete (C25–C45)", "#16A34A"
    elif mpa < 65.0:
        return "High-Strength Bio-Concrete (C45–C65)", "#0284C7"
    else:
        return "Ultra-High-Performance Bio-Concrete (> 65 MPa)", "#2563EB"


@st.cache_data
def get_whole_dataset_and_shap_results(_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes Whole-Dataset (100% Data, N=866) R² Parity Plot arrays, Percentage (%)
    Error metrics (MAPE %, RMSPE %, NRMSE %), and Whole-Dataset SHAP values (X1–X21).
    """
    raw_json = zlib.decompress(base64.b64decode(_EMBEDDED_FULL_DATASET_B64)).decode("utf-8")
    payload = json.loads(raw_json)
    full_df = pd.DataFrame(payload["full_records"])

    X_full_coll = full_df[FINAL_INPUT_VARIABLES].copy()
    y_full_actual = pd.to_numeric(full_df[TARGET], errors="coerce").to_numpy(dtype=float)

    # 1. Whole-Dataset R² Parity Predictions & Percentage (%) Error Metrics
    if "whole_parity_data" in _bundle and isinstance(_bundle["whole_parity_data"], dict):
        wp = _bundle["whole_parity_data"]
        y_actual = np.asarray(wp["y_actual"], dtype=float)
        y_pred = np.asarray(wp["y_pred"], dtype=float)
    else:
        y_actual = y_full_actual
        y_pred = predict_batch_from_bundle(_bundle, X_full_coll)

    safe_actual = np.where(np.abs(y_actual) > 1e-12, y_actual, 1e-12)
    pct_diff = np.abs(y_actual - y_pred) / safe_actual * 100.0

    ss_res = float(np.sum((y_actual - y_pred) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    r2_full = float(1.0 - ss_res / max(ss_tot, 1e-12))
    mape_pct = float(np.mean(pct_diff))
    rmspe_pct = float(np.sqrt(np.mean((pct_diff / 100.0) ** 2)) * 100.0)
    rmse_mpa = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
    nrmse_pct = float((rmse_mpa / max(float(np.mean(y_actual)), 1e-12)) * 100.0)

    if "whole_dataset_metrics" in _bundle and isinstance(_bundle["whole_dataset_metrics"], dict):
        m = _bundle["whole_dataset_metrics"]
        r2_full = float(m.get("R2", r2_full))
        mape_pct = float(m.get("MAPE_pct", mape_pct))
        rmspe_pct = float(m.get("RMSPE_pct", rmspe_pct))
        nrmse_pct = float(m.get("NRMSE_pct", nrmse_pct))

    # 2. Whole-Dataset SHAP Values (X1–X21 Global + X1–X16 Numerical Beeswarm)
    if "whole_shap_data" in _bundle and isinstance(_bundle["whole_shap_data"], dict):
        ws = _bundle["whole_shap_data"]
        imp_df = pd.DataFrame(ws["importance_records"])
        if "Relative_SHAP_Importance_Pct" not in imp_df.columns:
            tot_s = max(float(imp_df["Mean_Absolute_SHAP_MPa"].sum()), 1e-12)
            imp_df["Relative_SHAP_Importance_Pct"] = (imp_df["Mean_Absolute_SHAP_MPa"] / tot_s) * 100.0
        shap_num = np.asarray(ws["shap_values_num"], dtype=float)
        feat_num = np.asarray(ws["feature_values_num"], dtype=float)
        num_names = list(ws["numerical_feature_names"])
    else:
        baseline_row = X_full_coll.iloc[0:1].copy()
        for col in FINAL_NUMERICAL_VARIABLES:
            baseline_row[col] = float(pd.to_numeric(X_full_coll[col], errors="coerce").median())
        for col in FINAL_CATEGORICAL_VARIABLES:
            mode_vals = X_full_coll[col].mode()
            baseline_row[col] = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "None"

        n_rows = len(X_full_coll)
        n_vars = len(FINAL_INPUT_VARIABLES)
        shap_matrix = np.zeros((n_rows, n_vars), dtype=float)

        for j, var in enumerate(FINAL_INPUT_VARIABLES):
            perturbed = X_full_coll.copy()
            perturbed[var] = baseline_row[var].iloc[0]
            pred_without = predict_batch_from_bundle(_bundle, perturbed)
            shap_matrix[:, j] = y_pred - pred_without

        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        tot_abs = max(float(np.sum(mean_abs)), 1e-12)
        mean_abs_pct = (mean_abs / tot_abs) * 100.0
        display_names = [f"{ENCODING[v]}: {v}" for v in FINAL_INPUT_VARIABLES]
        imp_df = pd.DataFrame(
            {
                "Feature_ID": [ENCODING[v] for v in FINAL_INPUT_VARIABLES],
                "Feature": FINAL_INPUT_VARIABLES,
                "Display_Name": display_names,
                "Feature_Type": [
                    "Numerical" if v in FINAL_NUMERICAL_VARIABLES else "Categorical"
                    for v in FINAL_INPUT_VARIABLES
                ],
                "Mean_Absolute_SHAP_MPa": mean_abs,
                "Relative_SHAP_Importance_Pct": mean_abs_pct,
            }
        ).sort_values("Relative_SHAP_Importance_Pct", ascending=False)

        num_indices = [FINAL_INPUT_VARIABLES.index(v) for v in FINAL_NUMERICAL_VARIABLES]
        shap_num = shap_matrix[:, num_indices]
        feat_num = (
            X_full_coll[FINAL_NUMERICAL_VARIABLES]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        num_names = [f"{ENCODING[v]}: {v}" for v in FINAL_NUMERICAL_VARIABLES]

    return {
        "y_actual": y_actual,
        "y_pred": y_pred,
        "pct_diff": pct_diff,
        "r2": r2_full,
        "mape_pct": mape_pct,
        "rmspe_pct": rmspe_pct,
        "nrmse_pct": nrmse_pct,
        "shap_importance_df": imp_df,
        "shap_num": shap_num,
        "feat_num": feat_num,
        "num_names": num_names,
    }


def generate_suggested_mix(
    bundle: Dict[str, Any], base_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], float]:
    candidates: List[Dict[str, Any]] = []

    c1 = dict(base_params)
    c1["cement"] = float(np.clip(max(base_params["cement"], 450.0), 171.6, 650.0))
    c1["water"] = float(np.clip(min(base_params["water"], 150.0), 122.5, 300.0))
    c1["silica_fumes"] = 45.0
    c1["sp"] = max(base_params["sp"], 2.25)
    c1["bacteria_type"] = "Bacillus Megaterium"
    c1["bacteria_dosage"] = 22.5
    c1["bacteria_cells"] = 2.0e9
    c1["delivery_type"] = "Bacteria Only"
    c1["delivery_dosage"] = 0.0
    c1["nutrient_type"] = "Calcium Lactate"
    c1["nutrient_dosage"] = 2.25
    c1["curing_days"] = max(int(base_params["curing_days"]), 56)
    candidates.append(c1)

    c2 = dict(c1)
    c2["fiber_type"] = "Steel Fiber"
    c2["fiber_dosage"] = 78.0
    c2["bacteria_type"] = "Bacillus Sphaericus"
    c2["bacteria_dosage"] = 4.0
    c2["fly_ash"] = 80.0
    candidates.append(c2)

    c3 = dict(base_params)
    c3["cement"] = 475.0
    c3["water"] = 148.9
    c3["coarse_agg"] = 940.1
    c3["sand"] = 704.5
    c3["sp"] = 15.675
    c3["silica_fumes"] = 47.5
    c3["curing_days"] = max(int(base_params["curing_days"]), 28)
    candidates.append(c3)

    best_cand = dict(base_params)
    best_pred = predict_from_bundle(bundle, build_collective_input_df(base_params))

    for cand in candidates:
        pred = predict_from_bundle(bundle, build_collective_input_df(cand))
        if pred > best_pred:
            best_pred = pred
            best_cand = cand

    return best_cand, best_pred


def compute_local_feature_contributions(
    bundle: Dict[str, Any], current_params: Dict[str, Any], current_pred: float
) -> pd.DataFrame:
    baseline_params = {
        "cement": 400.0,
        "water": 160.0,
        "sand": 1000.0,
        "coarse_agg": 600.0,
        "silica_fumes": 0.0,
        "fly_ash": 0.0,
        "sp": 2.0,
        "temp_cube": 30.0,
        "fiber_type": "None",
        "fiber_dosage": 0.0,
        "bacteria_type": "None",
        "bacteria_dosage": 0.0,
        "bacteria_cells": 0.0,
        "delivery_type": "None",
        "delivery_dosage": 0.0,
        "nutrient_type": "None",
        "nutrient_dosage": 0.0,
        "curing_env": "Fresh Water Curing",
        "curing_days": 28,
        "healing_days": 0,
        "applied_load": 0.0,
    }
    feature_groups = [
        ("X3: Cement Content", ["cement"]),
        ("X6: Mixing Water", ["water"]),
        ("X4 & X5: Coarse & Fine Aggregates", ["coarse_agg", "sand"]),
        ("X8 & X9: Silica Fume & Fly Ash", ["silica_fumes", "fly_ash"]),
        ("X7: Superplasticizer", ["sp"]),
        ("X2: Cube Temperature", ["temp_cube"]),
        ("X13 & X17: Fiber Reinforcement", ["fiber_type", "fiber_dosage"]),
        ("X1, X14 & X18: Bacteria Agent & Cells", ["bacteria_type", "bacteria_dosage", "bacteria_cells"]),
        ("X15 & X19: Delivery Carrier", ["delivery_type", "delivery_dosage"]),
        ("X16 & X20: Chemical Nutrient", ["nutrient_type", "nutrient_dosage"]),
        ("X10 & X21: Curing Environment & Days", ["curing_env", "curing_days"]),
        ("X11 & X12: Healing Period & Load", ["healing_days", "applied_load"]),
    ]

    rows = []
    safe_base = max(abs(current_pred), 1e-12)
    for label, keys in feature_groups:
        perturbed = dict(current_params)
        for k in keys:
            perturbed[k] = baseline_params[k]
        pred_without = predict_from_bundle(bundle, build_collective_input_df(perturbed))
        delta_mpa = current_pred - pred_without
        delta_pct = (delta_mpa / safe_base) * 100.0
        rows.append({"Feature": label, "Impact_Pct": float(delta_pct), "Impact_MPa": float(delta_mpa)})

    df_imp = pd.DataFrame(rows)
    df_imp["Abs_Impact"] = df_imp["Impact_Pct"].abs()
    df_imp = df_imp.sort_values("Abs_Impact", ascending=True)
    return df_imp


# =============================================================================
# 5. POP-UP MODAL DIALOG (USER PREDICTION RESULT ONLY — NO DATASET ERROR)
# =============================================================================

_dialog_decorator = getattr(st, "dialog", getattr(st, "experimental_dialog", None))

if _dialog_decorator is not None:
    @_dialog_decorator("Self-Healing Concrete Prediction Result")
    def show_prediction_popup(
        pred_val: float,
        wc_ratio: float,
        wb_ratio: float,
        total_b: float,
        curing_env: str,
        curing_days: int,
        bacteria_type: str,
    ) -> None:
        grade_text, _ = classify_structural_grade(pred_val)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Predicted Compressive Strength</div>
                <div class="kpi-value">{pred_val:.2f}</div>
                <div class="kpi-unit">MPa</div>
                <div class="grade-pill">{grade_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_sub_heading("User Mix Design Summary", f"{curing_days}d Curing")
        st.markdown(
            f"""
            <div class="ratio-grid">
                <div class="ratio-tile">
                    <div class="ratio-tile-label">W/C Ratio</div>
                    <div class="ratio-tile-val">{wc_ratio:.3f}</div>
                </div>
                <div class="ratio-tile">
                    <div class="ratio-tile-label">W/B Ratio</div>
                    <div class="ratio-tile-val">{wb_ratio:.3f}</div>
                </div>
                <div class="ratio-tile">
                    <div class="ratio-tile-label">Total Binder</div>
                    <div class="ratio-tile-val">{total_b:.1f} kg/m³</div>
                </div>
                <div class="ratio-tile">
                    <div class="ratio-tile-label">Bio-Healing Agent</div>
                    <div class="ratio-tile-val" style="font-size:13.5px;">{bacteria_type}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Close Pop-Up", type="primary", use_container_width=True):
            st.rerun()


# =============================================================================
# 6. MAIN APPLICATION WORKFLOW
# =============================================================================

st.markdown(
    """
    <div class="app-hero-banner">
        <div>
            <h1 class="app-hero-title">AI-Based Prediction of Self-Healing Concrete Compressive Strength</h1>
            <p class="app-hero-subtitle">
                Optuna-Tuned Machine Learning Engine (100% Whole-Dataset + 5-Fold OOF CV Architecture) &bull;
                Verified Range: 9.06 &ndash; 115.13 MPa (N = 866)
            </p>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>TELEMETRY READY &bull; % ERROR NORMALIZED</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle = load_bundle()

col_left, col_right = st.columns([1.85, 1.15], gap="large")

with col_left:
    render_main_heading("STEP 01", "Input Mix & Self-Healing Parameters")

    with st.expander("Section 1: Base Mix Design (Continuous Inputs)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            cement = st.number_input(
                "Cement (kg/m³)", min_value=171.60, max_value=650.00, value=400.00, step=5.0
            )
            water = st.number_input(
                "Water (kg/m³)", min_value=122.50, max_value=300.00, value=160.00, step=2.5
            )
            sand = st.number_input(
                "Fine Aggregate / Sand (kg/m³)", min_value=350.00, max_value=1500.00, value=1000.00, step=10.0
            )
        with c2:
            coarse_agg = st.number_input(
                "Coarse Aggregate (kg/m³)", min_value=0.00, max_value=1477.74, value=600.00, step=10.0
            )
            silica_fumes = st.number_input(
                "Silica Fumes (kg/m³)", min_value=0.00, max_value=50.00, value=0.00, step=1.0
            )
            fly_ash = st.number_input(
                "Fly Ash (kg/m³)", min_value=0.00, max_value=350.00, value=0.00, step=5.0
            )
        with c3:
            sp = st.number_input(
                "Superplasticizer (kg/m³)", min_value=0.00, max_value=15.675, value=2.00, step=0.1
            )
            temp_cube = st.number_input(
                "Cube Temperature (°C)", min_value=22.0, max_value=200.0, value=30.0, step=1.0
            )

    with st.expander("Section 2: Self-Healing Agents & Fiber Reinforcement", expanded=True):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fiber_type = st.selectbox("Fiber Type", list(FIBER_MAX_DOSAGE.keys()))
        with f_col2:
            max_f = FIBER_MAX_DOSAGE[fiber_type]
            fiber_dosage = st.number_input(
                "Fiber Dosage (kg/m³)",
                min_value=0.0,
                max_value=max(0.01, max_f),
                value=0.0 if fiber_type == "None" else min(2.0, max_f),
                disabled=(fiber_type == "None"),
            )
            if fiber_type == "None":
                fiber_dosage = 0.0

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            bacteria_type = st.selectbox("Bacteria Type", list(BACTERIA_MAX_DOSAGE.keys()))
            max_b = BACTERIA_MAX_DOSAGE[bacteria_type]
            bacteria_dosage = st.number_input(
                "Bacteria Dosage (kg/m³)",
                min_value=0.0,
                max_value=max(0.01, max_b),
                value=0.0 if bacteria_type == "None" else min(4.0, max_b),
                disabled=(bacteria_type == "None"),
            )
            if bacteria_type == "None":
                bacteria_dosage = 0.0
        with b_col2:
            bacteria_cells = st.number_input(
                "Concentration (cell/ml)",
                min_value=0.0,
                max_value=2.8e9,
                value=0.0 if bacteria_type == "None" else 1.0e7,
                step=1.0e6,
                format="%.2e",
                disabled=(bacteria_type == "None"),
            )
            if bacteria_type == "None":
                bacteria_cells = 0.0
        with b_col3:
            delivery_type = st.selectbox("Delivery Carrier", DELIVERY_OPTIONS)
            max_del = (
                12.25
                if delivery_type in ("Sodium Silicate Microcapsule", "Cement Microcapsule")
                else 0.0
            )
            delivery_dosage = st.number_input(
                "Carrier Dosage (kg/m³)",
                min_value=0.0,
                max_value=max(0.01, max_del),
                value=0.0 if max_del == 0.0 else 5.0,
                disabled=(max_del == 0.0),
            )
            if max_del == 0.0:
                delivery_dosage = 0.0

        n_col1, n_col2 = st.columns(2)
        with n_col1:
            nutrient_type = st.selectbox("Nutrient Substrate", list(NUTRIENT_MAX_DOSAGE.keys()))
        with n_col2:
            max_nutr = NUTRIENT_MAX_DOSAGE[nutrient_type]
            nutrient_dosage = st.number_input(
                "Nutrient Dosage (kg/m³)",
                min_value=0.0,
                max_value=max(0.01, max_nutr),
                value=0.0 if nutrient_type == "None" else min(2.0, max_nutr),
                disabled=(nutrient_type == "None"),
            )
            if nutrient_type == "None":
                nutrient_dosage = 0.0

    with st.expander("Section 3: Environmental Curing & Testing Conditions", expanded=True):
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            curing_env = st.selectbox("Curing Exposure", CURING_OPTIONS)
        with e_col2:
            curing_days = st.number_input(
                "Curing Time (Days)", min_value=3, max_value=180, value=28, step=1
            )
        with e_col3:
            healing_days = st.number_input(
                "Healing Period (Days)", min_value=0, max_value=124, value=0, step=1
            )
        with e_col4:
            applied_load = st.number_input(
                "Pre-Damage Load (MPa)", min_value=0.0, max_value=60.3, value=0.0, step=1.0
            )


current_params = {
    "cement": cement,
    "water": water,
    "sand": sand,
    "coarse_agg": coarse_agg,
    "silica_fumes": silica_fumes,
    "fly_ash": fly_ash,
    "sp": sp,
    "temp_cube": temp_cube,
    "fiber_type": fiber_type,
    "fiber_dosage": fiber_dosage,
    "bacteria_type": bacteria_type,
    "bacteria_dosage": bacteria_dosage,
    "bacteria_cells": bacteria_cells,
    "delivery_type": delivery_type,
    "delivery_dosage": delivery_dosage,
    "nutrient_type": nutrient_type,
    "nutrient_dosage": nutrient_dosage,
    "curing_env": curing_env,
    "curing_days": curing_days,
    "healing_days": healing_days,
    "applied_load": applied_load,
}


with col_right:
    render_main_heading("STEP 03", "Prediction Output & Diagnostics")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_clicked = st.button(
            "Predict Compressive Strength", type="primary", use_container_width=True
        )
    with btn_col2:
        suggest_clicked = st.button(
            "Get Suggested Mix", use_container_width=True
        )

    if bundle is not None:
        try:
            input_row = build_collective_input_df(current_params)
            pred_val = predict_from_bundle(bundle, input_row)
            total_b = cement + silica_fumes + fly_ash
            wc_ratio = water / cement
            wb_ratio = water / total_b
            agg_b_ratio = (coarse_agg + sand) / total_b
            grade_text, _ = classify_structural_grade(pred_val)

            if predict_clicked and _dialog_decorator is not None:
                show_prediction_popup(
                    pred_val,
                    wc_ratio,
                    wb_ratio,
                    total_b,
                    curing_env,
                    int(curing_days),
                    bacteria_type,
                )

            # User Prediction KPI Card — Cleanly shows ONLY the predicted strength (no dataset error)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Predicted Compressive Strength</div>
                    <div class="kpi-value">{pred_val:.2f}</div>
                    <div class="kpi-unit">MPa</div>
                    <div class="grade-pill">{grade_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if go is not None:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=pred_val,
                        number={"suffix": " MPa", "font": {"size": 24, "color": "#0F172A", "family": "Fira Code"}},
                        gauge={
                            "axis": {"range": [0, 120], "tickwidth": 1, "tickcolor": "#64748B"},
                            "bar": {"color": "#0284C7"},
                            "bgcolor": "#FFFFFF",
                            "borderwidth": 1,
                            "bordercolor": "#CBD5E1",
                            "steps": [
                                {"range": [0, 25], "color": "rgba(245, 158, 11, 0.14)"},
                                {"range": [25, 45], "color": "rgba(22, 163, 74, 0.14)"},
                                {"range": [45, 65], "color": "rgba(2, 132, 199, 0.16)"},
                                {"range": [65, 120], "color": "rgba(37, 99, 235, 0.18)"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=195,
                    margin=dict(l=20, r=20, t=18, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#0F172A", "family": "Fira Sans"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            render_sub_heading("Mix Physical Ratios", "Real-Time")
            st.markdown(
                f"""
                <div class="ratio-grid">
                    <div class="ratio-tile">
                        <div class="ratio-tile-label">Water / Cement (W/C)</div>
                        <div class="ratio-tile-val">{wc_ratio:.3f}</div>
                    </div>
                    <div class="ratio-tile">
                        <div class="ratio-tile-label">Water / Binder (W/B)</div>
                        <div class="ratio-tile-val">{wb_ratio:.3f}</div>
                    </div>
                    <div class="ratio-tile">
                        <div class="ratio-tile-label">Total Binder</div>
                        <div class="ratio-tile-val">{total_b:.1f} kg/m³</div>
                    </div>
                    <div class="ratio-tile">
                        <div class="ratio-tile-label">Aggregate / Binder</div>
                        <div class="ratio-tile-val">{agg_b_ratio:.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as pred_err:
            st.error(f"Prediction calculation error: {repr(pred_err)}")
            st.code(traceback.format_exc())


# =============================================================================
# 7. STEP 2 & STEP 4: SUGGESTED MIX + WHOLE-DATASET R² PARITY & SHAP PLOTS (%)
# =============================================================================

if bundle is not None:
    st.write("")
    render_main_heading("STEP 04 & 02", "Model Interpretability, Validation (% Error) & Mix Optimization")

    tab_r2_shap, tab_local, tab_suggest = st.tabs(
        [
            "Step 4A: Whole-Dataset R² Parity Plot (% Error) & SHAP Plots (N=866)",
            "Step 4B: Current Mix Local Sensitivity (% Impact Waterfall)",
            "Step 2: Suggested Mix Proportion Comparison",
        ]
    )

    with tab_r2_shap:
        full_results = get_whole_dataset_and_shap_results(bundle)
        y_act = full_results["y_actual"]
        y_prd = full_results["y_pred"]
        pct_diff = full_results["pct_diff"]
        r2_f = full_results["r2"]
        mape_pct = full_results["mape_pct"]
        rmspe_pct = full_results["rmspe_pct"]
        nrmse_pct = full_results["nrmse_pct"]
        imp_df = full_results["shap_importance_df"]
        shap_num = full_results["shap_num"]
        feat_num = full_results["feat_num"]
        num_names = full_results["num_names"]

        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-badge">
                    <div class="metric-badge-label">Whole-Dataset R² (N={len(y_act)})</div>
                    <div class="metric-badge-val">{r2_f:.4f}</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-badge-label">Mean Abs. % Error (MAPE)</div>
                    <div class="metric-badge-val">{mape_pct:.2f}%</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-badge-label">RMS % Error (RMSPE)</div>
                    <div class="metric-badge-val">{rmspe_pct:.2f}%</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-badge-label">Normalized RMSE (NRMSE)</div>
                    <div class="metric-badge-val">{nrmse_pct:.2f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        plot_col1, plot_col2 = st.columns(2, gap="large")

        with plot_col1:
            render_sub_heading(
                "1. Whole-Dataset R² Parity Plot (Colored by % Error)",
                f"MAPE = {mape_pct:.2f}% | RMSPE = {rmspe_pct:.2f}%",
            )
            if go is not None:
                mn_v = float(min(y_act.min(), y_prd.min()))
                mx_v = float(max(y_act.max(), y_prd.max()))
                slope, intercept = np.polyfit(y_act, y_prd, 1)
                x_line = np.linspace(mn_v, mx_v, 100)

                fig_r2 = go.Figure()
                fig_r2.add_trace(
                    go.Scatter(
                        x=y_act,
                        y=y_prd,
                        mode="markers",
                        name=f"Whole Dataset Mixes (N={len(y_act)})",
                        marker=dict(
                            size=8,
                            color=pct_diff,
                            colorscale="YlGnBu",
                            line=dict(width=0.6, color="#0F172A"),
                            opacity=0.85,
                        ),
                        customdata=np.column_stack([pct_diff]),
                        hovertemplate=(
                            "Actual: %{x:.2f} MPa<br>"
                            "Predicted: %{y:.2f} MPa<br>"
                            "Percentage Error: %{customdata[0]:.2f}%<extra></extra>"
                        ),
                    )
                )
                fig_r2.add_trace(
                    go.Scatter(
                        x=[mn_v, mx_v],
                        y=[mn_v, mx_v],
                        mode="lines",
                        name="1:1 Ideal Line",
                        line=dict(color="#DC2626", width=2, dash="dash"),
                    )
                )
                fig_r2.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=slope * x_line + intercept,
                        mode="lines",
                        name=f"Fit Line (R² = {r2_f:.4f})",
                        line=dict(color="#0284C7", width=2.5),
                    )
                )
                fig_r2.update_layout(
                    xaxis=dict(
                        title="Actual Experimental Compressive Strength (MPa)",
                        gridcolor="#E2E8F0",
                        zerolinecolor="#CBD5E1",
                    ),
                    yaxis=dict(
                        title="Predicted Compressive Strength (MPa)",
                        gridcolor="#E2E8F0",
                        zerolinecolor="#CBD5E1",
                    ),
                    height=460,
                    margin=dict(l=20, r=20, t=25, b=20),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font={"color": "#0F172A", "family": "Fira Sans"},
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            else:
                st.dataframe(pd.DataFrame({"Actual_MPa": y_act, "Predicted_MPa": y_prd, "Error_Pct": pct_diff}))

        with plot_col2:
            render_sub_heading("2. Whole-Dataset Global SHAP Feature Importance (%) (X1–X21)", "Relative % Impact")
            bar_df = imp_df.sort_values("Relative_SHAP_Importance_Pct", ascending=True)
            if go is not None:
                bar_colors = [
                    "#0284C7" if t == "Numerical" else "#16A34A"
                    for t in bar_df["Feature_Type"]
                ]
                fig_shap_bar = go.Figure(
                    go.Bar(
                        x=bar_df["Relative_SHAP_Importance_Pct"],
                        y=bar_df["Display_Name"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:.2f}%" for v in bar_df["Relative_SHAP_Importance_Pct"]],
                        textposition="outside",
                    )
                )
                fig_shap_bar.update_layout(
                    xaxis=dict(
                        title="Relative SHAP Importance (%) Across Whole Dataset",
                        gridcolor="#E2E8F0",
                    ),
                    yaxis=dict(
                        title="Collective Input Variable (X1–X21)",
                        gridcolor="#F1F5F9",
                    ),
                    height=460,
                    margin=dict(l=20, r=55, t=25, b=20),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font={"color": "#0F172A", "family": "Fira Sans"},
                )
                st.plotly_chart(fig_shap_bar, use_container_width=True)
            else:
                st.dataframe(bar_df, use_container_width=True)

        render_sub_heading("3. Whole-Dataset Numerical SHAP Summary / Beeswarm Plot (X1–X16)", "N = 866 Mixes")
        if go is not None:
            mean_y_val = max(float(np.mean(y_act)), 1e-12)
            shap_num_pct = (shap_num / mean_y_val) * 100.0
            mean_num_order = np.argsort(np.mean(np.abs(shap_num_pct), axis=0))
            rng_jitter = np.random.RandomState(42)
            fig_bee = go.Figure()
            for rank_pos, idx in enumerate(mean_num_order):
                s_vals = shap_num_pct[:, idx]
                f_vals = feat_num[:, idx]
                f_min, f_max = float(np.min(f_vals)), float(np.max(f_vals))
                f_norm = (f_vals - f_min) / max(f_max - f_min, 1e-12)
                y_jit = rank_pos + rng_jitter.uniform(-0.24, 0.24, size=len(s_vals))
                fig_bee.add_trace(
                    go.Scatter(
                        x=s_vals,
                        y=y_jit,
                        mode="markers",
                        showlegend=False,
                        marker=dict(
                            size=6,
                            color=f_norm,
                            colorscale="Bluered",
                            opacity=0.78,
                            showscale=(rank_pos == len(mean_num_order) - 1),
                            colorbar=dict(
                                title="Feature Value<br>(Blue=Low, Red=High)",
                                tickvals=[0, 1],
                                ticktext=["Low", "High"],
                            )
                            if rank_pos == len(mean_num_order) - 1
                            else None,
                        ),
                        customdata=np.column_stack([f_vals]),
                        hovertemplate=(
                            f"<b>{num_names[idx]}</b><br>"
                            "SHAP Impact: %{x:+.2f}%<br>"
                            "Feature Value: %{customdata[0]:.3g}<extra></extra>"
                        ),
                    )
                )
            fig_bee.add_vline(x=0.0, line_width=1.5, line_dash="dash", line_color="#64748B")
            fig_bee.update_layout(
                xaxis=dict(
                    title="Normalized SHAP Impact (% of Mean Compressive Strength)",
                    gridcolor="#E2E8F0",
                ),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(mean_num_order))),
                    ticktext=[num_names[i] for i in mean_num_order],
                    gridcolor="#F1F5F9",
                ),
                height=520,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font={"color": "#0F172A", "family": "Fira Sans"},
            )
            st.plotly_chart(fig_bee, use_container_width=True)

    with tab_local:
        render_sub_heading("Single-Mix Local Parameter Contribution (% Impact vs. Control Baseline)", "Local % Sensitivity")
        df_imp = compute_local_feature_contributions(bundle, current_params, pred_val)
        if go is not None:
            colors = ["#16A34A" if v >= 0 else "#DC2626" for v in df_imp["Impact_Pct"]]
            fig_imp = go.Figure(
                go.Bar(
                    x=df_imp["Impact_Pct"],
                    y=df_imp["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f}%" for v in df_imp["Impact_Pct"]],
                    textposition="outside",
                )
            )
            fig_imp.update_layout(
                xaxis=dict(
                    title="Relative Strength Contribution (% Change)",
                    gridcolor="#E2E8F0",
                ),
                yaxis=dict(
                    title="Parameter Group",
                    gridcolor="#F1F5F9",
                ),
                height=420,
                margin=dict(l=20, r=55, t=25, b=20),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font={"color": "#0F172A", "family": "Fira Sans"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.dataframe(df_imp[["Feature", "Impact_Pct"]], use_container_width=True)

    with tab_suggest:
        sug_params, sug_pred = generate_suggested_mix(bundle, current_params)
        gain_pct = ((sug_pred - pred_val) / max(abs(pred_val), 1e-12)) * 100.0
        render_sub_heading(
            "Suggested High-Performance Bio-Concrete Mix Comparison",
            f"Target: {sug_pred:.2f} MPa ({gain_pct:+.2f}%)",
        )
        comp_df = pd.DataFrame(
            {
                "Parameter": [
                    "Cement (kg/m³)",
                    "Water (kg/m³)",
                    "Silica Fumes (kg/m³)",
                    "Fly Ash (kg/m³)",
                    "Superplasticizer (kg/m³)",
                    "Fiber Type",
                    "Fiber Dosage (kg/m³)",
                    "Bacteria Type",
                    "Bacteria Dosage (kg/m³)",
                    "Nutrient Type",
                    "Nutrient Dosage (kg/m³)",
                    "Curing Days",
                    "Predicted Strength (MPa)",
                    "Relative Strength Gain (%)",
                ],
                "Your Current Mix": [
                    f"{cement:.1f}",
                    f"{water:.1f}",
                    f"{silica_fumes:.1f}",
                    f"{fly_ash:.1f}",
                    f"{sp:.2f}",
                    fiber_type,
                    f"{fiber_dosage:.2f}",
                    bacteria_type,
                    f"{bacteria_dosage:.2f}",
                    nutrient_type,
                    f"{nutrient_dosage:.2f}",
                    f"{curing_days}",
                    f"{pred_val:.2f} MPa",
                    "Baseline (0.00%)",
                ],
                "Suggested High-Performance Mix": [
                    f"{sug_params['cement']:.1f}",
                    f"{sug_params['water']:.1f}",
                    f"{sug_params['silica_fumes']:.1f}",
                    f"{sug_params['fly_ash']:.1f}",
                    f"{sug_params['sp']:.2f}",
                    sug_params["fiber_type"],
                    f"{sug_params['fiber_dosage']:.2f}",
                    sug_params["bacteria_type"],
                    f"{sug_params['bacteria_dosage']:.2f}",
                    sug_params["nutrient_type"],
                    f"{sug_params['nutrient_dosage']:.2f}",
                    f"{sug_params['curing_days']}",
                    f"{sug_pred:.2f} MPa",
                    f"{gain_pct:+.2f}%",
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
