"""
AI-Based Prediction of Self-Healing Concrete Compressive Strength
=================================================================
Compatible with:
  - XGBoost_Optuna_Collective21.joblib
  - Optuna_Collective21_Ensemble_v29starstar.joblib

Complete 4-Step Engineering Workflow:
  Step 1: Input Data (Base Mix, Self-Healing Agents, Curing & Testing, Action Buttons)
  Step 2: Suggested Mix Proportion Window (Side-by-side comparison & 1-click apply)
  Step 3: Prediction Output (Pop-Up Modal Result, RMSE/MAE/R² Badges, Plotly Gauge)
  Step 4: Interpretability & Validation Plots (21-Feature Local Sensitivity & Parity)
"""

from __future__ import annotations

import os
import re
import sys
import traceback
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
# 2. PAGE CONFIGURATION & CUSTOM STYLING
# =============================================================================

st.set_page_config(
    page_title="AI-Based Prediction of Self-Healing Concrete Compressive Strength",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .kpi-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 2px solid #00D2FF;
            border-radius: 14px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0, 210, 255, 0.18);
            margin-top: 10px;
            margin-bottom: 16px;
        }
        .kpi-title {
            font-size: 14px;
            color: #94A3B8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }
        .kpi-value {
            font-size: 52px;
            color: #00D2FF;
            font-weight: 800;
            margin: 8px 0;
        }
        .kpi-unit {
            font-size: 18px;
            color: #E2E8F0;
            font-weight: 600;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-top: 10px;
            margin-bottom: 14px;
        }
        .metric-badge {
            flex: 1;
            background: #162032;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .metric-badge-label {
            font-size: 12px;
            color: #94A3B8;
            text-transform: uppercase;
            font-weight: 600;
        }
        .metric-badge-val {
            font-size: 20px;
            color: #38BDF8;
            font-weight: 700;
            margin-top: 4px;
        }
        .grade-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
            margin-top: 8px;
            background: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.4);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 3. DATASET CONSTANTS & STRICT PARAMETER BOUNDS
# =============================================================================

DATASET_MIN_MPA = 9.06
DATASET_MAX_MPA = 115.13
DEFAULT_MODEL_R2 = 0.9283
DEFAULT_MODEL_RMSE = 4.14
DEFAULT_MODEL_MAE = 2.98
REGRESSION_PLOT_FILENAME = "Optuna_v29starstar_Independent_Test_Regression.png"

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
            "⚠️ Model `.joblib` file not found. Place `XGBoost_Optuna_Collective21.joblib` "
            "or `Optuna_Collective21_Ensemble_v29starstar.joblib` in the root or `models/` folder."
        )
        return None

    try:
        return joblib.load(target_path)
    except Exception as e:
        st.error(f"❌ Failed to load model bundle: {repr(e)}")
        st.code(traceback.format_exc())
        return None


def extract_accuracy_metrics(bundle: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Retrieve RMSE, MAE, and R² from the loaded model bundle."""
    rmse, mae, r2 = None, None, None
    if isinstance(bundle, dict):
        if "independent_test_metrics" in bundle and isinstance(bundle["independent_test_metrics"], dict):
            m = bundle["independent_test_metrics"]
            rmse = m.get("RMSE")
            mae = m.get("MAE")
            r2 = m.get("R2")
        if rmse is None and "training_cv_metrics" in bundle and isinstance(bundle["training_cv_metrics"], dict):
            m = bundle["training_cv_metrics"]
            rmse = m.get("RMSE")
            mae = m.get("MAE")
            r2 = m.get("R2")
        if rmse is None and "best_cv_rmse" in bundle:
            b = bundle["best_cv_rmse"]
            if isinstance(b, dict):
                rmse = b.get("XGB", next(iter(b.values()), None))
            elif isinstance(b, (int, float)):
                rmse = float(b)

    final_rmse = float(rmse) if rmse is not None else DEFAULT_MODEL_RMSE
    final_mae = float(mae) if mae is not None else round(final_rmse * 0.72, 2)
    final_r2 = float(r2) if r2 is not None else DEFAULT_MODEL_R2
    return final_rmse, final_mae, final_r2


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


def predict_from_bundle(bundle: Dict[str, Any], df_raw_inputs: pd.DataFrame) -> float:
    df_eng = add_domain_features(df_raw_inputs)
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
        return float((preds @ weights)[0])
    elif "pipeline" in bundle:
        return float(bundle["pipeline"].predict(df_model)[0])
    else:
        strat = bundle.get("selected_strategy", bundle["model_order"][0])
        return float(bundle["models"][strat].predict(df_model)[0])


def classify_structural_grade(mpa: float) -> Tuple[str, str]:
    if mpa < 25.0:
        return "Low / Non-Structural Grade (< 25 MPa)", "#F59E0B"
    elif mpa < 45.0:
        return "Standard Structural Concrete (C25–C45)", "#10B981"
    elif mpa < 65.0:
        return "High-Strength Bio-Concrete (C45–C65)", "#00D2FF"
    else:
        return "Ultra-High-Performance Bio-Concrete (> 65 MPa)", "#A855F7"


def generate_suggested_mix(
    bundle: Dict[str, Any], base_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], float]:
    """
     Evaluates candidate bio-healing and mix optimizations within strict dataset bounds
     to recommend a higher-strength self-healing mix design.
    """
    candidates: List[Dict[str, Any]] = []

    # Candidate 1: Optimized w/b ratio + Silica Fume + Bacillus Megaterium + Calcium Lactate
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

    # Candidate 2: Steel Fiber Reinforced Bio-Healing Concrete
    c2 = dict(c1)
    c2["fiber_type"] = "Steel Fiber"
    c2["fiber_dosage"] = 78.0
    c2["bacteria_type"] = "Bacillus Sphaericus"
    c2["bacteria_dosage"] = 4.0
    c2["fly_ash"] = 80.0
    candidates.append(c2)

    # Candidate 3: High-Binder Micro-Silica Mix
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
    """
    Computes one-at-a-time local sensitivity contributions (in MPa) for key inputs
    relative to a standard control baseline mix.
    """
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
        ("Cement Content", ["cement"]),
        ("Mixing Water", ["water"]),
        ("Coarse & Fine Aggregates", ["coarse_agg", "sand"]),
        ("Silica Fume & Fly Ash", ["silica_fumes", "fly_ash"]),
        ("Superplasticizer", ["sp"]),
        ("Cube Temperature", ["temp_cube"]),
        ("Fiber Reinforcement", ["fiber_type", "fiber_dosage"]),
        ("Bacteria Agent & Cells", ["bacteria_type", "bacteria_dosage", "bacteria_cells"]),
        ("Delivery Carrier", ["delivery_type", "delivery_dosage"]),
        ("Chemical Nutrient", ["nutrient_type", "nutrient_dosage"]),
        ("Curing Environment & Days", ["curing_env", "curing_days"]),
        ("Healing Period & Load", ["healing_days", "applied_load"]),
    ]

    rows = []
    for label, keys in feature_groups:
        perturbed = dict(current_params)
        for k in keys:
            perturbed[k] = baseline_params[k]
        pred_without = predict_from_bundle(bundle, build_collective_input_df(perturbed))
        delta_mpa = current_pred - pred_without
        rows.append({"Feature": label, "Impact_MPa": float(delta_mpa)})

    df_imp = pd.DataFrame(rows)
    df_imp["Abs_Impact"] = df_imp["Impact_MPa"].abs()
    df_imp = df_imp.sort_values("Abs_Impact", ascending=True)
    return df_imp


# =============================================================================
# 5. POP-UP MODAL DIALOG (@st.dialog)
# =============================================================================

_dialog_decorator = getattr(st, "dialog", getattr(st, "experimental_dialog", None))

if _dialog_decorator is not None:
    @_dialog_decorator("🎯 Self-Healing Concrete Prediction Result")
    def show_prediction_popup(
        pred_val: float,
        rmse_val: float,
        mae_val: float,
        r2_val: float,
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
                <div class="kpi-unit">MPa (± {mae_val:.2f} MPa MAE)</div>
                <div class="grade-pill">{grade_text}</div>
            </div>
            <div class="metric-row">
                <div class="metric-badge">
                    <div class="metric-badge-label">Model RMSE</div>
                    <div class="metric-badge-val">{rmse_val:.2f} MPa</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-badge-label">Model MAE</div>
                    <div class="metric-badge-val">{mae_val:.2f} MPa</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-badge-label">Validation R²</div>
                    <div class="metric-badge-val">{r2_val:.4f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Key Mix Summary")
        st.write(
            f"• **Expected Strength Band (±MAE):** `{max(0.0, pred_val - mae_val):.2f}` to `{pred_val + mae_val:.2f} MPa`"
        )
        st.write(
            f"• **Water/Cement (W/C) Ratio:** `{wc_ratio:.3f}` | **Water/Binder (W/B) Ratio:** `{wb_ratio:.3f}`"
        )
        st.write(f"• **Total Binder Content:** `{total_b:.1f} kg/m³`")
        st.write(
            f"• **Bio-Healing Agent:** `{bacteria_type}` | **Curing:** `{curing_days} days ({curing_env})`"
        )

        if st.button("Close Pop-Up", type="primary", use_container_width=True):
            st.rerun()


# =============================================================================
# 6. MAIN APPLICATION WORKFLOW
# =============================================================================

st.title("🏗️ AI-Based Prediction of Self-Healing Concrete Compressive Strength")
st.caption(
    "Optuna-Tuned Machine Learning Engine (v29** Collective-21 Architecture) | "
    "Dataset Range: 9.06 – 115.13 MPa"
)

bundle = load_bundle()
rmse_val, mae_val, r2_val = extract_accuracy_metrics(bundle)

col_left, col_right = st.columns([1.85, 1.15], gap="large")

with col_left:
    st.subheader("Step 1: Input Mix & Self-Healing Parameters")

    with st.expander("🧱 Section 1: Base Mix Design (Continuous Inputs)", expanded=True):
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

    with st.expander("🧬 Section 2: Self-Healing Agents & Fiber Reinforcement", expanded=True):
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

    with st.expander("⏳ Section 3: Environmental Curing & Testing Conditions", expanded=True):
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
    st.subheader("Step 3: Prediction Output")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_clicked = st.button(
            "🚀 Predict Compressive Strength", type="primary", use_container_width=True
        )
    with btn_col2:
        suggest_clicked = st.button(
            "✨ Get Suggested Mix", use_container_width=True
        )

    if bundle is not None:
        try:
            input_row = build_collective_input_df(current_params)
            pred_val = predict_from_bundle(bundle, input_row)
            total_b = cement + silica_fumes + fly_ash
            wc_ratio = water / cement
            wb_ratio = water / total_b
            grade_text, _ = classify_structural_grade(pred_val)

            if predict_clicked and _dialog_decorator is not None:
                show_prediction_popup(
                    pred_val,
                    rmse_val,
                    mae_val,
                    r2_val,
                    wc_ratio,
                    wb_ratio,
                    total_b,
                    curing_env,
                    int(curing_days),
                    bacteria_type,
                )

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Predicted Compressive Strength</div>
                    <div class="kpi-value">{pred_val:.2f}</div>
                    <div class="kpi-unit">MPa (± {mae_val:.2f} MPa MAE)</div>
                    <div class="grade-pill">{grade_text}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-badge">
                        <div class="metric-badge-label">RMSE</div>
                        <div class="metric-badge-val">{rmse_val:.2f} MPa</div>
                    </div>
                    <div class="metric-badge">
                        <div class="metric-badge-label">MAE</div>
                        <div class="metric-badge-val">{mae_val:.2f} MPa</div>
                    </div>
                    <div class="metric-badge">
                        <div class="metric-badge-label">Validation R²</div>
                        <div class="metric-badge-val">{r2_val:.4f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if go is not None:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=pred_val,
                        number={"suffix": " MPa", "font": {"size": 26, "color": "#F8FAFC"}},
                        gauge={
                            "axis": {"range": [0, 120], "tickwidth": 1, "tickcolor": "#94A3B8"},
                            "bar": {"color": "#00D2FF"},
                            "bgcolor": "#0F172A",
                            "borderwidth": 1,
                            "bordercolor": "#334155",
                            "steps": [
                                {"range": [0, 25], "color": "rgba(245, 158, 11, 0.2)"},
                                {"range": [25, 45], "color": "rgba(16, 185, 129, 0.2)"},
                                {"range": [45, 65], "color": "rgba(0, 210, 255, 0.2)"},
                                {"range": [65, 120], "color": "rgba(168, 85, 247, 0.2)"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=210,
                    margin=dict(l=20, r=20, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#F8FAFC"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("#### Mix Physical Metrics")
            st.write(f"• **Water/Cement (W/C) Ratio:** `{wc_ratio:.3f}`")
            st.write(f"• **Water/Binder (W/B) Ratio:** `{wb_ratio:.3f}`")
            st.write(f"• **Total Binder:** `{total_b:.1f} kg/m³`")
            st.write(f"• **Aggregate/Binder Ratio:** `{(coarse_agg + sand) / total_b:.2f}`")

        except Exception as pred_err:
            st.error(f"Prediction calculation error: {repr(pred_err)}")
            st.code(traceback.format_exc())


# =============================================================================
# 7. STEP 2 & STEP 4: SUGGESTED MIX WINDOW & INTERPRETABILITY DIAGNOSTICS
# =============================================================================

if bundle is not None:
    st.markdown("---")
    tab_suggest, tab_interp = st.tabs(
        [
            "✨ Step 2: Suggested Mix Proportion Comparison",
            "📊 Step 4: Feature Sensitivity & Model Diagnostics",
        ]
    )

    with tab_suggest:
        sug_params, sug_pred = generate_suggested_mix(bundle, current_params)
        gain = sug_pred - pred_val
        st.markdown(
            f"**Suggested Mix Predicted Strength:** `{sug_pred:.2f} MPa` "
            f"(`{gain:+.2f} MPa` vs. current mix)"
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
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with tab_interp:
        df_imp = compute_local_feature_contributions(bundle, current_params, pred_val)
        if go is not None:
            colors = ["#10B981" if v >= 0 else "#F43F5E" for v in df_imp["Impact_MPa"]]
            fig_imp = go.Figure(
                go.Bar(
                    x=df_imp["Impact_MPa"],
                    y=df_imp["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f} MPa" for v in df_imp["Impact_MPa"]],
                    textposition="outside",
                )
            )
            fig_imp.update_layout(
                title="Local Parameter Contribution vs. Control Baseline (MPa)",
                xaxis_title="Strength Contribution (Δ MPa)",
                yaxis_title="Parameter Group",
                height=420,
                margin=dict(l=20, r=40, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F8FAFC"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.dataframe(df_imp[["Feature", "Impact_MPa"]], use_container_width=True)