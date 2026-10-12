import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================================================================
# 1. CROSS-VERSION SCIKIT-LEARN COMPATIBILITY BRIDGES
# ================================================================

# A. Fix for ColumnTransformer _RemainderColsList across versions
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# B. Fix for SimpleImputer _fill_dtype across versions
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

# C. Fix for OneHotEncoder _infrequent_enabled across versions
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

# D. Fix for CatBoost wrapper if loading ensemble bundle
from sklearn.base import BaseEstimator, RegressorMixin
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

class CatBoostSklearnWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, params=None):
        self.params = params
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.model_.predict(X)

sys.modules["__main__"].CatBoostSklearnWrapper = CatBoostSklearnWrapper

# ================================================================
# 2. STREAMLIT PAGE CONFIGURATION & STYLING
# ================================================================
st.set_page_config(
    page_title="Bio-Concrete Compressive Strength AI",
    page_icon="🏗️",
    layout="wide",
)

st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 2px solid #00D2FF;
        border-radius: 14px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 210, 255, 0.15);
        margin-top: 15px;
    }
    .kpi-title { font-size: 15px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 50px; color: #00D2FF; font-weight: 800; margin: 8px 0; }
    .kpi-unit { font-size: 20px; color: #E2E8F0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Bio-Concrete Compressive Strength Predictor")
st.caption("Optuna-Tuned Machine Learning Engine (v29** Architecture)")

# ================================================================
# 3. LOAD MODEL BUNDLE
# ================================================================
@st.cache_resource
def load_bundle():
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
        st.error("No model .joblib file found in the repository or models/ folder.")
        return None

    try:
        return joblib.load(target_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {repr(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

bundle = load_bundle()

# ================================================================
# 4. EXACT v29** FEATURE RATIOS & TRANSFORMATIONS
# ================================================================
def safe_divide(n, d):
    d = pd.to_numeric(d, errors="coerce")
    return n / d.where(np.abs(d) > 1e-12, np.nan)

def add_domain_features(frame):
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
        supp = (silica.fillna(0.0) if silica is not None else 0.0) + (fly_ash.fillna(0.0) if fly_ash is not None else 0.0)
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

def predict_from_bundle(bundle, df_raw_inputs):
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
        preds = np.column_stack([
            bundle["models"][name].predict(df_model) for name in bundle["model_order"]
        ])
        weights = np.array([bundle["weights"][name] for name in bundle["model_order"]])
        return float(preds @ weights)
    elif "pipeline" in bundle:
        return float(bundle["pipeline"].predict(df_model)[0])
    else:
        strat = bundle.get("selected_strategy", bundle["model_order"][0])
        return float(bundle["models"][strat].predict(df_model)[0])

# ================================================================
# 5. USER CONTROLS WITH DATASET BOUNDS
# ================================================================
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("1. Standard Mix Ingredients")
    c1, c2, c3 = st.columns(3)
    with c1:
        cement = st.number_input("Cement (kg/m³)", min_value=171.60, max_value=650.00, value=400.00, step=5.0)
        water = st.number_input("Water (kg/m³)", min_value=122.50, max_value=300.00, value=160.00, step=2.5)
        sand = st.number_input("Fine Aggregate (kg/m³)", min_value=350.00, max_value=1500.00, value=1000.00, step=10.0)
    with c2:
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.00, max_value=1477.74, value=600.00, step=10.0)
        silica_fumes = st.number_input("Silica Fumes (kg/m³)", min_value=0.00, max_value=50.00, value=0.00, step=1.0)
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.00, max_value=350.00, value=0.00, step=5.0)
    with c3:
        sp = st.number_input("Superplasticizer (kg/m³)", min_value=0.00, max_value=15.675, value=2.00, step=0.1)
        temp_cube = st.number_input("Cube Temperature (°C)", min_value=22.0, max_value=200.0, value=30.0, step=1.0)

    st.subheader("2. Fiber Reinforcement")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        fiber_type = st.selectbox("Fiber Type", ["None", "Steel Fiber", "Glass Fiber", "Basalt Fiber", "PVA Fiber"])
    with f_col2:
        max_f = {"None": 0.0, "Steel Fiber": 117.0, "Glass Fiber": 6.39, "Basalt Fiber": 4.05, "PVA Fiber": 16.04}[fiber_type]
        fiber_dosage = st.number_input("Fiber Dosage (kg/m³)", min_value=0.0, max_value=max(0.01, max_f), value=0.0 if fiber_type == "None" else min(2.0, max_f), disabled=(fiber_type == "None"))

    st.subheader("3. Bio-Healing Agent & Microcapsule Delivery")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bacteria_type = st.selectbox("Bacteria Type", ["None", "Bacillus Subtilis", "Rhizobium Leguminosarum", "Bacillus Sphaericus", "Bacillus Megaterium"])
        max_b = {"None": 0.0, "Bacillus Subtilis": 26.7, "Rhizobium Leguminosarum": 17.0, "Bacillus Sphaericus": 22.5, "Bacillus Megaterium": 22.5}[bacteria_type]
        bacteria_dosage = st.number_input("Bacteria Dosage (kg/m³)", min_value=0.0, max_value=max(0.01, max_b), value=0.0 if bacteria_type == "None" else min(4.0, max_b), disabled=(bacteria_type == "None"))
    with b_col2:
        bacteria_cells = st.number_input("Concentration (cell/ml)", min_value=0.0, max_value=2.8e9, value=0.0 if bacteria_type == "None" else 1.0e7, step=1.0e6, format="%.2e", disabled=(bacteria_type == "None"))
    with b_col3:
        delivery_type = st.selectbox("Delivery Carrier", ["None", "Bacteria Only", "Sodium Silicate Microcapsule", "Cement Microcapsule"])
        max_del = 12.25 if delivery_type in ["Sodium Silicate Microcapsule", "Cement Microcapsule"] else 0.0
        delivery_dosage = st.number_input("Carrier Dosage (kg/m³)", min_value=0.0, max_value=max(0.01, max_del), value=0.0 if max_del == 0.0 else 5.0, disabled=(max_del == 0.0))

    st.subheader("4. Nutrients & Curing Environment")
    n_col1, n_col2 = st.columns(2)
    with n_col1:
        nutrient_type = st.selectbox("Nutrient Substrate", ["None", "Calcium Lactate", "Calcium Lactate Pentahydrate", "Calcium Nitrate"])
        max_nutr = {"None": 0.0, "Calcium Lactate": 44.323, "Calcium Lactate Pentahydrate": 15.4, "Calcium Nitrate": 32.0}[nutrient_type]
        nutrient_dosage = st.number_input("Nutrient Dosage (kg/m³)", min_value=0.0, max_value=max(0.01, max_nutr), value=0.0 if nutrient_type == "None" else min(2.0, max_nutr), disabled=(nutrient_type == "None"))
    with n_col2:
        curing_env = st.selectbox("Curing Exposure", ["Fresh Water Curing", "Sulfate Water Curing", "Humidity Air Curing", "Dry Air Curing", "Sea Water Curing"])
        curing_days = st.number_input("Curing Time (Days)", min_value=3, max_value=180, value=28, step=1)

    c_extra1, c_extra2 = st.columns(2)
    with c_extra1:
        healing_days = st.number_input("Healing Period (Days)", min_value=0, max_value=124, value=0, step=1)
    with c_extra2:
        applied_load = st.number_input("Pre-Damage Load (MPa)", min_value=0.0, max_value=60.3, value=0.0, step=1.0)

# ================================================================
# 6. REAL-TIME PREDICTION DISPLAY
# ================================================================
with col_right:
    st.subheader("Predicted Strength Output")
    update_btn = st.button("Update Prediction", type="primary", use_container_width=True)

    if bundle is not None:
        input_row = pd.DataFrame([{
            "Bacteria(cell/ml)": float(bacteria_cells),
            "T(°C) OF CUBE": float(temp_cube),
            "Cement(kg/m3)": float(cement),
            "CA(kg/m3)": float(coarse_agg),
            "FA (kg/m3)SAND": float(sand),
            "WATER(kg/m3)": float(water),
            "SuperPlasticizer(kg/m3)": float(sp),
            "Silica Fumes(kg/m3)": float(silica_fumes),
            "Fly Ash(kg/m3)": float(fly_ash),
            "Curing Days": float(curing_days),
            "Healing Period Days": float(healing_days),
            "APPLIED LOAD FOR HEALING(MPa)": float(applied_load),
            "Fiber Dosage (kg/m³)": float(fiber_dosage),
            "Bacteria Dosage (kg/m³)": float(bacteria_dosage),
            "Healing Delivery Dosage (kg/m³)": float(delivery_dosage),
            "Nutrient Dosage (kg/m³)": float(nutrient_dosage),
            "Fiber Type": str(fiber_type),
            "Bacteria Type": str(bacteria_type),
            "Healing Delivery Type": str(delivery_type),
            "Nutrient Type": str(nutrient_type),
            "Curing Environment": str(curing_env),
        }])

        try:
            pred_val = predict_from_bundle(bundle, input_row)

            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Predicted Compressive Strength</div>
                <div class="kpi-value">{pred_val:.2f}</div>
                <div class="kpi-unit">MPa</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Mix Physical Metrics")
            total_b = cement + silica_fumes + fly_ash
            st.write(f"• **Water/Binder Ratio:** `{water / total_b:.3f}`")
            st.write(f"• **Total Binder:** `{total_b:.1f} kg/m³`")
            st.write(f"• **Aggregate/Binder Ratio:** `{(coarse_agg + sand) / total_b:.2f}`")
        except Exception as pred_err:
            st.error(f"Prediction calculation error: {repr(pred_err)}")
            import traceback
            st.code(traceback.format_exc())