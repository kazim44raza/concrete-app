```python
"""
Self-Healing Concrete Compressive Strength Prediction Dashboard
===============================================================
Model Bundle: Optuna_Collective21_Ensemble_v29starstar.joblib
Target Range: 9.06 MPa – 115.13 MPa | Validation R² = 0.9283 | RMSE = 4.136 MPa

Single-page, 4-step vertical engineering workflow:
  Step 1: Input Data (4 Visual Sections: Base Mix, Self-Healing Agents,
          Environmental & Testing, and "Get Suggested Mix Proportion" Action)
  Step 2: Suggested Mix Proportion Window (Side-by-side comparison & mix selector)
  Step 3: Prediction Output (Plotly Gauge Chart + Structural Grade Diagnostics)
  Step 4: Interpretability & Validation Plots (21-Feature SHAP Waterfall & Parity Plot)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore


# =============================================================================
# 1. CONSTANTS, STRICT BOUNDS & 30-COLUMN / 21-FEATURE SCHEMA DEFINITIONS
# =============================================================================

MODEL_BUNDLE_FILENAME = "Optuna_Collective21_Ensemble_v29starstar.joblib"
REGRESSION_PLOT_FILENAME = "Optuna_v29starstar_Independent_Test_Regression.png"

DATASET_MIN_MPA = 9.06
DATASET_MAX_MPA = 115.13
MODEL_R2 = 0.9283
MODEL_RMSE = 4.136

# Strict bounds for Section 1: Base Mix Design (Continuous Inputs)
BASE_MIX_SPECS: Dict[str, Dict[str, Any]] = {
    "Cement(kg/m3)": {
        "label": "Cement",
        "unit": "kg/m³",
        "min": 171.6,
        "max": 650.0,
        "default": 400.0,
        "step": 5.0,
        "help": "Ordinary Portland Cement content (171.6 – 650.0 kg/m³)",
    },
    "WATER(kg/m3)": {
        "label": "Mixing Water",
        "unit": "kg/m³",
        "min": 122.5,
        "max": 300.0,
        "default": 200.0,
        "step": 2.5,
        "help": "Net mixing water content (122.5 – 300.0 kg/m³)",
    },
    "CA(kg/m3)": {
        "label": "Coarse Aggregate (CA)",
        "unit": "kg/m³",
        "min": 0.0,
        "max": 1477.74,
        "default": 800.0,
        "step": 10.0,
        "help": "Crushed stone / gravel coarse aggregate (0.0 – 1477.74 kg/m³)",
    },
    "FA (kg/m3)SAND": {
        "label": "Fine Aggregate (FA / Sand)",
        "unit": "kg/m³",
        "min": 350.0,
        "max": 1500.0,
        "default": 700.0,
        "step": 10.0,
        "help": "Natural river or manufactured sand (350.0 – 1500.0 kg/m³)",
    },
    "SuperPlasticizer(kg/m3)": {
        "label": "Superplasticizer (HRWR)",
        "unit": "kg/m³",
        "min": 0.0,
        "max": 15.675,
        "default": 0.0,
        "step": 0.25,
        "help": "High-range water reducer dosage (0.0 – 15.675 kg/m³)",
    },
    "Silica Fumes(kg/m3)": {
        "label": "Silica Fume",
        "unit": "kg/m³",
        "min": 0.0,
        "max": 50.0,
        "default": 0.0,
        "step": 1.0,
        "help": "Micro-silica supplementary pozzolan (0.0 – 50.0 kg/m³)",
    },
    "Fly Ash(kg/m3)": {
        "label": "Fly Ash",
        "unit": "kg/m³",
        "min": 0.0,
        "max": 350.0,
        "default": 0.0,
        "step": 5.0,
        "help": "Pulverized fuel ash binder addition (0.0 – 350.0 kg/m³)",
    },
    "T(°C) OF CUBE": {
        "label": "Cube Temperature T(°C)",
        "unit": "°C",
        "min": 22.0,
        "max": 200.0,
        "default": 25.0,
        "step": 1.0,
        "help": "Thermal conditioning temperature of specimen (22.0 – 200.0 °C)",
    },
}

# Section 2: Self-Healing Agents mappings
FIBER_TYPE_MAP: Dict[str, Optional[str]] = {
    "None": None,
    "Steel Fiber": "STEEL FIBER(kg/m3)",
    "Glass Fiber": "GLASS FIBER(kg/m3)",
    "Basalt Fiber": "BASALT FIBER(kg/m3)",
    "PVA Fiber": "PVA FIBER(kg/m3)",
}

BACTERIA_TYPE_MAP: Dict[str, Optional[str]] = {
    "None": None,
    "Bacillus Subtilis": "BACILLUS SUBTILIS(kg/m3)",
    "Rhizobium Leguminosarum": "RHIZOBIUM LEGUMINOSARUM(kg/m3)",
    "Bacillus Sphaericus": "BACILLUS SPHAERICUS(kg/m3)",
    "Bacillus Megaterium": "BACILLUS MEGATERIUM(kg/m3)",
}

NUTRIENT_TYPE_MAP: Dict[str, Optional[str]] = {
    "None": None,
    "Calcium Lactate": "CALCIUM LACTATE(kg/m3)",
    "Calcium Lactate Pentahydrate": "CALCIUM LACTATE PENTAHYDRATE(kg/m3)",
    "Calcium Nitrate": "CALCIUM NITRATE(kg/m3)",
}

DELIVERY_SYSTEM_MAP: Dict[str, Optional[str]] = {
    "None": None,
    "Bacteria Only": None,  # Forces both microcapsule columns to 0.0
    "Sodium Silicate Microcapsule": "SODIUM SILICATE MICROCAPSULE(kg/m3)",
    "Cement Microcapsule": "CEMENT MICROCAPSULE(kg/m3)",
}

# Section 3: Environmental & Testing mappings
CURING_ENV_MAP: Dict[str, str] = {
    "Fresh Water Curing": "FRESH WATER CURING",
    "Sulfate Water Curing": "SULFATE WATER CURING",
    "Humidity Air Curing": "HUMIDITY AIR CURING",
    "Dry Air Curing": "DRY AIR CURING",
    "Sea Water Curing": "SEA WATER CURING",
}

# Exact 30 Raw Columns in canonical order
RAW_30_COLUMNS: List[str] = [
    # Section 1: Base Mix Design (8)
    "Cement(kg/m3)",
    "WATER(kg/m3)",
    "CA(kg/m3)",
    "FA (kg/m3)SAND",
    "SuperPlasticizer(kg/m3)",
    "Silica Fumes(kg/m3)",
    "Fly Ash(kg/m3)",
    "T(°C) OF CUBE",
    # Section 2: Self-Healing Agents (14)
    "STEEL FIBER(kg/m3)",
    "GLASS FIBER(kg/m3)",
    "BASALT FIBER(kg/m3)",
    "PVA FIBER(kg/m3)",
    "BACILLUS SUBTILIS(kg/m3)",
    "RHIZOBIUM LEGUMINOSARUM(kg/m3)",
    "BACILLUS SPHAERICUS(kg/m3)",
    "BACILLUS MEGATERIUM(kg/m3)",
    "Bacteria(cell/ml)",
    "CALCIUM LACTATE(kg/m3)",
    "CALCIUM LACTATE PENTAHYDRATE(kg/m3)",
    "CALCIUM NITRATE(kg/m3)",
    "SODIUM SILICATE MICROCAPSULE(kg/m3)",
    "CEMENT MICROCAPSULE(kg/m3)",
    # Section 3: Environmental & Testing (8)
    "FRESH WATER CURING",
    "SULFATE WATER CURING",
    "HUMIDITY AIR CURING",
    "DRY AIR CURING",
    "SEA WATER CURING",
    "Curing Days",
    "Healing Period Days",
    "APPLIED LOAD FOR HEALING(MPa)",
]

# Exact 21 Collective Features represented by the model architecture
COLLECTIVE_21_FEATURES: List[str] = [
    "Cement(kg/m3)",
    "WATER(kg/m3)",
    "CA(kg/m3)",
    "FA (kg/m3)SAND",
    "SuperPlasticizer(kg/m3)",
    "Silica Fumes(kg/m3)",
    "Fly Ash(kg/m3)",
    "T(°C) OF CUBE",
    "Fiber Type",
    "Fiber Dosage (kg/m3)",
    "Bacteria Type",
    "Bacteria Dosage (kg/m3)",
    "Bacteria(cell/ml)",
    "Nutrient Type",
    "Nutrient Dosage (kg/m3)",
    "Delivery System",
    "Delivery Dosage (kg/m3)",
    "Curing Environment",
    "Curing Days",
    "Healing Period Days",
    "APPLIED LOAD FOR HEALING(MPa)",
]


# =============================================================================
# 2. REQUIRED VERBATIM PREDICTION UTILITIES & SCHEMA MAPPING LOGIC
# =============================================================================

def normalized_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


# Ensure normalized_name is always registered on __main__ for joblib unpickling
sys.modules["__main__"].normalized_name = normalized_name


def map_ui_inputs_to_raw_30_df(ui_inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Maps the structured UI inputs from Sections 1, 2, and 3 into a single-row
    pandas DataFrame with the exact 30 raw columns specified by the schema.
    """
    row: Dict[str, float] = {col: 0.0 for col in RAW_30_COLUMNS}

    # Section 1: Base Mix Design (8 continuous inputs)
    for col, spec in BASE_MIX_SPECS.items():
        val = float(ui_inputs.get(col, spec["default"]))
        row[col] = float(np.clip(val, spec["min"], spec["max"]))

    # Section 2: Self-Healing Agents
    # Feature 1: Fibers
    fiber_type = str(ui_inputs.get("fiber_type", "None"))
    fiber_dosage = float(np.clip(float(ui_inputs.get("fiber_dosage", 0.0)), 0.0, 117.0))
    target_fiber_col = FIBER_TYPE_MAP.get(fiber_type)
    if target_fiber_col is not None:
        row[target_fiber_col] = fiber_dosage

    # Feature 2: Bacteria
    bacteria_type = str(ui_inputs.get("bacteria_type", "None"))
    bacteria_dosage = float(
        np.clip(float(ui_inputs.get("bacteria_dosage", 0.0)), 0.0, 26.7)
    )
    target_bacteria_col = BACTERIA_TYPE_MAP.get(bacteria_type)
    if target_bacteria_col is not None:
        row[target_bacteria_col] = bacteria_dosage

    bacteria_conc = float(
        np.clip(float(ui_inputs.get("bacteria_conc", 0.0)), 0.0, 2.8e9)
    )
    row["Bacteria(cell/ml)"] = bacteria_conc

    # Feature 3: Nutrients
    nutrient_type = str(ui_inputs.get("nutrient_type", "None"))
    nutrient_dosage = float(
        np.clip(float(ui_inputs.get("nutrient_dosage", 0.0)), 0.0, 44.323)
    )
    target_nutrient_col = NUTRIENT_TYPE_MAP.get(nutrient_type)
    if target_nutrient_col is not None:
        row[target_nutrient_col] = nutrient_dosage

    # Feature 4: Delivery System
    # Note: If "Bacteria Only" or "None" is selected, force delivery dosages to 0.
    delivery_system = str(ui_inputs.get("delivery_system", "None"))
    delivery_dosage = float(
        np.clip(float(ui_inputs.get("delivery_dosage", 0.0)), 0.0, 12.25)
    )
    if delivery_system in ("None", "Bacteria Only"):
        delivery_dosage = 0.0
    target_delivery_col = DELIVERY_SYSTEM_MAP.get(delivery_system)
    if target_delivery_col is not None:
        row[target_delivery_col] = delivery_dosage

    # Section 3: Environmental & Testing
    curing_env = str(ui_inputs.get("curing_env", "Fresh Water Curing"))
    target_curing_col = CURING_ENV_MAP.get(curing_env, "FRESH WATER CURING")
    for env_col in CURING_ENV_MAP.values():
        row[env_col] = 1.0 if env_col == target_curing_col else 0.0

    row["Curing Days"] = float(
        np.clip(float(ui_inputs.get("curing_days", 28.0)), 3.0, 180.0)
    )
    row["Healing Period Days"] = float(
        np.clip(float(ui_inputs.get("healing_days", 0.0)), 0.0, 124.0)
    )
    row["APPLIED LOAD FOR HEALING(MPa)"] = float(
        np.clip(float(ui_inputs.get("applied_load", 0.0)), 0.0, 60.3)
    )

    df = pd.DataFrame([row], columns=RAW_30_COLUMNS)
    # Preserve categorical selection metadata for Collective-21 reconstruction
    df.attrs["delivery_system"] = delivery_system
    df.attrs["fiber_type"] = fiber_type
    df.attrs["bacteria_type"] = bacteria_type
    df.attrs["nutrient_type"] = nutrient_type
    df.attrs["curing_env"] = curing_env
    return df


def derive_collective21_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives the 21 collective features from the 30-column raw DataFrame.
    Supports both categorical-string and numeric representations.
    """
    r = raw_df.iloc[0]

    # 1. Fiber Type & Dosage
    fiber_type = raw_df.attrs.get("fiber_type")
    fiber_dosage = 0.0
    for f_label, f_col in FIBER_TYPE_MAP.items():
        if f_col and f_col in r and float(r[f_col]) > 0:
            fiber_type = f_label
            fiber_dosage = float(r[f_col])
            break
    if not fiber_type or fiber_dosage == 0.0:
        fiber_type = fiber_type if fiber_type and fiber_dosage > 0 else "None"

    # 2. Bacteria Type & Dosage
    bacteria_type = raw_df.attrs.get("bacteria_type")
    bacteria_dosage = 0.0
    for b_label, b_col in BACTERIA_TYPE_MAP.items():
        if b_col and b_col in r and float(r[b_col]) > 0:
            bacteria_type = b_label
            bacteria_dosage = float(r[b_col])
            break
    if not bacteria_type or bacteria_dosage == 0.0:
        bacteria_type = bacteria_type if bacteria_type and bacteria_dosage > 0 else "None"

    # 3. Nutrient Type & Dosage
    nutrient_type = raw_df.attrs.get("nutrient_type")
    nutrient_dosage = 0.0
    for n_label, n_col in NUTRIENT_TYPE_MAP.items():
        if n_col and n_col in r and float(r[n_col]) > 0:
            nutrient_type = n_label
            nutrient_dosage = float(r[n_col])
            break
    if not nutrient_type or nutrient_dosage == 0.0:
        nutrient_type = nutrient_type if nutrient_type and nutrient_dosage > 0 else "None"

    # 4. Delivery System & Dosage
    delivery_system = raw_df.attrs.get("delivery_system", "None")
    delivery_dosage = 0.0
    for d_label, d_col in DELIVERY_SYSTEM_MAP.items():
        if d_col and d_col in r and float(r[d_col]) > 0:
            delivery_system = d_label
            delivery_dosage = float(r[d_col])
            break
    if delivery_system in ("None", "Bacteria Only"):
        delivery_dosage = 0.0

    # 5. Curing Environment
    curing_env = raw_df.attrs.get("curing_env", "Fresh Water Curing")
    for c_label, c_col in CURING_ENV_MAP.items():
        if c_col in r and float(r[c_col]) >= 0.5:
            curing_env = c_label
            break

    collective_dict = {
        "Cement(kg/m3)": float(r["Cement(kg/m3)"]),
        "WATER(kg/m3)": float(r["WATER(kg/m3)"]),
        "CA(kg/m3)": float(r["CA(kg/m3)"]),
        "FA (kg/m3)SAND": float(r["FA (kg/m3)SAND"]),
        "SuperPlasticizer(kg/m3)": float(r["SuperPlasticizer(kg/m3)"]),
        "Silica Fumes(kg/m3)": float(r["Silica Fumes(kg/m3)"]),
        "Fly Ash(kg/m3)": float(r["Fly Ash(kg/m3)"]),
        "T(°C) OF CUBE": float(r["T(°C) OF CUBE"]),
        "Fiber Type": fiber_type,
        "Fiber Dosage (kg/m3)": fiber_dosage,
        "Bacteria Type": bacteria_type,
        "Bacteria Dosage (kg/m3)": bacteria_dosage,
        "Bacteria(cell/ml)": float(r["Bacteria(cell/ml)"]),
        "Nutrient Type": nutrient_type,
        "Nutrient Dosage (kg/m3)": nutrient_dosage,
        "Delivery System": delivery_system,
        "Delivery Dosage (kg/m3)": delivery_dosage,
        "Curing Environment": curing_env,
        "Curing Days": float(r["Curing Days"]),
        "Healing Period Days": float(r["Healing Period Days"]),
        "APPLIED LOAD FOR HEALING(MPa)": float(r["APPLIED LOAD FOR HEALING(MPa)"]),
    }
    return pd.DataFrame([collective_dict], columns=COLLECTIVE_21_FEATURES)


def _align_schema_for_bundle(bundle: Dict[str, Any], new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Reconciles the 30-column raw DataFrame with whatever internal schema
    (`feature_names_in_`, `collective_columns`, `feature_names`, or custom
    mapping) is expected by `bundle["models"][bundle["selected_strategy"]]`.
    """
    if not isinstance(bundle, dict) or "models" not in bundle:
        return new_data

    # If the bundle provides an explicit transformation callable, apply it
    for fn_key in ("transform_fn", "to_collective", "schema_mapper", "preprocess_raw"):
        if callable(bundle.get(fn_key)):
            try:
                return bundle[fn_key](new_data)
            except Exception:
                pass

    model = bundle["models"][bundle["selected_strategy"]]
    collective_21 = derive_collective21_frame(new_data)

    # Build a normalized lookup pool combining raw 30, collective 21, and common aliases
    pool: Dict[str, Any] = {}
    for col in new_data.columns:
        pool[normalized_name(col)] = new_data[col].iloc[0]
    for col in collective_21.columns:
        pool[normalized_name(col)] = collective_21[col].iloc[0]

    # Common normalized aliases that training pipelines may use for the 21 features
    c21_row = collective_21.iloc[0]
    alias_groups = {
        c21_row["Fiber Type"]: ["fibertype", "fiber", "fibercategory", "typeoffiber"],
        c21_row["Fiber Dosage (kg/m3)"]: [
            "fiberdosagekgm3",
            "fiberdosage",
            "fiberkgm3",
            "fibercontentkgm3",
            "totalfiberkgm3",
        ],
        c21_row["Bacteria Type"]: [
            "bacteriatype",
            "bacteriaspecies",
            "typeofbacteria",
            "bacteriacategory",
        ],
        c21_row["Bacteria Dosage (kg/m3)"]: [
            "bacteriadosagekgm3",
            "bacteriadosage",
            "bacteriakgm3",
            "bacteriacontentkgm3",
        ],
        c21_row["Bacteria(cell/ml)"]: [
            "bacteriacellml",
            "bacteriaconcentrationcellml",
            "bacteriaconcentration",
        ],
        c21_row["Nutrient Type"]: ["nutrienttype", "nutrient", "typeofnutrient"],
        c21_row["Nutrient Dosage (kg/m3)"]: [
            "nutrientdosagekgm3",
            "nutrientdosage",
            "nutrientkgm3",
            "nutrientcontentkgm3",
        ],
        c21_row["Delivery System"]: [
            "deliverysystem",
            "deliverytype",
            "deliverymethod",
            "microcapsuletype",
            "immobilizationmethod",
        ],
        c21_row["Delivery Dosage (kg/m3)"]: [
            "deliverydosagekgm3",
            "deliverydosage",
            "deliverykgm3",
            "microcapsuledosagekgm3",
        ],
        c21_row["Curing Environment"]: [
            "curingenvironment",
            "curingcondition",
            "curingtype",
            "curingregime",
            "curingmedium",
        ],
        c21_row["Curing Days"]: ["curingdays", "curingage", "curingtime", "curingperioddays"],
        c21_row["Healing Period Days"]: [
            "healingperioddays",
            "healingdays",
            "healingtime",
            "healingperiod",
        ],
        c21_row["APPLIED LOAD FOR HEALING(MPa)"]: [
            "appliedloadforhealingmpa",
            "appliedloadforhealing",
            "appliedloadmpa",
            "preloadmpa",
        ],
    }
    for val, aliases in alias_groups.items():
        for alias in aliases:
            pool.setdefault(alias, val)

    # Determine expected column list from model or bundle metadata
    expected_cols: Optional[List[str]] = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
        expected_cols = list(model.steps[0][1].feature_names_in_)
    else:
        for key in (
            "collective_columns",
            "feature_names",
            "feature_columns",
            "input_columns",
            "columns",
        ):
            if key in bundle and isinstance(bundle[key], (list, tuple, pd.Index)):
                expected_cols = list(bundle[key])
                break

    if expected_cols:
        aligned_row = {}
        for col in expected_cols:
            norm_key = normalized_name(col)
            if col in new_data.columns:
                aligned_row[col] = new_data[col].iloc[0]
            elif col in collective_21.columns:
                aligned_row[col] = collective_21[col].iloc[0]
            elif norm_key in pool:
                aligned_row[col] = pool[norm_key]
            else:
                aligned_row[col] = 0.0
        return pd.DataFrame([aligned_row], columns=expected_cols)

    return new_data


def predict_with_bundle(bundle, new_data):
    # The bundle handles internal transformations. 
    import sys
    sys.modules['__main__'].normalized_name = normalized_name
    
    # Internal logic of bundle:
    collective_frame = _align_schema_for_bundle(bundle, new_data)
    
    # We must replicate the schema mapping logic expected by the model
    try:
        prediction = bundle["models"][bundle["selected_strategy"]].predict(collective_frame)
    except Exception:
        # Fallback if the model directly expects the derived 21-feature frame
        collective_frame = derive_collective21_frame(new_data)
        prediction = bundle["models"][bundle["selected_strategy"]].predict(collective_frame)
    return prediction


# =============================================================================
# 3. F
