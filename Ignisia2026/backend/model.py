"""
backend/model.py
----------------
CatBoost model loader. Loads from disk if available, otherwise triggers training.
"""

import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "credit_model.cbm")

_model: CatBoostRegressor | None = None


def _ensure_trained() -> None:
    """Auto-train if model file doesn't exist yet."""
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("⚠️  Model not found. Running train_model.py ...")
        # Dynamically import and run the training script
        root = os.path.join(os.path.dirname(__file__), "..")
        sys.path.insert(0, root)
        from train_model import train_and_save
        train_and_save(model_dir=model_dir)


def load_model() -> CatBoostRegressor:
    """Return a singleton CatBoost model instance."""
    global _model
    if _model is None:
        _ensure_trained()
        _model = CatBoostRegressor()
        _model.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    return _model


def predict(features: dict) -> int:
    """
    Run model inference.

    Args:
        features: dict with keys matching FEATURE_NAMES from pipeline.py

    Returns:
        credit_score: int clamped to [300, 900]
    """
    model = load_model()
    feature_order = [
        "gst_consistency",
        "gst_delay",
        "invoice_growth",
        "avg_upi_inflow",
        "inflow_outflow_ratio",
        "transaction_frequency",
        "cashflow_volatility",
        "shipment_rate",
        "business_growth",
        "fraud_flag",
    ]
    df = pd.DataFrame([{k: features[k] for k in feature_order}])
    raw_score = model.predict(df)[0]
    return int(np.clip(raw_score, 300, 900))
