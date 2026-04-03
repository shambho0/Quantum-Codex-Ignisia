"""
backend/explainability.py
--------------------------
SHAP-based explainability layer.
Generates top-5 plain-language reasons for a credit score.

reason_map and generate_reasons() extracted from Copy_of_mit_wpu_model.ipynb.
"""

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor

# ──────────────────────────────────────────────
# Human-readable reason map (from notebook)
# ──────────────────────────────────────────────

REASON_MAP = {
    "gst_consistency": lambda x: (
        "Excellent GST compliance improves creditworthiness" if x > 0.85 else
        "Moderate GST compliance indicates average reliability" if x > 0.65 else
        "Poor GST compliance reduces trustworthiness"
    ),
    "gst_delay": lambda x: (
        "Timely GST filings indicate strong discipline" if x <= 2 else
        "Occasional GST delays slightly affect reliability" if x <= 8 else
        "Frequent GST filing delays reduce reliability"
    ),
    "avg_upi_inflow": lambda x: (
        "Very strong UPI inflow indicates high business activity" if x > 150000 else
        "Stable UPI inflow supports steady operations" if x > 70000 else
        "Low UPI inflow suggests weak revenue generation"
    ),
    "cashflow_volatility": lambda x: (
        "Stable cash flow indicates low financial risk" if x < 0.3 else
        "Moderate cash flow variation is manageable" if x < 0.7 else
        "High cash flow volatility increases financial risk"
    ),
    "invoice_growth": lambda x: (
        "Strong sales growth boosts business outlook" if x > 0.15 else
        "Positive sales growth supports stability" if x > 0 else
        "Declining sales trend impacts business stability"
    ),
    "transaction_frequency": lambda x: (
        "High transaction activity indicates strong demand" if x > 70 else
        "Moderate transaction activity shows steady business" if x > 30 else
        "Low transaction activity indicates limited business operations"
    ),
    "inflow_outflow_ratio": lambda x: (
        "Healthy inflow-outflow balance indicates profitability" if x > 1.5 else
        "Balanced cash flow supports business stability" if x > 0.8 else
        "Poor cash flow balance raises financial concerns"
    ),
    "shipment_rate": lambda x: (
        "Strong shipment activity supports business expansion" if x > 30 else
        "Moderate shipment activity reflects stable operations" if x > 10 else
        "Low shipment activity suggests limited scale"
    ),
    "business_growth": lambda x: (
        "Strong business growth improves future outlook" if x > 0.15 else
        "Stable business growth supports consistency" if x > 0 else
        "Negative growth indicates business decline"
    ),
    "fraud_flag": lambda x: (
        "Potential fraud signals detected, significantly increasing risk"
        if x == 1 else
        "No fraud signals detected, improving trust profile"
    ),
}


def generate_reasons(features: dict, shap_values, feature_names: list) -> list[str]:
    """
    Generate top-5 plain-language reasons for the score.
    Negatives are listed first (most impactful risks up front).

    Extracted from notebook: generate_reasons()
    """
    shap_vals = shap_values.values[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:7]

    reasons = []
    for i in top_idx:
        feature = feature_names[i]
        value = features[feature]
        impact = shap_vals[i]
        if feature in REASON_MAP:
            text = REASON_MAP[feature](value)
            reasons.append((text, impact))

    # Negatives first (risk factors), then positives
    negatives = [r for r in reasons if r[1] < 0]
    positives = [r for r in reasons if r[1] > 0]
    ordered = negatives + positives

    return [r[0] for r in ordered[:5]]


def get_shap_explanation(model: CatBoostRegressor, features: dict) -> dict:
    """
    Returns:
        - top_reasons: list[str]   — top 5 plain-language reasons
        - shap_values: dict        — {feature_name: shap_value} for all features
        - base_value: float        — SHAP expected value
    """
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

    explainer = shap.Explainer(model)
    shap_vals = explainer(df)

    reasons = generate_reasons(features, shap_vals, feature_order)

    shap_dict = {
        feat: float(shap_vals.values[0][i])
        for i, feat in enumerate(feature_order)
    }

    return {
        "top_reasons": reasons,
        "shap_values": shap_dict,
        "base_value": float(shap_vals.base_values[0]),
    }
