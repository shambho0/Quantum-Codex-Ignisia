"""
backend/pipeline.py
-------------------
Mock real-time data ingestion pipeline.
Simulates GST filing velocity, UPI transaction cadence, and e-way bill volume trends.

Code extracted directly from Copy_of_mit_wpu_model.ipynb.
"""

import datetime
import numpy as np


# ──────────────────────────────────────────────
# Feature definitions (reference)
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    "gst_consistency",       # GST filing consistency ratio  [0, 1]
    "gst_delay",             # Avg days of GST filing delay  [0, 15]
    "invoice_growth",        # MoM invoice growth rate       [-0.2, 0.3]
    "avg_upi_inflow",        # Avg monthly UPI inflow (₹)   [10k, 200k]
    "inflow_outflow_ratio",  # UPI inflow / outflow ratio    [0.5, 2.0]
    "transaction_frequency", # Monthly UPI transaction count [5, 100]
    "cashflow_volatility",   # Cashflow volatility index     [0.1, 1.0]
    "shipment_rate",         # Monthly e-way bill count      [0, 50]
    "business_growth",       # MoM business growth signal    [-0.2, 0.3]
    "fraud_flag",            # Fraud indicator (0/1)
]


def _get_time_seed() -> int:
    """Returns an integer seed based on the current date (changes daily)."""
    now = datetime.datetime.now()
    return int(now.strftime("%Y%m%d"))


def get_base_features(gstin: str) -> dict:
    """
    Generate deterministic base features for a given GSTIN.
    Same GSTIN always returns the same base profile (business identity).

    Extracted from notebook: get_base_features()
    """
    seed = sum(ord(c) for c in gstin)
    rng = np.random.default_rng(seed)

    return {
        "gst_consistency":      float(np.clip(rng.normal(0.75, 0.2), 0, 1)),
        "gst_delay":            int(rng.integers(0, 15)),
        "invoice_growth":       float(rng.uniform(-0.2, 0.3)),
        "avg_upi_inflow":       int(rng.integers(10000, 200000)),
        "inflow_outflow_ratio": float(rng.uniform(0.5, 2.0)),
        "transaction_frequency":int(rng.integers(5, 100)),
        "cashflow_volatility":  float(rng.uniform(0.1, 1.0)),
        "shipment_rate":        int(rng.integers(0, 50)),
        "business_growth":      float(rng.uniform(-0.2, 0.3)),
        "fraud_flag":           int(rng.choice([0, 1], p=[0.9, 0.1])),
    }


def simulate_live_features(base: dict, gstin: str) -> dict:
    """
    Apply real-time micro-variations to the base features.
    Uses current date as part of the seed so scores update daily.

    Extracted from notebook: simulate_live_features()
    """
    seed = sum(ord(c) for c in gstin) + _get_time_seed()
    rng = np.random.default_rng(seed)

    live = base.copy()

    # UPI signal variations
    live["avg_upi_inflow"]       *= rng.uniform(0.95, 1.05)
    live["transaction_frequency"] += int(rng.integers(-3, 3))

    # Invoice & cashflow signal drift
    live["invoice_growth"]          += rng.normal(0, 0.02)
    live["inflow_outflow_ratio"]    += rng.normal(0, 0.05)
    live["cashflow_volatility"]     += rng.normal(0, 0.03)

    # Simulate occasional cash-flow shock (10% probability)
    if rng.random() < 0.1:
        live["avg_upi_inflow"]    *= rng.uniform(0.7, 0.9)
        live["cashflow_volatility"] += 0.2

    # Clamp to valid ranges
    live["cashflow_volatility"]  = float(np.clip(live["cashflow_volatility"],  0, 1))
    live["inflow_outflow_ratio"] = float(np.clip(live["inflow_outflow_ratio"], 0, 3))
    live["avg_upi_inflow"]       = float(live["avg_upi_inflow"])

    return live


def get_live_features(gstin: str) -> dict:
    """
    Full pipeline: base features → live simulation.
    Returns a flat dict ready to pass to the model.
    """
    base = get_base_features(gstin)
    live = simulate_live_features(base, gstin)
    return live
