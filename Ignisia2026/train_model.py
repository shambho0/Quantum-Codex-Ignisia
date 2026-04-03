"""
train_model.py
--------------
One-time script to generate synthetic MSME data and train the CatBoost model.
Code extracted directly from Copy_of_mit_wpu_model.ipynb.

Run once before starting the API:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ──────────────────────────────────────────────
# 1. Generate Synthetic Dataset (from notebook)
# ──────────────────────────────────────────────

def generate_dataset(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []

    for _ in range(n):
        g1 = np.clip(np.random.normal(0.75, 0.2), 0, 1)
        g2 = np.random.randint(0, 15)
        g3 = np.random.uniform(-0.2, 0.3)

        u1 = np.random.randint(10000, 200000)
        u2 = np.random.uniform(0.5, 2.0)
        u3 = np.random.randint(5, 100)
        u4 = np.random.uniform(0.1, 1.0)

        e1 = np.random.randint(0, 50)
        e2 = np.random.uniform(-0.2, 0.3)

        # Smarter fraud logic (not random only)
        if u4 > 0.8 and g2 > 10:
            f = 1
        else:
            f = np.random.choice([0, 1], p=[0.92, 0.08])

        # Small temporal variations
        g1 = np.clip(g1 + np.random.normal(0, 0.02), 0, 1)
        g2 = int(np.clip(g2 + np.random.choice([-1, 0, 1]), 0, 15))
        g3 += np.random.normal(0, 0.03)

        u1 = int(u1 * np.random.uniform(0.9, 1.1))
        u2 += np.random.normal(0, 0.05)
        u3 = max(1, u3 + np.random.randint(-5, 5))
        u4 = np.clip(u4 + np.random.normal(0, 0.05), 0, 1)

        e1 = max(0, e1 + np.random.randint(-3, 3))
        e2 += np.random.normal(0, 0.03)

        # ── Score calculation ──
        s = 500

        # Base contributions
        s += g1 * 120
        s -= g2 * 5
        s += g3 * 90

        s += u1 / 3000
        s += u2 * 35
        s += u3 * 1.0
        s -= u4 * 90

        s += e1 * 1.2
        s += e2 * 70

        # Hidden interaction patterns (model must learn)
        if g1 > 0.8 and u2 > 1.5 and e1 > 20:
            s += 60

        if u4 > 0.7 and u2 < 0.9:
            s -= 70

        if g3 > 0.15 and u3 > 60:
            s += 50

        # Fraud is probabilistic (NOT a fixed penalty)
        if f == 1:
            if np.random.rand() < 0.7:
                s -= np.random.randint(120, 220)
            else:
                s -= np.random.randint(20, 80)

        # Hidden external risk (unknown factor)
        hidden_risk = np.random.uniform(0, 1)
        if hidden_risk > 0.85:
            s -= np.random.randint(50, 150)

        # Major randomness → forces learning
        if np.random.rand() < 0.25:
            s += np.random.randint(-100, 100)

        # Continuous noise
        s += np.random.normal(0, 50)
        s = int(np.clip(s, 300, 900))

        # Risk band
        if s >= 750:
            r = "Low"
        elif s >= 600:
            r = "Medium"
        else:
            r = "High"

        records.append([g1, g2, g3, u1, u2, u3, u4, e1, e2, f, s, r])

    cols = [
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
        "credit_score",
        "risk_band",
    ]
    return pd.DataFrame(records, columns=cols)


# ──────────────────────────────────────────────
# 2. Train CatBoost Model (from notebook)
# ──────────────────────────────────────────────

def train_and_save(model_dir: str = "models") -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "credit_model.cbm")

    print("📊 Generating synthetic dataset (100,000 records)...")
    df = generate_dataset()

    # Drop target columns
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df.drop(["credit_score", "risk_band"], axis=1)
    y = df["credit_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training CatBoost model (iterations=1000, lr=0.05, depth=6)...")
    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n✅ Training complete — MSE: {mse:.2f} | R²: {r2:.4f}")

    # Save model
    model.save_model(model_path)
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_save()
