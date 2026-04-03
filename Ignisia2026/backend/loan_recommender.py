"""
backend/loan_recommender.py
----------------------------
Loan amount and tenure recommendation based on credit score and business features.
Extracted from Copy_of_mit_wpu_model.ipynb: recommend_loan()
"""


def recommend_loan(score: int, features: dict) -> dict:
    """
    Calculate recommended loan amount (₹) and tenure based on:
      - Credit score band
      - Average UPI inflow (revenue proxy)
      - Cash flow volatility (risk)
      - Inflow-outflow ratio (profitability)
      - Fraud flag (hard penalty)

    Extracted from notebook: recommend_loan()

    Returns:
        amount: int   — Recommended loan amount in ₹ (50,000 – 20,00,000)
        tenure: str   — Recommended repayment tenure
        breakdown: dict — Multiplier details for transparency
    """
    inflow     = features["avg_upi_inflow"]
    volatility = features["cashflow_volatility"]
    ratio      = features["inflow_outflow_ratio"]
    fraud      = features["fraud_flag"]

    base_amount = inflow * 2.5

    # Score-based multiplier + tenure
    if score >= 750:
        score_mult = 1.5
        tenure     = "36 months"
    elif score >= 650:
        score_mult = 1.2
        tenure     = "24 months"
    elif score >= 550:
        score_mult = 0.8
        tenure     = "18 months"
    else:
        score_mult = 0.5
        tenure     = "12 months"

    # Stability multiplier
    if volatility > 0.7:
        stability_mult = 0.6
    elif volatility > 0.4:
        stability_mult = 0.8
    else:
        stability_mult = 1.0

    # Profitability multiplier
    if ratio < 0.8:
        ratio_mult = 0.7
    elif ratio > 1.5:
        ratio_mult = 1.1
    else:
        ratio_mult = 1.0

    # Fraud hard penalty
    fraud_mult = 0.4 if fraud == 1 else 1.0

    amount = base_amount * score_mult * stability_mult * ratio_mult * fraud_mult
    amount = max(50_000, min(int(amount), 20_00_000))

    return {
        "amount": amount,
        "tenure": tenure,
        "breakdown": {
            "base_amount":     int(base_amount),
            "score_mult":      score_mult,
            "stability_mult":  stability_mult,
            "ratio_mult":      ratio_mult,
            "fraud_mult":      fraud_mult,
        },
    }
