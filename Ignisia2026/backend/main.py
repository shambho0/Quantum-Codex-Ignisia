"""
backend/main.py
---------------
FastAPI REST API for Real-Time MSME Credit Scoring.

Endpoints:
  GET  /health
  POST /score          { "gstin": "..." }
  GET  /score/{gstin}
  GET  /fraud-check/{gstin}
  GET  /score-history/{gstin}

Run with:
  uvicorn backend.main:app --reload --port 8000
"""

import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline        import get_live_features
from backend.model           import predict, load_model
from backend.explainability  import get_shap_explanation
from backend.fraud_detection import detect_circular_transactions
from backend.loan_recommender import recommend_loan
from backend.gstin           import validate_gstin

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="MSME Credit Scoring API",
    description=(
        "Real-time MSME credit scoring via alternative business signals "
        "(GST, UPI, e-Way Bill). Built for the Ignisia AI Hackathon."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load model at startup
@app.on_event("startup")
async def on_startup():
    load_model()


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class ScoreRequest(BaseModel):
    gstin: str = Field(..., example="27ABCDE1234F1Z5", description="15-character GSTIN")


class LoanRecommendation(BaseModel):
    amount: int
    tenure: str
    breakdown: dict


class FraudInfo(BaseModel):
    fraud_flag: int
    is_circular: bool
    cycle_count: int
    risk_score: float
    affected_nodes: list[str]


class ScoreResponse(BaseModel):
    gstin: str
    credit_score: int
    risk_band: str
    top_reasons: list[str]
    loan: LoanRecommendation
    fraud: FraudInfo
    shap_values: dict
    base_shap_value: float
    features: dict
    freshness_timestamp: str


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _risk_band(score: int) -> str:
    if score >= 750:
        return "Low"
    elif score >= 600:
        return "Medium"
    else:
        return "High"


def _raise_if_invalid_gstin(gstin: str) -> None:
    ok, err = validate_gstin(gstin)
    if not ok:
        raise HTTPException(status_code=422, detail=err)


def _score_gstin(gstin: str) -> ScoreResponse:
    """Core scoring logic shared by POST and GET endpoints."""
    # 1. Ingest live features
    features = get_live_features(gstin)

    # 2. ML prediction
    score = predict(features)
    band  = _risk_band(score)

    # 3. SHAP explainability
    model = load_model()
    explanation = get_shap_explanation(model, features)

    # 4. Fraud detection (graph layer)
    fraud_result = detect_circular_transactions(gstin)
    # Merge ML fraud_flag with graph fraud risk
    fraud_info = FraudInfo(
        fraud_flag    = int(features["fraud_flag"]),
        is_circular   = fraud_result["is_circular"],
        cycle_count   = fraud_result["cycle_count"],
        risk_score    = fraud_result["risk_score"],
        affected_nodes= fraud_result["affected_nodes"],
    ) 

    # 5. Loan recommendation
    loan_result = recommend_loan(score, features)
    loan_info   = LoanRecommendation(**loan_result)

    return ScoreResponse(
        gstin              = gstin,
        credit_score       = score,
        risk_band          = band,
        top_reasons        = explanation["top_reasons"],
        loan               = loan_info,
        fraud              = fraud_info,
        shap_values        = explanation["shap_values"],
        base_shap_value    = explanation["base_value"],
        features           = features,
        freshness_timestamp= datetime.datetime.utcnow().isoformat() + "Z",
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/score", response_model=ScoreResponse)
async def score_post(request: ScoreRequest):
    """Score a business by GSTIN (POST)."""
    gstin = request.gstin.strip().upper()
    _raise_if_invalid_gstin(gstin)
    return _score_gstin(gstin)


@app.get("/score/{gstin}", response_model=ScoreResponse)
async def score_get(gstin: str):
    """Score a business by GSTIN (GET convenience endpoint)."""
    gstin = gstin.strip().upper()
    _raise_if_invalid_gstin(gstin)
    return _score_gstin(gstin)


@app.get("/fraud-check/{gstin}")
async def fraud_check(gstin: str):
    """Standalone fraud graph analysis for a GSTIN."""
    gstin = gstin.strip().upper()
    _raise_if_invalid_gstin(gstin)
    result = detect_circular_transactions(gstin)
    return {
        "gstin": gstin,
        **result,
        "checked_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/score-history/{gstin}")
async def score_history(gstin: str, n: int = 7):
    """
    Simulate score history over the last N days.
    Uses different daily seeds to produce realistic variation.
    (In production, this would query a database.)
    """
    import datetime as dt
    import numpy as np
    from backend.pipeline import get_base_features, simulate_live_features
    from backend.model    import predict

    gstin = gstin.strip().upper()
    _raise_if_invalid_gstin(gstin)
    base  = get_base_features(gstin)
    history = []

    for days_ago in range(n - 1, -1, -1):
        target_date = dt.date.today() - dt.timedelta(days=days_ago)
        # Temporarily shift the daily seed
        import backend.pipeline as pipe_mod
        original_fn = pipe_mod._get_time_seed
        pipe_mod._get_time_seed = lambda d=target_date: int(d.strftime("%Y%m%d"))

        live  = simulate_live_features(base, gstin)
        score = predict(live)

        pipe_mod._get_time_seed = original_fn

        history.append({
            "date":  target_date.isoformat(),
            "score": score,
            "band":  _risk_band(score),
        })

    return {"gstin": gstin, "history": history}
