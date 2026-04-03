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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image as PILImage

from backend.pipeline        import get_live_features
from backend.model           import predict, load_model
from backend.explainability  import get_shap_explanation
from backend.fraud_detection import detect_circular_transactions
from backend.loan_recommender import recommend_loan
from backend.gstin           import validate_gstin
from backend.invoice_ocr     import extract_invoice_data

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


def _score_gstin(gstin: str, invoice_overrides: dict = None) -> ScoreResponse:
    """Core scoring logic shared by all endpoints."""
    # 1. Ingest live features
    features = get_live_features(gstin)

    # 1b. Override with invoice-extracted features if available
    if invoice_overrides and invoice_overrides.get("features"):
        for key, val in invoice_overrides["features"].items():
            if key in features:
                features[key] = val

    # 2. Fraud detection (graph layer) — run BEFORE prediction so
    #    the fraud_flag feature is consistent with the graph risk_score.
    fraud_result = detect_circular_transactions(gstin)
    graph_risk   = fraud_result["risk_score"]

    # Derive fraud_flag from graph risk so the two always agree:
    #   risk_score > 0.3  →  fraud_flag = 1 (suspicious)
    #   risk_score ≤ 0.3  →  fraud_flag = 0 (clean)
    features["fraud_flag"] = int(graph_risk > 0.3)

    # 3. ML prediction (now uses the aligned fraud_flag)
    score = predict(features)
    band  = _risk_band(score)

    # 4. SHAP explainability
    model = load_model()
    explanation = get_shap_explanation(model, features)

    # 5. Build unified fraud info
    fraud_info = FraudInfo(
        fraud_flag    = int(features["fraud_flag"]),
        is_circular   = fraud_result["is_circular"],
        cycle_count   = fraud_result["cycle_count"],
        risk_score    = graph_risk,
        affected_nodes= fraud_result["affected_nodes"],
    )

    # 6. Loan recommendation
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


@app.post("/score-invoice")
async def score_invoice(
    files: list[UploadFile] = File(...),
    gstin: str = Form(default=""),
):
    """Score from one or more uploaded GST invoice images (same company)."""
    all_ocr = []
    detected_gstin = None

    for f in files[:10]:  # cap at 10
        image = PILImage.open(f.file)
        ocr = extract_invoice_data(image)
        all_ocr.append(ocr)
        if not detected_gstin and ocr.get("gstin"):
            detected_gstin = ocr["gstin"]

    effective = gstin.strip().upper() or detected_gstin or ""
    if not effective:
        raise HTTPException(422, "GSTIN not found in invoices. Please enter it manually.")
    _raise_if_invalid_gstin(effective)

    # Aggregate across all invoices
    total_amount = sum(o.get("total_amount", 0) for o in all_ocr)
    tax_amount   = sum(o.get("tax_amount", 0) for o in all_ocr)
    line_items   = sum(o.get("line_items", 0) for o in all_ocr)
    n_invoices   = len(all_ocr)

    # Build aggregated feature overrides
    agg_features = {}
    if total_amount > 0:
        agg_features["avg_upi_inflow"] = total_amount / n_invoices
        agg_features["transaction_frequency"] = max(line_items * 5, 15)
        agg_features["inflow_outflow_ratio"] = min(2.0, 0.8 + (total_amount / n_invoices / 200000))
    if tax_amount > 0 and total_amount > 0:
        agg_features["gst_consistency"] = min(1.0, 0.7 + (tax_amount / total_amount))
        agg_features["gst_delay"] = 2
    elif total_amount > 0:
        agg_features["gst_consistency"] = 0.4
        agg_features["gst_delay"] = 10
    avg_amount = total_amount / max(n_invoices, 1)
    if avg_amount > 50000:
        agg_features["invoice_growth"] = 0.15
        agg_features["business_growth"] = 0.10
    elif avg_amount > 10000:
        agg_features["invoice_growth"] = 0.05
        agg_features["business_growth"] = 0.03
    else:
        agg_features["invoice_growth"] = -0.05
        agg_features["business_growth"] = -0.05
    agg_features["cashflow_volatility"] = 0.3 if n_invoices >= 3 else 0.5
    agg_features["shipment_rate"] = min(50, n_invoices * 5)

    aggregated = {
        "total_amount": total_amount,
        "tax_amount": tax_amount,
        "line_items": line_items,
        "has_ocr": any(o.get("has_ocr") for o in all_ocr),
        "ocr_text": "\n---\n".join(o.get("ocr_text", "") for o in all_ocr)[:1000],
        "features": agg_features,
    }

    score_resp = _score_gstin(effective, invoice_overrides=aggregated)
    resp = score_resp.model_dump()
    resp["invoice_data"] = {
        "total_amount": total_amount,
        "tax_amount": tax_amount,
        "line_items": line_items,
        "n_invoices": n_invoices,
        "ocr_text": aggregated["ocr_text"],
        "has_ocr": aggregated["has_ocr"],
        "extracted_features": agg_features,
    }
    return resp


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
        # Keep fraud_flag consistent with graph risk_score
        fraud_result = detect_circular_transactions(gstin)
        live["fraud_flag"] = int(fraud_result["risk_score"] > 0.3)
        score = predict(live)

        pipe_mod._get_time_seed = original_fn

        history.append({
            "date":  target_date.isoformat(),
            "score": score,
            "band":  _risk_band(score),
        })

    return {"gstin": gstin, "history": history}
