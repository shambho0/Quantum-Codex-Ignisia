from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel, Field
from backend.connectors.router import fetch_invoices
from backend.processing.gst_profile import generate_gst_profile
from backend.processing.aggregator import aggregate_invoices
from backend.model.features import engineer_features
from backend.model.scorer import generate_credit_score, generate_fraud_risk, recommend_loan
from backend.explain.explain import generate_explanations

router = APIRouter()

class AnalyzeRequest(BaseModel):
    gstin: str = Field(..., description="15-character GSTIN")
    source: str = Field(default="dummy", description="Data source: 'zoho' or 'dummy'")
    access_token: Optional[str] = Field(default=None, description="OAuth token for Zoho Books")

@router.post("/analyze")
async def analyze_gstin(request: AnalyzeRequest):
    """
    Process GSTIN and fetch external invoice pipelines to generate intelligence insights.
    """
    if not request.gstin or len(request.gstin) != 15:
        raise HTTPException(status_code=400, detail="Invalid GSTIN length. Must be 15 characters.")
    
    # 1. Fetch Invoices via Connector Router
    invoices = fetch_invoices(request.source, request.gstin, request.access_token)
    
    # 2. Determine final data source status (handle fallback to dummy)
    final_source = request.source
    if request.source == "zoho" and any(inv.get("invoice_id", "").startswith("DUMMY") for inv in invoices[:1]):
        final_source = "dummy_fallback"
        
    # 3. Generate GST deterministic profile
    gst_profile = generate_gst_profile(request.gstin)
    
    # 4. Aggregate extracted features
    aggregated_invoices = aggregate_invoices(invoices)
    
    # 5. Fuse features
    final_features = engineer_features(gst_profile, aggregated_invoices)
    
    # 6. Apply scoring logic
    credit_score = generate_credit_score(final_features)
    fraud_risk = generate_fraud_risk(final_features)
    
    # 7. Get loan logic
    total_turnover = aggregated_invoices.get("total_turnover", 0.0)
    loan_info = recommend_loan(credit_score, total_turnover)
    
    # 8. Explainability
    explanations = generate_explanations(final_features, credit_score, fraud_risk)
    
    return {
        "gstin": request.gstin,
        "data_source": final_source,
        "credit_score": credit_score,
        "fraud_risk": fraud_risk,
        "loan_amount": loan_info["amount"],
        "loan_tenure": loan_info["tenure"],
        "features": final_features,
        "explanations": explanations["insights"],
        "top_positive_factors": explanations["top_positive_factors"],
        "top_negative_factors": explanations["top_negative_factors"],
        "shap_values": explanations.get("shap_values", {})
    }

@router.get("/score-history/{gstin}")
async def score_history(gstin: str, n: int = 7):
    """Simulate score history over the last N days."""
    import datetime
    import random
    
    if len(gstin) != 15:
        raise HTTPException(400, "Invalid GSTIN")
        
    seed = sum(ord(c) for c in gstin)
    rng = random.Random(seed)
    base_score = rng.randint(400, 850)
    
    history = []
    for days_ago in range(n - 1, -1, -1):
        target_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        daily_score = base_score + rng.randint(-30, 30)
        daily_score = max(300, min(900, daily_score))
        
        history.append({
            "date": target_date.isoformat(),
            "score": daily_score,
            "band": "Low" if daily_score >= 750 else ("Medium" if daily_score >= 600 else "High")
        })
        # progress the base slowly
        base_score += rng.randint(-5, 5)
        
    return {"gstin": gstin, "history": history}
