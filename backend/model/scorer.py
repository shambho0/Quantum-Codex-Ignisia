from typing import Dict, Any

def generate_credit_score(features: Dict[str, Any]) -> int:
    """Multi-factor weighted model to generate score from 300 to 900."""
    base_score = 300
    
    # weights
    compliance = features.get("composite_compliance", 0) * 200     # up to +200
    turnover = features.get("norm_turnover", 0) * 150             # up to +150
    diversity = features.get("buyer_diversity", 0) * 100          # up to +100
    delay_penalty = features.get("norm_delay_days", 0) * -100      # up to -100
    consistency = features.get("gst_consistency", 0) * 150         # up to +150
    
    score = base_score + compliance + turnover + diversity + consistency + delay_penalty
    return max(300, min(900, int(score)))

def generate_fraud_risk(features: Dict[str, Any]) -> float:
    """Generate risk score from 0 to 1 based on mismatch, inconsistencies, anomalies."""
    base_risk = 0.1
    
    sector_risk = features.get("sector_risk", 0.5) * 0.3
    compliance_penalty = (1 - features.get("compliance_score", 1.0)) * 0.4
    anomaly = features.get("high_value_ratio", 0) * 0.2
    
    risk = base_risk + sector_risk + compliance_penalty + anomaly
    return max(0.0, min(1.0, float(risk)))

def recommend_loan(score: int, turnover: float) -> Dict[str, Any]:
    """Dynamic loan recommendation based on score and turnover."""
    if score < 500:
        return {"amount": 0, "tenure": "0 months", "status": "Rejected"}
        
    # Scale loan amount dynamically based on turnover
    max_loan = turnover * 0.20  # 20% of turnover
    
    if score >= 750:
        multiplier = 1.0
        tenure = "24 months"
    elif score >= 600:
        multiplier = 0.7
        tenure = "12 months"
    else:
        multiplier = 0.4
        tenure = "6 months"
        
    recommended_amount = max_loan * multiplier
    
    return {
        "amount": round(recommended_amount, 2),
        "tenure": tenure,
        "status": "Approved"
    }
