import joblib
import os
import logging
from typing import Dict, Any, Tuple
from backend.model.features import to_vector

logger = logging.getLogger(__name__)

ml_credit_model = None
ml_fraud_model = None

try:
    if os.path.exists("model/credit_model.pkl"):
        ml_credit_model = joblib.load("model/credit_model.pkl")
    if os.path.exists("model/fraud_model.pkl"):
        ml_fraud_model = joblib.load("model/fraud_model.pkl")
except Exception as e:
    logger.warning(f"ML models could not be loaded: {e}")

def generate_credit_score_rule(features: Dict[str, Any]) -> int:
    base_score = 300
    compliance = features.get("composite_compliance", 0) * 200
    turnover = features.get("norm_turnover", 0) * 150
    diversity = features.get("buyer_diversity", 0) * 100
    delay_penalty = features.get("norm_delay_days", 0) * -100
    consistency = features.get("gst_consistency", 0) * 150
    score = base_score + compliance + turnover + diversity + consistency + delay_penalty
    return max(300, min(900, int(score)))

def generate_fraud_risk_rule(features: Dict[str, Any]) -> float:
    base_risk = 0.1
    sector_risk = features.get("sector_risk", 0.5) * 0.3
    compliance_penalty = (1 - features.get("compliance_score", 1.0)) * 0.4
    anomaly = features.get("high_value_ratio", 0) * 0.2
    risk = base_risk + sector_risk + compliance_penalty + anomaly
    return max(0.0, min(1.0, float(risk)))

def generate_predictions(features: Dict[str, Any]) -> Tuple[int, float, str]:
    """Returns (credit_score, fraud_risk, model_used_status)"""
    rule_score = generate_credit_score_rule(features)
    rule_fraud = generate_fraud_risk_rule(features)
    
    model_used = "fallback"
    
    if ml_credit_model and ml_fraud_model:
        try:
            vec = to_vector(features)
            ml_score_val = ml_credit_model.predict([vec])[0]
            ml_fraud_val = ml_fraud_model.predict_proba([vec])[0][1]
            
            # Hybrid scoring logic
            final_score = 0.8 * float(ml_score_val) + 0.2 * rule_score
            final_fraud = 0.8 * float(ml_fraud_val) + 0.2 * rule_fraud
            
            credit_score = max(300, min(900, int(final_score)))
            fraud_risk = max(0.0, min(1.0, float(final_fraud)))
            model_used = "catboost"
            
            return credit_score, fraud_risk, model_used
        except Exception as e:
            logger.error(f"ML Prediction failed: {e}. Falling back.")
            
    return rule_score, rule_fraud, model_used

def recommend_loan(score: int, turnover: float, fraud_risk: float) -> Dict[str, Any]:
    if fraud_risk > 0.7:
        return {"amount": 0, "tenure": "0 months", "status": "Rejected - High Fraud Risk"}
        
    if score < 500:
        return {"amount": 0, "tenure": "0 months", "status": "Rejected"}
        
    max_loan = turnover * 0.20
    
    if score >= 750:
        multiplier = 1.0
        tenure = "24 months"
    elif score >= 600:
        multiplier = 0.7
        tenure = "12 months"
    else:
        multiplier = 0.4
        tenure = "6 months"
        
    k = 5.0
    risk_discount = 1.0 / (1.0 + (k * fraud_risk))
    multiplier *= risk_discount
    
    if fraud_risk > 0.35 and tenure == "24 months":
        tenure = "12 months"
    if fraud_risk > 0.5:
        tenure = "6 months"
        
    return {
        "amount": round(max_loan * multiplier, 2),
        "tenure": tenure,
        "status": "Approved"
    }
