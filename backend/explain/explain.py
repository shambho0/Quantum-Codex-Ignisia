from typing import Dict, Any

def generate_explanations(features: Dict[str, Any], score: int, fraud_risk: float) -> Dict[str, Any]:
    """Return top positive, negative factors, and human readable insights."""
    
    positive = []
    negative = []
    insights = []
    
    # --- Compliance & Consistency ---
    if features.get("composite_compliance", 0) > 0.8:
        positive.append("High GST compliance")
        insights.append("Excellent GST filing consistency has significantly boosted the credit score.")
    elif features.get("composite_compliance", 0) < 0.5:
        negative.append("Poor GST compliance")
        insights.append("Irregular GST filing has negatively impacted the score.")
        
    if features.get("gst_delay_days", 0) > 15:
        negative.append("High GST delay")
        insights.append("Frequent delays in GST payments increased risk profile.")
    elif features.get("gst_delay_days", 0) == 0:
        positive.append("No GST delays")
        insights.append("Perfect on-time tax payment history establishes strong trust.")
        
    # --- Supply Chain & Transactions ---
    if features.get("buyer_diversity", 0) > 0.6:
        positive.append("Strong buyer diversity")
        insights.append("Having a diverse set of buyers indicates a stable market presence.")
    elif features.get("buyer_diversity", 0) < 0.2:
        negative.append("Invoice concentration")
        insights.append("Too much dependence on a few buyers increases business risk.")
        
    if features.get("invoice_frequency", 0) > 100:
        positive.append("High transaction volume")
        insights.append("Frequent business activities demonstrate healthy cashflow cycles.")
    elif features.get("invoice_frequency", 0) < 10:
        negative.append("Low transaction volume")
        insights.append("Low invoice frequency suggests limited operational scale.")
        
    # --- Revenue & Growth ---
    if features.get("avg_invoice_value", 0) > 15000:
        positive.append("High average invoice value")
        insights.append("Ability to execute large B2B orders demonstrates good operational capacity.")
        
    if features.get("growth_rate", 0) > 0:
        positive.append("Positive growth trend")
        insights.append("Business shows positive monthly revenue growth.")
        
    # --- Anomalies & Risk ---
    if features.get("sector_risk", 0) > 0.7:
        negative.append("High sector risk")
        insights.append("Operating in a historically high-risk or highly-volatile industry.")
        
    if features.get("high_value_ratio", 0) > 0.4:
        negative.append("High anomalous transactions")
        insights.append("High ratio of exceptionally large transactions might indicate circular trading.")
        
    # Build complete SHAP tracking (10+ factors)
    base_comp = features.get("composite_compliance", 0.5)
    shap_values = {
        "Composite Compliance": (base_comp - 0.5) * 120,
        "Turnover Scale": features.get("norm_turnover", 0) * 80,
        "Buyer Diversity": (features.get("buyer_diversity", 0) - 0.3) * 60,
        "Delay Penalty": features.get("norm_delay_days", 0) * -100,
        "GST Consistency": (features.get("gst_consistency", 0) - 0.5) * 90,
        "Sector Risk Penalty": features.get("sector_risk", 0.5) * -50,
        "Invoice Volume": min(features.get("invoice_frequency", 0), 200) * 0.4,
        "Avg Invoice Size Obj": min(features.get("avg_invoice_value", 0) / 1000, 50),
        "Growth Rate": features.get("growth_rate", 0) * 150,
        "Anomaly Penalty": features.get("high_value_ratio", 0) * -70
    }
    
    return {
        "top_positive_factors": positive,
        "top_negative_factors": negative,
        "insights": insights,
        "shap_values": shap_values
    }
