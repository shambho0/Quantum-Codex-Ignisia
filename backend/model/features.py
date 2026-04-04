from typing import Dict, Any, List

FEATURE_ORDER = [
    "total_turnover",
    "avg_invoice_value",
    "invoice_frequency",
    "buyer_diversity",
    "growth_rate",
    "gst_consistency",
    "filing_consistency",
    "avg_turnover",
    "gst_delay_days",
    "sector_risk",
    "compliance_score",
    "buyer_concentration"
]

def engineer_features(gst_profile: Dict[str, Any], aggregated_invoices: Dict[str, Any]) -> Dict[str, Any]:
    features = {**gst_profile, **aggregated_invoices}
    
    features["norm_turnover"] = min(features.get("total_turnover", 0) / 10000000, 1.0)
    features["norm_delay_days"] = min(features.get("gst_delay_days", 0) / 90, 1.0)
    features["composite_compliance"] = (features.get("gst_consistency", 0) * 0.5) + (features.get("compliance_score", 0) * 0.5)
    
    features["buyer_concentration"] = 1.0 - features.get("buyer_diversity", 0.5)
    return features

def to_vector(features: Dict[str, Any]) -> List[float]:
    return [float(features.get(f, 0.0)) for f in FEATURE_ORDER]
