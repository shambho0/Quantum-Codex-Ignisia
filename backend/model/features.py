from typing import Dict, Any
from backend.utils.helpers import normalize

def engineer_features(gst_profile: Dict[str, Any], aggregated_invoices: Dict[str, Any]) -> Dict[str, Any]:
    """Combine GST profile and Aggregated features, normalize where needed."""
    
    # Start with base stats
    features = {**gst_profile, **aggregated_invoices}
    
    # Normalizing heavy values for ML models
    features["norm_turnover"] = normalize(features.get("total_turnover", 0), 0, 10000000)
    features["norm_avg_invoice"] = normalize(features.get("avg_invoice_value", 0), 0, 500000)
    features["norm_delay_days"] = normalize(features.get("gst_delay_days", 0), 0, 90)
    
    # Derived composite risk feature
    base_compliance = features.get("compliance_score", 0.5)
    consistency = features.get("filing_consistency", 0.5)
    features["composite_compliance"] = (base_compliance * 0.7) + (consistency * 0.3)
    
    # Remove strings from features to make ML ready
    features.pop("gstin", None)
    
    return features
