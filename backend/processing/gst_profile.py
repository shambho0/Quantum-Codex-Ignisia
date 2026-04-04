from backend.processing.parser import ParsedInvoice
from backend.utils.helpers import get_gstin_seed, deterministic_random

def generate_gst_profile(gstin: str) -> dict:
    """Simulate GSTIN historical financial profile (deterministic)."""
    seed = get_gstin_seed(gstin)
    rng = deterministic_random(seed)
    
    profile_type = seed % 3
    
    if profile_type == 0:
        # Struggling
        filing_consistency = rng.uniform(0.1, 0.5) 
        avg_turnover = rng.uniform(10000, 500000)
        gst_delay_days = rng.randint(10, 80)
        sector_risk = rng.uniform(0.6, 0.95)
        compliance_score = rng.uniform(0.1, 0.5)
    elif profile_type == 1:
        # Average
        filing_consistency = rng.uniform(0.5, 0.8) 
        avg_turnover = rng.uniform(400000, 2000000)
        gst_delay_days = rng.randint(0, 15)
        sector_risk = rng.uniform(0.3, 0.7)
        compliance_score = rng.uniform(0.5, 0.75)
    else:
        # Excellent
        filing_consistency = rng.uniform(0.8, 1.0) 
        avg_turnover = rng.uniform(1500000, 8000000)
        gst_delay_days = rng.randint(0, 3)
        sector_risk = rng.uniform(0.1, 0.4)
        compliance_score = rng.uniform(0.8, 0.99)
        
    return {
        "gstin": gstin,
        "filing_consistency": filing_consistency,
        "avg_turnover": avg_turnover,
        "gst_delay_days": gst_delay_days,
        "sector_risk": sector_risk,
        "compliance_score": compliance_score
    }
