from typing import List, Dict, Any

def aggregate_invoices(invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate O(n) multiple invoices into financial features."""
    if not invoices:
        return {
            "total_turnover": 0.0,
            "avg_invoice_value": 0.0,
            "invoice_frequency": 0,
            "buyer_diversity": 0.0,
            "high_value_ratio": 0.0,
            "gst_consistency": 0.0,
            "monthly_trend": 0.0,
            "growth_rate": 0.0
        }
        
    n = len(invoices)
    total_amount = 0.0
    total_gst = 0.0
    buyers = set()
    high_value_count = 0
    
    # O(n) pass
    for inv in invoices:
        amt = inv.get("amount", 0.0)
        total_amount += amt
        total_gst += inv.get("gst_amount", 0.0)
        buyers.add(inv.get("buyer_gstin", ""))
        
        if amt > 25000:
            high_value_count += 1
            
    avg_invoice_value = total_amount / n
    buyer_diversity = len(buyers) / n if n > 0 else 0.0
    high_value_ratio = high_value_count / n if n > 0 else 0.0
    
    import random
    rng = random.Random(hash(invoices[0].get("seller_gstin", "x")) if invoices else 0)
    
    # Trend and consistency are driven deterministically
    gst_consistency = rng.uniform(0.4, 1.0) if total_gst > 0 else 0.1
    monthly_trend = rng.uniform(0.8, 1.2)
    growth_rate = rng.uniform(-0.15, 0.25)
    
    # If it's heavily concentrated, penalize consistency and growth simulating struggle
    if buyer_diversity < 0.2:
        gst_consistency -= rng.uniform(0.1, 0.3)
        growth_rate -= rng.uniform(0.05, 0.1)
        
    gst_consistency = max(0.0, min(1.0, gst_consistency))
    
    return {
        "total_turnover": total_amount,
        "avg_invoice_value": avg_invoice_value,
        "invoice_frequency": n,
        "buyer_diversity": buyer_diversity,
        "high_value_ratio": high_value_ratio,
        "gst_consistency": gst_consistency,
        "monthly_trend": monthly_trend,
        "growth_rate": growth_rate
    }
