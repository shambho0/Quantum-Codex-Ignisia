from typing import List, Dict, Any
from fastapi import UploadFile
import asyncio
from pydantic import BaseModel

class ParsedInvoice(BaseModel):
    invoice_id: str
    date: str
    amount: float
    gst_amount: float
    buyer_gstin: str
    seller_gstin: str

async def parse_single_pdf(file: UploadFile) -> Dict[str, Any]:
    """Mock async PDF parsing using asyncio.sleep to simulate I/O."""
    await asyncio.sleep(0.05) # non-blocking I/O simulation
    
    # In a real app, use PyMuPDF/pdfplumber here
    # Mocking extraction based on filename or randomness
    size = file.size if file.size else 1000
    seed_val = hash(file.filename) + size
    import random
    rng = random.Random(seed_val)
    
    amount = round(rng.uniform(1000, 50000), 2)
    gst_amount = round(amount * 0.18, 2)
    
    return {
        "invoice_id": f"INV-{rng.randint(1000, 9999)}",
        "date": "2023-10-01",
        "amount": amount,
        "gst_amount": gst_amount,
        "buyer_gstin": "BUYER1234567890",
        "seller_gstin": "SELLER098765432"
    }

async def parse_pdfs(files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Process multiple invoice PDFs efficiently with asyncio.gather."""
    tasks = [parse_single_pdf(file) for file in files]
    results = await asyncio.gather(*tasks)
    return results

def expand_invoices_synthetically(invoices: List[Dict[str, Any]], target_count: int = 100) -> List[Dict[str, Any]]:
    """Simulate scaling by expanding small invoice batches to 100+ with variation."""
    if len(invoices) >= target_count or len(invoices) == 0:
        return invoices
        
    import random
    expanded = []
    
    while len(expanded) < target_count:
        base_inv = random.choice(invoices)
        variation = random.uniform(0.9, 1.1)
        new_inv = base_inv.copy()
        new_inv["amount"] = round(base_inv["amount"] * variation, 2)
        new_inv["gst_amount"] = round(base_inv["gst_amount"] * variation, 2)
        new_inv["invoice_id"] = f"{base_inv['invoice_id']}_EXT_{len(expanded)}"
        expanded.append(new_inv)
        
    return expanded
