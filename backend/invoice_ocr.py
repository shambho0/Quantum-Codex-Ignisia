"""
backend/invoice_ocr.py
-----------------------
Extract financial data from GST invoice images using EasyOCR,
then map extracted values to CatBoost model features.

No external binaries needed — easyocr is pure Python.
"""
import re
import numpy as np
from PIL import Image

try:
    import easyocr
    _reader = None
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Standard GSTIN pattern: 2 digits + 5 letters + 4 digits + 1 letter + 1 digit + Z + 1 alphanum
GSTIN_RE = re.compile(r'\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z][A-Z\d]')
AMOUNT_RE = re.compile(r'[\d,]+\.\d{2}')
TAX_RE = re.compile(r'(?:CGST|SGST|IGST|GST)\s*[@:]?\s*[\d.]*%?\s*[:\s]*([\d,]+\.\d{2})')

_NO_OCR = {
    "gstin": None, "total_amount": 0, "tax_amount": 0,
    "line_items": 0, "ocr_text": "", "has_ocr": False, "features": {},
}

# Common OCR misreads: fix letters/digits that get swapped
_OCR_FIXES = str.maketrans({
    'O': '0', 'o': '0',
    'I': '1', 'l': '1',
    'S': '5', 's': '5',
    'B': '8',
    'G': '6',
})


def _fix_gstin_ocr(text: str) -> str:
    """Apply common OCR corrections to help GSTIN detection."""
    # Remove spaces/dashes within potential GSTIN strings
    return re.sub(r'(?<=\d{2}[A-Z]{3})\s+', '', text)


def _find_gstin(text: str) -> str | None:
    """Try multiple strategies to find a GSTIN in OCR text."""
    upper = text.upper()

    # Strategy 1: Direct regex match
    m = GSTIN_RE.search(upper)
    if m:
        return m.group()

    # Strategy 2: Look near "GSTIN" or "GST" keywords
    for kw_match in re.finditer(r'(?:GSTIN|GST\s*(?:IN|No|Number|#)?)\s*[:\-]?\s*', upper):
        after = upper[kw_match.end():kw_match.end() + 25]
        # Remove spaces/special chars that OCR might insert
        cleaned = re.sub(r'[\s\-\.:]', '', after)
        m = GSTIN_RE.search(cleaned)
        if m:
            return m.group()

    # Strategy 3: Find any 15-char alphanumeric block that looks GSTIN-like
    # (2 digits at start, ends with Z + alphanum)
    blocks = re.findall(r'[A-Z0-9]{13,17}', re.sub(r'[\s\-\.]', '', upper))
    for block in blocks:
        m = GSTIN_RE.search(block)
        if m:
            return m.group()

    # Strategy 4: Fix common OCR misreads in digit/letter positions
    # GSTIN: DD LLLLL DDDD L D L X  (D=digit, L=letter, X=alphanum)
    for block in blocks:
        if len(block) >= 15:
            candidate = block[:15]
            # Fix positions that should be digits (0-1, 7-10, 12)
            fixed = list(candidate)
            digit_pos = [0, 1, 7, 8, 9, 10, 12]
            for p in digit_pos:
                if p < len(fixed) and fixed[p] in _OCR_FIXES:
                    fixed[p] = _OCR_FIXES[fixed[p]]
            fixed_str = ''.join(fixed)
            m = GSTIN_RE.search(fixed_str)
            if m:
                return m.group()

    return None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


def extract_invoice_data(image: Image.Image) -> dict:
    """
    OCR the invoice image, extract GSTIN + financials,
    and map to CatBoost-compatible feature overrides.
    """
    if not HAS_OCR:
        return dict(_NO_OCR)

    try:
        reader = _get_reader()
        img_array = np.array(image.convert("RGB"))
        results = reader.readtext(img_array)
        # Join with spaces, preserving line structure
        text = '\n'.join(r[1] for r in results)
    except Exception as e:
        return dict(_NO_OCR)

    # ── Extract GSTIN (with fuzzy matching) ──
    gstin = _find_gstin(text)

    # ── Extract monetary amounts ──
    amounts = []
    for a in AMOUNT_RE.findall(text):
        val = float(a.replace(',', ''))
        if val > 10:
            amounts.append(val)

    # ── Extract tax amounts (CGST/SGST/IGST) ──
    upper = text.upper()
    taxes = []
    for t in TAX_RE.findall(upper):
        taxes.append(float(t.replace(',', '')))

    total = max(amounts) if amounts else 0
    tax_total = sum(taxes)

    # ── Map OCR data → CatBoost features ──
    features = {}

    # avg_upi_inflow: use invoice total as a proxy for monthly inflow
    if total > 0:
        features["avg_upi_inflow"] = total

    # gst_consistency: tax present → compliant filing
    if tax_total > 0 and total > 0:
        tax_ratio = tax_total / total
        features["gst_consistency"] = min(1.0, 0.7 + tax_ratio)
        features["gst_delay"] = 2  # tax paid = timely filing
    elif total > 0:
        features["gst_consistency"] = 0.4
        features["gst_delay"] = 10

    # transaction_frequency: line items indicate activity
    if len(amounts) > 0:
        features["transaction_frequency"] = max(len(amounts) * 8, 15)

    # invoice_growth & business_growth: inferred from invoice size
    if total > 50000:
        features["invoice_growth"] = 0.15
        features["business_growth"] = 0.10
    elif total > 10000:
        features["invoice_growth"] = 0.05
        features["business_growth"] = 0.03
    elif total > 0:
        features["invoice_growth"] = -0.05
        features["business_growth"] = -0.05

    # inflow_outflow_ratio: higher invoice = healthier ratio
    if total > 0:
        features["inflow_outflow_ratio"] = min(2.0, 0.8 + (total / 200000))

    # cashflow_volatility
    features["cashflow_volatility"] = 0.4

    # shipment_rate: if e-way bill keywords found
    eway_kws = ["E-WAY", "EWAY", "EWB", "TRANSPORT", "SHIPMENT", "CONSIGNMENT"]
    if any(kw in upper for kw in eway_kws):
        features["shipment_rate"] = 25
    elif total > 0:
        features["shipment_rate"] = 10

    return {
        "gstin": gstin,
        "total_amount": total,
        "tax_amount": tax_total,
        "line_items": len(amounts),
        "ocr_text": text[:500],
        "has_ocr": True,
        "features": features,
    }
