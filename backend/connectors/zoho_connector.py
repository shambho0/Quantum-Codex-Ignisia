import requests


def get_zoho_invoices(access_token: str, org_id: str, gstin: str):
    url = "https://www.zohoapis.in/books/v3/invoices"

    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}"
    }

    params = {
        "organization_id": org_id
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        invoices = []

        for inv in data.get("invoices", []):
            buyer_gstin = inv.get("gst_no") or ""

            # Filter invoices related to GSTIN
            if gstin and gstin not in buyer_gstin:
                continue

            invoices.append({
                "invoice_id": inv.get("invoice_id"),
                "date": inv.get("date"),
                "amount": float(inv.get("total", 0)),
                "gst_amount": float(inv.get("tax_total", 0)),
                "buyer_gstin": buyer_gstin,
                "seller_gstin": inv.get("gst_no") or "UNKNOWN"
            })

        return invoices

    except Exception:
        return None