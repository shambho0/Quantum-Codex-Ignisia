from backend.connectors.zoho_connector import get_zoho_invoices
from backend.connectors.dummy_connector import generate_dummy_invoices


def fetch_invoices(source: str, gstin: str, token: str = None, org_id: str = None):
    
    if source == "zoho" and token and org_id:
        invoices = get_zoho_invoices(token, org_id, gstin)

        if invoices and len(invoices) > 0:
            return invoices

    # Fallback to dummy if Zoho fails
    return generate_dummy_invoices(gstin)