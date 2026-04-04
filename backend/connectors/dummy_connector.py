import numpy as np


def generate_dummy_invoices(gstin: str, n=200):
    seed = sum(ord(c) for c in gstin)
    np.random.seed(seed)

    invoices = []

    for i in range(n):
        amount = np.random.randint(1000, 100000)
        gst_amount = amount * 0.18

        invoices.append({
            "invoice_id": f"DUMMY-{i}",
            "date": f"2025-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
            "amount": float(amount),
            "gst_amount": float(gst_amount),
            "buyer_gstin": f"{np.random.randint(10,99)}ABCDE1234F1Z5",
            "seller_gstin": gstin
        })

    return invoices