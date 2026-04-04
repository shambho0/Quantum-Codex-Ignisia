"""
dashboard/app.py
-----------------
Streamlit dashboard for MSME Credit Scoring.

Shows:
  - Credit score gauge
  - Risk band
  - Top 5 SHAP-driven reasons
  - Fraud risk indicator
  - Loan recommendation
  - Feature contribution bar chart
  - Score trend over last 7 days

Run with:
  streamlit run dashboard/app.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.gstin import validate_gstin

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="MSME CreditIQ | Ignisia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #0a0e1a; }

.metric-box {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.3s ease;
}

.score-card {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1124 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}

.score-card:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 45px rgba(159,122,234,0.3);
    border-color: rgba(159,122,234,0.5);
}

.metric-box:hover, .reason-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 24px rgba(99,179,237,0.15);
    border-color: rgba(99,179,237,0.4);
}

.metric-label { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; margin-top: 0.3rem; }

.fraud-clean   { color: #38a169; }
.fraud-risk    { color: #e53e3e; }
.fraud-warning { color: #d69e2e; }

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #90cdf4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.8rem;
    border-bottom: 1px solid rgba(144,205,244,0.2);
    padding-bottom: 0.4rem;
}

.stTextInput > div > div > input {
    background: #1a1f35 !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4299e1, #805ad5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.6rem !important;
    font-size: 1rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

.timestamp { font-size: 0.72rem; color: #4a5568; text-align: right; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <h1 style="font-size:2.4rem; font-weight:800; color:#e2e8f0; margin:0;">
         MSME <span style="background:linear-gradient(135deg,#63b3ed,#9f7aea);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">CreditIQ</span>
    </h1>
    <p style="color:#718096; font-size:1rem; margin-top:0.3rem;">
        Real-Time Credit Scoring via Alternative Business Signals
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
# Input — Two modes: GSTIN text or Invoice upload
# ──────────────────────────────────────────────

tab_gstin, tab_invoice = st.tabs(["🔍 Enter GSTIN", "📄 Upload Invoice"])

with tab_gstin:
    col_input, col_btn, col_sample = st.columns([4, 1.2, 2])
    with col_input:
        gstin_input = st.text_input(
            "Enter GSTIN",
            placeholder="e.g. 27ABCDE1234F1Z5",
            label_visibility="collapsed",
            max_chars=15,
        )
    with col_btn:
        score_btn = st.button("⚡ Score Now")
    with col_sample:
        samples = [
            "27ABCDE1234F1Z5", "29AABCU9603R1ZX",
            "07AAACN9536B1ZQ", "24AAACR5055K1ZJ", "33AADCB2230M1Z8",
        ]
        sample_choice = st.selectbox("Try a sample", ["— select —"] + samples, label_visibility="collapsed")
        if sample_choice != "— select —":
            gstin_input = sample_choice

with tab_invoice:
    inv_col1, inv_col2 = st.columns([2, 1.5])
    with inv_col1:
        uploaded_files = st.file_uploader(
            "Upload GST Invoice Images (up to 10)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload one or more GST invoice images from the same company.",
        )
    with inv_col2:
        gstin_manual = st.text_input(
            "Enter GSTIN from the invoices",
            placeholder="e.g. 27ABCDE1234F1Z5",
            max_chars=15,
            help="Enter the 15-character GSTIN printed on the invoices.",
        )
    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 5))
        for i, uf in enumerate(uploaded_files[:5]):
            with cols[i]:
                st.image(uf, caption=f"Invoice {i+1}", width=150)
        if len(uploaded_files) > 5:
            st.caption(f"... and {len(uploaded_files) - 5} more")
    score_invoice_btn = st.button(f"📄 Score from {len(uploaded_files) if uploaded_files else 0} Invoice(s)")

# ──────────────────────────────────────────────
# API call
# ──────────────────────────────────────────────

def call_api(gstin: str) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/api/analyze", json={"gstin": gstin, "source": "dummy"}, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def call_api_invoice(file_list: list, gstin: str = "") -> dict | None:
    try:
        files = [("files", (f.name, f.read(), "image/png")) for f in file_list]
        r = requests.post(
            f"{API_BASE}/api/score-invoice",
            files=files,
            data={"gstin": gstin},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def call_history(gstin: str, current_score: int = None) -> list | None:
    try:
        url = f"{API_BASE}/api/score-history/{gstin}"
        if current_score is not None:
            url += f"?current_score={int(current_score)}"
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.json().get("history", [])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# Dashboard rendering
# ──────────────────────────────────────────────


# Determine which mode triggered scoring
data = None
gstin_display = None

if score_btn and gstin_input:
    gstin = gstin_input.strip().upper()
    _ok, _msg = validate_gstin(gstin)
    if not _ok:
        st.error(_msg)
    else:
        with st.spinner("Fetching live signals and computing score..."):
            data = call_api(gstin)
        gstin_display = gstin

elif score_invoice_btn and uploaded_files:
    for uf in uploaded_files:
        uf.seek(0)
    with st.spinner(f"Analyzing {len(uploaded_files)} invoice(s) and computing score..."):
        data = call_api_invoice(
            uploaded_files,
            (gstin_manual or "").strip().upper(),
        )
    if data:
        gstin_display = data.get("gstin", "")

if data:
    score   = data.get("credit_score", 0)
    band    = "Low" if score >= 750 else ("Medium" if score >= 600 else "High")
    reasons = data.get("explanations", [])
    features= data.get("features", {})
    fraud_risk   = data.get("fraud_risk", 0.0)
    loan_amount  = data.get("loan_amount", 0)
    loan_tenure  = data.get("loan_tenure", "")
    shap         = data.get("shap_values", {})
    ts      = "Live (Simulated Data)"

    # ── Row 1: Score card + metrics ──
    r1c1, r1c2 = st.columns([1.6, 2.4])

    with r1c1:
        band_class = {"Low": "band-low", "Medium": "band-medium", "High": "band-high"}[band]
        band_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[band]
        st.markdown(f"""
        <div class="score-card">
            <div style="color:#718096; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em;">Credit Score</div>
            <div class="score-number">{score}</div>
            <div class="{band_class}">{band_emoji} {band} Risk</div>
        <div style="color:#4a5568; font-size:0.75rem; margin-top:0.8rem;">Scale: 300 – 900</div>
        </div>
        """, unsafe_allow_html=True)
        # Dynamic progress bar for interactivity
        st.progress(max(0.0, min((score - 300) / 600, 1.0)))
        st.markdown(f'<div class="timestamp"> Freshness: {ts}</div>', unsafe_allow_html=True)

    with r1c2:
        # Metrics grid
        m1, m2, m3 = st.columns(3)
        with m1:
            trend = features.get("monthly_trend", 1.0)
            delta_val = (trend - 1.0) * 100
            st.metric("Total Turnover", f"₹{int(features.get('total_turnover', 0)):,}", delta=f"{delta_val:+.1f}% (vs last mo)")
        with m2:
            gst_pct = int(features.get("composite_compliance", 0) * 100)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Composite Compliance</div>
                <div class="metric-value">{gst_pct}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            growth = features.get("growth_rate", 0) * 100
            st.metric("Avg Invoice", f"₹{int(features.get('avg_invoice_value', 0)):,}", delta=f"{growth:+.1f}% YoY")

        m4, m5, m6 = st.columns(3)
        with m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Recommended Loan</div>
                <div class="metric-value">₹{loan_amount:,}</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Tenure</div>
                <div class="metric-value">{loan_tenure}</div>
            </div>""", unsafe_allow_html=True)
        with m6:
            f_class = "fraud-risk" if fraud_risk > 0.5 else ("fraud-warning" if fraud_risk > 0.2 else "fraud-clean")
            f_icon  = "🔴" if fraud_risk > 0.5 else ("🟡" if fraud_risk > 0.2 else "🟢")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Fraud Risk</div>
                <div class="metric-value {f_class}">{f_icon} {fraud_risk:.0%}</div>
            </div>""", unsafe_allow_html=True)

    # ── Row 2: Reasons + SHAP chart ──
    r2c1, r2c2 = st.columns([1.4, 2.6])

    with r2c1:
        st.markdown('<div class="section-title"> Score Reasons</div>', unsafe_allow_html=True)
        neg_kws = ["poor", "low", "weak", "declining", "negative", "high cash flow", "fraud", "delay", "limited", "concern"]
        for reason in reasons:
            is_neg = any(kw in reason.lower() for kw in neg_kws)
            cls    = "reason-negative" if is_neg else "reason-positive"
            icon   = "" if is_neg else ""
            st.markdown(f'<div class="reason-card {cls}">{icon} {reason}</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="section-title" style="margin-top:2rem;"> Data Source</div>', unsafe_allow_html=True)
        source_name = data.get("data_source", "dummy").upper()
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Current Strategy Backend</div>
            <div class="metric-value" style="color:#63b3ed; font-size:1.2rem;">{source_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="section-title"> Feature Contributions (SHAP)</div>', unsafe_allow_html=True)
        if shap:
            shap_df = (
                pd.DataFrame({"Feature": list(shap.keys()), "SHAP Value": list(shap.values())})
                .sort_values("SHAP Value", key=abs, ascending=True)
            )
            colors = ["#e53e3e" if v < 0 else "#38a169" for v in shap_df["SHAP Value"]]

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#0d1124")
            ax.set_facecolor("#0d1124")
            bars = ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, height=0.55)
            ax.axvline(0, color="#4a5568", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on score)", color="#718096", fontsize=9)
            ax.tick_params(colors="#a0aec0", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3748")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            pos_patch = mpatches.Patch(color="#38a169", label="Positive impact")
            neg_patch = mpatches.Patch(color="#e53e3e", label="Negative impact")
            ax.legend(handles=[pos_patch, neg_patch], facecolor="#1a1f35",
                      labelcolor="#a0aec0", fontsize=8, framealpha=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.markdown("<p style='color:#a0aec0;'>No SHAP data available from backend.</p>", unsafe_allow_html=True)

    # ── Row 3: Score trend + Fraud graph ──
    r3c1, r3c2 = st.columns([2, 1.4])

    with r3c1:
        st.markdown('<div class="section-title"> Score Trend (Last 7 Days)</div>', unsafe_allow_html=True)
        history = call_history(gstin_display, current_score=score)
        if history:
            hist_df = pd.DataFrame(history)
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            fig2.patch.set_facecolor("#0d1124")
            ax2.set_facecolor("#0d1124")

            scores_h = hist_df["score"].tolist()
            dates_h  = hist_df["date"].tolist()
            ax2.plot(dates_h, scores_h, color="#63b3ed", linewidth=2.5, marker="o",
                     markersize=6, markerfacecolor="#9f7aea", markeredgecolor="#63b3ed")
            ax2.fill_between(dates_h, scores_h, alpha=0.12, color="#63b3ed")

            ax2.axhline(750, color="#38a169", linewidth=0.8, linestyle="--", alpha=0.6, label="Low risk (750)")
            ax2.axhline(600, color="#d69e2e", linewidth=0.8, linestyle="--", alpha=0.6, label="Medium risk (600)")
            ax2.set_ylim(280, 920)
            ax2.tick_params(colors="#a0aec0", labelsize=8)
            plt.xticks(rotation=30, ha="right")
            ax2.legend(facecolor="#1a1f35", labelcolor="#a0aec0", fontsize=8, framealpha=0.8)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#2d3748")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    with r3c2:
        st.markdown('<div class="section-title"> Fraud Analysis</div>', unsafe_allow_html=True)
        fraud_icon  = "🔴 HIGH RISK" if fraud_risk > 0.5 else ("🟡 MODERATE" if fraud_risk > 0.2 else "🟢 CLEAN")
        fraud_color = "#e53e3e" if fraud_risk > 0.5 else ("#d69e2e" if fraud_risk > 0.2 else "#38a169")

        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:0.6rem;">
            <div class="metric-label">Graph Risk Level</div>
            <div style="font-size:1.3rem;font-weight:700;color:{fraud_color};">{fraud_icon}</div>
        </div>""", unsafe_allow_html=True)

        fd1, fd2 = st.columns(2)
        with fd1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Sector Risk</div>
                <div class="metric-value">{features.get('sector_risk', 0):.2f}</div>
            </div>""", unsafe_allow_html=True)
        with fd2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Anomaly Ratio</div>
                <div class="metric-value">{features.get('high_value_ratio', 0):.2f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Row 4: Feature table ──
    with st.expander(" Full Feature Breakdown"):
        feat_labels = {
            "gst_consistency":       "GST Consistency",
            "gst_delay":             "GST Delay (days)",
            "invoice_growth":        "Invoice Growth Rate",
            "avg_upi_inflow":        "Avg UPI Inflow (₹)",
            "inflow_outflow_ratio":  "Inflow/Outflow Ratio",
            "transaction_frequency": "Transaction Frequency",
            "cashflow_volatility":   "Cashflow Volatility",
            "shipment_rate":         "Shipment Rate (e-Way Bills)",
            "business_growth":       "Business Growth Rate",
            "fraud_flag":            "ML Fraud Flag",
        }
        feat_df = pd.DataFrame([
            {"Feature": feat_labels.get(k, k), "Value": round(v, 4) if isinstance(v, float) else v}
            for k, v in features.items()
        ])
        st.dataframe(feat_df, width="stretch", hide_index=True)

    # ── Row 5: Invoice OCR results (only for invoice uploads) ──
    inv_data = data.get("invoice_data")
    if inv_data and inv_data.get("has_ocr"):
        n_inv = inv_data.get("n_invoices", 1)
        st.markdown(f'<div class="section-title">📄 Invoice OCR Extraction ({n_inv} invoice{"s" if n_inv > 1 else ""})</div>', unsafe_allow_html=True)
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Invoice Total</div>
                <div class="metric-value">₹{inv_data['total_amount']:,.2f}</div>
            </div>""", unsafe_allow_html=True)
        with oc2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Tax Amount</div>
                <div class="metric-value">₹{inv_data['tax_amount']:,.2f}</div>
            </div>""", unsafe_allow_html=True)
        with oc3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Line Items</div>
                <div class="metric-value">{inv_data['line_items']}</div>
            </div>""", unsafe_allow_html=True)

        ext_feats = inv_data.get("extracted_features", {})
        if ext_feats:
            st.markdown('<div style="color:#90cdf4;font-size:0.85rem;margin-top:0.8rem;">⚡ Features extracted from invoice → fed to CatBoost model:</div>', unsafe_allow_html=True)
            ef_df = pd.DataFrame([
                {"Feature": feat_labels.get(k, k), "Extracted Value": round(v, 4) if isinstance(v, float) else v}
                for k, v in ext_feats.items()
            ])
            st.dataframe(ef_df, width="stretch", hide_index=True)

        with st.expander("🔍 Raw OCR Text"):
            st.code(inv_data.get("ocr_text", "(empty)"), language=None)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#4a5568;">
        <div style="font-size:4rem; margin-bottom:1rem;"></div>
        <h3 style="color:#718096;">Enter a GSTIN or upload a GST invoice to get started</h3>
        <p>The system will ingest live business signals and return a credit score in seconds.</p>
        <div style="display:flex; justify-content:center; gap:2rem; margin-top:2rem; flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:1rem 1.5rem;border:1px solid rgba(255,255,255,0.08);">
                📊 <b style="color:#e2e8f0;">Credit Score</b><br><span style="font-size:0.85rem;">300–900 scale</span>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:1rem 1.5rem;border:1px solid rgba(255,255,255,0.08);">
                📄 <b style="color:#e2e8f0;">Invoice OCR</b><br><span style="font-size:0.85rem;">Upload & auto-detect</span>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:1rem 1.5rem;border:1px solid rgba(255,255,255,0.08);">
                🕵️ <b style="color:#e2e8f0;">Fraud Detection</b><br><span style="font-size:0.85rem;">Graph analysis</span>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:1rem 1.5rem;border:1px solid rgba(255,255,255,0.08);">
                💰 <b style="color:#e2e8f0;">Loan Suggestion</b><br><span style="font-size:0.85rem;">Amount + tenure</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

