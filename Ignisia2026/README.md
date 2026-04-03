# Quantum-Codex-Ignisia
# Real-Time MSME Credit Scoring via Alternative Business Signals


> **Built for the Ignisia AI Hackathon (Domain: FinTech - FT02)** 

## The Problem

Banks and NBFCs turn down about 80% of MSME loan applications right now because new businesses (those less than two years old) don't have a formal credit history, ITR filings, or audited financials. Traditional underwriting only uses documents that are already months out of date.

At the same time, real-time digital footprints like the speed of GST e-invoice generation, the flow of UPI cash, and the trends in e-way bill volume give a very accurate picture of how healthy a business is right now, which traditional underwriting completely misses.

##  Our Solution

This project creates a dynamic, almost real-time scoring pipeline that uses different business signals to rate MSMEs. We give loan officers a credit score that is useful and easy to understand by using machine learning, network analysis, and explainable AI.

###  Key Features

* **Mock Data Pipeline:** Pretends to take in GST filing speed, UPI transaction cadence, and e-way bill volume in real time.
* Explainable ML Scoring: This method uses a Gradient Boosting model with SHAP to give a credit score (300–900) and clear, easy-to-understand reasons for the score.
* **Fraud Detection Engine:** Uses graph-based network analysis to find and flag high-risk circular transaction topologies where linked MSMEs move the same UPI funds around to make their scores look better.
* **REST API:** A quick and dependable endpoint that takes a GSTIN and gives back the credit score, risk band, top five SHAP reasons, suggested loan amount, and a timestamp for freshness.
* **Dashboard that lets you interact with it:** A frontend UI that shows contributions to features, score trends over time, and alerts for fraud.

##  Technology Stack

* **Backend Framework:** FastAPI 
* **Machine Learning:** CatBoost
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Network Analysis (Fraud):** NetworkX
* **Data Processing:** Pandas, NumPy 
* **Frontend Dashboard:** Streamlit 

##  Architecture & Workflow

1. **Data Ingestion:** Synthetic pipelines create time-series financial signals and work well with sparse data.
2. **Feature Engineering:** Raw timestamps and amounts are turned into aggregate features, like filing velocity and transaction frequency.
3. **Fraud Check:** A network graph looks at recent UPI flow patterns to find circular logic.
4. **Inference:** The gradient boosting model takes in the features and gives a base score.
5. Explainability Extraction: SHAP values turn the most important features into easy-to-understand insights.
6. **Delivery:** The FastAPI endpoint sends the full payload to the Streamlit dashboard.
