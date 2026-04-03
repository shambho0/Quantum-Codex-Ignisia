@echo off
title MSME CreditIQ Launcher

echo.
echo ============================================================
echo   MSME CreditIQ - Real-Time Credit Scoring System
echo   Ignisia AI Hackathon
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing dependencies...
pip install -r backend\requirements.txt -q
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done.

:: Train model if not already trained
echo [2/3] Checking / training ML model...
if not exist "models\credit_model.cbm" (
    python train_model.py
) else (
    echo       Model already trained. Skipping.
)

:: Start FastAPI in background
set API_PORT=8010
set API_BASE=http://localhost:%API_PORT%
echo [3/3] Starting FastAPI backend on port %API_PORT%...
start "FastAPI Backend" cmd /k "uvicorn backend.main:app --host 0.0.0.0 --port %API_PORT%"

:: Wait for API to start
timeout /t 4 /nobreak >nul

:: Start Streamlit dashboard
echo       Starting Streamlit dashboard on port 8501...
echo.
echo ============================================================
echo   Dashboard: http://localhost:8501
echo   API Docs:  %API_BASE%/docs
echo ============================================================
echo.
streamlit run dashboard/app.py --server.port 8501
