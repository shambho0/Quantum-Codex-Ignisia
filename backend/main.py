from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
from backend.routes.analyze import router as analyze_router

app = FastAPI(
    title="GST Intelligence Engine",
    description="Scalable MSME Credit Scoring and ML Analysis from GST Data Streams.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.include_router(analyze_router, prefix="/api", tags=["analyze"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "GST Intelligence Engine is running."}
