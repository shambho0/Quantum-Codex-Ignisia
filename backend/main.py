from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from backend.routes.analyze import router as analyze_router

app = FastAPI(
    title="GST Intelligence Engine",
    description="Scalable MSME Credit Scoring and ML Analysis from GST Data Streams.",
    version="2.1.0"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logger.info(f"{request.url.path} took {process_time:.4f}s")

    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

# Routes
app.include_router(analyze_router, prefix="/api", tags=["analyze"])

@app.get("/")
def root():
    return {
        "message": "GST Intelligence Engine API",
        "version": "2.1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "GST Intelligence Engine is running."}