from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
import os

app = FastAPI(
    title="Reddit Topic Content Agent Platform",
    description="Upgrade platform with Pydantic configurations and structured generation dashboard.",
    version="2.0.0"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router, prefix="/api")

# Verify frontend directory exists
frontend_dir = os.path.abspath("frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir, exist_ok=True)

# Mount Static Files (serves index.html at root '/')
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
