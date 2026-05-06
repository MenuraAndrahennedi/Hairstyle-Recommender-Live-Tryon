from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from generative_app.api.routes_assets import router as assets_router
from generative_app.api.routes_generative_tryon import router as generative_tryon_router
from generative_app.api.routes_health import router as health_router
from generative_app.config import BACKEND_ROOT


app = FastAPI(
    title="Hairstyle Recommender Generative Try-On API",
    version="0.1.0",
    description="Standalone backend for the generative try-on subsystem and its package-generation flow.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=BACKEND_ROOT), name="media")

app.include_router(health_router)
app.include_router(assets_router)
app.include_router(generative_tryon_router)
