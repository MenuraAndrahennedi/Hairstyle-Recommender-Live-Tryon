from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from manual_app.api.routes_assets import router as assets_router
from manual_app.api.routes_face import router as face_router
from manual_app.api.routes_health import router as health_router
from manual_app.api.routes_prediction import router as prediction_router
from manual_app.api.routes_recommend import router as recommend_router
from manual_app.config import BACKEND_ROOT


app = FastAPI(
    title="Hairstyle Recommender Static Manual Try-On API",
    version="0.1.0",
    description=(
        "Backend for static manual try-on. Keeps upload, segmentation, prediction, "
        "and recommendation steps separate from manual asset placement."
    ),
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
app.include_router(face_router)
app.include_router(prediction_router)
app.include_router(recommend_router)

