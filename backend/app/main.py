from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.factory import ALLOWED_ORIGINS
from systems.generative_tryon.generative_app.main import app as generative_tryon_app
from systems.live_2d.app.main import app as live_2d_app
from systems.live_3d.app.main import app as live_3d_app
from systems.static_auto_tryon.auto_app.main import app as static_auto_tryon_app


app = FastAPI(
    title="Hairstyle Recommender Multi-System Backend",
    version="0.1.0",
    description=(
        "Gateway backend that separates the project into static auto try-on, "
        "generative try-on, live 2D try-on, and live 3D try-on subsystems."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "system": "gateway"}


@app.get("/api/systems")
def systems_catalog() -> dict[str, list[dict[str, str]]]:
    return {
        "systems": [
            {
                "id": "static-auto",
                "name": "Static Auto Try-On",
                "mount_path": "/api/static-auto",
                "status": "active",
            },
            {
                "id": "generative",
                "name": "Generative Try-On",
                "mount_path": "/api/generative",
                "status": "experimental",
            },
            {
                "id": "live-2d",
                "name": "Live 2D Try-On",
                "mount_path": "/api/live-2d",
                "status": "empty",
            },
            {
                "id": "live-3d",
                "name": "Live 3D Try-On",
                "mount_path": "/api/live-3d",
                "status": "empty",
            },
        ]
    }


app.mount("/api/static-auto", static_auto_tryon_app)
app.mount("/api/generative", generative_tryon_app)
app.mount("/api/live-2d", live_2d_app)
app.mount("/api/live-3d", live_3d_app)
