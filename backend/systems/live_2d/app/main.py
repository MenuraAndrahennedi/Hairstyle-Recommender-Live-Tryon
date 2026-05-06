from __future__ import annotations

from fastapi import APIRouter

from app.factory import build_subsystem_app
from .config import FACE_LANDMARKER_PATH, WEBCAM_RUNNER_PATH
from .core.demo_engine import runtime_readiness


app = build_subsystem_app(
    title="Hairstyle Recommender Live 2D Try-On API",
    version="0.1.0",
    description=(
        "Dedicated live 2D webcam subsystem that uses the project's own recommendation system, "
        "segmentation model, and tryon_clean hairstyle assets while keeping the original live "
        "overlay logic."
    ),
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "system": "live_2d"}


@router.get("/info")
def info() -> dict[str, object]:
    readiness = runtime_readiness()
    return {
        "system": "live_2d",
        "status": "project_ready" if readiness["ready"] else "incomplete",
        "engine": "mediapipe_plus_project_recommender_plus_tryon_clean_original_overlay",
        "runner_script": str(WEBCAM_RUNNER_PATH),
        "models": {
            "face_landmarker": str(FACE_LANDMARKER_PATH),
        },
        "asset_root": readiness["tryon_clean_asset_root"],
        "readiness": readiness,
        "message": (
            "Run the dedicated webcam runner to use the live camera system with project recommendations and tryon_clean assets using the original overlay logic."
        ),
    }


app.include_router(router)
