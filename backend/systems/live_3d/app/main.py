from __future__ import annotations

from fastapi import APIRouter

from app.factory import build_subsystem_app


app = build_subsystem_app(
    title="Hairstyle Recommender Live 3D Try-On API",
    version="0.1.0",
    description="Reserved backend slot for the future live 3D try-on subsystem.",
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "system": "live_3d"}


@router.get("/info")
def info() -> dict[str, str]:
    return {
        "system": "live_3d",
        "status": "empty",
        "message": "Live 3D try-on is intentionally kept empty for now.",
    }


app.include_router(router)
