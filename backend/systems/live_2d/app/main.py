from __future__ import annotations

import base64
import threading

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.factory import build_subsystem_app
from .config import FACE_LANDMARKER_PATH, WEBCAM_RUNNER_PATH
from .core.demo_engine import Live2DDemoEngine, runtime_readiness


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
_ENGINE: Live2DDemoEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Live2DDemoEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Live2DDemoEngine()
    return _ENGINE


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


@router.post("/frame")
async def process_live_frame(
    image: UploadFile = File(...),
    selected_asset_id: str | None = Form(default=None),
) -> dict[str, object]:
    engine = get_engine()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded live frame is empty.",
        )

    frame_array = engine.np.frombuffer(image_bytes, dtype=engine.np.uint8)
    frame = engine.cv2.imdecode(frame_array, engine.cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded live frame could not be decoded as an image.",
        )

    with _ENGINE_LOCK:
        rendered = engine.process_frame(frame.copy())
        if selected_asset_id:
            if engine.set_selected_asset_by_id(selected_asset_id):
                rendered = engine.process_frame(frame.copy())
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Selected live asset '{selected_asset_id}' is not available in the current recommendations.",
                )

        ok, encoded = engine.cv2.imencode(
            ".jpg",
            rendered,
            [int(engine.cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Backend live frame encoding failed.",
            )

        frame_data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        payload = engine.current_response_payload()

    return {
        "success": True,
        "frame_data_url": frame_data_url,
        **payload,
    }


app.include_router(router)
