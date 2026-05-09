from __future__ import annotations

import base64
import json
import threading

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from app.factory import build_subsystem_app
from .config import FACE_LANDMARKER_PATH, WEBCAM_RUNNER_PATH
from .core.demo_engine import Live2DDemoEngine, ManualTuning, runtime_readiness

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
    target_gender: str | None = Form(default=None),
) -> dict[str, object]:
    engine = get_engine()
    engine.set_target_gender(target_gender)
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
        if selected_asset_id and engine.current_recommendations:
            engine.set_selected_asset_by_id(selected_asset_id)

        rendered = engine.process_frame(frame.copy())

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

@router.websocket("/ws")
async def live_2d_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    engine = get_engine()
    selected_asset_id: str | None = None
    target_gender: str | None = None

    try:
        while True:
            message = await websocket.receive()

            # Receive control messages from frontend
            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                next_asset_id = control.get("selected_asset_id")
                if next_asset_id:
                    selected_asset_id = next_asset_id

                next_target_gender = control.get("target_gender")
                if next_target_gender is not None:
                    target_gender = next_target_gender

                action = control.get("action")

                if action == "move_left":
                    engine.tuning.x_offset -= 5

                elif action == "move_right":
                    engine.tuning.x_offset += 5

                elif action == "move_up":
                    engine.tuning.y_offset -= 5

                elif action == "move_down":
                    engine.tuning.y_offset += 5

                elif action == "scale_up":
                    engine.tuning.scale += 0.05

                elif action == "scale_down":
                    engine.tuning.scale = max(
                        0.3,
                        engine.tuning.scale - 0.05
                    )

                elif action == "rotate_left":
                    engine.tuning.rotation -= 3

                elif action == "rotate_right":
                    engine.tuning.rotation += 3

                elif action == "reset":
                    engine.tuning = ManualTuning()
                    engine.reset_smoothing()

                elif action == "cycle":
                    if engine.current_recommendations:
                        engine.selected_index = (
                            engine.selected_index + 1
                        ) % len(engine.current_recommendations)

                        engine.reset_smoothing()

                continue

            # Receive camera frame as binary JPEG
            if "bytes" not in message or message["bytes"] is None:
                continue

            image_bytes = message["bytes"]

            frame_array = engine.np.frombuffer(image_bytes, dtype=engine.np.uint8)
            frame = engine.cv2.imdecode(frame_array, engine.cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Live frame could not be decoded.",
                        }
                    )
                )
                continue

            with _ENGINE_LOCK:
                engine.show_hud = False
                engine.set_target_gender(target_gender)

                if selected_asset_id and engine.current_recommendations:
                    engine.set_selected_asset_by_id(selected_asset_id)

                rendered = engine.process_frame(frame.copy())
                
                ok, encoded = engine.cv2.imencode(
                    ".jpg",
                    rendered,
                    [int(engine.cv2.IMWRITE_JPEG_QUALITY), 75],
                )

                if not ok:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "Backend live frame encoding failed.",
                            }
                        )
                    )
                    continue

                payload = engine.current_response_payload()

            # Send metadata as text
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "metadata",
                        "success": True,
                        **payload,
                    }
                )
            )

            # Send rendered frame as binary JPEG
            await websocket.send_bytes(encoded.tobytes())

    except WebSocketDisconnect:
        print("Live 2D websocket disconnected")

        
app.include_router(router)
