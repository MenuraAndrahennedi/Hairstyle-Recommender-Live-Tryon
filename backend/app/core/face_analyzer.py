from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import mediapipe as mp
import numpy as np
from PIL import Image

from app.config import LANDMARKS_DIR, MEDIAPIPE_FACE_LANDMARKER_PATH
from app.core.face_geometry import compute_face_geometry
from app.models.schemas import FaceAnalysisResult, FaceLandmark


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _compute_face_bbox(landmarks: list[FaceLandmark], image_width: int, image_height: int) -> dict[str, int] | None:
    if not landmarks:
        return None

    xs = [max(0, min(image_width - 1, int(point.x * image_width))) for point in landmarks]
    ys = [max(0, min(image_height - 1, int(point.y * image_height))) for point in landmarks]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def _save_landmarks_json(image_path: Path, landmarks: list[FaceLandmark], image_width: int, image_height: int) -> str:
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{image_path.stem}_{uuid4().hex[:8]}_landmarks.json"
    destination = LANDMARKS_DIR / filename
    payload = {
        "source_image": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "landmark_count": len(landmarks),
        "landmarks": [point.model_dump() for point in landmarks],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(destination)


def _missing_model_result() -> FaceAnalysisResult:
    return FaceAnalysisResult(
        face_detected=False,
        message=(
            "MediaPipe face landmarker model is missing. "
            f"Place the model file at: {MEDIAPIPE_FACE_LANDMARKER_PATH}"
        ),
        image_width=0,
        image_height=0,
        landmark_count=0,
        landmarks_path=None,
        face_bbox=None,
        landmarks=[],
        geometry=None,
        face_attributes=None,
    )


def _detect_face_from_mp_image(
    mp_image: mp.Image,
    *,
    source_image_path: Path | None,
    image_width: int,
    image_height: int,
) -> FaceAnalysisResult:
    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_FACE_LANDMARKER_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            results = landmarker.detect(mp_image)
    except Exception:
        return FaceAnalysisResult(
            face_detected=False,
            message="Face analysis failed while running the MediaPipe detector.",
            image_width=image_width,
            image_height=image_height,
            landmark_count=0,
            landmarks_path=None,
            face_bbox=None,
            landmarks=[],
            geometry=None,
            face_attributes=None,
        )

    if not results.face_landmarks:
        return FaceAnalysisResult(
            face_detected=False,
            message="No face detected in the current image.",
            image_width=image_width,
            image_height=image_height,
            landmark_count=0,
            landmarks_path=None,
            face_bbox=None,
            landmarks=[],
            geometry=None,
            face_attributes=None,
        )

    first_face = results.face_landmarks[0]
    landmarks = [
        FaceLandmark(index=index, x=point.x, y=point.y, z=point.z)
        for index, point in enumerate(first_face)
    ]
    landmarks_path = (
        _save_landmarks_json(source_image_path, landmarks, image_width, image_height)
        if source_image_path is not None
        else None
    )
    face_bbox = _compute_face_bbox(landmarks, image_width, image_height)
    geometry, face_attributes = compute_face_geometry(landmarks, image_width, image_height)

    return FaceAnalysisResult(
        face_detected=True,
        message="Face landmarks detected successfully.",
        image_width=image_width,
        image_height=image_height,
        landmark_count=len(landmarks),
        landmarks_path=landmarks_path,
        face_bbox=face_bbox,
        landmarks=landmarks,
        geometry=geometry,
        face_attributes=face_attributes,
    )


def analyze_face_image(image_path: str | Path) -> FaceAnalysisResult:
    image_path = Path(image_path)

    if not MEDIAPIPE_FACE_LANDMARKER_PATH.exists():
        return _missing_model_result()

    try:
        pil_image = Image.open(image_path)
        image_width, image_height = pil_image.size
    except Exception:
        return FaceAnalysisResult(
            face_detected=False,
            message="Image could not be read for face analysis.",
            image_width=0,
            image_height=0,
            landmark_count=0,
            landmarks_path=None,
            face_bbox=None,
            landmarks=[],
            geometry=None,
            face_attributes=None,
        )

    mp_image = mp.Image.create_from_file(str(image_path))
    return _detect_face_from_mp_image(
        mp_image,
        source_image_path=image_path,
        image_width=image_width,
        image_height=image_height,
    )


def analyze_face_pil_image(image: Image.Image) -> FaceAnalysisResult:
    if not MEDIAPIPE_FACE_LANDMARKER_PATH.exists():
        return _missing_model_result()

    rgb_image = image.convert("RGB")
    image_width, image_height = rgb_image.size
    image_array = np.asarray(rgb_image, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    return _detect_face_from_mp_image(
        mp_image,
        source_image_path=None,
        image_width=image_width,
        image_height=image_height,
    )
