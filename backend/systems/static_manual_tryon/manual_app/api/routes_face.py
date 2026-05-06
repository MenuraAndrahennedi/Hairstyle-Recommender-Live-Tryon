from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from manual_app.config import media_url_for_path
from manual_app.core.face_analyzer import analyze_face_image
from manual_app.core.uploads import save_uploaded_image
from manual_app.models.schemas import AnalyzeFaceResponse


router = APIRouter(prefix="/api", tags=["face"])


@router.post("/analyze-face", response_model=AnalyzeFaceResponse)
async def analyze_face(image: UploadFile = File(...)) -> AnalyzeFaceResponse:
    saved = await save_uploaded_image(image)
    analysis = analyze_face_image(saved["saved_path"])
    return AnalyzeFaceResponse(
        face_detected=analysis.face_detected,
        message=analysis.message,
        filename=saved["filename"],
        content_type=saved["content_type"],
        saved_path=saved["saved_path"],
        file_size_bytes=saved["file_size_bytes"],
        image_width=analysis.image_width,
        image_height=analysis.image_height,
        landmark_count=analysis.landmark_count,
        landmarks_path=analysis.landmarks_path,
        face_bbox=analysis.face_bbox,
        geometry=analysis.geometry,
        face_attributes=analysis.face_attributes,
    )

