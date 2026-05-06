from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from support_app.config import media_url_for_path
from support_app.core.asset_bank import get_asset_by_id
from support_app.core.face_analyzer import analyze_face_image
from support_app.core.hair_segmentation import save_predicted_hair_mask, predict_hair_mask_image
from support_app.core.tryon_2d_engine import render_static_tryon
from support_app.core.uploads import save_uploaded_image
from support_app.models.schemas import TryOnResponse


router = APIRouter(prefix="/api", tags=["tryon"])
logger = logging.getLogger(__name__)


@router.post("/tryon", response_model=TryOnResponse)
async def generate_tryon(
    image: UploadFile = File(...),
    asset_id: str = Form(...),
) -> TryOnResponse:
    asset = get_asset_by_id(asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset_id}' was not found in the processed asset bank.",
        )

    saved = await save_uploaded_image(image)
    analysis = analyze_face_image(saved["saved_path"])
    segmentation_mask_path = None
    subject_hair_mask = None
    used_segmentation = False
    if analysis.face_detected:
        try:
            with Image.open(Path(saved["saved_path"])) as source_image:
                subject_hair_mask = predict_hair_mask_image(source_image)
            segmentation_mask_path = save_predicted_hair_mask(subject_hair_mask, Path(saved["saved_path"]).stem)
            used_segmentation = True
        except Exception as exc:  # pragma: no cover - route fallback
            logger.exception("Hair segmentation failed during try-on generation: %s", exc)
            subject_hair_mask = None

    output_image_path = render_static_tryon(
        saved["saved_path"],
        analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
    )

    return TryOnResponse(
        success=output_image_path is not None,
        message=(
            (
                "Segmentation-guided 2D try-on image generated successfully."
                if used_segmentation
                else "2D try-on image generated successfully without segmentation guidance."
            )
            if output_image_path is not None
            else "Face was not detected, so try-on output was not generated."
        ),
        asset_id=asset_id,
        filename=saved["filename"],
        saved_input_path=saved["saved_path"],
        saved_input_url=media_url_for_path(saved["saved_path"]),
        output_image_path=output_image_path,
        output_image_url=media_url_for_path(output_image_path) if output_image_path is not None else None,
        segmentation_mask_path=str(segmentation_mask_path) if segmentation_mask_path is not None else None,
        segmentation_mask_url=media_url_for_path(segmentation_mask_path) if segmentation_mask_path is not None else None,
        segmentation_used=used_segmentation,
        face_detected=analysis.face_detected,
        face_bbox=analysis.face_bbox,
    )

