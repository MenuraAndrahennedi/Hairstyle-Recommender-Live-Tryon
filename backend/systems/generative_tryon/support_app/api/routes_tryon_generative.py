from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from support_app.config import media_url_for_path
from support_app.core.asset_bank import get_asset_by_id
from support_app.core.face_analyzer import analyze_face_image
from support_app.core.hair_segmentation import save_predicted_hair_mask, predict_hair_mask_image
from support_app.core.tryon_2d_generative_prototype import prepare_generative_tryon_prototype
from support_app.core.uploads import save_uploaded_image
from support_app.models.schemas import GenerativeTryOnPrototypeResponse


router = APIRouter(prefix="/api", tags=["tryon-generative"])
logger = logging.getLogger(__name__)


@router.post("/tryon-generative-prototype", response_model=GenerativeTryOnPrototypeResponse)
async def generate_generative_tryon_prototype(
    image: UploadFile = File(...),
    asset_id: str = Form(...),
) -> GenerativeTryOnPrototypeResponse:
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

    if analysis.face_detected:
        try:
            with Image.open(Path(saved["saved_path"])) as source_image:
                subject_hair_mask = predict_hair_mask_image(source_image)
            segmentation_mask_path = save_predicted_hair_mask(subject_hair_mask, Path(saved["saved_path"]).stem)
        except Exception as exc:  # pragma: no cover - route fallback
            logger.exception("Hair segmentation failed during generative prototype preparation: %s", exc)
            subject_hair_mask = None

    prototype = prepare_generative_tryon_prototype(
        saved["saved_path"],
        analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
    )

    return GenerativeTryOnPrototypeResponse(
        success=prototype is not None,
        message=(
            "Generative try-on prototype package prepared successfully."
            if prototype is not None
            else "Face was not detected, so the generative try-on prototype package was not generated."
        ),
        asset_id=asset_id,
        input_image_path=saved["saved_path"],
        input_image_url=media_url_for_path(saved["saved_path"]),
        erased_subject_path=prototype["erased_subject_path"] if prototype is not None else None,
        erased_subject_url=(
            media_url_for_path(prototype["erased_subject_path"])
            if prototype is not None and prototype["erased_subject_path"] is not None
            else None
        ),
        inpaint_mask_path=prototype["inpaint_mask_path"] if prototype is not None else None,
        inpaint_mask_url=(
            media_url_for_path(prototype["inpaint_mask_path"])
            if prototype is not None and prototype["inpaint_mask_path"] is not None
            else None
        ),
        reference_image_path=prototype["reference_image_path"] if prototype is not None else None,
        reference_image_url=(
            media_url_for_path(prototype["reference_image_path"])
            if prototype is not None and prototype["reference_image_path"] is not None
            else None
        ),
        reference_mask_path=prototype["reference_mask_path"] if prototype is not None else None,
        reference_mask_url=(
            media_url_for_path(prototype["reference_mask_path"])
            if prototype is not None and prototype["reference_mask_path"] is not None
            else None
        ),
        preview_path=prototype["preview_path"] if prototype is not None else None,
        preview_url=(
            media_url_for_path(prototype["preview_path"])
            if prototype is not None and prototype["preview_path"] is not None
            else None
        ),
        manifest_path=prototype["manifest_path"] if prototype is not None else None,
        manifest_url=(
            media_url_for_path(prototype["manifest_path"])
            if prototype is not None and prototype["manifest_path"] is not None
            else None
        ),
        face_detected=analysis.face_detected,
        face_bbox=analysis.face_bbox,
    )

