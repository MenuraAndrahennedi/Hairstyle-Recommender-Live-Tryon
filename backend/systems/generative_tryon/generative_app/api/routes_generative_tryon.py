from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from generative_app.bootstrap_static import ensure_static_2d_on_path
from generative_app.config import media_url_for_path
from generative_app.core.generative_inpaint_runner import run_generative_inpaint
from generative_app.core.generative_tryon_pipeline import prepare_generative_tryon_package
from generative_app.core.uploads import save_uploaded_image
from generative_app.models.schemas import GenerativeTryOnPackageResponse


ensure_static_2d_on_path()

from support_app.core.asset_bank import get_asset_by_id  # noqa: E402
from support_app.core.face_analyzer import analyze_face_image  # noqa: E402
from support_app.core.hair_segmentation import predict_hair_mask_image, save_predicted_hair_mask  # noqa: E402


router = APIRouter(prefix="/api/generative-tryon", tags=["generative-tryon"])
logger = logging.getLogger(__name__)


@router.post("/package", response_model=GenerativeTryOnPackageResponse)
async def generate_generative_tryon_package(
    image: UploadFile = File(...),
    asset_id: str = Form(...),
) -> GenerativeTryOnPackageResponse:
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
            logger.exception("Hair segmentation failed during generative package preparation: %s", exc)
            subject_hair_mask = None

    package = prepare_generative_tryon_package(
        saved["saved_path"],
        analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
    )

    final_image_path = None
    final_metadata_path = None
    final_generation_completed = False
    final_generation_error = None

    if package is not None and package.get("manifest_path"):
        try:
            final_generation = run_generative_inpaint(
                package["manifest_path"],
                extra_prompt=(
                    "Match the reference hairstyle shape, volume, color, and texture. "
                    "Keep the person clothed and edit only the hair."
                ),
                use_ip_adapter=True,
                ip_adapter_scale=0.85,
                strength=0.92,
                guidance_scale=7.0,
                num_inference_steps=35,
            )
            final_image_path = final_generation.get("output_image_path")
            final_metadata_path = final_generation.get("metadata_path")
            final_generation_completed = True
            
        except Exception as exc:  # pragma: no cover - route fallback
            logger.exception("Final generative inpaint failed: %s", exc)
            final_generation_error = str(exc)

    return GenerativeTryOnPackageResponse(
        success=package is not None,
        message=(
            (
                "Generative try-on package prepared and final image generated successfully."
                if final_generation_completed
                else (
                    "Generative try-on package prepared successfully, but final generation is unavailable."
                    if package is not None
                    else "Face was not detected, so the generative try-on package was not generated."
                )
            )
        ),
        asset_id=asset_id,
        input_image_path=saved["saved_path"],
        input_image_url=media_url_for_path(saved["saved_path"]),
        segmentation_mask_path=str(segmentation_mask_path) if segmentation_mask_path is not None else None,
        segmentation_mask_url=media_url_for_path(segmentation_mask_path) if segmentation_mask_path is not None else None,
        erased_subject_path=package["erased_subject_path"] if package is not None else None,
        erased_subject_url=media_url_for_path(package["erased_subject_path"]) if package is not None else None,
        inpaint_mask_path=package["inpaint_mask_path"] if package is not None else None,
        inpaint_mask_url=media_url_for_path(package["inpaint_mask_path"]) if package is not None else None,
        reference_image_path=package["reference_image_path"] if package is not None else None,
        reference_image_url=media_url_for_path(package["reference_image_path"]) if package is not None else None,
        reference_mask_path=package["reference_mask_path"] if package is not None else None,
        reference_mask_url=media_url_for_path(package["reference_mask_path"]) if package is not None else None,
        preview_path=package["preview_path"] if package is not None else None,
        preview_url=media_url_for_path(package["preview_path"]) if package is not None else None,
        manifest_path=package["manifest_path"] if package is not None else None,
        manifest_url=media_url_for_path(package["manifest_path"]) if package is not None else None,
        prompt_path=package["prompt_path"] if package is not None else None,
        prompt_url=media_url_for_path(package["prompt_path"]) if package is not None else None,
        final_image_path=str(final_image_path) if final_image_path is not None else None,
        final_image_url=media_url_for_path(final_image_path) if final_image_path is not None else None,
        final_metadata_path=str(final_metadata_path) if final_metadata_path is not None else None,
        final_metadata_url=media_url_for_path(final_metadata_path) if final_metadata_path is not None else None,
        final_generation_completed=final_generation_completed,
        final_generation_error=final_generation_error,
        face_detected=analysis.face_detected,
        face_bbox=analysis.face_bbox,
    )
