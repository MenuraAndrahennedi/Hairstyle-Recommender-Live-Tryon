from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from PIL import Image

from support_app.config import media_url_for_path
from support_app.core.hair_segmentation import (
    SEGMENTATION_CHECKPOINT_PATH,
    SEGMENTATION_THRESHOLD,
    predict_hair_mask_image,
    save_predicted_hair_mask,
)
from support_app.core.uploads import save_uploaded_image
from support_app.models.schemas import HairSegmentationPredictionResponse


router = APIRouter(prefix="/api/predict", tags=["prediction"])


@router.post("/hair-mask", response_model=HairSegmentationPredictionResponse)
async def predict_hair_mask_route(image: UploadFile = File(...)) -> HairSegmentationPredictionResponse:
    saved = await save_uploaded_image(image)

    input_path = Path(saved["saved_path"])
    with Image.open(input_path) as source_image:
        mask_image = predict_hair_mask_image(source_image)

    output_path = save_predicted_hair_mask(mask_image, input_path.stem)

    return HairSegmentationPredictionResponse(
        success=True,
        message="Hair mask predicted successfully.",
        filename=saved["filename"],
        saved_input_path=saved["saved_path"],
        saved_input_url=media_url_for_path(saved["saved_path"]),
        output_mask_path=str(output_path),
        output_mask_url=media_url_for_path(output_path),
        checkpoint_path=str(SEGMENTATION_CHECKPOINT_PATH),
        threshold=SEGMENTATION_THRESHOLD,
    )

