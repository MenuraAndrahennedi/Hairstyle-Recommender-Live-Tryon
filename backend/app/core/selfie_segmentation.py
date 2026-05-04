from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.image_segmenter import (
    ImageSegmenter,
    ImageSegmenterOptions,
)

from app.core.hair_segmentation import (
    clean_predicted_hair_mask,
    predict_hair_mask_image_from_face_roi,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MEDIAPIPE_SEGMENTER_MODEL_PATH = (
    BACKEND_ROOT / "models" / "media_pipe_deeplab_v3.tflite"
)
FOREGROUND_CLASS_INDEX = 15


def selfie_segmentation_available() -> bool:
    return MEDIAPIPE_SEGMENTER_MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _get_selfie_segmenter() -> ImageSegmenter:
    if not selfie_segmentation_available():
        raise RuntimeError(
            "MediaPipe ImageSegmenter model asset is missing at "
            f"{MEDIAPIPE_SEGMENTER_MODEL_PATH}."
        )
    options = ImageSegmenterOptions(
        base_options=BaseOptions(
            model_asset_path=str(MEDIAPIPE_SEGMENTER_MODEL_PATH.resolve())
        ),
        running_mode=VisionTaskRunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=True,
    )
    return ImageSegmenter.create_from_options(options)


def _binary_mask_to_image(mask_array: np.ndarray) -> Image.Image:
    return Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")


def _cleanup_selfie_mask(mask_image: Image.Image) -> Image.Image:
    cleaned = mask_image.convert("L")
    cleaned = cleaned.filter(ImageFilter.MaxFilter(size=3))
    cleaned = cleaned.filter(ImageFilter.MinFilter(size=3))
    cleaned = cleaned.point(lambda value: 255 if value >= 128 else 0, mode="L")
    return clean_predicted_hair_mask(cleaned)


def predict_selfie_foreground_mask(
    image: Image.Image,
    threshold: float = 0.35,
) -> Image.Image:
    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    results = _get_selfie_segmenter().segment(mp_image)

    if results.confidence_masks:
        class_index = min(FOREGROUND_CLASS_INDEX, len(results.confidence_masks) - 1)
        confidence_view = results.confidence_masks[class_index].numpy_view()
        if confidence_view.ndim == 3:
            confidence_view = confidence_view[:, :, 0]
        mask_array = confidence_view >= threshold
        return _cleanup_selfie_mask(_binary_mask_to_image(mask_array))

    if results.category_mask is None:
        return Image.new("L", rgb_image.size, color=0)

    category_view = results.category_mask.numpy_view()
    if category_view.ndim == 3:
        category_view = category_view[:, :, 0]
    mask_array = category_view != 0
    return _cleanup_selfie_mask(_binary_mask_to_image(mask_array))


def predict_selfie_head_region_mask(
    image: Image.Image,
    face_bbox: dict[str, int] | None,
    threshold: float = 0.35,
) -> Image.Image:
    foreground_mask = predict_selfie_foreground_mask(image, threshold=threshold)
    if face_bbox is None:
        return foreground_mask

    width, height = foreground_mask.size
    face_x = int(face_bbox.get("x", 0))
    face_y = int(face_bbox.get("y", 0))
    face_width = max(int(face_bbox.get("width", 0)), 1)
    face_height = max(int(face_bbox.get("height", 0)), 1)

    x0 = max(face_x - int(face_width * 0.45), 0)
    x1 = min(face_x + face_width + int(face_width * 0.45), width)
    y0 = max(face_y - int(face_height * 1.05), 0)
    y1 = min(face_y + int(face_height * 0.42), height)

    mask_array = np.asarray(foreground_mask.convert("L")) > 0
    constrained = np.zeros_like(mask_array, dtype=bool)
    constrained[y0:y1, x0:x1] = mask_array[y0:y1, x0:x1]
    return _cleanup_selfie_mask(_binary_mask_to_image(constrained))


def predict_hybrid_live_hair_mask(
    image: Image.Image,
    face_bbox: dict[str, int] | None,
    *,
    selfie_threshold: float = 0.35,
) -> Image.Image:
    roi_hair_mask = predict_hair_mask_image_from_face_roi(image, face_bbox)
    selfie_head_mask = predict_selfie_head_region_mask(
        image,
        face_bbox,
        threshold=selfie_threshold,
    )

    roi_array = np.asarray(roi_hair_mask.convert("L")) > 0
    selfie_array = np.asarray(selfie_head_mask.convert("L")) > 0

    # Keep the hair-specific ROI prediction, but only where it agrees with the
    # person/head silhouette. If ROI is empty, fall back to the selfie head mask.
    if roi_array.any():
        hybrid_array = roi_array & selfie_array
        if not hybrid_array.any():
            hybrid_array = roi_array
    else:
        hybrid_array = selfie_array

    return _cleanup_selfie_mask(_binary_mask_to_image(hybrid_array))
