from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from app.config import PREDICTIONS_DIR, PROJECT_ROOT
from app.ml.inference import load_segmentation_checkpoint, predict_hair_mask
from app.ml.transforms import ResizeImage


SEGMENTATION_CHECKPOINT_PATH = (
    PROJECT_ROOT / "backend" / "checkpoints" / "hair_segmentation" / "v2_unet_product.pt"
)
SEGMENTATION_IMAGE_SIZE = (256, 256)
SEGMENTATION_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def get_segmentation_model() -> object:
    return load_segmentation_checkpoint(SEGMENTATION_CHECKPOINT_PATH, device="cpu")


def _largest_connected_component(mask_array: np.ndarray) -> np.ndarray:
    if not mask_array.any():
        return mask_array

    height, width = mask_array.shape
    visited = np.zeros_like(mask_array, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    best_score = -1.0
    image_center_x = width / 2.0

    for start_y, start_x in np.argwhere(mask_array):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask_array[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        area = len(pixels)
        centroid_y = sum(y for y, _ in pixels) / area
        centroid_x = sum(x for _, x in pixels) / area
        top_bias = 1.0 - (centroid_y / max(height, 1))
        center_bias = 1.0 - min(abs(centroid_x - image_center_x) / max(image_center_x, 1.0), 1.0)
        score = area + (top_bias * 500.0) + (center_bias * 120.0)

        if score > best_score:
            best_score = score
            best_pixels = pixels

    cleaned = np.zeros_like(mask_array, dtype=bool)
    for y, x in best_pixels:
        cleaned[y, x] = True
    return cleaned


def clean_predicted_hair_mask(mask_image: Image.Image) -> Image.Image:
    mask = mask_image.convert("L")
    mask_array = np.asarray(mask) >= 128
    mask_array = _largest_connected_component(mask_array)

    cleaned = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
    cleaned = cleaned.filter(ImageFilter.MaxFilter(size=3))
    cleaned = cleaned.filter(ImageFilter.MinFilter(size=3))
    cleaned = cleaned.point(lambda value: 255 if value >= 128 else 0, mode="L")
    return cleaned


def _predict_mask_on_rgb_image(rgb_image: Image.Image) -> Image.Image:
    original_size = rgb_image.size
    image_tensor = ResizeImage(SEGMENTATION_IMAGE_SIZE)(rgb_image)
    mask_tensor = predict_hair_mask(
        get_segmentation_model(),
        image_tensor,
        threshold=SEGMENTATION_THRESHOLD,
    )
    mask_array = (mask_tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(mask_array, mode="L").resize(original_size, Image.Resampling.NEAREST)


def _expanded_face_hair_roi(
    image_size: tuple[int, int],
    face_bbox: dict[str, int] | None,
) -> tuple[int, int, int, int] | None:
    if face_bbox is None:
        return None

    image_width, image_height = image_size
    face_x = int(face_bbox.get("x", 0))
    face_y = int(face_bbox.get("y", 0))
    face_width = max(int(face_bbox.get("width", 0)), 1)
    face_height = max(int(face_bbox.get("height", 0)), 1)

    expand_left = int(face_width * 0.38)
    expand_right = int(face_width * 0.38)
    expand_top = int(face_height * 0.95)
    expand_bottom = int(face_height * 0.18)

    x0 = max(face_x - expand_left, 0)
    y0 = max(face_y - expand_top, 0)
    x1 = min(face_x + face_width + expand_right, image_width)
    y1 = min(face_y + face_height + expand_bottom, image_height)

    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    return x0, y0, x1, y1


def predict_hair_mask_image(image: Image.Image) -> Image.Image:
    rgb_image = image.convert("RGB")
    raw_mask = _predict_mask_on_rgb_image(rgb_image)
    return clean_predicted_hair_mask(raw_mask)


def predict_hair_mask_image_from_face_roi(
    image: Image.Image,
    face_bbox: dict[str, int] | None,
) -> Image.Image:
    rgb_image = image.convert("RGB")
    roi = _expanded_face_hair_roi(rgb_image.size, face_bbox)
    if roi is None:
        return predict_hair_mask_image(rgb_image)

    x0, y0, x1, y1 = roi
    cropped_rgb = rgb_image.crop((x0, y0, x1, y1))
    cropped_raw_mask = _predict_mask_on_rgb_image(cropped_rgb)
    cropped_clean_mask = clean_predicted_hair_mask(cropped_raw_mask)

    full_mask = Image.new("L", rgb_image.size, color=0)
    full_mask.paste(cropped_clean_mask, (x0, y0))
    return clean_predicted_hair_mask(full_mask)


def save_predicted_hair_mask(mask_image: Image.Image, stem: str) -> Path:
    output_dir = PREDICTIONS_DIR / "hair_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_hair_mask.png"
    mask_image.save(output_path)
    return output_path
