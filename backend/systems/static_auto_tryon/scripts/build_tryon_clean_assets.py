from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = PROJECT_ROOT / "backend" / "data" / "processed" / "celeba_full_hair_assets"
METADATA_DIR = ASSET_ROOT / "metadata"
OUTPUT_IMAGE_DIR = ASSET_ROOT / "tryon_clean" / "images"
OUTPUT_MASK_DIR = ASSET_ROOT / "tryon_clean" / "masks"

TARGET_ASSET_IDS = [
    "celeba_full_hair_000064",
    "celeba_full_hair_000068",
    "celeba_full_hair_000084",
]

CLEANUP_PROFILES: dict[str, dict[str, float]] = {
    "celeba_full_hair_000068": {
        "trim_left_below_ratio": 0.58,
        "trim_left_before_ratio": 0.18,
        "trim_right_below_ratio": 0.66,
        "trim_right_after_ratio": 0.84,
    },
    "celeba_full_hair_000084": {
        "trim_left_below_ratio": 0.50,
        "trim_left_before_ratio": 0.18,
        "trim_right_below_ratio": 0.72,
        "trim_right_after_ratio": 0.90,
    },
}


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _hair_mask_bbox(mask_image: Image.Image) -> dict[str, int] | None:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    if not mask_array.any():
        return None

    y_coords, x_coords = np.where(mask_array)
    min_x = int(x_coords.min())
    max_x = int(x_coords.max())
    min_y = int(y_coords.min())
    max_y = int(y_coords.max())
    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x + 1,
        "height": max_y - min_y + 1,
    }


def _crop_asset_to_mask_bounds(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    pad_ratio: float = 0.06,
) -> tuple[Image.Image, Image.Image]:
    bbox = _hair_mask_bbox(asset_mask)
    if bbox is None:
        return asset_image, asset_mask

    pad_x = max(int(bbox["width"] * pad_ratio), 2)
    pad_y = max(int(bbox["height"] * pad_ratio), 2)
    left = max(bbox["x"] - pad_x, 0)
    top = max(bbox["y"] - pad_y, 0)
    right = min(bbox["x"] + bbox["width"] + pad_x, asset_image.width)
    bottom = min(bbox["y"] + bbox["height"] + pad_y, asset_image.height)

    if right <= left or bottom <= top:
        return asset_image, asset_mask

    crop_box = (left, top, right, bottom)
    return asset_image.crop(crop_box), asset_mask.crop(crop_box)


def _apply_asset_mask(asset_image: Image.Image, asset_mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    softened_mask = asset_mask.convert("L").filter(ImageFilter.MinFilter(size=3))
    alpha = softened_mask.point(lambda value: 255 if value >= 24 else 0, mode="L")
    masked_asset = asset_image.copy()
    masked_asset.putalpha(alpha)
    return masked_asset, alpha


def _cleanup_asset(
    asset_id: str,
    asset_image: Image.Image,
    asset_mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    profile = CLEANUP_PROFILES.get(asset_id)
    if not profile:
        return asset_image, asset_mask

    bbox = _hair_mask_bbox(asset_mask)
    if bbox is None:
        return asset_image, asset_mask

    rgba = np.asarray(asset_image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.asarray(asset_mask.convert("L"), dtype=np.uint8).copy()

    left_below = profile.get("trim_left_below_ratio")
    left_before = profile.get("trim_left_before_ratio")
    if left_below is not None and left_before is not None:
        y_threshold = bbox["y"] + int(bbox["height"] * left_below)
        x_threshold = bbox["x"] + int(bbox["width"] * left_before)
        alpha[y_threshold:, :x_threshold] = 0
        rgba[y_threshold:, :x_threshold, 3] = 0

    right_below = profile.get("trim_right_below_ratio")
    right_after = profile.get("trim_right_after_ratio")
    if right_below is not None and right_after is not None:
        y_threshold = bbox["y"] + int(bbox["height"] * right_below)
        x_threshold = bbox["x"] + int(bbox["width"] * right_after)
        alpha[y_threshold:, x_threshold:] = 0
        rgba[y_threshold:, x_threshold:, 3] = 0

    cleaned_mask = Image.fromarray(alpha, mode="L")
    cleaned_image = Image.fromarray(rgba, mode="RGBA")
    cleaned_image.putalpha(cleaned_mask)
    return cleaned_image, cleaned_mask


def main() -> None:
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    for asset_id in TARGET_ASSET_IDS:
        metadata_path = METADATA_DIR / f"{asset_id}.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        image_path = _resolve_path(str(payload["image_path"]))
        mask_path = _resolve_path(str(payload["mask_path"]))

        hair_asset = Image.open(image_path).convert("RGBA")
        hair_asset_mask = Image.open(mask_path).convert("L")
        hair_asset, hair_asset_mask = _crop_asset_to_mask_bounds(hair_asset, hair_asset_mask)
        hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
        hair_asset, hair_asset_mask = _cleanup_asset(asset_id, hair_asset, hair_asset_mask)

        output_image_path = OUTPUT_IMAGE_DIR / image_path.name
        output_mask_path = OUTPUT_MASK_DIR / mask_path.name
        hair_asset.save(output_image_path)
        hair_asset_mask.save(output_mask_path)
        print(f"saved {asset_id} -> {output_image_path}")


if __name__ == "__main__":
    main()
