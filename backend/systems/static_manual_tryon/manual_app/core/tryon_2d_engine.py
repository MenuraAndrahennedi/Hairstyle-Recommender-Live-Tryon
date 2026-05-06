from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from manual_app.config import PROJECT_ROOT, TRYON_DIR
from manual_app.models.schemas import AssetMetadata, FaceAnalysisResult, FaceLandmark


def _resolve_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


TRYON_CLEAN_ASSET_ROOT = PROJECT_ROOT / "backend" / "data" / "processed" / "celeba_full_hair_assets" / "tryon_clean"
TRYON_CLEAN_IMAGE_DIR = TRYON_CLEAN_ASSET_ROOT / "images"
TRYON_CLEAN_MASK_DIR = TRYON_CLEAN_ASSET_ROOT / "masks"


def _tryon_clean_asset_paths(asset: AssetMetadata) -> tuple[Path, Path]:
    image_name = Path(asset.image_path).name
    mask_name = Path(asset.mask_path).name
    return TRYON_CLEAN_IMAGE_DIR / image_name, TRYON_CLEAN_MASK_DIR / mask_name


def _resolve_tryon_asset_paths(
    asset: AssetMetadata,
    *,
    prefer_clean_variant: bool,
) -> tuple[Path, Path]:
    if prefer_clean_variant:
        clean_image_path, clean_mask_path = _tryon_clean_asset_paths(asset)
        if clean_image_path.exists() and clean_mask_path.exists():
            return clean_image_path, clean_mask_path

    return _resolve_project_path(asset.image_path), _resolve_project_path(asset.mask_path)


def _average_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    return x, y


def _pixel_point(landmark: FaceLandmark, image_width: int, image_height: int) -> tuple[float, float]:
    return landmark.x * image_width, landmark.y * image_height


def _landmark_by_index(landmarks: list[FaceLandmark], index: int) -> FaceLandmark:
    return landmarks[index]


def _hair_mask_bbox(mask_image: Image.Image | None) -> dict[str, int] | None:
    if mask_image is None:
        return None
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


def _hair_mask_top_band_bbox(
    mask_image: Image.Image | None,
    band_ratio: float = 0.38,
) -> dict[str, int] | None:
    bbox = _hair_mask_bbox(mask_image)
    if bbox is None:
        return None

    mask_array = np.asarray(mask_image.convert("L")) > 0
    min_y = bbox["y"]
    max_y = bbox["y"] + bbox["height"] - 1
    band_height = max(int(bbox["height"] * band_ratio), 1)
    band_max_y = min(min_y + band_height - 1, max_y)

    top_band = mask_array[min_y : band_max_y + 1, :]
    if not top_band.any():
        return bbox

    y_coords, x_coords = np.where(top_band)
    min_x = int(x_coords.min())
    max_x = int(x_coords.max())
    local_min_y = int(y_coords.min())
    local_max_y = int(y_coords.max())

    return {
        "x": min_x,
        "y": min_y + local_min_y,
        "width": max_x - min_x + 1,
        "height": local_max_y - local_min_y + 1,
    }


def subject_hair_layout_bboxes(mask_image: Image.Image | None) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    cleaned_mask = _largest_subject_hair_component(mask_image)
    return _hair_mask_bbox(cleaned_mask), _hair_mask_top_band_bbox(cleaned_mask)


def _largest_subject_hair_component(mask_image: Image.Image | None) -> Image.Image | None:
    if mask_image is None:
        return None

    mask_array = np.asarray(mask_image.convert("L")) > 0
    if not mask_array.any():
        return None

    height, width = mask_array.shape
    visited = np.zeros_like(mask_array, dtype=bool)
    components: list[dict[str, object]] = []

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

        ys = [pixel[0] for pixel in pixels]
        xs = [pixel[1] for pixel in pixels]
        area = len(pixels)
        centroid_y = sum(ys) / area
        centroid_x = sum(xs) / area
        components.append(
            {
                "pixels": pixels,
                "area": area,
                "min_y": min(ys),
                "max_y": max(ys),
                "min_x": min(xs),
                "max_x": max(xs),
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
            }
        )

    if not components:
        return None

    image_center_x = width / 2.0

    def component_score(component: dict[str, object]) -> float:
        area = float(component["area"])
        top_bias = 1.0 - (float(component["centroid_y"]) / max(height, 1))
        center_bias = 1.0 - min(abs(float(component["centroid_x"]) - image_center_x) / max(image_center_x, 1.0), 1.0)
        return area + (top_bias * 500.0) + (center_bias * 120.0)

    best_component = max(components, key=component_score)
    cleaned = np.zeros_like(mask_array, dtype=np.uint8)
    for y, x in best_component["pixels"]:
        cleaned[y, x] = 255
    return Image.fromarray(cleaned, mode="L")


def _asset_mask_profile(mask_image: Image.Image) -> dict[str, float]:
    bbox = _hair_mask_bbox(mask_image)
    if bbox is None:
        return {"aspect_ratio": 1.0, "coverage": 0.0, "bottom_ratio": 0.0}

    mask_array = np.asarray(mask_image.convert("L")) > 0
    canvas_height, canvas_width = mask_array.shape
    bottom = bbox["y"] + bbox["height"] - 1
    return {
        "aspect_ratio": bbox["height"] / max(bbox["width"], 1),
        "coverage": float(mask_array.mean()),
        "bottom_ratio": bottom / max(canvas_height - 1, 1),
    }


def _is_short_open_static_asset(asset: AssetMetadata | None) -> bool:
    if asset is None:
        return False

    attrs = asset.normalized_attributes
    return (
        attrs.bang == "none"
        and attrs.side_hair == "exposed"
        and attrs.length in {"short", "medium"}
        and attrs.style_family in {"other", "pompadour", "regent", "side_part", "curly_crop"}
    )


def _crop_asset_to_mask_bounds(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    padding_ratio: float = 0.08,
) -> tuple[Image.Image, Image.Image]:
    bbox = _hair_mask_bbox(asset_mask)
    if bbox is None:
        return asset_image, asset_mask

    pad_x = max(int(bbox["width"] * padding_ratio), 4)
    pad_y = max(int(bbox["height"] * padding_ratio), 4)

    left = max(bbox["x"] - pad_x, 0)
    top = max(bbox["y"] - pad_y, 0)
    right = min(bbox["x"] + bbox["width"] + pad_x, asset_image.width)
    bottom = min(bbox["y"] + bbox["height"] + pad_y, asset_image.height)

    if right <= left or bottom <= top:
        return asset_image, asset_mask

    crop_box = (left, top, right, bottom)
    return asset_image.crop(crop_box), asset_mask.crop(crop_box)


def _apply_asset_mask(
    asset_image: Image.Image,
    asset_mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    # Clamp the hairstyle RGB strictly to the reviewed mask to remove bright
    # extraction halos before any resize or rotation spreads those pixels.
    softened_mask = asset_mask.convert("L").filter(ImageFilter.MinFilter(size=3))
    alpha = softened_mask.point(lambda value: 255 if value >= 24 else 0, mode="L")
    masked_asset = asset_image.copy()
    masked_asset.putalpha(alpha)
    return masked_asset, alpha


STATIC_TRYON_ASSET_CLEANUP_PROFILES: dict[str, dict[str, float]] = {
    "celeba_full_hair_000068": {
        "trim_left_below_ratio": 0.72,
        "trim_left_before_ratio": 0.22,
    },
    "celeba_full_hair_000084": {
        "trim_left_below_ratio": 0.58,
        "trim_left_before_ratio": 0.22,
    },
}


STATIC_TRYON_ASSET_LAYOUT_PROFILES: dict[str, dict[str, float]] = {
    "celeba_full_hair_000064": {
        "scale_multiplier": 1.08,
        "offset_x": 0.0,
        "offset_y": -5.0,
    },
    "celeba_full_hair_000068": {
        "scale_multiplier": 1.17,
        "offset_x": -2.0,
        "offset_y": -10.0,
    },
    "celeba_full_hair_000084": {
        "scale_multiplier": 1.12,
        "offset_x": 0.0,
        "offset_y": -7.0,
    },
}


def _static_tryon_layout_profile(asset: AssetMetadata | None) -> dict[str, float]:
    if asset is None:
        return {}
    return STATIC_TRYON_ASSET_LAYOUT_PROFILES.get(asset.asset_id, {})


def _cleanup_static_tryon_asset(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    asset: AssetMetadata,
) -> tuple[Image.Image, Image.Image]:
    profile = STATIC_TRYON_ASSET_CLEANUP_PROFILES.get(asset.asset_id)
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


def _estimate_anchor_data(face_analysis: FaceAnalysisResult) -> dict[str, float]:
    image_width = face_analysis.image_width
    image_height = face_analysis.image_height
    landmarks = face_analysis.landmarks

    left_eye_indices = [33, 133, 159, 145]
    right_eye_indices = [362, 263, 386, 374]
    left_temple_indices = [127, 234]
    right_temple_indices = [356, 454]
    forehead_index = 10
    chin_index = 152

    left_eye_center = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in left_eye_indices]
    )
    right_eye_center = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in right_eye_indices]
    )
    left_temple_center = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in left_temple_indices]
    )
    right_temple_center = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in right_temple_indices]
    )
    forehead_point = _pixel_point(_landmark_by_index(landmarks, forehead_index), image_width, image_height)
    chin_point = _pixel_point(_landmark_by_index(landmarks, chin_index), image_width, image_height)

    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
    eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
    eye_distance = abs(right_eye_center[0] - left_eye_center[0])
    temple_span = abs(right_temple_center[0] - left_temple_center[0])
    roll_radians = math.atan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
    upper_face_height = max(chin_point[1] - forehead_point[1], 1.0)

    return {
        "eye_center_x": eye_center_x,
        "eye_center_y": eye_center_y,
        "eye_distance": eye_distance,
        "left_temple_x": left_temple_center[0],
        "right_temple_x": right_temple_center[0],
        "temple_center_x": (left_temple_center[0] + right_temple_center[0]) / 2,
        "temple_center_y": (left_temple_center[1] + right_temple_center[1]) / 2,
        "temple_span": temple_span,
        "forehead_y": forehead_point[1],
        "chin_y": chin_point[1],
        "upper_face_height": upper_face_height,
        "roll_degrees": math.degrees(roll_radians),
    }


def _resize_hair_asset(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    face_analysis: FaceAnalysisResult,
    subject_hair_bbox: dict[str, int] | None = None,
    subject_hair_top_bbox: dict[str, int] | None = None,
    asset: AssetMetadata | None = None,
) -> Image.Image:
    face_bbox = face_analysis.face_bbox or {"width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)
    face_width = max(face_bbox["width"], 1)
    face_height = max(face_bbox["height"], 1)
    subject_hair_width = max((subject_hair_bbox or {}).get("width", 0), 0)
    subject_hair_height = max((subject_hair_bbox or {}).get("height", 0), 0)
    subject_hair_top_width = max((subject_hair_top_bbox or {}).get("width", 0), 0)
    asset_profile = _asset_mask_profile(asset_mask)

    landmark_target_width = max(
        int(max(face_width * 1.48, anchor["eye_distance"] * 2.32, anchor["temple_span"] * 1.38)),
        1,
    )
    target_width = landmark_target_width
    live_landmark_only = subject_hair_top_width <= 0 and subject_hair_width <= 0
    if live_landmark_only:
        target_width = max(int(landmark_target_width * 1.12), 1)
    elif subject_hair_width > 0:
        # Static/uploaded images have a reliable full hair mask. Size to the full
        # hair mass first; the top band alone makes short assets look like bangs.
        short_lifted_style = _is_short_open_static_asset(asset)
        if short_lifted_style:
            target_width = max(
                int(max(landmark_target_width * 0.98, subject_hair_width * 1.16, subject_hair_top_width * 1.22)),
                1,
            )
        else:
            target_width = max(
                int(max(landmark_target_width * 1.02, subject_hair_width * 1.18)),
                1,
            )
    elif subject_hair_top_width > 0:
        target_width = max(int(max(landmark_target_width * 1.0, subject_hair_top_width * 1.55)), 1)
    scale = target_width / max(asset_image.width, 1)
    target_height = int(asset_image.height * scale)
    if live_landmark_only:
        target_height = max(
            target_height,
            int(anchor["upper_face_height"] * 1.28),
            int(face_height * 1.08),
        )
    elif subject_hair_height > 0 and not _is_short_open_static_asset(asset):
        target_height = max(target_height, int(subject_hair_height * 0.96))

    max_allowed_height = int(
        max(
            face_height * 1.42,
            anchor["upper_face_height"] * 1.42,
        )
    )
    if asset_profile["aspect_ratio"] > 1.45:
        max_allowed_height = int(max_allowed_height * 0.88)
    if asset_profile["bottom_ratio"] > 0.9:
        max_allowed_height = int(max_allowed_height * 0.9)

    # Short, lifted men's styles need room to lift above the existing hairline.
    # 1.06 was too tight and compressed the pompadour into the hair region.
    # 1.45 gives a natural upward extension without overflowing the face.
    if (
        asset is not None
        and subject_hair_height > 0
        and _is_short_open_static_asset(asset)
        and asset_profile["bottom_ratio"] <= 0.90
    ):
        max_allowed_height = min(
            max_allowed_height,
            int(subject_hair_height * 1.30),
        )

    if max_allowed_height > 0 and target_height > max_allowed_height:
        shrink_scale = max_allowed_height / max(target_height, 1)
        target_height = max_allowed_height
        target_width = max(int(target_width * shrink_scale), 1)

    return asset_image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _resize_asset_mask(mask_image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask_image.convert("L").resize(size, Image.Resampling.NEAREST)


def _rotate_hair_asset(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    face_analysis: FaceAnalysisResult,
) -> tuple[Image.Image, Image.Image]:
    anchor = _estimate_anchor_data(face_analysis)
    roll_degrees = max(min(anchor["roll_degrees"], 16.0), -16.0)
    if abs(roll_degrees) < 1.0:
        return asset_image, asset_mask

    rotated_image = asset_image.rotate(-roll_degrees, resample=Image.Resampling.BICUBIC, expand=True)
    rotated_mask = asset_mask.rotate(-roll_degrees, resample=Image.Resampling.NEAREST, expand=True)
    return rotated_image, rotated_mask


def _overlay_position(
    face_analysis: FaceAnalysisResult,
    hair_size: tuple[int, int],
    canvas_size: tuple[int, int],
    subject_hair_bbox: dict[str, int] | None = None,
    subject_hair_top_bbox: dict[str, int] | None = None,
    asset_mask_bbox: dict[str, int] | None = None,
    asset_top_bbox: dict[str, int] | None = None,
    asset: AssetMetadata | None = None,
) -> tuple[int, int]:
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)
    canvas_width, canvas_height = canvas_size
    hair_width, hair_height = hair_size

    short_lifted_style = _is_short_open_static_asset(asset)

    horizontal_bbox = subject_hair_bbox or subject_hair_top_bbox
    vertical_bbox = subject_hair_bbox or subject_hair_top_bbox
    x = int(anchor["temple_center_x"] - (hair_width / 2))
    y = int(anchor["forehead_y"] - hair_height * (0.34 if vertical_bbox is None else 0.20))

    if horizontal_bbox is not None:
        subject_center_x = horizontal_bbox["x"] + (horizontal_bbox["width"] / 2)
        if asset_mask_bbox is not None:
            asset_center_x = asset_mask_bbox["x"] + (asset_mask_bbox["width"] / 2)
            x = int(subject_center_x - asset_center_x)
        else:
            bbox_x = subject_center_x - (hair_width / 2)
            x = int((x * 0.56) + (bbox_x * 0.44))

    if vertical_bbox is not None:
        if asset_mask_bbox is not None:
            asset_bottom_y = asset_mask_bbox["y"] + asset_mask_bbox["height"]
            reference_bottom_y = vertical_bbox["y"] + vertical_bbox["height"]

            # Short open men's assets should sit on the upper head, not on the
            # full predicted hair bbox, whose sides can extend down near the ears.
            if short_lifted_style and subject_hair_bbox is not None:
                target_bottom_y = min(
                    reference_bottom_y,
                    int(face_bbox["y"] + face_bbox["height"] * 0.32),
                )
            else:
                target_bottom_y = reference_bottom_y
            y = int(target_bottom_y - asset_bottom_y)
        else:
            bbox_y = vertical_bbox["y"] - hair_height * 0.12
            y = int((y * 0.24) + (bbox_y * 0.76))

    fallback_x = int(face_bbox["x"] + (face_bbox["width"] / 2) - (hair_width / 2))
    fallback_y = int(face_bbox["y"] - hair_height * 0.34)
    if hair_width <= 0 or hair_height <= 0:
        x, y = fallback_x, fallback_y

    x = max(0, min(x, max(canvas_width - hair_width, 0)))
    y = max(0, min(y, max(canvas_height - hair_height, 0)))
    return x, y


def _build_face_protection_mask(
    canvas_size: tuple[int, int],
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
) -> Image.Image:
    canvas_width, canvas_height = canvas_size
    protection = Image.new("L", (canvas_width, canvas_height), 0)
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}

    left = max(int(face_bbox["x"] - face_bbox["width"] * 0.08), 0)
    right = min(int(face_bbox["x"] + face_bbox["width"] * 1.08), canvas_width)

    if asset.normalized_attributes.bang == "none":
        top = max(int(face_bbox["y"] + face_bbox["height"] * 0.20), 0)
    elif asset.normalized_attributes.bang in {"side", "see_through"}:
        top = max(int(face_bbox["y"] + face_bbox["height"] * 0.28), 0)
    else:
        top = max(int(face_bbox["y"] + face_bbox["height"] * 0.38), 0)

    bottom = min(int(face_bbox["y"] + face_bbox["height"] * 1.02), canvas_height)

    if right <= left or bottom <= top:
        return protection

    draw = ImageDraw.Draw(protection)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(int(face_bbox["width"] * 0.18), 12),
        fill=255,
    )
    return protection.filter(ImageFilter.GaussianBlur(radius=8))


def _build_subject_hair_region_mask(
    canvas_size: tuple[int, int],
    subject_hair_mask: Image.Image | None,
) -> Image.Image | None:
    cleaned_mask = _largest_subject_hair_component(subject_hair_mask)
    if cleaned_mask is None:
        return None
    mask = cleaned_mask.convert("L").resize(canvas_size, Image.Resampling.NEAREST)
    if not np.any(np.asarray(mask) > 0):
        return None
    return mask


def _estimate_background_rgb(
    input_image: Image.Image,
    subject_hair_mask: Image.Image | None,
) -> tuple[int, int, int]:
    rgb = input_image.convert("RGB")
    rgb_array = np.asarray(rgb, dtype=np.uint8)
    height, width = rgb_array.shape[:2]

    samples: list[np.ndarray] = []

    corner = max(min(width, height) // 8, 16)
    samples.extend(
        [
            rgb_array[:corner, :corner].reshape(-1, 3),
            rgb_array[:corner, width - corner :].reshape(-1, 3),
            # Bottom corners excluded â€” portrait photos typically have clothing
            # there, which biases the fill estimate toward dark clothing colors.
        ]
    )

    if subject_hair_mask is not None:
        mask = np.asarray(subject_hair_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)) > 0
        if mask.any():
            non_hair = ~mask
            top_band = np.zeros_like(non_hair, dtype=bool)
            top_band[: max(height // 3, 1), :] = True
            candidate = non_hair & top_band
            if candidate.any():
                samples.append(rgb_array[candidate].reshape(-1, 3))

    merged = np.concatenate([sample for sample in samples if sample.size > 0], axis=0)
    median = np.median(merged, axis=0)
    return tuple(int(x) for x in median.tolist())


def _build_static_support_mask(
    canvas_size: tuple[int, int],
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None,
) -> Image.Image | None:
    region = _build_subject_hair_region_mask(canvas_size, subject_hair_mask)
    if region is None:
        return None

    region_array = np.asarray(region, dtype=np.uint8)
    support = Image.fromarray(region_array, mode="L")

    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    subject_bbox = _hair_mask_bbox(region)
    if subject_bbox is None:
        return support

    expand_up = int(face_bbox["height"] * 0.12)
    expand_side = int(face_bbox["width"] * 0.06)

    if asset.normalized_attributes.style_family in {"pompadour", "regent", "side_part", "curly_crop"}:
        expand_up = int(face_bbox["height"] * 0.74)
        expand_side = int(face_bbox["width"] * 0.24)

    overlay = Image.new("L", canvas_size, 0)
    draw = ImageDraw.Draw(overlay)
    left = max(subject_bbox["x"] - expand_side, 0)
    top = max(subject_bbox["y"] - expand_up, 0)
    right = min(subject_bbox["x"] + subject_bbox["width"] + expand_side, canvas_size[0])
    bottom = min(subject_bbox["y"] + subject_bbox["height"], canvas_size[1])
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(int(face_bbox["width"] * 0.10), 10),
        fill=255,
    )

    support = ImageChops.lighter(
        support.filter(ImageFilter.MaxFilter(size=19)),
        overlay.filter(ImageFilter.GaussianBlur(radius=6)),
    )
    return support


def _paste_resized_mask(
    source_mask: Image.Image,
    canvas_size: tuple[int, int],
    paste_x: int,
    paste_y: int,
) -> Image.Image:
    canvas = Image.new("L", canvas_size, 0)
    canvas.paste(source_mask, (paste_x, paste_y))
    return canvas


def _compose_tryon(
    input_image: Image.Image,
    resized_hair: Image.Image,
    resized_asset_mask: Image.Image,
    paste_x: int,
    paste_y: int,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
) -> Image.Image:
    canvas_size = input_image.size
    asset_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    asset_layer.paste(resized_hair, (paste_x, paste_y))
    original_asset_alpha = np.asarray(asset_layer.getchannel("A"), dtype=np.float32) / 255.0

    base = input_image.copy()
    final_asset_alpha = original_asset_alpha
    final_alpha = Image.fromarray((255.0 * final_asset_alpha).astype(np.uint8), mode="L")
    asset_layer.putalpha(final_alpha)

    composed = base.copy()
    composed.alpha_composite(asset_layer)
    return composed


def _compose_headfit_tryon(
    input_image: Image.Image,
    resized_hair: Image.Image,
    resized_asset_mask: Image.Image,
    paste_x: int,
    paste_y: int,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None,
) -> Image.Image:
    canvas_size = input_image.size
    asset_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    asset_layer.paste(resized_hair, (paste_x, paste_y))
    asset_alpha = np.asarray(asset_layer.getchannel("A"), dtype=np.float32) / 255.0

    subject_region = _build_subject_hair_region_mask(canvas_size, subject_hair_mask)
    if subject_region is None:
        asset_layer.putalpha(Image.fromarray((asset_alpha * 255.0).astype(np.uint8), mode="L"))
        composed = input_image.copy()
        composed.alpha_composite(asset_layer)
        return composed

    background_rgb = _estimate_background_rgb(input_image, subject_region)
    base = input_image.convert("RGBA").copy()
    fill_layer = Image.new("RGBA", canvas_size, (*background_rgb, 255))

    subject_alpha = np.asarray(subject_region, dtype=np.float32) / 255.0
    # Static replacement should fully remove the subject's original hair.
    # The real fix is donor placement; keeping old hair behind the donor just
    # creates a double-style result.
    erase_mask = Image.fromarray((subject_alpha * 255.0).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.6)
    )
    erased = Image.composite(fill_layer, base, erase_mask)

    # Preserve the full donor silhouette. We only soften the matte slightly to
    # avoid hard pasted edges, not to crop the hairstyle to the subject mask.
    alpha_blur = 0.9 if _is_short_open_static_asset(asset) else 0.12
    final_alpha = Image.fromarray(
        np.clip(asset_alpha * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=alpha_blur))
    asset_layer.putalpha(final_alpha)

    composed = erased.copy()
    composed.alpha_composite(asset_layer)
    return composed


def _headfit_scale_asset(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    asset: AssetMetadata,
    subject_hair_bbox: dict[str, int] | None,
    subject_hair_top_bbox: dict[str, int] | None,
    subject_hair_mask: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image]:
    asset_bbox = _hair_mask_bbox(asset_mask)
    asset_top_bbox = _hair_mask_top_band_bbox(asset_mask)
    cap_bbox = _short_style_cap_bbox(asset_mask) if _is_short_open_static_asset(asset) else None
    asset_open = _mask_opening_geometry(asset_mask)
    subject_open = _mask_opening_geometry(subject_hair_mask) if subject_hair_mask is not None else None

    if asset_bbox is None:
        return asset_image, asset_mask

    short_open_style = _is_short_open_static_asset(asset)
    weighted_scales: list[tuple[float, float]] = []

    if subject_open is not None and asset_open is not None:
        asset_open_width = max(asset_open["right_inner_x"] - asset_open["left_inner_x"], 1.0)
        subject_open_width = max(subject_open["right_inner_x"] - subject_open["left_inner_x"], 1.0)
        weighted_scales.append((subject_open_width / asset_open_width, 0.54 if short_open_style else 0.44))

    if short_open_style and subject_hair_top_bbox is not None and cap_bbox is not None:
        weighted_scales.append(((subject_hair_top_bbox["width"] * 1.02) / max(cap_bbox["width"], 1), 0.42))
    else:
        if subject_hair_bbox is not None:
            weighted_scales.append(((subject_hair_bbox["width"] * 1.02) / max(asset_bbox["width"], 1), 0.28))
        if subject_hair_top_bbox is not None and asset_top_bbox is not None:
            weighted_scales.append(((subject_hair_top_bbox["width"] * 1.00) / max(asset_top_bbox["width"], 1), 0.28))
        if subject_hair_top_bbox is not None and cap_bbox is not None:
            weighted_scales.append(((subject_hair_top_bbox["width"] * 1.00) / max(cap_bbox["width"], 1), 0.24))

        if not weighted_scales:
            return asset_image, asset_mask

    if not weighted_scales:
        return asset_image, asset_mask

    total_weight = sum(weight for _, weight in weighted_scales)
    scale = sum(scale_value * weight for scale_value, weight in weighted_scales) / max(total_weight, 1e-6)

    if subject_hair_bbox is not None:
        max_width_scale = (subject_hair_bbox["width"] * (1.06 if short_open_style else 1.02)) / max(asset_bbox["width"], 1)
        scale = min(scale, max_width_scale)

    scale = max(scale, 0.35)
    target_size = (
        max(int(round(asset_image.width * scale)), 1),
        max(int(round(asset_image.height * scale)), 1),
    )
    return (
        asset_image.resize(target_size, Image.Resampling.LANCZOS),
        asset_mask.resize(target_size, Image.Resampling.NEAREST),
    )


def _headfit_position(
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    asset_mask: Image.Image,
    canvas_size: tuple[int, int],
    subject_hair_bbox: dict[str, int] | None,
    subject_hair_top_bbox: dict[str, int] | None,
    subject_hair_mask: Image.Image | None,
) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)
    asset_bbox = _hair_mask_bbox(asset_mask)
    asset_top_bbox = _hair_mask_top_band_bbox(asset_mask)

    if asset_bbox is None:
        return (
            max(0, min(int(anchor["temple_center_x"] - asset_mask.width / 2), max(canvas_width - asset_mask.width, 0))),
            max(0, min(int(anchor["forehead_y"] - asset_mask.height * 0.30), max(canvas_height - asset_mask.height, 0))),
        )

    subject_open = _mask_opening_geometry(subject_hair_mask) if subject_hair_mask is not None else None
    asset_open = _mask_opening_geometry(asset_mask)
    short_open_style = _is_short_open_static_asset(asset)

    if subject_open is not None and asset_open is not None:
        subject_open_center_x = (subject_open["left_inner_x"] + subject_open["right_inner_x"]) / 2.0
        asset_open_center_x = (asset_open["left_inner_x"] + asset_open["right_inner_x"]) / 2.0
        x_open = subject_open_center_x - asset_open_center_x
        x_anchor = anchor["temple_center_x"] - asset_open_center_x
        x = int(round((x_open * 0.72) + (x_anchor * 0.28)))

        subject_anchor_y = subject_open["y"]
        asset_anchor_y = asset_open["y"]
        x -= int(round((asset_open["gap"] - subject_open["gap"]) * 0.04))
        y_open = float(subject_anchor_y - asset_anchor_y)
        forehead_reference_y = float(anchor["forehead_y"] + face_bbox["height"] * (0.06 if short_open_style else 0.04))
        y_landmark = forehead_reference_y - asset_anchor_y

        y = int(round((y_open * 0.78) + (y_landmark * 0.22)))
        if subject_hair_bbox is not None:
            y_top = float(subject_hair_bbox["y"] - asset_bbox["y"])
            y = int(round((y * 0.80) + (y_top * 0.20)))

        if subject_hair_top_bbox is not None and asset_top_bbox is not None:
            subject_crown_y = subject_hair_top_bbox["y"]
            asset_crown_y = asset_top_bbox["y"] + y
            crown_delta = subject_crown_y - asset_crown_y
            y += int(round(crown_delta * (0.16 if short_open_style else 0.18)))
    else:
        if subject_hair_top_bbox is not None and asset_top_bbox is not None:
            subject_center_x = subject_hair_top_bbox["x"] + subject_hair_top_bbox["width"] / 2
            asset_center_x = asset_top_bbox["x"] + asset_top_bbox["width"] / 2
            x = int(round(subject_center_x - asset_center_x))
            y = int(round(subject_hair_top_bbox["y"] - asset_top_bbox["y"]))
        elif subject_hair_bbox is not None:
            subject_center_x = subject_hair_bbox["x"] + subject_hair_bbox["width"] / 2
            asset_center_x = asset_bbox["x"] + asset_bbox["width"] / 2
            x = int(round(subject_center_x - asset_center_x))
            y = int(round(subject_hair_bbox["y"] - asset_bbox["y"]))
        else:
            x = int(round(anchor["temple_center_x"] - asset_mask.width / 2))
            y = int(round(anchor["forehead_y"] - asset_bbox["y"] - face_bbox["height"] * 0.08))

    # Small upward bias keeps the style seated on the head rather than sliding
    # down into the forehead after matching the internal opening.
    if short_open_style:
        y -= max(int(face_bbox["height"] * 0.042), 7)
    else:
        y += max(int(face_bbox["height"] * 0.012), 2)

    x = max(0, min(x, max(canvas_width - asset_mask.width, 0)))
    y = max(0, min(y, max(canvas_height - asset_mask.height, 0)))
    return x, y


def _locally_fit_hairline_band(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    paste_x: int,
    paste_y: int,
    subject_hair_mask: Image.Image | None,
    asset: AssetMetadata,
) -> tuple[Image.Image, Image.Image]:
    if subject_hair_mask is None or not _is_short_open_static_asset(asset):
        return asset_image, asset_mask

    subject_open = _mask_opening_geometry(subject_hair_mask)
    asset_open = _mask_opening_geometry(asset_mask)
    asset_bbox = _hair_mask_bbox(asset_mask)
    if subject_open is None or asset_open is None or asset_bbox is None:
        return asset_image, asset_mask

    local_target_left = float(subject_open["left_outer_x"] - paste_x)
    local_target_right = float(subject_open["right_outer_x"] - paste_x)
    local_inner_left = float(subject_open["left_inner_x"] - paste_x)
    local_inner_right = float(subject_open["right_inner_x"] - paste_x)

    rgba = np.asarray(asset_image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.asarray(asset_mask.convert("L"), dtype=np.uint8).copy()
    h, w = alpha.shape

    if local_target_right <= 0 or local_target_left >= w:
        return asset_image, asset_mask

    band_top = max(int(asset_open["y"] - asset_bbox["height"] * 0.10), 0)
    band_mid = min(int(asset_open["y"] + asset_bbox["height"] * 0.02), h - 1)
    band_bottom = min(int(asset_open["y"] + asset_bbox["height"] * 0.08), h - 1)
    if band_bottom <= band_top:
        return asset_image, asset_mask

    fitted_rgba = rgba.copy()
    fitted_alpha = alpha.copy()

    for y in range(band_top, band_bottom + 1):
        xs = np.where(alpha[y] > 0)[0]
        if xs.size < 2:
            continue

        segments: list[tuple[int, int]] = []
        start_x = int(xs[0])
        prev_x = int(xs[0])
        for raw_x in xs[1:]:
            x = int(raw_x)
            if x == prev_x + 1:
                prev_x = x
                continue
            segments.append((start_x, prev_x))
            start_x = x
            prev_x = x
        segments.append((start_x, prev_x))
        if len(segments) < 2:
            continue

        left_seg = segments[0]
        right_seg = segments[-1]
        if left_seg[1] - left_seg[0] < 1 or right_seg[1] - right_seg[0] < 1:
            continue

        if y <= band_mid:
            t = (y - band_top) / max(band_mid - band_top, 1)
            desired_left_outer = (1.0 - t) * left_seg[0] + t * local_target_left
            desired_left_inner = (1.0 - t) * left_seg[1] + t * local_inner_left
            desired_right_inner = (1.0 - t) * right_seg[0] + t * local_inner_right
            desired_right_outer = (1.0 - t) * right_seg[1] + t * local_target_right
            blend = 0.55
        else:
            t = (y - band_mid) / max(band_bottom - band_mid, 1)
            desired_left_outer = (1.0 - t) * local_target_left + t * (local_target_left + 1.0)
            desired_left_inner = (1.0 - t) * local_inner_left + t * local_inner_left
            desired_right_inner = (1.0 - t) * local_inner_right + t * local_inner_right
            desired_right_outer = (1.0 - t) * local_target_right + t * (local_target_right - 1.0)
            blend = 0.80

        left_dst_left = max(int(round(desired_left_outer)), 0)
        left_dst_right = min(int(round(desired_left_inner)), w - 1)
        right_dst_left = max(int(round(desired_right_inner)), 0)
        right_dst_right = min(int(round(desired_right_outer)), w - 1)
        if left_dst_right - left_dst_left < 1 or right_dst_right - right_dst_left < 1:
            continue

        left_row_rgba = rgba[y : y + 1, left_seg[0] : left_seg[1] + 1, :]
        left_row_alpha = alpha[y : y + 1, left_seg[0] : left_seg[1] + 1]
        right_row_rgba = rgba[y : y + 1, right_seg[0] : right_seg[1] + 1, :]
        right_row_alpha = alpha[y : y + 1, right_seg[0] : right_seg[1] + 1]

        left_warp_rgba = cv2.resize(
            left_row_rgba,
            (left_dst_right - left_dst_left + 1, 1),
            interpolation=cv2.INTER_LINEAR,
        )
        left_warp_alpha = cv2.resize(
            left_row_alpha,
            (left_dst_right - left_dst_left + 1, 1),
            interpolation=cv2.INTER_LINEAR,
        )
        right_warp_rgba = cv2.resize(
            right_row_rgba,
            (right_dst_right - right_dst_left + 1, 1),
            interpolation=cv2.INTER_LINEAR,
        )
        right_warp_alpha = cv2.resize(
            right_row_alpha,
            (right_dst_right - right_dst_left + 1, 1),
            interpolation=cv2.INTER_LINEAR,
        )

        new_row_rgba = fitted_rgba[y].copy()
        new_row_alpha = fitted_alpha[y].copy()

        clear_left = max(left_seg[0] - 1, 0)
        clear_right = min(right_seg[1] + 1, w - 1)
        clear_strength = blend if y <= band_mid else 1.0
        if clear_right >= clear_left:
            source_region = slice(clear_left, clear_right + 1)
            new_row_rgba[source_region, :3] = (
                new_row_rgba[source_region, :3].astype(np.float32) * (1.0 - clear_strength)
            ).astype(np.uint8)
            new_row_alpha[source_region] = (
                new_row_alpha[source_region].astype(np.float32) * (1.0 - clear_strength)
            ).astype(np.uint8)
            new_row_rgba[source_region, 3] = new_row_alpha[source_region]

        for dst_left, dst_right, incoming_rgba_arr, incoming_alpha_arr in (
            (left_dst_left, left_dst_right, left_warp_rgba[0], left_warp_alpha[0]),
            (right_dst_left, right_dst_right, right_warp_rgba[0], right_warp_alpha[0]),
        ):
            existing_rgba = new_row_rgba[dst_left : dst_right + 1].astype(np.float32)
            existing_alpha = new_row_alpha[dst_left : dst_right + 1].astype(np.float32) / 255.0
            incoming_rgba = incoming_rgba_arr.astype(np.float32)
            incoming_alpha = (incoming_alpha_arr.astype(np.float32) / 255.0) * blend

            out_alpha = incoming_alpha + existing_alpha * (1.0 - incoming_alpha)
            out_rgb = incoming_rgba[:, :3] * incoming_alpha[:, None] + existing_rgba[:, :3] * (
                1.0 - incoming_alpha[:, None]
            )

            result = existing_rgba.copy()
            result[:, :3] = out_rgb
            result[:, 3] = np.clip(out_alpha * 255.0, 0.0, 255.0)
            new_row_rgba[dst_left : dst_right + 1] = result.astype(np.uint8)
        new_row_alpha = new_row_rgba[:, 3]

        fitted_rgba[y] = new_row_rgba
        fitted_alpha[y] = new_row_alpha

    out_image = Image.fromarray(fitted_rgba, mode="RGBA")
    out_mask = Image.fromarray(fitted_alpha, mode="L")
    out_image.putalpha(out_mask)
    return out_image, out_mask


def render_headfit_static_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
    else:
        input_image = Image.open(Path(input_image_path)).convert("RGBA")

    asset_image_path, asset_mask_path = _resolve_tryon_asset_paths(
        asset,
        prefer_clean_variant=False,
    )
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _cleanup_static_tryon_asset(hair_asset, hair_asset_mask, asset)

    cleaned_subject_hair_mask = _largest_subject_hair_component(subject_hair_mask)
    if layout_hint is None:
        subject_hair_bbox = _hair_mask_bbox(cleaned_subject_hair_mask)
        subject_hair_top_bbox = _hair_mask_top_band_bbox(cleaned_subject_hair_mask)
    else:
        subject_hair_bbox = layout_hint.get("subject_hair_bbox")
        subject_hair_top_bbox = layout_hint.get("subject_hair_top_bbox")

    warped_short_layer = None
    if _is_short_open_static_asset(asset):
        warped_short_layer = _warp_short_open_asset_layer(
            input_image.size,
            face_analysis,
            asset,
            hair_asset,
            hair_asset_mask,
            cleaned_subject_hair_mask,
        )
    if warped_short_layer is not None:
        warped_hair, warped_mask = warped_short_layer
        return _compose_headfit_tryon(
            input_image,
            warped_hair,
            warped_mask,
            0,
            0,
            face_analysis,
            asset,
            subject_hair_mask=cleaned_subject_hair_mask,
        )

    resized_hair, resized_mask = _headfit_scale_asset(
        hair_asset,
        hair_asset_mask,
        asset,
        subject_hair_bbox,
        subject_hair_top_bbox,
        cleaned_subject_hair_mask,
    )
    resized_hair, resized_mask = _rotate_hair_asset(resized_hair, resized_mask, face_analysis)

    paste_x, paste_y = _headfit_position(
        face_analysis,
        asset,
        resized_mask,
        input_image.size,
        subject_hair_bbox,
        subject_hair_top_bbox,
        cleaned_subject_hair_mask,
    )
    return _compose_headfit_tryon(
        input_image,
        resized_hair,
        resized_mask,
        paste_x,
        paste_y,
        face_analysis,
        asset,
        subject_hair_mask=cleaned_subject_hair_mask,
    )


def render_headfit_static_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> str | None:
    composed = render_headfit_static_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
        layout_hint=layout_hint,
    )
    if composed is None:
        return None

    if isinstance(input_image_path, Image.Image):
        input_stem = "headfit_frame"
    else:
        input_stem = Path(input_image_path).stem

    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_stem}_{asset.asset_id}_{uuid4().hex[:8]}.png"
    output_path = TRYON_DIR / output_name
    composed.convert("RGB").save(output_path)
    return str(output_path)


def render_experimental_static_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
    else:
        input_image = Image.open(Path(input_image_path)).convert("RGBA")

    asset_image_path = _resolve_project_path(asset.image_path)
    asset_mask_path = _resolve_project_path(asset.mask_path)
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
    cleaned_subject_hair_mask = _largest_subject_hair_component(subject_hair_mask)
    if layout_hint is None:
        subject_hair_bbox = _hair_mask_bbox(cleaned_subject_hair_mask)
        subject_hair_top_bbox = _hair_mask_top_band_bbox(cleaned_subject_hair_mask)
    else:
        subject_hair_bbox = layout_hint.get("subject_hair_bbox")
        subject_hair_top_bbox = layout_hint.get("subject_hair_top_bbox")

    resized_hair = _resize_hair_asset(
        hair_asset,
        hair_asset_mask,
        face_analysis,
        subject_hair_bbox=subject_hair_bbox,
        subject_hair_top_bbox=subject_hair_top_bbox,
        asset=asset,
    )
    resized_asset_mask = _resize_asset_mask(hair_asset_mask, resized_hair.size)
    resized_hair, resized_asset_mask = _rotate_hair_asset(
        resized_hair,
        resized_asset_mask,
        face_analysis,
    )
    resized_asset_mask_bbox = _hair_mask_bbox(resized_asset_mask)
    resized_asset_top_bbox = _hair_mask_top_band_bbox(resized_asset_mask)
    paste_x, paste_y = _overlay_position(
        face_analysis,
        resized_hair.size,
        input_image.size,
        subject_hair_bbox=subject_hair_bbox,
        subject_hair_top_bbox=subject_hair_top_bbox,
        asset_mask_bbox=resized_asset_mask_bbox,
        asset_top_bbox=resized_asset_top_bbox,
        asset=asset,
    )

    composed = _compose_tryon(
        input_image,
        resized_hair,
        resized_asset_mask,
        paste_x,
        paste_y,
        face_analysis,
        asset,
        subject_hair_mask=cleaned_subject_hair_mask,
    )
    return composed


def render_experimental_static_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> str | None:
    composed = render_experimental_static_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
        layout_hint=layout_hint,
    )
    if composed is None:
        return None

    if isinstance(input_image_path, Image.Image):
        input_stem = "live_frame"
    else:
        input_stem = Path(input_image_path).stem

    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_stem}_{asset.asset_id}_{uuid4().hex[:8]}.png"
    output_path = TRYON_DIR / output_name
    composed.convert("RGB").save(output_path)
    return str(output_path)


def _legacy_visible_bboxes(
    asset_mask: Image.Image,
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    return _hair_mask_bbox(asset_mask), _hair_mask_top_band_bbox(asset_mask)


def _short_style_cap_bbox(asset_mask: Image.Image) -> dict[str, int] | None:
    bbox = _hair_mask_bbox(asset_mask)
    if bbox is None:
        return None

    mask_array = np.asarray(asset_mask.convert("L")) > 0
    top = bbox["y"]
    bottom = bbox["y"] + bbox["height"] - 1
    cap_bottom = min(top + int(bbox["height"] * 0.62), bottom)

    row_widths: list[tuple[int, int]] = []
    for y in range(top, cap_bottom + 1):
        xs = np.where(mask_array[y])[0]
        if xs.size == 0:
            continue
        row_widths.append((y, int(xs.size)))

    if not row_widths:
        return bbox

    max_width = max(width for _, width in row_widths)
    keep_threshold = max(int(max_width * 0.72), 8)

    kept_pixels: list[tuple[int, int]] = []
    for y, width in row_widths:
        if width < keep_threshold:
            continue
        xs = np.where(mask_array[y])[0]
        kept_pixels.extend((y, int(x)) for x in xs)

    if not kept_pixels:
        return bbox

    ys = [pixel[0] for pixel in kept_pixels]
    xs = [pixel[1] for pixel in kept_pixels]
    return {
        "x": int(min(xs)),
        "y": int(min(ys)),
        "width": int(max(xs) - min(xs) + 1),
        "height": int(max(ys) - min(ys) + 1),
    }


def _donor_crown_bbox(
    asset_mask: Image.Image,
    asset: AssetMetadata | None = None,
) -> dict[str, int] | None:
    if asset is not None and _is_short_open_static_asset(asset):
        cap_bbox = _short_style_cap_bbox(asset_mask)
        if cap_bbox is not None:
            return cap_bbox

    crown_bbox = _hair_mask_top_band_bbox(asset_mask, band_ratio=0.34)
    if crown_bbox is not None:
        return crown_bbox
    return _hair_mask_bbox(asset_mask)


def _legacy_forehead_opening_anchor(
    asset_mask: Image.Image,
) -> tuple[float, float] | None:
    mask_array = np.asarray(asset_mask.convert("L")) > 0
    bbox = _hair_mask_bbox(asset_mask)
    if bbox is None:
        return None

    start_y = bbox["y"] + int(bbox["height"] * 0.35)
    end_y = bbox["y"] + int(bbox["height"] * 0.80)
    best_gap = -1.0
    best_anchor: tuple[float, float] | None = None

    for y in range(start_y, end_y):
        xs = np.where(mask_array[y])[0]
        if xs.size < 2:
            continue

        segments: list[tuple[int, int]] = []
        start_x = int(xs[0])
        prev_x = int(xs[0])
        for raw_x in xs[1:]:
            x = int(raw_x)
            if x == prev_x + 1:
                prev_x = x
                continue
            segments.append((start_x, prev_x))
            start_x = x
            prev_x = x
        segments.append((start_x, prev_x))

        if len(segments) < 2:
            continue

        left_segment = segments[0]
        right_segment = segments[-1]
        gap = float(right_segment[0] - left_segment[1] - 1)
        if gap < max(bbox["width"] * 0.08, 4.0):
            continue

        anchor_x = (left_segment[1] + right_segment[0]) / 2.0
        if gap > best_gap:
            best_gap = gap
            best_anchor = (anchor_x, float(y))

    return best_anchor


def _mask_opening_geometry(
    mask_image: Image.Image,
) -> dict[str, float] | None:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    bbox = _hair_mask_bbox(mask_image)
    if bbox is None:
        return None

    start_y = bbox["y"] + int(bbox["height"] * 0.35)
    end_y = bbox["y"] + int(bbox["height"] * 0.80)
    best: dict[str, float] | None = None
    best_gap = -1.0

    for y in range(start_y, end_y):
        xs = np.where(mask_array[y])[0]
        if xs.size < 2:
            continue

        segments: list[tuple[int, int]] = []
        start_x = int(xs[0])
        prev_x = int(xs[0])
        for raw_x in xs[1:]:
            x = int(raw_x)
            if x == prev_x + 1:
                prev_x = x
                continue
            segments.append((start_x, prev_x))
            start_x = x
            prev_x = x
        segments.append((start_x, prev_x))

        if len(segments) < 2:
            continue

        left_segment = segments[0]
        right_segment = segments[-1]
        gap = float(right_segment[0] - left_segment[1] - 1)
        if gap < max(bbox["width"] * 0.08, 4.0):
            continue

        if gap > best_gap:
            best_gap = gap
            best = {
                "y": float(y),
                "left_outer_x": float(left_segment[0]),
                "left_inner_x": float(left_segment[1]),
                "right_inner_x": float(right_segment[0]),
                "right_outer_x": float(right_segment[1]),
                "gap": gap,
            }
    return best


def _mask_hairline_opening_geometry(
    mask_image: Image.Image,
) -> dict[str, float] | None:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    bbox = _hair_mask_bbox(mask_image)
    if bbox is None:
        return None

    start_y = bbox["y"] + int(bbox["height"] * 0.12)
    end_y = bbox["y"] + int(bbox["height"] * 0.46)
    best: dict[str, float] | None = None

    for y in range(start_y, end_y):
        xs = np.where(mask_array[y])[0]
        if xs.size < 2:
            continue

        segments: list[tuple[int, int]] = []
        start_x = int(xs[0])
        prev_x = int(xs[0])
        for raw_x in xs[1:]:
            x = int(raw_x)
            if x == prev_x + 1:
                prev_x = x
                continue
            segments.append((start_x, prev_x))
            start_x = x
            prev_x = x
        segments.append((start_x, prev_x))

        if len(segments) < 2:
            continue

        left_segment = segments[0]
        right_segment = segments[-1]
        gap = float(right_segment[0] - left_segment[1] - 1)
        if gap < max(bbox["width"] * 0.14, 8.0):
            continue

        left_width = float(left_segment[1] - left_segment[0] + 1)
        right_width = float(right_segment[1] - right_segment[0] + 1)
        if left_width < 3.0 or right_width < 3.0:
            continue

        best = {
            "y": float(y),
            "left_outer_x": float(left_segment[0]),
            "left_inner_x": float(left_segment[1]),
            "right_inner_x": float(right_segment[0]),
            "right_outer_x": float(right_segment[1]),
            "gap": gap,
        }
        break

    return best


def _warp_short_open_asset_layer(
    canvas_size: tuple[int, int],
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    asset_image: Image.Image,
    asset_mask: Image.Image,
    subject_hair_mask: Image.Image | None,
) -> tuple[Image.Image, Image.Image] | None:
    if subject_hair_mask is None:
        return None

    subject_bbox = _hair_mask_bbox(subject_hair_mask)
    subject_top_bbox = _hair_mask_top_band_bbox(subject_hair_mask)
    subject_open = _mask_opening_geometry(subject_hair_mask)
    asset_bbox = _hair_mask_bbox(asset_mask)
    asset_cap_bbox = _short_style_cap_bbox(asset_mask)
    asset_open = _mask_opening_geometry(asset_mask)

    if (
        subject_bbox is None
        or subject_top_bbox is None
        or subject_open is None
        or asset_bbox is None
        or asset_cap_bbox is None
        or asset_open is None
    ):
        return None

    src_top_y = float(asset_cap_bbox["y"] + max(int(asset_cap_bbox["height"] * 0.28), 1))
    dst_top_y = float(subject_top_bbox["y"] + max(int(subject_top_bbox["height"] * 0.20), 1))

    src_quad = np.array(
        [
            [float(asset_cap_bbox["x"]), src_top_y],
            [float(asset_cap_bbox["x"] + asset_cap_bbox["width"] - 1), src_top_y],
            [float(asset_open["right_inner_x"]), float(asset_open["y"])],
            [float(asset_open["left_inner_x"]), float(asset_open["y"])],
        ],
        dtype=np.float32,
    )
    dst_quad = np.array(
        [
            [float(subject_top_bbox["x"]), dst_top_y],
            [float(subject_top_bbox["x"] + subject_top_bbox["width"] - 1), dst_top_y],
            [float(subject_open["right_inner_x"]), float(subject_open["y"])],
            [float(subject_open["left_inner_x"]), float(subject_open["y"])],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
    canvas_w, canvas_h = canvas_size

    asset_rgba = np.asarray(asset_image.convert("RGBA"), dtype=np.uint8)
    warped_rgba = cv2.warpPerspective(
        asset_rgba,
        transform,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    asset_alpha = np.asarray(asset_mask.convert("L"), dtype=np.uint8)
    warped_alpha = cv2.warpPerspective(
        asset_alpha,
        transform,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_rgba[..., 3] = warped_alpha
    layer = Image.fromarray(warped_rgba, mode="RGBA")
    mask = Image.fromarray(warped_alpha, mode="L")
    layer.putalpha(mask)
    return layer, mask


def _legacy_scale_asset(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_bbox: dict[str, int] | None,
    subject_hair_top_bbox: dict[str, int] | None,
) -> tuple[Image.Image, Image.Image]:
    asset_bbox, asset_top_bbox = _legacy_visible_bboxes(asset_mask)
    if asset_bbox is None:
        return asset_image, asset_mask

    subject_bbox = subject_hair_bbox
    subject_top = subject_hair_top_bbox or subject_hair_bbox
    anchor = _estimate_anchor_data(face_analysis)
    face_bbox = face_analysis.face_bbox or {"width": 1, "height": 1}

    scale_candidates: list[float] = []
    short_lifted_style = _is_short_open_static_asset(asset)
    crown_bbox = _donor_crown_bbox(asset_mask, asset)
    crown_anchor_style = crown_bbox is not None and not short_lifted_style
    asset_open = _mask_hairline_opening_geometry(asset_mask) if not short_lifted_style else None

    if asset_open is not None:
        opening_target_width = max(
            anchor["temple_span"] * 1.08,
            face_bbox["width"] * 0.82,
            anchor["eye_distance"] * 1.68,
        )
        scale_candidates.append(opening_target_width / max(asset_open["gap"], 1.0))

    if crown_bbox is not None:
        if short_lifted_style:
            crown_target_width = max(
                anchor["temple_span"] * 1.48,
                face_bbox["width"] * 0.92,
                anchor["eye_distance"] * 1.90,
            )
        else:
            crown_target_width = max(
                anchor["temple_span"] * 1.56,
                face_bbox["width"] * 1.02,
                anchor["eye_distance"] * 2.02,
            )
        scale_candidates.append(crown_target_width / max(crown_bbox["width"], 1))

    if subject_top is not None and asset_top_bbox is not None:
        scale_candidates.append((subject_top["width"] * (0.96 if short_lifted_style else 1.02)) / max(asset_top_bbox["width"], 1))
        if short_lifted_style:
            scale_candidates.append((subject_top["height"] * 0.96) / max(asset_top_bbox["height"], 1))
        elif crown_bbox is not None:
            scale_candidates.append((subject_top["height"] * 1.14) / max(crown_bbox["height"], 1))
    if subject_bbox is not None:
        if short_lifted_style:
            scale_candidates.append((subject_bbox["height"] * 0.96) / max(asset_bbox["height"], 1))
            scale_candidates.append((subject_bbox["width"] * 0.92) / max(asset_bbox["width"], 1))
        else:
            scale_candidates.append((subject_bbox["width"] * 0.86) / max(asset_bbox["width"], 1))

    if not scale_candidates:
        return asset_image, asset_mask

    if asset_open is not None:
        opening_weighted = scale_candidates[0]
        remaining = scale_candidates[1:]
        if remaining:
            remaining_mean = sum(remaining) / len(remaining)
            scale = (opening_weighted * 0.68) + (remaining_mean * 0.32)
        else:
            scale = opening_weighted
    elif crown_anchor_style:
        crown_weighted = scale_candidates[0]
        remaining = scale_candidates[1:]
        if remaining:
            remaining_mean = sum(remaining) / len(remaining)
            scale = (crown_weighted * 0.74) + (remaining_mean * 0.26)
        else:
            scale = crown_weighted
    elif crown_bbox is not None:
        crown_weighted = scale_candidates[0]
        remaining = scale_candidates[1:]
        if remaining:
            remaining_mean = sum(remaining) / len(remaining)
            scale = (crown_weighted * (0.72 if short_lifted_style else 0.64)) + (
                remaining_mean * (0.28 if short_lifted_style else 0.36)
            )
        else:
            scale = crown_weighted
    else:
        scale = sum(scale_candidates) / len(scale_candidates)

    if short_lifted_style:
        scale *= 1.04

    layout_profile = _static_tryon_layout_profile(asset)
    scale *= layout_profile.get("scale_multiplier", 1.0)
    scale = max(scale, 0.35)
    target_size = (
        max(int(round(asset_image.width * scale)), 1),
        max(int(round(asset_image.height * scale)), 1),
    )
    return (
        asset_image.resize(target_size, Image.Resampling.LANCZOS),
        asset_mask.resize(target_size, Image.Resampling.NEAREST),
    )


def _legacy_paste_position(
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    asset_mask: Image.Image,
    canvas_size: tuple[int, int],
    subject_hair_bbox: dict[str, int] | None,
    subject_hair_top_bbox: dict[str, int] | None,
    subject_hair_mask: Image.Image | None = None,
) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)
    asset_bbox, asset_top_bbox = _legacy_visible_bboxes(asset_mask)
    crown_bbox = _donor_crown_bbox(asset_mask, asset)
    if asset_bbox is None:
        fallback_x = int(anchor["temple_center_x"] - (asset_mask.width / 2))
        fallback_y = int(anchor["forehead_y"] - (asset_mask.height * 0.30))
        return (
            max(0, min(fallback_x, max(canvas_width - asset_mask.width, 0))),
            max(0, min(fallback_y, max(canvas_height - asset_mask.height, 0))),
        )

    subject_bbox = subject_hair_bbox
    subject_top = subject_hair_top_bbox or subject_hair_bbox
    short_lifted_style = _is_short_open_static_asset(asset)

    forehead_anchor = _legacy_forehead_opening_anchor(asset_mask) if short_lifted_style else None
    subject_open = _mask_opening_geometry(subject_hair_mask) if short_lifted_style and subject_hair_mask is not None else None
    asset_open = _mask_opening_geometry(asset_mask) if short_lifted_style else _mask_hairline_opening_geometry(asset_mask)

    if subject_open is not None and asset_open is not None:
        subject_open_center_x = (subject_open["left_inner_x"] + subject_open["right_inner_x"]) / 2.0
        asset_open_center_x = (asset_open["left_inner_x"] + asset_open["right_inner_x"]) / 2.0
        x_open = subject_open_center_x - asset_open_center_x
        if crown_bbox is not None:
            x_crown = anchor["temple_center_x"] - (crown_bbox["x"] + (crown_bbox["width"] / 2))
        else:
            x_crown = anchor["temple_center_x"] - asset_open_center_x
        x = int(round((x_open * 0.72) + (x_crown * 0.28)))
        x -= int(round((asset_open["gap"] - subject_open["gap"]) * 0.04))
    elif asset_open is not None and not short_lifted_style:
        asset_open_center_x = (asset_open["left_inner_x"] + asset_open["right_inner_x"]) / 2.0
        x_open = anchor["temple_center_x"] - asset_open_center_x
        if crown_bbox is not None:
            x_crown = anchor["temple_center_x"] - (crown_bbox["x"] + (crown_bbox["width"] / 2))
            x = int(round((x_open * 0.78) + (x_crown * 0.22)))
        else:
            x = int(round(x_open))
    elif crown_bbox is not None:
        x = int(round(anchor["temple_center_x"] - (crown_bbox["x"] + (crown_bbox["width"] / 2))))
    elif forehead_anchor is not None:
        subject_center_x = face_bbox["x"] + (face_bbox["width"] / 2)
        x = int(round(subject_center_x - forehead_anchor[0]))
    elif subject_top is not None and asset_top_bbox is not None:
        subject_center_x = subject_top["x"] + (subject_top["width"] / 2)
        asset_center_x = asset_top_bbox["x"] + (asset_top_bbox["width"] / 2)
        x = int(round(subject_center_x - asset_center_x))
    elif subject_bbox is not None:
        subject_center_x = subject_bbox["x"] + (subject_bbox["width"] / 2)
        asset_center_x = asset_bbox["x"] + (asset_bbox["width"] / 2)
        x = int(round(subject_center_x - asset_center_x))
    else:
        x = int(round(anchor["temple_center_x"] - (asset_mask.width / 2)))

    if short_lifted_style and subject_open is not None and asset_open is not None:
        y_open = subject_open["y"] - asset_open["y"]
        if crown_bbox is not None:
            forehead_target_y = anchor["forehead_y"] - (face_bbox["height"] * 0.08)
            y_top = forehead_target_y - crown_bbox["y"]
        elif subject_bbox is not None:
            y_top = subject_bbox["y"] - asset_bbox["y"]
        else:
            y_top = anchor["forehead_y"] - asset_open["y"]
        y = int(round((y_open * 0.74) + (y_top * 0.26)))
        if crown_bbox is not None:
            current_asset_top = crown_bbox["y"] + y
            desired_crown_top = anchor["forehead_y"] - (face_bbox["height"] * 0.08)
            crown_delta = desired_crown_top - current_asset_top
            y += int(round(crown_delta * 0.18))
        y -= max(int(face_bbox["height"] * 0.028), 4)
    elif asset_open is not None and not short_lifted_style:
        forehead_offset = 0.03 if asset.normalized_attributes.bang in {"full", "side", "see_through"} else 0.06
        target_open_y = anchor["forehead_y"] + (face_bbox["height"] * forehead_offset)
        y_open = target_open_y - asset_open["y"]
        if crown_bbox is not None and subject_top is not None:
            crown_target_y = min(
                float(subject_top["y"]),
                anchor["forehead_y"] - (face_bbox["height"] * 0.12),
            )
            y_crown = crown_target_y - crown_bbox["y"]
            y = int(round((y_open * 0.72) + (y_crown * 0.28)))
        else:
            y = int(round(y_open))
    elif crown_bbox is not None and not short_lifted_style and subject_top is not None:
        crown_target_y = min(
            float(subject_top["y"]),
            anchor["forehead_y"] - (face_bbox["height"] * 0.12),
        )
        y = int(round(crown_target_y - crown_bbox["y"]))
    elif crown_bbox is not None:
        desired_crown_top = anchor["forehead_y"] - (
            face_bbox["height"] * (0.11 if short_lifted_style else 0.07)
        )
        y = int(round(desired_crown_top - crown_bbox["y"]))
    elif short_lifted_style and subject_bbox is not None and asset_bbox is not None:
        # Fallback for lifted styles when no stable inner opening is detected.
        target_bottom_y = subject_bbox["y"] + subject_bbox["height"]
        y = int(round(target_bottom_y - (asset_bbox["y"] + asset_bbox["height"])))
    elif subject_top is not None and asset_top_bbox is not None:
        target_top_y = subject_top["y"]
        if asset.normalized_attributes.bang in {"full", "side", "see_through"}:
            target_top_y += int(face_bbox["height"] * 0.05)
        y = int(round(target_top_y - asset_top_bbox["y"]))
    elif subject_bbox is not None:
        target_top_y = subject_bbox["y"]
        y = int(round(target_top_y - asset_bbox["y"]))
    else:
        y = int(round(anchor["forehead_y"] - asset_bbox["y"] - (face_bbox["height"] * 0.08)))
    x = max(0, min(x, max(canvas_width - asset_mask.width, 0)))
    y = max(0, min(y, max(canvas_height - asset_mask.height, 0)))
    layout_profile = _static_tryon_layout_profile(asset)
    x += int(round(layout_profile.get("offset_x", 0.0)))
    y += int(round(layout_profile.get("offset_y", 0.0)))
    x = max(0, min(x, max(canvas_width - asset_mask.width, 0)))
    y = max(0, min(y, max(canvas_height - asset_mask.height, 0)))
    return x, y


def _legacy_compose_tryon(
    input_image: Image.Image,
    asset_image: Image.Image,
    asset_mask: Image.Image,
    paste_x: int,
    paste_y: int,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None,
) -> Image.Image:
    canvas_size = input_image.size
    asset_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    asset_layer.paste(asset_image, (paste_x, paste_y))

    asset_mask_canvas = _paste_resized_mask(asset_mask, canvas_size, paste_x, paste_y)
    asset_mask_array = np.asarray(asset_mask_canvas, dtype=np.float32) / 255.0

    if subject_hair_mask is not None:
        subject_region = _build_subject_hair_region_mask(canvas_size, subject_hair_mask)
    else:
        subject_region = None

    if subject_region is not None:
        subject_hair_array = np.asarray(subject_region, dtype=np.float32) / 255.0
        support_mask = _build_static_support_mask(canvas_size, face_analysis, asset, subject_region)
        short_open_style = _is_short_open_static_asset(asset)
        if support_mask is not None:
            support_array = np.asarray(support_mask, dtype=np.float32) / 255.0
            if short_open_style:
                # Do not crop short replacement styles back into the supported
                # subject-hair region. Once we commit to full replacement, the
                # donor alpha should define the visible silhouette.
                final_asset_alpha_array = asset_mask_array
            else:
                final_asset_alpha_array = asset_mask_array * support_array
        else:
            final_asset_alpha_array = asset_mask_array

        background_rgb = _estimate_background_rgb(input_image, subject_region)
        erased = input_image.convert("RGBA").copy()
        fill_layer = Image.new("RGBA", canvas_size, (*background_rgb, 255))

        # Conservative erase: keep the previous best scaling/alignment, but avoid
        # erasing the whole predicted subject-hair mask. Many CelebA hair assets
        # have forehead holes/openings; full erasing creates gray scalp/forehead
        # patches. Erase mainly where the new asset itself will be visible.
        visible_asset = np.clip(final_asset_alpha_array, 0.0, 1.0)
        soft_asset = Image.fromarray((visible_asset * 255.0).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=2.0)
        )
        soft_asset_array = np.asarray(soft_asset, dtype=np.float32) / 255.0
        if short_open_style and support_mask is not None:
            # For the static full-replacement path, fully clear the supported
            # subject-hair region so the donor hairstyle becomes the only hair
            # visible. We preserve silhouette via the donor alpha itself rather
            # than by leaving the old hair underneath.
            erase_seed = np.clip(
                np.maximum(visible_asset, subject_hair_array * np.clip(support_array * 1.02, 0.0, 1.0)),
                0.0,
                1.0,
            )
            erase_blur = 2.8
        else:
            erase_seed = np.clip(
                visible_asset + (subject_hair_array * soft_asset_array * 0.12),
                0.0,
                1.0,
            )
            erase_blur = 4
        erase_mask = Image.fromarray((255.0 * np.clip(erase_seed, 0.0, 1.0)).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=erase_blur)
        )
        erased = Image.composite(fill_layer, erased, erase_mask)
        base = erased
    else:
        base = input_image.convert("RGBA").copy()
        final_asset_alpha_array = asset_mask_array

    final_asset_alpha = Image.fromarray((final_asset_alpha_array * 255.0).astype(np.uint8), mode="L")
    asset_layer.putalpha(final_asset_alpha)
    composed = base.copy()
    composed.alpha_composite(asset_layer)
    return composed


def _render_subject_silhouette_texture_tryon(
    input_image: Image.Image,
    asset_image: Image.Image,
    asset_mask: Image.Image,
    face_analysis: FaceAnalysisResult,
    subject_hair_mask: Image.Image,
) -> Image.Image:
    canvas_size = input_image.size
    subject_region = _build_subject_hair_region_mask(canvas_size, subject_hair_mask)
    subject_bbox = _hair_mask_bbox(subject_region)
    asset_bbox = _hair_mask_bbox(asset_mask)
    if subject_region is None or subject_bbox is None or asset_bbox is None:
        return input_image

    background_rgb = _estimate_background_rgb(input_image, subject_region)
    base = input_image.convert("RGBA").copy()
    fill_layer = Image.new("RGBA", canvas_size, (*background_rgb, 255))
    erase_mask = subject_region
    erased = Image.composite(fill_layer, base, erase_mask)

    subject_array = np.asarray(subject_region.convert("L")) > 0
    donor_rgb = np.asarray(asset_image.convert("RGB"), dtype=np.uint8)
    donor_mask = np.asarray(asset_mask.convert("L")) > 0
    canvas_rgb = np.asarray(erased.convert("RGB"), dtype=np.uint8).copy()

    donor_y0 = asset_bbox["y"]
    donor_y1 = asset_bbox["y"] + asset_bbox["height"] - 1
    subject_y0 = subject_bbox["y"]
    subject_y1 = subject_bbox["y"] + subject_bbox["height"] - 1
    subject_height = max(subject_bbox["height"], 1)
    donor_height = max(asset_bbox["height"], 1)

    for y in range(subject_y0, subject_y1 + 1):
        xs = np.where(subject_array[y])[0]
        if xs.size == 0:
            continue

        subject_left = int(xs.min())
        subject_right = int(xs.max())
        subject_width = subject_right - subject_left + 1
        if subject_width <= 0:
            continue

        y_ratio = (y - subject_y0) / max(subject_height - 1, 1)
        donor_y = donor_y0 + int(round(y_ratio * max(donor_height - 1, 0)))
        donor_y = max(donor_y0, min(donor_y, donor_y1))

        donor_xs = np.where(donor_mask[donor_y])[0]
        if donor_xs.size == 0:
            probe = donor_y
            found = None
            for offset in range(1, 8):
                for candidate in (probe - offset, probe + offset):
                    if donor_y0 <= candidate <= donor_y1:
                        candidate_xs = np.where(donor_mask[candidate])[0]
                        if candidate_xs.size > 0:
                            found = candidate
                            donor_xs = candidate_xs
                            break
                if found is not None:
                    donor_y = found
                    break
        if donor_xs.size == 0:
            continue

        donor_left = int(donor_xs.min())
        donor_right = int(donor_xs.max())
        donor_row = donor_rgb[donor_y, donor_left : donor_right + 1, :]
        if donor_row.size == 0:
            continue

        row_patch = Image.fromarray(donor_row[np.newaxis, :, :], mode="RGB").resize(
            (subject_width, 1),
            Image.Resampling.BILINEAR,
        )
        row_rgb = np.asarray(row_patch, dtype=np.uint8)[0]
        canvas_rgb[y, subject_left : subject_right + 1, :] = row_rgb

    texture_layer = Image.fromarray(canvas_rgb, mode="RGB").convert("RGBA")
    alpha = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
    alpha[subject_array] = 255
    alpha_image = Image.fromarray(alpha, mode="L")
    overlay_only = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    overlay_only.paste(texture_layer, (0, 0), alpha_image)

    composed = erased.copy()
    composed.alpha_composite(overlay_only)
    return composed


def render_legacy_static_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
    else:
        input_image = Image.open(Path(input_image_path)).convert("RGBA")

    asset_image_path, asset_mask_path = _resolve_tryon_asset_paths(
        asset,
        prefer_clean_variant=True,
    )
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _crop_asset_to_mask_bounds(
        hair_asset,
        hair_asset_mask,
        padding_ratio=0.16,
    )
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _cleanup_static_tryon_asset(hair_asset, hair_asset_mask, asset)

    cleaned_subject_hair_mask = _largest_subject_hair_component(subject_hair_mask)

    if layout_hint is None:
        subject_hair_bbox = _hair_mask_bbox(cleaned_subject_hair_mask)
        subject_hair_top_bbox = _hair_mask_top_band_bbox(cleaned_subject_hair_mask)
    else:
        subject_hair_bbox = layout_hint.get("subject_hair_bbox")
        subject_hair_top_bbox = layout_hint.get("subject_hair_top_bbox")

    resized_hair, resized_mask = _legacy_scale_asset(
        hair_asset,
        hair_asset_mask,
        face_analysis,
        asset,
        subject_hair_bbox,
        subject_hair_top_bbox,
    )
    resized_hair, resized_mask = _rotate_hair_asset(
        resized_hair,
        resized_mask,
        face_analysis,
    )

    paste_x, paste_y = _legacy_paste_position(
        face_analysis,
        asset,
        resized_mask,
        input_image.size,
        subject_hair_bbox,
        subject_hair_top_bbox,
        cleaned_subject_hair_mask,
    )

    return _legacy_compose_tryon(
        input_image,
        resized_hair,
        resized_mask,
        paste_x,
        paste_y,
        face_analysis,
        asset,
        cleaned_subject_hair_mask,
    )


def render_legacy_static_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> str | None:
    composed = render_legacy_static_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
        layout_hint=layout_hint,
    )
    if composed is None:
        return None

    if isinstance(input_image_path, Image.Image):
        input_stem = "legacy_live_frame"
    else:
        input_stem = Path(input_image_path).stem

    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_stem}_{asset.asset_id}_{uuid4().hex[:8]}.png"
    output_path = TRYON_DIR / output_name
    composed.convert("RGB").save(output_path)
    return str(output_path)


def render_static_texture_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
    else:
        input_image = Image.open(Path(input_image_path)).convert("RGBA")

    asset_image_path = _resolve_project_path(asset.image_path)
    asset_mask_path = _resolve_project_path(asset.mask_path)
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _crop_asset_to_mask_bounds(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _cleanup_static_tryon_asset(hair_asset, hair_asset_mask, asset)

    cleaned_subject_hair_mask = _largest_subject_hair_component(subject_hair_mask)
    if cleaned_subject_hair_mask is None:
        return render_headfit_static_tryon_image(
            input_image,
            face_analysis,
            asset,
            subject_hair_mask=subject_hair_mask,
            layout_hint=layout_hint,
        )

    return _render_subject_silhouette_texture_tryon(
        input_image,
        hair_asset,
        hair_asset_mask,
        face_analysis,
        cleaned_subject_hair_mask,
    )


def render_static_hybrid_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
    else:
        input_image = Image.open(Path(input_image_path)).convert("RGBA")

    asset_image_path = _resolve_project_path(asset.image_path)
    asset_mask_path = _resolve_project_path(asset.mask_path)
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _crop_asset_to_mask_bounds(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _cleanup_static_tryon_asset(hair_asset, hair_asset_mask, asset)

    cleaned_subject_hair_mask = _largest_subject_hair_component(subject_hair_mask)
    if cleaned_subject_hair_mask is None:
        return render_headfit_static_tryon_image(
            input_image,
            face_analysis,
            asset,
            subject_hair_mask=subject_hair_mask,
            layout_hint=layout_hint,
        )

    # Base: keep the good head-fitting behavior from subject-silhouette texture
    # transfer inside the actual subject hair region.
    base = _render_subject_silhouette_texture_tryon(
        input_image,
        hair_asset,
        hair_asset_mask,
        face_analysis,
        cleaned_subject_hair_mask,
    )

    if layout_hint is None:
        subject_hair_bbox = _hair_mask_bbox(cleaned_subject_hair_mask)
        subject_hair_top_bbox = _hair_mask_top_band_bbox(cleaned_subject_hair_mask)
    else:
        subject_hair_bbox = layout_hint.get("subject_hair_bbox")
        subject_hair_top_bbox = layout_hint.get("subject_hair_top_bbox")

    resized_hair, resized_mask = _legacy_scale_asset(
        hair_asset,
        hair_asset_mask,
        face_analysis,
        asset,
        subject_hair_bbox,
        subject_hair_top_bbox,
    )
    resized_hair, resized_mask = _rotate_hair_asset(resized_hair, resized_mask, face_analysis)

    paste_x, paste_y = _legacy_paste_position(
        face_analysis,
        asset,
        resized_mask,
        input_image.size,
        subject_hair_bbox,
        subject_hair_top_bbox,
        cleaned_subject_hair_mask,
    )

    overlay_layer = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
    overlay_layer.paste(resized_hair, (paste_x, paste_y))
    overlay_alpha = np.asarray(overlay_layer.getchannel("A"), dtype=np.float32) / 255.0
    overlay_bbox = _hair_mask_bbox(overlay_layer.getchannel("A"))

    subject_region = _build_subject_hair_region_mask(input_image.size, cleaned_subject_hair_mask)
    subject_alpha = np.asarray(
        subject_region.filter(ImageFilter.MaxFilter(size=19)).filter(ImageFilter.GaussianBlur(radius=2.2)),
        dtype=np.float32,
    ) / 255.0

    subject_open = _legacy_forehead_opening_anchor(cleaned_subject_hair_mask)
    face_bbox = face_analysis.face_bbox or {"height": 1}

    top_ref = subject_hair_top_bbox or subject_hair_bbox or {"x": 0, "y": 0, "width": input_image.size[0], "height": 1}
    anchor = _estimate_anchor_data(face_analysis)

    if subject_open is not None:
        fade_start_y = int(subject_open[1] - face_bbox["height"] * 0.24)
        fade_end_y = int(subject_open[1] - face_bbox["height"] * 0.12)
    else:
        fade_start_y = int(top_ref["y"] + top_ref["height"] * 0.06)
        fade_end_y = int(top_ref["y"] + top_ref["height"] * 0.36)
    fade_start_y = max(0, min(fade_start_y, input_image.size[1] - 1))
    fade_end_y = max(fade_start_y + 1, min(fade_end_y, input_image.size[1]))

    row_gate = np.zeros(input_image.size[1], dtype=np.float32)
    row_gate[:fade_start_y] = 1.0
    row_gate[fade_start_y:fade_end_y] = np.linspace(
        1.0,
        0.0,
        fade_end_y - fade_start_y,
        dtype=np.float32,
    )

    width_top = max(int(top_ref["width"] * 1.10), 1)
    width_bottom = max(int(top_ref["width"] * 0.86), 1)
    center_x = int(round(anchor["temple_center_x"]))
    top_y = max(int(top_ref["y"] - top_ref["height"] * 0.22), 0)
    bottom_y = min(fade_end_y, input_image.size[1] - 1)
    crown_support = np.zeros((input_image.size[1], input_image.size[0]), dtype=np.uint8)
    crown_poly = np.array(
        [
            [max(center_x - width_top // 2, 0), top_y],
            [min(center_x + width_top // 2, input_image.size[0] - 1), top_y],
            [min(center_x + width_bottom // 2, input_image.size[0] - 1), bottom_y],
            [max(center_x - width_bottom // 2, 0), bottom_y],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(crown_support, crown_poly, 255)
    crown_support = cv2.GaussianBlur(crown_support, (11, 11), 0).astype(np.float32) / 255.0

    extension_alpha = overlay_alpha * (1.0 - subject_alpha) * row_gate[:, None] * crown_support

    if overlay_bbox is not None:
        crown_limit_y = overlay_bbox["y"] + int(overlay_bbox["height"] * 0.44)
        crown_limit_y = max(0, min(crown_limit_y, input_image.size[1]))
        if crown_limit_y < input_image.size[1]:
            extension_alpha[crown_limit_y:, :] *= 0.02

    extension_mask = Image.fromarray(
        np.clip(extension_alpha * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=1.4))
    extension_layer = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
    extension_layer.paste(overlay_layer, (0, 0), extension_mask)

    composed = base.copy()
    composed.alpha_composite(extension_layer)
    return composed


def render_static_hybrid_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> str | None:
    composed = render_static_hybrid_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
        layout_hint=layout_hint,
    )
    if composed is None:
        return None

    if isinstance(input_image_path, Image.Image):
        input_stem = "static_frame"
    else:
        input_stem = Path(input_image_path).stem

    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_stem}_{asset.asset_id}_{uuid4().hex[:8]}.png"
    output_path = TRYON_DIR / output_name
    composed.convert("RGB").save(output_path)
    return str(output_path)


def render_static_texture_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
    layout_hint: dict[str, dict[str, int] | None] | None = None,
) -> str | None:
    composed = render_static_texture_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
        layout_hint=layout_hint,
    )
    if composed is None:
        return None

    if isinstance(input_image_path, Image.Image):
        input_stem = "static_frame"
    else:
        input_stem = Path(input_image_path).stem

    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_stem}_{asset.asset_id}_{uuid4().hex[:8]}.png"
    output_path = TRYON_DIR / output_name
    composed.convert("RGB").save(output_path)
    return str(output_path)


render_static_tryon_image = render_legacy_static_tryon_image
render_static_tryon = render_legacy_static_tryon

