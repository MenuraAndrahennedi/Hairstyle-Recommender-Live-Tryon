from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.config import PROJECT_ROOT, TRYON_DIR
from app.models.schemas import AssetMetadata, FaceAnalysisResult, FaceLandmark


def _resolve_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


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
        short_lifted_style = (
            asset is not None
            and asset.normalized_attributes.style_family in {"pompadour", "regent", "side_part", "curly_crop"}
            and asset.normalized_attributes.side_hair == "exposed"
            and asset.normalized_attributes.bang == "none"
        )
        if short_lifted_style:
            target_width = max(
                int(max(landmark_target_width * 1.10, subject_hair_width * 1.32)),
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
    target_height = max(
        int(asset_image.height * scale),
        int(anchor["upper_face_height"] * (1.28 if live_landmark_only else 1.18)),
        int(face_height * (1.08 if live_landmark_only else 0.98)),
        int(subject_hair_height * 0.96) if subject_hair_height > 0 else 1,
    )

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
        and asset.normalized_attributes.style_family in {"pompadour", "regent", "side_part", "curly_crop"}
        and asset.normalized_attributes.side_hair == "exposed"
        and asset_profile["bottom_ratio"] <= 0.90
    ):
        max_allowed_height = min(
            max_allowed_height,
            int(subject_hair_height * 1.45),   # <-- FIXED
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

    short_lifted_style = (
        asset is not None
        and asset.normalized_attributes.style_family in {"pompadour", "regent", "side_part", "curly_crop"}
        and asset.normalized_attributes.side_hair == "exposed"
        and asset.normalized_attributes.bang == "none"
    )

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

            # Align hair asset mask bottom directly to the detected subject hair mask
            # bottom so the generated hairstyle sits on the existing hair region.
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
        top = max(int(face_bbox["y"] + face_bbox["height"] * 0.18), 0)
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
            # Bottom corners excluded — portrait photos typically have clothing
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
        expand_up = int(face_bbox["height"] * 0.55)   
        expand_side = int(face_bbox["width"] * 0.12)

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


def _legacy_scale_asset(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    asset: AssetMetadata,
    subject_hair_bbox: dict[str, int] | None,
    subject_hair_top_bbox: dict[str, int] | None,
) -> tuple[Image.Image, Image.Image]:
    asset_bbox, asset_top_bbox = _legacy_visible_bboxes(asset_mask)
    if asset_bbox is None:
        return asset_image, asset_mask

    subject_bbox = subject_hair_bbox
    subject_top = subject_hair_top_bbox or subject_hair_bbox

    scale_candidates: list[float] = []
    short_lifted_style = (
        asset.normalized_attributes.style_family in {"pompadour", "regent", "side_part", "curly_crop"}
        and asset.normalized_attributes.side_hair == "exposed"
        and asset.normalized_attributes.bang == "none"
    )
    cap_bbox = _short_style_cap_bbox(asset_mask) if short_lifted_style else None
    if subject_bbox is not None:
        scale_candidates.append((subject_bbox["height"] * 1.0) / max(asset_bbox["height"], 1))
        scale_candidates.append((subject_bbox["width"] * 1.02) / max(asset_bbox["width"], 1))
    if subject_top is not None and asset_top_bbox is not None:
        scale_candidates.append((subject_top["width"] * 1.02) / max(asset_top_bbox["width"], 1))
        scale_candidates.append((subject_top["height"] * 1.02) / max(asset_top_bbox["height"], 1))

    if not scale_candidates:
        return asset_image, asset_mask

    if short_lifted_style and subject_top is not None and cap_bbox is not None:
        cap_width_scale = (subject_top["width"] * 1.04) / max(cap_bbox["width"], 1)
        full_width_scale = (
            (subject_bbox["width"] * 1.10) / max(asset_bbox["width"], 1)
            if subject_bbox is not None
            else cap_width_scale
        )
        max_height_scale = (
            (subject_bbox["height"] * 1.50) / max(asset_bbox["height"], 1)
            if subject_bbox is not None
            else cap_width_scale
        )
        scale = min(max(cap_width_scale, full_width_scale), max_height_scale)
    else:
        scale = sum(scale_candidates) / len(scale_candidates)

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
) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)
    asset_bbox, asset_top_bbox = _legacy_visible_bboxes(asset_mask)
    if asset_bbox is None:
        fallback_x = int(anchor["temple_center_x"] - (asset_mask.width / 2))
        fallback_y = int(anchor["forehead_y"] - (asset_mask.height * 0.30))
        return (
            max(0, min(fallback_x, max(canvas_width - asset_mask.width, 0))),
            max(0, min(fallback_y, max(canvas_height - asset_mask.height, 0))),
        )

    subject_bbox = subject_hair_bbox
    subject_top = subject_hair_top_bbox or subject_hair_bbox
    short_lifted_style = (
        asset.normalized_attributes.style_family in {"pompadour", "regent", "side_part", "curly_crop"}
        and asset.normalized_attributes.side_hair == "exposed"
    )

    forehead_anchor = _legacy_forehead_opening_anchor(asset_mask) if short_lifted_style else None

    if forehead_anchor is not None:
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

    if short_lifted_style and subject_bbox is not None and asset_bbox is not None:
        # Pompadour/lifted styles: anchor the bottom of the asset to the bottom
        # of the subject hair so the style sits on the head and lifts above it.
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
        if support_mask is not None:
            support_array = np.asarray(support_mask, dtype=np.float32) / 255.0
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
        erase_seed = np.maximum(
            visible_asset,                               # Full coverage — 0.95 left a 1-px exposed gap
            subject_hair_array * soft_asset_array * 0.4,
        )
        erase_mask = Image.fromarray((255.0 * np.clip(erase_seed, 0.0, 1.0)).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=10)          # Softer transition at hair boundary
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

    asset_image_path = _resolve_project_path(asset.image_path)
    asset_mask_path = _resolve_project_path(asset.mask_path)
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _crop_asset_to_mask_bounds(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)

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


# Static non-live path uses the legacy compositor which erases the original
# hair, fills the background, applies the support mask, and then composites.
# The experimental compositor skips all of those steps and produces a green-
# tinted, unnatural overlay on static uploads — do not use it for this path.
render_static_tryon_image = render_legacy_static_tryon_image
render_static_tryon = render_legacy_static_tryon