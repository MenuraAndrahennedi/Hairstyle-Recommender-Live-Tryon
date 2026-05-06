from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from support_app.config import TRYON_DIR
from support_app.core.tryon_2d_engine import (
    _average_point,
    _apply_asset_mask,
    _cleanup_static_tryon_asset,
    _crop_asset_to_mask_bounds,
    _estimate_anchor_data,
    _estimate_background_rgb,
    _hair_mask_bbox,
    _landmark_by_index,
    _pixel_point,
    _resolve_tryon_asset_paths,
    _build_subject_hair_region_mask,
)
from support_app.models.schemas import AssetMetadata, FaceAnalysisResult


TESTING_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "celeba_full_hair_assets"
    / "testing_annotations.json"
)


def _load_testing_annotations() -> dict[str, dict[str, object]]:
    if not TESTING_ANNOTATIONS_PATH.exists():
        return {}
    return json.loads(TESTING_ANNOTATIONS_PATH.read_text(encoding="utf-8"))


def _donor_top_bbox(mask_image: Image.Image, band_ratio: float = 0.34) -> dict[str, int] | None:
    bbox = _hair_mask_bbox(mask_image)
    if bbox is None:
        return None

    mask_array = np.asarray(mask_image.convert("L")) > 0
    top = bbox["y"]
    bottom = min(top + max(int(bbox["height"] * band_ratio), 1), bbox["y"] + bbox["height"])
    band = mask_array[top:bottom, :]
    if not band.any():
        return bbox

    ys, xs = np.where(band)
    return {
        "x": int(xs.min()),
        "y": int(top + ys.min()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def _donor_hairline_opening(mask_image: Image.Image) -> dict[str, float] | None:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    bbox = _hair_mask_bbox(mask_image)
    if bbox is None:
        return None

    start_y = bbox["y"] + int(bbox["height"] * 0.10)
    end_y = bbox["y"] + int(bbox["height"] * 0.45)

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

        return {
            "y": float(y),
            "left_outer_x": float(left_segment[0]),
            "left_inner_x": float(left_segment[1]),
            "right_inner_x": float(right_segment[0]),
            "right_outer_x": float(right_segment[1]),
            "gap": gap,
        }
    return None


def _top_row_span(mask_image: Image.Image, y: int) -> tuple[float, float] | None:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    if y < 0 or y >= mask_array.shape[0]:
        return None
    xs = np.where(mask_array[y])[0]
    if xs.size == 0:
        return None
    return float(xs.min()), float(xs.max())


def _extract_donor_anchor_model(mask_image: Image.Image) -> dict[str, tuple[float, float] | float] | None:
    bbox = _hair_mask_bbox(mask_image)
    donor_open = _donor_hairline_opening(mask_image)
    donor_top = _donor_top_bbox(mask_image)
    if bbox is None or donor_top is None or donor_open is None:
        return None

    crown_row_y = donor_top["y"] + max(int(donor_top["height"] * 0.16), 1)
    crown_span = _top_row_span(mask_image, crown_row_y)
    if crown_span is None:
        crown_row_y = donor_top["y"] + max(int(donor_top["height"] * 0.28), 1)
        crown_span = _top_row_span(mask_image, crown_row_y)
    if crown_span is None:
        return None

    crown_left_x, crown_right_x = crown_span
    crown_center_x = (crown_left_x + crown_right_x) / 2.0
    open_center_x = (donor_open["left_inner_x"] + donor_open["right_inner_x"]) / 2.0
    crown_bottom_y = donor_open["y"] + max(float(bbox["height"]) * 0.10, 8.0)

    return {
        "left_hairline": (donor_open["left_inner_x"], donor_open["y"]),
        "right_hairline": (donor_open["right_inner_x"], donor_open["y"]),
        "scalp_left": (donor_open["left_outer_x"], crown_bottom_y),
        "scalp_right": (donor_open["right_outer_x"], crown_bottom_y),
        "left_outer_hairline": (donor_open["left_outer_x"], donor_open["y"]),
        "right_outer_hairline": (donor_open["right_outer_x"], donor_open["y"]),
        "crown_top_left": (crown_left_x, float(crown_row_y)),
        "crown_top_right": (crown_right_x, float(crown_row_y)),
        "crown_left": (crown_left_x, float(crown_row_y)),
        "crown_right": (crown_right_x, float(crown_row_y)),
        "crown_center": (crown_center_x, float(crown_row_y)),
        "open_center": (open_center_x, donor_open["y"]),
        "crown_width": crown_right_x - crown_left_x,
        "open_width": donor_open["gap"],
        "bbox_width": float(bbox["width"]),
        "bbox_height": float(bbox["height"]),
    }


def _override_donor_anchor_model(
    asset: AssetMetadata,
    donor_model: dict[str, tuple[float, float] | float] | None,
) -> dict[str, tuple[float, float] | float] | None:
    if donor_model is None:
        return None
    asset_data = _load_testing_annotations().get(asset.asset_id)
    if not asset_data:
        return donor_model

    anchors = asset_data.get("anchors", {})
    updated = dict(donor_model)
    for key, value in anchors.items():
        if isinstance(value, list) and len(value) == 2:
            updated[key] = (float(value[0]), float(value[1]))
    return updated


def _subject_anchor_model(
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
) -> dict[str, tuple[float, float] | float]:
    image_width = face_analysis.image_width
    image_height = face_analysis.image_height
    landmarks = face_analysis.landmarks
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)

    left_temple = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in [127, 234]]
    )
    right_temple = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in [356, 454]]
    )
    forehead = _pixel_point(_landmark_by_index(landmarks, 10), image_width, image_height)

    if asset.normalized_attributes.bang in {"full", "side", "see_through"}:
        hairline_y = forehead[1] + (face_bbox["height"] * 0.03)
        crown_y = forehead[1] - (face_bbox["height"] * 0.10)
    else:
        hairline_y = forehead[1] + (face_bbox["height"] * 0.01)
        crown_y = forehead[1] - (face_bbox["height"] * 0.16)

    outer_half_width = max(anchor["temple_span"] * 0.74, face_bbox["width"] * 0.62)
    crown_half_width = max(anchor["temple_span"] * 0.82, face_bbox["width"] * 0.68)
    scalp_half_width = max(anchor["temple_span"] * 0.66, face_bbox["width"] * 0.56)
    scalp_y = hairline_y + face_bbox["height"] * (0.02 if asset.normalized_attributes.bang != "none" else 0.00)

    return {
        "left_hairline": (left_temple[0], hairline_y),
        "right_hairline": (right_temple[0], hairline_y),
        "scalp_left": (anchor["temple_center_x"] - scalp_half_width, scalp_y),
        "scalp_right": (anchor["temple_center_x"] + scalp_half_width, scalp_y),
        "left_outer_hairline": (anchor["temple_center_x"] - outer_half_width, hairline_y),
        "right_outer_hairline": (anchor["temple_center_x"] + outer_half_width, hairline_y),
        "crown_top_left": (anchor["temple_center_x"] - crown_half_width, crown_y),
        "crown_top_right": (anchor["temple_center_x"] + crown_half_width, crown_y),
        "crown_left": (anchor["temple_center_x"] - crown_half_width, crown_y),
        "crown_right": (anchor["temple_center_x"] + crown_half_width, crown_y),
        "crown_center": (anchor["temple_center_x"], crown_y),
        "open_center": (anchor["temple_center_x"], hairline_y),
        "temple_span": anchor["temple_span"],
    }


def _bbox_from_boolean_mask(mask_array: np.ndarray) -> dict[str, int] | None:
    if not mask_array.any():
        return None
    ys, xs = np.where(mask_array)
    return {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def _crop_region_from_mask(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    region_mask_array: np.ndarray,
) -> tuple[Image.Image, Image.Image, dict[str, int]] | None:
    bbox = _bbox_from_boolean_mask(region_mask_array)
    if bbox is None:
        return None

    rgba = np.asarray(asset_image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.zeros_like(np.asarray(asset_mask.convert("L"), dtype=np.uint8))
    alpha[region_mask_array] = np.asarray(asset_mask.convert("L"), dtype=np.uint8)[region_mask_array]
    rgba[..., 3] = alpha

    crop_box = (
        bbox["x"],
        bbox["y"],
        bbox["x"] + bbox["width"],
        bbox["y"] + bbox["height"],
    )
    cropped_image = Image.fromarray(rgba, mode="RGBA").crop(crop_box)
    cropped_mask = Image.fromarray(alpha, mode="L").crop(crop_box)
    cropped_image.putalpha(cropped_mask)
    return cropped_image, cropped_mask, bbox


def _extract_donor_regions(
    asset: AssetMetadata,
    asset_image: Image.Image,
    asset_mask: Image.Image,
) -> dict[str, tuple[Image.Image, Image.Image, dict[str, int]]] | None:
    donor_model = _override_donor_anchor_model(asset, _extract_donor_anchor_model(asset_mask))
    bbox = _hair_mask_bbox(asset_mask)
    if donor_model is None or bbox is None:
        return None

    mask_array = np.asarray(asset_mask.convert("L")) > 0
    asset_data = _load_testing_annotations().get(asset.asset_id, {})
    region_data = asset_data.get("regions", {})
    crown_bottom_y = int(region_data.get("crown_bottom_y", round(donor_model["scalp_left"][1] + bbox["height"] * 0.18)))
    crown_bottom_y = min(crown_bottom_y, bbox["y"] + bbox["height"] - 1)
    left_split_x = int(region_data.get("left_split_x", round(donor_model["left_hairline"][0] + bbox["width"] * 0.06)))
    right_split_x = int(region_data.get("right_split_x", round(donor_model["right_hairline"][0] - bbox["width"] * 0.06)))

    crown_mask = mask_array.copy()
    crown_mask[crown_bottom_y + 1 :, :] = False

    left_mask = mask_array.copy()
    left_mask[:crown_bottom_y, :] = False
    left_mask[:, left_split_x + 1 :] = False

    right_mask = mask_array.copy()
    right_mask[:crown_bottom_y, :] = False
    right_mask[:, :right_split_x] = False

    regions: dict[str, tuple[Image.Image, Image.Image, dict[str, int]]] = {}
    crown_region = _crop_region_from_mask(asset_image, asset_mask, crown_mask)
    left_region = _crop_region_from_mask(asset_image, asset_mask, left_mask)
    right_region = _crop_region_from_mask(asset_image, asset_mask, right_mask)
    if crown_region is None:
        return None
    regions["crown"] = crown_region
    if left_region is not None:
        regions["left"] = left_region
    if right_region is not None:
        regions["right"] = right_region
    return regions


def _load_donor_asset(asset: AssetMetadata) -> tuple[Image.Image, Image.Image]:
    asset_image_path, asset_mask_path = _resolve_tryon_asset_paths(
        asset,
        prefer_clean_variant=True,
    )
    hair_asset = Image.open(asset_image_path).convert("RGBA")
    hair_asset_mask = Image.open(asset_mask_path).convert("L")
    hair_asset, hair_asset_mask = _crop_asset_to_mask_bounds(
        hair_asset,
        hair_asset_mask,
        padding_ratio=0.18,
    )
    hair_asset, hair_asset_mask = _apply_asset_mask(hair_asset, hair_asset_mask)
    hair_asset, hair_asset_mask = _cleanup_static_tryon_asset(hair_asset, hair_asset_mask, asset)
    return hair_asset, hair_asset_mask


def _warp_region_quad(
    region_image: Image.Image,
    region_mask: Image.Image,
    src_quad: np.ndarray,
    dst_quad: np.ndarray,
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    transform = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad.astype(np.float32))
    canvas_w, canvas_h = canvas_size

    rgba = np.asarray(region_image.convert("RGBA"), dtype=np.uint8)
    warped_rgba = cv2.warpPerspective(
        rgba,
        transform,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha = np.asarray(region_mask.convert("L"), dtype=np.uint8)
    warped_alpha = cv2.warpPerspective(
        alpha,
        transform,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_rgba[..., 3] = warped_alpha
    warped_image = Image.fromarray(warped_rgba, mode="RGBA")
    warped_mask = Image.fromarray(warped_alpha, mode="L")
    warped_image.putalpha(warped_mask)
    return warped_image, warped_mask


def _place_region_box(
    region_image: Image.Image,
    region_mask: Image.Image,
    target_box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    left, top, right, bottom = target_box
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    resized_image = region_image.resize((width, height), Image.Resampling.LANCZOS)
    resized_mask = region_mask.resize((width, height), Image.Resampling.NEAREST)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer_mask = Image.new("L", canvas_size, 0)
    layer.paste(resized_image, (left, top))
    layer_mask.paste(resized_mask, (left, top))
    layer.putalpha(layer_mask)
    return layer, layer_mask


def _compose_region_layers(layers: list[tuple[Image.Image, Image.Image]], canvas_size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    rgba = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    alpha = Image.new("L", canvas_size, 0)
    for layer, layer_mask in layers:
        rgba.alpha_composite(layer)
        alpha = Image.fromarray(
            np.maximum(np.asarray(alpha, dtype=np.uint8), np.asarray(layer_mask, dtype=np.uint8)),
            mode="L",
        )
    rgba.putalpha(alpha)
    return rgba, alpha


def _warp_donor_to_subject(
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    asset_image: Image.Image,
    asset_mask: Image.Image,
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image] | None:
    donor_model = _override_donor_anchor_model(asset, _extract_donor_anchor_model(asset_mask))
    if donor_model is None:
        return None
    subject_model = _subject_anchor_model(face_analysis, asset)
    regions = _extract_donor_regions(asset, asset_image, asset_mask)
    if regions is None:
        return None

    crown_image, crown_mask, crown_bbox = regions["crown"]
    crown_src = np.array(
        [
            [donor_model["crown_top_left"][0] - crown_bbox["x"], donor_model["crown_top_left"][1] - crown_bbox["y"]],
            [donor_model["crown_top_right"][0] - crown_bbox["x"], donor_model["crown_top_right"][1] - crown_bbox["y"]],
            [donor_model["scalp_right"][0] - crown_bbox["x"], donor_model["scalp_right"][1] - crown_bbox["y"]],
            [donor_model["scalp_left"][0] - crown_bbox["x"], donor_model["scalp_left"][1] - crown_bbox["y"]],
        ],
        dtype=np.float32,
    )
    crown_dst = np.array(
        [
            subject_model["crown_top_left"],
            subject_model["crown_top_right"],
            subject_model["scalp_right"],
            subject_model["scalp_left"],
        ],
        dtype=np.float32,
    )
    layers: list[tuple[Image.Image, Image.Image]] = [
        _warp_region_quad(crown_image, crown_mask, crown_src, crown_dst, canvas_size)
    ]

    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    temple_span = float(subject_model["temple_span"])
    hairline_y = float(subject_model["scalp_left"][1])
    side_top = int(round(hairline_y - face_bbox["height"] * 0.01))
    side_bottom = int(round(face_bbox["y"] + face_bbox["height"] * (1.12 if asset.normalized_attributes.length == "long" else 0.88)))
    left_outer_x = int(round(subject_model["left_outer_hairline"][0] - temple_span * 0.10))
    left_inner_x = int(round(subject_model["scalp_left"][0] + temple_span * 0.04))
    right_inner_x = int(round(subject_model["scalp_right"][0] - temple_span * 0.04))
    right_outer_x = int(round(subject_model["right_outer_hairline"][0] + temple_span * 0.10))

    if "left" in regions:
        left_image, left_mask, _ = regions["left"]
        layers.append(
            _place_region_box(
                left_image,
                left_mask,
                (left_outer_x, side_top, left_inner_x, side_bottom),
                canvas_size,
            )
        )
    if "right" in regions:
        right_image, right_mask, _ = regions["right"]
        layers.append(
            _place_region_box(
                right_image,
                right_mask,
                (right_inner_x, side_top, right_outer_x, side_bottom),
                canvas_size,
            )
        )

    return _compose_region_layers(layers, canvas_size)


def _build_mediapipe_keepout_mask(
    canvas_size: tuple[int, int],
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
) -> Image.Image:
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)
    mask = Image.new("L", canvas_size, 0)
    draw = ImageDraw.Draw(mask)

    left = max(int(face_bbox["x"] - face_bbox["width"] * 0.10), 0)
    right = min(int(face_bbox["x"] + face_bbox["width"] * 1.10), canvas_size[0])
    top = max(int(anchor["eye_center_y"] - face_bbox["height"] * 0.04), 0)
    bottom = min(int(face_bbox["y"] + face_bbox["height"] * 1.02), canvas_size[1])

    if asset.normalized_attributes.bang in {"full", "side", "see_through"}:
        top = max(int(face_bbox["y"] + face_bbox["height"] * 0.34), 0)
    elif asset.normalized_attributes.bang == "none":
        top = max(int(face_bbox["y"] + face_bbox["height"] * 0.14), 0)

    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(int(face_bbox["width"] * 0.16), 10),
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=10))


def _erase_subject_hair(
    input_image: Image.Image,
    subject_hair_mask: Image.Image | None,
) -> Image.Image:
    subject_region = _build_subject_hair_region_mask(input_image.size, subject_hair_mask)
    if subject_region is None:
        return input_image.convert("RGBA")

    background_rgb = _estimate_background_rgb(input_image, subject_region)
    base = input_image.convert("RGBA").copy()
    fill = Image.new("RGBA", input_image.size, (*background_rgb, 255))
    erase_mask = subject_region.filter(ImageFilter.MaxFilter(size=11)).filter(ImageFilter.GaussianBlur(radius=2.2))
    return Image.composite(fill, base, erase_mask)


def _compose_mediapipe_tryon(
    input_image: Image.Image,
    asset_image: Image.Image,
    asset_mask: Image.Image,
    paste_x: int,
    paste_y: int,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None,
) -> Image.Image:
    base = _erase_subject_hair(input_image, subject_hair_mask)
    layer = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
    layer.paste(asset_image, (paste_x, paste_y))

    alpha = np.asarray(layer.getchannel("A"), dtype=np.float32) / 255.0
    keepout = np.asarray(_build_mediapipe_keepout_mask(input_image.size, face_analysis, asset), dtype=np.float32) / 255.0
    alpha = np.clip(alpha * (1.0 - keepout), 0.0, 1.0)

    final_alpha = Image.fromarray(
        np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=0.8))
    layer.putalpha(final_alpha)

    composed = base.copy()
    composed.alpha_composite(layer)
    return composed


def render_mediapipe_static_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
    else:
        input_image = Image.open(Path(input_image_path)).convert("RGBA")

    donor_image, donor_mask = _load_donor_asset(asset)
    warped = _warp_donor_to_subject(
        face_analysis,
        asset,
        donor_image,
        donor_mask,
        input_image.size,
    )
    if warped is None:
        return None
    warped_image, warped_mask = warped
    return _compose_mediapipe_tryon(
        input_image,
        warped_image,
        warped_mask,
        0,
        0,
        face_analysis,
        asset,
        subject_hair_mask,
    )


def render_mediapipe_static_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
) -> str | None:
    composed = render_mediapipe_static_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
    )
    if composed is None:
        return None

    input_stem = "mediapipe_frame" if isinstance(input_image_path, Image.Image) else Path(input_image_path).stem
    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRYON_DIR / f"{input_stem}_{asset.asset_id}_mediapipe_{uuid4().hex[:8]}.png"
    composed.convert("RGB").save(output_path)
    return str(output_path)

