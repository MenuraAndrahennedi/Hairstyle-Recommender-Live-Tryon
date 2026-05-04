from __future__ import annotations

"""Experimental pose-aware 2.5D warp helpers.

This module is intentionally kept separate from the strong 2D baseline.
It is a bridge experiment for future pose-aware rendering work, but it is
not the default runtime path because current single-view assets do not
produce a consistent quality gain under this warp.
"""

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from app.config import TRYON_DIR
from app.core.tryon_2d_engine import (
    _apply_asset_mask,
    _compose_tryon,
    _crop_asset_to_mask_bounds,
    _hair_mask_bbox,
    _hair_mask_top_band_bbox,
    _largest_subject_hair_component,
    _overlay_position,
    _resize_asset_mask,
    _resize_hair_asset,
    _resolve_project_path,
    _rotate_hair_asset,
)
from app.core.tryon_3d_metadata import build_3d_tryon_candidate_metadata
from app.models.schemas import AssetMetadata, FaceAnalysisResult


def estimate_pose_warp_parameters(
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
) -> dict[str, float] | None:
    candidate = build_3d_tryon_candidate_metadata(asset, face_analysis)
    if candidate is None:
        return None

    anchor = candidate["head_anchor_frame"]
    priors = candidate["placement_priors"]

    yaw_proxy = float(anchor["yaw_proxy"])
    pitch_proxy = float(anchor["pitch_proxy"])
    roll_degrees = float(anchor["roll_degrees"])

    yaw_strength = max(min(yaw_proxy * 1.8, 0.42), -0.42)
    pitch_strength = max(min(pitch_proxy * 1.5, 0.28), -0.28)
    depth_bias = float(priors["depth_bias"])
    side_fall_bias = float(priors["side_fall_bias"])
    scale_multiplier = float(priors["scale_multiplier"])

    return {
        "yaw_strength": yaw_strength,
        "pitch_strength": pitch_strength,
        "roll_degrees": roll_degrees,
        "depth_bias": depth_bias,
        "side_fall_bias": side_fall_bias,
        "scale_multiplier": scale_multiplier,
    }


def _warp_asset_by_pose(
    asset_image: Image.Image,
    asset_mask: Image.Image,
    pose_parameters: dict[str, float],
) -> tuple[Image.Image, Image.Image]:
    yaw = pose_parameters["yaw_strength"]
    pitch = pose_parameters["pitch_strength"]
    depth_bias = pose_parameters["depth_bias"]
    side_fall_bias = pose_parameters["side_fall_bias"]
    scale_multiplier = pose_parameters["scale_multiplier"]

    width, height = asset_image.size
    resized_width = max(int(width * (scale_multiplier + (abs(yaw) * 0.04))), 1)
    resized_height = max(
        int(
            height
            * (
                scale_multiplier
                + (max(-pitch, 0.0) * 0.05)
                + (max(pitch, 0.0) * 0.02)
                + (depth_bias * 0.04)
            )
        ),
        1,
    )

    warped_image = asset_image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    warped_mask = asset_mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)

    near_scale = 1.0 + abs(yaw) * (0.28 + side_fall_bias * 0.12)
    far_scale = max(0.72, 1.0 - abs(yaw) * (0.34 + depth_bias * 0.16))

    if yaw >= 0:
        left_scale, right_scale = far_scale, near_scale
    else:
        left_scale, right_scale = near_scale, far_scale

    split_x = max(int(resized_width * 0.5), 1)
    left_image = warped_image.crop((0, 0, split_x, resized_height))
    right_image = warped_image.crop((split_x, 0, resized_width, resized_height))
    left_mask = warped_mask.crop((0, 0, split_x, resized_height))
    right_mask = warped_mask.crop((split_x, 0, resized_width, resized_height))

    left_width = max(int(left_image.width * left_scale), 1)
    right_width = max(int(right_image.width * right_scale), 1)
    left_image = left_image.resize((left_width, resized_height), Image.Resampling.BICUBIC)
    right_image = right_image.resize((right_width, resized_height), Image.Resampling.BICUBIC)
    left_mask = left_mask.resize((left_width, resized_height), Image.Resampling.NEAREST)
    right_mask = right_mask.resize((right_width, resized_height), Image.Resampling.NEAREST)

    merged_width = left_width + right_width
    merged_image = Image.new("RGBA", (merged_width, resized_height), (0, 0, 0, 0))
    merged_mask = Image.new("L", (merged_width, resized_height), 0)
    merged_image.paste(left_image, (0, 0))
    merged_image.paste(right_image, (left_width, 0))
    merged_mask.paste(left_mask, (0, 0))
    merged_mask.paste(right_mask, (left_width, 0))

    top_ratio = 0.42
    top_h = max(int(resized_height * top_ratio), 1)
    bottom_h = max(resized_height - top_h, 1)
    top_image = merged_image.crop((0, 0, merged_width, top_h))
    bottom_image = merged_image.crop((0, top_h, merged_width, resized_height))
    top_mask = merged_mask.crop((0, 0, merged_width, top_h))
    bottom_mask = merged_mask.crop((0, top_h, merged_width, resized_height))

    crown_scale = max(0.92, 1.0 - abs(yaw) * 0.05)
    top_pitch_scale = max(0.86, 1.0 - (max(pitch, 0.0) * 0.18) + (max(-pitch, 0.0) * 0.08))
    bottom_pitch_scale = min(1.14, 1.0 + (max(pitch, 0.0) * 0.18) - (max(-pitch, 0.0) * 0.06))

    top_w = max(int(merged_width * crown_scale), 1)
    top_h2 = max(int(top_h * top_pitch_scale), 1)
    bottom_h2 = max(int(bottom_h * bottom_pitch_scale), 1)

    top_image = top_image.resize((top_w, top_h2), Image.Resampling.BICUBIC)
    top_mask = top_mask.resize((top_w, top_h2), Image.Resampling.NEAREST)
    bottom_image = bottom_image.resize((merged_width, bottom_h2), Image.Resampling.BICUBIC)
    bottom_mask = bottom_mask.resize((merged_width, bottom_h2), Image.Resampling.NEAREST)

    final_width = max(merged_width, top_w)
    final_height = top_h2 + bottom_h2
    final_image = Image.new("RGBA", (final_width, final_height), (0, 0, 0, 0))
    final_mask = Image.new("L", (final_width, final_height), 0)

    top_x = max((final_width - top_w) // 2 + int(yaw * merged_width * 0.05), 0)
    bottom_x = max((final_width - merged_width) // 2, 0)

    final_image.paste(top_image, (top_x, 0))
    final_mask.paste(top_mask, (top_x, 0))
    final_image.paste(bottom_image, (bottom_x, top_h2))
    final_mask.paste(bottom_mask, (bottom_x, top_h2))

    mask_bbox = final_mask.getbbox()
    if mask_bbox is None:
        return merged_image, merged_mask

    final_image = final_image.crop(mask_bbox)
    final_mask = final_mask.crop(mask_bbox)

    # Light cleanup so the asymmetric warp does not leave rough edge specks.
    mask_arr = np.asarray(final_mask, dtype=np.uint8)
    mask_arr = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
    final_mask = Image.fromarray(mask_arr, mode="L")
    return final_image, final_mask


def render_pose_aware_tryon_image(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
) -> Image.Image | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    pose_parameters = estimate_pose_warp_parameters(face_analysis, asset)
    if pose_parameters is None:
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
    subject_hair_bbox = _hair_mask_bbox(cleaned_subject_hair_mask)
    subject_hair_top_bbox = _hair_mask_top_band_bbox(cleaned_subject_hair_mask)

    resized_hair = _resize_hair_asset(
        hair_asset,
        hair_asset_mask,
        face_analysis,
        subject_hair_bbox=subject_hair_bbox,
        subject_hair_top_bbox=subject_hair_top_bbox,
    )
    resized_asset_mask = _resize_asset_mask(hair_asset_mask, resized_hair.size)
    resized_hair, resized_asset_mask = _rotate_hair_asset(
        resized_hair,
        resized_asset_mask,
        face_analysis,
    )
    warped_hair, warped_mask = _warp_asset_by_pose(
        resized_hair,
        resized_asset_mask,
        pose_parameters,
    )

    warped_mask_bbox = _hair_mask_bbox(warped_mask)
    paste_x, paste_y = _overlay_position(
        face_analysis,
        warped_hair.size,
        input_image.size,
        subject_hair_bbox=subject_hair_bbox,
        subject_hair_top_bbox=subject_hair_top_bbox,
        asset_mask_bbox=warped_mask_bbox,
    )

    return _compose_tryon(
        input_image,
        warped_hair,
        warped_mask,
        paste_x,
        paste_y,
        face_analysis,
        asset,
        subject_hair_mask=cleaned_subject_hair_mask,
    )


def render_pose_aware_tryon(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
) -> str | None:
    composed = render_pose_aware_tryon_image(
        input_image_path,
        face_analysis,
        asset,
        subject_hair_mask=subject_hair_mask,
    )
    if composed is None:
        return None

    if isinstance(input_image_path, Image.Image):
        input_stem = "pose_aware_frame"
    else:
        input_stem = Path(input_image_path).stem

    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_stem}_{asset.asset_id}_25d_{uuid4().hex[:8]}.png"
    output_path = TRYON_DIR / output_name
    composed.convert("RGB").save(output_path)
    return str(output_path)
