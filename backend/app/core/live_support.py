from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.core.tryon_3d_metadata import estimate_head_anchor_frame
from app.core.tryon_2d_engine import subject_hair_layout_bboxes


def face_patch_brightness(
    image: Image.Image,
    face_bbox: dict[str, int] | None,
) -> float | None:
    if face_bbox is None:
        return None

    gray = image.convert("L")
    left = max(int(face_bbox["x"]), 0)
    top = max(int(face_bbox["y"]), 0)
    right = min(int(face_bbox["x"] + face_bbox["width"]), gray.width)
    bottom = min(int(face_bbox["y"] + face_bbox["height"]), gray.height)
    if right <= left or bottom <= top:
        return None

    patch = np.asarray(gray.crop((left, top, right, bottom)), dtype=np.float32)
    if patch.size == 0:
        return None
    return float(patch.mean())


def evaluate_supported_live_range(
    image: Image.Image,
    analysis: Any,
) -> dict[str, Any]:
    support = {
        "frame_supported": False,
        "support_reason": "unknown",
        "yaw_proxy": None,
        "pitch_proxy": None,
        "roll_degrees": None,
        "face_width_ratio": None,
        "brightness_mean": None,
    }

    if not analysis.face_detected or not analysis.face_bbox or not analysis.landmarks:
        support["support_reason"] = "no_face"
        return support

    anchor = estimate_head_anchor_frame(analysis)
    if anchor is None:
        support["support_reason"] = "no_anchor"
        return support

    yaw_proxy = float(anchor["yaw_proxy"])
    pitch_proxy = float(anchor["pitch_proxy"])
    roll_degrees = abs(float(anchor["roll_degrees"]))
    face_width_ratio = analysis.face_bbox["width"] / max(analysis.image_width, 1)
    brightness_mean = face_patch_brightness(image, analysis.face_bbox)

    support.update(
        {
            "yaw_proxy": yaw_proxy,
            "pitch_proxy": pitch_proxy,
            "roll_degrees": roll_degrees,
            "face_width_ratio": face_width_ratio,
            "brightness_mean": brightness_mean,
        }
    )

    if face_width_ratio < 0.16:
        support["support_reason"] = "face_too_small"
        return support
    if brightness_mean is None or brightness_mean < 38 or brightness_mean > 232:
        support["support_reason"] = "lighting_out_of_range"
        return support
    if abs(yaw_proxy) > 0.16:
        support["support_reason"] = "yaw_out_of_range"
        return support
    # Webcam posture tends to sit noticeably lower than the curated still-image
    # set, so live support needs a more permissive pitch band.
    if abs(pitch_proxy) > 0.38:
        support["support_reason"] = "pitch_out_of_range"
        return support
    if roll_degrees > 14.0:
        support["support_reason"] = "roll_out_of_range"
        return support

    support["frame_supported"] = True
    support["support_reason"] = "supported"
    return support


def summarize_live_mask_quality(
    analysis: Any,
    mask_image: Image.Image | None,
) -> dict[str, Any]:
    summary = {
        "mask_nonzero_ratio": 0.0,
        "subject_hair_bbox": None,
        "subject_hair_top_bbox": None,
        "width_ratio": None,
        "height_ratio": None,
        "center_offset_ratio": None,
        "top_offset_ratio": None,
        "mask_quality_reason": "empty",
        "mask_reliable": False,
    }

    if mask_image is None or not analysis.face_bbox:
        return summary

    mask_array = np.asarray(mask_image.convert("L")) > 0
    summary["mask_nonzero_ratio"] = float(mask_array.mean())
    if not mask_array.any():
        return summary

    subject_hair_bbox, subject_hair_top_bbox = subject_hair_layout_bboxes(mask_image)
    summary["subject_hair_bbox"] = subject_hair_bbox
    summary["subject_hair_top_bbox"] = subject_hair_top_bbox
    if subject_hair_bbox is None or subject_hair_top_bbox is None:
        summary["mask_quality_reason"] = "no_bbox"
        return summary

    face_bbox = analysis.face_bbox
    face_width = max(int(face_bbox.get("width", 0) or 0), 1)
    face_height = max(int(face_bbox.get("height", 0) or 0), 1)
    face_center_x = face_bbox["x"] + (face_width / 2)
    hair_center_x = subject_hair_top_bbox["x"] + (subject_hair_top_bbox["width"] / 2)

    width_ratio = subject_hair_top_bbox["width"] / face_width
    height_ratio = subject_hair_bbox["height"] / face_height
    center_offset_ratio = abs(hair_center_x - face_center_x) / face_width
    top_offset_ratio = (subject_hair_top_bbox["y"] - face_bbox["y"]) / face_height

    summary.update(
        {
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "center_offset_ratio": center_offset_ratio,
            "top_offset_ratio": top_offset_ratio,
        }
    )

    if width_ratio < 0.30:
        summary["mask_quality_reason"] = "mask_too_narrow"
        return summary
    if width_ratio > 1.35:
        summary["mask_quality_reason"] = "mask_too_wide"
        return summary
    if height_ratio < 0.18:
        summary["mask_quality_reason"] = "mask_too_short"
        return summary
    if height_ratio > 1.2:
        summary["mask_quality_reason"] = "mask_too_tall"
        return summary
    if center_offset_ratio > 0.34:
        summary["mask_quality_reason"] = "mask_off_center"
        return summary
    if top_offset_ratio < -0.52:
        summary["mask_quality_reason"] = "mask_too_high"
        return summary
    if top_offset_ratio > 0.36:
        summary["mask_quality_reason"] = "mask_too_low"
        return summary

    summary["mask_quality_reason"] = "reliable"
    summary["mask_reliable"] = True
    return summary
