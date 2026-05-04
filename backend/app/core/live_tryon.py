from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app.core.asset_bank import get_asset_by_id, load_live_top_tier_asset_bank
from app.core.face_analyzer import analyze_face_pil_image
from app.core.hair_segmentation import (
    clean_predicted_hair_mask,
    predict_hair_mask_image_from_face_roi,
)
from app.core.live_support import evaluate_supported_live_range, summarize_live_mask_quality
from app.core.recommender import recommend_hairstyles
from app.core.selfie_segmentation import (
    predict_selfie_head_region_mask,
    selfie_segmentation_available,
)
from app.core.tryon_2d_engine import (
    render_static_tryon,
    render_static_tryon_image,
    subject_hair_layout_bboxes,
)
from app.models.schemas import FaceAttributes, RecommendationPreferences


@dataclass
class LiveTryOnState:
    last_asset_ids: list[str] = field(default_factory=list)
    stable_asset_id: str | None = None
    stable_asset_hold_remaining: int = 0
    last_face_signature: tuple[str, ...] | None = None
    last_recommendations: list[Any] = field(default_factory=list)
    last_subject_hair_mask: Image.Image | None = None
    last_good_subject_hair_mask: Image.Image | None = None
    last_good_mask_source: str | None = None
    last_analysis: Any | None = None
    smoothed_subject_hair_bbox: dict[str, float] | None = None
    smoothed_subject_hair_top_bbox: dict[str, float] | None = None
    last_good_subject_hair_bbox: dict[str, int] | None = None
    last_good_subject_hair_top_bbox: dict[str, int] | None = None
    last_good_preview_image: Image.Image | None = None
    last_good_preview_path: str | None = None
    preview_grace_remaining: int = 0
    support_hold_remaining: int = 0
    last_support_summary: dict[str, Any] | None = None


def frame_to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    rgb = frame_bgr[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _face_signature(face_attributes: dict[str, str] | None) -> tuple[str, ...] | None:
    if not face_attributes:
        return None
    return (
        face_attributes.get("face_shape", "unknown"),
        face_attributes.get("forehead", "unknown"),
        face_attributes.get("jaw", "unknown"),
        face_attributes.get("cheekbone", "unknown"),
        face_attributes.get("face_width", "unknown"),
    )


def _smoothed_primary_asset_id(
    state: LiveTryOnState,
    recommendation_asset_ids: list[str],
    face_signature: tuple[str, ...] | None,
    stability_window: int = 4,
    hold_frames: int = 18,
) -> str | None:
    if not recommendation_asset_ids:
        state.last_asset_ids.clear()
        state.stable_asset_id = None
        state.stable_asset_hold_remaining = 0
        state.last_face_signature = face_signature
        return None

    top_asset_id = recommendation_asset_ids[0]
    if face_signature != state.last_face_signature:
        state.last_asset_ids.clear()
        state.stable_asset_hold_remaining = 0

    state.last_asset_ids.append(top_asset_id)
    state.last_asset_ids = state.last_asset_ids[-stability_window:]

    most_common_asset_id, frequency = Counter(state.last_asset_ids).most_common(1)[0]
    required_frequency = 2 if len(state.last_asset_ids) < stability_window else max(2, stability_window - 1)

    if state.stable_asset_id is not None and state.stable_asset_hold_remaining > 0:
        state.stable_asset_hold_remaining -= 1
    elif frequency >= required_frequency:
        if state.stable_asset_id != most_common_asset_id:
            state.stable_asset_id = most_common_asset_id
            state.stable_asset_hold_remaining = hold_frames
    elif state.stable_asset_id is None:
        state.stable_asset_id = top_asset_id
        state.stable_asset_hold_remaining = hold_frames

    state.last_face_signature = face_signature
    return state.stable_asset_id


def _smooth_bbox(
    previous: dict[str, float] | None,
    current: dict[str, int] | None,
    *,
    alpha: float,
) -> dict[str, float] | None:
    if current is None:
        return previous
    if previous is None:
        return {key: float(value) for key, value in current.items()}

    return {
        key: (previous[key] * (1.0 - alpha)) + (float(current[key]) * alpha)
        for key in ("x", "y", "width", "height")
    }


def _rounded_bbox(bbox: dict[str, float] | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    return {
        "x": int(round(bbox["x"])),
        "y": int(round(bbox["y"])),
        "width": max(int(round(bbox["width"])), 1),
        "height": max(int(round(bbox["height"])), 1),
    }


def _mask_is_reliable(
    analysis: Any,
    subject_hair_bbox: dict[str, int] | None,
    subject_hair_top_bbox: dict[str, int] | None,
) -> bool:
    face_bbox = analysis.face_bbox
    if face_bbox is None or subject_hair_bbox is None or subject_hair_top_bbox is None:
        return False

    face_width = max(int(face_bbox.get("width", 0) or 0), 1)
    face_height = max(int(face_bbox.get("height", 0) or 0), 1)
    face_center_x = face_bbox["x"] + (face_width / 2)
    hair_center_x = subject_hair_top_bbox["x"] + (subject_hair_top_bbox["width"] / 2)

    width_ratio = subject_hair_top_bbox["width"] / face_width
    height_ratio = subject_hair_bbox["height"] / face_height
    center_offset_ratio = abs(hair_center_x - face_center_x) / face_width
    top_offset_ratio = (subject_hair_top_bbox["y"] - face_bbox["y"]) / face_height

    if width_ratio < 0.34 or width_ratio > 1.35:
        return False
    if height_ratio < 0.18 or height_ratio > 1.2:
        return False
    if center_offset_ratio > 0.26:
        return False
    if top_offset_ratio < -0.42 or top_offset_ratio > 0.3:
        return False

    return True


def _face_geometry_is_stable(
    previous_analysis: Any | None,
    current_analysis: Any,
) -> bool:
    if previous_analysis is None:
        return False

    previous_bbox = getattr(previous_analysis, "face_bbox", None)
    current_bbox = getattr(current_analysis, "face_bbox", None)
    if previous_bbox is None or current_bbox is None:
        return False

    prev_width = max(int(previous_bbox.get("width", 0) or 0), 1)
    prev_height = max(int(previous_bbox.get("height", 0) or 0), 1)
    curr_width = max(int(current_bbox.get("width", 0) or 0), 1)
    curr_height = max(int(current_bbox.get("height", 0) or 0), 1)

    prev_center_x = previous_bbox["x"] + (prev_width / 2)
    prev_center_y = previous_bbox["y"] + (prev_height / 2)
    curr_center_x = current_bbox["x"] + (curr_width / 2)
    curr_center_y = current_bbox["y"] + (curr_height / 2)

    width_change = abs(curr_width - prev_width) / prev_width
    height_change = abs(curr_height - prev_height) / prev_height
    center_shift_x = abs(curr_center_x - prev_center_x) / prev_width
    center_shift_y = abs(curr_center_y - prev_center_y) / prev_height

    if width_change > 0.12 or height_change > 0.12:
        return False
    if center_shift_x > 0.10 or center_shift_y > 0.12:
        return False

    return True


def _build_live_replacement_mask(
    subject_hair_mask: Image.Image | None,
    analysis: Any,
    mask_source: str,
    source_image: Image.Image,
) -> Image.Image | None:
    if subject_hair_mask is None or analysis.face_bbox is None:
        return subject_hair_mask

    if "selfie" not in (mask_source or ""):
        return subject_hair_mask

    mask = subject_hair_mask.convert("L")
    mask_array = np.asarray(mask) > 0
    if not mask_array.any():
        return subject_hair_mask

    face_bbox = analysis.face_bbox
    width, height = mask.size
    face_x = int(face_bbox["x"])
    face_y = int(face_bbox["y"])
    face_width = max(int(face_bbox["width"]), 1)
    face_height = max(int(face_bbox["height"]), 1)

    landmarks = getattr(analysis, "landmarks", None) or []

    def _point(index: int) -> tuple[float, float] | None:
        if not landmarks or index >= len(landmarks):
            return None
        landmark = landmarks[index]
        return landmark.x * width, landmark.y * height

    left_temple = _point(127)
    right_temple = _point(356)
    forehead = _point(10)

    temple_center_x = face_x + (face_width / 2)
    temple_span = float(face_width)
    forehead_y = float(face_y)
    if left_temple is not None and right_temple is not None:
        temple_center_x = (left_temple[0] + right_temple[0]) / 2.0
        temple_span = max(abs(right_temple[0] - left_temple[0]), face_width * 0.62)
    if forehead is not None:
        forehead_y = forehead[1]

    x0 = max(int(temple_center_x - (temple_span * 0.66)), 0)
    x1 = min(int(temple_center_x + (temple_span * 0.66)), width)
    y0 = max(int(forehead_y - (face_height * 0.70)), 0)
    y1 = min(int(forehead_y + (face_height * 0.05)), height)

    constrained = np.zeros_like(mask_array, dtype=bool)
    constrained[y0:y1, x0:x1] = mask_array[y0:y1, x0:x1]
    if not constrained.any():
        return None

    # Refine the coarse support mask into a narrower forehead-to-crown band
    # centered on the temples so we do not replace broad side-face regions.
    y_coords, x_coords = np.where(constrained)
    top_y = int(y_coords.min())
    bottom_y = int(min(y_coords.max(), forehead_y + (face_height * 0.03)))
    vertical_span = max(bottom_y - top_y, 1)

    band = np.zeros_like(constrained, dtype=bool)
    min_half_width = max(int(face_width * 0.22), 1)
    max_half_width = max(int(temple_span * 0.54), min_half_width)
    for y in range(top_y, bottom_y + 1):
        t = (y - top_y) / vertical_span
        half_width = int(round(max_half_width * (1.0 - (0.42 * t))))
        half_width = max(half_width, min_half_width)
        row_left = max(int(round(temple_center_x - half_width)), 0)
        row_right = min(int(round(temple_center_x + half_width)), width)
        band[y, row_left:row_right] = True

    constrained = constrained & band
    if not constrained.any():
        constrained = band & mask_array
    if not constrained.any():
        return None

    # Keep the darker upper-head pixels inside that band.
    grayscale = np.asarray(source_image.convert("L"), dtype=np.float32)
    masked_values = grayscale[constrained]
    if masked_values.size == 0:
        refined = constrained
    else:
        darkness_threshold = float(np.percentile(masked_values, 68))
        dark_hair = grayscale <= darkness_threshold
        refined = constrained & dark_hair

    if refined.any():
        return clean_predicted_hair_mask(
            Image.fromarray((refined.astype(np.uint8) * 255), mode="L")
        )

    # If darkness-based refinement becomes too sparse, fall back to the narrow
    # temple/forehead band rather than the full head cap.
    fallback = np.zeros_like(mask_array, dtype=bool)
    fallback_y0 = top_y
    fallback_y1 = min(bottom_y, int(forehead_y))
    fallback[fallback_y0:fallback_y1, :] = constrained[fallback_y0:fallback_y1, :]
    if not fallback.any():
        fallback = constrained

    return clean_predicted_hair_mask(
        Image.fromarray((fallback.astype(np.uint8) * 255), mode="L")
    )


def process_live_frame(
    frame_bgr: np.ndarray,
    *,
    target_gender: str = "male",
    top_k: int = 3,
    state: LiveTryOnState | None = None,
    refresh_recommendations: bool = True,
    refresh_segmentation: bool = True,
) -> dict[str, Any]:
    preview_grace_frames = 12
    support_hold_frames = 18
    state = state or LiveTryOnState()
    pil_image = frame_to_pil_rgb(frame_bgr)
    analysis = analyze_face_pil_image(pil_image)
    support_summary = evaluate_supported_live_range(pil_image, analysis)

    payload: dict[str, Any] = {
        "face_detected": analysis.face_detected,
        "analysis": analysis,
        "recommendations": [],
        "stable_asset_id": None,
        "subject_hair_mask": None,
        "mask_reliable": False,
        "mask_source": "none",
        "used_mask_fallback": False,
        "used_preview_grace": False,
        "mask_status": "invalid",
        "preview_path": None,
        "preview_image": None,
        "frame_supported": support_summary["frame_supported"],
        "support_reason": support_summary["support_reason"],
        "support_summary": support_summary,
    }

    if not analysis.face_detected or not analysis.face_attributes:
        return payload

    if support_summary["frame_supported"]:
        state.support_hold_remaining = support_hold_frames
        state.last_support_summary = support_summary
    elif state.support_hold_remaining > 0:
        state.support_hold_remaining -= 1
    payload["frame_supported"] = support_summary["frame_supported"]
    payload["support_reason"] = support_summary["support_reason"]

    geometry_stable = _face_geometry_is_stable(state.last_analysis, analysis)
    allow_active_render = support_summary["frame_supported"] or state.support_hold_remaining > 0
    should_refresh_segmentation = (
        allow_active_render and (refresh_segmentation or state.last_good_subject_hair_mask is None)
    )

    fresh_subject_hair_mask = None
    fresh_subject_hair_bbox = None
    fresh_subject_hair_top_bbox = None
    mask_reliable = False
    fresh_mask_source = "none"

    if should_refresh_segmentation:
        roi_subject_hair_mask = predict_hair_mask_image_from_face_roi(
            pil_image,
            analysis.face_bbox,
        )
        roi_quality = summarize_live_mask_quality(analysis, roi_subject_hair_mask)

        if roi_quality["mask_reliable"]:
            fresh_subject_hair_mask = roi_subject_hair_mask
            fresh_subject_hair_bbox = roi_quality["subject_hair_bbox"]
            fresh_subject_hair_top_bbox = roi_quality["subject_hair_top_bbox"]
            mask_reliable = True
            fresh_mask_source = "roi"
        elif selfie_segmentation_available():
            selfie_subject_hair_mask = predict_selfie_head_region_mask(
                pil_image,
                analysis.face_bbox,
            )
            selfie_quality = summarize_live_mask_quality(analysis, selfie_subject_hair_mask)
            if selfie_quality["mask_reliable"]:
                fresh_subject_hair_mask = selfie_subject_hair_mask
                fresh_subject_hair_bbox = selfie_quality["subject_hair_bbox"]
                fresh_subject_hair_top_bbox = selfie_quality["subject_hair_top_bbox"]
                mask_reliable = True
                fresh_mask_source = "selfie"
            else:
                fresh_subject_hair_mask = roi_subject_hair_mask
                fresh_subject_hair_bbox = roi_quality["subject_hair_bbox"]
                fresh_subject_hair_top_bbox = roi_quality["subject_hair_top_bbox"]
                fresh_mask_source = "roi_weak"
        else:
            fresh_subject_hair_mask = roi_subject_hair_mask
            fresh_subject_hair_bbox = roi_quality["subject_hair_bbox"]
            fresh_subject_hair_top_bbox = roi_quality["subject_hair_top_bbox"]
            fresh_mask_source = "roi_weak"

    if should_refresh_segmentation and mask_reliable:
        subject_hair_mask = fresh_subject_hair_mask
        subject_hair_bbox = fresh_subject_hair_bbox
        subject_hair_top_bbox = fresh_subject_hair_top_bbox
        state.last_good_subject_hair_mask = fresh_subject_hair_mask
        state.last_good_mask_source = fresh_mask_source
        state.last_good_subject_hair_bbox = subject_hair_bbox
        state.last_good_subject_hair_top_bbox = subject_hair_top_bbox
        payload["mask_status"] = "fresh"
        payload["mask_source"] = fresh_mask_source
    elif state.last_good_subject_hair_mask is not None and (geometry_stable or should_refresh_segmentation):
        subject_hair_mask = state.last_good_subject_hair_mask
        subject_hair_bbox = state.last_good_subject_hair_bbox
        subject_hair_top_bbox = state.last_good_subject_hair_top_bbox
        payload["used_mask_fallback"] = True
        payload["mask_status"] = "fallback" if should_refresh_segmentation else "held"
        payload["mask_source"] = f"fallback:{state.last_good_mask_source or 'unknown'}"
    else:
        subject_hair_mask = fresh_subject_hair_mask
        subject_hair_bbox = fresh_subject_hair_bbox
        subject_hair_top_bbox = fresh_subject_hair_top_bbox
        payload["mask_status"] = "invalid"
        payload["mask_source"] = fresh_mask_source

    if (refresh_recommendations and support_summary["frame_supported"]) or not state.last_recommendations:
        preferences = RecommendationPreferences(allow_bangs=True, target_gender=target_gender)
        face_attrs = FaceAttributes(**analysis.face_attributes)
        live_assets = load_live_top_tier_asset_bank()
        recommendation_response = recommend_hairstyles(
            face_attrs,
            preferences=preferences,
            top_k=top_k,
            candidate_assets=live_assets or None,
        )
        recommendations = recommendation_response.recommendations
        state.last_recommendations = recommendations
    else:
        recommendations = state.last_recommendations

    allowed_asset_ids = {item.asset_id for item in recommendations}
    if state.stable_asset_id is not None and state.stable_asset_id not in allowed_asset_ids:
        state.stable_asset_id = None
        state.stable_asset_hold_remaining = 0
        state.last_asset_ids.clear()

    payload["recommendations"] = recommendations
    payload["subject_hair_mask"] = subject_hair_mask
    payload["mask_reliable"] = mask_reliable
    state.last_subject_hair_mask = subject_hair_mask
    state.last_analysis = analysis

    if mask_reliable and subject_hair_bbox is not None:
        state.smoothed_subject_hair_bbox = _smooth_bbox(
            state.smoothed_subject_hair_bbox,
            subject_hair_bbox,
            alpha=0.24,
        )
    if mask_reliable and subject_hair_top_bbox is not None:
        state.smoothed_subject_hair_top_bbox = _smooth_bbox(
            state.smoothed_subject_hair_top_bbox,
            subject_hair_top_bbox,
            alpha=0.20,
        )

    if not mask_reliable and payload["used_mask_fallback"]:
        if state.last_good_subject_hair_bbox is not None and state.smoothed_subject_hair_bbox is None:
            state.smoothed_subject_hair_bbox = {key: float(value) for key, value in state.last_good_subject_hair_bbox.items()}
        if state.last_good_subject_hair_top_bbox is not None and state.smoothed_subject_hair_top_bbox is None:
            state.smoothed_subject_hair_top_bbox = {
                key: float(value) for key, value in state.last_good_subject_hair_top_bbox.items()
            }

    # MediaPipe support masks are great for live stability and blending, but
    # they are too coarse to drive the primary hairstyle fit. Only let the
    # tighter ROI hair mask influence placement hints.
    use_mask_layout_hint = (
        mask_reliable
        and fresh_mask_source == "roi"
        and subject_hair_bbox is not None
        and subject_hair_top_bbox is not None
    )
    layout_hint = {
        "subject_hair_bbox": _rounded_bbox(state.smoothed_subject_hair_bbox) if use_mask_layout_hint else None,
        "subject_hair_top_bbox": _rounded_bbox(state.smoothed_subject_hair_top_bbox) if use_mask_layout_hint else None,
    }

    face_signature = _face_signature(analysis.face_attributes)
    stable_asset_id = _smoothed_primary_asset_id(
        state,
        [item.asset_id for item in recommendations],
        face_signature,
        hold_frames=28,
    )
    payload["stable_asset_id"] = stable_asset_id

    if stable_asset_id is None:
        return payload

    if not allow_active_render:
        payload["mask_status"] = "unsupported"
        payload["mask_source"] = "unsupported"
        if state.last_good_preview_image is not None and state.preview_grace_remaining > 0:
            payload["preview_image"] = state.last_good_preview_image.copy()
            payload["preview_path"] = state.last_good_preview_path
            payload["used_preview_grace"] = True
            state.preview_grace_remaining -= 1
        return payload

    if payload["mask_status"] == "invalid":
        if state.last_good_preview_image is not None and state.preview_grace_remaining > 0:
            payload["preview_image"] = state.last_good_preview_image.copy()
            payload["preview_path"] = state.last_good_preview_path
            payload["used_preview_grace"] = True
            payload["mask_status"] = "grace"
            payload["mask_source"] = "grace"
            state.preview_grace_remaining -= 1
        return payload

    asset = get_asset_by_id(stable_asset_id)
    if asset is None:
        return payload

    render_subject_hair_mask = _build_live_replacement_mask(
        subject_hair_mask,
        analysis,
        str(payload.get("mask_source", "")),
        pil_image,
    )
    render_mask_source = payload.get("mask_source", "")
    if (
        isinstance(render_mask_source, str)
        and "selfie" in render_mask_source
        and "roi" not in render_mask_source
    ):
        render_subject_hair_mask = render_subject_hair_mask

    payload["subject_hair_mask"] = render_subject_hair_mask

    preview_image = render_static_tryon_image(
        pil_image,
        analysis,
        asset,
        subject_hair_mask=render_subject_hair_mask,
        layout_hint=layout_hint,
    )
    if preview_image is None:
        return payload

    preview_path = render_static_tryon(
        pil_image,
        analysis,
        asset,
        subject_hair_mask=render_subject_hair_mask,
        layout_hint=layout_hint,
    )
    payload["preview_image"] = preview_image
    payload["preview_path"] = preview_path
    state.last_good_preview_image = preview_image.copy()
    state.last_good_preview_path = str(preview_path) if preview_path is not None else None
    state.preview_grace_remaining = preview_grace_frames
    return payload
