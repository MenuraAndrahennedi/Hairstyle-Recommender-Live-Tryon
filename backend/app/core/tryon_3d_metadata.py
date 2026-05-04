from __future__ import annotations

import math
from typing import Any

from app.core.asset_bank import load_asset_bank
from app.models.schemas import FaceAnalysisResult


def _pixel_point(landmark: Any, image_width: int, image_height: int) -> tuple[float, float]:
    return landmark.x * image_width, landmark.y * image_height


def _average_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _landmark_point(face_analysis: FaceAnalysisResult, indices: list[int]) -> tuple[float, float]:
    return _average_point(
        [
            _pixel_point(face_analysis.landmarks[index], face_analysis.image_width, face_analysis.image_height)
            for index in indices
        ]
    )


def estimate_head_anchor_frame(face_analysis: FaceAnalysisResult) -> dict[str, float | dict[str, float] | str] | None:
    if not face_analysis.face_detected or not face_analysis.landmarks or not face_analysis.face_bbox:
        return None

    left_eye = _landmark_point(face_analysis, [33, 133, 159, 145])
    right_eye = _landmark_point(face_analysis, [362, 263, 386, 374])
    left_temple = _landmark_point(face_analysis, [127, 234])
    right_temple = _landmark_point(face_analysis, [356, 454])
    forehead = _landmark_point(face_analysis, [10])
    chin = _landmark_point(face_analysis, [152])
    nose_bridge = _landmark_point(face_analysis, [6, 168])

    eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    temple_center = ((left_temple[0] + right_temple[0]) / 2.0, (left_temple[1] + right_temple[1]) / 2.0)

    eye_distance = max(abs(right_eye[0] - left_eye[0]), 1.0)
    temple_span = max(abs(right_temple[0] - left_temple[0]), 1.0)
    face_height = max(chin[1] - forehead[1], 1.0)
    roll_radians = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

    face_center_x = face_analysis.face_bbox["x"] + (face_analysis.face_bbox["width"] / 2.0)
    face_center_y = face_analysis.face_bbox["y"] + (face_analysis.face_bbox["height"] / 2.0)

    yaw_proxy = (nose_bridge[0] - face_center_x) / max(face_analysis.face_bbox["width"], 1)
    pitch_proxy = ((eye_center[1] + forehead[1]) / 2.0 - face_center_y) / max(face_analysis.face_bbox["height"], 1)

    return {
        "anchor_strategy": "mediapipe_temples_forehead",
        "head_anchor": {
            "x": round(temple_center[0], 2),
            "y": round(forehead[1], 2),
        },
        "eye_center": {
            "x": round(eye_center[0], 2),
            "y": round(eye_center[1], 2),
        },
        "temple_center": {
            "x": round(temple_center[0], 2),
            "y": round(temple_center[1], 2),
        },
        "forehead_center": {
            "x": round(forehead[0], 2),
            "y": round(forehead[1], 2),
        },
        "eye_distance_px": round(eye_distance, 2),
        "temple_span_px": round(temple_span, 2),
        "face_height_px": round(face_height, 2),
        "roll_degrees": round(math.degrees(roll_radians), 3),
        "yaw_proxy": round(float(yaw_proxy), 4),
        "pitch_proxy": round(float(pitch_proxy), 4),
        "lighting_mode": "estimated_from_input",
        "status": "lightweight_head_anchor",
    }


def build_3d_tryon_candidate_metadata(
    asset: Any,
    face_analysis: FaceAnalysisResult,
) -> dict[str, object] | None:
    anchor_frame = estimate_head_anchor_frame(face_analysis)
    if anchor_frame is None:
        return None

    normalized = asset.normalized_attributes
    style = normalized.style_family
    length = normalized.length
    volume = normalized.volume

    depth_bias = {
        "pompadour": 0.18,
        "regent": 0.16,
        "side_part": 0.10,
        "curly_crop": 0.09,
        "long_layered": 0.04,
        "bob": 0.02,
        "other": 0.08,
    }.get(style, 0.08)

    side_fall_bias = {
        "exposed": 0.15,
        "framed": 0.35,
        "covered": 0.55,
    }.get(normalized.side_hair, 0.25)

    volume_scale = {
        "low": 0.92,
        "medium": 1.0,
        "high": 1.12,
    }.get(volume, 1.0)

    length_scale = {
        "short": 0.94,
        "medium": 1.0,
        "long": 1.1,
    }.get(length, 1.0)

    return {
        "asset_id": asset.asset_id,
        "source_dataset": getattr(asset, "source_dataset", "runtime_asset_bank"),
        "gender_suitability": getattr(asset, "gender_suitability", None),
        "normalized_attributes": normalized.model_dump(),
        "head_anchor_frame": anchor_frame,
        "placement_priors": {
            "scale_multiplier": round(volume_scale * length_scale, 3),
            "depth_bias": round(depth_bias, 3),
            "side_fall_bias": round(side_fall_bias, 3),
            "rotation_follows_roll": True,
            "yaw_reactive": True,
        },
        "status": "lightweight_3d_candidate",
    }


def build_top_3d_candidate_metadata(
    face_analysis: FaceAnalysisResult,
    limit: int = 12,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for asset in load_asset_bank()[:limit]:
        candidate = build_3d_tryon_candidate_metadata(asset, face_analysis)
        if candidate is not None:
            rows.append(candidate)
    return rows


def get_default_tryon_3d_metadata() -> dict[str, object]:
    return {
        "anchor_strategy": "mediapipe_temples_forehead",
        "head_anchor": "temple_center_forehead",
        "lighting_mode": "estimated_from_input",
        "status": "lightweight_head_anchor",
    }
