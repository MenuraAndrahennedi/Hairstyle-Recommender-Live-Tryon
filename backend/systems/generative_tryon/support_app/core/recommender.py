from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from support_app.config import FACE_TO_HAIR_MAPPER_CONFIG_PATH, PROJECT_ROOT, media_url_for_path
from support_app.core.asset_bank import load_asset_bank
from support_app.ml.face_to_hair_mapper import load_mapper_payload, predict_hair_labels_from_candidates
from support_app.models.schemas import (
    AssetMetadata,
    FaceAttributes,
    RecommendationItem,
    RecommendationPreferences,
    RecommendationResponse,
)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reason: str


FACE_SHAPE_STYLE_RULES = {
    "round": {
        "prefer_lengths": {"medium", "long"},
        "prefer_styles": {"long_layered", "side_part", "long_wave", "pompadour"},
        "prefer_curls": {"wavy", "curly"},
        "avoid_bangs": {"full"},
        "reason": "adds vertical balance and reduces facial roundness",
    },
    "long": {
        "prefer_lengths": {"medium"},
        "prefer_styles": {"bob", "long_layered", "long_wave", "side_part"},
        "prefer_curls": {"wavy", "curly", "coily"},
        "avoid_bangs": set(),
        "reason": "adds side volume and softens facial length",
    },
    "square": {
        "prefer_lengths": {"medium", "long"},
        "prefer_styles": {"long_layered", "side_part", "long_wave", "pompadour"},
        "prefer_curls": {"wavy", "curly"},
        "avoid_bangs": set(),
        "reason": "softens stronger jaw lines with movement and layering",
    },
    "heart": {
        "prefer_lengths": {"medium", "short"},
        "prefer_styles": {"bob", "long_layered", "side_part", "regent"},
        "prefer_curls": {"wavy"},
        "avoid_bangs": set(),
        "reason": "balances a wider upper face with softer lower framing",
    },
    "oval": {
        "prefer_lengths": {"short", "medium", "long"},
        "prefer_styles": {"long_layered", "side_part", "bob", "long_wave", "pompadour", "regent", "curly_crop", "other"},
        "prefer_curls": {"straight", "wavy", "curly", "coily"},
        "avoid_bangs": set(),
        "reason": "works well with balanced facial proportions",
    },
    "unknown": {
        "prefer_lengths": {"short", "medium", "long"},
        "prefer_styles": {"side_part", "pompadour", "regent", "long_layered", "other"},
        "prefer_curls": {"straight", "wavy"},
        "avoid_bangs": set(),
        "reason": "is a safe style match while face geometry remains uncertain",
    },
}


MASCULINE_STYLE_FAMILIES = {"regent", "pompadour", "two_block"}
FEMININE_STYLE_FAMILIES = {"bob", "long_wave", "one_length", "layered"}


@lru_cache(maxsize=1)
def _load_mapper_payload_if_available() -> dict[str, object] | None:
    if not FACE_TO_HAIR_MAPPER_CONFIG_PATH.exists():
        return None
    try:
        return load_mapper_payload(FACE_TO_HAIR_MAPPER_CONFIG_PATH)
    except Exception:
        return None


def _positive_field_tokens(asset: AssetMetadata) -> set[str]:
    positive_fields = asset.translated_labels.get("positive_fields", "")
    return {field.strip().lower() for field in positive_fields.split(",") if field.strip()}


def _gender_evidence(asset: AssetMetadata) -> tuple[float, float]:
    attrs = asset.normalized_attributes
    tokens = _positive_field_tokens(asset)
    explicit_label = (asset.gender_suitability or "").strip().lower()

    male_score = 0.0
    female_score = 0.0

    if explicit_label == "male":
        male_score += 1.5
    elif explicit_label == "female":
        female_score += 1.5
    elif explicit_label == "neutral":
        male_score += 0.2
        female_score += 0.2

    if "male" in tokens:
        male_score += 2.5
    if "female" in tokens:
        female_score += 2.5

    if attrs.style_family in MASCULINE_STYLE_FAMILIES:
        male_score += 2.0
    if attrs.style_family in FEMININE_STYLE_FAMILIES:
        female_score += 2.0

    if attrs.style_family == "side_part":
        if attrs.length == "long":
            female_score += 1.0
        elif attrs.length in {"short", "medium"} and attrs.bang == "none":
            male_score += 1.5

    if attrs.style_family == "pixie":
        female_score += 1.0
        if "male" in tokens:
            male_score += 1.25

    if attrs.length == "long":
        female_score += 2.0
    elif attrs.length == "short":
        male_score += 0.8

    if attrs.bang not in {"none"}:
        female_score += 0.8
    else:
        male_score += 0.3

    if attrs.side_hair == "covered" and attrs.length in {"short", "medium"}:
        male_score += 0.4
    if attrs.side_hair == "exposed" and attrs.length == "long":
        female_score += 0.4

    return male_score, female_score


def infer_gender_suitability(asset: AssetMetadata) -> str:
    """Infer asset suitability from reviewed labels plus hairstyle evidence."""
    male_score, female_score = _gender_evidence(asset)
    if male_score >= female_score + 1.0:
        return "male"
    if female_score >= male_score + 1.0:
        return "female"
    return "neutral"


def _score_face_match(face: FaceAttributes, asset: AssetMetadata) -> ScoreResult:
    attrs = asset.normalized_attributes
    rules = FACE_SHAPE_STYLE_RULES.get(face.face_shape, FACE_SHAPE_STYLE_RULES["unknown"])
    score = 0.0
    reasons: list[str] = []

    if attrs.length in rules["prefer_lengths"]:
        score += 0.30
        reasons.append("length matches face-shape guidance")

    if attrs.style_family in rules["prefer_styles"]:
        score += 0.30
        reasons.append("style family suits the detected face shape")

    if attrs.curl in rules["prefer_curls"]:
        score += 0.15
        reasons.append("texture complements the current facial proportions")

    if face.forehead == "large" and attrs.bang in {"side", "see_through"}:
        score += 0.10
        reasons.append("bang styling can frame a larger forehead")
    elif face.forehead == "small" and attrs.bang == "none":
        score += 0.08
        reasons.append("open forehead styling avoids shortening the face")

    if face.jaw == "sharp" and attrs.curl in {"wavy", "curly"}:
        score += 0.08
        reasons.append("softer texture helps balance a sharper jaw")

    if face.face_width == "wide" and attrs.style_family in {"long_layered", "side_part", "pompadour"}:
        score += 0.07
        reasons.append("shape adds vertical structure for a wider face")

    if attrs.bang in rules["avoid_bangs"]:
        score -= 0.18
        reasons.append("bang type can work against the detected face shape")

    if not reasons:
        reasons.append(rules["reason"])

    return ScoreResult(score=max(score, 0.0), reason=f"{attrs.style_family.replace('_', ' ')} {rules['reason']}.")


def _score_preferences(
    preferences: RecommendationPreferences | None,
    asset: AssetMetadata,
) -> float:
    if preferences is None:
        return 0.0

    attrs = asset.normalized_attributes
    score = 0.0

    if preferences.preferred_length and attrs.length == preferences.preferred_length:
        score += 0.08
    if preferences.preferred_color and attrs.color == preferences.preferred_color:
        score += 0.05
    if preferences.preferred_style_family and attrs.style_family == preferences.preferred_style_family:
        score += 0.08
    if not preferences.allow_bangs and attrs.bang not in {"none"}:
        score -= 0.12

    return score


def _score_gender_preference(
    preferences: RecommendationPreferences | None,
    asset: AssetMetadata,
) -> float:
    if preferences is None:
        return 0.0

    target_gender = (preferences.target_gender or "any").lower()
    if target_gender not in {"male", "female"}:
        return 0.0

    suitability = infer_gender_suitability(asset)
    if suitability == target_gender:
        return 0.40
    if suitability == "neutral":
        return -0.05
    return -0.65


def _target_gender(preferences: RecommendationPreferences | None) -> str:
    if preferences is None:
        return "any"
    target_gender = (preferences.target_gender or "any").strip().lower()
    if target_gender not in {"male", "female"}:
        return "any"
    return target_gender


def _filter_assets_by_gender(
    assets: list[AssetMetadata],
    preferences: RecommendationPreferences | None,
) -> list[AssetMetadata]:
    target_gender = _target_gender(preferences)
    if target_gender == "any":
        return assets

    strict_matches = [asset for asset in assets if infer_gender_suitability(asset) == target_gender]
    if strict_matches:
        return strict_matches

    neutral_fallback = [asset for asset in assets if infer_gender_suitability(asset) == "neutral"]
    if neutral_fallback:
        return neutral_fallback

    return assets


def _score_quality(asset: AssetMetadata) -> float:
    score = 0.0
    attrs = asset.normalized_attributes
    horizontal = abs(asset.quality.horizontal)

    if asset.quality.front:
        score += 0.18
    else:
        score -= 0.08

    if asset.quality.vertical == "center":
        score += 0.12
    elif asset.quality.vertical == "upper":
        score += 0.04
    else:
        score -= 0.06

    if horizontal <= 6:
        score += 0.12
    elif horizontal <= 12:
        score += 0.06
    elif horizontal <= 20:
        score += 0.02
    else:
        score -= 0.08

    if attrs.length == "long" and not asset.quality.front:
        score -= 0.08
    if attrs.style_family in {"layered", "long_wave", "one_length"} and asset.quality.vertical != "center":
        score -= 0.06
    if attrs.style_family in {"regent", "pompadour", "side_part"} and horizontal <= 8:
        score += 0.04

    return score


@lru_cache(maxsize=512)
def _asset_mask_profile(mask_path: str) -> dict[str, float]:
    resolved = Path(mask_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / mask_path

    with Image.open(resolved) as mask_image:
        mask_array = np.asarray(mask_image.convert("L")) > 0
        if not mask_array.any():
            return {
                "aspect_ratio": 1.0,
                "coverage": 0.0,
                "bottom_ratio": 0.0,
                "side_ratio": 0.0,
            }

        y_coords, x_coords = np.where(mask_array)
        min_x = int(x_coords.min())
        max_x = int(x_coords.max())
        min_y = int(y_coords.min())
        max_y = int(y_coords.max())
        width = max(max_x - min_x + 1, 1)
        height = max(max_y - min_y + 1, 1)
        canvas_width = mask_array.shape[1]
        canvas_height = mask_array.shape[0]

        return {
            "aspect_ratio": height / width,
            "coverage": float(mask_array.mean()),
            "bottom_ratio": max_y / max(canvas_height - 1, 1),
            "side_ratio": min(min_x, canvas_width - 1 - max_x) / max(canvas_width, 1),
        }


def _score_tryon_suitability(
    preferences: RecommendationPreferences | None,
    asset: AssetMetadata,
) -> float:
    profile = _asset_mask_profile(asset.mask_path)
    attrs = asset.normalized_attributes
    target_gender = _target_gender(preferences)

    score = 0.0

    if profile["aspect_ratio"] <= 1.15:
        score += 0.08
    elif profile["aspect_ratio"] <= 1.35:
        score += 0.03
    elif profile["aspect_ratio"] <= 1.55:
        score -= 0.06
    else:
        score -= 0.18

    if profile["bottom_ratio"] >= 0.95:
        score -= 0.08
    elif profile["bottom_ratio"] >= 0.88:
        score -= 0.04

    if profile["coverage"] >= 0.58:
        score -= 0.08
    elif profile["coverage"] <= 0.12:
        score -= 0.05

    if profile["side_ratio"] <= 0.03:
        score -= 0.04

    if target_gender == "male":
        if attrs.length == "long":
            score -= 0.16
        if profile["aspect_ratio"] > 1.35:
            score -= 0.12
    elif target_gender == "female":
        if attrs.length == "long" and profile["aspect_ratio"] <= 1.45:
            score += 0.04

    if attrs.style_family in {"pompadour", "side_part", "regent"} and profile["aspect_ratio"] <= 1.2:
        score += 0.05
    if attrs.style_family in {"long_wave", "one_length"} and profile["aspect_ratio"] > 1.55:
        score -= 0.08

    return score


def _score_style_clarity(
    preferences: RecommendationPreferences | None,
    asset: AssetMetadata,
) -> float:
    attrs = asset.normalized_attributes
    target_gender = _target_gender(preferences)
    score = 0.0

    if attrs.style_family == "other":
        score -= 0.14

    if target_gender == "male":
        if attrs.style_family in {"bob", "long_layered", "long_wave", "one_length"}:
            score -= 0.22
        if attrs.length == "long":
            score -= 0.12
        if attrs.bang == "full":
            score -= 0.08
    elif target_gender == "female":
        if attrs.style_family in {"regent", "pompadour", "two_block"}:
            score -= 0.16

    return score


def _display_scores_from_raw(raw_scores: list[float]) -> list[float]:
    if not raw_scores:
        return []
    if len(raw_scores) == 1:
        return [0.92]

    max_score = max(raw_scores)
    min_score = min(raw_scores)
    spread = max(max_score - min_score, 1e-6)
    display_scores: list[float] = []

    for rank_index, raw_score in enumerate(raw_scores):
        normalized = (raw_score - min_score) / spread
        rank_penalty = rank_index * 0.015
        display = 0.58 + 0.38 * normalized - rank_penalty
        display_scores.append(round(min(max(display, 0.05), 0.98), 4))

    return display_scores


def _mapper_face_labels(
    face: FaceAttributes,
    preferences: RecommendationPreferences | None,
) -> dict[str, str]:
    labels: dict[str, str] = {}

    target_gender = _target_gender(preferences)
    if target_gender in {"male", "female"}:
        labels["gender"] = target_gender

    if face.face_width == "wide":
        labels["face_fullness"] = "full"
    elif face.face_width in {"medium", "narrow"}:
        labels["face_fullness"] = "slim"

    labels["cheekbones"] = "high" if face.cheekbone == "high" else "soft"
    labels["hairline"] = "receding" if face.forehead == "large" else "regular"
    return labels


def _score_mapper_alignment(
    face: FaceAttributes,
    preferences: RecommendationPreferences | None,
    asset: AssetMetadata,
    candidate_assets: list[AssetMetadata],
) -> tuple[float, str | None]:
    mapper_payload = _load_mapper_payload_if_available()
    if mapper_payload is None:
        return 0.0, None

    face_labels = _mapper_face_labels(face, preferences)
    if not face_labels:
        return 0.0, None

    candidate_rows = [
        {
            "asset_id": item.asset_id,
            "gender_label": infer_gender_suitability(item),
            "labeling": {
                "normalized_attributes": {
                    "length": item.normalized_attributes.length,
                    "curl": item.normalized_attributes.curl,
                    "style_family": item.normalized_attributes.style_family,
                }
            },
        }
        for item in candidate_assets
    ]
    target_predictions = predict_hair_labels_from_candidates(face_labels, mapper_payload, candidate_rows)

    score = 0.0
    matched_fields: list[str] = []
    if target_predictions.get("length", {}).get("label") == asset.normalized_attributes.length:
        score += 0.14
        matched_fields.append("length")
    if target_predictions.get("curl", {}).get("label") == asset.normalized_attributes.curl:
        score += 0.10
        matched_fields.append("texture")
    if target_predictions.get("style_family", {}).get("label") == asset.normalized_attributes.style_family:
        score += 0.16
        matched_fields.append("style family")

    if not matched_fields:
        return 0.0, None

    if len(matched_fields) == 1:
        reason = f"Mapper alignment: {matched_fields[0]} matches the lightweight face-to-hair profile."
    else:
        reason = (
            "Mapper alignment: "
            + ", ".join(matched_fields[:-1])
            + f" and {matched_fields[-1]} match the lightweight face-to-hair profile."
        )
    return score, reason


def recommend_hairstyles(
    face_attributes: FaceAttributes,
    preferences: RecommendationPreferences | None = None,
    top_k: int = 5,
    candidate_assets: list[AssetMetadata] | None = None,
) -> RecommendationResponse:
    scored_rows: list[tuple[float, RecommendationItem]] = []
    asset_pool = load_asset_bank() if candidate_assets is None else candidate_assets
    candidate_assets = _filter_assets_by_gender(asset_pool, preferences)

    for asset in candidate_assets:
        face_match = _score_face_match(face_attributes, asset)
        preference_score = _score_preferences(preferences, asset)
        gender_score = _score_gender_preference(preferences, asset)
        quality_score = _score_quality(asset)
        tryon_score = _score_tryon_suitability(preferences, asset)
        style_clarity_score = _score_style_clarity(preferences, asset)
        mapper_score, mapper_reason = _score_mapper_alignment(
            face_attributes,
            preferences,
            asset,
            candidate_assets,
        )
        raw_score = max(
            face_match.score
            + preference_score
            + gender_score
            + quality_score
            + tryon_score
            + style_clarity_score
            + mapper_score,
            0.0,
        )
        reason = face_match.reason if mapper_reason is None else f"{face_match.reason} {mapper_reason}"

        scored_rows.append(
            (
                raw_score,
                RecommendationItem(
                    asset_id=asset.asset_id,
                    score=0.0,
                    reason=reason,
                    gender_suitability=infer_gender_suitability(asset),
                    image_path=asset.image_path,
                    mask_path=asset.mask_path,
                    image_url=media_url_for_path(asset.image_path),
                    mask_url=media_url_for_path(asset.mask_path),
                    normalized_attributes=asset.normalized_attributes,
                ),
            )
        )

    scored_rows.sort(key=lambda row: row[0], reverse=True)
    display_scores = _display_scores_from_raw([row[0] for row in scored_rows])

    ranked: list[RecommendationItem] = []
    for display_score, (_, item) in zip(display_scores, scored_rows):
        item.score = display_score
        ranked.append(item)

    return RecommendationResponse(recommendations=ranked[: max(1, top_k)])

