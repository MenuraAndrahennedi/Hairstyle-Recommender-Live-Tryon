from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from manual_app.config import (
    FULL_HAIR_ASSET_METADATA_DIR,
    REVIEWED_ASSET_BANK_JSONL,
    REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL,
)
from manual_app.models.schemas import AssetMetadata


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_optional_label(value: object, fallback: str = "none") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan"}:
        return fallback
    return text


def _reviewed_row_to_asset_metadata(row: dict) -> AssetMetadata:
    normalized = row.get("labeling", {}).get("normalized_attributes", {})
    celeba_hints = row.get("celeba_attribute_hints", {})

    translated_labels = {
        "taxonomy_source": str(row.get("labeling", {}).get("taxonomy_source", "manual_review")),
        "source_celeba_file": str(row.get("original_celeba_file", "")),
        "color_hint": str(celeba_hints.get("color_hint") or ""),
        "texture_hint": str(celeba_hints.get("texture_hint") or ""),
        "positive_fields": ", ".join(celeba_hints.get("positive_fields", [])),
        "notes": str(row.get("labeling", {}).get("notes") or row.get("review", {}).get("notes") or ""),
    }

    payload = {
        "asset_id": row["asset_id"],
        "source_dataset": row.get("source_dataset", "CelebA"),
        "gender_suitability": row.get("gender_label"),
        "raw_label_path": row.get("raw_mask_path", ""),
        "raw_image_path": row.get("raw_image_path", ""),
        "image_path": row.get("image_path", ""),
        "mask_path": row.get("mask_path", ""),
        "translated_labels": translated_labels,
        "normalized_attributes": {
            "length": _normalize_optional_label(normalized.get("length"), fallback="medium"),
            "curl": _normalize_optional_label(normalized.get("curl"), fallback="straight"),
            "bang": _normalize_optional_label(normalized.get("bang"), fallback="none"),
            "volume": _normalize_optional_label(normalized.get("volume"), fallback="medium"),
            "side_hair": _normalize_optional_label(normalized.get("side_hair"), fallback="covered"),
            "color": _normalize_optional_label(normalized.get("color"), fallback="black"),
            "style_family": _normalize_optional_label(normalized.get("style_family"), fallback="other"),
        },
        "quality": {
            "front": True,
            "horizontal": 0,
            "vertical": "center",
        },
    }
    return AssetMetadata.model_validate(payload)


def _color_from_celeba_hint(value: object) -> str:
    hint = str(value or "").strip().lower()
    mapping = {
        "black_hair": "black",
        "blond_hair": "blonde",
        "brown_hair": "brown",
        "gray_hair": "gray",
    }
    return mapping.get(hint, "black")


def _curl_from_celeba_hint(value: object) -> str:
    hint = str(value or "").strip().lower()
    mapping = {
        "straight_hair": "straight",
        "wavy_hair": "wavy",
        "curly_hair": "curly",
    }
    return mapping.get(hint, "straight")


def _length_from_mask_stats(stats: dict) -> str:
    width = max(int(stats.get("bbox_width", 0) or 0), 1)
    height = max(int(stats.get("bbox_height", 0) or 0), 1)
    aspect_ratio = height / width
    coverage = float(stats.get("coverage_ratio", 0.0) or 0.0)

    if aspect_ratio >= 1.25 or coverage >= 0.24:
        return "long"
    if aspect_ratio >= 0.95 or coverage >= 0.16:
        return "medium"
    return "short"


def _aspect_ratio_from_mask_stats(stats: dict) -> float:
    width = max(int(stats.get("bbox_width", 0) or 0), 1)
    height = max(int(stats.get("bbox_height", 0) or 0), 1)
    return height / width


def _volume_from_full_hair_clues(stats: dict, shape_stats: dict) -> str:
    coverage = float(stats.get("coverage_ratio", 0.0) or 0.0)
    top_band_coverage = float(shape_stats.get("top_band_coverage", 0.0) or 0.0)
    top_band_span_ratio = float(shape_stats.get("top_band_span_ratio", 0.0) or 0.0)

    if coverage >= 0.30 or top_band_coverage >= 0.24 or top_band_span_ratio >= 0.58:
        return "high"
    if coverage <= 0.18 and top_band_coverage <= 0.10:
        return "low"
    return "medium"


def _side_hair_from_full_hair_clues(stats: dict, shape_stats: dict, length: str) -> str:
    aspect_ratio = _aspect_ratio_from_mask_stats(stats)
    outer_side_ratio = float(shape_stats.get("outer_side_ratio", 0.0) or 0.0)

    if length == "long" or aspect_ratio >= 1.2 or outer_side_ratio >= 0.16:
        return "covered"
    return "exposed"


def _style_family_from_seed_labels(
    gender_label: str,
    length: str,
    curl: str,
    bangs_hint: bool,
) -> str:
    gender = (gender_label or "").strip().lower()
    if length == "short" and gender == "male" and curl == "straight":
        return "side_part"
    if length == "short" and curl == "curly":
        return "curly_crop"
    if length == "medium" and bangs_hint:
        return "bob"
    if length == "long" and curl == "wavy":
        return "long_layered"
    return "other"


def _style_family_from_full_hair_clues(
    gender_label: str,
    length: str,
    curl: str,
    bangs_hint: bool,
    stats: dict,
    shape_stats: dict,
) -> str:
    gender = (gender_label or "").strip().lower()
    top_band_coverage = float(shape_stats.get("top_band_coverage", 0.0) or 0.0)
    top_band_span_ratio = float(shape_stats.get("top_band_span_ratio", 0.0) or 0.0)
    side_bias_ratio = float(shape_stats.get("side_bias_ratio", 0.0) or 0.0)
    outer_side_ratio = float(shape_stats.get("outer_side_ratio", 0.0) or 0.0)
    aspect_ratio = _aspect_ratio_from_mask_stats(stats)

    if gender == "male":
        if curl in {"wavy", "curly"} and length == "short":
            return "curly_crop"
        if top_band_coverage >= 0.22 and top_band_span_ratio >= 0.48 and outer_side_ratio <= 0.12:
            return "pompadour"
        if top_band_span_ratio <= 0.46 and side_bias_ratio <= 0.16 and outer_side_ratio <= 0.10:
            return "regent"
        if bangs_hint or side_bias_ratio >= 0.16 or outer_side_ratio >= 0.12 or length == "medium":
            return "side_part"
        return "other"

    if length == "long" and (curl in {"wavy", "curly"} or outer_side_ratio >= 0.18):
        return "long_layered"
    if length in {"medium", "long"} and curl == "straight" and outer_side_ratio <= 0.12 and side_bias_ratio <= 0.16:
        return "one_length"
    if bangs_hint and top_band_span_ratio >= 0.44:
        return "bob"
    if length == "short" and bangs_hint:
        return "bob"
    if aspect_ratio >= 1.28 and outer_side_ratio >= 0.16:
        return "long_layered"
    return "other"


def _full_hair_row_to_asset_metadata(row: dict, infer_style_family: bool = False) -> AssetMetadata:
    normalized = row.get("labeling", {}).get("normalized_attributes", {})
    stats = row.get("hair_mask_stats", {})
    shape_stats = row.get("mask_shape_stats", {})
    hints = row.get("celeba_attribute_hints", {})

    length = _normalize_optional_label(normalized.get("length"), fallback=_length_from_mask_stats(stats))
    curl = _normalize_optional_label(normalized.get("curl"), fallback=_curl_from_celeba_hint(hints.get("texture_hint")))
    color = _normalize_optional_label(normalized.get("color"), fallback=_color_from_celeba_hint(hints.get("color_hint")))
    bangs_hint = bool(hints.get("bangs_hint", False))

    translated_labels = {
        "taxonomy_source": str(row.get("labeling", {}).get("taxonomy_source", "CelebA_native_attributes")),
        "source_celeba_file": str(row.get("original_celeba_file", "")),
        "color_hint": str(hints.get("color_hint") or ""),
        "texture_hint": str(hints.get("texture_hint") or ""),
        "positive_fields": ", ".join(hints.get("positive_fields", [])),
        "notes": str(row.get("labeling", {}).get("notes") or ""),
        "confidence_bucket": str(row.get("confidence_bucket") or ""),
        "full_hair_score": str(row.get("full_hair_score") or ""),
    }

    inferred_style_family = _style_family_from_full_hair_clues(
        row.get("gender_label", ""),
        length,
        curl,
        bangs_hint,
        stats,
        shape_stats,
    )
    style_family_fallback = inferred_style_family if infer_style_family else "other"

    payload = {
        "asset_id": row["asset_id"],
        "source_dataset": row.get("source_dataset", "CelebA_full_hair"),
        "gender_suitability": row.get("gender_label"),
        "raw_label_path": row.get("raw_mask_path", ""),
        "raw_image_path": row.get("raw_image_path", ""),
        "image_path": row.get("image_path", ""),
        "mask_path": row.get("mask_path", ""),
        "translated_labels": translated_labels,
        "normalized_attributes": {
            "length": length,
            "curl": curl,
            "bang": _normalize_optional_label(normalized.get("bang"), fallback="full" if bangs_hint else "none"),
            "volume": _normalize_optional_label(
                normalized.get("volume"),
                fallback=_volume_from_full_hair_clues(stats, shape_stats),
            ),
            "side_hair": _normalize_optional_label(
                normalized.get("side_hair"),
                fallback=_side_hair_from_full_hair_clues(stats, shape_stats, length),
            ),
            "color": color,
            "style_family": _normalize_optional_label(
                normalized.get("style_family"),
                fallback=style_family_fallback,
            ),
        },
        "quality": {
            "front": True,
            "horizontal": 0,
            "vertical": "center",
        },
    }
    return AssetMetadata.model_validate(payload)


def list_full_hair_asset_metadata_paths() -> List[Path]:
    if not FULL_HAIR_ASSET_METADATA_DIR.exists():
        return []
    return sorted(FULL_HAIR_ASSET_METADATA_DIR.glob("*.json"))


def load_full_hair_asset_bank() -> List[AssetMetadata]:
    assets: List[AssetMetadata] = []
    for path in list_full_hair_asset_metadata_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assets.append(_full_hair_row_to_asset_metadata(payload, infer_style_family=False))
    return assets


def load_reviewed_render_safe_asset_bank() -> List[AssetMetadata]:
    if not REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL.exists():
        return []
    reviewed_rows = _read_jsonl(REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL)
    return [_full_hair_row_to_asset_metadata(row, infer_style_family=False) for row in reviewed_rows]


def _is_live_top_tier_row(row: dict) -> bool:
    if str(row.get("confidence_bucket", "")).strip().lower() != "high":
        return False

    try:
        quality_score = float(row.get("quality_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        quality_score = 0.0
    if quality_score < 0.99:
        return False

    try:
        full_hair_score = float(row.get("full_hair_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        full_hair_score = 0.0
    if full_hair_score < 0.95:
        return False

    stats = row.get("hair_mask_stats", {}) or {}
    topology = row.get("mask_topology_stats", {}) or {}
    shape = row.get("mask_shape_stats", {}) or {}
    normalized = row.get("labeling", {}).get("normalized_attributes", {}) or {}

    bbox_width = max(int(stats.get("bbox_width", 0) or 0), 1)
    bbox_height = max(int(stats.get("bbox_height", 0) or 0), 1)
    aspect_ratio = bbox_height / bbox_width
    coverage = float(stats.get("coverage_ratio", 0.0) or 0.0)

    component_count = int(topology.get("component_count", 0) or 0)
    stray_ratio = float(topology.get("stray_ratio", 0.0) or 0.0)

    top_band_coverage = float(shape.get("top_band_coverage", 0.0) or 0.0)
    top_band_span_ratio = float(shape.get("top_band_span_ratio", 0.0) or 0.0)
    side_bias_ratio = float(shape.get("side_bias_ratio", 0.0) or 0.0)
    outer_side_ratio = float(shape.get("outer_side_ratio", 0.0) or 0.0)
    top_rows_with_pixels = int(shape.get("top_rows_with_pixels", 0) or 0)

    length = _normalize_optional_label(normalized.get("length"), fallback=_length_from_mask_stats(stats))
    style_family = _normalize_optional_label(
        normalized.get("style_family"),
        fallback=_style_family_from_full_hair_clues(
            row.get("gender_label", ""),
            length,
            _normalize_optional_label(normalized.get("curl"), fallback=_curl_from_celeba_hint(
                row.get("celeba_attribute_hints", {}).get("texture_hint")
            )),
            bool(row.get("celeba_attribute_hints", {}).get("bangs_hint", False)),
            stats,
            shape,
        ),
    )

    if component_count > 1 or stray_ratio > 0.01:
        return False
    if length == "long":
        return False
    if style_family in {"long_layered", "one_length", "bob"}:
        return False
    if not (0.10 <= coverage <= 0.30):
        return False
    if not (0.72 <= aspect_ratio <= 1.20):
        return False
    if not (0.12 <= top_band_coverage <= 0.26):
        return False
    if not (0.38 <= top_band_span_ratio <= 0.58):
        return False
    if side_bias_ratio > 0.18:
        return False
    if outer_side_ratio > 0.12:
        return False
    if top_rows_with_pixels < 12:
        return False

    return True


@lru_cache(maxsize=1)
def load_live_top_tier_asset_bank() -> List[AssetMetadata]:
    if not REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL.exists():
        return []
    reviewed_rows = _read_jsonl(REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL)
    filtered_rows = [row for row in reviewed_rows if _is_live_top_tier_row(row)]
    return [_full_hair_row_to_asset_metadata(row, infer_style_family=True) for row in filtered_rows]


def load_reviewed_asset_bank() -> List[AssetMetadata]:
    if not REVIEWED_ASSET_BANK_JSONL.exists():
        return []
    reviewed_rows = _read_jsonl(REVIEWED_ASSET_BANK_JSONL)
    return [_reviewed_row_to_asset_metadata(row) for row in reviewed_rows]


def load_asset_bank() -> List[AssetMetadata]:
    """Runtime try-on bank: prefer the reviewed render-safe live bank."""
    reviewed_render_safe_assets = load_reviewed_render_safe_asset_bank()
    if reviewed_render_safe_assets:
        return reviewed_render_safe_assets

    full_hair_assets = load_full_hair_asset_bank()
    if full_hair_assets:
        return full_hair_assets
    return load_reviewed_asset_bank()


def asset_bank_summary() -> dict:
    assets = load_asset_bank()
    if REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL.exists() and load_reviewed_render_safe_asset_bank():
        metadata_dir = str(REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL)
    elif FULL_HAIR_ASSET_METADATA_DIR.exists() and load_full_hair_asset_bank():
        metadata_dir = str(FULL_HAIR_ASSET_METADATA_DIR)
    else:
        metadata_dir = str(REVIEWED_ASSET_BANK_JSONL)
    return {
        "asset_count": len(assets),
        "metadata_dir": metadata_dir,
    }


def get_asset_by_id(asset_id: str) -> AssetMetadata | None:
    for asset in load_asset_bank():
        if asset.asset_id == asset_id:
            return asset
    return None

