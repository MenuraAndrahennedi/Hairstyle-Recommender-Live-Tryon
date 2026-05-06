from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


CELEBA_PARTITION_MAP = {
    0: "train",
    1: "val",
    2: "test",
}

CELEBA_HAIR_ATTRIBUTE_COLUMNS = [
    "Male",
    "Bald",
    "Bangs",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Straight_Hair",
    "Wavy_Hair",
    "Receding_Hairline",
    "Wearing_Hat",
]


def discover_celeba_image_dir(raw_celeba_root: str | Path) -> Path:
    root = Path(raw_celeba_root)
    candidates = [
        root / "img_align_celeba" / "img_align_celeba",
        root / "img_align_celeba",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate CelebA image directory inside: {root}")


def load_celeba_attributes(raw_celeba_root: str | Path) -> pd.DataFrame:
    csv_path = Path(raw_celeba_root) / "list_attr_celeba.csv"
    frame = pd.read_csv(csv_path)
    for column in frame.columns:
        if column != "image_id":
            frame[column] = frame[column].astype(int)
    return frame


def load_celeba_partitions(raw_celeba_root: str | Path) -> pd.DataFrame:
    csv_path = Path(raw_celeba_root) / "list_eval_partition.csv"
    frame = pd.read_csv(csv_path)
    frame["partition_name"] = frame["partition"].map(CELEBA_PARTITION_MAP)
    return frame


def build_celeba_seed_table(raw_celeba_root: str | Path) -> pd.DataFrame:
    attrs = load_celeba_attributes(raw_celeba_root)
    partitions = load_celeba_partitions(raw_celeba_root)
    merged = attrs.merge(partitions, on="image_id", how="inner")

    merged["has_visible_hair_signal"] = (
        (merged["Bald"] != 1)
        & (merged["Wearing_Hat"] != 1)
        & (
            (merged["Black_Hair"] == 1)
            | (merged["Blond_Hair"] == 1)
            | (merged["Brown_Hair"] == 1)
            | (merged["Gray_Hair"] == 1)
            | (merged["Straight_Hair"] == 1)
            | (merged["Wavy_Hair"] == 1)
            | (merged["Bangs"] == 1)
            | (merged["Receding_Hairline"] == 1)
        )
    )
    return merged


def select_lightweight_celeba_subset(
    seed_table: pd.DataFrame,
    max_train: int = 400,
    max_val: int = 100,
    max_test: int = 100,
) -> pd.DataFrame:
    filtered = seed_table[seed_table["has_visible_hair_signal"]].copy()
    limits = {"train": max_train, "val": max_val, "test": max_test}
    sampled_frames: list[pd.DataFrame] = []

    for partition_name, limit in limits.items():
        partition_rows = filtered[filtered["partition_name"] == partition_name].copy()
        sampled_frames.append(partition_rows.head(limit))

    subset = pd.concat(sampled_frames, ignore_index=True)
    subset["gender_label"] = subset["Male"].map({1: "male", -1: "female"})
    return subset


def image_path_for_row(raw_celeba_root: str | Path, image_id: str) -> Path:
    return discover_celeba_image_dir(raw_celeba_root) / image_id


def celeba_attribute_payload(row: pd.Series) -> dict[str, int]:
    payload: dict[str, int] = {}
    for column in CELEBA_HAIR_ATTRIBUTE_COLUMNS:
        payload[column] = int(row[column])
    return payload


def build_subset_record(raw_celeba_root: str | Path, row: pd.Series) -> dict[str, object]:
    image_path = image_path_for_row(raw_celeba_root, row["image_id"])
    gender_label = row["gender_label"] if "gender_label" in row.index else ("male" if int(row["Male"]) == 1 else "female")
    partition_name = row["partition_name"] if "partition_name" in row.index else CELEBA_PARTITION_MAP.get(int(row["partition"]), "train")
    return {
        "image_id": row["image_id"],
        "image_path": str(image_path),
        "partition": partition_name,
        "gender_label": gender_label,
        "celeba_attributes": celeba_attribute_payload(row),
    }


def mask_statistics(mask_image: Image.Image) -> dict[str, float | int]:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    total_pixels = int(mask_array.size)
    positive_pixels = int(mask_array.sum())
    if positive_pixels == 0:
        return {
            "positive_pixels": 0,
            "coverage_ratio": 0.0,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width": 0,
            "bbox_height": 0,
        }

    y_coords, x_coords = np.where(mask_array)
    min_x = int(x_coords.min())
    max_x = int(x_coords.max())
    min_y = int(y_coords.min())
    max_y = int(y_coords.max())
    return {
        "positive_pixels": positive_pixels,
        "coverage_ratio": round(positive_pixels / max(total_pixels, 1), 6),
        "bbox_x": min_x,
        "bbox_y": min_y,
        "bbox_width": max_x - min_x + 1,
        "bbox_height": max_y - min_y + 1,
    }


def _largest_component_mask(mask_array: np.ndarray) -> np.ndarray:
    binary = mask_array.astype(bool)
    if not binary.any():
        return binary

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    largest_coords: list[tuple[int, int]] = []
    largest_size = 0

    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        coords: list[tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(coords) > largest_size:
            largest_size = len(coords)
            largest_coords = coords

    cleaned = np.zeros_like(binary, dtype=bool)
    for y, x in largest_coords:
        cleaned[y, x] = True
    return cleaned


def _component_areas(mask_array: np.ndarray) -> list[int]:
    binary = mask_array.astype(bool)
    if not binary.any():
        return []

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    areas: list[int] = []

    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0

        while stack:
            y, x = stack.pop()
            area += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        areas.append(area)

    return sorted(areas, reverse=True)


def cleaned_mask_image(mask_image: Image.Image) -> Image.Image:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    cleaned_array = _largest_component_mask(mask_array)
    cleaned = Image.fromarray((cleaned_array.astype(np.uint8) * 255), mode="L")
    cleaned = cleaned.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    final_array = np.asarray(cleaned.convert("L")) > 127
    final_array = _largest_component_mask(final_array)
    return Image.fromarray((final_array.astype(np.uint8) * 255), mode="L")


def mask_topology_statistics(mask_image: Image.Image) -> dict[str, float | int]:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    total_pixels = int(mask_array.sum())
    component_areas = _component_areas(mask_array)
    component_count = len(component_areas)
    largest_area = int(component_areas[0]) if component_areas else 0
    secondary_area = int(component_areas[1]) if component_count > 1 else 0
    stray_area = int(sum(component_areas[1:])) if component_count > 1 else 0
    stray_ratio = round(stray_area / max(total_pixels, 1), 6) if total_pixels else 0.0
    return {
        "component_count": component_count,
        "largest_component_pixels": largest_area,
        "secondary_component_pixels": secondary_area,
        "stray_pixels": stray_area,
        "stray_ratio": stray_ratio,
    }


def mask_shape_statistics(mask_image: Image.Image) -> dict[str, float | int]:
    mask_array = np.asarray(mask_image.convert("L")) > 0
    if not mask_array.any():
        return {
            "top_band_coverage": 0.0,
            "top_band_span_ratio": 0.0,
            "side_bias_ratio": 0.0,
            "outer_side_ratio": 0.0,
            "top_rows_with_pixels": 0,
        }

    height, width = mask_array.shape
    top_band_height = max(int(height * 0.18), 12)
    top_band = mask_array[:top_band_height, :]
    top_band_pixels = int(top_band.sum())
    top_band_coverage = round(top_band_pixels / max(top_band.size, 1), 6)

    top_cols = np.where(top_band.any(axis=0))[0]
    if len(top_cols) == 0:
        top_span_ratio = 0.0
    else:
        top_span_ratio = round((int(top_cols.max()) - int(top_cols.min()) + 1) / max(width, 1), 6)

    left_half = int(mask_array[:, : width // 2].sum())
    right_half = int(mask_array[:, width // 2 :].sum())
    total_pixels = max(int(mask_array.sum()), 1)
    side_bias_ratio = round(abs(left_half - right_half) / total_pixels, 6)

    outer_band_width = max(int(width * 0.18), 10)
    outer_left = int(mask_array[:, :outer_band_width].sum())
    outer_right = int(mask_array[:, width - outer_band_width :].sum())
    outer_side_ratio = round((outer_left + outer_right) / total_pixels, 6)

    top_rows_with_pixels = int(np.count_nonzero(mask_array.any(axis=1)[:top_band_height]))

    return {
        "top_band_coverage": top_band_coverage,
        "top_band_span_ratio": top_span_ratio,
        "side_bias_ratio": side_bias_ratio,
        "outer_side_ratio": outer_side_ratio,
        "top_rows_with_pixels": top_rows_with_pixels,
    }


def build_hair_rich_record(
    raw_celeba_root: str | Path,
    row: pd.Series,
    segmentation_mask_path: str | Path,
) -> dict[str, object]:
    mask_path = Path(segmentation_mask_path)
    with Image.open(mask_path) as mask_image:
        stats = mask_statistics(mask_image)

    return {
        **build_subset_record(raw_celeba_root, row),
        "segmentation_mask_path": str(mask_path),
        "hair_mask_stats": stats,
    }


def write_jsonl(records: list[dict[str, object]], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def quality_flags(
    record: dict[str, object],
    coverage_min: float = 0.08,
    coverage_max: float = 0.62,
    min_bbox_width: int = 105,
    min_bbox_height: int = 105,
    allow_touching_top: bool = True,
) -> list[str]:
    stats = record["hair_mask_stats"]
    flags: list[str] = []

    coverage = float(stats["coverage_ratio"])
    bbox_x = int(stats["bbox_x"])
    bbox_y = int(stats["bbox_y"])
    bbox_width = int(stats["bbox_width"])
    bbox_height = int(stats["bbox_height"])

    if coverage < coverage_min:
        flags.append("low_coverage")
    if coverage > coverage_max:
        flags.append("high_coverage")
    if bbox_width < min_bbox_width:
        flags.append("narrow_mask")
    if bbox_height < min_bbox_height:
        flags.append("short_mask")
    if bbox_x <= 0:
        flags.append("touches_left_edge")
    if bbox_y <= 0 and not allow_touching_top:
        flags.append("touches_top_edge")
    if bbox_y <= 0 and allow_touching_top:
        flags.append("touches_top_edge")

    if coverage >= coverage_min and coverage <= coverage_max and bbox_width >= min_bbox_width and bbox_height >= min_bbox_height:
        if not flags or flags == ["touches_top_edge"] or flags == ["touches_left_edge"] or sorted(flags) == ["touches_left_edge", "touches_top_edge"]:
            flags.append("usable_candidate")

    return flags


def quality_score(record: dict[str, object]) -> float:
    stats = record["hair_mask_stats"]
    coverage = float(stats["coverage_ratio"])
    bbox_width = int(stats["bbox_width"])
    bbox_height = int(stats["bbox_height"])
    bbox_x = int(stats["bbox_x"])
    bbox_y = int(stats["bbox_y"])

    score = 1.0
    if coverage < 0.08:
        score -= 0.45
    elif coverage < 0.12:
        score -= 0.20
    if coverage > 0.62:
        score -= 0.30
    elif coverage > 0.52:
        score -= 0.12

    if bbox_width < 105:
        score -= 0.20
    if bbox_height < 105:
        score -= 0.20
    if bbox_x <= 0:
        score -= 0.08
    if bbox_y <= 0:
        score -= 0.04

    return round(max(score, 0.0), 4)


def enrich_quality_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for record in records:
        enriched_record = dict(record)
        flags = quality_flags(record)
        enriched_record["quality_flags"] = flags
        enriched_record["quality_score"] = quality_score(record)
        enriched_record["usable_candidate"] = "usable_candidate" in flags
        enriched.append(enriched_record)
    return enriched


def quality_summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "total_records": 0,
            "usable_candidates": 0,
            "usable_ratio": 0.0,
            "avg_quality_score": 0.0,
            "flag_counts": {},
        }

    enriched = enrich_quality_records(records)
    usable_count = sum(1 for record in enriched if record["usable_candidate"])
    avg_score = round(sum(float(record["quality_score"]) for record in enriched) / len(enriched), 4)
    flag_counts: dict[str, int] = {}
    for record in enriched:
        for flag in record["quality_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "total_records": len(enriched),
        "usable_candidates": usable_count,
        "usable_ratio": round(usable_count / len(enriched), 4),
        "avg_quality_score": avg_score,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def celeba_attribute_hints_from_record(record: dict[str, object]) -> dict[str, object]:
    attrs = record["celeba_attributes"]
    positive_fields = [name for name, value in attrs.items() if int(value) == 1]

    color_hint = next(
        (
            name
            for name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
            if int(attrs.get(name, -1)) == 1
        ),
        None,
    )
    texture_hint = next(
        (
            name
            for name in ["Straight_Hair", "Wavy_Hair"]
            if int(attrs.get(name, -1)) == 1
        ),
        None,
    )

    return {
        "positive_fields": positive_fields,
        "color_hint": color_hint,
        "texture_hint": texture_hint,
        "bangs_hint": int(attrs.get("Bangs", -1)) == 1,
        "male_hint": int(attrs.get("Male", -1)) == 1,
        "receding_hairline_hint": int(attrs.get("Receding_Hairline", -1)) == 1,
    }


def crop_box_from_stats(
    stats: dict[str, float | int],
    image_size: tuple[int, int],
    pad_x_ratio: float = 0.16,
    pad_top_ratio: float = 0.14,
    pad_bottom_ratio: float = 0.08,
) -> dict[str, int]:
    image_width, image_height = image_size
    bbox_x = int(stats["bbox_x"])
    bbox_y = int(stats["bbox_y"])
    bbox_width = int(stats["bbox_width"])
    bbox_height = int(stats["bbox_height"])

    pad_x = max(int(bbox_width * pad_x_ratio), 12)
    pad_top = max(int(bbox_height * pad_top_ratio), 12)
    pad_bottom = max(int(bbox_height * pad_bottom_ratio), 8)

    left = max(bbox_x - pad_x, 0)
    top = max(bbox_y - pad_top, 0)
    right = min(bbox_x + bbox_width + pad_x, image_width)
    bottom = min(bbox_y + bbox_height + pad_bottom, image_height)

    return {
        "left": int(left),
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
    }


def extract_hair_asset_from_record(
    record: dict[str, object],
    output_image_path: str | Path,
    output_mask_path: str | Path,
) -> dict[str, object]:
    image_path = Path(record["image_path"])
    mask_path = Path(record["segmentation_mask_path"])
    output_image_path = Path(output_image_path)
    output_mask_path = Path(output_mask_path)

    with Image.open(image_path) as source_image:
        rgba_image = source_image.convert("RGBA")
        image_size = rgba_image.size

    with Image.open(mask_path) as source_mask:
        mask_image = source_mask.convert("L").resize(image_size, Image.Resampling.NEAREST)
    mask_image = cleaned_mask_image(mask_image)

    cleaned_stats = mask_statistics(mask_image)
    topology_stats = mask_topology_statistics(mask_image)
    shape_stats = mask_shape_statistics(mask_image)
    stats = cleaned_stats
    crop_box = crop_box_from_stats(stats, image_size)
    crop_tuple = (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])

    cropped_image = rgba_image.crop(crop_tuple)
    cropped_mask = mask_image.crop(crop_tuple)

    alpha_image = cropped_image.copy()
    alpha_image.putalpha(cropped_mask)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_image.save(output_image_path)
    cropped_mask.save(output_mask_path)

    return {
        "cleaned_hair_mask_stats": cleaned_stats,
        "mask_topology_stats": topology_stats,
        "mask_shape_stats": shape_stats,
        "crop_box": crop_box,
        "crop_size": {
            "width": crop_box["right"] - crop_box["left"],
            "height": crop_box["bottom"] - crop_box["top"],
        },
    }


def build_extracted_asset_metadata(
    record: dict[str, object],
    asset_id: str,
    image_path: str | Path,
    mask_path: str | Path,
    extraction_info: dict[str, object],
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "source_dataset": "CelebA",
        "original_celeba_file": record["image_id"],
        "raw_image_path": record["image_path"],
        "raw_mask_path": record["segmentation_mask_path"],
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "partition": record["partition"],
        "gender_label": record["gender_label"],
        "quality_score": record.get("quality_score"),
        "confidence_bucket": record.get("confidence_bucket"),
        "quality_flags": record.get("quality_flags", []),
        "hair_mask_stats": extraction_info.get("cleaned_hair_mask_stats", record["hair_mask_stats"]),
        "mask_topology_stats": extraction_info.get("mask_topology_stats", {}),
        "mask_shape_stats": extraction_info.get("mask_shape_stats", {}),
        "crop_box": extraction_info["crop_box"],
        "crop_size": extraction_info["crop_size"],
        "celeba_attribute_hints": celeba_attribute_hints_from_record(record),
        "labeling": {
            "status": "seed_from_celeba_attributes",
            "taxonomy_source": "CelebA_native_attributes",
            "normalized_attributes": {
                "length": None,
                "curl": None,
                "bang": None,
                "volume": None,
                "side_hair": None,
                "color": None,
                "style_family": None,
            },
            "notes": "",
        },
    }


def load_extracted_asset_metadata(metadata_root: str | Path) -> list[dict[str, object]]:
    metadata_root = Path(metadata_root)
    rows: list[dict[str, object]] = []
    for path in sorted(metadata_root.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def review_row_from_asset_metadata(metadata: dict[str, object]) -> dict[str, object]:
    hints = metadata.get("celeba_attribute_hints", {})
    return {
        "asset_id": metadata["asset_id"],
        "image_path": metadata["image_path"],
        "mask_path": metadata["mask_path"],
        "partition": metadata.get("partition"),
        "gender_label": metadata.get("gender_label"),
        "confidence_bucket": metadata.get("confidence_bucket"),
        "quality_score": metadata.get("quality_score"),
        "color_hint": hints.get("color_hint"),
        "texture_hint": hints.get("texture_hint"),
        "bangs_hint": hints.get("bangs_hint"),
        "male_hint": hints.get("male_hint"),
        "review_keep": "",
        "review_gender": "",
        "review_length": "",
        "review_curl": "",
        "review_bang": "",
        "review_volume": "",
        "review_side_hair": "",
        "review_color": "",
        "review_style_family": "",
        "review_notes": "",
    }


def select_manual_verification_sample(
    metadata_rows: list[dict[str, object]],
    per_group: int = 20,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in metadata_rows:
        key = (
            str(row.get("gender_label", "unknown")),
            str(row.get("confidence_bucket", "unknown")),
        )
        grouped.setdefault(key, []).append(row)

    sample: list[dict[str, object]] = []
    for key in sorted(grouped):
        sample.extend(grouped[key][:per_group])
    return sample


def write_review_csv(rows: list[dict[str, object]], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8")
    return destination


def load_review_table(review_csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(review_csv_path)
    if "asset_id" not in frame.columns:
        raise ValueError("Review CSV must contain an asset_id column.")
    frame["asset_id"] = frame["asset_id"].astype(str)
    return frame


def reviewed_asset_record(
    metadata: dict[str, object],
    review_row: dict[str, object],
) -> dict[str, object]:
    reviewed = dict(metadata)
    normalized = dict(reviewed.get("labeling", {}).get("normalized_attributes", {}))
    field_map = {
        "length": "review_length",
        "curl": "review_curl",
        "bang": "review_bang",
        "volume": "review_volume",
        "side_hair": "review_side_hair",
        "color": "review_color",
        "style_family": "review_style_family",
    }
    for target_field, source_field in field_map.items():
        value = review_row.get(source_field)
        if pd.notna(value) and str(value).strip():
            normalized[target_field] = str(value).strip()

    review_gender = review_row.get("review_gender")
    if pd.notna(review_gender) and str(review_gender).strip():
        reviewed["gender_label"] = str(review_gender).strip()

    review_keep = str(review_row.get("review_keep", "")).strip().lower()
    reviewed["review"] = {
        "keep": review_keep,
        "notes": "" if pd.isna(review_row.get("review_notes")) else str(review_row.get("review_notes")),
    }

    reviewed["labeling"] = {
        "status": "reviewed_keep" if review_keep == "yes" else "reviewed_reject",
        "taxonomy_source": "manual_review",
        "normalized_attributes": normalized,
        "notes": "" if pd.isna(review_row.get("review_notes")) else str(review_row.get("review_notes")),
    }
    return reviewed


def split_reviewed_assets(
    metadata_rows: list[dict[str, object]],
    review_frame: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    review_lookup = {
        str(row["asset_id"]): row
        for row in review_frame.to_dict(orient="records")
    }

    kept: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    missing_reviews: list[str] = []

    for metadata in metadata_rows:
        asset_id = str(metadata["asset_id"])
        review_row = review_lookup.get(asset_id)
        if review_row is None:
            missing_reviews.append(asset_id)
            continue

        reviewed = reviewed_asset_record(metadata, review_row)
        keep_value = reviewed["review"]["keep"]
        if keep_value == "yes":
            kept.append(reviewed)
        elif keep_value == "no":
            rejected.append(reviewed)
        else:
            missing_reviews.append(asset_id)

    return kept, rejected, missing_reviews


def normalized_label_hints_from_asset(metadata: dict[str, object]) -> dict[str, object]:
    hints = metadata.get("celeba_attribute_hints", {})
    color_map = {
        "Black_Hair": "black",
        "Blond_Hair": "blonde",
        "Brown_Hair": "brown",
        "Gray_Hair": "gray",
    }
    curl_map = {
        "Straight_Hair": "straight",
        "Wavy_Hair": "wavy",
    }
    return {
        "suggested_gender": metadata.get("gender_label"),
        "suggested_color": color_map.get(hints.get("color_hint"), ""),
        "suggested_curl": curl_map.get(hints.get("texture_hint"), ""),
        "suggested_bang": "bangs" if bool(hints.get("bangs_hint")) else "",
        "source_color_hint": hints.get("color_hint"),
        "source_texture_hint": hints.get("texture_hint"),
        "source_positive_fields": ",".join(hints.get("positive_fields", [])),
    }


def normalized_label_row_from_asset(metadata: dict[str, object]) -> dict[str, object]:
    normalized = metadata.get("labeling", {}).get("normalized_attributes", {})
    hint_payload = normalized_label_hints_from_asset(metadata)
    return {
        "asset_id": metadata["asset_id"],
        "image_path": metadata["image_path"],
        "mask_path": metadata["mask_path"],
        "gender_label": metadata.get("gender_label"),
        "confidence_bucket": metadata.get("confidence_bucket"),
        "quality_score": metadata.get("quality_score"),
        **hint_payload,
        "label_gender": metadata.get("gender_label") or "",
        "label_length": normalized.get("length") or "",
        "label_curl": normalized.get("curl") or "",
        "label_bang": normalized.get("bang") or "",
        "label_volume": normalized.get("volume") or "",
        "label_side_hair": normalized.get("side_hair") or "",
        "label_color": normalized.get("color") or "",
        "label_style_family": normalized.get("style_family") or "",
        "label_notes": metadata.get("labeling", {}).get("notes", "") or "",
    }


def apply_label_row_to_asset(
    metadata: dict[str, object],
    label_row: dict[str, object],
) -> dict[str, object]:
    updated = dict(metadata)
    normalized = dict(updated.get("labeling", {}).get("normalized_attributes", {}))

    field_map = {
        "length": "label_length",
        "curl": "label_curl",
        "bang": "label_bang",
        "volume": "label_volume",
        "side_hair": "label_side_hair",
        "color": "label_color",
        "style_family": "label_style_family",
    }
    for target_field, source_field in field_map.items():
        value = label_row.get(source_field)
        if pd.notna(value) and str(value).strip():
            normalized[target_field] = str(value).strip()

    label_gender = label_row.get("label_gender")
    if pd.notna(label_gender) and str(label_gender).strip():
        updated["gender_label"] = str(label_gender).strip()

    notes = label_row.get("label_notes")
    updated["labeling"] = {
        "status": "reviewed_labeled",
        "taxonomy_source": "manual_review_plus_visual_labeling",
        "normalized_attributes": normalized,
        "notes": "" if pd.isna(notes) else str(notes),
    }
    return updated


def merge_labels_into_kept_assets(
    kept_assets: list[dict[str, object]],
    label_frame: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[str]]:
    label_lookup = {
        str(row["asset_id"]): row
        for row in label_frame.to_dict(orient="records")
    }

    labeled_assets: list[dict[str, object]] = []
    missing_labels: list[str] = []

    for metadata in kept_assets:
        asset_id = str(metadata["asset_id"])
        label_row = label_lookup.get(asset_id)
        if label_row is None:
            missing_labels.append(asset_id)
            labeled_assets.append(metadata)
            continue
        labeled_assets.append(apply_label_row_to_asset(metadata, label_row))

    return labeled_assets, missing_labels


def merge_basic_predictions_into_assets(
    asset_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    prediction_lookup = {
        str(row["asset_id"]): row
        for row in prediction_rows
    }

    merged_rows: list[dict[str, object]] = []
    missing_predictions: list[str] = []

    for asset in asset_rows:
        asset_id = str(asset["asset_id"])
        merged = dict(asset)
        prediction_row = prediction_lookup.get(asset_id)
        if prediction_row is None:
            missing_predictions.append(asset_id)
            merged_rows.append(merged)
            continue

        merged["predicted_basic_attributes"] = prediction_row.get("predictions", {})
        merged_rows.append(merged)

    return merged_rows, missing_predictions


def full_hair_visibility_flags(
    record: dict[str, object],
    image_width: int = 178,
    image_height: int = 218,
    min_coverage: float = 0.14,
    max_coverage: float = 0.58,
    min_bbox_width: int = 120,
    min_bbox_height: int = 120,
    min_width_ratio: float = 0.58,
    min_height_ratio: float = 0.56,
    max_top_margin_ratio: float = 0.08,
    allow_touching_top: bool = False,
) -> list[str]:
    stats = record["hair_mask_stats"]
    coverage = float(stats["coverage_ratio"])
    bbox_x = int(stats["bbox_x"])
    bbox_y = int(stats["bbox_y"])
    bbox_width = int(stats["bbox_width"])
    bbox_height = int(stats["bbox_height"])
    bbox_right = bbox_x + bbox_width
    bbox_bottom = bbox_y + bbox_height

    flags: list[str] = []

    if coverage < min_coverage:
        flags.append("low_coverage")
    if coverage > max_coverage:
        flags.append("high_coverage")
    if bbox_width < min_bbox_width:
        flags.append("narrow_mask")
    if bbox_height < min_bbox_height:
        flags.append("short_mask")

    if (bbox_width / max(image_width, 1)) < min_width_ratio:
        flags.append("weak_side_span")
    if (bbox_height / max(image_height, 1)) < min_height_ratio:
        flags.append("weak_vertical_span")

    if bbox_x <= 0:
        flags.append("touches_left_edge")
    if bbox_right >= image_width:
        flags.append("touches_right_edge")
    if bbox_bottom >= image_height:
        flags.append("touches_bottom_edge")
    if bbox_y <= 0:
        flags.append("touches_top_edge")

    if (bbox_y / max(image_height, 1)) > max_top_margin_ratio:
        flags.append("hair_too_low")

    hard_rejects = {
        "low_coverage",
        "high_coverage",
        "narrow_mask",
        "short_mask",
        "weak_side_span",
        "weak_vertical_span",
        "touches_left_edge",
        "touches_right_edge",
        "touches_bottom_edge",
        "hair_too_low",
    }
    if not allow_touching_top:
        hard_rejects.add("touches_top_edge")

    if not any(flag in hard_rejects for flag in flags):
        flags.append("full_hair_candidate")

    return flags


def full_hair_visibility_score(
    record: dict[str, object],
    image_width: int = 178,
    image_height: int = 218,
) -> float:
    stats = record["hair_mask_stats"]
    coverage = float(stats["coverage_ratio"])
    bbox_x = int(stats["bbox_x"])
    bbox_y = int(stats["bbox_y"])
    bbox_width = int(stats["bbox_width"])
    bbox_height = int(stats["bbox_height"])
    bbox_right = bbox_x + bbox_width
    bbox_bottom = bbox_y + bbox_height

    score = 1.0

    if coverage < 0.14:
        score -= 0.35
    elif coverage < 0.18:
        score -= 0.15
    if coverage > 0.58:
        score -= 0.20

    if bbox_width < 120:
        score -= 0.18
    if bbox_height < 120:
        score -= 0.18

    width_ratio = bbox_width / max(image_width, 1)
    height_ratio = bbox_height / max(image_height, 1)
    if width_ratio < 0.58:
        score -= 0.18
    if height_ratio < 0.56:
        score -= 0.18

    if bbox_x <= 0:
        score -= 0.10
    if bbox_right >= image_width:
        score -= 0.10
    if bbox_bottom >= image_height:
        score -= 0.12
    if bbox_y <= 0:
        score -= 0.08
    if (bbox_y / max(image_height, 1)) > 0.08:
        score -= 0.10

    return round(max(score, 0.0), 4)


def enrich_full_hair_records(
    records: list[dict[str, object]],
    image_width: int = 178,
    image_height: int = 218,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        flags = full_hair_visibility_flags(
            row,
            image_width=image_width,
            image_height=image_height,
        )
        row["full_hair_flags"] = flags
        row["full_hair_score"] = full_hair_visibility_score(
            row,
            image_width=image_width,
            image_height=image_height,
        )
        row["full_hair_candidate"] = "full_hair_candidate" in flags
        enriched.append(row)
    return enriched


def full_hair_summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "total_records": 0,
            "full_hair_candidates": 0,
            "accepted_ratio": 0.0,
            "avg_full_hair_score": 0.0,
            "flag_counts": {},
        }

    enriched = enrich_full_hair_records(records)
    accepted = sum(1 for row in enriched if row["full_hair_candidate"])
    avg_score = round(sum(float(row["full_hair_score"]) for row in enriched) / len(enriched), 4)
    flag_counts: dict[str, int] = {}
    for row in enriched:
        for flag in row["full_hair_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "total_records": len(enriched),
        "full_hair_candidates": accepted,
        "accepted_ratio": round(accepted / len(enriched), 4),
        "avg_full_hair_score": avg_score,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def render_safe_flags(
    metadata: dict[str, object],
    min_quality_score: float = 0.96,
    min_full_hair_score: float = 0.96,
    min_crop_width: int = 118,
    min_crop_height: int = 108,
    min_bbox_width: int = 118,
    min_bbox_height: int = 118,
    min_coverage: float = 0.13,
    max_coverage: float = 0.44,
    max_component_count: int = 1,
    max_stray_ratio: float = 0.01,
    require_high_confidence: bool = True,
) -> list[str]:
    stats = metadata.get("hair_mask_stats", {}) or {}
    crop_size = metadata.get("crop_size", {}) or {}
    topology = metadata.get("mask_topology_stats", {}) or {}
    shape = metadata.get("mask_shape_stats", {}) or {}
    full_hair_flags = set(metadata.get("full_hair_flags", []) or [])

    quality_score = float(metadata.get("quality_score") or 0.0)
    full_hair_score = float(metadata.get("full_hair_score") or 0.0)
    confidence_bucket = str(metadata.get("confidence_bucket") or "").strip().lower()
    coverage = float(stats.get("coverage_ratio") or 0.0)
    bbox_width = int(stats.get("bbox_width") or 0)
    bbox_height = int(stats.get("bbox_height") or 0)
    crop_width = int(crop_size.get("width") or 0)
    crop_height = int(crop_size.get("height") or 0)
    component_count = int(topology.get("component_count") or 0)
    stray_ratio = float(topology.get("stray_ratio") or 0.0)
    top_band_coverage = float(shape.get("top_band_coverage") or 0.0)
    top_band_span_ratio = float(shape.get("top_band_span_ratio") or 0.0)
    side_bias_ratio = float(shape.get("side_bias_ratio") or 0.0)
    outer_side_ratio = float(shape.get("outer_side_ratio") or 0.0)

    flags: list[str] = []

    if quality_score < min_quality_score:
        flags.append("low_quality_score")
    if full_hair_score < min_full_hair_score:
        flags.append("low_full_hair_score")
    if require_high_confidence and confidence_bucket != "high":
        flags.append("not_high_confidence")
    if crop_width < min_crop_width:
        flags.append("narrow_crop")
    if crop_height < min_crop_height:
        flags.append("short_crop")
    if bbox_width < min_bbox_width:
        flags.append("narrow_mask")
    if bbox_height < min_bbox_height:
        flags.append("short_mask")
    if coverage < min_coverage:
        flags.append("low_coverage")
    if coverage > max_coverage:
        flags.append("high_coverage")
    if component_count > max_component_count:
        flags.append("multi_component_mask")
    if stray_ratio > max_stray_ratio:
        flags.append("high_stray_ratio")
    if top_band_coverage > 0.72:
        flags.append("blocky_top_region")
    if top_band_span_ratio > 0.92:
        flags.append("overwide_top_span")
    if side_bias_ratio > 0.38:
        flags.append("asymmetric_side_mass")
    if outer_side_ratio > 0.36:
        flags.append("heavy_outer_side_mass")

    inherited_render_risks = [
        "touches_left_edge",
        "touches_right_edge",
        "touches_bottom_edge",
        "hair_too_low",
        "weak_side_span",
        "weak_vertical_span",
    ]
    for risk in inherited_render_risks:
        if risk in full_hair_flags:
            flags.append(risk)

    hard_rejects = {
        "low_quality_score",
        "low_full_hair_score",
        "not_high_confidence",
        "narrow_crop",
        "short_crop",
        "narrow_mask",
        "short_mask",
        "low_coverage",
        "high_coverage",
        "multi_component_mask",
        "high_stray_ratio",
        "blocky_top_region",
        "overwide_top_span",
        "asymmetric_side_mass",
        "heavy_outer_side_mass",
        "touches_left_edge",
        "touches_right_edge",
        "touches_bottom_edge",
        "hair_too_low",
        "weak_side_span",
        "weak_vertical_span",
    }
    if not any(flag in hard_rejects for flag in flags):
        flags.append("render_safe_candidate")

    return flags


def render_safe_score(metadata: dict[str, object]) -> float:
    stats = metadata.get("hair_mask_stats", {}) or {}
    crop_size = metadata.get("crop_size", {}) or {}
    topology = metadata.get("mask_topology_stats", {}) or {}
    shape = metadata.get("mask_shape_stats", {}) or {}
    full_hair_flags = set(metadata.get("full_hair_flags", []) or [])

    quality_score = float(metadata.get("quality_score") or 0.0)
    full_hair_score = float(metadata.get("full_hair_score") or 0.0)
    confidence_bucket = str(metadata.get("confidence_bucket") or "").strip().lower()
    coverage = float(stats.get("coverage_ratio") or 0.0)
    bbox_width = int(stats.get("bbox_width") or 0)
    bbox_height = int(stats.get("bbox_height") or 0)
    crop_width = int(crop_size.get("width") or 0)
    crop_height = int(crop_size.get("height") or 0)
    component_count = int(topology.get("component_count") or 0)
    stray_ratio = float(topology.get("stray_ratio") or 0.0)
    top_band_coverage = float(shape.get("top_band_coverage") or 0.0)
    top_band_span_ratio = float(shape.get("top_band_span_ratio") or 0.0)
    side_bias_ratio = float(shape.get("side_bias_ratio") or 0.0)
    outer_side_ratio = float(shape.get("outer_side_ratio") or 0.0)

    score = 1.0

    if quality_score < 0.96:
        score -= 0.18
    elif quality_score < 0.98:
        score -= 0.06

    if full_hair_score < 0.96:
        score -= 0.18
    elif full_hair_score < 0.98:
        score -= 0.06

    if confidence_bucket != "high":
        score -= 0.12

    if coverage < 0.13:
        score -= 0.18
    elif coverage < 0.16:
        score -= 0.08
    if coverage > 0.44:
        score -= 0.12
    if component_count > 1:
        score -= 0.22
    if stray_ratio > 0.01:
        score -= 0.22
    elif stray_ratio > 0.002:
        score -= 0.10
    if top_band_coverage > 0.72:
        score -= 0.20
    elif top_band_coverage > 0.62:
        score -= 0.08
    if top_band_span_ratio > 0.92:
        score -= 0.18
    elif top_band_span_ratio > 0.84:
        score -= 0.08
    if side_bias_ratio > 0.38:
        score -= 0.16
    elif side_bias_ratio > 0.28:
        score -= 0.08
    if outer_side_ratio > 0.36:
        score -= 0.16
    elif outer_side_ratio > 0.28:
        score -= 0.08

    if bbox_width < 118:
        score -= 0.12
    if bbox_height < 118:
        score -= 0.12
    if crop_width < 118:
        score -= 0.12
    if crop_height < 108:
        score -= 0.10

    inherited_render_risks = [
        "touches_left_edge",
        "touches_right_edge",
        "touches_bottom_edge",
        "hair_too_low",
        "weak_side_span",
        "weak_vertical_span",
    ]
    for risk in inherited_render_risks:
        if risk in full_hair_flags:
            score -= 0.16

    return round(max(score, 0.0), 4)


def enrich_render_safe_assets(metadata_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for metadata in metadata_rows:
        row = dict(metadata)
        flags = render_safe_flags(row)
        row["render_safe_flags"] = flags
        row["render_safe_score"] = render_safe_score(row)
        row["render_safe_candidate"] = "render_safe_candidate" in flags
        enriched.append(row)
    return enriched


def render_safe_summary(metadata_rows: list[dict[str, object]]) -> dict[str, object]:
    if not metadata_rows:
        return {
            "total_assets": 0,
            "render_safe_candidates": 0,
            "accepted_ratio": 0.0,
            "avg_render_safe_score": 0.0,
            "flag_counts": {},
        }

    enriched = enrich_render_safe_assets(metadata_rows)
    accepted = sum(1 for row in enriched if row["render_safe_candidate"])
    avg_score = round(sum(float(row["render_safe_score"]) for row in enriched) / len(enriched), 4)
    flag_counts: dict[str, int] = {}
    for row in enriched:
        for flag in row["render_safe_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "total_assets": len(enriched),
        "render_safe_candidates": accepted,
        "accepted_ratio": round(accepted / len(enriched), 4),
        "avg_render_safe_score": avg_score,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def reviewed_render_safe_asset_record(
    metadata: dict[str, object],
    review_row: dict[str, object],
) -> dict[str, object]:
    reviewed = dict(metadata)
    render_keep = str(review_row.get("render_keep", "")).strip().lower()
    render_notes = "" if pd.isna(review_row.get("render_notes")) else str(review_row.get("render_notes"))

    reviewed["render_review"] = {
        "keep": render_keep,
        "notes": render_notes,
    }
    reviewed["render_bank"] = {
        "status": "reviewed_render_safe_keep" if render_keep == "yes" else "reviewed_render_safe_reject",
        "source": "manual_render_review",
    }
    return reviewed


def split_render_safe_reviewed_assets(
    metadata_rows: list[dict[str, object]],
    review_frame: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    review_lookup = {
        str(row["asset_id"]): row
        for row in review_frame.to_dict(orient="records")
    }

    kept: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    missing_reviews: list[str] = []

    for metadata in metadata_rows:
        asset_id = str(metadata["asset_id"])
        review_row = review_lookup.get(asset_id)
        if review_row is None:
            missing_reviews.append(asset_id)
            continue

        reviewed = reviewed_render_safe_asset_record(metadata, review_row)
        keep_value = reviewed["render_review"]["keep"]
        if keep_value == "yes":
            kept.append(reviewed)
        elif keep_value == "no":
            rejected.append(reviewed)
        else:
            missing_reviews.append(asset_id)

    return kept, rejected, missing_reviews
