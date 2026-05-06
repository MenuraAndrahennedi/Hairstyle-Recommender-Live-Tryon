from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from support_app.config import PROJECT_ROOT
from support_app.ml.datasets import read_jsonl_manifest, write_jsonl_manifest


RAW_CELEBA_ROOT = PROJECT_ROOT / "backend" / "data" / "raw" / "celeba"

FACE_FIELDS = ("gender", "face_fullness", "cheekbones", "hairline")
HAIR_FIELDS = ("length", "curl", "style_family")
FEATURE_WEIGHTS = {
    "gender": 1.6,
    "face_fullness": 0.8,
    "cheekbones": 1.0,
    "hairline": 1.0,
}
TARGET_WEIGHTS = {
    "length": 1.0,
    "curl": 1.0,
    "style_family": 1.25,
}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_celeba_attribute_rows(raw_celeba_root: str | Path | None = None) -> dict[str, dict[str, int]]:
    root = _resolve_project_path(raw_celeba_root or RAW_CELEBA_ROOT)
    attr_path = root / "list_attr_celeba.csv"
    rows: dict[str, dict[str, int]] = {}
    with attr_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            rows[image_id] = {
                key: int(value)
                for key, value in row.items()
                if key != "image_id"
            }
    return rows


def derive_face_labels(celeba_attrs: dict[str, int]) -> dict[str, str] | None:
    if celeba_attrs.get("Wearing_Hat", -1) == 1:
        return None
    if celeba_attrs.get("Bald", -1) == 1:
        return None
    return {
        "gender": "male" if celeba_attrs.get("Male", -1) == 1 else "female",
        "face_fullness": "full" if celeba_attrs.get("Chubby", -1) == 1 else "slim",
        "cheekbones": "high" if celeba_attrs.get("High_Cheekbones", -1) == 1 else "soft",
        "hairline": "receding" if celeba_attrs.get("Receding_Hairline", -1) == 1 else "regular",
    }


def _normalized_hair_labels(asset_row: dict[str, object]) -> dict[str, str]:
    normalized = asset_row.get("labeling", {}).get("normalized_attributes", {})
    labels: dict[str, str] = {}
    for field in HAIR_FIELDS:
        value = normalized.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "null", "nan"}:
            labels[field] = text
    return labels


def build_face_to_hair_training_records(
    enriched_asset_jsonl_path: str | Path,
    raw_celeba_root: str | Path | None = None,
) -> list[dict[str, object]]:
    asset_rows = read_jsonl_manifest(enriched_asset_jsonl_path)
    celeba_lookup = load_celeba_attribute_rows(raw_celeba_root=raw_celeba_root)
    records: list[dict[str, object]] = []

    for asset_row in asset_rows:
        source_id = str(asset_row.get("original_celeba_file") or asset_row.get("source_id") or "")
        if not source_id:
            continue
        celeba_attrs = celeba_lookup.get(source_id)
        if celeba_attrs is None:
            continue

        face_labels = derive_face_labels(celeba_attrs)
        hair_labels = _normalized_hair_labels(asset_row)
        if face_labels is None or len(hair_labels) != len(HAIR_FIELDS):
            continue

        records.append(
            {
                "asset_id": asset_row["asset_id"],
                "source_id": source_id,
                "image_path": asset_row["image_path"],
                "asset_partition": asset_row.get("partition", "train"),
                "confidence_bucket": asset_row.get("confidence_bucket", "unknown"),
                "quality_score": asset_row.get("quality_score", 0.0),
                "face_labels": face_labels,
                "hair_labels": hair_labels,
            }
        )

    return records


def split_face_to_hair_records(
    records: list[dict[str, object]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    return shuffled[:split_index], shuffled[split_index:]


def _safe_prob(count: int, total: int, vocab_size: int, alpha: float = 1.0) -> float:
    return (count + alpha) / max(total + alpha * vocab_size, 1.0)


def build_mapper_payload(
    train_records: list[dict[str, object]],
    feature_fields: Iterable[str] = FACE_FIELDS,
    target_fields: Iterable[str] = HAIR_FIELDS,
) -> dict[str, object]:
    feature_fields = tuple(feature_fields)
    target_fields = tuple(target_fields)

    target_value_counts: dict[str, Counter[str]] = {field: Counter() for field in target_fields}
    conditional_counts: dict[str, dict[str, dict[str, Counter[str]]]] = {
        target_field: {
            feature_field: defaultdict(Counter)
            for feature_field in feature_fields
        }
        for target_field in target_fields
    }

    for record in train_records:
        face_labels = record["face_labels"]
        hair_labels = record["hair_labels"]
        for target_field in target_fields:
            target_value = hair_labels[target_field]
            target_value_counts[target_field][target_value] += 1
            for feature_field in feature_fields:
                feature_value = face_labels[feature_field]
                conditional_counts[target_field][feature_field][feature_value][target_value] += 1

    target_vocab = {
        field: sorted(counter.keys())
        for field, counter in target_value_counts.items()
    }

    serializable_conditional = {
        target_field: {
            feature_field: {
                feature_value: dict(counter)
                for feature_value, counter in feature_map.items()
            }
            for feature_field, feature_map in feature_maps.items()
        }
        for target_field, feature_maps in conditional_counts.items()
    }

    return {
        "mapper_type": "lightweight_frequency_mapper",
        "feature_fields": list(feature_fields),
        "target_fields": list(target_fields),
        "feature_weights": FEATURE_WEIGHTS,
        "target_weights": TARGET_WEIGHTS,
        "train_count": len(train_records),
        "target_value_counts": {
            field: dict(counter)
            for field, counter in target_value_counts.items()
        },
        "target_vocab": target_vocab,
        "conditional_counts": serializable_conditional,
    }


def mapper_score_for_labels(
    face_labels: dict[str, str],
    hair_labels: dict[str, str],
    mapper_payload: dict[str, object],
) -> float:
    target_value_counts = mapper_payload["target_value_counts"]
    target_vocab = mapper_payload["target_vocab"]
    conditional_counts = mapper_payload["conditional_counts"]
    feature_fields = mapper_payload["feature_fields"]
    target_fields = mapper_payload["target_fields"]
    feature_weights = mapper_payload.get("feature_weights", FEATURE_WEIGHTS)
    target_weights = mapper_payload.get("target_weights", TARGET_WEIGHTS)

    total_score = 0.0
    for target_field in target_fields:
        target_value = hair_labels.get(target_field)
        if not target_value:
            continue

        field_counts = target_value_counts[target_field]
        vocab = target_vocab[target_field]
        field_total = sum(field_counts.values())
        base_prob = _safe_prob(field_counts.get(target_value, 0), field_total, len(vocab))
        field_score = math.log(base_prob)

        for feature_field in feature_fields:
            feature_value = face_labels.get(feature_field)
            if not feature_value:
                continue
            feature_map = conditional_counts[target_field][feature_field]
            value_counts = feature_map.get(feature_value, {})
            condition_total = sum(value_counts.values())
            conditional_prob = _safe_prob(
                value_counts.get(target_value, 0),
                condition_total,
                len(vocab),
            )
            field_score += feature_weights.get(feature_field, 1.0) * math.log(conditional_prob)

        total_score += target_weights.get(target_field, 1.0) * field_score

    asset_gender = hair_labels.get("gender_label")
    face_gender = face_labels.get("gender")
    if asset_gender and face_gender and asset_gender != face_gender:
        total_score -= 6.0

    return float(total_score)


def predict_hair_labels_for_face(
    face_labels: dict[str, str],
    mapper_payload: dict[str, object],
) -> dict[str, dict[str, float | str]]:
    predictions: dict[str, dict[str, float | str]] = {}
    for target_field in mapper_payload["target_fields"]:
        best_label = None
        best_score = None
        for target_value in mapper_payload["target_vocab"][target_field]:
            candidate_labels = {target_field: target_value}
            score = mapper_score_for_labels(face_labels, candidate_labels, mapper_payload)
            if best_score is None or score > best_score:
                best_label = target_value
                best_score = score
        predictions[target_field] = {
            "label": best_label or "",
            "score": float(best_score or 0.0),
        }
    return predictions


def predict_hair_labels_from_candidates(
    face_labels: dict[str, str],
    mapper_payload: dict[str, object],
    candidate_rows: list[dict[str, object]],
) -> dict[str, dict[str, float | str]]:
    predictions: dict[str, dict[str, float | str]] = {}
    for target_field in mapper_payload["target_fields"]:
        best_label = ""
        best_score = None
        seen_values: set[str] = set()
        for row in candidate_rows:
            hair_labels = _normalized_hair_labels(row)
            target_value = hair_labels.get(target_field)
            if not target_value or target_value in seen_values:
                continue
            seen_values.add(target_value)
            candidate_labels = {
                target_field: target_value,
                "gender_label": str(row.get("gender_label", "")),
            }
            score = mapper_score_for_labels(face_labels, candidate_labels, mapper_payload)
            if best_score is None or score > best_score:
                best_label = target_value
                best_score = score
        predictions[target_field] = {
            "label": best_label,
            "score": float(best_score or 0.0),
        }
    return predictions


def recommend_assets_for_face(
    face_labels: dict[str, str],
    asset_rows: list[dict[str, object]],
    mapper_payload: dict[str, object],
    top_k: int = 5,
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for asset_row in asset_rows:
        hair_labels = _normalized_hair_labels(asset_row)
        if len(hair_labels) != len(HAIR_FIELDS):
            continue
        score = mapper_score_for_labels(
            face_labels,
            {
                **hair_labels,
                "gender_label": str(asset_row.get("gender_label", "")),
            },
            mapper_payload,
        )
        ranked.append(
            {
                "asset_id": asset_row["asset_id"],
                "source_id": asset_row.get("original_celeba_file"),
                "image_path": asset_row["image_path"],
                "score": score,
                "hair_labels": hair_labels,
                "gender_label": asset_row.get("gender_label", ""),
            }
        )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]


def evaluate_mapper_predictions(
    val_records: list[dict[str, object]],
    mapper_payload: dict[str, object],
) -> dict[str, object]:
    per_field_correct = Counter()
    total = len(val_records)
    exact_matches = 0
    prediction_rows: list[dict[str, object]] = []

    for record in val_records:
        face_labels = record["face_labels"]
        true_hair = record["hair_labels"]
        predicted = predict_hair_labels_for_face(face_labels, mapper_payload)
        correct_all = True
        row = {
            "asset_id": record["asset_id"],
            "source_id": record["source_id"],
        }
        for field in HAIR_FIELDS:
            pred_label = str(predicted[field]["label"])
            true_label = str(true_hair[field])
            correct = pred_label == true_label
            row[f"true_{field}"] = true_label
            row[f"pred_{field}"] = pred_label
            row[f"correct_{field}"] = correct
            if correct:
                per_field_correct[field] += 1
            else:
                correct_all = False
        if correct_all:
            exact_matches += 1
        prediction_rows.append(row)

    metrics = {
        "record_count": total,
        "exact_match_accuracy": exact_matches / max(total, 1),
        "field_accuracy": {
            field: per_field_correct[field] / max(total, 1)
            for field in HAIR_FIELDS
        },
        "prediction_rows": prediction_rows,
    }
    return metrics


def save_mapper_payload(mapper_payload: dict[str, object], output_path: str | Path) -> Path:
    path = _resolve_project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapper_payload, indent=2), encoding="utf-8")
    return path


def load_mapper_payload(path: str | Path) -> dict[str, object]:
    resolved = _resolve_project_path(path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_mapper_records(records: list[dict[str, object]], output_path: str | Path) -> Path:
    return write_jsonl_manifest(records, output_path)

