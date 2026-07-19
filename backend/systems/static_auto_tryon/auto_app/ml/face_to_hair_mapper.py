from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from auto_app.config import PROJECT_ROOT, resolve_project_path
from auto_app.ml.datasets import read_jsonl_manifest, write_jsonl_manifest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None


RAW_CELEBA_ROOT = PROJECT_ROOT / "backend" / "data" / "raw" / "celeba"

FACE_FIELDS = ("gender", "face_fullness", "cheekbones", "hairline")
HAIR_FIELDS = ("length", "curl", "style_family")
UNKNOWN_TOKEN = "__unk__"
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


def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for neural face-to-hair mapper training or inference. "
            "Install the packages from requirements.txt before running the mapper trainer."
        )


def _resolve_project_path(path: str | Path) -> Path:
    return resolve_project_path(path)


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


def build_face_label_vocab(
    records: list[dict[str, object]],
    feature_fields: Iterable[str] = FACE_FIELDS,
) -> dict[str, dict[str, int]]:
    vocab: dict[str, dict[str, int]] = {}
    for field in feature_fields:
        values = sorted(
            {
                str(record["face_labels"].get(field, "")).strip()
                for record in records
                if str(record["face_labels"].get(field, "")).strip()
            }
        )
        values.append(UNKNOWN_TOKEN)
        vocab[field] = {value: index for index, value in enumerate(values)}
    return vocab


def build_hair_label_vocab(
    records: list[dict[str, object]],
    target_fields: Iterable[str] = HAIR_FIELDS,
) -> dict[str, dict[str, int]]:
    vocab: dict[str, dict[str, int]] = {}
    for field in target_fields:
        values = sorted({str(record["hair_labels"][field]) for record in records})
        vocab[field] = {value: index for index, value in enumerate(values)}
    return vocab


if nn is not None:
    class NeuralFaceToHairMapper(nn.Module):
        def __init__(
            self,
            face_vocab: dict[str, dict[str, int]],
            hair_vocab: dict[str, dict[str, int]],
            feature_fields: Iterable[str] = FACE_FIELDS,
            target_fields: Iterable[str] = HAIR_FIELDS,
            embedding_dim: int = 8,
            hidden_dim: int = 64,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.feature_fields = tuple(feature_fields)
            self.target_fields = tuple(target_fields)
            self.face_vocab = face_vocab
            self.hair_vocab = hair_vocab
            self.embeddings = nn.ModuleDict(
                {
                    field: nn.Embedding(len(face_vocab[field]), embedding_dim)
                    for field in self.feature_fields
                }
            )
            self.encoder = nn.Sequential(
                nn.Linear(len(self.feature_fields) * embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.heads = nn.ModuleDict(
                {
                    field: nn.Linear(hidden_dim, len(hair_vocab[field]))
                    for field in self.target_fields
                }
            )

        def forward(self, features: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            embedded = []
            for index, field in enumerate(self.feature_fields):
                embedded.append(self.embeddings[field](features[:, index]))
            fused = torch.cat(embedded, dim=1)
            encoded = self.encoder(fused)
            return {field: head(encoded) for field, head in self.heads.items()}
else:
    class NeuralFaceToHairMapper:  # pragma: no cover - runtime fallback
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _ensure_torch()


def build_neural_mapper_model(
    face_vocab: dict[str, dict[str, int]],
    hair_vocab: dict[str, dict[str, int]],
    feature_fields: Iterable[str] = FACE_FIELDS,
    target_fields: Iterable[str] = HAIR_FIELDS,
    embedding_dim: int = 8,
    hidden_dim: int = 64,
    dropout: float = 0.2,
) -> NeuralFaceToHairMapper:
    _ensure_torch()
    return NeuralFaceToHairMapper(
        face_vocab=face_vocab,
        hair_vocab=hair_vocab,
        feature_fields=feature_fields,
        target_fields=target_fields,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def encode_face_labels_tensor(
    face_labels: dict[str, str],
    face_vocab: dict[str, dict[str, int]],
    feature_fields: Iterable[str] = FACE_FIELDS,
    device: str | None = None,
) -> "torch.Tensor":
    _ensure_torch()
    values = []
    for field in feature_fields:
        field_vocab = face_vocab[field]
        token = str(face_labels.get(field, "")).strip()
        values.append(field_vocab.get(token, field_vocab[UNKNOWN_TOKEN]))
    return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)


def encode_mapper_batch(
    records: list[dict[str, object]],
    face_vocab: dict[str, dict[str, int]],
    hair_vocab: dict[str, dict[str, int]],
    feature_fields: Iterable[str] = FACE_FIELDS,
    target_fields: Iterable[str] = HAIR_FIELDS,
    device: str | None = None,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    _ensure_torch()
    feature_rows = []
    target_rows: dict[str, list[int]] = {field: [] for field in target_fields}
    for record in records:
        feature_rows.append(
            [
                face_vocab[field].get(
                    str(record["face_labels"].get(field, "")).strip(),
                    face_vocab[field][UNKNOWN_TOKEN],
                )
                for field in feature_fields
            ]
        )
        for field in target_fields:
            target_rows[field].append(hair_vocab[field][str(record["hair_labels"][field])])
    features = torch.tensor(feature_rows, dtype=torch.long, device=device)
    targets = {
        field: torch.tensor(values, dtype=torch.long, device=device)
        for field, values in target_rows.items()
    }
    return features, targets


def multitask_mapper_loss(
    outputs: dict[str, "torch.Tensor"],
    targets: dict[str, "torch.Tensor"],
    class_weights: dict[str, "torch.Tensor"] | None = None,
) -> "torch.Tensor":
    _ensure_torch()
    losses = []
    for field, logits in outputs.items():
        weights = None if class_weights is None else class_weights.get(field)
        losses.append(nn.CrossEntropyLoss(weight=weights)(logits, targets[field]))
    return sum(losses)


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


@lru_cache(maxsize=4)
def _load_neural_mapper_runtime(
    checkpoint_path: str,
) -> tuple[object, dict[str, dict[str, int]], dict[str, dict[int, str]], tuple[str, ...], tuple[str, ...]]:
    _ensure_torch()
    payload = torch.load(_resolve_project_path(checkpoint_path), map_location="cpu")
    feature_fields = tuple(payload.get("feature_fields", FACE_FIELDS))
    target_fields = tuple(payload.get("target_fields", HAIR_FIELDS))
    face_vocab = payload["face_vocab"]
    hair_vocab = payload["hair_vocab"]
    model = build_neural_mapper_model(
        face_vocab=face_vocab,
        hair_vocab=hair_vocab,
        feature_fields=feature_fields,
        target_fields=target_fields,
        embedding_dim=int(payload.get("embedding_dim", 8)),
        hidden_dim=int(payload.get("hidden_dim", 64)),
        dropout=float(payload.get("dropout", 0.2)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    inverse_vocab = {
        field: {index: label for label, index in field_vocab.items()}
        for field, field_vocab in hair_vocab.items()
    }
    return model, face_vocab, inverse_vocab, feature_fields, target_fields


def _predict_neural_mapper_labels(
    face_labels: dict[str, str],
    mapper_payload: dict[str, object],
) -> dict[str, dict[str, float | str]]:
    _ensure_torch()
    checkpoint_path = str(mapper_payload["checkpoint_path"])
    model, face_vocab, inverse_vocab, feature_fields, target_fields = _load_neural_mapper_runtime(checkpoint_path)
    encoded = encode_face_labels_tensor(face_labels, face_vocab, feature_fields=feature_fields)
    with torch.inference_mode():
        outputs = model(encoded)
    predictions: dict[str, dict[str, float | str]] = {}
    for field in target_fields:
        probabilities = torch.softmax(outputs[field], dim=1)
        confidence, index = probabilities.max(dim=1)
        predictions[field] = {
            "label": inverse_vocab[field][int(index.item())],
            "score": float(confidence.item()),
        }
    return predictions


def predict_hair_labels_for_face(
    face_labels: dict[str, str],
    mapper_payload: dict[str, object],
) -> dict[str, dict[str, float | str]]:
    if mapper_payload.get("mapper_type") == "neural_multitask_mapper":
        return _predict_neural_mapper_labels(face_labels, mapper_payload)

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
    if mapper_payload.get("mapper_type") == "neural_multitask_mapper":
        base_predictions = _predict_neural_mapper_labels(face_labels, mapper_payload)
        checkpoint_path = str(mapper_payload["checkpoint_path"])
        _model, _face_vocab, inverse_vocab, _feature_fields, target_fields = _load_neural_mapper_runtime(checkpoint_path)
        predictions: dict[str, dict[str, float | str]] = {}

        for field in target_fields:
            field_index_to_label = inverse_vocab[field]
            field_probs = {
                label: 0.0
                for label in field_index_to_label.values()
            }
            predicted_label = str(base_predictions[field]["label"])
            field_probs[predicted_label] = float(base_predictions[field]["score"])

            best_label = ""
            best_score = None
            seen_values: set[str] = set()
            for row in candidate_rows:
                hair_labels = _normalized_hair_labels(row)
                target_value = hair_labels.get(field)
                if not target_value or target_value in seen_values:
                    continue
                seen_values.add(target_value)
                score = field_probs.get(target_value, 0.0)
                face_gender = face_labels.get("gender")
                asset_gender = str(row.get("gender_label", ""))
                if face_gender and asset_gender and asset_gender not in {"", "neutral", face_gender}:
                    score -= 0.15
                if best_score is None or score > best_score:
                    best_label = target_value
                    best_score = score

            predictions[field] = {
                "label": best_label,
                "score": float(best_score or 0.0),
            }
        return predictions

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

        if mapper_payload.get("mapper_type") == "neural_multitask_mapper":
            predictions = predict_hair_labels_for_face(face_labels, mapper_payload)
            score = 0.0
            if predictions.get("length", {}).get("label") == hair_labels.get("length"):
                score += 1.0
            if predictions.get("curl", {}).get("label") == hair_labels.get("curl"):
                score += 1.0
            if predictions.get("style_family", {}).get("label") == hair_labels.get("style_family"):
                score += 1.25
        else:
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
                "score": float(score),
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
