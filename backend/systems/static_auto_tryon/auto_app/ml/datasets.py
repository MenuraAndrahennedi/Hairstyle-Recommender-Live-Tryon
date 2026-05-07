from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from auto_app.config import PROJECT_ROOT
from auto_app.ml.khairstyle_translation import build_normalized_attributes, translate_labels

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None
    Dataset = object

from auto_app.ml.transforms import pil_image_to_tensor, pil_mask_to_tensor


ATTRIBUTE_FIELDS = (
    "length",
    "curl",
    "bang",
    "volume",
    "side_hair",
    "color",
    "style_family",
)

RAW_CELEBAMASK_ROOT = PROJECT_ROOT / "backend" / "data" / "raw" / "celebamask_hq"
PROCESSED_STAGE1_REVIEW_CSV = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "processed"
    / "celebamask_hq_hair_assets"
    / "manual_labels"
    / "celebamask_hq_label_template.csv"
)
RAW_KHAIRSTYLE_ROOT = PROJECT_ROOT / "backend" / "data" / "raw" / "khairstyle" / "mqset"
RAW_KHAIRSTYLE_LABEL_DIR = RAW_KHAIRSTYLE_ROOT / "labels" / "labels_mqset"
RAW_KHAIRSTYLE_IMAGE_DIR = RAW_KHAIRSTYLE_ROOT / "images" / "images_mqset001"


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: str
    mask_path: str
    source_id: str
    source_dataset: str = "CelebAMask-HQ"

    def to_dict(self) -> dict[str, str]:
        return {
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "source_id": self.source_id,
            "source_dataset": self.source_dataset,
        }


@dataclass(frozen=True)
class AttributeRecord:
    image_path: str
    labels: dict[str, str]
    source_id: str
    source_dataset: str

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "labels": self.labels,
            "source_id": self.source_id,
            "source_dataset": self.source_dataset,
        }


def _ensure_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for dataset tensors and training. "
            "Install the packages from the project root requirements.txt before running the notebooks."
        )


def _project_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def write_jsonl_manifest(records: Iterable[dict[str, object]], output_path: str | Path) -> Path:
    path = _resolve_project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl_manifest(path: str | Path) -> list[dict[str, object]]:
    resolved = _resolve_project_path(path)
    if not resolved.exists():
        return []
    rows: list[dict[str, object]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_val_split(
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


def build_segmentation_records_from_celebamask(
    raw_root: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    root = _resolve_project_path(raw_root or RAW_CELEBAMASK_ROOT)
    image_root = root / "CelebA-HQ-img"
    mask_root = root / "CelebAMask-HQ-mask-anno"
    records: list[dict[str, str]] = []

    for mask_path in sorted(mask_root.rglob("*_hair.png")):
        source_id = mask_path.stem.split("_")[0]
        image_path = image_root / f"{int(source_id)}.jpg"
        if not image_path.exists():
            continue
        records.append(
            SegmentationRecord(
                image_path=_project_relative(image_path),
                mask_path=_project_relative(mask_path),
                source_id=source_id,
            ).to_dict()
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def build_attribute_records_from_khairstyle(
    metadata_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    directory = _resolve_project_path(metadata_dir or RAW_KHAIRSTYLE_LABEL_DIR)
    records: list[dict[str, object]] = []

    for metadata_path in sorted(directory.rglob("*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8", errors="replace"))
        translated_labels = translate_labels(payload)
        labels = build_normalized_attributes(translated_labels)
        if not all(labels.get(field) for field in ATTRIBUTE_FIELDS):
            continue

        relative_parent = metadata_path.relative_to(directory).parent
        filename = str(payload.get("filename") or "").strip()
        image_candidates = []
        if filename:
            image_candidates.append(RAW_KHAIRSTYLE_IMAGE_DIR / relative_parent / filename)
        image_candidates.extend(
            [
                RAW_KHAIRSTYLE_IMAGE_DIR / relative_parent / f"{metadata_path.stem}.jpg",
                RAW_KHAIRSTYLE_IMAGE_DIR / relative_parent / f"{metadata_path.stem.replace('_', '-')}.jpg",
                RAW_KHAIRSTYLE_IMAGE_DIR / relative_parent / f"{metadata_path.stem}.png",
                RAW_KHAIRSTYLE_IMAGE_DIR / relative_parent / f"{metadata_path.stem.replace('_', '-')}.png",
            ]
        )
        image_path = next((candidate for candidate in image_candidates if candidate.exists()), None)
        if image_path is None:
            continue

        records.append(
            AttributeRecord(
                image_path=_project_relative(image_path),
                labels={field: str(labels[field]) for field in ATTRIBUTE_FIELDS},
                source_id=str(payload.get("id") or metadata_path.stem),
                source_dataset="K-Hairstyle",
            ).to_dict()
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def build_attribute_records_from_stage1_review(
    csv_path: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    path = _resolve_project_path(csv_path or PROCESSED_STAGE1_REVIEW_CSV)
    if not path.exists():
        return []

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("review_status", "").strip().lower() != "approved":
                continue
            labels = {field: (row.get(field) or "").strip() for field in ATTRIBUTE_FIELDS}
            if not all(labels.values()):
                continue
            records.append(
                AttributeRecord(
                    image_path=row["image_path"],
                    labels=labels,
                    source_id=row["asset_id"],
                    source_dataset="CelebAMask-HQ-reviewed",
                ).to_dict()
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def build_label_vocab(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    return build_label_vocab_for_fields(records, ATTRIBUTE_FIELDS)


def build_label_vocab_for_fields(
    records: list[dict[str, object]],
    fields: Iterable[str],
) -> dict[str, dict[str, int]]:
    field_list = tuple(fields)
    vocab: dict[str, dict[str, int]] = {field: {} for field in field_list}
    for field in field_list:
        values = sorted({str(record["labels"][field]) for record in records if field in record.get("labels", {})})
        vocab[field] = {value: index for index, value in enumerate(values)}
    return vocab


def build_attribute_records_from_reviewed_assets(
    labeled_jsonl_path: str | Path,
    required_fields: Iterable[str],
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows = read_jsonl_manifest(labeled_jsonl_path)
    field_list = tuple(required_fields)
    records: list[dict[str, object]] = []

    def _normalized_label_value(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"", "none", "nan", "null"}:
            return ""
        return text

    for payload in rows:
        normalized = payload.get("labeling", {}).get("normalized_attributes", {})
        labels = {field: _normalized_label_value(normalized.get(field, "")) for field in field_list}
        if not all(labels.values()):
            continue
        records.append(
            AttributeRecord(
                image_path=str(payload["image_path"]),
                labels=labels,
                source_id=str(payload["asset_id"]),
                source_dataset="CelebA-reviewed-kept",
            ).to_dict()
        )
        if limit is not None and len(records) >= limit:
            break
    return records


class HairSegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, object]],
        transform: Callable[[Image.Image, Image.Image], tuple[object, object]] | None = None,
    ) -> None:
        _ensure_torch()
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = Image.open(_resolve_project_path(str(record["image_path"]))).convert("RGB")
        mask = Image.open(_resolve_project_path(str(record["mask_path"]))).convert("L")
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            image_tensor = pil_image_to_tensor(image)
            mask_tensor = pil_mask_to_tensor(mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "source_id": record["source_id"],
        }


class MultiAttributeDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, object]],
        label_vocab: dict[str, dict[str, int]],
        transform: Callable[[Image.Image], object] | None = None,
        fields: Iterable[str] | None = None,
    ) -> None:
        _ensure_torch()
        self.records = records
        self.label_vocab = label_vocab
        self.transform = transform
        self.fields = tuple(fields or label_vocab.keys())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = Image.open(_resolve_project_path(str(record["image_path"]))).convert("RGB")
        image_tensor = self.transform(image) if self.transform is not None else pil_image_to_tensor(image)
        labels = {
            field: self.label_vocab[field][str(record["labels"][field])]
            for field in self.fields
        }
        return {
            "image": image_tensor,
            "labels": {field: torch.tensor(value, dtype=torch.long) for field, value in labels.items()},
            "source_id": record["source_id"],
        }

