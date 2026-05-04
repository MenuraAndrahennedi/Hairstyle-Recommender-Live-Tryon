from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import PROJECT_ROOT
from app.core.face_analyzer import analyze_face_image
from app.core.hair_segmentation import predict_hair_mask_image, predict_hair_mask_image_from_face_roi
from app.core.live_support import evaluate_supported_live_range, summarize_live_mask_quality
from app.ml.datasets import read_jsonl_manifest, write_jsonl_manifest


LIVE_SEGMENTATION_INPUT_DIR = (
    PROJECT_ROOT / "backend" / "outputs" / "live_segmentation_eval" / "inputs"
)
LIVE_SEGMENTATION_DATASET_DIR = (
    PROJECT_ROOT / "backend" / "data" / "datasets" / "live_segmentation_adaptation"
)


def _relative_project_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_live_frame_paths(
    input_dir: str | Path | None = None,
) -> list[Path]:
    directory = Path(input_dir or LIVE_SEGMENTATION_INPUT_DIR)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [path for path in sorted(directory.glob("*")) if path.suffix.lower() in image_exts]


def build_live_segmentation_seed_records(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    use_face_roi: bool = True,
    keep_only_supported_frames: bool = False,
) -> dict[str, Any]:
    frame_paths = collect_live_frame_paths(input_dir)
    dataset_dir = Path(output_dir or LIVE_SEGMENTATION_DATASET_DIR)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    mask_dir = dataset_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(frame_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        analysis = analyze_face_image(image_path)
        support = evaluate_supported_live_range(image, analysis)

        if use_face_roi and analysis.face_bbox is not None:
            mask_image = predict_hair_mask_image_from_face_roi(image, analysis.face_bbox)
        else:
            mask_image = predict_hair_mask_image(image)

        mask_quality = summarize_live_mask_quality(analysis, mask_image)
        if keep_only_supported_frames and not support["frame_supported"]:
            continue

        mask_filename = f"live_mask_{index:04d}_{image_path.stem}.png"
        saved_mask_path = mask_dir / mask_filename
        mask_image.save(saved_mask_path)

        records.append(
            {
                "image_path": _relative_project_path(image_path),
                "mask_path": _relative_project_path(saved_mask_path),
                "source_id": image_path.stem,
                "source_dataset": "Live-Webcam-Pseudo",
                "face_detected": bool(analysis.face_detected),
                "frame_supported": bool(support["frame_supported"]),
                "support_reason": support["support_reason"],
                "mask_quality_reason": mask_quality["mask_quality_reason"],
                "mask_reliable": bool(mask_quality["mask_reliable"]),
                "mask_nonzero_ratio": float(mask_quality["mask_nonzero_ratio"]),
                "face_width_ratio": float(support["face_width_ratio"] or 0.0),
                "brightness_mean": float(support["brightness_mean"] or 0.0),
                "yaw_proxy": float(support["yaw_proxy"] or 0.0),
                "pitch_proxy": float(support["pitch_proxy"] or 0.0),
                "roll_degrees": float(support["roll_degrees"] or 0.0),
            }
        )

    seed_path = dataset_dir / "seed_records.jsonl"
    write_jsonl_manifest(records, seed_path)

    summary = summarize_live_segmentation_seed_records(records)
    summary_path = dataset_dir / "seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "records": records,
        "seed_path": seed_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def summarize_live_segmentation_seed_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    supported = sum(1 for row in records if row.get("frame_supported"))
    reliable = sum(1 for row in records if row.get("mask_reliable"))
    support_reason_counts: dict[str, int] = {}
    mask_reason_counts: dict[str, int] = {}
    for row in records:
        support_reason = str(row.get("support_reason", "unknown"))
        mask_reason = str(row.get("mask_quality_reason", "unknown"))
        support_reason_counts[support_reason] = support_reason_counts.get(support_reason, 0) + 1
        mask_reason_counts[mask_reason] = mask_reason_counts.get(mask_reason, 0) + 1

    return {
        "total_frames": total,
        "supported_frames": supported,
        "reliable_masks": reliable,
        "supported_ratio": (supported / total) if total else 0.0,
        "reliable_ratio": (reliable / total) if total else 0.0,
        "support_reason_counts": support_reason_counts,
        "mask_reason_counts": mask_reason_counts,
    }


def split_live_segmentation_seed_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    require_reliable_masks: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    filtered = [
        row for row in records
        if (not require_reliable_masks or row.get("mask_reliable"))
    ]

    shuffled = list(filtered)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_records = shuffled[:train_end]
    val_records = shuffled[train_end:val_end]
    test_records = shuffled[val_end:]

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def export_live_segmentation_split(
    split_records: dict[str, list[dict[str, Any]]],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    dataset_dir = Path(output_dir or LIVE_SEGMENTATION_DATASET_DIR)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for split_name, rows in split_records.items():
        path = dataset_dir / f"{split_name}.jsonl"
        write_jsonl_manifest(rows, path)
        output_paths[split_name] = path

    summary = {
        "train_records": len(split_records.get("train", [])),
        "val_records": len(split_records.get("val", [])),
        "test_records": len(split_records.get("test", [])),
        "total_records": sum(len(rows) for rows in split_records.values()),
    }
    summary_path = dataset_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_paths["summary"] = summary_path
    return output_paths


def load_live_segmentation_records(
    dataset_dir: str | Path | None = None,
    split_name: str = "train",
) -> list[dict[str, Any]]:
    directory = Path(dataset_dir or LIVE_SEGMENTATION_DATASET_DIR)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return read_jsonl_manifest(directory / f"{split_name}.jsonl")


def prepare_live_segmentation_review(
    dataset_dir: str | Path | None = None,
) -> dict[str, Path]:
    directory = Path(dataset_dir or LIVE_SEGMENTATION_DATASET_DIR)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory

    records = read_jsonl_manifest(directory / "seed_records.jsonl")
    review_dir = directory / "review"
    editable_mask_dir = review_dir / "editable_masks"
    editable_mask_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    for row in records:
        source_id = str(row["source_id"])
        original_mask_path = PROJECT_ROOT / str(row["mask_path"])
        editable_mask_path = editable_mask_dir / f"{source_id}.png"
        if original_mask_path.exists() and not editable_mask_path.exists():
            shutil.copy2(original_mask_path, editable_mask_path)

        review_rows.append(
            {
                "source_id": source_id,
                "image_path": str(row["image_path"]),
                "original_mask_path": str(row["mask_path"]),
                "editable_mask_path": _relative_project_path(editable_mask_path),
                "support_reason": str(row.get("support_reason", "")),
                "initial_mask_quality_reason": str(row.get("mask_quality_reason", "")),
                "initial_mask_reliable": str(bool(row.get("mask_reliable", False))).lower(),
                "review_keep": "",
                "mask_fixed": "",
                "review_notes": "",
            }
        )

    review_csv_path = review_dir / "live_segmentation_review_template.csv"
    import csv

    with review_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(review_rows[0].keys()) if review_rows else (
            "source_id",
            "image_path",
            "original_mask_path",
            "editable_mask_path",
            "support_reason",
            "initial_mask_quality_reason",
            "initial_mask_reliable",
            "review_keep",
            "mask_fixed",
            "review_notes",
        ))
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    summary_path = review_dir / "review_prep_summary.json"
    summary_payload = {
        "total_rows": len(review_rows),
        "review_csv_path": _relative_project_path(review_csv_path),
        "editable_mask_dir": _relative_project_path(editable_mask_dir),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "review_csv_path": review_csv_path,
        "editable_mask_dir": editable_mask_dir,
        "summary_path": summary_path,
    }


def apply_live_segmentation_review(
    review_csv_path: str | Path,
    dataset_dir: str | Path | None = None,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Path]:
    directory = Path(dataset_dir or LIVE_SEGMENTATION_DATASET_DIR)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory

    review_csv = Path(review_csv_path)
    if not review_csv.is_absolute():
        review_csv = PROJECT_ROOT / review_csv

    import csv

    approved_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            review_keep = str(row.get("review_keep", "")).strip().lower()
            editable_mask_path = PROJECT_ROOT / str(row.get("editable_mask_path", ""))
            original_mask_path = str(row.get("original_mask_path", ""))
            selected_mask_path = editable_mask_path if editable_mask_path.exists() else (PROJECT_ROOT / original_mask_path)

            payload = {
                "image_path": str(row.get("image_path", "")),
                "mask_path": _relative_project_path(selected_mask_path),
                "source_id": str(row.get("source_id", "")),
                "source_dataset": "Live-Webcam-Reviewed",
                "review_notes": str(row.get("review_notes", "")),
                "mask_fixed": str(row.get("mask_fixed", "")).strip().lower(),
            }

            if review_keep == "yes":
                approved_rows.append(payload)
            else:
                rejected_rows.append(payload)

    split_records = split_live_segmentation_seed_records(
        approved_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        require_reliable_masks=False,
    )
    export_paths = export_live_segmentation_split(split_records, directory)

    reviewed_dir = directory / "reviewed"
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    approved_path = reviewed_dir / "approved_records.jsonl"
    rejected_path = reviewed_dir / "rejected_records.jsonl"
    write_jsonl_manifest(approved_rows, approved_path)
    write_jsonl_manifest(rejected_rows, rejected_path)

    review_summary = {
        "approved_records": len(approved_rows),
        "rejected_records": len(rejected_rows),
        "train_records": len(split_records.get("train", [])),
        "val_records": len(split_records.get("val", [])),
        "test_records": len(split_records.get("test", [])),
    }
    review_summary_path = reviewed_dir / "review_summary.json"
    review_summary_path.write_text(json.dumps(review_summary, indent=2), encoding="utf-8")

    export_paths.update(
        {
            "approved": approved_path,
            "rejected": rejected_path,
            "review_summary": review_summary_path,
        }
    )
    return export_paths
