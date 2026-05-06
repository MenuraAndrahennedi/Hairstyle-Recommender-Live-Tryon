from __future__ import annotations

from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SYSTEM_ROOT.parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

FULL_HAIR_ASSET_ROOT = BACKEND_ROOT / "data" / "processed" / "celeba_full_hair_assets"
FULL_HAIR_ASSET_METADATA_DIR = FULL_HAIR_ASSET_ROOT / "metadata"
REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL = (
    FULL_HAIR_ASSET_ROOT
    / "reviewed_render_safe"
    / "kept_assets.jsonl"
)
REVIEWED_ASSET_BANK_JSONL = (
    BACKEND_ROOT
    / "data"
    / "processed"
    / "celeba_hair_rich_assets"
    / "reviewed"
    / "kept_assets_labeled_enriched.jsonl"
)

OUTPUT_ROOT = BACKEND_ROOT / "outputs"
UPLOADS_DIR = OUTPUT_ROOT / "uploads"
DEBUG_DIR = OUTPUT_ROOT / "debug"
LANDMARKS_DIR = DEBUG_DIR / "landmarks"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
TRYON_DIR = OUTPUT_ROOT / "tryon_2d"

MODEL_ASSETS_DIR = BACKEND_ROOT / "models"
MEDIAPIPE_FACE_LANDMARKER_PATH = MODEL_ASSETS_DIR / "face_landmarker.task"
FACE_TO_HAIR_MAPPER_CONFIG_PATH = (
    MODEL_ASSETS_DIR / "face_to_hair_mapper_light" / "mapper_config.json"
)


def media_url_for_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        relative = resolved.relative_to(BACKEND_ROOT)
    except ValueError:
        relative = resolved.relative_to(PROJECT_ROOT)
        if relative.parts and relative.parts[0] == "backend":
            relative = Path(*relative.parts[1:])
    return f"/media/{relative.as_posix()}"
