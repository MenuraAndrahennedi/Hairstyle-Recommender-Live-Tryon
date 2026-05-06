from __future__ import annotations

from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SYSTEM_ROOT.parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

OUTPUT_ROOT = BACKEND_ROOT / "outputs" / "generative_tryon"
UPLOADS_DIR = OUTPUT_ROOT / "uploads"
PACKAGE_DIR = OUTPUT_ROOT / "packages"
MODEL_CACHE_DIR = OUTPUT_ROOT / "hf_cache"


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
