from __future__ import annotations

from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SYSTEM_ROOT.parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

OUTPUT_ROOT = BACKEND_ROOT / "outputs" / "generative_tryon"
UPLOADS_DIR = OUTPUT_ROOT / "uploads"
PACKAGE_DIR = OUTPUT_ROOT / "packages"
MODEL_CACHE_DIR = OUTPUT_ROOT / "hf_cache"


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        return PROJECT_ROOT / resolved

    normalized_parts = [part.lower() for part in resolved.parts]
    if "backend" in normalized_parts:
        backend_index = normalized_parts.index("backend")
        return BACKEND_ROOT.joinpath(*resolved.parts[backend_index + 1 :])

    return resolved


def media_url_for_path(path: str | Path) -> str:
    resolved = resolve_project_path(path)
    try:
        relative = resolved.relative_to(BACKEND_ROOT)
    except ValueError:
        try:
            relative = resolved.relative_to(PROJECT_ROOT)
            if relative.parts and relative.parts[0] == "backend":
                relative = Path(*relative.parts[1:])
        except ValueError:
            relative = Path(resolved.name)
    return f"/media/{relative.as_posix()}"
