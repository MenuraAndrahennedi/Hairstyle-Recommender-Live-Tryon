from __future__ import annotations

from pathlib import Path

_STATIC_APP_ROOT = (
    Path(__file__).resolve().parents[1] / "systems" / "static_2d" / "app"
)

__path__ = [str(_STATIC_APP_ROOT)]
