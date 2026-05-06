from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
AUTO_APP_ROOT = (
    APP_ROOT.parent / "systems" / "static_auto_tryon" / "auto_app"
)

__path__ = [str(APP_ROOT), str(AUTO_APP_ROOT)]
