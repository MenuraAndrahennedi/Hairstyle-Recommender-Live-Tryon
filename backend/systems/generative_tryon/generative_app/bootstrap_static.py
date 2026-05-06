from __future__ import annotations

import sys
from pathlib import Path


SUPPORT_ROOT = Path(__file__).resolve().parents[1]


def ensure_static_2d_on_path() -> None:
    static_root = str(SUPPORT_ROOT)
    if static_root not in sys.path:
        sys.path.append(static_root)
