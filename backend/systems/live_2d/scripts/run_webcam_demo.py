from __future__ import annotations

import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SYSTEM_ROOT.parents[1]

for path in (BACKEND_ROOT, SYSTEM_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from systems.live_2d.app.core.demo_engine import Live2DDemoEngine


def main() -> None:
    engine = Live2DDemoEngine()
    print("Live 2D project engine loaded.")
    print(f"System root: {SYSTEM_ROOT}")
    engine.run_webcam()


if __name__ == "__main__":
    main()
