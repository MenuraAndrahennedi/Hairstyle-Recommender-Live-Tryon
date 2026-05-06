from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = APP_ROOT.parent
BACKEND_ROOT = SYSTEM_ROOT.parents[1]
MODEL_ROOT = BACKEND_ROOT / "models"
SCRIPT_ROOT = SYSTEM_ROOT / "scripts"

FACE_LANDMARKER_PATH = MODEL_ROOT / "face_landmarker.task"

WEBCAM_RUNNER_PATH = SCRIPT_ROOT / "run_webcam_demo.py"
