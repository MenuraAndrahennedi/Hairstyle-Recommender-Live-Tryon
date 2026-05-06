from __future__ import annotations

import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parent
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.append(str(SYSTEM_ROOT))
