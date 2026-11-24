"""Permite ejecutar los scripts como `python scripts/<nombre>.py` desde la raíz del repo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
