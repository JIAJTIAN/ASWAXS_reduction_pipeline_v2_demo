"""Run the ASWAXS live curve viewer without installing the project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aswaxs_live.viewer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

