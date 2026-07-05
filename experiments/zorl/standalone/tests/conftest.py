from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
STANDALONE_DIR = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(STANDALONE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
