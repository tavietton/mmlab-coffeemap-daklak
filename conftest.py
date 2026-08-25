"""Pytest root conftest: make `src/` importable as `coffeemap`, matching the
same sys.path convention every notebook already uses (see notebooks/00...ipynb)."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
