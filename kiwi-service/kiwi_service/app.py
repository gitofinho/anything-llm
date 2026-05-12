import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app import app  # noqa: E402

__all__ = ["app"]
