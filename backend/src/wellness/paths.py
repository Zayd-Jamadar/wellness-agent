"""Application paths (XDG dirs). Adapted from toad's paths module."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Final

from xdg_base_dirs import xdg_config_home, xdg_data_home, xdg_state_home

APP_NAME: Final[str] = "wellness"

# Repository root is three parents up from this file:
# src/wellness/paths.py -> src/wellness -> src -> <package root: wellness/>
_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def get_data() -> Path:
    """Return (creating if needed) the application data directory."""
    path = xdg_data_home() / APP_NAME
    with suppress(OSError):
        path.mkdir(0o700, exist_ok=True, parents=True)
    return path


def get_config() -> Path:
    """Return (creating if needed) the application config directory."""
    path = xdg_config_home() / APP_NAME
    with suppress(OSError):
        path.mkdir(0o700, exist_ok=True, parents=True)
    return path


def get_state() -> Path:
    """Return (creating if needed) the application state directory."""
    path = xdg_state_home() / APP_NAME
    with suppress(OSError):
        path.mkdir(0o700, exist_ok=True, parents=True)
    return path


def get_log() -> Path:
    """Return (creating if needed) the log directory."""
    path = get_state() / "logs"
    with suppress(OSError):
        path.mkdir(0o700, exist_ok=True, parents=True)
    return path


def get_db_dir() -> Path:
    """Return (creating if needed) the backend data directory (``backend/data``)."""
    path = _PACKAGE_ROOT / "data"
    with suppress(OSError):
        path.mkdir(0o700, exist_ok=True, parents=True)
    return path


def get_db() -> Path:
    """Path to the single shared SQLite DB (KB index + conversation memory)."""
    return get_db_dir() / "wellness.db"


def get_memory_db() -> Path:
    """Path to the SQLite database holding short-term conversation memory.

    Now backed by the single shared DB (:func:`get_db`).
    """
    return get_db()


def get_prompts_dir() -> Path:
    """Directory holding YAML prompt files (bundled with the package)."""
    return _PACKAGE_ROOT / "prompts"


def get_kb_dir() -> Path:
    """Directory holding the knowledge-base markdown documents.

    Always the backend ``data/kb`` folder (``backend/data/kb``), created if
    needed.
    """
    path = get_db_dir() / "kb"
    with suppress(OSError):
        path.mkdir(0o700, exist_ok=True, parents=True)
    return path
