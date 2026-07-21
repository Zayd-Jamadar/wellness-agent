"""YAML prompt loader.

All prompt text lives in ``prompts/*.yml``; no prompt strings are written in
Python. Each YAML file maps keys to string templates.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wellness import paths


@lru_cache(maxsize=None)
def _load_file(name: str) -> dict[str, Any]:
    path: Path = paths.get_prompts_dir() / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must contain a mapping at top level.")
    return data


def get_prompt(name: str, key: str) -> str:
    """Return a raw prompt template string.

    Args:
        name: File stem, e.g. ``"system"`` for ``prompts/system.yml``.
        key: Top-level key within the file.

    Returns:
        The template string.
    """
    data = _load_file(name)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in prompt file '{name}.yml'.")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"Prompt '{name}.{key}' must be a string.")
    return value


def render_prompt(name: str, key: str, /, **variables: Any) -> str:
    """Return a prompt template with ``str.format`` variables substituted.

    Args:
        name: File stem.
        key: Key within the file.
        **variables: Substitution variables.

    Returns:
        The rendered string.
    """
    template = get_prompt(name, key)
    return template.format(**variables)
