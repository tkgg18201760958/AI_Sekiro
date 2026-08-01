"""Tiny YAML loader for config/config.yaml, shared by every module that
needs configuration (key mappings, reward weights, training hyperparams).

Kept deliberately dumb: just parses YAML into a plain dict. Each module owns
its own top-level key (e.g. `controller:`) and is responsible for validating
its own section -- this loader doesn't know or care what's inside.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
