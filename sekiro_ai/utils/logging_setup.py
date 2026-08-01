"""Shared logging setup so every module/test script logs consistently.

Each module gets its own log file under logs/ (e.g. logs/state_reader.log)
plus console output, so a module's behavior can be inspected in isolation
without digging through a single combined log.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def get_logger(name: str, filename: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured, avoid duplicate handlers

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_path = LOG_DIR / (filename or f"{name}.log")
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
