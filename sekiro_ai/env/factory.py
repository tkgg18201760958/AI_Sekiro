"""Builds a ready-to-use SekiroEnv from CLI-style options.

Shared by train.py and play.py (and usable from a REPL/notebook) so the
"which reader, which controller, live-vs-mock fallback" wiring logic lives
in exactly one place instead of being copy-pasted into every entrypoint --
the same fallback pattern tests/test_state_reader.py already established
for the reader alone, extended here to the full env.
"""
from __future__ import annotations

import logging

from ..controller.input_controller import InputController
from ..state_reader.base import StateReader
from ..state_reader.mock_reader import MockStateReader
from .sekiro_env import SekiroEnv

logger = logging.getLogger(__name__)


def build_reader(live: bool, mock_mode: str, seed: int | None) -> StateReader:
    if live:
        from ..state_reader.pixel_reader import PixelStateReader

        reader = PixelStateReader()
        if reader.is_available():
            logger.info("Found real game window. Using PixelStateReader.")
            return reader
        logger.warning(
            "Real game window not found (or screen calibration unavailable). "
            "Falling back to MockStateReader(mode=%r).",
            mock_mode,
        )

    return MockStateReader(mode=mock_mode, seed=seed)


def build_env(
    live: bool = False,
    mock_mode: str = "scripted",
    seed: int | None = None,
    dry_run: bool = True,
    max_episode_steps: int | None = 2000,
) -> SekiroEnv:
    """`dry_run` defaults to True (no real input sent) since the common case
    -- training against the mock env -- should never need pydirectinput to
    actually work. Pass dry_run=False explicitly once running --live against
    a real game window."""
    reader = build_reader(live, mock_mode, seed)
    controller = InputController(dry_run=dry_run)
    return SekiroEnv(reader, controller, max_episode_steps=max_episode_steps)
