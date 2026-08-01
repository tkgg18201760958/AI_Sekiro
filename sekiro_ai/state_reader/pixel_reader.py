"""Real StateReader backed by reading Sekiro's screen pixels.

Replaces the earlier memory-reading approach (see docs/architecture.md and
docs/risks.md for why: no stable public memory offsets exist for Sekiro,
and the reference implementation that does read memory relies on pattern
scanning + runtime code injection, a much higher-effort and version-fragile
technique than reading the HP/posture bars the game already renders on
screen for the player).

This is a stage-1 placeholder: it can detect whether the game process is
running (`is_available()`), by matching the owning process's executable
name rather than window title (title substring matching produces false
positives, see PROCESS_NAME comment below). The actual screenshot region
calibration and
color-threshold based HP/posture/"危" detection is stage 8 work, done once
against a real running game at a known resolution/UI scale. Until that
calibration exists, `read()` raises NotImplementedError so callers know to
fall back to `MockStateReader` rather than silently getting wrong data.
"""
from __future__ import annotations

import logging

from ..utils.config_loader import load_config
from .base import StateReader
from .schema import GameState

logger = logging.getLogger(__name__)

REQUIRED_CALIBRATION_KEYS = (
    "player_hp_bar",
    "player_posture_bar",
    "boss_hp_bar",
    "boss_posture_bar",
)

# NOTE: the actual executable name of the Sekiro client process. Matching on
# this (rather than window title) avoids false positives: pygetwindow's
# title-based lookup does a case-insensitive *substring* match, so a window
# titled e.g. "sekiro-rl-agent-design - VSCode" would otherwise match a
# search for "Sekiro" even though the game isn't running at all.
PROCESS_NAME = "sekiro.exe"


class PixelStateReader(StateReader):
    def __init__(self, process_name: str = PROCESS_NAME):
        self.process_name = process_name
        self._window = None
        self._sct = None
        self.calibration = load_config().get("calibration", {})
        self._find_window()

    def missing_calibration(self) -> list[str]:
        """Calibration keys from config.yaml's `calibration` section that are
        still null. Non-empty until someone measures bar rects against a
        real running game window (see config.yaml's comments)."""
        return [key for key in REQUIRED_CALIBRATION_KEYS if not self.calibration.get(key)]

    def _find_window(self) -> None:
        try:
            import pygetwindow as gw
            import win32process
        except ImportError:
            logger.warning("pygetwindow/pywin32 is not installed; PixelStateReader unavailable.")
            return

        for window in gw.getAllWindows():
            try:
                _, pid = win32process.GetWindowThreadProcessId(window._hWnd)
                exe_name = _process_exe_name(pid)
            except Exception:
                continue
            if exe_name and exe_name.lower() == self.process_name.lower():
                self._window = window
                return

        logger.info("No window owned by process %r found.", self.process_name)
        self._window = None

    def is_available(self) -> bool:
        return self._window is not None

    def read(self) -> GameState:
        if not self.is_available():
            raise RuntimeError(
                f"PixelStateReader could not find a window owned by {self.process_name!r}."
            )
        missing = self.missing_calibration()
        if missing:
            raise NotImplementedError(
                "PixelStateReader is not calibrated yet -- config.yaml's "
                f"`calibration` section is missing: {missing}. This must be "
                "measured against the live game window (see the comments "
                "above `calibration:` in config.yaml for the manual steps); "
                "it cannot be filled in from code. Use MockStateReader "
                "until calibration is done."
            )
        raise NotImplementedError(
            "Calibration values are present but the screenshot/color-mask "
            "reading pipeline (mss capture + OpenCV bar-fill measurement) "
            "is not implemented yet. Use MockStateReader for now."
        )

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
        self._sct = None
        self._window = None


def _process_exe_name(pid: int) -> str | None:
    import win32api
    import win32con
    import win32process

    handle = win32api.OpenProcess(
        win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid
    )
    try:
        path = win32process.GetModuleFileNameEx(handle, 0)
    finally:
        win32api.CloseHandle(handle)
    return path.rsplit("\\", 1)[-1]
