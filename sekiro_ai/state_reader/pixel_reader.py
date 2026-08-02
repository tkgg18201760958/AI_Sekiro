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
import time

import numpy as np

from ..utils.config_loader import load_config
from .base import StateReader
from .observation_config import ObservationConfig
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

        image = capture_client_area(self._window)
        hp_range = self.calibration.get("hp_color_hsv_range") or [[0, 100, 90], [10, 255, 255]]
        posture_range = self.calibration.get("posture_color_hsv_range") or [[15, 80, 80], [35, 255, 255]]

        player_hp = bar_fill_ratio(image, tuple(self.calibration["player_hp_bar"]), hp_range)
        boss_hp = bar_fill_ratio(image, tuple(self.calibration["boss_hp_bar"]), hp_range)
        player_posture = bar_fill_ratio(image, tuple(self.calibration["player_posture_bar"]), posture_range)
        boss_posture = bar_fill_ratio(image, tuple(self.calibration["boss_posture_bar"]), posture_range)

        return GameState(
            player_hp=player_hp,
            boss_hp=boss_hp,
            player_posture=player_posture,
            boss_posture=boss_posture,
            player_dead=player_hp <= 0.01,
            boss_dead=boss_hp <= 0.01,
            timestamp=time.time(),
        )

    def read_frame(self) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                f"PixelStateReader could not find a window owned by {self.process_name!r}."
            )
        import cv2

        obs_cfg = ObservationConfig.from_config()
        image = capture_client_area(self._window)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, obs_cfg.frame_size, interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
        self._sct = None
        self._window = None


def bar_fill_ratio(bgr_image, rect: tuple[int, int, int, int], hsv_range) -> float:
    """`rect` 的宽度中，在其高度范围内至少有一个像素匹配 `hsv_range`
    （一对 OpenCV-HSV 的 [[H,S,V]min, [H,S,V]max]）的比例——这是一种
    对水平血条填充比例的估算方式，对血条填充色在亮度/明暗上的轻微变化
    比较鲁棒（每一列只需要有一个匹配的像素，不要求整列都匹配）。"""
    import cv2
    import numpy as np

    x, y, w, h = rect
    roi = bgr_image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(hsv_range[0]), np.array(hsv_range[1])
    mask = cv2.inRange(hsv, lo, hi)
    if mask.shape[1] == 0:
        return 0.0
    column_has_fill = (mask > 0).any(axis=0)
    return float(column_has_fill.mean())


def capture_client_area(window):
    """截取 `window` 客户区（不包括标题栏/边框）的截图，返回 BGR
    ndarray，使用该窗口当前在屏幕上的位置（通过 win32gui.ClientToScreen）
    和尺寸（通过 win32gui.GetClientRect）。"""
    import cv2
    import mss
    import numpy as np
    import win32gui

    hwnd = window._hWnd
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_x, screen_y = win32gui.ClientToScreen(hwnd, (left, top))
    width, height = right - left, bottom - top

    with mss.mss() as sct:
        shot = sct.grab({"left": screen_x, "top": screen_y, "width": width, "height": height})
    bgra = np.array(shot)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


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
