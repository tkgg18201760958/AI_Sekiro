"""Executes an `Action` as real key/mouse input via pydirectinput.

pydirectinput is used instead of pyautogui because it emits DirectInput scan
codes rather than Windows message-based input -- most game engines (Sekiro
included) only react to the former (see architecture.md section 6).

`dry_run=True` skips all real input and only logs what *would* have been
sent, so the action->key mapping can be sanity-checked without a game
window, a focused input target, or even pydirectinput installed correctly on
a machine without admin rights (DirectInput injection on Windows can require
elevation depending on the target process).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .action_map import Action, load_keymap

logger = logging.getLogger(__name__)


class InputController:
    def __init__(self, dry_run: bool = False, keymap: dict[Action, dict[str, Any]] | None = None):
        self.dry_run = dry_run
        self.keymap = keymap if keymap is not None else load_keymap()
        self._pdi = None
        if not dry_run:
            import pydirectinput

            pydirectinput.FAILSAFE = False
            self._pdi = pydirectinput

    def execute(self, action: int | Action) -> bool:
        """Send the input(s) for `action`. Returns True if it was (or would
        have been, in dry_run) executed, False for an unknown action."""
        action = Action(action)
        spec = self.keymap.get(action)
        if spec is None:
            logger.warning("No keymap entry for action %s; skipping.", action.name)
            return False

        if self.dry_run:
            logger.info("[dry_run] action=%s spec=%s", action.name, spec)
            return True

        self._send(spec)
        logger.info("action=%s spec=%s", action.name, spec)
        return True

    def _send(self, spec: dict[str, Any]) -> None:
        kind = spec["type"]
        if kind == "none":
            return
        elif kind == "mouse":
            self._send_mouse(spec)
        elif kind == "key":
            self._send_key(spec)
        elif kind == "combo":
            self._send_combo(spec)
        else:
            raise ValueError(f"Unknown input spec type: {kind!r}")

    def _send_mouse(self, spec: dict[str, Any]) -> None:
        button = spec["button"]
        duration = spec.get("duration")
        if duration is None:
            self._pdi.click(button=button)
        else:
            self._pdi.mouseDown(button=button)
            time.sleep(duration)
            self._pdi.mouseUp(button=button)

    def _send_key(self, spec: dict[str, Any]) -> None:
        key = spec["key"]
        duration = spec.get("duration")
        if duration is None:
            self._pdi.press(key)
        else:
            self._pdi.keyDown(key)
            time.sleep(duration)
            self._pdi.keyUp(key)

    def _send_combo(self, spec: dict[str, Any]) -> None:
        keys = spec["keys"]
        duration = spec.get("duration", 0.1)
        for key in keys:
            self._pdi.keyDown(key)
        time.sleep(duration)
        for key in reversed(keys):
            self._pdi.keyUp(key)

    def send_raw(self, spec: dict[str, Any], label: str = "raw") -> None:
        """Send an input spec directly, bypassing the Action keymap.

        Used by RestartManager for restart-sequence keys (confirm/skip
        prompts) that aren't part of the combat Action enum, so there's
        still exactly one place ("_send") that knows how to talk to
        pydirectinput.
        """
        if self.dry_run:
            logger.info("[dry_run] %s: %s", label, spec)
            return
        self._send(spec)
        logger.info("%s: %s", label, spec)

    def close(self) -> None:
        """No persistent resources to release; kept for interface symmetry
        with StateReader.close()."""
        return None
