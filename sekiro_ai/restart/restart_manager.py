"""Detects episode-ending conditions (death/boss kill) and drives the
in-game restart flow back to a fresh fight.

Per architecture.md, this is state-driven, not delay-driven: each step it
only asks "has the current sub-step's expected state condition been met
yet?" before sending the next key in the restart sequence, with a per-step
timeout as a safety net. That's deliberately more robust than "sleep N
seconds then press key" against loading-time / cutscene-length variance.

RestartManager doesn't talk to pydirectinput itself -- it reuses
InputController.send_raw() so there's exactly one code path that knows how
to speak to the input backend (see input_controller.py).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..controller.input_controller import InputController
from ..state_reader.base import StateReader
from ..state_reader.schema import GameState

logger = logging.getLogger(__name__)


@dataclass
class RestartStep:
    """One step of the restart sequence.

    `input_spec`: the InputController-compatible spec to send (see
    action_map.py's spec shapes) -- e.g. a key press to confirm a dialog.
    `wait_until`: predicate over the latest GameState that must become True
    before moving to the next step (e.g. "player_dead is False again", i.e.
    we've respawned). If None, the step is considered done as soon as its
    input is sent (used for steps with no observable state signal, like a
    generic "confirm" tap).
    `timeout`: max seconds to wait for `wait_until` before giving up on this
    step and moving on anyway (safety net against a state signal that never
    fires, e.g. because the reader is temporarily wrong).
    """

    input_spec: dict
    wait_until: Optional[Callable[[GameState], bool]] = None
    timeout: float = 15.0
    label: str = "restart_step"


# Default sequence: after death, Sekiro shows a "You Died" prompt, then an
# idol/respawn menu. A single confirm tap advances through most of this;
# the second step waits for player_dead to clear (i.e. we're back in
# control) as the real completion signal, with a generous timeout for the
# loading screen.
DEFAULT_RESTART_SEQUENCE: list[RestartStep] = [
    RestartStep(
        input_spec={"type": "key", "key": "e"},
        wait_until=None,
        timeout=3.0,
        label="confirm_death_prompt",
    ),
    RestartStep(
        input_spec={"type": "key", "key": "e"},
        wait_until=lambda s: not s.player_dead,
        timeout=15.0,
        label="confirm_respawn_and_wait_for_control",
    ),
]


class RestartManager:
    def __init__(
        self,
        reader: StateReader,
        controller: InputController,
        sequence: list[RestartStep] | None = None,
        poll_interval: float = 0.2,
    ):
        self.reader = reader
        self.controller = controller
        self.sequence = sequence if sequence is not None else DEFAULT_RESTART_SEQUENCE
        self.poll_interval = poll_interval

    @staticmethod
    def needs_restart(state: GameState) -> bool:
        return state.player_dead or state.boss_dead

    def run(self) -> GameState:
        """Execute the restart sequence, returning the state once complete.

        Blocking: polls `reader.read()` between steps. Safe to call against
        MockStateReader (no real sleeping needed there beyond poll_interval)
        or a real reader once stage 8 pixel calibration exists.
        """
        logger.info("Restart sequence starting (%d steps).", len(self.sequence))
        state = self.reader.read()

        for step in self.sequence:
            self.controller.send_raw(step.input_spec, label=step.label)

            if step.wait_until is None:
                continue

            deadline = time.monotonic() + step.timeout
            while time.monotonic() < deadline:
                state = self.reader.read()
                if step.wait_until(state):
                    logger.info("Restart step %r condition met.", step.label)
                    break
                time.sleep(self.poll_interval)
            else:
                logger.warning(
                    "Restart step %r timed out after %.1fs; continuing anyway.",
                    step.label,
                    step.timeout,
                )

        if hasattr(self.reader, "reset"):
            state = self.reader.reset()
        else:
            state = self.reader.read()

        logger.info("Restart sequence finished.")
        return state
