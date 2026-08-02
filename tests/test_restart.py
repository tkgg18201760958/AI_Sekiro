"""Stage 3 standalone test: exercise the Restart Manager module in isolation.

Usage:
    python tests/test_restart.py                    # simulate player death, dry-run input
    python tests/test_restart.py --scenario boss     # simulate boss death instead
    python tests/test_restart.py --send-real         # actually send restart keys (needs focus!)

Uses a tiny scripted StateReader stub (not MockStateReader) because the
roadmap's acceptance test wants an explicit player_dead=True -> False
transition to verify RestartManager's wait_until logic, and
MockStateReader's scripted fight trajectory only ever ends in boss_dead.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.controller import InputController
from sekiro_ai.restart import RestartManager
from sekiro_ai.state_reader.base import StateReader
from sekiro_ai.state_reader.schema import GameState
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("restart", "restart.log")


class ScriptedDeathReader(StateReader):
    """Reports dead=True for `dead_reads` calls, then flips to False.

    Models "you died -> confirm prompts -> loading -> back in control" as a
    fixed number of read() calls, which is enough to exercise
    RestartManager.run()'s poll loop and wait_until predicate without any
    real timing dependency.
    """

    def __init__(self, scenario: str, dead_reads: int = 3):
        self.scenario = scenario
        self.dead_reads = dead_reads
        self._reads = 0

    def read(self) -> GameState:
        state = GameState(timestamp=time.time())
        self._reads += 1
        still_dead = self._reads <= self.dead_reads
        if self.scenario == "player":
            state.player_dead = still_dead
        else:
            state.boss_dead = still_dead
        return state

    def reset(self) -> GameState:
        logger.info("ScriptedDeathReader.reset() called -- new episode's initial state.")
        self._reads = 0
        return GameState(timestamp=time.time())

    def read_frame(self):
        import numpy as np
        return np.zeros((84, 84), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Restart Manager module.")
    parser.add_argument("--scenario", choices=["player", "boss"], default="player", help="Which death condition to simulate.")
    parser.add_argument("--dead-reads", type=int, default=3, help="How many read() calls report dead=True before clearing.")
    parser.add_argument("--send-real", action="store_true", help="Actually send restart keys instead of dry-run logging only.")
    parser.add_argument("--poll-interval", type=float, default=0.1, help="Seconds between state polls while waiting on a restart step.")
    args = parser.parse_args()

    reader = ScriptedDeathReader(args.scenario, dead_reads=args.dead_reads)
    controller = InputController(dry_run=not args.send_real)
    manager = RestartManager(reader, controller, poll_interval=args.poll_interval)

    state = reader.read()
    logger.info(
        "Initial state: player_dead=%s boss_dead=%s needs_restart=%s",
        state.player_dead,
        state.boss_dead,
        RestartManager.needs_restart(state),
    )

    if not RestartManager.needs_restart(state):
        logger.error("Test setup bug: initial state doesn't need a restart.")
        return

    new_state = manager.run()
    logger.info(
        "Post-restart state: player_dead=%s boss_dead=%s (should both be False -- fresh episode)",
        new_state.player_dead,
        new_state.boss_dead,
    )

    if new_state.player_dead or new_state.boss_dead:
        logger.error("FAIL: restart did not produce a fresh initial state.")
    else:
        logger.info("PASS: restart sequence completed and returned a fresh state.")

    controller.close()
    reader.close()
    logger.info("Test finished. Log written to logs/restart.log")


if __name__ == "__main__":
    main()
