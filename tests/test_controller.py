"""Stage 2 standalone test: exercise the Action Controller module in isolation.

Usage:
    python tests/test_controller.py --dry-run              # print all 7 actions, send nothing
    python tests/test_controller.py --dry-run --action 1   # print just ATTACK's spec
    python tests/test_controller.py --action 1             # actually send ATTACK's input
    python tests/test_controller.py --delay 3               # 3s countdown before sending, per action

Real (non dry-run) runs give you `--delay` seconds before each action fires,
so you can alt-tab to a notepad window / the game window and place focus
there -- key/mouse events go to whatever window has focus, this script does
not (and, per architecture.md, will not) attempt to force focus itself.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.controller import Action, InputController
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("controller", "controller.log")


def run_one(controller: InputController, action: Action, delay: float) -> None:
    spec = controller.keymap.get(action)
    if delay > 0 and not controller.dry_run:
        logger.info("Sending %s (%s) in %.1fs -- focus the target window now.", action.name, spec, delay)
        time.sleep(delay)
    ok = controller.execute(action)
    if ok:
        logger.info("%s%s: %s -> %s", "[dry_run] " if controller.dry_run else "", "OK", action.name, spec)
    else:
        logger.error("Failed to execute action %s (no keymap entry).", action.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Action Controller module.")
    parser.add_argument("--action", type=int, default=None, help="Run a single Action id (0-6). Omit to run all actions in order.")
    parser.add_argument("--dry-run", action="store_true", help="Only log what would be sent; do not send real input.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait before each real (non dry-run) action, to allow re-focusing a target window.")
    args = parser.parse_args()

    controller = InputController(dry_run=args.dry_run)
    logger.info("Starting controller test: dry_run=%s", args.dry_run)

    try:
        if args.action is not None:
            actions = [Action(args.action)]
        else:
            actions = list(Action)

        for action in actions:
            run_one(controller, action, args.delay)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        controller.close()
        logger.info("Test finished. Log written to logs/controller.log")


if __name__ == "__main__":
    main()
