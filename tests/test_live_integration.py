"""Stage 8 standalone test: pair real state reading with real input sending.

Unlike test_state_reader.py (read-only, mock-fallback) and test_controller.py
(send-only), this script does both against the *real* game in one run: read
a baseline state, send one real Action, wait for it to land, read again, and
print the before/after delta so a human can see whether what the pipeline
"saw" actually matches what the input did (e.g. sending ATTACK should show
boss_posture go up).

Usage:
    python tests/test_live_integration.py                       # ATTACK, real input
    python tests/test_live_integration.py --action 2             # GUARD
    python tests/test_live_integration.py --action 1 --dry-run   # log the input spec, don't send it
    python tests/test_live_integration.py --before-samples 5 --after-samples 5 --settle 0.5

This requires a real Sekiro window (PixelStateReader.is_available()) and a
fully filled-in config.yaml `calibration` section -- see docs/live_game.md.
As of this writing, PixelStateReader.read() unconditionally raises
NotImplementedError (the mss-capture + OpenCV pipeline itself isn't written
yet), so running this script today will fail fast with a clear message
pointing here instead of silently doing nothing. Once read() is implemented,
this script becomes the way to confirm recognition and input are actually
wired together correctly, not just individually functional.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.controller import Action, InputController
from sekiro_ai.state_reader.pixel_reader import PixelStateReader
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("live_integration", "live_integration.log")

# Fields worth diffing; player_pos/boss_pos are omitted since pixel reading
# can't recover them (see docs/live_game.md) and they'd just show as an
# unchanging (0.0, 0.0, 0.0) placeholder.
DIFF_FIELDS = (
    "player_hp",
    "player_posture",
    "boss_hp",
    "boss_posture",
    "distance",
    "boss_action",
    "player_hit",
    "can_parry",
    "player_dead",
    "boss_dead",
)


def sample_states(reader: PixelStateReader, count: int, interval: float, label: str) -> list:
    states = []
    for i in range(count):
        state = reader.read()
        d = state.to_dict()
        logger.info(
            "%s[%d] hp(p/b)=%.2f/%.2f posture(p/b)=%.2f/%.2f boss_action=%r",
            label, i, d["player_hp"], d["boss_hp"], d["player_posture"], d["boss_posture"], d["boss_action"],
        )
        states.append(d)
        if i < count - 1:
            time.sleep(interval)
    return states


def print_delta(before: dict, after: dict) -> None:
    logger.info("---- before -> after delta ----")
    for field in DIFF_FIELDS:
        b, a = before[field], after[field]
        if isinstance(b, float):
            change = a - b
            marker = "" if abs(change) < 1e-6 else (" (+)" if change > 0 else " (-)")
            logger.info("  %-14s %.3f -> %.3f  delta=%+.3f%s", field, b, a, change, marker)
        else:
            marker = "" if b == a else "  <-- changed"
            logger.info("  %-14s %s -> %s%s", field, b, a, marker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair real state reading with real input sending.")
    parser.add_argument("--action", type=int, default=Action.ATTACK.value, help="Action id (0-6) to send between the before/after reads.")
    parser.add_argument("--dry-run", action="store_true", help="Log the input spec instead of actually sending it (reads still happen for real).")
    parser.add_argument("--before-samples", type=int, default=3, help="Number of read() calls before sending the action.")
    parser.add_argument("--after-samples", type=int, default=3, help="Number of read() calls after sending the action.")
    parser.add_argument("--interval", type=float, default=0.15, help="Seconds between consecutive reads within a before/after batch.")
    parser.add_argument("--settle", type=float, default=0.4, help="Seconds to wait after sending the action before the 'after' reads start.")
    args = parser.parse_args()

    action = Action(args.action)
    reader = PixelStateReader()
    controller = InputController(dry_run=args.dry_run)

    try:
        if not reader.is_available():
            logger.error(
                "No window owned by %r found. Launch Sekiro and focus its window, then re-run. "
                "(This script intentionally does not fall back to MockStateReader -- a mock "
                "read paired with a real input would just be misleading.)",
                reader.process_name,
            )
            sys.exit(1)

        missing = reader.missing_calibration()
        if missing:
            logger.error(
                "config.yaml's `calibration` section is missing: %s. Fill these in first -- "
                "see docs/configuration.md and docs/live_game.md.",
                missing,
            )
            sys.exit(1)

        logger.info(
            "Reading baseline (%d samples), then sending %s (dry_run=%s), then reading again (%d samples).",
            args.before_samples, action.name, args.dry_run, args.after_samples,
        )

        try:
            before = sample_states(reader, args.before_samples, args.interval, "before")
        except NotImplementedError as exc:
            logger.error(
                "PixelStateReader.read() is not implemented yet, so this script can't do "
                "anything real today: %s\nSee docs/live_game.md for what's left to build "
                "(mss capture + OpenCV bar-fill / template-match code in pixel_reader.py).",
                exc,
            )
            sys.exit(1)

        if not controller.dry_run:
            logger.info("Sending %s now -- make sure the game window has focus.", action.name)
        controller.execute(action)

        time.sleep(args.settle)
        after = sample_states(reader, args.after_samples, args.interval, "after")

        print_delta(before[-1], after[-1])
        logger.info(
            "Compare the delta above against what you'd expect for %s (e.g. ATTACK should "
            "raise boss_posture if it landed). This is a human judgment call, not a pass/fail "
            "assertion -- the pixel-reading pipeline has no ground truth to check itself against.",
            action.name,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        reader.close()
        controller.close()
        logger.info("Test finished. Log written to logs/live_integration.log")


if __name__ == "__main__":
    main()
