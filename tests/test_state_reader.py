"""Stage 1 standalone test: exercise the State Reader module in isolation.

Usage:
    python tests/test_state_reader.py                     # mock, scripted mode
    python tests/test_state_reader.py --mode random        # mock, random mode
    python tests/test_state_reader.py --live               # try real pixel reader first
    python tests/test_state_reader.py --steps 20 --interval 0.2

With --live, the script tries PixelStateReader first; if the game window
isn't found (or screen-region calibration isn't done yet), it logs a warning
and falls back to MockStateReader automatically so the script never just
dies.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader import MockStateReader, StateReader
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("state_reader", "state_reader.log")

FRAME_PREVIEW_DIR = Path(__file__).resolve().parent.parent / "logs" / "frame_preview"


def build_reader(live: bool, mode: str, seed: int | None) -> StateReader:
    if live:
        from sekiro_ai.state_reader.pixel_reader import PixelStateReader

        reader = PixelStateReader()
        if reader.is_available():
            logger.info("Found real game window. Using PixelStateReader.")
            return reader
        logger.warning(
            "Real game window not found (or screen calibration unavailable). "
            "Falling back to MockStateReader(mode=%r).",
            mode,
        )

    return MockStateReader(mode=mode, seed=seed)


def format_state(state) -> str:
    d = state.to_dict()
    return (
        f"hp(p/b)={d['player_hp']:.2f}/{d['boss_hp']:.2f} "
        f"posture(p/b)={d['player_posture']:.2f}/{d['boss_posture']:.2f} "
        f"dist={d['distance']:.2f} boss_action={d['boss_action']!r} "
        f"hit={d['player_hit']} parry={d['can_parry']} "
        f"dead(p/b)={d['player_dead']}/{d['boss_dead']}"
    )


def save_frame_previews(reader, n: int) -> None:
    import cv2

    if FRAME_PREVIEW_DIR.exists():
        shutil.rmtree(FRAME_PREVIEW_DIR)
    FRAME_PREVIEW_DIR.mkdir(parents=True)

    for i in range(n):
        frame = reader.read_frame()
        path = FRAME_PREVIEW_DIR / f"frame_{i:03d}.png"
        cv2.imwrite(str(path), frame)
    logger.info("Saved %d frame(s) to %s", n, FRAME_PREVIEW_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the State Reader module.")
    parser.add_argument("--live", action="store_true", help="Try a real PixelStateReader before falling back to mock.")
    parser.add_argument("--mode", choices=["scripted", "random"], default="scripted", help="Mock reader mode.")
    parser.add_argument("--steps", type=int, default=40, help="Number of read() calls to perform.")
    parser.add_argument("--interval", type=float, default=0.15, help="Seconds to sleep between reads.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible mock output.")
    parser.add_argument("--save-frames", type=int, default=0, help="Save N read_frame() outputs as PNGs to logs/frame_preview/ for visual inspection.")
    args = parser.parse_args()

    reader = build_reader(args.live, args.mode, args.seed)
    logger.info("Starting state reader test: reader=%s steps=%d interval=%.2fs", type(reader).__name__, args.steps, args.interval)

    if hasattr(reader, "reset"):
        reader.reset()

    if args.save_frames > 0:
        save_frame_previews(reader, args.save_frames)

    try:
        for i in range(args.steps):
            state = reader.read()
            logger.info("step=%02d %s", i, format_state(state))
            if state.boss_dead or state.player_dead:
                logger.info("Episode-ending condition reached (boss_dead=%s player_dead=%s).", state.boss_dead, state.player_dead)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        reader.close()
        logger.info("Test finished. Log written to logs/state_reader.log")


if __name__ == "__main__":
    main()
