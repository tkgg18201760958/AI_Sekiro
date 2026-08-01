"""Stage 4 standalone test: exercise the Reward Calculator with hand-built
(prev_state, curr_state) sample pairs.

Usage:
    python tests/test_reward.py

Each sample below documents the sign it's expected to produce (per
roadmap.md's acceptance criterion) and asserts it, so a bad weight edit or
a sign-flip bug fails loudly instead of just "looking plausible" in a log.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.reward import RewardCalculator
from sekiro_ai.state_reader.schema import GameState
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("reward", "reward.log")


def base_state(**overrides) -> GameState:
    return GameState(**overrides)


SAMPLES = [
    (
        "boss takes damage, nothing else changes -> reward should be positive",
        base_state(boss_hp=0.5),
        base_state(boss_hp=0.4),
        lambda r: r > 0,
    ),
    (
        "player takes damage, nothing else changes -> reward should be negative",
        base_state(player_hp=0.5),
        base_state(player_hp=0.4),
        lambda r: r < 0,
    ),
    (
        "both take equal damage -> boss_hp_delta weight (100) == player_hp_delta weight (100), "
        "so net should be ~0 (just the small per-step cost)",
        base_state(player_hp=0.5, boss_hp=0.5),
        base_state(player_hp=0.4, boss_hp=0.4),
        lambda r: abs(r) < 0.1,
    ),
    (
        "player_hit flag set -> extra flat penalty on top of any hp change",
        base_state(player_hp=0.5, player_hit=False),
        base_state(player_hp=0.5, player_hit=True),
        lambda r: r < 0,
    ),
    (
        "boss posture increases (getting staggered) -> positive, even with no hp change",
        base_state(boss_posture=0.2),
        base_state(boss_posture=0.6),
        lambda r: r > 0,
    ),
    (
        "player posture increases (we're the ones getting staggered) -> negative",
        base_state(player_posture=0.2),
        base_state(player_posture=0.6),
        lambda r: r < 0,
    ),
    (
        "boss_dead transitions False->True -> large positive terminal bonus dominates",
        base_state(boss_hp=0.05, boss_dead=False),
        base_state(boss_hp=0.0, boss_dead=True),
        lambda r: r > 100,
    ),
    (
        "player_dead transitions False->True -> large negative terminal penalty dominates",
        base_state(player_hp=0.05, player_dead=False),
        base_state(player_hp=0.0, player_dead=True),
        lambda r: r < -100,
    ),
    (
        "no change at all -> reward is just the small negative per-step cost (anti-stalling)",
        base_state(),
        base_state(),
        lambda r: -0.1 < r < 0,
    ),
    (
        "already-dead -> still-dead (no new transition) -> no terminal bonus re-applied",
        base_state(boss_dead=True, boss_hp=0.0),
        base_state(boss_dead=True, boss_hp=0.0),
        lambda r: -0.1 < r < 0,
    ),
]


def main() -> None:
    calc = RewardCalculator()
    logger.info("Starting reward calculator test: %d samples, weights=%s", len(SAMPLES), calc.weights)

    failures = 0
    for description, prev, curr, expect in SAMPLES:
        reward = calc.compute(prev, curr)
        ok = expect(reward)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        logger.info("[%s] reward=%.3f :: %s", status, reward, description)

    if failures:
        logger.error("%d/%d samples FAILED expected-sign check.", failures, len(SAMPLES))
    else:
        logger.info("All %d samples matched expected sign/range.", len(SAMPLES))

    logger.info("Test finished. Log written to logs/reward.log")


if __name__ == "__main__":
    main()
