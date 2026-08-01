"""Stage 6 standalone test: run a random-action agent against SekiroEnv for
many episodes, purely to shake out crashes across the whole mock pipeline
(StateReader -> RewardCalculator -> SekiroEnv -> InputController ->
RestartManager) before any real RL algorithm is involved.

Per roadmap.md, this stage deliberately adds no new module -- it's a
scripted exercise of stage 1-5's existing pieces. Runs against both mock
modes (scripted: reliably reaches a terminal state every episode; random:
fuzzes arbitrary state combinations) since each stresses a different thing.

Usage:
    python tests/test_random_agent.py                       # 20 episodes, both mock modes
    python tests/test_random_agent.py --episodes 50 --mode random
    python tests/test_random_agent.py --max-steps 500
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.controller import InputController
from sekiro_ai.env import SekiroEnv
from sekiro_ai.state_reader.mock_reader import MockStateReader
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("random_agent", "random_agent.log")


def run_episodes(mode: str, episodes: int, max_steps: int, seed: int | None) -> dict:
    reader = MockStateReader(mode=mode, seed=seed)
    controller = InputController(dry_run=True)
    env = SekiroEnv(reader, controller, max_episode_steps=max_steps)

    stats = {"episodes": 0, "terminated": 0, "truncated": 0, "crashed": 0, "total_reward": 0.0, "total_steps": 0}

    for ep in range(episodes):
        # env.reset(seed=...) only seeds the Env's own np_random (gymnasium
        # convention); it doesn't reach MockStateReader's RNG. Vary the
        # reader's seed directly per episode so "random" mode actually
        # fuzzes different trajectories across episodes instead of
        # replaying the same one every time, while staying reproducible for
        # a given --seed.
        reader._seed = None if seed is None else seed + ep
        try:
            obs, info = env.reset(seed=seed)
            ep_reward = 0.0
            steps = 0
            for steps in range(1, max_steps + 1):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    stats["terminated" if terminated else "truncated"] += 1
                    break

            stats["episodes"] += 1
            stats["total_reward"] += ep_reward
            stats["total_steps"] += steps
            logger.info(
                "mode=%s episode=%03d steps=%d reward=%.2f terminated=%s truncated=%s",
                mode, ep, steps, ep_reward, terminated, truncated,
            )
        except Exception:
            stats["crashed"] += 1
            logger.error("mode=%s episode=%03d CRASHED:\n%s", mode, ep, traceback.format_exc())

    env.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-agent smoke test over SekiroEnv.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per mock mode.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode (also env's truncation limit).")
    parser.add_argument("--mode", choices=["scripted", "random", "both"], default="both", help="Which mock mode(s) to run.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    modes = ["scripted", "random"] if args.mode == "both" else [args.mode]

    logger.info("Starting random agent smoke test: modes=%s episodes=%d max_steps=%d", modes, args.episodes, args.max_steps)

    overall_crashed = 0
    for mode in modes:
        stats = run_episodes(mode, args.episodes, args.max_steps, args.seed)
        overall_crashed += stats["crashed"]
        avg_reward = stats["total_reward"] / max(1, stats["episodes"])
        avg_steps = stats["total_steps"] / max(1, stats["episodes"])
        logger.info(
            "mode=%s summary: episodes=%d terminated=%d truncated=%d crashed=%d avg_reward=%.2f avg_steps=%.1f",
            mode, stats["episodes"], stats["terminated"], stats["truncated"], stats["crashed"], avg_reward, avg_steps,
        )

    if overall_crashed:
        logger.error("FAIL: %d episode(s) crashed across all modes.", overall_crashed)
    else:
        logger.info("PASS: no crashes across all modes/episodes.")

    logger.info("Test finished. Log written to logs/random_agent.log")


if __name__ == "__main__":
    main()
