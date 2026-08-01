"""Stage 7: load a trained PPO model and run it against SekiroEnv.

Usage:
    python play.py --model models/ppo_sekiro.zip                       # mock env
    python play.py --model models/ppo_sekiro.zip --live                # real game (stage 8)
    python play.py --model models/ppo_sekiro.zip --episodes 5 --deterministic
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from sekiro_ai.env import build_env
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("play", "play.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained PPO agent on SekiroEnv.")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained model .zip.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument("--mode", choices=["scripted", "random"], default="scripted", help="Mock reader mode (ignored with --live).")
    parser.add_argument("--live", action="store_true", help="Try a real PixelStateReader before falling back to mock.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions instead of sampling.")
    parser.add_argument("--max-episode-steps", type=int, default=2000, help="Truncation limit per episode.")
    args = parser.parse_args()

    env = build_env(
        live=args.live,
        mock_mode=args.mode,
        dry_run=not args.live,
        max_episode_steps=args.max_episode_steps,
    )
    model = PPO.load(args.model, env=env)
    logger.info("Loaded model from %s. Playing %d episode(s), live=%s deterministic=%s", args.model, args.episodes, args.live, args.deterministic)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                steps += 1
            logger.info("episode=%d steps=%d total_reward=%.2f terminated=%s truncated=%s", ep, steps, total_reward, terminated, truncated)
    finally:
        env.close()

    logger.info("Play finished. Log written to logs/play.log")


if __name__ == "__main__":
    main()
