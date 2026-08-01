"""Stage 7: train a PPO agent against SekiroEnv.

Usage:
    python train.py --total-timesteps 1000                  # mock env, quick smoke test
    python train.py --total-timesteps 200000 --mode scripted
    python train.py --live --total-timesteps 200000          # real game (stage 8)

Saves the final model to models/<run-name>.zip and streams metrics to
TensorBoard under logs/tensorboard/<run-name>/ (per architecture.md's
"SB3 自带 tensorboard_log 参数直接接入 TensorBoard"). View with:
    tensorboard --logdir logs/tensorboard
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sekiro_ai.env import build_env
from sekiro_ai.logging import EpisodeCsvLogger
from sekiro_ai.utils.logging_setup import get_logger

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
TENSORBOARD_DIR = ROOT / "logs" / "tensorboard"
EPISODES_DIR = ROOT / "logs" / "episodes"

logger = get_logger("train", "train.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent on SekiroEnv.")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="PPO total_timesteps.")
    parser.add_argument("--run-name", type=str, default="ppo_sekiro", help="Name for the saved model and tensorboard run.")
    parser.add_argument("--mode", choices=["scripted", "random"], default="scripted", help="Mock reader mode (ignored with --live).")
    parser.add_argument("--live", action="store_true", help="Try a real PixelStateReader before falling back to mock.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument("--max-episode-steps", type=int, default=2000, help="Truncation limit per episode.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to an existing model .zip to continue training from.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env(
        live=args.live,
        mock_mode=args.mode,
        seed=args.seed,
        dry_run=not args.live,
        max_episode_steps=args.max_episode_steps,
    )
    # info_keywords=("state",) tells Monitor to copy env.step()'s "state"
    # key into the info dict it passes upward -- without this, Monitor only
    # keeps the reward/length it computes itself, and EpisodeCsvLogger would
    # have no way to know the death reason for its CSV row.
    env = Monitor(env, info_keywords=("state",))

    if args.resume_from:
        logger.info("Resuming training from %s", args.resume_from)
        model = PPO.load(args.resume_from, env=env, tensorboard_log=str(TENSORBOARD_DIR))
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(TENSORBOARD_DIR))

    logger.info(
        "Starting PPO training: total_timesteps=%d run_name=%s live=%s mode=%s",
        args.total_timesteps, args.run_name, args.live, args.mode,
    )

    episode_csv = EPISODES_DIR / f"{args.run_name}.csv"
    episode_logger = EpisodeCsvLogger(episode_csv)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=args.run_name,
            progress_bar=False,
            callback=episode_logger,
        )
    finally:
        save_path = MODELS_DIR / f"{args.run_name}.zip"
        model.save(str(save_path))
        logger.info("Model saved to %s", save_path)
        env.close()

    logger.info("Training finished. Log written to logs/train.log")


if __name__ == "__main__":
    main()
