"""Per-episode CSV logging, as an SB3 BaseCallback.

Per architecture.md: SB3's own `tensorboard_log` already covers step-level
training metrics (loss, value estimates, etc.) -- this module's job is
specifically the *episode-level* summary (death reason, kill time) that SB3
doesn't track on its own, written to a plain CSV that's easy to load into
pandas/Excel without needing a TensorBoard event-file parser.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

CSV_FIELDS = [
    "episode",
    "timesteps",
    "wall_time",
    "steps",
    "total_reward",
    "end_reason",
    "final_boss_hp",
    "final_player_hp",
]


class EpisodeCsvLogger(BaseCallback):
    """Appends one CSV row per completed episode.

    Relies on stable_baselines3.common.monitor.Monitor wrapping the env
    (train.py always does this) for episode boundaries -- specifically the
    `"episode"` key Monitor puts in `info` on the step where an episode
    ends, which contains that episode's total reward/length.
    """

    def __init__(self, csv_path: str | Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self._episode_count = 0
        self._t_start = time.time()

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.csv_path.exists()
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            state = info.get("state", {})
            if state.get("boss_dead"):
                end_reason = "boss_dead"
            elif state.get("player_dead"):
                end_reason = "player_dead"
            else:
                end_reason = "truncated"

            self._episode_count += 1
            self._writer.writerow(
                {
                    "episode": self._episode_count,
                    "timesteps": self.num_timesteps,
                    "wall_time": round(time.time() - self._t_start, 2),
                    "steps": episode_info["l"],
                    "total_reward": round(episode_info["r"], 4),
                    "end_reason": end_reason,
                    "final_boss_hp": round(state.get("boss_hp", -1.0), 4),
                    "final_player_hp": round(state.get("player_hp", -1.0), 4),
                }
            )
            self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()
