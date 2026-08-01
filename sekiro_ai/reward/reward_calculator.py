"""Turns a (prev_state, curr_state) pair into a scalar reward.

Per architecture.md, reward is purely a function of the state *delta* --
it never talks to StateReader/InputController itself, which is what makes
it trivially testable with hand-built GameState pairs (see
tests/test_reward.py) and reusable unchanged once real pixel-based states
replace mock ones.

Sign convention (roadmap.md's acceptance criterion): boss losing HP is
positive, player losing HP is negative. Everything else is an extension of
that same idea -- posture symmetric to HP, plus terminal bonuses/penalties
and small per-step shaping terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..state_reader.schema import GameState
from ..utils.config_loader import load_config

# Fallback weights, used when config.yaml has no reward.weights section.
DEFAULT_WEIGHTS: dict[str, float] = {
    "boss_hp_delta": 100.0,       # per unit of boss_hp lost (boss_hp in [0,1])
    "player_hp_delta": 100.0,     # per unit of player_hp lost
    "boss_posture_delta": 20.0,   # per unit of boss_posture gained (staggering it)
    "player_posture_delta": 20.0, # per unit of player_posture gained
    "player_hit": -5.0,           # flat penalty for taking a hit this step
    "boss_dead": 500.0,           # terminal bonus
    "player_dead": -500.0,        # terminal penalty
    "step": -0.01,                # small per-step cost, discourages stalling
}


@dataclass
class RewardWeights:
    boss_hp_delta: float = DEFAULT_WEIGHTS["boss_hp_delta"]
    player_hp_delta: float = DEFAULT_WEIGHTS["player_hp_delta"]
    boss_posture_delta: float = DEFAULT_WEIGHTS["boss_posture_delta"]
    player_posture_delta: float = DEFAULT_WEIGHTS["player_posture_delta"]
    player_hit: float = DEFAULT_WEIGHTS["player_hit"]
    boss_dead: float = DEFAULT_WEIGHTS["boss_dead"]
    player_dead: float = DEFAULT_WEIGHTS["player_dead"]
    step: float = DEFAULT_WEIGHTS["step"]

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "RewardWeights":
        cfg = config if config is not None else load_config()
        overrides = (cfg or {}).get("reward", {}).get("weights", {})
        merged = dict(DEFAULT_WEIGHTS)
        merged.update({k: v for k, v in overrides.items() if k in DEFAULT_WEIGHTS})
        return cls(**merged)


class RewardCalculator:
    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights if weights is not None else RewardWeights.from_config()

    def compute(self, prev: GameState, curr: GameState) -> float:
        w = self.weights
        reward = w.step

        boss_hp_lost = max(0.0, prev.boss_hp - curr.boss_hp)
        player_hp_lost = max(0.0, prev.player_hp - curr.player_hp)
        boss_posture_gained = max(0.0, curr.boss_posture - prev.boss_posture)
        player_posture_gained = max(0.0, curr.player_posture - prev.player_posture)

        reward += w.boss_hp_delta * boss_hp_lost
        reward -= w.player_hp_delta * player_hp_lost
        reward += w.boss_posture_delta * boss_posture_gained
        reward -= w.player_posture_delta * player_posture_gained

        if curr.player_hit:
            reward += w.player_hit

        if curr.boss_dead and not prev.boss_dead:
            reward += w.boss_dead
        if curr.player_dead and not prev.player_dead:
            reward += w.player_dead

        return reward
