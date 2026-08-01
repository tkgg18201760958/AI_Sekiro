"""Unified game state schema shared by every module in the pipeline.

Any StateReader implementation must return a `GameState` (or an equivalent
plain dict via `GameState.to_dict()`) so downstream modules (reward
calculator, gym env, restart manager) never need to know whether the data
came from a mock, memory reads, or pixel analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Tuple

# Known boss action labels. Readers are free to emit other strings (e.g. a
# new attack name discovered later), but these are the ones the rest of the
# pipeline is expected to reason about.
BOSS_ACTIONS = (
    "idle",
    "walk",
    "attack",
    "perilous_attack",
    "stagger",
    "dead",
)


@dataclass
class GameState:
    player_hp: float = 1.0
    player_posture: float = 0.0
    boss_hp: float = 1.0
    boss_posture: float = 0.0
    player_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    boss_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 0.0
    boss_action: str = "idle"
    player_hit: bool = False
    can_parry: bool = False
    player_dead: bool = False
    boss_dead: bool = False
    # Wall-clock timestamp (seconds, time.time()) the state was captured at.
    # Downstream modules use this to reason about observation staleness.
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


DEFAULT_STATE: dict[str, Any] = GameState().to_dict()
