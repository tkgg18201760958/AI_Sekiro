"""Gymnasium Env wrapping StateReader + InputController + RewardCalculator +
RestartManager into the standard reset()/step() API (architecture.md
section 3's per-step data flow).

Design choice on *when* the restart sequence actually runs: `step()` only
detects the death/boss-kill condition and reports it via `terminated=True`
(with the terminal state as the returned observation, per Gym convention).
The actual restart key sequence is executed inside `reset()`, since that's
exactly the moment a new episode is expected to begin -- this keeps `step()`
fast and side-effect-free beyond the one action it was asked to take, and
matches how SB3's VecEnv auto-calls reset() right after a terminated step.

Observation vector (Box, 13 dims, all roughly in [0,1] per architecture.md's
"归一化到 0-1"):
    [player_hp, player_posture, boss_hp, boss_posture, distance_norm,
     player_hit, can_parry, *one_hot(boss_action over BOSS_ACTIONS)]
`distance_norm` is `distance / max_distance` clipped to [0,1] -- an
arbitrary but documented normalization since raw distance is unbounded.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..controller.action_map import Action
from ..controller.input_controller import InputController
from ..restart.restart_manager import RestartManager
from ..reward.reward_calculator import RewardCalculator
from ..state_reader.base import StateReader
from ..state_reader.schema import BOSS_ACTIONS, GameState

OBS_DIM = 7 + len(BOSS_ACTIONS)


class SekiroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        reader: StateReader,
        controller: InputController,
        reward_calculator: Optional[RewardCalculator] = None,
        restart_manager: Optional[RestartManager] = None,
        max_distance: float = 10.0,
        max_episode_steps: Optional[int] = 2000,
        action_delay: float = 0.0,
    ):
        super().__init__()
        self.reader = reader
        self.controller = controller
        self.reward_calculator = reward_calculator if reward_calculator is not None else RewardCalculator()
        self.restart_manager = restart_manager if restart_manager is not None else RestartManager(reader, controller)
        self.max_distance = max_distance
        self.max_episode_steps = max_episode_steps
        self.action_delay = action_delay

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self._prev_state: GameState = GameState()
        self._elapsed_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        state = self.reader.read()
        if RestartManager.needs_restart(state):
            state = self.restart_manager.run()
        elif hasattr(self.reader, "reset"):
            state = self.reader.reset()
        # else: state as read() returned it -- no restart needed, and this
        # reader has no reset() (e.g. a real PixelStateReader has nothing to
        # "reset", the game world is just whatever it currently is).

        self._prev_state = state
        self._elapsed_steps = 0
        return self.state_to_obs(state), {"state": state.to_dict()}

    def step(self, action: int):
        self.controller.execute(Action(action))
        if self.action_delay > 0:
            time.sleep(self.action_delay)

        state = self.reader.read()
        reward = self.reward_calculator.compute(self._prev_state, state)
        terminated = RestartManager.needs_restart(state)

        self._elapsed_steps += 1
        truncated = self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps

        self._prev_state = state
        info = {"state": state.to_dict()}
        return self.state_to_obs(state), reward, terminated, truncated, info

    def close(self):
        self.controller.close()
        self.reader.close()

    def state_to_obs(self, state: GameState) -> np.ndarray:
        max_distance = self.max_distance if self.max_distance > 0 else 1.0
        distance_norm = min(1.0, max(0.0, state.distance / max_distance))
        one_hot = [1.0 if state.boss_action == name else 0.0 for name in BOSS_ACTIONS]
        vec = [
            state.player_hp,
            state.player_posture,
            state.boss_hp,
            state.boss_posture,
            distance_norm,
            1.0 if state.player_hit else 0.0,
            1.0 if state.can_parry else 0.0,
            *one_hot,
        ]
        return np.array(vec, dtype=np.float32)
