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

Observation (Box, uint8, shape (H, W, stack_size)): a sliding window of
grayscale game-window frames from StateReader.read_frame(), stacked by
FrameStack (see docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md).
This REPLACES the earlier 13-dim scalar-state vector entirely -- the agent
only ever sees pixels. GameState (from StateReader.read(), numeric HP/
posture/death flags) is still read every step, but purely to drive
RewardCalculator and RestartManager; it never enters the observation.
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
from ..state_reader.frame_stack import FrameStack
from ..state_reader.observation_config import ObservationConfig
from ..state_reader.schema import GameState


class SekiroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        reader: StateReader,
        controller: InputController,
        reward_calculator: Optional[RewardCalculator] = None,
        restart_manager: Optional[RestartManager] = None,
        max_episode_steps: Optional[int] = 2000,
        action_delay: float = 0.0,
        observation_config: Optional[ObservationConfig] = None,
    ):
        super().__init__()
        self.reader = reader
        self.controller = controller
        self.reward_calculator = reward_calculator if reward_calculator is not None else RewardCalculator()
        self.restart_manager = restart_manager if restart_manager is not None else RestartManager(reader, controller)
        self.max_episode_steps = max_episode_steps
        self.action_delay = action_delay
        self.observation_config = observation_config if observation_config is not None else ObservationConfig.from_config()

        self.action_space = spaces.Discrete(len(Action))
        width, height = self.observation_config.frame_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, self.observation_config.stack_size), dtype=np.uint8
        )
        self._frame_stack = FrameStack(
            stack_size=self.observation_config.stack_size, frame_skip=self.observation_config.frame_skip
        )

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
        obs = self._frame_stack.reset(self.reader.read_frame())
        return obs, {"state": state.to_dict()}

    def step(self, action: int):
        self.controller.execute(Action(action))
        if self.action_delay > 0:
            time.sleep(self.action_delay)

        obs = self._frame_stack.push(self.reader.read_frame())
        state = self.reader.read()
        reward = self.reward_calculator.compute(self._prev_state, state)
        terminated = RestartManager.needs_restart(state)

        self._elapsed_steps += 1
        truncated = self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps

        self._prev_state = state
        info = {"state": state.to_dict()}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.controller.close()
        self.reader.close()
