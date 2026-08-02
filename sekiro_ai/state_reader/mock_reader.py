"""Simulated StateReader used while no real game connection is available.

Two modes:

- "scripted" (default): replays a deterministic, hand-authored fight
  trajectory (idle -> approach -> exchange -> boss staggers -> boss dies),
  then loops. Useful for reproducible tests of downstream modules.
- "random": jitters values within valid bounds every call. Useful for
  fuzzing / making sure nothing crashes on arbitrary state combinations.
"""
from __future__ import annotations

import random
import time
from typing import List

import numpy as np

from .base import StateReader
from .observation_config import ObservationConfig
from .schema import GameState
from ..utils.config_loader import load_config

# 要绘制的血条 rect，键名与 config.yaml 的 calibration 部分中使用的名称
# 一致，值是每个 rect 的填充比例应该跟踪的 GameState 属性名。顺序无关紧要，
# 各自独立绘制。
_BAR_SPECS = (
    ("player_hp_bar", "player_hp"),
    ("player_posture_bar", "player_posture"),
    ("boss_hp_bar", "boss_hp"),
    ("boss_posture_bar", "boss_posture"),
)

# Each scripted phase describes the *target* values the state should move
# towards, plus the boss_action for that phase, and how many read() calls to
# stay in the phase. Values are interpolated linearly from the previous
# phase's end state.
_SCRIPTED_PHASES: List[dict] = [
    {"steps": 5, "boss_action": "idle", "player_hp": 1.0, "player_posture": 0.0,
     "boss_hp": 1.0, "boss_posture": 0.0, "distance": 8.0},
    {"steps": 5, "boss_action": "walk", "player_hp": 1.0, "player_posture": 0.0,
     "boss_hp": 1.0, "boss_posture": 0.0, "distance": 3.0},
    {"steps": 6, "boss_action": "attack", "player_hp": 0.85, "player_posture": 0.2,
     "boss_hp": 0.9, "boss_posture": 0.1, "distance": 1.5},
    {"steps": 6, "boss_action": "perilous_attack", "player_hp": 0.7, "player_posture": 0.4,
     "boss_hp": 0.75, "boss_posture": 0.3, "distance": 1.2},
    {"steps": 5, "boss_action": "stagger", "player_hp": 0.65, "player_posture": 0.3,
     "boss_hp": 0.55, "boss_posture": 0.8, "distance": 1.0},
    {"steps": 6, "boss_action": "attack", "player_hp": 0.5, "player_posture": 0.5,
     "boss_hp": 0.35, "boss_posture": 0.4, "distance": 1.3},
    {"steps": 5, "boss_action": "stagger", "player_hp": 0.45, "player_posture": 0.4,
     "boss_hp": 0.1, "boss_posture": 0.9, "distance": 1.0},
    {"steps": 2, "boss_action": "dead", "player_hp": 0.45, "player_posture": 0.4,
     "boss_hp": 0.0, "boss_posture": 0.9, "distance": 1.0},
]


class MockStateReader(StateReader):
    def __init__(self, mode: str = "scripted", seed: int | None = None, noise: float = 0.02):
        if mode not in ("scripted", "random"):
            raise ValueError(f"Unknown mock mode: {mode!r}")
        self.mode = mode
        self.noise = noise
        self._seed = seed
        self._rng = random.Random(seed)

        self._phase_idx = 0
        self._phase_step = 0
        self._state = GameState()

    def is_available(self) -> bool:
        return True

    def reset(self) -> GameState:
        """Restart from phase 0, reseeding the RNG so the same `seed` always
        reproduces the same trajectory (needed for e.g. gymnasium's
        check_env, which resets with a fixed seed and expects step() to then
        behave identically across repeated resets)."""
        self._rng = random.Random(self._seed)
        self._phase_idx = 0
        self._phase_step = 0
        self._state = GameState(timestamp=time.time())
        return self._state

    def read(self) -> GameState:
        if self.mode == "scripted":
            self._advance_scripted()
        else:
            self._advance_random()
        self._state.timestamp = time.time()
        return self._state

    def read_frame(self) -> np.ndarray:
        """合成的、大约 84x84 大小的灰度帧：把 config.yaml 的 calibration
        部分中的 4 个血条 rect（从它们原生的 calibration.resolution 缩放
        到 ObservationConfig.frame_size）绘制成水平色条，其填充宽度对应
        相应的 GameState 字段。这样 mock 模式下的预览图像在视觉位置上
        就能和 PixelStateReader 最终从真实游戏窗口截取到的图像保持一致，
        符合设计文档里对 mock 图像的要求。"""
        obs_cfg = ObservationConfig.from_config()
        width, height = obs_cfg.frame_size
        frame = np.zeros((height, width), dtype=np.uint8)

        calibration = load_config().get("calibration", {})
        resolution = calibration.get("resolution") or [1280, 720]

        for rect_key, state_attr in _BAR_SPECS:
            rect = calibration.get(rect_key)
            if not rect:
                continue
            x, y, w, h = _scale_rect(rect, resolution, obs_cfg.frame_size)
            fill_ratio = max(0.0, min(1.0, getattr(self._state, state_attr)))
            filled_w = max(1, int(w * fill_ratio))
            frame[y : y + h, x : x + filled_w] = 200

        return frame

    def _advance_scripted(self) -> None:
        phase = _SCRIPTED_PHASES[self._phase_idx]

        for key in ("player_hp", "player_posture", "boss_hp", "boss_posture", "distance"):
            target = phase[key]
            current = getattr(self._state, key)
            jitter = self._rng.uniform(-self.noise, self.noise)
            new_value = current + (target - current) * 0.5 + jitter
            setattr(self._state, key, _clamp01(new_value) if key != "distance" else max(0.0, new_value))

        self._state.boss_action = phase["boss_action"]
        self._state.player_hit = phase["boss_action"] in ("attack", "perilous_attack") and self._rng.random() < 0.3
        self._state.can_parry = phase["boss_action"] == "attack"
        # Sticky once True: the "dead" phase's jitter can nudge boss_hp back
        # above the 0.01 threshold on a later read() (noise is applied every
        # call, even once the target is reached), which would otherwise
        # flicker boss_dead False again -- not a real "un-death", just noise.
        self._state.boss_dead = self._state.boss_dead or (
            phase["boss_action"] == "dead" and self._state.boss_hp <= 0.01
        )
        self._state.player_dead = self._state.player_dead or self._state.player_hp <= 0.0

        self._phase_step += 1
        if self._phase_step >= phase["steps"]:
            self._phase_step = 0
            if self._phase_idx < len(_SCRIPTED_PHASES) - 1:
                self._phase_idx += 1
            # Once the last phase is reached, stay there (boss_dead=True)
            # until the caller calls reset() to start a new episode.

    def _advance_random(self) -> None:
        s = self._state
        s.player_hp = _clamp01(s.player_hp + self._rng.uniform(-0.1, 0.05))
        s.player_posture = _clamp01(s.player_posture + self._rng.uniform(-0.1, 0.1))
        s.boss_hp = _clamp01(s.boss_hp + self._rng.uniform(-0.08, 0.02))
        s.boss_posture = _clamp01(s.boss_posture + self._rng.uniform(-0.1, 0.1))
        s.distance = max(0.0, s.distance + self._rng.uniform(-1.0, 1.0))
        s.boss_action = self._rng.choice(("idle", "walk", "attack", "perilous_attack", "stagger"))
        s.player_hit = self._rng.random() < 0.15
        s.can_parry = s.boss_action == "attack"
        s.player_dead = s.player_hp <= 0.0
        s.boss_dead = s.boss_hp <= 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scale_rect(
    rect: list[int], from_size: list[int], to_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """把一个在 `from_size` 分辨率下测得的 [x, y, w, h] 像素 rect
    映射到 `to_size` 的画布上，宽/高至少钳制为 1px，这样一个很小的
    目标画布（比如 84x84）永远不会产生面积为零的 rect。"""
    x, y, w, h = rect
    from_w, from_h = from_size
    to_w, to_h = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    return (
        int(x * scale_x),
        int(y * scale_y),
        max(1, int(w * scale_x)),
        max(1, int(h * scale_y)),
    )
