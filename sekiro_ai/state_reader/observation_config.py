"""像素帧观测管线的配置（帧尺寸、跳帧采样间隔、堆栈大小）—— 由
MockStateReader、PixelStateReader 和 SekiroEnv 三者共用，让它们对观测
形状达成一致，而不需要各自硬编码
（参见 docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md）。

沿用 sekiro_ai/reward/reward_calculator.py 的 RewardWeights 那种
"dataclass + from_config() 在默认值上合并覆盖"的模式。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.config_loader import load_config

DEFAULT_FRAME_SIZE: tuple[int, int] = (84, 84)
DEFAULT_FRAME_SKIP = 4
DEFAULT_STACK_SIZE = 4


@dataclass
class ObservationConfig:
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE
    frame_skip: int = DEFAULT_FRAME_SKIP
    stack_size: int = DEFAULT_STACK_SIZE

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "ObservationConfig":
        cfg = config if config is not None else load_config()
        section = (cfg or {}).get("observation", {})
        frame_size = section.get("frame_size", list(DEFAULT_FRAME_SIZE))
        return cls(
            frame_size=tuple(frame_size),
            frame_skip=section.get("frame_skip", DEFAULT_FRAME_SKIP),
            stack_size=section.get("stack_size", DEFAULT_STACK_SIZE),
        )
