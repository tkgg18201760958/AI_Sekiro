"""对灰度图像帧进行固定大小、支持跳帧采样的滑动窗口处理。

把一串单帧的、大约 84x84 大小的灰度图（每个环境步骤一帧）转换成
Atari 风格 RL 常用的 (H, W, stack_size) observation：不是把每一个连续帧都堆进去
（那样在 60fps 下画面几乎不变，会把堆栈的时间窗口浪费在几乎重复的帧上），
而是只有每第 `frame_skip` 次调用才真正轮转进一个新帧——中间的调用会重复
最近一次入栈的帧。这个模块不依赖 StateReader/config，只处理 ndarray，
因此可以独立测试和复用
（参见 docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md）。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class FrameStack:
    def __init__(self, stack_size: int = 4, frame_skip: int = 4):
        if stack_size < 1:
            raise ValueError(f"stack_size must be >= 1, got {stack_size}")
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        self.stack_size = stack_size
        self.frame_skip = frame_skip
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)
        self._calls_since_rotation = 0

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames = deque((frame.copy() for _ in range(self.stack_size)), maxlen=self.stack_size)
        self._calls_since_rotation = 0
        return self._stacked()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._calls_since_rotation += 1
        if self._calls_since_rotation >= self.frame_skip:
            self._frames.append(frame.copy())
            self._calls_since_rotation = 0
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        return np.stack(self._frames, axis=-1)
