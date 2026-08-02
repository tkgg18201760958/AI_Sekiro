# Pixel Frame Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SekiroEnv`'s 13-dim scalar observation with a stacked 84x84 grayscale image observation read directly from game pixels, while keeping numeric HP/posture extraction alive (for reward/restart only) and preserving a fully offline mock-mode test path.

**Architecture:** `StateReader` gains a second read method, `read_frame() -> np.ndarray` (single 84x84 uint8 grayscale frame), alongside the existing `read() -> GameState` (numeric state for reward/restart). A new `FrameStack` module samples frames at a configurable interval (`frame_skip`) and keeps a sliding window of `stack_size` frames, which becomes `SekiroEnv`'s `Box(shape=(84,84,stack_size), dtype=uint8)` observation. `MockStateReader` renders synthetic bar-graphics for `read_frame()` (positioned using the same `calibration` rects `PixelStateReader` uses for its numeric extraction), so the whole training pipeline still runs with no game window. `train.py`/`play.py` switch PPO's policy from `MlpPolicy` to `CnnPolicy`.

**Tech Stack:** Python 3.13, OpenCV (`opencv-python`, already in requirements.txt), `mss` (screen capture), NumPy, Gymnasium, stable-baselines3 PPO `CnnPolicy`.

## Global Constraints

- `observation_space` becomes `spaces.Box(low=0, high=255, shape=(84,84,stack_size), dtype=np.uint8)`, replacing the old `Box(shape=(13,), dtype=np.float32)`. Default `frame_size=[84,84]`, `frame_skip=4`, `stack_size=4`, all overridable via `config.yaml`'s new `observation:` section.
- `StateReader.read_frame()` is a new **abstract** method (not optional) — every `StateReader` subclass, including test doubles, must implement it.
- `read()` and `read_frame()` must NOT share a screenshot cache. `RestartManager.run()` polls `reader.read()` in a blocking loop waiting for state to change (`restart/restart_manager.py:88-124`); a shared/stale capture would make that loop see a frozen state and time out. Each capture is independent, even though real-mode this costs one extra `mss` grab per env step.
- Numeric HP/posture extraction in `PixelStateReader.read()` only needs to produce `player_hp`, `boss_hp`, `player_posture`, `boss_posture`, `player_dead`, `boss_dead`. `boss_action`, `can_parry`, `player_hit`, `player_pos`, `boss_pos`, `distance` stay at `GameState` defaults — perilous-icon template matching and position/distance estimation are explicitly out of scope (per the design spec).
- Training visualization UI is explicitly out of scope for this plan (separate future brainstorming round, per user request).
- Real-game features stay Windows-only and keep the project's lazy-import convention: `cv2`/`mss`/`win32gui` are imported inside `PixelStateReader` methods, never at module top level.
- `mock_reader.py` must NOT import `cv2` (or any other package from the "only needed for `--live`" list in `docs/installation.md`) even for its new image rendering — draw synthetic frames with plain NumPy slicing instead, so stage 1-7's "runs with no extra deps installed" guarantee still holds. `numpy`/`PyYAML` are already always-installed core deps (per `docs/installation.md`'s dependency table) and may be imported at module top level anywhere.
- This project has no pytest and no test runner config (confirmed: `pytest` is not installed in `venv`). Tests are standalone argparse-free or argparse scripts under `tests/`, run directly with `python tests/test_X.py`, using hand-rolled `assert`/PASS-FAIL-log patterns (see `tests/test_reward.py` for the house style). All new tests in this plan follow that same style, not pytest.
- Design spec (read before starting): `docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md`.

---

## File Structure

New files:
- `sekiro_ai/state_reader/frame_stack.py` — `FrameStack` class (pure NumPy, no StateReader/config dependency).
- `sekiro_ai/state_reader/observation_config.py` — `ObservationConfig` dataclass, loads `config.yaml`'s `observation:` section (mirrors `reward/reward_calculator.py`'s `RewardWeights.from_config()` pattern already in the codebase).
- `tests/test_frame_stack.py`, `tests/test_mock_frame.py`, `tests/test_pixel_bar_extraction.py` — new test scripts, one per new testable unit.

Modified files:
- `sekiro_ai/state_reader/base.py` — add abstract `read_frame()` to `StateReader`.
- `sekiro_ai/state_reader/mock_reader.py` — add `render_frame()`/`read_frame()`, calibration-scaled synthetic bar drawing.
- `sekiro_ai/state_reader/pixel_reader.py` — implement `read()` (bar-rect + HSV fill ratio) and `read_frame()` (capture + grayscale + resize); add `_capture_client_area()` and `bar_fill_ratio()` helpers.
- `sekiro_ai/env/sekiro_env.py` — `observation_space` becomes an image `Box`; `reset()`/`step()` call `read_frame()` through a `FrameStack`; remove `OBS_DIM`/`state_to_obs`.
- `sekiro_ai/env/__init__.py` — drop the `OBS_DIM` export (no longer exists).
- `tests/test_restart.py` — `ScriptedDeathReader` gains a `read_frame()` stub (required now that it's abstract).
- `tests/test_env.py` — assert the new image `observation_space` shape/dtype.
- `tests/test_state_reader.py` — add `--save-frames N` PNG-export option.
- `config/config.yaml` — new `observation:` section; new `calibration.posture_color_hsv_range` field.
- `train.py`, `play.py` — `PPO("CnnPolicy", ...)` instead of `PPO("MlpPolicy", ...)`.
- `docs/architecture.md`, `docs/installation.md`, `docs/training.md`, `docs/configuration.md` — update the "13-dim vector / MlpPolicy, no CNN needed" statements to describe the new image observation.

---

### Task 1: FrameStack module

**Files:**
- Create: `sekiro_ai/state_reader/frame_stack.py`
- Test: `tests/test_frame_stack.py`

**Interfaces:**
- Consumes: nothing (pure NumPy, no project imports).
- Produces: `class FrameStack` with:
  - `__init__(self, stack_size: int = 4, frame_skip: int = 4)`
  - `reset(self, frame: np.ndarray) -> np.ndarray` — clears state, fills the whole stack with `frame` (copied `stack_size` times), returns the `(H, W, stack_size)` stacked array.
  - `push(self, frame: np.ndarray) -> np.ndarray` — call once per env step with the newest captured frame; internally counts calls and only rotates a new frame into the stack every `frame_skip` calls (repeating the most recent stacked frame on skipped calls), returns the current `(H, W, stack_size)` stacked array.
  - Frames passed in are always full-size `(H, W)` 2D uint8 arrays (grayscale, already resized by the caller); `FrameStack` does no resizing/color conversion itself, it only stacks.

Downstream (`SekiroEnv`, Task 5) relies on exactly these two method names and this exact behavior: `reset()` is called once per episode start, `push()` once per `step()`, and both return `(H, W, stack_size)` uint8 ndarrays with the oldest frame at index 0 and newest at index `stack_size - 1` along the last axis.

- [ ] **Step 1: Write the failing tests**

```python
"""Standalone test for sekiro_ai.state_reader.frame_stack.FrameStack.

Usage:
    python tests/test_frame_stack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.frame_stack import FrameStack
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("frame_stack", "frame_stack.log")


def make_frame(value: int) -> np.ndarray:
    """An 8x8 frame filled with a single value, so stacked slices are
    trivially distinguishable by their scalar fill value."""
    return np.full((8, 8), value, dtype=np.uint8)


def test_reset_fills_stack_with_first_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=4)
    stacked = fs.reset(make_frame(10))
    ok = stacked.shape == (8, 8, 4) and stacked.dtype == np.uint8 and np.all(stacked == 10)
    logger.info("[%s] reset fills all %d slots with first frame", "PASS" if ok else "FAIL", 4)
    return ok


def test_push_before_skip_interval_repeats_last_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=4)
    fs.reset(make_frame(1))
    # frame_skip=4: only every 4th push() call should rotate in a new frame.
    # Calls 1-3 after reset should NOT change the stack.
    before = fs.push(make_frame(99)).copy()
    stacked = fs.push(make_frame(50))
    ok = np.array_equal(before, stacked)
    logger.info("[%s] pushes before frame_skip interval don't change the stack", "PASS" if ok else "FAIL")
    return ok


def test_push_at_skip_interval_rotates_newest_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=2)
    fs.reset(make_frame(0))
    fs.push(make_frame(11))  # call 1: within skip interval, no rotation
    stacked = fs.push(make_frame(22))  # call 2: skip interval reached, rotate in 22
    ok = stacked[:, :, -1][0, 0] == 22
    logger.info("[%s] newest frame lands at last stack index after frame_skip calls", "PASS" if ok else "FAIL")
    return ok


def test_stack_order_oldest_to_newest() -> bool:
    fs = FrameStack(stack_size=3, frame_skip=1)
    fs.reset(make_frame(0))
    fs.push(make_frame(1))
    fs.push(make_frame(2))
    stacked = fs.push(make_frame(3))
    ok = (
        stacked[0, 0, 0] == 1
        and stacked[0, 0, 1] == 2
        and stacked[0, 0, 2] == 3
    )
    logger.info("[%s] stack orders oldest frame at index 0, newest at last index", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    tests = [
        test_reset_fills_stack_with_first_frame,
        test_push_before_skip_interval_repeats_last_frame,
        test_push_at_skip_interval_rotates_newest_frame,
        test_stack_order_oldest_to_newest,
    ]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d FrameStack tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d FrameStack tests PASSED.", len(tests))
    logger.info("Test finished. Log written to logs/frame_stack.log")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_frame_stack.py`
Expected: `ModuleNotFoundError: No module named 'sekiro_ai.state_reader.frame_stack'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Fixed-size, frame-skipping sliding window over grayscale image frames.

Turns a stream of single 84x84-ish grayscale frames (one per env step) into
a (H, W, stack_size) observation the way Atari-style RL setups do: instead
of stacking every consecutive frame (which barely changes at 60fps and
wastes the stack's temporal window on near-duplicates), only every
`frame_skip`-th call actually rotates a new frame in -- calls in between
repeat the most recently stacked frame. This has no dependency on
StateReader/config -- it only knows about ndarrays, so it's testable and
reusable standalone (see docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md).
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_frame_stack.py`
Expected: `All 4 FrameStack tests PASSED.` (no `[FAIL]` lines, exit without traceback)

- [ ] **Step 5: Commit**

```bash
git add sekiro_ai/state_reader/frame_stack.py tests/test_frame_stack.py
git commit -m "feat: add FrameStack for frame-skip image observation stacking"
```

---

### Task 2: Observation config + calibration posture color range

**Files:**
- Create: `sekiro_ai/state_reader/observation_config.py`
- Modify: `config/config.yaml`
- Modify: `docs/configuration.md`

**Interfaces:**
- Consumes: `sekiro_ai.utils.config_loader.load_config()` (existing, `sekiro_ai/utils/config_loader.py:20`).
- Produces: `class ObservationConfig` with fields `frame_size: tuple[int, int] = (84, 84)`, `frame_skip: int = 4`, `stack_size: int = 4`, and classmethod `from_config(config: dict | None = None) -> "ObservationConfig"` (mirrors `RewardWeights.from_config` in `sekiro_ai/reward/reward_calculator.py:46-52` — same merge-over-defaults pattern). Task 3 (`MockStateReader`) and Task 4 (`PixelStateReader`) both construct one of these; Task 5 (`SekiroEnv`) constructs one to size its `FrameStack`.
- Also produces: `config.yaml`'s new `calibration.posture_color_hsv_range` field, read directly via `load_config().get("calibration", {}).get("posture_color_hsv_range")` (no new config class needed for this one field — it lives alongside the existing `hp_color_hsv_range` sibling in the same `calibration:` section, following that section's existing plain-dict-access style in `pixel_reader.py`).

- [ ] **Step 1: Write the failing test**

Add to a new `tests/test_observation_config.py`:

```python
"""Standalone test for sekiro_ai.state_reader.observation_config.ObservationConfig.

Usage:
    python tests/test_observation_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.observation_config import ObservationConfig
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("observation_config", "observation_config.log")


def test_defaults_with_no_config() -> bool:
    cfg = ObservationConfig.from_config({})
    ok = cfg.frame_size == (84, 84) and cfg.frame_skip == 4 and cfg.stack_size == 4
    logger.info("[%s] defaults with empty config: %s", "PASS" if ok else "FAIL", cfg)
    return ok


def test_overrides_from_config() -> bool:
    cfg = ObservationConfig.from_config(
        {"observation": {"frame_size": [42, 42], "frame_skip": 2, "stack_size": 3}}
    )
    ok = cfg.frame_size == (42, 42) and cfg.frame_skip == 2 and cfg.stack_size == 3
    logger.info("[%s] overrides from config.yaml's observation section: %s", "PASS" if ok else "FAIL", cfg)
    return ok


def test_partial_override_keeps_other_defaults() -> bool:
    cfg = ObservationConfig.from_config({"observation": {"frame_skip": 8}})
    ok = cfg.frame_size == (84, 84) and cfg.frame_skip == 8 and cfg.stack_size == 4
    logger.info("[%s] partial override keeps unspecified fields at default: %s", "PASS" if ok else "FAIL", cfg)
    return ok


def main() -> None:
    tests = [test_defaults_with_no_config, test_overrides_from_config, test_partial_override_keeps_other_defaults]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d ObservationConfig tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d ObservationConfig tests PASSED.", len(tests))
    logger.info("Test finished. Log written to logs/observation_config.log")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_observation_config.py`
Expected: `ModuleNotFoundError: No module named 'sekiro_ai.state_reader.observation_config'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Config for the pixel-frame observation pipeline (frame size, frame-skip
sampling interval, stack size) -- shared by MockStateReader, PixelStateReader,
and SekiroEnv so all three agree on the observation shape without any of them
hardcoding it (see docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md).

Follows the same "dataclass + from_config() merges over defaults" pattern as
sekiro_ai/reward/reward_calculator.py's RewardWeights.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_observation_config.py`
Expected: `All 3 ObservationConfig tests PASSED.`

- [ ] **Step 5: Add `observation:` section and `posture_color_hsv_range` to config.yaml**

Add this new top-level section to `config/config.yaml` (after the existing `calibration:` section):

```yaml
observation:
  # Pixel-frame observation shape fed to the PPO agent (sekiro_ai.state_reader.observation_config).
  # frame_size: single-frame resize target [width, height] before stacking.
  # frame_skip: sample a new frame into the stack every N env steps (in between,
  #   the most recently stacked frame repeats) -- at 60fps this keeps the stack's
  #   time window wide enough to show motion instead of 4 nearly-identical frames.
  # stack_size: number of frames kept in the sliding window; observation shape
  #   is (frame_size[1], frame_size[0], stack_size).
  frame_size: [84, 84]
  frame_skip: 4
  stack_size: 4
```

Also add `posture_color_hsv_range` to the existing `calibration:` section, right after `hp_color_hsv_range`:

```yaml
  posture_color_hsv_range: [[15, 80, 80], [35, 255, 255]]   # yellow chevron posture bar fill, distinct from the red/salmon HP bars above
```

(Measured against `GAME_PIC/BOSS.png`/`GAME_PIC/WITHOUT_BOSS.png`: this range gives a HP-bar-vs-posture-bar HSV separation that correctly reads ~0 fill on an empty posture bar and high fill on a full one in both sample screenshots — verified by manual `cv2.inRange` + column-fill-ratio checks during planning, not by an automated test since there's no live game window yet.)

- [ ] **Step 6: Document the new config fields**

In `docs/configuration.md`, add a subsection after the existing `calibration` section documenting `observation:` (fields, defaults, what changing `frame_skip`/`stack_size` does to training) and add `posture_color_hsv_range` to the `calibration` fields table, following the existing doc's style (see the `hp_color_hsv_range` row already there).

- [ ] **Step 7: Commit**

```bash
git add sekiro_ai/state_reader/observation_config.py tests/test_observation_config.py config/config.yaml docs/configuration.md
git commit -m "feat: add ObservationConfig and observation/posture_color config fields"
```

---

### Task 3: `StateReader.read_frame()` abstract method + `MockStateReader` implementation

**Files:**
- Modify: `sekiro_ai/state_reader/base.py`
- Modify: `sekiro_ai/state_reader/mock_reader.py`
- Modify: `tests/test_restart.py` (its `ScriptedDeathReader` test double must implement the now-abstract method)
- Test: `tests/test_mock_frame.py`

**Interfaces:**
- Consumes: `ObservationConfig.from_config()` (Task 2, `sekiro_ai/state_reader/observation_config.py`), `load_config()` (existing, `sekiro_ai/utils/config_loader.py`).
- Produces: `StateReader.read_frame(self) -> np.ndarray` as an `@abstractmethod` on the base class (`sekiro_ai/state_reader/base.py`) — every subclass must implement it or fail to instantiate. `MockStateReader.read_frame()` returns a `(H, W)` uint8 grayscale ndarray sized per `ObservationConfig.frame_size` (note: NumPy shape order is `(height, width)`, i.e. `(frame_size[1], frame_size[0])`), synthesizing bar graphics from the current `GameState`'s `player_hp`/`boss_hp`/`player_posture`/`boss_posture`. This is what Task 5 (`SekiroEnv`) and Task 6 (test updates) call.

- [ ] **Step 1: Update `StateReader.read_frame()` as abstract, and `ScriptedDeathReader` stub, before writing MockStateReader's real implementation**

Edit `sekiro_ai/state_reader/base.py`, adding the import and abstract method:

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .schema import GameState


class StateReader(ABC):
    @abstractmethod
    def read(self) -> GameState:
        """Return the current game state as a GameState instance."""
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Return a single (H, W) uint8 grayscale frame for the observation
        pipeline's FrameStack (see sekiro_ai.state_reader.frame_stack).

        Unlike read(), this must never share a capture/cache with read() --
        RestartManager polls read() in a blocking loop waiting for state
        changes (restart/restart_manager.py), and a shared stale capture
        would make that loop see a frozen state forever."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this reader can currently produce real data.

        Real readers (e.g. screen-based) should return False when the game
        window isn't found, so callers can decide to fall back to a mock.
        Mock readers are always available.
        """
        return True

    def close(self) -> None:
        """Release any resources (process handles, sockets, etc.)."""
        return None
```

Edit `tests/test_restart.py`'s `ScriptedDeathReader` (currently `tests/test_restart.py:31-58`) to add a stub `read_frame()` — it's now required for the class to instantiate at all, even though this test doesn't exercise frame reading:

```python
    def read_frame(self):
        import numpy as np
        return np.zeros((84, 84), dtype=np.uint8)
```

Add this method inside `class ScriptedDeathReader(StateReader):`, alongside its existing `read()`/`reset()`.

- [ ] **Step 2: Run `test_restart.py` to confirm it still passes (proves the abstract method doesn't break existing test doubles once stubbed)**

Run: `python tests/test_restart.py`
Expected: `PASS: restart sequence completed and returned a fresh state.` (same as before this task's changes)

- [ ] **Step 3: Write the failing test for MockStateReader.read_frame()**

```python
"""Standalone test for MockStateReader's read_frame() synthetic image
generation.

Usage:
    python tests/test_mock_frame.py
    python tests/test_mock_frame.py --save-frames 5   # also dump PNGs for visual inspection
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.mock_reader import MockStateReader
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("mock_frame", "mock_frame.log")

FRAME_PREVIEW_DIR = Path(__file__).resolve().parent.parent / "logs" / "frame_preview"


def test_read_frame_shape_and_dtype() -> bool:
    reader = MockStateReader(mode="scripted", seed=0)
    frame = reader.read_frame()
    ok = frame.shape == (84, 84) and frame.dtype == np.uint8
    logger.info("[%s] read_frame() shape=%s dtype=%s (expected (84,84) uint8)", "PASS" if ok else "FAIL", frame.shape, frame.dtype)
    return ok


def test_full_hp_bar_brighter_than_empty_hp_bar() -> bool:
    """A full HP bar should have more bright (filled) pixels than a nearly
    empty one, at the same bar location -- proxy for "the drawn bar tracks
    the GameState value" without needing exact pixel-perfect assertions."""
    reader = MockStateReader(mode="scripted", seed=0)
    reader._state.player_hp = 1.0
    full_frame = reader.read_frame()
    reader._state.player_hp = 0.05
    empty_frame = reader.read_frame()
    ok = full_frame.sum() > empty_frame.sum()
    logger.info(
        "[%s] full-HP frame brightness sum (%d) > near-empty-HP frame brightness sum (%d)",
        "PASS" if ok else "FAIL", int(full_frame.sum()), int(empty_frame.sum()),
    )
    return ok


def save_frames(n: int) -> None:
    if FRAME_PREVIEW_DIR.exists():
        shutil.rmtree(FRAME_PREVIEW_DIR)
    FRAME_PREVIEW_DIR.mkdir(parents=True)

    import cv2

    reader = MockStateReader(mode="scripted", seed=0)
    for i in range(n):
        reader.read()  # advance the scripted trajectory
        frame = reader.read_frame()
        path = FRAME_PREVIEW_DIR / f"mock_frame_{i:03d}.png"
        cv2.imwrite(str(path), frame)
    logger.info("Saved %d mock frames to %s", n, FRAME_PREVIEW_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MockStateReader.read_frame().")
    parser.add_argument("--save-frames", type=int, default=0, help="Also save N preview PNGs to logs/frame_preview/.")
    args = parser.parse_args()

    tests = [test_read_frame_shape_and_dtype, test_full_hp_bar_brighter_than_empty_hp_bar]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d MockStateReader.read_frame() tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d MockStateReader.read_frame() tests PASSED.", len(tests))

    if args.save_frames > 0:
        save_frames(args.save_frames)

    logger.info("Test finished. Log written to logs/mock_frame.log")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python tests/test_mock_frame.py`
Expected: `AttributeError: 'MockStateReader' object has no attribute 'read_frame'`

- [ ] **Step 5: Implement `MockStateReader.read_frame()`**

Add to `sekiro_ai/state_reader/mock_reader.py` (imports at top of file, method on the class, module-level helper functions at the bottom):

```python
# Add to the top-level imports:
import numpy as np

from .observation_config import ObservationConfig
from ..utils.config_loader import load_config

# Bar rects to draw, keyed by the same names used in config.yaml's
# calibration section, and the GameState attribute each one's fill ratio
# should track. Order doesn't matter; drawn independently.
_BAR_SPECS = (
    ("player_hp_bar", "player_hp"),
    ("player_posture_bar", "player_posture"),
    ("boss_hp_bar", "boss_hp"),
    ("boss_posture_bar", "boss_posture"),
)
```

Add this method to `class MockStateReader(StateReader)`, e.g. right after `read()`:

```python
    def read_frame(self) -> np.ndarray:
        """Synthetic 84x84-ish grayscale frame: draws each of the 4 bar
        rects from config.yaml's calibration section (scaled from their
        native calibration.resolution down to ObservationConfig.frame_size)
        as a horizontal bar whose filled width matches the corresponding
        GameState field. Lets mock-mode preview images look positionally
        consistent with what PixelStateReader would eventually capture from
        a real game window, per the design spec's mock-image requirement."""
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
```

Add these module-level helpers near the bottom of `mock_reader.py`, alongside the existing `_clamp01`:

```python
def _scale_rect(
    rect: list[int], from_size: list[int], to_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Map a [x, y, w, h] pixel rect measured at `from_size` resolution onto
    a `to_size` canvas, clamping width/height to at least 1px so a tiny
    target canvas (e.g. 84x84) never produces a zero-area rect."""
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python tests/test_mock_frame.py`
Expected: `All 2 MockStateReader.read_frame() tests PASSED.`

- [ ] **Step 7: Generate and visually inspect preview frames**

Run: `python tests/test_mock_frame.py --save-frames 8`
Expected: log line `Saved 8 mock frames to <path>\logs\frame_preview`, and 8 PNG files exist there. Open a couple in an image viewer -- confirm 4 short bright horizontal bar segments appear roughly where the HP/posture bars sit in `GAME_PIC/BOSS.png` (bottom-left, bottom-center, top-left, top-center), against a black background, and that the segments visibly change length across the 8 frames as the scripted trajectory advances.

- [ ] **Step 8: Commit**

```bash
git add sekiro_ai/state_reader/base.py sekiro_ai/state_reader/mock_reader.py tests/test_restart.py tests/test_mock_frame.py
git commit -m "feat: add read_frame() to StateReader; mock synthetic bar rendering"
```

---

### Task 4: `PixelStateReader.read_frame()` and `read()` implementation

**Files:**
- Modify: `sekiro_ai/state_reader/pixel_reader.py`
- Modify: `config/config.yaml` (add `posture_color_hsv_range`, already done in Task 2 -- this task only *consumes* it)
- Test: `tests/test_pixel_bar_extraction.py`

**Interfaces:**
- Consumes: `ObservationConfig.from_config()` (Task 2). `calibration.player_hp_bar`/`boss_hp_bar`/`player_posture_bar`/`boss_posture_bar`/`resolution`/`hp_color_hsv_range`/`posture_color_hsv_range` from `config.yaml` (all pre-existing except `posture_color_hsv_range`, added in Task 2).
- Produces: `PixelStateReader.read_frame(self) -> np.ndarray` (whole-window capture, grayscale, resized to `ObservationConfig.frame_size`) and a working `PixelStateReader.read(self) -> GameState` that fills `player_hp`, `boss_hp`, `player_posture`, `boss_posture`, `player_dead`, `boss_dead` from bar-rect HSV fill ratios (everything else in `GameState` stays at its dataclass default, per the design spec's explicit scope exclusions for `boss_action`/`player_pos`/`boss_pos`/`distance`/`can_parry`/`player_hit`). Also produces two importable helpers other code/tests can call directly: `bar_fill_ratio(bgr_image: np.ndarray, rect: tuple[int,int,int,int], hsv_range: tuple) -> float` and `capture_client_area(window) -> np.ndarray` (BGR ndarray).

This task cannot be exercised against a real game window (none is running during implementation) -- it's tested against the two static screenshots already in the repo, `GAME_PIC/BOSS.png` (boss engaged, mid-fight bar levels) and `GAME_PIC/WITHOUT_BOSS.png` (player at rest, high HP, empty posture bars), by feeding them in as a substitute for a live `mss` capture. This validates the HSV-threshold and fill-ratio math is correct; it does NOT validate real capture timing/`mss`/window-finding against a live game, which per the design spec's scope boundary remains a manual step for later (documented in `docs/live_game.md`, unchanged by this task).

- [ ] **Step 1: Write the failing test**

```python
"""Standalone test for PixelStateReader's bar-fill-ratio extraction math,
run against the two static screenshots already checked into GAME_PIC/ (no
live game window available during development -- see
docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md's scope
notes on what this can and can't validate).

Usage:
    python tests/test_pixel_bar_extraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.pixel_reader import bar_fill_ratio
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("pixel_bar_extraction", "pixel_bar_extraction.log")

GAME_PIC_DIR = Path(__file__).resolve().parent.parent / "GAME_PIC"

HP_HSV_RANGE = ([0, 100, 90], [10, 255, 255])
POSTURE_HSV_RANGE = ([15, 80, 80], [35, 255, 255])

# Rects copied from config/config.yaml's calibration section (measured
# against GAME_PIC/BOSS.png, a 1280x720 screenshot).
PLAYER_HP_BAR = (70, 647, 184, 15)
PLAYER_POSTURE_BAR = (507, 617, 266, 14)
BOSS_HP_BAR = (71, 60, 279, 14)
BOSS_POSTURE_BAR = (580, 40, 119, 10)


def load_image(name: str):
    img = cv2.imread(str(GAME_PIC_DIR / name))
    if img is None:
        raise FileNotFoundError(f"Could not load {GAME_PIC_DIR / name}")
    return img


def test_boss_hp_bar_mostly_full_in_boss_screenshot() -> bool:
    img = load_image("BOSS.png")
    ratio = bar_fill_ratio(img, BOSS_HP_BAR, HP_HSV_RANGE)
    ok = ratio > 0.8
    logger.info("[%s] BOSS.png boss_hp_bar fill=%.3f (expected > 0.8, boss near full HP)", "PASS" if ok else "FAIL", ratio)
    return ok


def test_boss_hp_bar_near_empty_without_boss_engaged() -> bool:
    """WITHOUT_BOSS.png has no boss HP bar rendered at all (no boss engaged)
    -- the rect should read near-zero fill since there's no red bar-fill
    color present there."""
    img = load_image("WITHOUT_BOSS.png")
    ratio = bar_fill_ratio(img, BOSS_HP_BAR, HP_HSV_RANGE)
    ok = ratio < 0.1
    logger.info("[%s] WITHOUT_BOSS.png boss_hp_bar fill=%.3f (expected < 0.1, no boss HUD present)", "PASS" if ok else "FAIL", ratio)
    return ok


def test_player_hp_bar_full_when_at_rest() -> bool:
    img = load_image("WITHOUT_BOSS.png")
    ratio = bar_fill_ratio(img, PLAYER_HP_BAR, HP_HSV_RANGE)
    ok = ratio > 0.8
    logger.info("[%s] WITHOUT_BOSS.png player_hp_bar fill=%.3f (expected > 0.8, player at rest/full HP)", "PASS" if ok else "FAIL", ratio)
    return ok


def test_posture_bars_empty_when_not_in_combat() -> bool:
    img = load_image("WITHOUT_BOSS.png")
    player_ratio = bar_fill_ratio(img, PLAYER_POSTURE_BAR, POSTURE_HSV_RANGE)
    ok = player_ratio < 0.1
    logger.info("[%s] WITHOUT_BOSS.png player_posture_bar fill=%.3f (expected < 0.1, not in combat)", "PASS" if ok else "FAIL", player_ratio)
    return ok


def test_boss_posture_bar_high_during_boss_fight() -> bool:
    img = load_image("BOSS.png")
    ratio = bar_fill_ratio(img, BOSS_POSTURE_BAR, POSTURE_HSV_RANGE)
    ok = ratio > 0.8
    logger.info("[%s] BOSS.png boss_posture_bar fill=%.3f (expected > 0.8, mid-fight)", "PASS" if ok else "FAIL", ratio)
    return ok


def main() -> None:
    tests = [
        test_boss_hp_bar_mostly_full_in_boss_screenshot,
        test_boss_hp_bar_near_empty_without_boss_engaged,
        test_player_hp_bar_full_when_at_rest,
        test_posture_bars_empty_when_not_in_combat,
        test_boss_posture_bar_high_during_boss_fight,
    ]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d bar extraction tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d bar extraction tests PASSED.", len(tests))
    logger.info("Test finished. Log written to logs/pixel_bar_extraction.log")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pixel_bar_extraction.py`
Expected: `ImportError: cannot import name 'bar_fill_ratio' from 'sekiro_ai.state_reader.pixel_reader'`

- [ ] **Step 3: Implement `bar_fill_ratio`, `capture_client_area`, `read()`, and `read_frame()` in `pixel_reader.py`**

Replace the whole `read()` method and add new helpers. The module keeps its existing lazy-import convention (`cv2`/`mss` imported inside functions, never at module top).

```python
def bar_fill_ratio(bgr_image, rect: tuple[int, int, int, int], hsv_range) -> float:
    """Fraction of `rect`'s width that has at least one pixel matching
    `hsv_range` (an OpenCV-HSV [[H,S,V]min, [H,S,V]max] pair) somewhere in
    its height -- a horizontal-bar-fill estimate that's robust to the bar's
    fill color varying slightly in brightness/shading (only requires ONE
    matching pixel per column, not the whole column)."""
    import cv2
    import numpy as np

    x, y, w, h = rect
    roi = bgr_image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(hsv_range[0]), np.array(hsv_range[1])
    mask = cv2.inRange(hsv, lo, hi)
    if mask.shape[1] == 0:
        return 0.0
    column_has_fill = (mask > 0).any(axis=0)
    return float(column_has_fill.mean())


def capture_client_area(window):
    """Screenshot of `window`'s client area (excludes title bar/border) as a
    BGR ndarray, using the window's current on-screen position (via
    win32gui.ClientToScreen) and size (via win32gui.GetClientRect)."""
    import cv2
    import mss
    import numpy as np
    import win32gui

    hwnd = window._hWnd
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_x, screen_y = win32gui.ClientToScreen(hwnd, (left, top))
    width, height = right - left, bottom - top

    with mss.mss() as sct:
        shot = sct.grab({"left": screen_x, "top": screen_y, "width": width, "height": height})
    bgra = np.array(shot)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
```

Replace the existing `read()` method body (currently always raising `NotImplementedError`, `pixel_reader.py:83-102`) with:

```python
    def read(self) -> GameState:
        if not self.is_available():
            raise RuntimeError(
                f"PixelStateReader could not find a window owned by {self.process_name!r}."
            )
        missing = self.missing_calibration()
        if missing:
            raise NotImplementedError(
                "PixelStateReader is not calibrated yet -- config.yaml's "
                f"`calibration` section is missing: {missing}. This must be "
                "measured against the live game window (see the comments "
                "above `calibration:` in config.yaml for the manual steps); "
                "it cannot be filled in from code. Use MockStateReader "
                "until calibration is done."
            )

        image = capture_client_area(self._window)
        hp_range = self.calibration.get("hp_color_hsv_range") or [[0, 100, 90], [10, 255, 255]]
        posture_range = self.calibration.get("posture_color_hsv_range") or [[15, 80, 80], [35, 255, 255]]

        player_hp = bar_fill_ratio(image, tuple(self.calibration["player_hp_bar"]), hp_range)
        boss_hp = bar_fill_ratio(image, tuple(self.calibration["boss_hp_bar"]), hp_range)
        player_posture = bar_fill_ratio(image, tuple(self.calibration["player_posture_bar"]), posture_range)
        boss_posture = bar_fill_ratio(image, tuple(self.calibration["boss_posture_bar"]), posture_range)

        return GameState(
            player_hp=player_hp,
            boss_hp=boss_hp,
            player_posture=player_posture,
            boss_posture=boss_posture,
            player_dead=player_hp <= 0.01,
            boss_dead=boss_hp <= 0.01,
            timestamp=time.time(),
        )

    def read_frame(self) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                f"PixelStateReader could not find a window owned by {self.process_name!r}."
            )
        import cv2

        obs_cfg = ObservationConfig.from_config()
        image = capture_client_area(self._window)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, obs_cfg.frame_size, interpolation=cv2.INTER_AREA)
```

Add the two new imports at the top of `pixel_reader.py` (alongside the existing `from ..utils.config_loader import load_config`):

```python
import time

import numpy as np

from .observation_config import ObservationConfig
```

Note: `read_frame()` deliberately does NOT check `missing_calibration()` -- it only needs the window to exist, not the bar-rect calibration, matching the design spec's error-handling section ("即使 HP 校准没填,图像分支也能跑").

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pixel_bar_extraction.py`
Expected: `All 5 bar extraction tests PASSED.`

If any HSV-range assertion fails, adjust `HP_HSV_RANGE`/`POSTURE_HSV_RANGE` in the test (and the matching `hp_color_hsv_range`/`posture_color_hsv_range` defaults in `pixel_reader.py` and `config/config.yaml`) based on what the actual pixel values in `GAME_PIC/BOSS.png`/`GAME_PIC/WITHOUT_BOSS.png` are -- inspect with `cv2.cvtColor(cv2.imread(...), cv2.COLOR_BGR2HSV)` on the specific rect region to find the real range, rather than guessing blindly.

- [ ] **Step 5: Commit**

```bash
git add sekiro_ai/state_reader/pixel_reader.py tests/test_pixel_bar_extraction.py
git commit -m "feat: implement PixelStateReader.read() bar extraction and read_frame() capture"
```

---

### Task 5: `SekiroEnv` image observation space

**Files:**
- Modify: `sekiro_ai/env/sekiro_env.py`
- Modify: `sekiro_ai/env/__init__.py`
- Modify: `tests/test_env.py`

**Interfaces:**
- Consumes: `StateReader.read()`/`read_frame()` (Tasks 3-4), `FrameStack` (Task 1, `sekiro_ai/state_reader/frame_stack.py`'s `FrameStack(stack_size, frame_skip)` with `.reset(frame)`/`.push(frame)` both returning `(H, W, stack_size)` uint8 ndarrays), `ObservationConfig.from_config()` (Task 2).
- Produces: `SekiroEnv.observation_space = spaces.Box(low=0, high=255, shape=(frame_size[1], frame_size[0], stack_size), dtype=np.uint8)`. Removes the `OBS_DIM` constant and `state_to_obs()` method entirely (no longer meaningful once observation is image-based) -- Task 6 depends on `OBS_DIM` no longer being imported anywhere.

- [ ] **Step 1: Update `tests/test_env.py`'s assertions for the new observation shape (failing test first)**

In `tests/test_env.py`, add this check inside `run_manual_episodes` right after `obs, info = env.reset(seed=seed)` (currently `tests/test_env.py:68`):

```python
            expected_shape = env.observation_space.shape
            if obs.shape != expected_shape or obs.dtype != np.uint8:
                logger.error(
                    "episode=%d FAIL: obs.shape=%s dtype=%s, expected shape=%s dtype=uint8",
                    ep, obs.shape, obs.dtype, expected_shape,
                )
```

Add `import numpy as np` to `tests/test_env.py`'s imports (currently missing since the old scalar-vector code never needed it directly in the test file).

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_env.py --skip-check-env`
Expected: fails or errors, since `SekiroEnv.observation_space` is still the old `Box(shape=(13,))` and `obs.shape` won't match an image shape expectation once Task 5's implementation changes are in (this step establishes the "before" baseline showing the old shape; if run before any code changes it will just pass trivially against the old 13-dim space, which is fine -- the meaningful failing-then-passing transition happens across Steps 2 and 4 as the implementation changes underneath it).

- [ ] **Step 3: Rewrite `sekiro_ai/env/sekiro_env.py`**

Replace the file's imports, module docstring, `OBS_DIM`, `__init__`, `reset()`, `step()`, and `state_to_obs()`:

```python
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
```

Note this removes the `max_distance` constructor parameter entirely along with `state_to_obs()` -- `distance_norm` was only ever used to build the old scalar observation vector, which no longer exists. `GameState.distance` is unused by the new pipeline (per the design spec's scope exclusions).

- [ ] **Step 4: Update `sekiro_ai/env/__init__.py`**

```python
from .factory import build_env, build_reader
from .sekiro_env import SekiroEnv

__all__ = ["SekiroEnv", "build_env", "build_reader"]
```

(Drops `OBS_DIM` from both the import and `__all__` since `sekiro_env.py` no longer defines it.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python tests/test_env.py`
Expected: `gymnasium check_env PASSED.` and `All ... FrameStack tests PASSED`-style manual episode logs with no `FAIL` lines; `obs.shape` logged for each episode reset should equal `(84, 84, 4)` (or whatever `config.yaml`'s `observation` section currently specifies).

If `check_env` raises a new (not the pre-existing, already-expected timestamp-determinism) assertion about the image space's bounds/dtype, verify `observation_space`'s `low=0, high=255, dtype=np.uint8` matches exactly what `MockStateReader.read_frame()` / `FrameStack` actually produce -- `check_env` is strict about this for `Box` spaces.

- [ ] **Step 6: Commit**

```bash
git add sekiro_ai/env/sekiro_env.py sekiro_ai/env/__init__.py tests/test_env.py
git commit -m "feat: switch SekiroEnv observation_space to stacked grayscale image"
```

---

### Task 6: `--save-frames` PNG export in `tests/test_state_reader.py`

**Files:**
- Modify: `tests/test_state_reader.py`

**Interfaces:**
- Consumes: `StateReader.read_frame()` (Tasks 3-4), works against whichever reader `build_reader()` returns (mock or, if `--live` finds a window, `PixelStateReader`).
- Produces: a `--save-frames N` CLI flag that dumps the last N captured frames as PNGs to `logs/frame_preview/`, for the user's own visual sanity-checking (per the design spec's explicit request: "测试文件中最好有直接显示mock生成图像的，方便你测试").

- [ ] **Step 1: Add the flag and export logic**

In `tests/test_state_reader.py`, add near the top (after the existing imports):

```python
import shutil

FRAME_PREVIEW_DIR = Path(__file__).resolve().parent.parent / "logs" / "frame_preview"
```

Add a new function, and call it from `main()`:

```python
def save_frame_previews(reader, n: int) -> None:
    import cv2

    if FRAME_PREVIEW_DIR.exists():
        shutil.rmtree(FRAME_PREVIEW_DIR)
    FRAME_PREVIEW_DIR.mkdir(parents=True)

    for i in range(n):
        frame = reader.read_frame()
        path = FRAME_PREVIEW_DIR / f"frame_{i:03d}.png"
        cv2.imwrite(str(path), frame)
    logger.info("Saved %d frame(s) to %s", n, FRAME_PREVIEW_DIR)
```

In `main()`, add the argparse flag:

```python
    parser.add_argument("--save-frames", type=int, default=0, help="Save N read_frame() outputs as PNGs to logs/frame_preview/ for visual inspection.")
```

And call it right before the existing `try:`/`for i in range(args.steps):` loop (so it runs once per invocation, using whichever reader `build_reader()` already picked):

```python
    if args.save_frames > 0:
        save_frame_previews(reader, args.save_frames)
```

- [ ] **Step 2: Run and inspect**

Run: `python tests/test_state_reader.py --save-frames 5`
Expected: log line `Saved 5 frame(s) to <path>\logs\frame_preview`, and 5 PNGs exist there afterward. Open one -- should look like Task 3's mock bar-graphic preview (or, if run with `--live` against a real game window, an actual downscaled grayscale game screenshot).

- [ ] **Step 3: Commit**

```bash
git add tests/test_state_reader.py
git commit -m "feat: add --save-frames PNG export to test_state_reader.py"
```

---

### Task 7: Switch PPO policy to `CnnPolicy`; update docs

**Files:**
- Modify: `train.py:65`
- Modify: `docs/architecture.md` (section 6's "RL 算法" bullet, and directory-structure header comment referencing scalar state)
- Modify: `docs/installation.md` (CUDA/CPU note referencing "13维向量,不是图像")
- Modify: `docs/training.md` (`policy="MlpPolicy"` step description)
- Modify: `docs/configuration.md` (already updated in Task 2 for the `observation`/`posture_color_hsv_range` fields; this task only touches prose describing the observation shape elsewhere in the file, if any)

**Interfaces:**
- Consumes: nothing new -- this task is a policy-string change plus documentation, using the `observation_space` already wired up in Task 5.
- Produces: `train.py` builds `PPO("CnnPolicy", ...)` instead of `PPO("MlpPolicy", ...)`, so `stable_baselines3` picks its default Atari-style CNN feature extractor (`NatureCNN`) matching an `(84,84,4)` uint8 image `Box` space. `play.py` needs no change -- `PPO.load()` restores the policy architecture from the saved model, so it only ever needs to know a model path, not a policy string (confirmed: `play.py` does not construct `PPO(...)` directly, only calls `PPO.load(args.model, env=env)`, so nothing in `play.py` currently hardcodes `MlpPolicy`/`CnnPolicy`).

- [ ] **Step 1: Change `train.py`'s policy string**

In `train.py:65`, change:

```python
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(TENSORBOARD_DIR))
```

to:

```python
        model = PPO("CnnPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(TENSORBOARD_DIR))
```

- [ ] **Step 2: Run the mock training smoke test**

Run: `python train.py --total-timesteps 1000 --run-name pixel_obs_smoke_test`
Expected: runs to completion without error, logs `Model saved to models/pixel_obs_smoke_test.zip`. SB3 will print `Using cpu device` and construct its default CNN feature extractor for the image observation space -- no manual policy_kwargs needed for this default architecture.

- [ ] **Step 3: Update documentation to match the new observation pipeline**

In `docs/architecture.md`:
- Update section 2 ("统一状态格式") to add a note that `GameState` is now only used for reward/restart, not the agent's observation (which is the stacked grayscale image described in the new `docs/configuration.md` `observation` section from Task 2).
- Update section 6's bullet `**RL 算法**：stable-baselines3 的 PPO，policy="MlpPolicy"（状态是低维向量，不需要 CNN）。` to describe `policy="CnnPolicy"` and the `(84,84,stack_size)` uint8 image observation instead.

In `docs/installation.md`, update the line `不需要提前安装 CUDA/GPU 相关依赖，stable-baselines3 的 MlpPolicy 在 CPU 上跑训练完全够用（状态是13维向量，不是图像，不需要卷积网络）。` to reflect that observations are now images processed by a CNN, but CPU training is still fine for this project's short episodes/small image size (no GPU requirement introduced, just note the change from vector to image).

In `docs/training.md`, update step 3 of "训练过程中发生了什么" (currently `创建（或加载）PPO模型，policy="MlpPolicy"（状态是低维数值向量，不需要卷积网络处理图像）。`) to say `policy="CnnPolicy"` and describe the image observation.

- [ ] **Step 4: Commit**

```bash
git add train.py docs/architecture.md docs/installation.md docs/training.md
git commit -m "feat: switch training to CnnPolicy for image observations; update docs"
```

---

## Final Verification

After all 7 tasks are complete, run the full mock-mode test suite in order (per `docs/installation.md`'s "验证安装" checklist, updated for this plan's new/changed test files) to confirm nothing regressed:

```powershell
python tests/test_frame_stack.py
python tests/test_observation_config.py
python tests/test_mock_frame.py
python tests/test_pixel_bar_extraction.py
python tests/test_state_reader.py
python tests/test_controller.py --dry-run
python tests/test_restart.py
python tests/test_reward.py
python tests/test_env.py
python tests/test_random_agent.py
python train.py --total-timesteps 1000 --run-name final_check
```

Every script should finish with no unhandled traceback and no `[FAIL]`/`FAILED` log lines (the pre-existing, expected `check_env` timestamp-determinism warning in `test_env.py` is the one known exception, already handled by that test's own logic). This is the point at which the user's follow-up request -- a training visualization UI -- becomes a new, separate brainstorming topic, per this plan's explicit scope boundary.

