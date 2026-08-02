# 像素帧观测（Pixel Frame Observation）实施计划

> **给自动化执行者的提示：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务（task-by-task）执行本计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 用一个直接从游戏像素读取、经过堆叠的 84x84 灰度图像观测，替换掉 `SekiroEnv` 现有的 13 维标量观测，同时保留数值型 HP/架势提取（仅用于 reward/restart），并保证完全离线的 mock 模式测试路径依然可用。

**架构：** `StateReader` 新增第二个读取方法 `read_frame() -> np.ndarray`（单帧 84x84 uint8 灰度图），与现有的 `read() -> GameState`（用于 reward/restart 的数值状态）并存。新增的 `FrameStack` 模块按可配置的间隔（`frame_skip`）采样帧，并维护一个大小为 `stack_size` 的滑动窗口，这个窗口构成 `SekiroEnv` 的 `Box(shape=(84,84,stack_size), dtype=uint8)` 观测。`MockStateReader` 为 `read_frame()` 渲染合成的血条图形（使用与 `PixelStateReader` 数值提取相同的 `calibration` rect 来定位），因此整条训练管线在没有游戏窗口的情况下依然能跑通。`train.py`/`play.py` 将 PPO 的 policy 从 `MlpPolicy` 切换为 `CnnPolicy`。

**技术栈：** Python 3.13、OpenCV（`opencv-python`，requirements.txt 中已有）、`mss`（屏幕截图）、NumPy、Gymnasium、stable-baselines3 PPO 的 `CnnPolicy`。

## 全局约束

- `observation_space` 变为 `spaces.Box(low=0, high=255, shape=(84,84,stack_size), dtype=np.uint8)`，替换掉原来的 `Box(shape=(13,), dtype=np.float32)`。默认 `frame_size=[84,84]`、`frame_skip=4`、`stack_size=4`，均可通过 `config.yaml` 新增的 `observation:` 部分覆盖。
- `StateReader.read_frame()` 是新增的**抽象**方法（不是可选的）——每一个 `StateReader` 子类，包括测试替身（test doubles），都必须实现它。
- `read()` 和 `read_frame()` 不能共享截图缓存。`RestartManager.run()` 会在阻塞循环中反复调用 `reader.read()`（`restart/restart_manager.py:88-124`）等待状态发生变化；如果共享/使用过期的截图，这个循环会一直看到冻结的状态并最终超时。每次截图都是独立的，即便在真实模式下这意味着每个环境步骤要多做一次 `mss` 抓图。
- `PixelStateReader.read()` 中的数值型 HP/架势提取只需要产出 `player_hp`、`boss_hp`、`player_posture`、`boss_posture`、`player_dead`、`boss_dead`。`boss_action`、`can_parry`、`player_hit`、`player_pos`、`boss_pos`、`distance` 保持 `GameState` 的默认值——"危"字图标模板匹配以及位置/距离估算明确不在本次范围内（依据设计文档）。
- 训练可视化 UI 明确不在本计划范围内（根据用户要求，将作为单独的一轮后续 brainstorming）。
- 真实游戏相关功能保持仅限 Windows，并遵循项目现有的惰性导入约定：`cv2`/`mss`/`win32gui` 在 `PixelStateReader` 的方法内部导入，绝不在模块顶层导入。
- `mock_reader.py` 即便为了新增的图像渲染功能，也**不能**导入 `cv2`（或 `docs/installation.md` 中"仅 `--live` 需要"清单里的任何其他包）——改用纯 NumPy 切片来绘制合成帧，这样阶段 1-7 "无需安装额外依赖即可运行"的保证才能继续成立。`numpy`/`PyYAML` 已经是始终安装的核心依赖（依据 `docs/installation.md` 的依赖表），可以在任何地方的模块顶层导入。
- 本项目没有 pytest，也没有测试运行器配置（已确认：`venv` 中未安装 `pytest`）。测试是 `tests/` 目录下不依赖/依赖 argparse 的独立脚本，直接用 `python tests/test_X.py` 运行，采用手写的 `assert`/PASS-FAIL 日志模式（参见 `tests/test_reward.py` 的风格）。本计划新增的所有测试都遵循同样的风格，而不是 pytest。
- 设计文档（开始前请先阅读）：`docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md`。

---

## 文件结构

新建文件：
- `sekiro_ai/state_reader/frame_stack.py` —— `FrameStack` 类（纯 NumPy，不依赖 StateReader/config）。
- `sekiro_ai/state_reader/observation_config.py` —— `ObservationConfig` 数据类，加载 `config.yaml` 的 `observation:` 部分（沿用代码库中已有的 `reward/reward_calculator.py` 里 `RewardWeights.from_config()` 的模式）。
- `tests/test_frame_stack.py`、`tests/test_mock_frame.py`、`tests/test_pixel_bar_extraction.py` —— 新增的测试脚本，每个新的可测试单元对应一个。

修改文件：
- `sekiro_ai/state_reader/base.py` —— 为 `StateReader` 添加抽象方法 `read_frame()`。
- `sekiro_ai/state_reader/mock_reader.py` —— 添加 `render_frame()`/`read_frame()`，以及按 calibration 缩放的合成血条绘制。
- `sekiro_ai/state_reader/pixel_reader.py` —— 实现 `read()`（bar-rect + HSV 填充比例）和 `read_frame()`（截图 + 灰度化 + resize）；添加 `_capture_client_area()` 和 `bar_fill_ratio()` 辅助函数。
- `sekiro_ai/env/sekiro_env.py` —— `observation_space` 变为图像型 `Box`；`reset()`/`step()` 通过 `FrameStack` 调用 `read_frame()`；移除 `OBS_DIM`/`state_to_obs`。
- `sekiro_ai/env/__init__.py` —— 去掉 `OBS_DIM` 的导出（已不存在）。
- `tests/test_restart.py` —— `ScriptedDeathReader` 增加 `read_frame()` 占位实现（现在该方法是抽象方法，必须实现）。
- `tests/test_env.py` —— 断言新的图像型 `observation_space` 形状/dtype。
- `tests/test_state_reader.py` —— 添加 `--save-frames N` 的 PNG 导出选项。
- `config/config.yaml` —— 新增 `observation:` 部分；新增 `calibration.posture_color_hsv_range` 字段。
- `train.py`、`play.py` —— 用 `PPO("CnnPolicy", ...)` 替代 `PPO("MlpPolicy", ...)`。
- `docs/architecture.md`、`docs/installation.md`、`docs/training.md`、`docs/configuration.md` —— 更新"13维向量 / MlpPolicy，不需要 CNN"的相关表述，改为描述新的图像观测。

---

### 任务 1：FrameStack 模块

**文件：**
- 新建：`sekiro_ai/state_reader/frame_stack.py`
- 测试：`tests/test_frame_stack.py`

**接口：**
- 依赖：无（纯 NumPy，不依赖项目内其他模块）。
- 产出：`class FrameStack`，包含：
  - `__init__(self, stack_size: int = 4, frame_skip: int = 4)`
  - `reset(self, frame: np.ndarray) -> np.ndarray` —— 清空内部状态，用 `frame`（复制 `stack_size` 次）填满整个堆栈，返回 `(H, W, stack_size)` 的堆叠数组。
  - `push(self, frame: np.ndarray) -> np.ndarray` —— 每个环境步骤调用一次，传入最新捕获的帧；内部记录调用次数，只有每隔 `frame_skip` 次调用才把新帧轮转进堆栈（在被跳过的调用中重复最近一次入栈的帧），返回当前的 `(H, W, stack_size)` 堆叠数组。
  - 传入的帧始终是完整尺寸的 `(H, W)` 二维 uint8 数组（灰度图，调用方已完成 resize）；`FrameStack` 本身不做 resize/颜色转换，只负责堆叠。

下游模块（`SekiroEnv`，任务 5）依赖这两个方法名和这个确切行为：`reset()` 在每个 episode 开始时调用一次，`push()` 在每次 `step()` 调用一次，两者都返回 `(H, W, stack_size)` 的 uint8 ndarray，最旧的帧在最后一维的索引 0 处，最新的帧在索引 `stack_size - 1` 处。

- [ ] **步骤 1：编写会失败的测试**

```python
"""sekiro_ai.state_reader.frame_stack.FrameStack 的独立测试脚本。

用法：
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
    """一个填充单一数值的 8x8 帧，这样堆叠后的切片可以通过标量填充值轻易区分。"""
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
    # frame_skip=4：只有每第 4 次 push() 调用才会轮转进新帧。
    # reset 之后的第 1-3 次调用不应改变堆栈。
    before = fs.push(make_frame(99)).copy()
    stacked = fs.push(make_frame(50))
    ok = np.array_equal(before, stacked)
    logger.info("[%s] pushes before frame_skip interval don't change the stack", "PASS" if ok else "FAIL")
    return ok


def test_push_at_skip_interval_rotates_newest_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=2)
    fs.reset(make_frame(0))
    fs.push(make_frame(11))  # 第 1 次调用：还在跳过间隔内，不轮转
    stacked = fs.push(make_frame(22))  # 第 2 次调用：达到跳过间隔，轮转进 22
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

- [ ] **步骤 2：运行测试确认它会失败**

运行：`python tests/test_frame_stack.py`
预期：`ModuleNotFoundError: No module named 'sekiro_ai.state_reader.frame_stack'`

- [ ] **步骤 3：编写最小实现**

```python
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
```

- [ ] **步骤 4：运行测试确认它通过**

运行：`python tests/test_frame_stack.py`
预期：`All 4 FrameStack tests PASSED.`（没有 `[FAIL]` 行，且没有 traceback 退出）

- [ ] **步骤 5：提交**

```bash
git add sekiro_ai/state_reader/frame_stack.py tests/test_frame_stack.py
git commit -m "feat: add FrameStack for frame-skip image observation stacking"
```

---

### 任务 2：Observation 配置 + calibration 架势颜色范围

**文件：**
- 新建：`sekiro_ai/state_reader/observation_config.py`
- 修改：`config/config.yaml`
- 修改：`docs/configuration.md`

**接口：**
- 依赖：`sekiro_ai.utils.config_loader.load_config()`（已有，`sekiro_ai/utils/config_loader.py:20`）。
- 产出：`class ObservationConfig`，字段为 `frame_size: tuple[int, int] = (84, 84)`、`frame_skip: int = 4`、`stack_size: int = 4`，以及类方法 `from_config(config: dict | None = None) -> "ObservationConfig"`（沿用 `sekiro_ai/reward/reward_calculator.py:46-52` 中 `RewardWeights.from_config` 的模式——同样的"在默认值基础上合并覆盖"模式）。任务 3（`MockStateReader`）和任务 4（`PixelStateReader`）都会各自构造一个这样的实例；任务 5（`SekiroEnv`）会构造一个用来确定其 `FrameStack` 的大小。
- 同时产出：`config.yaml` 新增的 `calibration.posture_color_hsv_range` 字段，直接通过 `load_config().get("calibration", {}).get("posture_color_hsv_range")` 读取（这一个字段不需要新的配置类——它和现有的 `hp_color_hsv_range` 一起放在同一个 `calibration:` 部分里，沿用该部分在 `pixel_reader.py` 中现有的纯字典访问风格）。

- [ ] **步骤 1：编写会失败的测试**

新建 `tests/test_observation_config.py`：

```python
"""sekiro_ai.state_reader.observation_config.ObservationConfig 的独立测试脚本。

用法：
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

- [ ] **步骤 2：运行测试确认它会失败**

运行：`python tests/test_observation_config.py`
预期：`ModuleNotFoundError: No module named 'sekiro_ai.state_reader.observation_config'`

- [ ] **步骤 3：编写最小实现**

```python
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
```

- [ ] **步骤 4：运行测试确认它通过**

运行：`python tests/test_observation_config.py`
预期：`All 3 ObservationConfig tests PASSED.`

- [ ] **步骤 5：向 config.yaml 添加 `observation:` 部分和 `posture_color_hsv_range`**

在 `config/config.yaml` 中添加这个新的顶层部分（放在现有的 `calibration:` 部分之后）：

```yaml
observation:
  # 喂给 PPO agent 的像素帧观测形状（sekiro_ai.state_reader.observation_config）。
  # frame_size：堆叠前单帧 resize 的目标尺寸 [width, height]。
  # frame_skip：每隔 N 个环境步骤才采样一个新帧入栈（在此期间，
  #   最近入栈的帧会重复）——在 60fps 下，这样能让堆栈的时间窗口
  #   足够宽以体现出运动，而不是 4 张几乎一样的帧。
  # stack_size：滑动窗口中保留的帧数；observation 的形状为
  #   (frame_size[1], frame_size[0], stack_size)。
  frame_size: [84, 84]
  frame_skip: 4
  stack_size: 4
```

再向现有的 `calibration:` 部分添加 `posture_color_hsv_range`，紧跟在 `hp_color_hsv_range` 之后：

```yaml
  posture_color_hsv_range: [[15, 80, 80], [35, 255, 255]]   # 黄色箭头架势条填充色，与上面红/橙的 HP 条区分开
```

（依据 `GAME_PIC/BOSS.png`/`GAME_PIC/WITHOUT_BOSS.png` 测得：这个范围能让 HP 条和架势条的 HSV 区分度做到——在两张样例截图中，空架势条正确读出接近 0 的填充比例，满架势条读出高填充比例——是在规划阶段通过手动 `cv2.inRange` + 逐列填充比例检查验证的，不是通过自动化测试验证的，因为目前还没有真实的游戏窗口。）

- [ ] **步骤 6：为新增的配置字段编写文档**

在 `docs/configuration.md` 中，在现有的 `calibration` 部分之后添加一个小节，说明 `observation:`（字段、默认值、修改 `frame_skip`/`stack_size` 对训练的影响），并在 `calibration` 字段表中添加 `posture_color_hsv_range`，遵循该文档现有的风格（参见文档中已有的 `hp_color_hsv_range` 那一行）。

- [ ] **步骤 7：提交**

```bash
git add sekiro_ai/state_reader/observation_config.py tests/test_observation_config.py config/config.yaml docs/configuration.md
git commit -m "feat: add ObservationConfig and observation/posture_color config fields"
```

---

### 任务 3：`StateReader.read_frame()` 抽象方法 + `MockStateReader` 实现

**文件：**
- 修改：`sekiro_ai/state_reader/base.py`
- 修改：`sekiro_ai/state_reader/mock_reader.py`
- 修改：`tests/test_restart.py`（其 `ScriptedDeathReader` 测试替身必须实现这个现已变为抽象方法的方法）
- 测试：`tests/test_mock_frame.py`

**接口：**
- 依赖：`ObservationConfig.from_config()`（任务 2，`sekiro_ai/state_reader/observation_config.py`）、`load_config()`（已有，`sekiro_ai/utils/config_loader.py`）。
- 产出：在基类（`sekiro_ai/state_reader/base.py`）上把 `StateReader.read_frame(self) -> np.ndarray` 定义为 `@abstractmethod`——每个子类都必须实现它，否则无法实例化。`MockStateReader.read_frame()` 返回一个按 `ObservationConfig.frame_size` 确定大小的 `(H, W)` uint8 灰度 ndarray（注意：NumPy 的形状顺序是 `(height, width)`，即 `(frame_size[1], frame_size[0])`），根据当前 `GameState` 的 `player_hp`/`boss_hp`/`player_posture`/`boss_posture` 合成血条图形。这是任务 5（`SekiroEnv`）和任务 6（测试更新）会调用的内容。

- [ ] **步骤 1：先把 `StateReader.read_frame()` 更新为抽象方法，并给 `ScriptedDeathReader` 加占位实现，然后再编写 MockStateReader 的真实实现**

编辑 `sekiro_ai/state_reader/base.py`，添加导入和抽象方法：

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

编辑 `tests/test_restart.py` 中的 `ScriptedDeathReader`（目前在 `tests/test_restart.py:31-58`），添加一个占位的 `read_frame()`——现在这个方法是抽象方法，类要能实例化就必须实现它，即便这个测试本身不会真正用到帧读取：

```python
    def read_frame(self):
        import numpy as np
        return np.zeros((84, 84), dtype=np.uint8)
```

把这个方法加到 `class ScriptedDeathReader(StateReader):` 内部，和它现有的 `read()`/`reset()` 放在一起。

- [ ] **步骤 2：运行 `test_restart.py` 确认它依然通过（证明加了占位实现之后，抽象方法不会破坏现有的测试替身）**

运行：`python tests/test_restart.py`
预期：`PASS: restart sequence completed and returned a fresh state.`（和本任务改动之前一样）

- [ ] **步骤 3：为 MockStateReader.read_frame() 编写会失败的测试**

```python
"""MockStateReader 的 read_frame() 合成图像生成功能的独立测试脚本。

用法：
    python tests/test_mock_frame.py
    python tests/test_mock_frame.py --save-frames 5   # 同时导出 PNG 供肉眼检查
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
    """满血条应该比同一位置几乎空的血条有更多的亮（已填充）像素 ——
    这是"绘制出的血条确实跟随 GameState 数值变化"的一个代理验证方式，
    不需要精确到像素级的断言。"""
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
        reader.read()  # 推进脚本化轨迹
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

- [ ] **步骤 4：运行测试确认它会失败**

运行：`python tests/test_mock_frame.py`
预期：`AttributeError: 'MockStateReader' object has no attribute 'read_frame'`

- [ ] **步骤 5：实现 `MockStateReader.read_frame()`**

添加到 `sekiro_ai/state_reader/mock_reader.py`（文件顶部的导入、类上的方法、文件底部的模块级辅助函数）：

```python
# 添加到顶层导入中：
import numpy as np

from .observation_config import ObservationConfig
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
```

把这个方法添加到 `class MockStateReader(StateReader)` 上，比如放在 `read()` 之后：

```python
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
```

把这些模块级辅助函数添加到 `mock_reader.py` 靠底部的位置，和现有的 `_clamp01` 放在一起：

```python
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
```

- [ ] **步骤 6：运行测试确认它通过**

运行：`python tests/test_mock_frame.py`
预期：`All 2 MockStateReader.read_frame() tests PASSED.`

- [ ] **步骤 7：生成并肉眼检查预览帧**

运行：`python tests/test_mock_frame.py --save-frames 8`
预期：日志行 `Saved 8 mock frames to <path>\logs\frame_preview`，且该目录下存在 8 个 PNG 文件。用图片查看器打开其中几张——确认出现了 4 段短的亮色水平条，大致位于 `GAME_PIC/BOSS.png` 中 HP/架势条所在的位置（左下、中下、左上、中上），背景为黑色，并且这些色条的长度随着这 8 帧、脚本化轨迹的推进而明显变化。

- [ ] **步骤 8：提交**

```bash
git add sekiro_ai/state_reader/base.py sekiro_ai/state_reader/mock_reader.py tests/test_restart.py tests/test_mock_frame.py
git commit -m "feat: add read_frame() to StateReader; mock synthetic bar rendering"
```

---

### 任务 4：`PixelStateReader.read_frame()` 和 `read()` 实现

**文件：**
- 修改：`sekiro_ai/state_reader/pixel_reader.py`
- 修改：`config/config.yaml`（添加 `posture_color_hsv_range`，已经在任务 2 中完成——本任务只是*使用*它）
- 测试：`tests/test_pixel_bar_extraction.py`

**接口：**
- 依赖：`ObservationConfig.from_config()`（任务 2）。`config.yaml` 中的 `calibration.player_hp_bar`/`boss_hp_bar`/`player_posture_bar`/`boss_posture_bar`/`resolution`/`hp_color_hsv_range`/`posture_color_hsv_range`（除 `posture_color_hsv_range`（任务 2 新增）外均为已有字段）。
- 产出：`PixelStateReader.read_frame(self) -> np.ndarray`（整窗口截图、灰度化、resize 到 `ObservationConfig.frame_size`），以及一个能真正工作的 `PixelStateReader.read(self) -> GameState`，它根据 bar-rect 的 HSV 填充比例填充 `player_hp`、`boss_hp`、`player_posture`、`boss_posture`、`player_dead`、`boss_dead`（`GameState` 中的其他字段保持其数据类默认值，符合设计文档中对 `boss_action`/`player_pos`/`boss_pos`/`distance`/`can_parry`/`player_hit` 的明确范围排除）。同时产出两个可供其他代码/测试直接导入调用的辅助函数：`bar_fill_ratio(bgr_image: np.ndarray, rect: tuple[int,int,int,int], hsv_range: tuple) -> float` 和 `capture_client_area(window) -> np.ndarray`（BGR ndarray）。

本任务在实现期间无法对着真实游戏窗口验证（因为开发期间没有游戏在运行）——它是对着仓库里已有的两张静态截图测试的，`GAME_PIC/BOSS.png`（boss 交战中，战斗中期的血条水平）和 `GAME_PIC/WITHOUT_BOSS.png`（玩家处于非战斗状态，高 HP，架势条为空），把它们当作 `mss` 实时截图的替代品喂进去。这能验证 HSV 阈值和填充比例的数学计算是正确的；但**不能**验证真实截图时机/`mss`/窗口查找相对于真实游戏的正确性，依据设计文档的范围边界，这部分留作以后的手动步骤（记录在 `docs/live_game.md` 中，本任务不改动该文档）。

- [ ] **步骤 1：编写会失败的测试**

```python
"""PixelStateReader 血条填充比例提取数学计算的独立测试脚本，针对已经
提交到 GAME_PIC/ 目录下的两张静态截图运行（开发期间没有真实游戏窗口可用——
关于这个测试能验证什么、不能验证什么，参见
docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md
的范围说明）。

用法：
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

# 这些 rect 是从 config/config.yaml 的 calibration 部分复制过来的
# （对着 GAME_PIC/BOSS.png，一张 1280x720 的截图测量得到）。
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
    """WITHOUT_BOSS.png 里完全没有渲染 boss HP 条（没有 boss 交战）——
    该 rect 应该读出接近零的填充比例，因为那里根本没有红色的血条填充色。"""
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

- [ ] **步骤 2：运行测试确认它会失败**

运行：`python tests/test_pixel_bar_extraction.py`
预期：`ImportError: cannot import name 'bar_fill_ratio' from 'sekiro_ai.state_reader.pixel_reader'`

- [ ] **步骤 3：在 `pixel_reader.py` 中实现 `bar_fill_ratio`、`capture_client_area`、`read()` 和 `read_frame()`**

替换整个 `read()` 方法，并添加新的辅助函数。这个模块保持其现有的惰性导入约定（`cv2`/`mss` 在函数内部导入，绝不在模块顶层）。

```python
def bar_fill_ratio(bgr_image, rect: tuple[int, int, int, int], hsv_range) -> float:
    """`rect` 的宽度中，在其高度范围内至少有一个像素匹配 `hsv_range`
    （一对 OpenCV-HSV 的 [[H,S,V]min, [H,S,V]max]）的比例——这是一种
    对水平血条填充比例的估算方式，对血条填充色在亮度/明暗上的轻微变化
    比较鲁棒（每一列只需要有一个匹配的像素，不要求整列都匹配）。"""
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
    """截取 `window` 客户区（不包括标题栏/边框）的截图，返回 BGR
    ndarray，使用该窗口当前在屏幕上的位置（通过 win32gui.ClientToScreen）
    和尺寸（通过 win32gui.GetClientRect）。"""
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

把现有的 `read()` 方法体（目前一直抛出 `NotImplementedError`，见 `pixel_reader.py:83-102`）替换为：

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

在 `pixel_reader.py` 顶部添加两个新的导入（和现有的 `from ..utils.config_loader import load_config` 放在一起）：

```python
import time

import numpy as np

from .observation_config import ObservationConfig
```

注意：`read_frame()` 故意**不**检查 `missing_calibration()`——它只需要窗口存在，不需要 bar-rect 校准，这和设计文档的错误处理小节一致（"即使 HP 校准没填，图像分支也能跑"）。

- [ ] **步骤 4：运行测试确认它通过**

运行：`python tests/test_pixel_bar_extraction.py`
预期：`All 5 bar extraction tests PASSED.`

如果任何 HSV 范围断言失败，根据 `GAME_PIC/BOSS.png`/`GAME_PIC/WITHOUT_BOSS.png` 中实际的像素值来调整测试里的 `HP_HSV_RANGE`/`POSTURE_HSV_RANGE`（以及 `pixel_reader.py` 和 `config/config.yaml` 中相应的 `hp_color_hsv_range`/`posture_color_hsv_range` 默认值）——用 `cv2.cvtColor(cv2.imread(...), cv2.COLOR_BGR2HSV)` 检查具体 rect 区域来找出真实的范围，而不是盲目猜测。

- [ ] **步骤 5：提交**

```bash
git add sekiro_ai/state_reader/pixel_reader.py tests/test_pixel_bar_extraction.py
git commit -m "feat: implement PixelStateReader.read() bar extraction and read_frame() capture"
```

---

### 任务 5：`SekiroEnv` 图像观测空间

**文件：**
- 修改：`sekiro_ai/env/sekiro_env.py`
- 修改：`sekiro_ai/env/__init__.py`
- 修改：`tests/test_env.py`

**接口：**
- 依赖：`StateReader.read()`/`read_frame()`（任务 3-4）、`FrameStack`（任务 1，`sekiro_ai/state_reader/frame_stack.py` 的 `FrameStack(stack_size, frame_skip)`，其 `.reset(frame)`/`.push(frame)` 都返回 `(H, W, stack_size)` 的 uint8 ndarray）、`ObservationConfig.from_config()`（任务 2）。
- 产出：`SekiroEnv.observation_space = spaces.Box(low=0, high=255, shape=(frame_size[1], frame_size[0], stack_size), dtype=np.uint8)`。彻底移除 `OBS_DIM` 常量和 `state_to_obs()` 方法（一旦观测变为图像型，它们就没有意义了）——任务 6 依赖于任何地方都不再导入 `OBS_DIM` 这一前提。

- [ ] **步骤 1：先更新 `tests/test_env.py` 中针对新观测形状的断言（先写会失败的测试）**

在 `tests/test_env.py` 中，在 `run_manual_episodes` 内部、紧跟在 `obs, info = env.reset(seed=seed)` 之后（目前在 `tests/test_env.py:68`）添加这个检查：

```python
            expected_shape = env.observation_space.shape
            if obs.shape != expected_shape or obs.dtype != np.uint8:
                logger.error(
                    "episode=%d FAIL: obs.shape=%s dtype=%s, expected shape=%s dtype=uint8",
                    ep, obs.shape, obs.dtype, expected_shape,
                )
```

在 `tests/test_env.py` 的导入中添加 `import numpy as np`（目前缺失，因为旧的标量向量代码在测试文件里从未直接需要它）。

- [ ] **步骤 2：运行测试确认它会失败**

运行：`python tests/test_env.py --skip-check-env`
预期：失败或报错，因为此时 `SekiroEnv.observation_space` 还是旧的 `Box(shape=(13,))`，一旦任务 5 的实现改动落地，`obs.shape` 就不会再匹配图像形状的预期（这一步是在建立"改动前"的基线，展示旧的形状；如果在任何代码改动之前运行，它只会针对旧的 13 维空间平凡地通过，这也没问题——真正有意义的"先失败后通过"的转变发生在步骤 2 和步骤 4 之间，随着底层实现的改动而完成）。

- [ ] **步骤 3：重写 `sekiro_ai/env/sekiro_env.py`**

替换该文件的导入、模块文档字符串、`OBS_DIM`、`__init__`、`reset()`、`step()` 和 `state_to_obs()`：

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

请注意这个改动彻底移除了构造函数参数 `max_distance`，也一并移除了 `state_to_obs()`——`distance_norm` 以前只用于构建旧的标量观测向量，而这个向量已经不存在了。`GameState.distance` 在新的管线中不再被使用（符合设计文档的范围排除）。

- [ ] **步骤 4：更新 `sekiro_ai/env/__init__.py`**

```python
from .factory import build_env, build_reader
from .sekiro_env import SekiroEnv

__all__ = ["SekiroEnv", "build_env", "build_reader"]
```

（从导入和 `__all__` 中都去掉了 `OBS_DIM`，因为 `sekiro_env.py` 已经不再定义它。）

- [ ] **步骤 5：运行测试确认它通过**

运行：`python tests/test_env.py`
预期：`gymnasium check_env PASSED.`，以及类似 `All ... FrameStack tests PASSED` 风格的手动 episode 日志，没有任何 `FAIL` 行；每次 episode reset 时记录的 `obs.shape` 应该等于 `(84, 84, 4)`（或者 `config.yaml` 的 `observation` 部分当前指定的任何值）。

如果 `check_env` 针对图像空间的边界/dtype 抛出一个新的（不是既有的、已经预期到的时间戳确定性）断言错误，验证 `observation_space` 的 `low=0, high=255, dtype=np.uint8` 是否与 `MockStateReader.read_frame()` / `FrameStack` 实际产出的完全一致——`check_env` 对 `Box` 空间这方面的检查是很严格的。

- [ ] **步骤 6：提交**

```bash
git add sekiro_ai/env/sekiro_env.py sekiro_ai/env/__init__.py tests/test_env.py
git commit -m "feat: switch SekiroEnv observation_space to stacked grayscale image"
```

---

### 任务 6：在 `tests/test_state_reader.py` 中添加 `--save-frames` PNG 导出功能

**文件：**
- 修改：`tests/test_state_reader.py`

**接口：**
- 依赖：`StateReader.read_frame()`（任务 3-4），针对 `build_reader()` 返回的任意 reader 工作（mock，或者如果 `--live` 找到了窗口，则是 `PixelStateReader`）。
- 产出：一个 `--save-frames N` 的命令行参数，把最近 N 次捕获的帧导出为 PNG 保存到 `logs/frame_preview/`，供用户自行进行肉眼合理性检查（依据设计文档的明确要求："测试文件中最好有直接显示mock生成图像的，方便你测试"）。

- [ ] **步骤 1：添加该参数和导出逻辑**

在 `tests/test_state_reader.py` 中，在现有导入之后靠近顶部的位置添加：

```python
import shutil

FRAME_PREVIEW_DIR = Path(__file__).resolve().parent.parent / "logs" / "frame_preview"
```

添加一个新函数，并在 `main()` 中调用它：

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

在 `main()` 中添加这个 argparse 参数：

```python
    parser.add_argument("--save-frames", type=int, default=0, help="Save N read_frame() outputs as PNGs to logs/frame_preview/ for visual inspection.")
```

并在现有的 `try:`/`for i in range(args.steps):` 循环之前调用它（这样它在每次运行时只会执行一次，使用 `build_reader()` 已经选好的那个 reader）：

```python
    if args.save_frames > 0:
        save_frame_previews(reader, args.save_frames)
```

- [ ] **步骤 2：运行并检查**

运行：`python tests/test_state_reader.py --save-frames 5`
预期：日志行 `Saved 5 frame(s) to <path>\logs\frame_preview`，此后该目录下存在 5 个 PNG 文件。打开其中一个——应该看起来和任务 3 的 mock 血条图预览类似（或者，如果加 `--live` 对着真实游戏窗口运行，则是一张真实的、经过降采样的灰度游戏截图）。

- [ ] **步骤 3：提交**

```bash
git add tests/test_state_reader.py
git commit -m "feat: add --save-frames PNG export to test_state_reader.py"
```

---

### 任务 7：把 PPO 的 policy 切换为 `CnnPolicy`；更新文档

**文件：**
- 修改：`train.py:65`
- 修改：`docs/architecture.md`（第 6 节的"RL 算法"要点，以及引用标量状态的目录结构头部注释）
- 修改：`docs/installation.md`（引用"13维向量,不是图像"的 CUDA/CPU 说明）
- 修改：`docs/training.md`（`policy="MlpPolicy"` 那一步的描述）
- 修改：`docs/configuration.md`（已在任务 2 中为 `observation`/`posture_color_hsv_range` 字段更新过；本任务只涉及文件中其他地方描述观测形状的文字，如果有的话）

**接口：**
- 依赖：无新增内容——本任务是一个 policy 字符串的改动加文档更新，使用的是任务 5 中已经接好的 `observation_space`。
- 产出：`train.py` 构建 `PPO("CnnPolicy", ...)` 而不是 `PPO("MlpPolicy", ...)`，这样 `stable_baselines3` 就会选用它默认的 Atari 风格 CNN 特征提取器（`NatureCNN`），匹配一个 `(84,84,4)` uint8 图像 `Box` 空间。`play.py` 不需要任何改动——`PPO.load()` 会从保存的模型中恢复 policy 架构，所以它只需要知道模型路径，不需要知道 policy 字符串（已确认：`play.py` 不会直接构造 `PPO(...)`，只调用 `PPO.load(args.model, env=env)`，所以 `play.py` 中目前没有任何地方硬编码 `MlpPolicy`/`CnnPolicy`）。

- [ ] **步骤 1：修改 `train.py` 的 policy 字符串**

在 `train.py:65` 中，把：

```python
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(TENSORBOARD_DIR))
```

改为：

```python
        model = PPO("CnnPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(TENSORBOARD_DIR))
```

- [ ] **步骤 2：运行 mock 训练冒烟测试**

运行：`python train.py --total-timesteps 1000 --run-name pixel_obs_smoke_test`
预期：无错误地跑完全程，日志输出 `Model saved to models/pixel_obs_smoke_test.zip`。SB3 会打印 `Using cpu device`，并为图像观测空间构建它默认的 CNN 特征提取器——这个默认架构不需要手动传 policy_kwargs。

- [ ] **步骤 3：更新文档以匹配新的观测管线**

在 `docs/architecture.md` 中：
- 更新第 2 节（"统一状态格式"），加一条说明：`GameState` 现在只用于 reward/restart，不再是 agent 的观测（观测是任务 2 中 `docs/configuration.md` 新增的 `observation` 部分所描述的堆叠灰度图像）。
- 更新第 6 节的要点 `**RL 算法**：stable-baselines3 的 PPO，policy="MlpPolicy"（状态是低维向量，不需要 CNN）。`，改为描述 `policy="CnnPolicy"` 和 `(84,84,stack_size)` 的 uint8 图像观测。

在 `docs/installation.md` 中，更新这一行：`不需要提前安装 CUDA/GPU 相关依赖，stable-baselines3 的 MlpPolicy 在 CPU 上跑训练完全够用（状态是13维向量，不是图像，不需要卷积网络）。`，改为体现观测现在是经 CNN 处理的图像，但对于本项目较短的 episode/较小的图像尺寸，CPU 训练依然完全够用（不引入 GPU 需求，只需说明从向量到图像的变化）。

在 `docs/training.md` 中，更新"训练过程中发生了什么"的第 3 步（目前是 `创建（或加载）PPO模型，policy="MlpPolicy"（状态是低维数值向量，不需要卷积网络处理图像）。`），改为 `policy="CnnPolicy"` 并描述图像观测。

- [ ] **步骤 4：提交**

```bash
git add train.py docs/architecture.md docs/installation.md docs/training.md
git commit -m "feat: switch training to CnnPolicy for image observations; update docs"
```

---

## 最终验证

在全部 7 个任务完成之后，按顺序运行完整的 mock 模式测试套件（依据 `docs/installation.md` 中"验证安装"清单，针对本计划新增/改动的测试文件做了更新），确认没有出现回归：

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

每个脚本都应该在没有未处理的 traceback、没有 `[FAIL]`/`FAILED` 日志行的情况下运行完毕（`test_env.py` 中既有的、已经预期到的 `check_env` 时间戳确定性警告是唯一已知例外，那个测试自身的逻辑已经处理了它）。到了这一步，用户的后续请求——训练可视化 UI——就会成为一个新的、独立的 brainstorming 话题，依据本计划明确的范围边界。

