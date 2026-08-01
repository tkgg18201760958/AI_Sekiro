# Sekiro RL Agent — 开发路线图

配套文档：[architecture.md](./architecture.md)（目录结构/模块职责/数据流/技术方案）、[risks.md](./risks.md)（潜在困难与解决方案）。

## 分阶段开发目标与验收标准

| 阶段 | 目标 | 交付物 | 验收标准 |
|---|---|---|---|
| 1 | State Reader | `mock_reader.py`, `tests/test_state_reader.py` | 运行测试脚本能实时打印模拟状态字典，字段与 schema 一致；若检测到游戏进程则尝试真实读取，否则自动降级为 mock 并打印提示 |
| 2 | Action Controller | `input_controller.py`, `action_map.py`, `tests/test_controller.py` | 手动运行 `python tests/test_controller.py --action 1`，在记事本/游戏窗口观察到对应键鼠事件；支持 `--dry-run` 只打印不执行 |
| 3 | Restart Manager | `restart_manager.py`, `tests/test_restart.py` | 用 mock 状态模拟 `player_dead=True`，观察日志显示执行了预设的重开按键序列，且返回新的初始状态 |
| 4 | Reward Calculator | `reward_calculator.py`, `tests/test_reward.py` | 输入若干组 (prev_state, curr_state) 手工样例，输出 reward 数值符合预期符号（Boss掉血为正、玩家掉血为负等） |
| 5 | Gymnasium 环境 | `sekiro_env.py`, `tests/test_env.py` | 用 mock 组件跑 `env.reset()`/`env.step()`，输出符合 `gymnasium.utils.env_checker.check_env` 校验 |
| 6 | 随机 Agent 测试 | 无新增模块，脚本化跑 env | 用随机动作跑若干 episode，日志系统正确记录每步数据，无异常崩溃 |
| 7 | PPO 训练接入 | `train.py`, `play.py`, `models/`, `logs/` | `train.py --total-timesteps 1000`（mock 环境）能跑通并保存模型；`play.py` 能加载模型跑推理 |
| 8 | 真实游戏对接 + 调优 | 切换到 `pixel_reader.py`（截图+图像识别，标定窗口坐标/颜色阈值），调整奖励/状态空间 | 在真实游戏中完整跑一次 episode 不崩溃，reward 曲线在 TensorBoard 可见 |

阶段1-6 全部可以在**不启动只狼游戏**的情况下用 mock 组件完成，这是模块化设计的核心价值：先把 RL 管线打通，最后才接真实游戏，降低调试成本。

## 测试脚本作用一览

- `tests/test_state_reader.py`：单独验证状态读取模块，支持 `--mock`（默认）和 `--live` 两种模式，实时打印状态。
- `tests/test_controller.py`：单独验证动作到键鼠的映射，支持指定单个 action 或遍历全部 action 逐一测试。
- `tests/test_restart.py`：注入模拟的死亡/胜利状态，验证重开流程被正确触发。
- `tests/test_reward.py`：喂入手工构造的状态对，打印计算出的 reward，用于校准奖励权重。
- `tests/test_env.py`：跑通 `reset`/`step` 循环（可配合随机动作），打印每步 action/state/reward，并可跑 gymnasium 的环境校验器。

## 当前进度

- [x] 阶段1：State Reader（`sekiro_ai/state_reader/` + `tests/test_state_reader.py`，已验证 scripted/random/--live fallback 三种模式）
- [x] 阶段2：Action Controller（`sekiro_ai/controller/` + `config/config.yaml` + `tests/test_controller.py`，已验证 `--dry-run` 全部7个action及单个action选择，config覆盖按键映射生效）
- [x] 阶段3：Restart Manager（`sekiro_ai/restart/` + `tests/test_restart.py`，已验证 player_dead/boss_dead 两种场景状态驱动重开）
- [x] 阶段4：Reward Calculator（`sekiro_ai/reward/` + `tests/test_reward.py`，10组手工样例符号全部符合预期）
- [x] 阶段5：Gymnasium 环境（`sekiro_ai/env/sekiro_env.py` + `tests/test_env.py`，通过 `check_env` 校验及手动多episode跑通）
- [x] 阶段6：随机 Agent 测试（`tests/test_random_agent.py`，scripted/random 模式各跑多episode无崩溃）
- [x] 阶段7：PPO 训练接入（`train.py`, `play.py`，`--total-timesteps 1000` 跑通并保存/加载模型）
- [x] 阶段8：Episode Logger + 真实游戏对接骨架（`sekiro_ai/logging/episode_logger.py` 已接入 `train.py` 的 `model.learn(callback=...)`，CSV按episode记录死亡原因/HP；`pixel_reader.py` 加入 `config.yaml` 的 `calibration` 配置段与 `missing_calibration()` 检查，`read()` 会明确报告缺失的标定项。**真实截图标定（HP/体力条像素坐标、颜色阈值、"危"图标模板）仍需在游戏实际运行时手动测量填入 `config.yaml`，无法在没有游戏窗口的情况下完成**）

> 每完成一个阶段，回到本文件勾选对应项，保持进度可追溯。
