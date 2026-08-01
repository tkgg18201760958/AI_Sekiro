# Sekiro RL Agent

用强化学习（PPO）训练一个自动打只狼 Boss 的 Agent。

项目采用模块化设计：State Reader → Reward Calculator → Gymnasium Env → PPO Agent → Input Controller → Restart Manager → Episode Logger。阶段1-7 全部可以在**不启动只狼游戏**的情况下用 mock 组件跑通完整训练管线；只有真正对接真实游戏画面（阶段8）时才需要游戏运行。

## 快速开始

```powershell
git clone https://github.com/tkgg18201760958/AI_Sekiro.git
cd AI_Sekiro
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python tests/test_random_agent.py   # 不需要打开游戏，跑通即说明环境没问题
python train.py --total-timesteps 1000   # mock 环境训练一把试试
```

更详细的步骤和依赖说明见 [docs/installation.md](docs/installation.md)。

## 文档导览

| 文档 | 内容 |
|---|---|
| [docs/installation.md](docs/installation.md) | 环境部署：前置要求、安装步骤、依赖说明、安装后如何验证 |
| [docs/configuration.md](docs/configuration.md) | `config/config.yaml` 每一节怎么填：按键映射、真实游戏标定参数、奖励权重 |
| [docs/testing.md](docs/testing.md) | `tests/` 下每个测试脚本的作用、参数和使用示例 |
| [docs/training.md](docs/training.md) | 怎么训练、模型怎么保存、怎么继续训练、怎么用 `play.py` 跑推理、怎么看 TensorBoard |
| [docs/live_game.md](docs/live_game.md) | 对接真实游戏（阶段8）：现状说明、手动标定步骤、还差什么 |
| [docs/architecture.md](docs/architecture.md) | 整体架构：目录结构、模块职责、数据流、技术方案选型理由 |
| [docs/roadmap.md](docs/roadmap.md) | 分阶段开发计划与当前进度 |
| [docs/risks.md](docs/risks.md) | 已知风险与应对思路 |

## 目录结构

```
AI_Sekiro/
├── config/config.yaml         # 全局配置：按键映射、标定参数、奖励权重（见 docs/configuration.md）
├── sekiro_ai/
│   ├── state_reader/          # 状态读取（mock_reader.py / pixel_reader.py）
│   ├── controller/             # 动作 -> 键鼠映射与执行
│   ├── restart/                # 死亡/胜利检测 + 自动重开
│   ├── reward/                 # 状态差 -> reward
│   ├── env/                    # Gymnasium Env 封装
│   ├── logging/                # episode CSV 记录回调
│   └── utils/                  # 配置加载、日志setup
├── tests/                       # 各模块独立测试脚本（见 docs/testing.md）
├── train.py                     # PPO 训练入口（见 docs/training.md）
├── play.py                      # 加载模型跑推理（见 docs/training.md）
├── models/                      # 训练产出的模型 .zip
└── logs/                        # 运行日志 / tensorboard / episode CSV
```

## 常见问题

- **`--live` 一直降级成 mock**：检查游戏窗口是否已打开，以及 `config.yaml` 的 `calibration` 段落是否已按 [docs/live_game.md](docs/live_game.md) 的步骤填写完整；日志会明确列出缺失的字段名。
- **键鼠没有反应**：确认没有加 `--dry-run`（`test_controller.py`）或用的是mock模式（`train.py`/`play.py` 不加 `--live` 时默认 `dry_run=True`，只打印不发送真实输入）；同时确认游戏窗口处于前台聚焦状态，`pydirectinput` 发送的输入依赖当前激活窗口。
- **想同时保留多个训练版本对比效果**：每次训练换一个不同的 `--run-name`，模型/日志/CSV 都会按名字分开，不会互相覆盖，详见 [docs/training.md](docs/training.md)。
- **改了 `config.yaml` 没生效**：配置只在脚本启动时读取一次（并缓存），改完需要重新运行脚本，不会在运行中的进程里热更新。
