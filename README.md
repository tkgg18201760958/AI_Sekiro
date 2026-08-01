# Sekiro RL Agent

用强化学习（PPO）训练一个自动打只狼 Boss 的 Agent。整体架构、模块职责、数据流见 [docs/architecture.md](docs/architecture.md)；分阶段开发计划见 [docs/roadmap.md](docs/roadmap.md)；已知风险与应对见 [docs/risks.md](docs/risks.md)。

项目采用模块化设计：State Reader → Reward Calculator → Gymnasium Env → PPO Agent → Input Controller → Restart Manager → Episode Logger。阶段1-7 全部可以在**不启动只狼游戏**的情况下用 mock 组件跑通训练管线；只有真正对接真实游戏画面时才需要游戏运行。

## 1. 环境部署

### 前置要求

- Windows（`pydirectinput`/`pywin32` 依赖 Windows API，仅支持 Windows 上真实控制游戏；mock 模式理论上跨平台）
- Python 3.11+（已在 3.13 上验证）

### 安装步骤

```powershell
# 1. 创建并激活虚拟环境（项目已包含 venv/ 目录结构，也可以重新创建）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 里的依赖：

| 包 | 用途 |
|---|---|
| `mss` | 高性能屏幕截图（真实画面读取用） |
| `opencv-python` | 图像识别（血条/架势条颜色阈值、"危"图标模板匹配） |
| `pygetwindow` / `pywin32` | 定位游戏窗口、获取客户区坐标 |
| `numpy` | 状态向量/观测空间运算 |
| `PyYAML` | 读取 `config/config.yaml` |
| `pydirectinput` | 模拟键鼠输入（DirectInput 扫描码，游戏引擎更容易识别） |
| `gymnasium` | RL 环境标准接口 |
| `stable-baselines3` | PPO 算法实现 |
| `tensorboard` | 训练曲线可视化 |

安装完成后可直接运行 `tests/` 下任意脚本验证环境（全部用 mock 数据，不需要打开游戏）：

```powershell
python tests/test_state_reader.py
python tests/test_controller.py --dry-run
python tests/test_restart.py
python tests/test_reward.py
python tests/test_env.py
python tests/test_random_agent.py
```

## 2. 配置

所有可调参数集中在 [config/config.yaml](config/config.yaml)，按模块分节，缺失的键会回退到代码里的默认值：

- `controller.keymap`：动作名 → 键鼠映射，按自己的游戏内按键习惯调整。
- `calibration`：真实游戏画面识别所需的像素标定（血条/架势条区域、颜色阈值、"危"图标模板）。**这一节必须在游戏实际运行时手动测量填写**，无法凭代码猜出来，未填写时 `--live` 模式会自动降级为 mock 并在日志里报告具体缺失哪些键。标定步骤见该配置段落内的注释。
- `reward.weights`：各奖励项权重（Boss掉血、玩家掉血、击杀/死亡等），用 `tests/test_reward.py` 校验符号，训练后观察 TensorBoard 曲线再迭代调整。

## 3. 使用方式

### 3.1 训练（train.py）

```powershell
# mock 环境快速跑通（几千步，几分钟内跑完）
python train.py --total-timesteps 1000

# 更长时间的正式训练，自定义运行名（影响模型文件名/TensorBoard子目录/CSV文件名）
python train.py --total-timesteps 200000 --run-name my_run

# 对接真实游戏（需要先完成 config.yaml 的 calibration 标定，且游戏窗口已打开）
python train.py --live --total-timesteps 200000 --run-name live_run
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--total-timesteps` | 1000 | PPO 训练的总步数 |
| `--run-name` | `ppo_sekiro` | 模型文件名 / TensorBoard 子目录名 / episode CSV 文件名 |
| `--mode` | `scripted` | mock 数据模式：`scripted`（固定脚本状态序列）或 `random`（随机状态），`--live` 时忽略 |
| `--live` | 关闭 | 尝试使用真实 `PixelStateReader`，找不到游戏窗口或标定未完成时自动降级为 mock |
| `--seed` | 0 | 随机种子，保证可复现 |
| `--max-episode-steps` | 2000 | 单个 episode 的最大步数（超过则 truncated） |
| `--resume-from` | 无 | 传入已有模型 `.zip` 路径，从该模型继续训练 |

训练结束（或中途 Ctrl+C 中断，`finally` 块会保证保存）后会产出：

- `models/<run-name>.zip` — 保存的 PPO 模型
- `logs/tensorboard/<run-name>_1/` — TensorBoard 事件文件
- `logs/episodes/<run-name>.csv` — 逐 episode 记录（步数、总reward、死亡原因、终局HP等）
- `logs/train.log` — 运行日志

### 3.2 保存模型

模型保存是 `train.py` 自动完成的（无需额外命令），保存时机：

- 训练正常跑完 `--total-timesteps` 后
- 或训练被 Ctrl+C 中断时（`finally` 块保证一定会存盘一次）

保存路径固定为 `models/<run-name>.zip`。如果想在训练中途手动多存几个检查点，目前需要自己改 `train.py` 加 SB3 的 `CheckpointCallback`（当前实现只在训练结束时存一次）。

继续训练某个已保存模型：

```powershell
python train.py --resume-from models/my_run.zip --run-name my_run_continued --total-timesteps 100000
```

### 3.3 推理 / 观看 Agent 表现（play.py）

```powershell
# 用 mock 环境跑 3 个 episode（默认）
python play.py --model models/my_run.zip

# 跑 10 个 episode，用确定性策略（不采样，每次选最大概率动作）
python play.py --model models/my_run.zip --episodes 10 --deterministic

# 对接真实游戏
python play.py --model models/my_run.zip --live
```

参数与 `train.py` 的 `--mode`/`--live`/`--max-episode-steps` 含义一致，另外：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 必填 | 要加载的模型 `.zip` 路径 |
| `--episodes` | 3 | 跑多少个 episode |
| `--deterministic` | 关闭 | 关闭时用策略概率分布采样动作，开启则总是选最优动作 |

每个 episode 的步数/总reward/终止原因会打印到 `logs/play.log`。

### 3.4 查看 TensorBoard

```powershell
tensorboard --logdir logs/tensorboard
```

启动后浏览器打开命令行输出的地址（默认 `http://localhost:6006`），可以看到：

- `rollout/ep_rew_mean`、`rollout/ep_len_mean` 等 SB3 自带的训练曲线
- 每个 `--run-name` 对应一条独立曲线（子目录按 `<run-name>_<序号>` 命名，同名多次运行会自动编号递增，不会互相覆盖）

也可以直接打开 `logs/episodes/<run-name>.csv` 用 Excel/pandas 分析每个 episode 的死亡原因分布、平均生存步数等。

## 4. 对接真实游戏（阶段8）

在 `--live` 模式真正生效之前，必须完成以下手动步骤（`config.yaml` 的 `calibration` 段落里有更详细的注释）：

1. 用固定分辨率启动只狼（推荐 1920x1080 无边框窗口化）。
2. 截图 HUD，测量血条/架势条相对游戏窗口客户区的像素矩形：`player_hp_bar`、`player_posture_bar`、`boss_hp_bar`、`boss_posture_bar`，格式 `[x, y, width, height]`。
3. 采样血条在"满"和"空"状态下的颜色，得到一组 HSV 颜色阈值 `hp_color_hsv_range`，供 OpenCV 做颜色掩膜。
4. 截取一张"危"（危险攻击提示）图标的模板图，保存路径填入 `perilous_icon_template`，供 `cv2.matchTemplate` 使用。

未完成标定时，`--live` 会在日志里明确报告缺失哪些字段并自动降级为 mock，不会直接报错崩溃。这一部分无法在没有游戏窗口的情况下自动完成，需要人工测量填入 `config.yaml`。

## 5. 目录结构

```
AI_Sekiro/
├── config/config.yaml         # 全局配置：按键映射、标定参数、奖励权重
├── sekiro_ai/
│   ├── state_reader/          # 状态读取（mock_reader.py / pixel_reader.py）
│   ├── controller/             # 动作 -> 键鼠映射与执行
│   ├── restart/                # 死亡/胜利检测 + 自动重开
│   ├── reward/                 # 状态差 -> reward
│   ├── env/                    # Gymnasium Env 封装
│   ├── logging/                # episode CSV 记录回调
│   └── utils/                  # 配置加载、日志setup
├── tests/                       # 各模块独立测试脚本（全部支持 mock，不依赖游戏）
├── train.py                     # PPO 训练入口
├── play.py                      # 加载模型跑推理
├── models/                      # 训练产出的模型 .zip
└── logs/                        # 运行日志 / tensorboard / episode CSV
```

## 6. 常见问题

- **`--live` 一直降级成 mock**：检查游戏窗口是否已打开，以及 `config.yaml` 的 `calibration` 段落是否已按上文步骤填写完整；日志会明确列出缺失的字段名。
- **键鼠没有反应**：确认没有加 `--dry-run`（`test_controller.py`）或用的是 `--live`（`train.py`/`play.py` 默认 `dry_run=True`，只打印不发送真实输入）；同时确认游戏窗口处于前台聚焦状态，`pydirectinput` 发送的输入依赖当前激活窗口。
- **想同时保留多个训练版本对比效果**：每次训练换一个不同的 `--run-name`，模型/日志/CSV 都会按名字分开，不会互相覆盖。
