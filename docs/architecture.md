# Sekiro RL Agent — 架构设计

## 1. 项目目录结构

```
AI_Sekiro/
├── venv/
├── docs/
│   ├── architecture.md          # 本文档：目录结构 / 模块职责 / 数据流 / 技术方案
│   ├── roadmap.md               # 分阶段开发目标与验收标准
│   └── risks.md                 # 潜在困难与解决方案
├── config/
│   └── config.yaml              # 全局配置：按键映射、奖励权重、训练超参数
├── sekiro_ai/
│   ├── __init__.py
│   ├── state_reader/
│   │   ├── __init__.py
│   │   ├── base.py               # StateReader 抽象接口
│   │   ├── pixel_reader.py       # 真实画面读取实现（阶段8，截图+图像识别）
│   │   ├── mock_reader.py        # 模拟数据实现（阶段1起主用）
│   │   └── schema.py             # 统一状态字典的数据结构定义
│   ├── controller/
│   │   ├── __init__.py
│   │   ├── action_map.py         # 动作枚举 <-> 按键映射
│   │   └── input_controller.py   # 键鼠执行（pydirectinput）
│   ├── restart/
│   │   ├── __init__.py
│   │   └── restart_manager.py    # 死亡/胜利检测 + 自动重开流程
│   ├── reward/
│   │   ├── __init__.py
│   │   └── reward_calculator.py  # 状态差 -> reward
│   ├── env/
│   │   ├── __init__.py
│   │   └── sekiro_env.py         # Gymnasium Env 封装
│   ├── logging/
│   │   ├── __init__.py
│   │   └── episode_logger.py     # CSV/JSON 记录 + TensorBoard 回调
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py
├── tests/
│   ├── test_state_reader.py
│   ├── test_controller.py
│   ├── test_restart.py
│   ├── test_reward.py
│   └── test_env.py
├── train.py
├── play.py
├── models/          # PPO 保存的 .zip
├── logs/            # tensorboard + episode 记录
└── requirements.txt
```

设计原则：`sekiro_ai/` 下每个子包对应一个模块，彼此只通过明确定义的数据结构（`schema.py` 里的状态字典 / 动作枚举）交互，不直接互相 import 内部实现。测试脚本放在 `tests/`，可单独运行，且都不强制依赖真实游戏进程。

> **变更说明（真实数据源方案调整）**：原方案计划以内存读取（`memory_reader.py` + `pymem`）为首选真实数据源，画面像素识别作为降级方案。经调研已有开源实现（`docs/案例/RL-Sekiro-env`）后发现：
> 1. 只狼没有公开、稳定的内存偏移表可直接使用；该开源案例是通过**特征码扫描（pattern scan）+ 运行时代码注入（code injection，手写x86-64机器码打补丁）**才拿到角色/Boss结构体基址的，属于较深的逆向工程技术，且实现被硬编码死绑定在游戏 `v1.06` 版本，版本一变基本全部失效。
> 2. 该案例本身也尝试过用颜色识别血条（HSV阈值），但实测"生命值/耐力值识别不准确，会有小波动，偶尔有大波动"，因此才转向内存注入方案——但这恰恰说明**该案例作者当时的画面识别方案做得不够充分**（例如只做了粗糙的颜色阈值匹配，没有做区域校准、多帧平滑、动态范围归一化等），不代表画面识别本身不可行，只代表那一次实现不够好。
> 3. 综合评估：内存注入方案的开发门槛（逆向工具链、特征码维护、版本兼容性、代码注入对游戏进程的侵入性和风险）明显高于画面识别方案；而画面上的 HP 条、架势条、"危"字提示本来就是游戏主动渲染给玩家看的信息，读取它们不需要绕过或修改游戏本身，只要把颜色识别、区域校准做扎实（而不是像参考案例那样简单阈值一把梭），可靠性是可以做上去的。
>
> 因此调整为：**画面读取（截图 + 图像识别）作为首要且唯一规划的真实数据源，取消内存读取路线**，不再规划 `memory_reader.py`。详见下文技术方案第6节与 [risks.md](./risks.md)。

## 2. 统一状态格式

由 `state_reader/schema.py` 定义，贯穿全项目：

```
{
  "player_hp": float [0,1],
  "player_posture": float [0,1],
  "boss_hp": float [0,1],
  "boss_posture": float [0,1],
  "player_pos": (x, y, z),
  "boss_pos": (x, y, z),
  "distance": float,
  "boss_action": str,        # e.g. "idle","attack","perilous_attack","stagger"
  "player_hit": bool,
  "can_parry": bool,
  "player_dead": bool,
  "boss_dead": bool,
}
```

> **注意**：这个状态字典现在只供 `RewardCalculator` 和 `RestartManager` 使用（计算 reward、判断死亡/重开），**不再是 agent 的观测**。agent 的观测是堆叠的灰度像素帧（形状见 [configuration.md](configuration.md) 的"`observation` —— 像素帧观测形状"一节），由 `SekiroEnv`/`FrameStack` 独立于这份状态字典单独产出。

## 3. 模块间数据流

单个 step 循环：

```
StateReader.read() -> state(t)
        │
        ▼
RewardCalculator.compute(state(t-1), state(t)) -> reward
        │
SekiroEnv 组装 observation (state(t) 转成 np.array)
        │
        ▼
PPO Agent.predict(observation) -> action (int)
        │
        ▼
InputController.execute(action) -> 键鼠事件
        │
        ▼
RestartManager.check(state(t)) -> 若死亡/结束则触发重开, terminated=True
        │
        ▼
EpisodeLogger.log(step, action, state, reward)
```

## 4. 模块职责一览

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| State Reader | 获取当前游戏状态，统一格式 | 游戏窗口截图 或 mock 配置 | 状态 dict |
| Action Controller | 把离散动作映射为键鼠输入 | action:int | 无（副作用：键鼠事件），返回 bool 执行成功 |
| Restart Manager | 检测死亡/胜利，执行重开按键序列 | 状态 dict | bool: 是否已重开, 新状态 |
| Reward Calculator | 根据状态差计算 reward | state(t-1), state(t) | float reward |
| SekiroEnv | 组合以上模块，符合 Gym API | action | obs, reward, terminated, truncated, info |
| PPO Agent (train/play) | 用 SB3 训练/推理策略 | env | 保存模型 / 实时动作 |
| Episode Logger | 记录训练过程数据 | step级数据 | CSV + TensorBoard 事件 |

关键设计点：

- **StateReader 是抽象接口**（`base.py` 定义 `read() -> dict`），`MockStateReader` 和 `PixelStateReader` 都实现它。`SekiroEnv` 只依赖接口，初期用 mock，后期切换真实读取无需改环境代码。这是解耦真实游戏依赖的核心手段，让阶段1-6可以完全离线测试。
- **InputController** 同样支持 `dry_run` 模式（只打印将要执行的动作，不真正发送键鼠事件），方便在没有游戏窗口时测试映射逻辑。
- **RestartManager** 依赖 StateReader 提供的 `player_dead`/`boss_dead`，重开的按键序列本身也通过 InputController 执行，避免重复实现输入逻辑。

## 5. 动作空间定义

| Action ID | 含义 | 键鼠映射 |
|---|---|---|
| 0 | 等待 | 无操作 |
| 1 | 攻击 | 鼠标左键 |
| 2 | 防御 | 鼠标右键 |
| 3 | 弹刀 | 鼠标右键（时机敏感的短按） |
| 4 | 左闪 | Shift + A |
| 5 | 右闪 | Shift + D |
| 6 | 后退 | Shift + S（或 S） |

具体按键与游戏内设置对应，实际映射表放在 `config/config.yaml`，代码从配置读取而非硬编码，方便用户按自己的键位习惯调整。

## 6. 推荐技术方案

- **游戏状态读取**：初期用 `MockStateReader`（随机/脚本化状态序列）。真实读取阶段采用**画面读取（截图 + 图像识别）**作为唯一规划路线（不再规划内存读取，原因见上方"变更说明"与 [risks.md](./risks.md)），具体技术栈：
  - 窗口定位：`pygetwindow` 或 Win32 API（`FindWindow` + `GetClientRect`，注意要用 ClientRect 而非 WindowRect，否则标题栏/边框会导致截图整体偏移）定位只狼窗口的客户区坐标。
  - 截图：`mss`（性能优于 `PIL.ImageGrab`，适合高频率 step 循环）按窗口客户区坐标截图。
  - HP/架势条识别：OpenCV 在截图中裁出血条/架势条固定区域，用 HSV 颜色阈值检测高亮像素长度占比换算成百分比。区域坐标和颜色阈值需要针对当前分辨率/UI缩放**手工标定一次**（配套一个可视化标定脚本，截图后画出识别框方便肉眼核对）。
  - Boss危险动作识别：只狼会在Boss发起突刺/横扫/下段攻击时弹出醒目的"危"字提示，这是游戏主动渲染给玩家的强视觉信号，用模板匹配（`cv2.matchTemplate`）或简单OCR识别该图标出现，比试图从像素反推"boss_action"分类更可靠。
  - 已知局限：`player_pos`/`boss_pos`/精确 `distance` 这类数值在游戏UI上不显示，画面方案拿不到精确坐标，只能退化为粗略估计（如根据锁定框大小/角色画面占比间接判断远近）或直接从 observation 中去掉这些字段，改用相对距离的模糊分档。
  - 两种实现（mock / 画面读取）都实现同一 `StateReader` 接口，可在 config 里切换。
- **键鼠控制**：`pydirectinput`（比 pyautogui 更适合游戏，因为它模拟的是 DirectInput 扫描码，游戏引擎更容易识别）。闪避的 Shift+方向键组合用 `pydirectinput.keyDown/keyUp` 手动控制按下时长。
- **Gym 环境**：`gymnasium.Env`，`observation_space` 用 `Box` 表示堆叠的灰度像素帧（`(84,84,stack_size)`，uint8，具体形状见 [configuration.md](configuration.md)），`action_space` 用 `Discrete(7)`。
- **RL 算法**：`stable-baselines3` 的 `PPO`，`policy="CnnPolicy"`（观测是 `(84,84,stack_size)` 的 uint8 灰度图像堆叠，SB3 会自动为这个图像 `Box` 空间构建默认的 `NatureCNN` 特征提取器，不需要手动传 `policy_kwargs`）。
- **日志**：SB3 自带 `tensorboard_log` 参数直接接入 TensorBoard；额外自定义 `BaseCallback` 记录每个 episode 的死亡原因/击杀用时到 CSV。
- **配置管理**：用一个 `config.yaml` + 简单 dataclass 加载，统一管理按键映射、奖励权重、训练超参，避免硬编码分散在各文件。
