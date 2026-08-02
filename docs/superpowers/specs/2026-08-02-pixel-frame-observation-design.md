# 像素帧观测（Pixel Frame Observation）设计

## 背景与目标

当前训练管线（`StateReader → RewardCalculator → SekiroEnv → PPO`）全部跑在 `MockStateReader` 提供的标量状态向量上（13 维 `Box`，`MlpPolicy`）。真实游戏对接（阶段 8）目前只完成了窗口检测和校准检查，`PixelStateReader.read()` 遇到截图+颜色识别这部分始终 `NotImplementedError`。

本次改造目标：让 agent 的 observation **完全替换**为直接从游戏画面截取、灰度化、缩放到 84×84 并做多帧堆叠的图像张量，替代现有的标量向量 observation。数值型状态提取（HP/架势/死亡判定）保留，但只服务于 reward 计算和重启判断，不再进入 observation。

## 范围边界

**包含：**
- `PixelStateReader` 新增截图能力：抓取整个游戏窗口客户区画面。
- 新增图像预处理：灰度化 + resize 84×84（OpenCV）。
- 新增帧堆叠模块：按可配置的 `frame_skip`（默认 4）间隔采样，堆叠 `stack_size`（默认 4）帧,形成 `(84, 84, 4)` 的 observation。
- `PixelStateReader` 补完 `read()` 中数值状态提取部分（bar-rect + HSV 颜色掩膜 → HP/架势百分比），这部分是 reward/restart 依赖的已有设计，之前只是未写实现。
- `MockStateReader` 新增图像模式：生成假的 `(84, 84, 4)` 图像堆栈（用几何图形模拟血条变化），供无游戏环境下跑通训练管线。
- `SekiroEnv` 改造：`observation_space` 从 `Box(shape=(13,))` 改为 `Box(shape=(84,84,4), dtype=np.uint8)`；`step()`/`reset()` 同时消费 `GameState`（reward/restart用）和图像堆栈（observation用）。
- `train.py`/`play.py` 的 policy 从 `MlpPolicy` 切换为 `CnnPolicy`。
- 相关测试脚本更新以覆盖新 observation 形状和帧堆叠逻辑。

**不包含（明确排除）：**
- "危"字图标模板匹配（`perilous_icon_template`）——超出本次改造范围，boss_action 的 perilous_attack 识别留待后续单独任务，本次只要求 HP/架势数值提取可用。
- `player_pos`/`boss_pos`/精确 `distance` 的像素反推——docs/live_game.md 已经指出这在纯画面方案下拿不到精确值，本次保留 `GameState` 里这些字段为默认值（不影响 reward，reward 不使用这些字段）。
- 训练可视化 UI——用户已确认为独立后续项目，测试全部通过后另开一轮 brainstorming。
- 实际真人对战联调（需要真实游戏窗口验证颜色阈值），本次只做到代码实现 + mock 图像模式下的自动化测试;真实窗口下的手动验证步骤会写进 docs，但不作为本次任务的完成判据。

## 架构

```
真实模式（--live）:
  PixelStateReader.read_frame()
    -> mss 截取窗口客户区原始画面（一次截图）
    -> 数值提取分支：按 calibration 的 4 个 bar rect 裁剪 + cv2.inRange HSV 掩膜 -> GameState(player_hp, boss_hp, player_posture, boss_posture, ...)
    -> 图像提取分支：整帧 cv2.cvtColor 转灰度 + cv2.resize 到 84x84 -> uint8 ndarray (84,84)
  FrameStack（新模块）：按 frame_skip 间隔采样,滑动窗口保留最近 stack_size 帧 -> (84,84,stack_size) ndarray

Mock 模式:
  MockStateReader 保持现有 GameState 数值轨迹（scripted/random）不变，
  新增 render_frame() 方法：把当前 GameState 的 hp/posture 数值绘制成 84x84 简易几何图形（色块长度模拟血条填充比例），
  绘制位置/尺寸从 config.yaml 的 calibration.player_hp_bar / boss_hp_bar / player_posture_bar / boss_posture_bar
  四个 rect 按 calibration.resolution 缩放映射到 84x84 画布上得到（而非随意摆放），
  这样 mock 图像里血条的视觉位置和真实截图的血条位置逻辑一致，方便后续对照真实模式调试。
  经同一个 FrameStack 逻辑堆叠。

SekiroEnv:
  __init__ 时创建一个 FrameStack 实例。
  reset()/step() 调用 reader.read() 拿 GameState（reward/restart用，不变）,
  再调用 reader.read_frame() 拿单帧灰度图,喂给 FrameStack,得到 (84,84,stack_size) observation。
  observation_space = Box(low=0, high=255, shape=(84,84,stack_size), dtype=np.uint8)
```

关键设计点：
- **`StateReader` 接口扩展**：新增抽象方法 `read_frame() -> np.ndarray`（返回单帧 84×84 灰度图,dtype uint8）。`MockStateReader`和`PixelStateReader`都必须实现。这保持了`SekiroEnv`不关心数据来源的既有设计原则（base.py 的开篇注释）。
- **一次截图两用**：`PixelStateReader` 内部把 mss 截图和数值提取、图像缩放共享同一次截屏结果，避免每步截两次屏幕拖慢帧率。`read()`（数值）和`read_frame()`（图像）在真实实现中会共享一个"截一次、算两次"的内部缓存,通过每步开始时调用一次`_capture()`来实现,`read()`和`read_frame()`都读这个缓存。
- **FrameStack 是独立、可单测的模块**（`sekiro_ai/state_reader/frame_stack.py`），不依赖 StateReader，只接收单帧 ndarray,内部维护 deque 和帧计数器,决定何时因为 frame_skip 而重复上一帧还是采集新帧。
- **配置新增**：`config.yaml` 新增 `observation` 顶层字段：
  ```yaml
  observation:
    frame_size: [84, 84]
    frame_skip: 4
    stack_size: 4
  ```
  `utils/config_loader.py` 的既有 lru_cache 模式复用,不需要改动。

## 数据流（每一步）

```
step(action):
  controller.execute(action)
  frame = reader.read_frame()       # 单帧 84x84 灰度图（原始或mock生成）
  obs = self._frame_stack.push(frame)  # (84,84,stack_size)
  state = reader.read()             # GameState，用于 reward/restart（可能复用同一次截图缓存）
  reward = reward_calculator.compute(prev_state, state)
  terminated = RestartManager.needs_restart(state)
  ...
  return obs, reward, terminated, truncated, info
```

`reset()` 时 `FrameStack` 清空并用首帧填满整个堆栈（标准 Atari 处理方式，避免开局出现全零帧）。

## 错误处理

- `PixelStateReader.read_frame()` 在窗口不可用时抛 `RuntimeError`（与现有 `read()` 一致的错误处理风格）。
- 校准缺失时，`read()`（数值分支）仍抛 `NotImplementedError` 提示缺哪些字段；但 `read_frame()`（图像分支）不依赖 bar-rect 校准，只需要窗口存在即可工作——所以即使 HP 校准没填,图像分支也能跑（这样图像 observation 可以先行验证，数值提取可以后补）。
- `perilous_icon_template` 未提供时,`boss_action` 相关字段维持默认值（"idle"），不报错，reward 计算不受影响（reward 不区分 boss_action 种类）。

## 测试策略

- `tests/test_state_reader.py` 新增图像模式断言：mock 模式下 `read_frame()` 返回 `(84,84)` uint8 ndarray。
- 新增 `tests/test_frame_stack.py`：单测 `FrameStack` 的 push/reset 行为（frame_skip 间隔、stack_size 滑窗、reset 首帧填满）。
- `tests/test_env.py` 更新 `check_env` 断言：`observation_space.shape == (84,84,stack_size)`,dtype uint8。
- 现有 `tests/test_reward.py`/`tests/test_restart.py` 不受影响（它们只测 `GameState`，不测 observation）。
- `train.py --total-timesteps 1000` 的 mock 冒烟测试需要能跑通并用 `CnnPolicy`。
- **可视化验证**：`tests/test_state_reader.py` 新增 `--save-frames N` 参数：把最近 N 次 `read_frame()`/`render_frame()` 的输出用 `cv2.imwrite` 保存成 PNG 到 `logs/frame_preview/frame_XXX.png`（每次运行先清空该目录），运行结束打印保存路径，方便用户直接打开图片肉眼检查血条位置和灰度效果是否合理。

## 待用户审阅的关键假设

1. `read_frame()` 作为 `StateReader` 接口的新增抽象方法（而不是可选方法）——`MockStateReader` 和 `PixelStateReader` 都必须实现，这是接口的破坏性变更（任何未来新增的 reader 实现都必须提供这个方法）。
2. Mock 模式下的图像生成使用简单几何图形（矩形色块长度模拟血条百分比）而非纯随机噪声，代价是多写一点绘图代码,好处是训练冒烟测试时能看到"reward 变化"和"画面变化"有粗略对应关系,便于调试。
