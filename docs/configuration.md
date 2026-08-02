# 配置文件说明（config/config.yaml）

所有可调参数集中在 `config/config.yaml`，按模块分成几个顶层节（`controller`/`calibration`/`reward`）。**每个字段都是可选的**——代码里都写了默认值，`config.yaml` 里没写的字段会自动回退到默认值，不会报错。这意味着你可以只改自己关心的那一小部分，不用把整个文件填满。

加载逻辑在 `sekiro_ai/utils/config_loader.py`：`load_config()` 读一次 YAML 文件、缓存结果（`functools.lru_cache`），后续调用不会重复读磁盘。**注意**：这也意味着如果你在一个还在运行的 Python 进程里改了 `config.yaml`，不会立即生效——需要重新启动脚本（`train.py`/`play.py`/`tests/*.py`）才会读到新内容。

## `controller.keymap` —— 动作到键鼠的映射

```yaml
controller:
  keymap:
    wait:
      type: none
    attack:
      type: mouse
      button: left
    guard:
      type: mouse
      button: right
    parry:
      type: mouse
      button: right
      duration: 0.08
    dodge_left:
      type: combo
      keys: [shift, a]
      duration: 0.15
    dodge_right:
      type: combo
      keys: [shift, d]
      duration: 0.15
    backstep:
      type: combo
      keys: [shift, s]
      duration: 0.15
```

七个动作（`Action` 枚举，`sekiro_ai/controller/action_map.py`）对应七个键：`wait`（等待/无操作）、`attack`（攻击）、`guard`（防御）、`parry`（弹刀）、`dodge_left`（左闪）、`dodge_right`（右闪）、`backstep`（后退）。**键名必须是这七个之一的小写**，写错名字（比如打成 `atack`）不会报错，只会被静默忽略、该动作继续用默认映射——所以改完键位建议跑一下 `tests/test_controller.py --dry-run` 确认输出的映射跟你改的一致。

每条映射是以下四种"形状"之一：

| type | 字段 | 含义 |
|---|---|---|
| `none` | 无 | 什么都不做（`wait`默认用这个） |
| `mouse` | `button`（`left`/`right`）, `duration`（可选，秒） | 点一下鼠标；给了 `duration` 就变成"按住这么久再松开"而不是单击 |
| `key` | `key`（键名，如 `e`）, `duration`（可选，秒） | 敲一下键；同上，给了 `duration` 就是按住再松开 |
| `combo` | `keys`（键名列表）, `duration`（秒，默认 0.1） | 同时按住列表里所有键，等 `duration` 秒后再依次松开（顺序与按下时相反） |

`key`/`keys` 里的按键名是 `pydirectinput` 认识的名字（字母、数字直接写，`shift`/`ctrl`/`alt`/`space` 等特殊键用其小写英文名）。如果你的游戏内按键设置跟默认不一样（比如把闪避键改成了别的组合键），改这里对应的 `keys` 列表就行，不需要改任何代码。

## `calibration` —— 真实游戏画面标定（阶段8，必须手动测量）

```yaml
calibration:
  resolution: null            # e.g. [1920, 1080]
  player_hp_bar: null         # [x, y, width, height]
  player_posture_bar: null
  boss_hp_bar: null
  boss_posture_bar: null
  perilous_icon_template: null   # path to a saved template image, e.g. assets/perilous_icon.png
  hp_color_hsv_range: null       # e.g. [[0, 120, 70], [10, 255, 255]]
  posture_color_hsv_range: [[15, 80, 80], [35, 255, 255]]   # e.g. 黄色箭头架势条填充色的 HSV 范围
```

这一节是**唯一一个无法凭代码或经验预填、必须在游戏实际运行时手动测量**的配置。原因很直接：血条/架势条在屏幕上的像素位置、颜色，只有启动游戏截个图量一量才知道，没有游戏窗口时纯靠猜是不可靠的。

在没有填这些字段之前，`--live` 模式会被自动降级为 mock（并在日志里打印出具体缺了哪些字段），不会直接崩溃报错。也就是说**这一节留空完全不影响 mock 训练**，只影响真正对接真实游戏（`--live`）时的可用性。

标定步骤（详细的手动操作指南见 [live_game.md](live_game.md)）：

1. `resolution`：以固定分辨率启动游戏，填 `[宽, 高]`，比如 `[1920, 1080]`。之后如果换了分辨率/窗口大小，这里连同下面几个像素坐标都要重新量。
2. `player_hp_bar` / `player_posture_bar` / `boss_hp_bar` / `boss_posture_bar`：四条血条/架势条相对游戏窗口客户区（不含标题栏/边框）的像素矩形，格式 `[x, y, width, height]`。
3. `hp_color_hsv_range`：血条在"满"和"空"状态下取色，得到一组 HSV 阈值范围 `[[H_min, S_min, V_min], [H_max, S_max, V_max]]`，供 OpenCV 做颜色掩膜识别血条填充比例。
4. `posture_color_hsv_range`：架势条（黄色箭头填充）的 HSV 阈值范围，取法跟 `hp_color_hsv_range` 一样，但要注意架势条的黄色要跟血条的红/橙色在 HSV 上区分开，避免同一个掩膜误把两条都识别进去。默认值 `[[15, 80, 80], [35, 255, 255]]` 是基于 `GAME_PIC/BOSS.png`/`GAME_PIC/WITHOUT_BOSS.png` 两张样例截图，用手动 `cv2.inRange` + 逐列填充比例检查估出来的——不是自动化测试验证的，因为规划阶段还没有真实游戏窗口可用；接入真实游戏时建议重新用你自己截的图核对一遍。
5. `perilous_icon_template`：截一张只狼"危"字警示图标的图，存成图片文件，把文件路径填进来（比如 `assets/perilous_icon.png`），供模板匹配识别 boss 危险动作。

`sekiro_ai/state_reader/pixel_reader.py` 里的 `PixelStateReader.missing_calibration()` 会检查 `player_hp_bar`/`player_posture_bar`/`boss_hp_bar`/`boss_posture_bar` 这四项是否都已填写（`hp_color_hsv_range`/`perilous_icon_template`/`resolution` 目前不参与这个检查，因为读取管线本身——mss截图+OpenCV颜色分析——还没实现，参见下方"当前实现状态"）。

**当前实现状态**：即使把 `calibration` 全填满，`PixelStateReader.read()` 目前仍会抛 `NotImplementedError`——因为截图+颜色识别的具体处理逻辑（mss 截图、按矩形裁剪、HSV 掩膜、算填充比例）还没有写。也就是说 `calibration` 这一节和 `missing_calibration()` 检查，是"阶段8对接真实游戏"这件事目前唯一已经做好的部分（把需要标定什么、缺了什么讲清楚），实际的图像处理代码还是空的。如果你要接入真实游戏，这部分需要你自己在 `pixel_reader.py` 的 `read()` 里实现。

## `observation` —— 像素帧观测形状

```yaml
observation:
  frame_size: [84, 84]
  frame_skip: 4
  stack_size: 4
```

`sekiro_ai/state_reader/observation_config.py` 里的 `ObservationConfig.from_config()` 读取这一节，供 `MockStateReader`、`PixelStateReader`、`SekiroEnv` 三者共用，让它们对"喂给 PPO agent 的观测长什么样"达成一致，不用各自硬编码。三个字段都可选，缺的字段回退到下面的默认值：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `frame_size` | `[84, 84]` | 堆叠前单帧 resize 的目标尺寸 `[width, height]` |
| `frame_skip` | `4` | 每隔 N 个环境步骤才采样一个新帧入栈，期间最近入栈的帧会重复 |
| `stack_size` | `4` | 滑动窗口中保留的帧数；最终 observation 形状为 `(frame_size[1], frame_size[0], stack_size)` |

调这两个参数会直接影响训练：

- `frame_skip` 越大，堆栈内几帧之间的时间跨度越宽，越容易从静态帧里看出"动作方向"（比如 boss 是在举刀还是收刀），但采样密度变低，可能错过极短的判定窗口（比如弹刀时机）；调小则相反——帧之间几乎没差异，agent 更难从像素堆栈里推断出速度/趋势信息。
- `stack_size` 越大，agent 能看到更长的历史，对连续动作的建模更准，但观测维度线性增加，会拖慢前向推理和训练速度，同时增大显存占用；调小则观测更省资源，但可能丢失判断"招式正在进行到哪一步"所需的上下文。

改完这两个参数后，`FrameStack`（`sekiro_ai/state_reader/frame_stack.py`）产出的堆栈形状会跟着变，训练前建议确认 CNN 特征提取器的输入形状与之匹配。

## `reward.weights` —— 奖励权重

```yaml
reward:
  weights:
    boss_hp_delta: 100.0
    player_hp_delta: 100.0
    boss_posture_delta: 20.0
    player_posture_delta: 20.0
    player_hit: -5.0
    boss_dead: 500.0
    player_dead: -500.0
    step: -0.01
```

`sekiro_ai/reward/reward_calculator.py` 里 `RewardCalculator.compute(prev_state, curr_state)` 用这些权重把两帧状态的差异换算成一个标量 reward。含义：

| 权重 | 含义 | 符号/方向 |
|---|---|---|
| `boss_hp_delta` | 每单位 boss 掉血（`boss_hp` 在 [0,1]）獎励多少 | 正数，boss掉血越多奖励越高 |
| `player_hp_delta` | 每单位玩家掉血惩罚多少 | 正数，但在公式里是被减去的（玩家掉血是负向） |
| `boss_posture_delta` | boss 架势条上升（被打出破绽）奖励多少 | 正数 |
| `player_posture_delta` | 玩家架势条上升惩罚多少 | 正数，同样是被减去的 |
| `player_hit` | 玩家这一步被打中的固定惩罚 | 负数（直接加到reward上） |
| `boss_dead` | boss 死亡瞬间（False→True那一步）的终局奖励 | 正数，一次性 |
| `player_dead` | 玩家死亡瞬间的终局惩罚 | 负数，一次性 |
| `step` | 每一步固定的小额惩罚，防止agent"摆烂拖时间" | 负数，每步都加 |

默认权重下，boss 和玩家等量掉血时净 reward 接近 0（`100.0`权重相同，互相抵消），只剩下 `step` 的微小负值——这是为了不让 agent 靠"和boss同归于尽换正reward"这种取巧策略刷分。改权重后建议先跑 `python tests/test_reward.py`（用法见 [testing.md](testing.md)）确认10组手工样例的符号预期依然成立，再拿去训练，避免改出符号反了的bug（比如不小心把 `player_hp_delta` 写成正数，会变成"玩家掉血越多奖励越高"）。

调参建议：先用默认值跑通训练观察 TensorBoard 曲线（见 [training.md](training.md)），如果发现 agent 学会了"贴脸对拼血量"而不躲闪，可以尝试调大 `player_hp_delta`（让掉血惩罚更重）或调大 `player_hit`（让被打中本身更疼）；如果发现 agent 在拖时间不进攻，加大 `step` 的绝对值（比如从 `-0.01` 改成 `-0.05`）。
