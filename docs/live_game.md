# 对接真实游戏（阶段8）

阶段1-7的整条训练管线（StateReader→RewardCalculator→SekiroEnv→PPO→InputController→RestartManager→EpisodeLogger）都可以用 `MockStateReader` 在不启动游戏的情况下跑通。这份文档讲的是怎么把 `MockStateReader` 换成读取真实游戏画面的 `PixelStateReader`，以及目前这一步实际做到了什么程度。

## 现状：还差什么

先说清楚现状，避免产生"装好依赖就能直接对战"的误解：

- `--live` 参数在 `train.py`/`play.py`/`tests/test_state_reader.py` 里都已经接好了：会先尝试用 `PixelStateReader` 找游戏窗口，找不到（或标定没做）就自动降级为 mock，日志里会打印清楚的原因。
- `PixelStateReader` 目前能做到：通过进程名（`sekiro.exe`）定位游戏窗口、判断游戏是否在运行（`is_available()`）、检查 `config.yaml` 里 `calibration` 该填的字段填了没有（`missing_calibration()`）。
- `PixelStateReader` **目前做不到**：真正的截图+颜色识别（`read()` 会抛 `NotImplementedError`，无论标定填了没填）。也就是说，即使你完整做完下面的标定步骤，`read()` 现在依然会报错，因为mss截图、OpenCV裁剪血条区域、HSV颜色掩膜换算血量百分比这部分代码本身还没写。

这不是bug，是这个项目目前的真实完成状态：**标定所需的"缺什么"已经讲清楚（`calibration`配置节+`missing_calibration()`检查），但"拿到标定值之后怎么处理"这部分图像处理代码需要你自己在 `sekiro_ai/state_reader/pixel_reader.py` 的 `read()` 方法里实现**。之所以做不完是因为这一步必须对着真实运行的游戏窗口反复调试颜色阈值/裁剪区域，没有游戏环境的情况下没法验证代码对不对，纯靠猜写出来的代码几乎肯定是错的（宁可留空不实现，也不留一份没验证过、看起来能跑但实际读数是错的代码）。

## 为什么选画面识别而不是内存读取

`docs/architecture.md` 里有更完整的调研说明，简单版本：只狼没有公开稳定的内存偏移表，已知的开源方案是靠特征码扫描+运行时代码注入拿到角色/Boss结构体地址，属于较深的逆向工程技术，且死绑定某个游戏版本，版本一更新基本失效。相比之下，血条/架势条/"危"字提示本来就是游戏主动渲染给玩家看的信息，截图读取不需要绕过或修改游戏本身，只要标定和图像处理做扎实，可靠性是可以做上去的——所以项目选择只规划画面识别这一条路，不做内存读取。

## 手动标定步骤

如果你要继续完成这部分工作，标定步骤如下（对应 `config/config.yaml` 里 `calibration` 段的注释，字段说明见 [configuration.md](configuration.md)）：

### 1. 固定分辨率启动游戏

推荐 1920x1080 无边框窗口化模式。**分辨率或窗口大小一变，下面所有像素坐标都要重新量**——所以先固定好分辨率再开始标定，中途不要改。

把实际用的分辨率填进 `config.yaml`：

```yaml
calibration:
  resolution: [1920, 1080]
```

### 2. 测量血条/架势条的像素矩形

截一张游戏HUD的图（游戏内直接按截图键，或用系统自带截图工具），用图片编辑器（画图、PS等）打开，量出以下四个区域相对**游戏窗口客户区**（不含标题栏和边框）左上角的坐标和宽高：

- `player_hp_bar`：玩家血条
- `player_posture_bar`：玩家架势条
- `boss_hp_bar`：Boss血条
- `boss_posture_bar`：Boss架势条

每个填成 `[x, y, width, height]`，比如：

```yaml
  player_hp_bar: [40, 980, 320, 12]
```

注意事项：
- 一定要用**客户区坐标**，不是整个窗口（含标题栏/边框）的坐标——`architecture.md` 里特别提到这点，用 `GetClientRect` 而非 `WindowRect`，否则截图会整体偏移。
- 矩形尽量只框住血条本身的填充区域，不要包含血条外框/装饰，否则颜色识别会被边框颜色干扰。

### 3. 采样颜色阈值

分别在血条"满"（刚接敌或刚复活）和"空"（快死或Boss濒死）状态下截图，用取色工具（很多截图软件自带，或用 `cv2`/PS的取色器）读出血条填充色的HSV值，取一个覆盖两种状态、又不会误识别背景/空血条颜色的范围：

```yaml
  hp_color_hsv_range: [[0, 120, 70], [10, 255, 255]]
```

格式是 `[[H_min, S_min, V_min], [H_max, S_max, V_max]]`（OpenCV的HSV范围，H是0-179，不是0-360）。

### 4. 截取"危"字图标模板

只狼在Boss发起突刺/横扫/下段攻击等危险动作时会弹出醒目的"危"字视觉提示。截一张这个图标的裁剪图（只要图标本身，不要背景），保存成图片文件（比如 `assets/perilous_icon.png`），路径填进配置：

```yaml
  perilous_icon_template: assets/perilous_icon.png
```

用于后续 `cv2.matchTemplate` 模板匹配识别这个图标是否出现——这比试图从像素反推"boss在做什么动作"分类要可靠得多，因为这是游戏主动渲染给玩家的强视觉信号。

## 标定完成之后还需要做什么

填完 `calibration` 只是让 `missing_calibration()` 检查通过，`PixelStateReader.read()` 目前仍会抛第二个 `NotImplementedError`（"标定值有了，但截图/颜色识别的处理逻辑还没写"）。要让 `--live` 真正跑起来，需要在 `pixel_reader.py` 里补上：

1. 用 `mss` 按 `player_hp_bar` 等矩形截取对应区域的画面。
2. 用 `cv2.inRange` 配合 `hp_color_hsv_range` 做颜色掩膜，统计高亮像素占矩形宽度的比例，换算成 0-1 的血量/架势值。
3. 用 `cv2.matchTemplate` 拿 `perilous_icon_template` 在截图里找"危"字图标，判断 `boss_action` 是否应该标记为 `perilous_attack`。
4. `player_pos`/`boss_pos`/精确 `distance` 这几个字段在真实画面方案下**拿不到精确数值**（游戏UI不显示坐标），需要退化处理——比如从 `GameState` 里去掉这些字段改用相对距离的模糊分档，或者干脆固定填0/忽略。这是画面识别方案本身的已知局限，不是标定没做好能解决的。

写完这部分之后，建议先写一个独立的可视化标定脚本（截图后把识别出的矩形画框显示出来，方便肉眼核对坐标对不对），再接入正式训练——参照 `docs/architecture.md` 第6节的建议。

## 验证真实对接效果

标定和识别代码都做完之后：

```powershell
python tests/test_state_reader.py --live --steps 20
```

观察打印出的状态是否跟游戏里实际的血量/架势/危险提示对得上。确认没问题后再用 `--live` 跑 `train.py`/`play.py`。真正对战训练时，**务必让游戏窗口保持在前台并获得焦点**——`InputController` 发送的键鼠事件是发给"当前焦点窗口"的，不会主动抢焦点或切换窗口。
