# 安装与环境部署

## 前置要求

- **操作系统**：Windows。真实游戏对接依赖 `pydirectinput`（DirectInput 键鼠模拟）和 `pywin32`/`pygetwindow`（定位游戏窗口），这些库只支持 Windows。mock 模式（不涉及真实键鼠/截图）理论上可以在其他系统上跑，但项目没有在非 Windows 上测试过。
- **Python**：3.11 及以上，已在 3.13 上验证过。
- 不需要提前安装 CUDA/GPU 相关依赖。观测现在是 `84x84` 灰度图像堆叠，`stable-baselines3` 的 `CnnPolicy` 会用它默认的 `NatureCNN` 提取特征，但图像尺寸小、episode 也不长，CPU 上跑训练依然完全够用，不需要额外配 GPU。

## 安装步骤

```powershell
# 1. 克隆项目
git clone https://github.com/tkgg18201760958/AI_Sekiro.git
cd AI_Sekiro

# 2. 创建并激活虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt
```

激活虚拟环境后，命令行提示符前面会出现 `(venv)`。以下所有命令都假定你已经激活了虚拟环境、且当前目录在项目根目录下。

## 依赖说明

`requirements.txt` 锁定了每个包的精确版本，安装时不会拉取更新的不兼容版本：

| 包 | 用途 | 什么时候会用到 |
|---|---|---|
| `mss` | 高性能屏幕截图 | 只有 `--live` 真实对战模式（阶段8）会用，mock 模式不需要 |
| `opencv-python` | 图像识别（颜色阈值、模板匹配） | 同上 |
| `pygetwindow` | 按标题/进程查找游戏窗口 | 同上 |
| `pywin32` | Win32 API（进程查询、窗口句柄） | 同上 |
| `numpy` | 状态向量、观测空间数值运算 | 全程都用，mock/真实模式通用 |
| `PyYAML` | 解析 `config/config.yaml` | 全程都用 |
| `pydirectinput` | 模拟键鼠输入（DirectInput 扫描码） | `--dry-run`/`dry_run=True` 时不会真正调用，但仍需装上（`InputController` 非 dry-run 时才会 `import pydirectinput`，见下方"依赖延迟加载"） |
| `gymnasium` | RL 环境标准接口（`Env`/`spaces`/`check_env`） | 全程都用 |
| `stable-baselines3` | PPO 算法实现、训练回调、`Monitor` | `train.py`/`play.py` 用 |
| `tensorboard` | 训练曲线可视化 | 查看训练曲线时用 |

**依赖延迟加载**：`PixelStateReader`（真实截图读取）和 `InputController(dry_run=False)`（真实键鼠输出）内部对 `mss`/`opencv-python`/`pygetwindow`/`pywin32`/`pydirectinput` 都是**用到时才 `import`**，不是模块顶层导入。这意味着即使这几个包因为某些原因没装成功，只要你一直用默认的 mock/dry-run 模式，也完全不受影响——这也是为什么阶段1-7的整条训练管线可以在没有真实游戏、甚至没装全依赖的机器上跑通。

## 验证安装

装完之后，不需要打开游戏就可以跑一遍全部测试脚本，确认环境搭好了（每个脚本的具体作用和参数见 [testing.md](testing.md)）：

```powershell
python tests/test_state_reader.py
python tests/test_controller.py --dry-run
python tests/test_restart.py
python tests/test_reward.py
python tests/test_env.py
python tests/test_random_agent.py
```

全部正常跑完（没有报错退出）说明环境没问题。接下来可以直接开始训练（见 [training.md](training.md)），或者先看 [configuration.md](configuration.md) 了解可调参数。
