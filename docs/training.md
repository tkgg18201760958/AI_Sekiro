# 训练、保存模型、查看 TensorBoard

## 训练（train.py）

```powershell
# mock 环境快速跑通（几千步，几分钟内跑完，用于验证整条管线没问题）
python train.py --total-timesteps 1000

# 更长时间的正式训练，自定义运行名
python train.py --total-timesteps 200000 --run-name my_run

# 对接真实游戏（需要先完成 config.yaml 的 calibration 标定，见 live_game.md）
python train.py --live --total-timesteps 200000 --run-name live_run
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--total-timesteps` | 1000 | PPO 训练的总步数（不是episode数，是环境step的总次数） |
| `--run-name` | `ppo_sekiro` | 这次运行的名字，决定模型文件名、TensorBoard子目录名、episode CSV文件名——**换一个名字就不会覆盖之前的运行**，方便对比不同超参/权重的效果 |
| `--mode` | `scripted` | mock数据模式：`scripted`（固定剧本，稳定复现）或`random`（随机抖动，测鲁棒性），`--live`生效时忽略 |
| `--live` | 关闭 | 尝试连接真实游戏；游戏没打开或标定未完成会自动降级为mock，日志会打印具体原因 |
| `--seed` | 0 | 随机种子，保证同样参数下训练过程可复现 |
| `--max-episode-steps` | 2000 | 单个episode最多跑多少步，超过就算`truncated`（不是死亡，是"打太久了强制结束"） |
| `--resume-from` | 无 | 传入已有模型 `.zip` 路径，从这个模型的参数继续训练（不是从0开始） |

### 训练过程中发生了什么

`train.py` 内部会：

1. 构建环境（`build_env`），根据 `--live`/`--mode` 决定用 `PixelStateReader` 还是 `MockStateReader`。
2. 用 `stable_baselines3.common.monitor.Monitor` 包一层环境，专门用来在每个episode结束时收集这一整局的统计数据（步数、总reward），并通过 `info_keywords=("state",)` 把最后一步的完整状态字典也带出来（供CSV记录死亡原因用）。
3. 创建（或加载）PPO模型，`policy="MlpPolicy"`（状态是低维数值向量，不需要卷积网络处理图像）。
4. 调用 `model.learn(total_timesteps=..., callback=episode_logger)` 开始训练，`episode_logger` 是自定义的 `EpisodeCsvLogger`（见下方"episode记录"）。
5. 训练结束后（或被 Ctrl+C 中断）会走到 `finally` 块，**保证模型一定会被保存一次**。

### Ctrl+C 中断是安全的

训练时按 `Ctrl+C` 会触发 `KeyboardInterrupt`，但因为保存逻辑写在 `finally` 里，模型仍会被存到 `models/<run-name>.zip`——不用担心训练一半强行退出会白跑。

## 训练产出的文件

一次训练结束后会产出：

- **`models/<run-name>.zip`** —— 保存的 PPO 模型，可以用 `play.py` 加载它跑推理，或用 `--resume-from` 继续训练。
- **`logs/tensorboard/<run-name>_N/`** —— TensorBoard 事件文件。同名 `--run-name` 多次运行会自动在末尾加序号（`_1`、`_2`...），**不会覆盖前一次的曲线**，方便在TensorBoard里对比同名字不同尝试的效果。
- **`logs/episodes/<run-name>.csv`** —— 逐episode记录（见下方"episode记录"）。**这个文件是追加写入的**：同一个 `--run-name` 再跑一次，会接着往同一个CSV后面加行，不会清空重写。
- **`logs/train.log`** —— 完整运行日志（配置、开始/结束时间、模型保存路径等）。

## Episode 记录（logs/episodes/<run-name>.csv）

`sekiro_ai/logging/episode_logger.py` 里的 `EpisodeCsvLogger` 是一个 SB3 `BaseCallback`，每当 `Monitor` 检测到一个episode结束（`terminated`或`truncated`），就往CSV里追加一行。字段：

| 列名 | 含义 |
|---|---|
| `episode` | 本次运行内的episode序号（从1开始，重新跑不会清零，是CSV里的累计行号） |
| `timesteps` | 到这个episode结束时，模型已经训练了多少个环境step（累计值） |
| `wall_time` | 从这次`train.py`启动到这个episode结束经过了多少秒（真实时间，不是仿真时间） |
| `steps` | 这个episode本身跑了多少步 |
| `total_reward` | 这个episode的总reward |
| `end_reason` | 这个episode怎么结束的：`boss_dead`（打死boss）、`player_dead`（玩家死亡）、`truncated`（跑到`--max-episode-steps`上限，强制结束，非死亡） |
| `final_boss_hp` | episode最后一步的boss血量（0~1） |
| `final_player_hp` | episode最后一步的玩家血量（0~1） |

这个CSV **不是**给SB3自己用的（SB3自己的训练曲线走的是TensorBoard），而是为了拿到"死亡原因分布""平均生存步数""boss有没有真的被打死过"这类粒度的信息——TensorBoard默认曲线看不到这些。可以直接用Excel或者`pandas.read_csv(...)`打开分析，比如统计 `end_reason` 的分布看agent更容易死还是更容易磨死boss。

## 查看 TensorBoard

```powershell
tensorboard --logdir logs/tensorboard
```

命令跑起来后，浏览器打开命令行打印的地址（默认 `http://localhost:6006`）。里面能看到：

- `rollout/ep_rew_mean` —— 最近若干个episode的平均reward，训练是否在变好看这条最直接。
- `rollout/ep_len_mean` —— 平均episode长度（步数）。配合reward一起看：如果reward在涨但episode变短，可能是agent学会了更快打死boss；如果episode变长但reward没涨，可能是在拖时间。
- `train/loss`、`train/value_loss`、`train/policy_gradient_loss` 等PPO内部训练指标，用于判断训练本身是否稳定（比如loss剧烈震荡通常意味着学习率或其他超参需要调整）。

每个 `--run-name` 对应左侧一条独立曲线（TensorBoard会按目录名自动分组），可以勾选多条同时对比。

## 保存模型（补充说明）

保存不需要额外命令，是 `train.py` 自动完成的：训练正常跑完，或者被Ctrl+C中断，都会在退出前把模型存到 `models/<run-name>.zip`。

如果想在**训练过程中**多存几个中间检查点（而不是只在结束时存一次），当前实现没有内置这个功能——`train.py` 只在 `finally` 块存一次。要加的话需要自己在 `train.py` 里给 `model.learn()` 加上 SB3 自带的 `stable_baselines3.common.callbacks.CheckpointCallback`（比如每N步存一次到 `models/<run-name>_step_N.zip`），或者把 `EpisodeCsvLogger` 所在的 callback list 扩展成多个callback（SB3的 `callback` 参数支持传 `CallbackList`）。这不是当前代码自带的功能，需要你根据自己的训练时长/中断风险自行决定是否加。

### 继续训练已有模型

```powershell
python train.py --resume-from models/my_run.zip --run-name my_run_continued --total-timesteps 100000
```

注意 `--run-name` 建议换一个新名字（比如加个 `_continued` 后缀），这样新产出的CSV/TensorBoard曲线不会跟原来的运行混在一起，方便区分"从哪个checkpoint接着训练的"。

## 推理 / 观看 Agent 表现（play.py）

```powershell
# 用 mock 环境跑 3 个 episode（默认）
python play.py --model models/my_run.zip

# 跑 10 个 episode，用确定性策略（不随机采样，每次都选最高概率的动作）
python play.py --model models/my_run.zip --episodes 10 --deterministic

# 对接真实游戏
python play.py --model models/my_run.zip --live
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 必填 | 要加载的模型 `.zip` 路径 |
| `--episodes` | 3 | 跑多少个episode |
| `--mode` | `scripted` | 同`train.py`，mock数据模式，`--live`生效时忽略 |
| `--live` | 关闭 | 同`train.py` |
| `--deterministic` | 关闭 | 关闭时按策略输出的概率分布随机采样动作（保留一定探索性/随机性）；开启则每次都选概率最高的动作（更稳定、可复现，适合"展示效果"而不是"继续学习"） |
| `--max-episode-steps` | 2000 | 同`train.py` |

每个episode跑完会在终端和 `logs/play.log` 里打印：`episode=N steps=... total_reward=... terminated=... truncated=...`。`play.py` **不会**训练或更新模型，纯推理，也不产出CSV/TensorBoard数据——想看某个模型在实际对战中的具体表现，看这里的日志输出或者直接用 `--live` 打开游戏观察。

## 下一步

如果想对接真实游戏而不是一直用mock数据训练，看 [live_game.md](live_game.md)——里面有标定步骤和已知局限的说明。
