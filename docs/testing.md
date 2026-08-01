# 测试脚本使用指南

`tests/` 下每个脚本单独验证一个模块，全部基于 mock 数据，**不需要打开游戏**就能跑（除非你主动加 `--live` 或 `--send-real`/去掉 `--dry-run`）。按阶段顺序介绍，方便理解模块之间的依赖关系；日常验证环境装得对不对，直接照 [installation.md](installation.md) 里"验证安装"那一节挨个跑一遍就行。

每个脚本都会把详细输出写到 `logs/<模块名>.log`（比如 `test_reward.py` 写到 `logs/reward.log`），终端上也会同步打印，看不清可以去对应日志文件里翻。

## test_state_reader.py —— 状态读取

验证"能不能拿到游戏状态"这一步，不涉及键鼠、reward、环境逻辑。

```powershell
python tests/test_state_reader.py                     # mock，scripted 模式（默认）
python tests/test_state_reader.py --mode random        # mock，random 模式
python tests/test_state_reader.py --live               # 先尝试真实读取，找不到游戏窗口自动降级mock
python tests/test_state_reader.py --steps 20 --interval 0.2
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--live` | 关闭 | 尝试用 `PixelStateReader` 先找游戏窗口；找不到（或标定没做）会自动降级到 mock，并在日志里打印警告，不会报错崩溃 |
| `--mode` | `scripted` | mock 数据模式：`scripted`（固定剧本）或 `random`（随机抖动），`--live` 生效时忽略这个参数 |
| `--steps` | 40 | 调用几次 `read()` |
| `--interval` | 0.15 | 每次 `read()` 之间间隔多少秒 |
| `--seed` | 无 | 随机种子，给了就能复现同样的 mock 输出序列 |

跑起来后，终端每一行会打印当前状态：血量/架势条（玩家/boss）、距离、boss动作、是否被打中/可弹反、是否死亡。用 `scripted` 模式能看到一套完整的"接近→交锋→boss被打出破绽→boss死亡"的剧本；`random` 模式则是各数值在合法范围内随机抖动，主要用来测试极端/异常状态组合会不会让下游代码崩溃。

## test_controller.py —— 键鼠控制器

验证"动作编号 → 实际键鼠输入"的映射对不对。**这是唯一一个默认会真的发送键鼠事件的测试**，用之前一定看清楚参数。

```powershell
python tests/test_controller.py --dry-run              # 只打印，不真的发送，安全（推荐先跑这个）
python tests/test_controller.py --dry-run --action 1   # 只打印ATTACK这一个动作的映射
python tests/test_controller.py --action 1              # 真的发送ATTACK对应的键鼠事件
python tests/test_controller.py --delay 3               # 每个动作发送前倒计时3秒，留时间切到目标窗口
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--action` | 无（遍历全部） | 只测试单个动作，传 0-6 的整数（对应 `Action` 枚举：0=等待,1=攻击,2=防御,3=弹刀,4=左闪,5=右闪,6=后退） |
| `--dry-run` | 关闭 | 加上就只打印会发生什么，不发送真实输入。**没有游戏窗口/不想意外触发键鼠事件时务必加这个** |
| `--delay` | 2.0 | 真实发送（非dry-run）模式下，每个动作执行前等待几秒——用来倒计时切到目标窗口（记事本或游戏窗口），因为键鼠事件是发给"当前焦点窗口"的，脚本本身不会（也不会尝试）抢焦点 |

不加 `--dry-run` 时，脚本会真的调用 `pydirectinput` 发送键鼠事件——如果这时候焦点在记事本上，你会看到字符被打出来/鼠标点击；如果焦点在游戏窗口上，游戏里的角色就会做出对应动作。**没有明确要验证真实键鼠效果时，一律加 `--dry-run`**。

## test_restart.py —— 死亡检测与自动重开

模拟"死了/boss死了"这个状态转换，验证重开流程是否按预期执行。

```powershell
python tests/test_restart.py                    # 模拟玩家死亡（默认），dry-run方式发送重开按键
python tests/test_restart.py --scenario boss     # 换成模拟boss死亡
python tests/test_restart.py --send-real         # 真的发送重开按键（需要游戏窗口获得焦点）
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--scenario` | `player` | 模拟哪种死亡：`player`（玩家死亡）或 `boss`（boss死亡） |
| `--dead-reads` | 3 | 模拟"死亡状态"要持续几次 `read()` 调用才清除（模拟"你死了"提示+读盘的耗时） |
| `--send-real` | 关闭 | 加上才会真的发送重开按键序列（`e`键确认对话框、等待复活状态清除）；不加则只是dry-run打印日志 |
| `--poll-interval` | 0.1 | 等待某个重开步骤条件满足时，轮询间隔多少秒 |

脚本里用的不是 `MockStateReader`，而是一个专门写的 `ScriptedDeathReader`（只在这个测试文件里定义），逻辑更简单直接：连续 N 次报告"死亡=True"，之后变成"死亡=False"。跑完会打印"PASS/FAIL"，判断依据是重开后的新状态里 `player_dead`/`boss_dead` 是否都变回了 `False`（代表确实回到了一局新的开始）。

## test_reward.py —— 奖励计算

用10组手工构造的"改动前状态→改动后状态"样例对，检查 reward 计算出来的符号/范围是否符合预期。**不需要任何参数**，改完 `config.yaml` 的 `reward.weights` 后适合先跑这个再去训练。

```powershell
python tests/test_reward.py
```

10组样例覆盖的场景：boss单独掉血（应为正）、玩家单独掉血（应为负）、双方等量掉血互相抵消（应接近0）、玩家被击中额外惩罚、boss/玩家架势条上升（正/负）、boss死亡瞬间大额正奖励、玩家死亡瞬间大额负奖励、无任何变化只扣步数成本、"已经死亡→仍然死亡"不重复触发终局奖励。每条样例跑完会打印 `[PASS]`或`[FAIL]`，任何一条 `FAIL` 都值得注意——通常意味着权重改动带来了非预期的符号问题。

## test_env.py —— Gymnasium 环境

验证 `SekiroEnv` 是否符合 Gymnasium 的标准接口，以及能否正常跑完 `reset()`/`step()` 循环。

```powershell
python tests/test_env.py                        # 跑 gymnasium 官方 check_env 校验 + 手动跑几个episode
python tests/test_env.py --skip-check-env        # 跳过check_env，只跑手动episode（更快）
python tests/test_env.py --episodes 3 --steps 50
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--skip-check-env` | 关闭 | 跳过 `gymnasium.utils.env_checker.check_env` 的严格校验，只跑手动episode循环 |
| `--episodes` | 2 | 手动跑几个episode |
| `--steps` | 30 | 每个episode最多跑几步 |
| `--seed` | 0 | 随机种子 |

`check_env` 会报一个已知的、可以忽略的警告：关于 `info` 字典里 `timestamp` 字段在两次相同seed的reset之间不完全一致——这是因为 `timestamp` 记录的是真实墙钟时间（`time.time()`），故意如此设计（用于判断状态是否"过期"），跟 obs/reward/terminated 的确定性（真正重要的部分）无关，脚本会自动识别这种情况并仍然判定为通过。

## test_random_agent.py —— 随机动作全流程冒烟测试

不涉及新模块，纯粹是拿"随机选动作"的策略把 StateReader→RewardCalculator→SekiroEnv→InputController→RestartManager 整条链路跑很多遍，专门用来抓崩溃，不关心reward高不高。

```powershell
python tests/test_random_agent.py                       # 20个episode，scripted和random模式各跑一遍（默认）
python tests/test_random_agent.py --episodes 50 --mode random
python tests/test_random_agent.py --max-steps 500
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--episodes` | 20 | 每种模式跑多少个episode |
| `--max-steps` | 200 | 每个episode最多跑多少步（同时也是环境的截断上限） |
| `--mode` | `both` | 跑哪种/哪些mock模式：`scripted`、`random`，或 `both`（两种都跑） |
| `--seed` | 0 | 随机种子 |

跑完会汇总打印每种模式下：完成的episode数、正常终止(`terminated`)数、被截断(`truncated`)数、崩溃(`crashed`)数、平均reward、平均步数。最后一行如果是 `PASS: no crashes across all modes/episodes.` 说明整条管线在mock环境下是稳的；如果有 `crashed`，日志里会有完整的异常堆栈，需要照着定位问题。

## 下一步

测试脚本全部跑通之后，就可以开始正式训练了，见 [training.md](training.md)。如果要对接真实游戏，看 [live_game.md](live_game.md)。
