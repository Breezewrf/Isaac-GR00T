# 执行策略离线评估

使用 LeRobot dataset 的 observation 回放 GR00T 推理，将各策略最终输出的物理动作与同一 tick 的 `action.*` GT 比较。支持纯同步执行、RoboJuDo 的 `double_buffer`、`temporal_ensemble`、`rtc`，以及关闭 guidance 的 RTC 调度对照 `rtc_schedule_no_guidance`。

这衡量的是**示范一致性、指令平滑性和时间滞后**。observation 始终来自录制轨迹，输出不会改变后续 observation，因此不能据此推断闭环任务成功率或实际机器人的运动平稳性。较大的 GT 误差也可能来自其他合理动作解。

## 快速开始

在仓库根目录、已安装项目依赖的环境中执行。优先使用未参与训练的 episode。

```bash
source .venv/bin/activate
python -m evaluation.execution_strategy_eval \
  --dataset-path /path/to/lerobot_dataset \
  --model-path /path/to/checkpoint \
  --embodiment-tag new_embodiment \
  --episode-ids 0 1 2 \
  --latency-ticks 0 1 2 4 8 \
  --replay-clock execution_clock \
  --execution-horizon 8 \
  --rtc-guidance-horizon 8 \
  --repeats 3 \
  --output-dir evaluation/results/comparison
```

`--episode-ids` 指 `meta/episodes.jsonl` 中的 `episode_index`，不是 loader 的数组下标；省略时评估所有 episode。默认完整回放，`--steps 300` 可限制长度。输出目录必须尚不存在；省略 `--output-dir` 时自动生成时间戳目录。

连接已启动的 GR00T policy server 时，将 `--model-path` 替换为：

```bash
--host 127.0.0.1 --port 5555
```

远端模式需要在服务端设置 denoising steps。当前服务端 API 不支持设置随机种子，脚本会明确提示；`--repeats` 仍可用于独立随机采样，但不能声称不同策略使用了相同采样噪声。本地模式按 episode、repeat、query tick 设置 Python/NumPy/PyTorch seed，匹配相同请求 tick 的初始噪声；GPU 算子仍可能有非确定性。

先做小规模检查可使用 `--episode-ids 0 --steps 100 --latency-ticks 0 2 --no-plots`。确认动作表示、采样周期和 GT 曲线合理后再扩大范围。

## 时间轴和对照条件

模型 chunk 长度和 observation modalities 从 policy 读取。控制周期固定为 dataset `meta/info.json` 的 `fps`；如果指定 `--control-fps`，必须与其一致。若 episode 有 `timestamp` 列，会检查等间隔采样。脚本不隐式重采样、不自动修正 action/state 时间偏移。

`--replay-clock` 控制 dataset 进度：

| 模式 | Dataset tick | 适用场景 |
|---|---|---|
| `wall_clock`（默认） | 每个 wall/control tick 都推进，包括 unavailable 和 hold | 动态环境、移动目标、必须按现实时间响应的任务；与旧结果直接兼容 |
| `execution_clock` | 只有真正输出预测动作（`status=1`）时推进；unavailable/hold 时冻结 observation 和 GT 进度 | 静态操作任务中按示范进度公平比较；同时保留 wall time 的暂停代价 |

`execution_clock` 下两个时钟被分别保存。模型读取 `dataset_tick` 对应的 observation；延迟、请求到达、hold、差分平滑性和完成耗时使用 `wall_tick`。TE 的时间对齐以及 RTC 返回后的跳步使用推理期间实际推进的 dataset step 数，启动等待不会跳过 chunk 前缀。所有策略都遵循同一规则；异步策略正常连续输出时，dataset 仍每个 wall tick 推进一步。

action 的 `delta_indices` 必须为从 0 开始的连续步；observation 不允许包含未来帧。历史 observation 在 episode 起点用第一帧填充，避免负索引误读 episode 尾部。language 当前要求单个 key、`delta_indices=[0]`。

每个虚拟 tick 按以下顺序执行：

1. 检查任务切换，清空旧队列和 hold 状态。
2. 接收已到达的推理结果；double buffer 在当前队列空时激活待执行 chunk。
3. 若没有在途请求且调度允许，使用当前 observation 发起一次推理。
4. 接收零延迟结果，再输出本 tick 的 command。

始终最多一个在途请求，每 tick 最多发起一次请求。`--query-interval` 指两次请求的最小间隔，默认 1 tick。真实模型顺序调用，结果由虚拟时钟决定何时可用，无需 sleep。尚未返回的预测不能提前进入 ensemble 或 prefix。

Synchronous 的完整周期是 `推理 D tick → 执行 execution_horizon 个动作 → 再推理 D tick`。它在最后一个动作输出后的下一 tick 采集 observation 并开始推理；推理返回后从新 chunk 的第 0 项执行，不跳过前缀。延迟为 0 时不会插入 hold tick。固定延迟为 D、执行长度为 E 时，稳态预测动作占比约为 `E / (E + D)`。

| 模式 | 行为 | 参数 |
|---|---|---|
| `synchronous` | 当前 chunk 执行完后才发起下一次推理；等待期间保持位置、速度置零 | `--execution-horizon`，默认 min(8, 模型长度) |
| `double_buffer` | 执行当前 chunk 时异步预取，消费完后切换，最多一个待执行 chunk；耗尽后重复上个完整 command | `--execution-horizon`，默认 min(8, 模型长度) |
| `temporal_ensemble` | 复用部署 `ACTTemporalEnsembler`，按 query tick 对齐重叠预测 | `--ensemble-horizon` 默认完整 chunk；`--temporal-ensemble-coeff` 默认 0.01 |
| `rtc` | 复用 `RTCActionQueue`，剩余物理预测作为模型 guidance prefix | `--rtc-guidance-horizon` 默认 min(8, 模型长度)，schedule 默认 exp，weight 默认 10 |
| `rtc_schedule_no_guidance` | 与 RTC 相同的队列、启动和延迟跳步方式，调用模型时不传 guidance | 隔离 guidance 的影响 |

当前 TE 正系数偏重较早请求，0 表示等权平均。RTC guidance horizon 与 synchronous/double-buffer 执行长度、TE 有效长度是三个独立参数。RTC 和 rtc_schedule_no_guidance 均保留完整模型 chunk。

RTC 在**发起请求时有剩余 prefix**的情况下，结果返回后跳过实际延迟步数；首次启动及无 prefix 的恢复从 chunk 的第 0 项开始，忠实保留现有客户端语义。`rtc_schedule_no_guidance` 也保留此行为。TE 始终按 query tick 对齐，因此启动延迟下即使预测完美，不同策略也可能产生不同 GT 误差；这是执行语义造成的结果。

Synchronous 在推理等待期、TE/RTC/rtc_schedule_no_guidance 在队列耗尽时保持位置类动作，并将指定速度组置零。默认仅将存在的 `navigate_command` 识别为速度组，其他数据集请显式传 `--velocity-groups group1 group2`；传空列表可禁用速度归零。程序不会根据维数猜测动作语义。Double buffer 保留部署行为，耗尽时保持完整的上一条 command。

首次预测返回之前不虚构 GT 或初始 command：NPZ 中对应输出为 NaN，状态为 unavailable。task 切换清空队列，旧任务在途结果会被丢弃。每个 episode/repeat/mode 都调用 `policy.reset()`。回放假设 observation 每 tick 正常更新、控制始终启用，不模拟网络断开、takeover 或 observation timeout。

## 延迟设置

三种设置互斥：

- `--latency-ticks 0 1 2 4 8`：分别运行五个固定延迟场景。TE、RTC、rtc_schedule_no_guidance 共享连续异步请求机会；synchronous 和 double buffer 按各自执行状态调度。所有方法在相同 query tick 上读取同一延迟环境。
- `--latency-trace trace.json`：文件是非负整数列表，例如 `[2, 2, 3, 2, 8, 1]`。按 **query tick % trace 长度**取值，表示随控制时间变化的延迟环境，所有方法使用同一份 trace；不是按请求编号取值。
- `--measured-latency`：每次请求使用 `ceil(get_action 耗时 × fps)`。本地包含采样 seed 设置、预处理、模型推理和动作解码；远端还包含 RPC 往返。dataset 解码、报告绘图不计入。RTC 延迟估计仅读取已返回请求，使用最近 N 次耗时的最大值。

固定延迟/trace 适合控制变量；实测延迟适合衡量各方法自身计算开销下的表现。实测第一轮包含可能的冷启动开销，不能当作稳态硬件 benchmark；需要先预热模型后再测，或用实测记录另行构造受控 trace。测量运行的结果不缓存复用。

## 指标

所有主指标保持原始时间对齐，使用 policy 已解码的**物理动作**与 dataset `action.*` 比较，不能将内部归一化/相对动作直接混入。评估前需要确认 dataset action 本身就是目标物理指令。各 action group 分别计算，不混合弧度、米、速度等单位，也不按本次测试轨迹重新归一化。

| 指标 | 解释 |
|---|---|
| `mae`, `rmse`, `p95_abs_error` | 对应时间 GT 误差 |
| `d1_rmse` 至 `d3_rmse` | 一至三阶时间差分相对 GT 的误差 |
| `dN_rms`, `dN_p95` | 输出差分的 RMS、绝对值 P95 |
| `gt_dN_rms`, `gt_dN_p95` | GT 在相同有效区间的参考值 |
| `boundary_jump_rms/p95` | chunk 激活/TE 新有效 chunk 加入时，输出相对上一 tick 的跳变，单位为 action units |
| `boundary_delta_rmse` | 同一边界处输出变化与 GT 变化的差异 |
| `lag_ticks`, `lag_seconds` | 一阶差分相关性最高的有符号时移，正数表示预测滞后 GT |
| `lag_correlation` | 上述时移的相关性，辅助判断 lag 是否可信 |
| `__execution__` | 无输出、hold、预测占比，首个 command 时间，过期 chunk 比例和推理延迟 |

`dN = diff(action, n=N) × fps**N`。位置指令的 d1/d2/d3 对应速度/加速度/jerk；速度指令的 d1/d2 才对应加速度/jerk。平滑性是**指令**的属性，不是执行器实测运动。三阶差分对采样噪声敏感，本版不做滤波，也不自动处理角度 wrap 或离散 gripper 通道；按数据语义解释这些组的指标。

lag 在 `±--max-lag-ticks` 内搜索，各候选时移使用相同有效时间支持；静止、恒速或数据不足返回 null。lag 不修正主指标的时间轴，也不等同于精确的抓取/启停事件延迟。

每个 run 输出两套指标：

- `available`：wall-time 序列中本策略有输出的全部有效 tick，包含重复的 hold tick；用于观察实际暂停、恢复和控制平滑性。
- `common`：`wall_clock` 下是所有选定策略都有输出的共同 wall tick；`execution_clock` 下是每个 dataset tick 恰好一次的真正执行动作序列，去掉 unavailable/hold 后比较任务进度上的动作准确性。

`--warmup-ticks` 额外排除起始若干 tick 的准确性/平滑性统计。执行可用性指标始终统计完整回放。所有差分要求整个差分窗口有效且属于同一任务，绝不跨缺失帧或任务边界。若共同有效区间为空，误差指标是 null，仍报告无输出比例。

## 输出

```text
manifest.json                         参数、完整延迟 trace、policy seed 支持和评估范围
metrics.json                          所有 episode/scenario/repeat/mode 指标
summary.csv                           episode 均值、95% CI、相对 double_buffer 的配对差值
tradeoff_*.png                        每个动作组的 RMSE vs d2/d3 RMS
episode_000000/
  layout.json                         动作组顺序/宽度和速度组
  gt.npz                              GT action 和时间轴
  delay_2/repeat_000/
    common_mask.npy                   wall_clock 才生成；所有策略共同可用的原始 mask
    rtc.npz                           最终 action、status、边界、contributors、segments、dataset_ticks、原始 chunks
    rtc.events.json                   query/ready/activation tick、prefix 长度、延迟、过期事件
    rtc.metrics.json                  available/common 指标
    trajectory_*.png                   各维度 GT/四策略动作、d1、d2，标注更新与 hold
```

NPZ 的 `status`：0 无输出，1 预测，2 hold。`dataset_ticks` 给出每个 wall tick 对齐的数据帧；execution clock 的等待区间会出现重复值。`chunk.<group>` 的形状为 `(requests, model_horizon, group_dim)`；`query_ticks` 与 requests 对齐，事件 JSON 另存 `query_dataset_tick`。不保存完整 observation 或图片副本。

summary 使用 `common` 指标，先在每个 episode 内平均 repeats，再对 episode 等权平均；配对差值为 `method - double_buffer`。95% CI 通过 2000 次 episode bootstrap 得到，少于两个有效 episode 时留空，不把相邻帧当独立样本。只比较 TE/RTC 时可以用 `--modes temporal_ensemble rtc`，此时 baseline 差值列为空。边界指标基于各策略自身更新时刻，不保证具有相同采样位置，应结合整体差分指标解读。

每个完整 repeat 后增量更新汇总；中途错误会保留已经写出的轨迹和结果。`--no-plots` 可以减少大量 episode/关节时的磁盘和绘图开销。

## 测试

```bash
python -m pytest evaluation/tests/test_execution_strategy_eval.py tests/test_robojudo_client.py -q --timeout=300
ruff check evaluation
ruff format --check evaluation
```

若 shell 加载了 ROS 的 pytest 插件但项目环境缺少 ROS 依赖，可只加载本次需要的 timeout 插件：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_timeout \
  evaluation/tests/test_execution_strategy_eval.py tests/test_robojudo_client.py -q --timeout=300
```

测试使用可控预测器验证时间对齐、RTC prefix、耗尽恢复、任务切换、差分窗口、滞后、episode bootstrap，并覆盖完整报告生成。真实模型评估仍需提供 checkpoint/运行中的 policy server 和 dataset。
