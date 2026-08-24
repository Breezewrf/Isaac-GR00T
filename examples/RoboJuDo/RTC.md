# RoboJuDo Inference-time RTC

本文档说明 RoboJuDo 中已经实现的 inference-time Real-Time Chunking（RTC）。实现参考
[LeRobot RTC](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py)，
使用现有 16-step RoboJuDo checkpoint，不需要重新训练。

当前实现范围：

- 支持 `examples/RoboJuDo/run_robojudo_client.py --execution-mode rtc`。
- 支持 X2 和 G1 23-DoF RoboJuDo profile。
- 当前 RTC batch size 固定为 1，并保持单个 inference in-flight。
- `double_buffer` 和 `temporal_ensemble` 的原有行为保持不变。
- N1.7 action head 中原有的 inpainting RTC 分支仍作为 legacy 路径保留；RoboJuDo 使用新的梯度 guidance 路径。

## 启动方式

先启动 Policy Server：

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/x2_move_box_center_test/ \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 5555
```

再启动 RTC client：

```bash
uv run python examples/RoboJuDo/run_robojudo_client.py \
  --profile x2 \
  --robot-endpoint tcp://127.0.0.1:8561 \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --command-endpoint tcp://*:8559 \
  --execution-mode rtc \
  --execution-horizon 8 \
  --rtc-prefix-schedule exp \
  --rtc-max-guidance-weight 10
```

RTC 参数：

- `--execution-horizon`：RTC guidance 结束位置 `H`，默认 8；模型仍生成并排队 checkpoint
  定义的完整 action chunk。客户端会从第一次 policy 返回动态读取 action horizon。
- `--rtc-prefix-schedule`：`zeros`、`ones`、`linear` 或 `exp`，默认 `exp`。
- `--rtc-max-guidance-weight`：denoising correction 最大增益，默认 10。
- `--rtc-latency-window`：估计延迟时保留的最近推理耗时数量，默认 10。
- `--command-fps`：控制频率，默认 30 Hz。

## 整体数据流

```text
RoboJuDo observation（图像、当前关节、任务）
        ↓
run_robojudo_client.py
  截取旧队列中尚未执行的物理动作
  估计推理延迟 D_est
        ↓
deploy_adapter.py
  构造 GR00T observation
  通过 PolicyClient options 发送 RTC prefix
        ↓
gr00t_policy.py
  物理 absolute prefix + 最新 state
  → relative/absolute 动作处理
  → normalization
        ↓
gr00t_n1d7.py
  在每个 flow denoising step 中执行 RTC guidance
        ↓
gr00t_policy.py
  模型动作 denormalize
  relative 双臂动作恢复成物理 absolute target
        ↓
deploy_adapter.py
  同时返回完整物理 action chunk 和 RoboJuDo commands
        ↓
run_robojudo_client.py
  根据实际延迟 D_actual 跳过新 chunk 前若干步
  原子替换执行队列
```

最重要的动作空间关系：

```text
客户端 RTC 队列
  保存物理 absolute action
        ↓ 结合最新 observation 重新处理
模型 RTC prefix
  使用 normalized model-space action

双臂：物理 absolute target → 相对最新关节状态转换 → normalize
导航：保持 absolute → normalize
高度：保持 absolute → normalize
```

客户端没有把 raw absolute action 直接送进 flow sampler。保存 absolute target 的目的是在下一轮推理时，
能根据机器人最新实测状态重新计算 relative action。

## 实际代码修改

### `examples/RoboJuDo/deploy_adapter.py`

该文件负责 RoboJuDo 与 GR00T Policy 之间的格式转换，不负责队列调度和 RTC 数学。

新增 `PolicyActionChunk`：

```python
@dataclass(frozen=True)
class PolicyActionChunk:
    actions: dict[str, np.ndarray]
    commands: list[dict[str, Any]]
    info: dict[str, Any]
```

一次推理结果同时保留两种表示：

```text
actions
  分组物理动作，shape 为 (1, 16, D)
  用于下一轮 RTC prefix

commands
  RoboJuDo 可直接执行的 positions + locomotion_command
  用于当前控制循环
```

`actions` 包含：

```python
{
    "left_arm": (1, 16, arm_dim),
    "right_arm": (1, 16, arm_dim),
    "navigate_command": (1, 16, 3),
    "base_height_command": (1, 16, 1),
}
```

主要接口：

- `_validate_action_chunk()` 验证四个动作组、batch、horizon、动作维度和有限值。
- `decode_action_chunk()` 将每一步转换成 RoboJuDo command。
- `get_action_chunk()` 将 RTC options 传给 `PolicyClient.get_action()`，并返回完整物理 chunk。
- 原来的 `get_action()` 保留，内部调用 `get_action_chunk()` 后只返回 commands，供其他 execution mode 使用。

RTC 调用 `get_action_chunk(execution_horizon=None)`，因此保存模型返回的完整 chunk，而不是只截取
`--execution-horizon` 步。chunk 长度来自 checkpoint 对应 embodiment 的 action
`delta_indices`，客户端不再将其固定为 16。

### `examples/RoboJuDo/run_robojudo_client.py`

该文件负责异步推理、RTC 队列、延迟估计、实际延迟切片和控制安全。

`ActionChunk` 新增：

```text
physical_actions
  完整分组物理动作，用于下一次 guidance

task
  防止跨任务复用 prefix

estimated_delay_steps
  本轮 sampler 使用的 D_est
```

新增 `RTCActionQueue`，内部只保存：

```python
self._chunk
self._next_index
```

`_next_index` 同时表示：

- 下一条要执行的 command。
- 下一条尚未执行的 physical action。

主要操作：

```text
pop()
  返回 commands[next_index]
  next_index += 1

get_left_over()
  返回 physical_actions[:, next_index:]
  同时校验 stream、control session 和 task

replace(new_chunk, D_actual)
  保存新 chunk
  next_index = D_actual
```

推理开始时，推理线程在同一把 condition lock 下完成：

```text
固定最新 observation
记录 query_tick
固定 stream / control session / task
截取尚未执行的 physical prefix
根据最近延迟计算 D_est
```

延迟估计公式：

```text
D_est = ceil(max(recent_inference_latency) / command_period)

等价于：
D_est = ceil(max(recent_inference_latency) × command_fps)
```

然后裁剪到：

```text
0 <= D_est <= min(prefix_length, execution_horizon)
```

发送给服务端的 RTC options：

```python
{
    "rtc": {
        "prefix_actions": {
            "left_arm": ...,
            "right_arm": ...,
            "navigate_command": ...,
            "base_height_command": ...,
        },
        "prefix_length": L,
        "estimated_delay_steps": D_est,
        "guidance_horizon": min(H, L),
        "prefix_schedule": "exp",
        "max_guidance_weight": 10.0,
    }
}
```

推理返回后重新读取 control tick：

```text
D_actual = ready_tick - query_tick
```

`D_est` 和 `D_actual` 的职责不同：

```text
D_est
  告诉模型哪些 prefix step 应该强约束

D_actual
  告诉客户端新 chunk 的哪些 step 已经过期
```

新队列从 `new_chunk[D_actual]` 开始执行。第一次推理没有旧 prefix，因此 `skipped_steps=0`，不会错误地
跳过第一批动作。

控制安全：

- observation timeout、takeover disabled、stream 改变、control session 改变或 task 改变时清空 RTC 队列。
- in-flight 结果若属于旧 session 或旧 task，会在进入队列前被丢弃。
- RTC 队列耗尽时保持双臂和高度，并将 `vx`、`vy`、`yaw_rate` 置零。
- 若 `D_actual >= 16`，整段新预测已经过期，丢弃该 chunk 并等待无 prefix 刷新。

### `gr00t/policy/gr00t_policy.py`

该文件负责把客户端传来的物理 absolute prefix 转换成模型使用的 normalized action tensor。

`_to_vla_step_data()` 现在可以接收 actions：

```python
VLAStepData(
    states=current_state,
    actions=physical_prefix,
    ...,
)
```

`_prepare_rtc_options()` 负责：

- 当前只允许 batch size 1。
- 校验 action groups、`float32`、shape `(1, T, D)` 和有限值。
- 校验 `0 <= D_est <= H <= action_horizon`。
- 从 model options 中移除大型 `prefix_actions` 数组。
- 把物理 prefix 放入 `VLAStepData.actions`，让现有 processor 处理。

RoboJuDo 的动作配置是：

```text
left_arm             RELATIVE
right_arm            RELATIVE
navigate_command     ABSOLUTE
base_height_command  ABSOLUTE
```

例如旧队列保存的双臂物理目标为 35°：

```text
最新实测关节为 30° → model prefix = +5°
最新实测关节为 28° → model prefix = +7°
```

物理目标仍然是 35°，但 relative 表示根据最新实测状态重新锚定。随后 processor 使用 checkpoint
statistics 归一化动作，并生成 action padding 和 action mask。

#### 短 prefix 与 16-step statistics

RoboJuDo checkpoint 的双臂 relative-action statistics 是按时间步保存的：

```text
left_arm relative statistics   (16, 7)
right_arm relative statistics  (16, 7)
```

RTC 运行一轮后，leftover prefix 通常短于 16。例如实际推理延迟为 3：

```text
原始 chunk    16 steps
跳过          3 steps
leftover      13 steps
```

如果直接把 `(13, 7)` 交给 `(16, 7)` statistics 归一化，会产生 shape mismatch。当前实现会在进入
processor 前重复最后一个物理目标，补齐到完整 horizon：

```text
真实 prefix： A3 A4 ... A15                 共 13 步
processor：   A3 A4 ... A15 A15 A15 A15    补齐到 16 步
```

真实 `prefix_length` 仍为 13，sampler 会将 13 步之后的 guidance 权重设为 0；padding 只用于满足
per-timestep normalization shape，不会扩大有效 RTC prefix。

普通推理继续使用：

```python
with torch.inference_mode():
```

RTC 推理使用：

```python
with torch.no_grad():
```

原因是 `torch.inference_mode()` 不能被内部 `torch.enable_grad()` 覆盖，而 RTC sampler 需要局部
autograd correction。backbone 和普通推理部分仍然不构建梯度。

模型输出后，`processor.decode_action()` 会：

```text
denormalize
双臂 relative action → 当前状态下的物理 absolute target
导航和高度保持 absolute
```

因此形成闭环：

```text
normalized model output
        ↓ decode
physical absolute queue
        ↓ 下一轮结合最新 state
normalized model prefix
```

### `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`

该文件实现每个 flow denoising step 中的 RTC correction。

`get_rtc_prefix_weights()` 构造时间权重。设：

```text
D_est = 3
H = 8
T = 16
```

则：

```text
step       0    1    2    3    4    5    6    7    8 ... 15
weight     1    1    1    ↓    ↓    ↓    ↓    ↓    0 ...  0
           └ 强约束 ┘    └ 平滑接管区域 ┘       └ 自由生成 ┘
```

不同 schedule：

- `zeros`：只有 `[0, D_est)` 为 1。
- `ones`：`[0, H)` 全部为 1。
- `linear`：`[D_est, H)` 线性衰减。
- `exp`：`[D_est, H)` 按 LeRobot exponential 形状衰减。

时间权重还会乘 processor 的 action mask，屏蔽无效 action dimension。`guidance_horizon` 会裁剪到真实
`prefix_length`，因此短 prefix 的 padding step 不参与 guidance。

GR00T flow 的时间方向是：

```text
t = 0  noise
t = 1  clean action
```

每个 denoising step 首先预测基础 velocity：

```text
v_t = model(x_t, observation, t)
```

当前 latent 的 clean-action estimate：

```text
x_clean = x_t + (1 - t) × v_t
```

计算加权 prefix error：

```text
error = (prefix - x_clean) × prefix_weights
```

使用局部 autograd 计算 correction：

```text
correction = grad(x_clean, x_t, grad_outputs=error)
```

与 LeRobot 当前实现一致，本次 denoiser 输出 `v_t` 在 correction 中视为固定值，不通过整个 denoiser
构建反向图。然后修正 velocity：

```text
v_guided = v_t + guidance_gain(t) × correction
```

最后继续正常 Euler integration：

```text
x_next = x_t + dt × v_guided
```

每一步结束后 detach latent，避免多个 denoising step 的计算图连接起来。

LeRobot reverse-time guidance gain 已转换到 GR00T 的正向 flow convention：

```text
gain(t) = ((1 - t)² + t²) / ((1 - t) × t)
gain(t) = min(gain(t), max_guidance_weight)
```

## 完整 RTC 时序可视化

下面假设：

- 控制频率 30 Hz。
- 每次推理耗时 3 tick。
- `D_est = 3`。
- `guidance_horizon H = 8`。
- 每个模型 chunk 实际仍为 16 步，为了可读性只画前几步。

```text
30 Hz tick              0       1       2       3       4       5       6       7       8       9

infer chunk A           [--------- inference -------->]
A prediction time                               A0      A1      A2      A3      A4      A5      A6
published action        -       -       -       A0      A1      A2

infer chunk B                                   [--------- inference -------->]
old prefix for B                                A0      A1      A2      A3      A4      A5      A6
B prediction time                               B0      B1      B2      B3      B4      B5      B6
RTC guidance                                    强      强      强      渐弱    渐弱    渐弱    渐弱
published action        -       -       -       A0      A1      A2      B3      B4      B5

infer chunk C                                                           [--------- inference -------->]
old prefix for C                                                        B3      B4      B5      B6
C prediction time                                                       C0      C1      C2      C3
RTC guidance                                                            强      强      强      渐弱
published action        -       -       -       A0      A1      A2      B3      B4      B5      C3
```

第一次推理 A 没有旧 prefix：

```text
tick 0～2    A 正在推理，没有可执行动作
tick 3      A 返回；因为没有 prefix，所以从 A0 开始执行
```

推理 B 在 tick 3 发起，快照中的旧 prefix 是：

```text
prefix_B = [A0, A1, A2, A3, ...]
```

B 推理期间控制线程继续执行：

```text
tick 3 → A0
tick 4 → A1
tick 5 → A2
```

B 在 tick 6 返回：

```text
D_actual = ready_tick - query_tick
         = 6 - 3
         = 3
```

因此 `B0`、`B1`、`B2` 对应的时间已经过去，新队列从 `B3` 开始：

```text
旧执行序列： A0 → A1 → A2
新执行序列：                B3 → B4 → B5
最终发布：   A0 → A1 → A2 → B3 → B4 → B5 → C3 ...
```

模型内部的 prefix 对齐：

```text
旧 prefix       A0      A1      A2      A3      A4      A5      A6      A7
                 │       │       │       │       │       │       │       │
                 ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
新 chunk        B0      B1      B2      B3      B4      B5      B6      B7
weight          1.0     1.0     1.0      ↓       ↓       ↓       ↓       ↓
                └── D_est=3 ──┘  └──── transition，直到 H=8 ────────────┘
```

RTC 与 Temporal Ensemble 的区别：

```text
Temporal Ensemble
  A 和 B 独立生成
  客户端按同一 control tick 做 avg(A, B)

RTC
  生成 B 的过程中已经使用 A 作为 prefix guidance
  客户端按 D_actual 跳过过期 step 后直接执行 B3
```

## 日志解读

示例：

```text
[inference] chunk ready: actions=16, latency=0.190s,
query_tick=0, ready_tick=5, skipped_steps=0,
rtc_prefix=0, rtc_estimated_delay=0

[inference] chunk ready: actions=16, latency=0.101s,
query_tick=5, ready_tick=8, skipped_steps=3,
rtc_prefix=16, rtc_estimated_delay=6
```

第一行：

- 首次推理没有 prefix，所以 `rtc_prefix=0`。
- 即使推理期间 control tick 增长，首次结果仍从 step 0 开始，因此 `skipped_steps=0`。

第二行：

- 推理开始时旧 chunk 尚有完整 16 步，所以 `rtc_prefix=16`。
- 首轮 warm-up latency 为 0.190 秒，在 30 Hz 下得到 `ceil(0.190 × 30)=6`，所以 `D_est=6`。
- 第二轮实际只经过 3 tick，所以客户端使用 `D_actual=3`，从新 chunk 的 step 3 开始执行。
- 下一次推理看到的 leftover 通常为 `16 - 3 = 13` 步；Policy 会为 normalization 补齐到 16，
  但日志和 sampler 中的真实 `prefix_length` 仍是 13。

## 当前限制与调参建议

- 16-step checkpoint 可以运行 RTC，但比 32-step chunk 留给 transition/free region 的空间更小。
- warm-up 推理通常比稳态慢，rolling maximum 可能让最初几轮 `D_est` 偏大，这是保守行为。
- `D_est` 偏大只会让更多 prefix step 被强约束；实际队列切片始终使用 `D_actual`。
- 如果动作过于依赖旧轨迹，可减小 `--execution-horizon` 或降低 `--rtc-max-guidance-weight`。
- 如果 chunk 边界仍明显，可增大 `--execution-horizon`，但不能超过客户端从 policy 首个 chunk
  动态读取到的 action horizon。
- 实机调参期间可随时切回 `--execution-mode double_buffer` 作为回滚模式。

建议重点观察：

- chunk 切换时双臂 position jump。
- 速度和加速度突变。
- `D_est` 与 `skipped_steps` 的长期差异。
- RTC queue 是否频繁耗尽并进入 safe hold。
- task、takeover、session 或 stream 切换时是否正确清空历史动作。

## 验证覆盖

当前测试覆盖：

- prefix weight 的 frozen、transition 和 free 区域。
- guidance gain 的边界截断。
- guidance weight 为 0 时与普通 sampling 等价。
- Policy 物理 prefix 传递、短 prefix 补齐和 model options 清理。
- RTC queue 按 `D_actual` 替换、leftover 与 commands 锁步。
- task/session 不匹配时拒绝复用 prefix。
- 推理线程发送 leftover prefix，并按实际 tick 跳过新 chunk。
- PolicyClient/PolicyServer 对包含 NumPy prefix 的 RTC options 往返传输。
- `double_buffer` 和 `temporal_ensemble` 相关回归行为。
