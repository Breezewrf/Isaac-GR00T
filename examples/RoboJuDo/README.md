# RoboJuDo X2 and G1 23-DoF

This example adapts datasets produced by `robojudo_recorder` for GR00T N1.7. The recorder
writes LeRobot v3.0; GR00T currently trains from its LeRobot v2.1 layout plus
`meta/modality.json`.

The two profiles are intentionally separate:

| Profile | State | Action | Config |
| --- | --- | --- | --- |
| G1 23-DoF | 5 left-arm + 5 right-arm joints | 10 joint targets + `vx`, `vy`, yaw rate, height | `robojudo_g1_23dof_config.py` |
| X2 | 7 left-arm + 7 right-arm joints | 14 joint targets + `vx`, `vy`, yaw rate, height | `robojudo_x2_config.py` |

Arm targets are trained as actions relative to the measured joint state. Navigation and height
commands remain absolute. Both profiles use a 16-frame action horizon and the episode task text
as language input.

## Prepare G1 23-DoF data

Convert the recorder output in the conversion helper's isolated environment:

```bash
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id g1_23dof_upper_body \
  --root /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/record_data
```

The converter moves the original directory to `g1_23dof_upper_body_v3.0` and places the v2.1
dataset at the original path. Install the G1 modality file:

```bash
cp examples/RoboJuDo/g1_23dof_modality.json \
  /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/record_data/g1_23dof_upper_body/meta/modality.json
```

Fine-tune a separate G1 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/record_data/g1_23dof_upper_body \
  --modality-config-path examples/RoboJuDo/robojudo_g1_23dof_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/robojudo_g1_23dof_finetune
```

## Prepare X2 data

Convert the X2 recorder output:

```bash
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id x2_upper_body \
  --root /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/record_data
```

Install the X2 modality file:

```bash
cp examples/RoboJuDo/x2_modality.json \
  /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/record_data/x2_upper_body/meta/modality.json
```

Fine-tune a separate X2 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/record_data/x2_upper_body \
  --modality-config-path examples/RoboJuDo/robojudo_x2_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/robojudo_x2_finetune
```

## Policy interface

The trained policies expect observations grouped as follows:

```python
observation = {
    "video": {"ego_view": images},
    "state": {
        "left_arm": left_joint_positions,
        "right_arm": right_joint_positions,
    },
    "language": {"task": [[instruction]]},
}
```

They return four action groups: `left_arm`, `right_arm`, `navigate_command`, and
`base_height_command`. `navigate_command` is ordered as `[vx, vy, yaw_rate]`. The decoded arm
outputs are absolute joint targets because the policy converts the learned relative actions back
using the current state.

X2 and G1 have different state/action dimensions. Do not mix them in one `NEW_EMBODIMENT`
training run or use one robot's checkpoint for the other.

## Deploy

Start the policy server with the checkpoint produced by the matching training run. The processor
saved inside the checkpoint contains the custom `NEW_EMBODIMENT` modality configuration, so no
extra config path is needed at inference time:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/robojudo_g1_23dof_finetune/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555
```

Start the matching coupled arm and locomotion pipeline in RoboJuDo-Plus. The optional task
override is published with every observation:

```bash
cd /home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus
python scripts/run_pipeline.py \
  -c g1_23_gr00t_locomanipulation_default_real \
  --gr00t-task "pick up the red cup"
```

G1 uses a RealSense camera by default. X2 uses the configured ROS2 compressed image topic. In
both cases, `Gr00tZmqCtrl` publishes a msgpack/JPEG multipart observation stream on port 8561;
the deploy client does not open the robot camera itself.

On the deploy machine, run the subscriber/client after starting the policy server:

```bash
uv run python examples/RoboJuDo/run_robojudo_client.py \
  --profile g1_23dof \
  --robot-endpoint tcp://<robot-ip>:8561 \
  --policy-host <policy-server-ip> \
  --policy-port 5555 \
  --status-interval 5
```

Use `--profile x2` for X2. The client validates the profile and exact joint order before inference.
It double-buffers action chunks: one thread receives the latest RoboJuDo observation, one performs
policy inference, and the command loop keeps publishing at 30 Hz while the next chunk is prepared.

### Double-buffer execution

#### Logic

The deploy client has three concurrent parts:

1. The observation thread receives the camera image, measured upper-body joints, and task from
   RoboJuDo. It drains already queued ZeroMQ messages and keeps only the newest valid observation,
   so inference does not work through a backlog of old camera frames.
2. The inference thread sends the newest observation that has not already been inferred to the
   GR00T policy server. The returned commands are stored in a single `pending` chunk. While that
   slot is occupied, the thread does not request another chunk.
3. The command loop owns the `active` chunk and publishes one command from it every
   `1 / --command-fps` seconds. The default command rate is 30 Hz.

The buffer transition is:

```text
latest observation
       |
       v
GR00T inference ---> pending chunk
                         |
                         | active chunk is empty
                         v
                   active chunk ---> command[0], command[1], ...
                         |
                         | activating it frees the pending slot
                         v
              infer the newest observation while active executes
```

The same flow on a time axis looks like this (`A0` means the first command in chunk A):

```text
time / 30 Hz tick --->  0       1       2       3       4       5       6

observation thread     O0      O1      O2      O3      O4      O5      O6
latest observation     O0      O1      O2      O3      O4      O5      O6

inference thread       [------ infer chunk A ------] [------ infer chunk B ------]
pending slot           empty   empty   empty   A       empty   empty   B
                                           transfer A             wait for A
                                               |                  to finish
                                               v
active queue           empty   empty   empty   A0..A3  A1..A3  A2..A3  A3
published command      none    none    none    A0      A1      A2      A3
```

Inference B can run while A is active because transferring A clears the pending slot. If B becomes
ready before A finishes, B waits in the pending slot. The command loop still consumes all of A and
then switches directly to B; predictions from A and B are not blended.

At startup, publishing waits for takeover to be enabled, a fresh observation from that control
session, and its first inferred chunk. Once a pending chunk is activated, its commands are copied
into the active queue and the pending slot is cleared. This immediately permits inference of the
newest observation while the active commands continue to execute. If that inference finishes
early, its result waits in the pending slot until the active queue is exhausted.

A shared `threading.Condition` coordinates this handoff. The inference thread waits on the
condition until the pending slot is empty and a newer observation is available, then stores its
result in the pending slot while holding the condition lock. When the command loop moves that
pending chunk into its local active queue, it clears the pending slot and calls `notify_all()`,
waking the inference thread so it can start the next request. The active queue itself is accessed
only by the command loop, so it does not need separate locking.

RoboJuDo publishes `takeover_enabled`, `control_session`, and a process-unique `stream_id` with
every observation. Disabling takeover leaves the camera stream running but makes the deploy client
clear active, pending, and held commands. Each disabled-to-enabled transition increments
`control_session`; the client then starts inference from a fresh observation for that session. An
inference result from an older session is discarded even if it returns after the new session has
started. A changed `stream_id` provides the same reset boundary when the whole RoboJuDo pipeline is
restarted. Every command carries the same `stream_id` and `control_session`; RoboJuDo rejects a
command unless it belongs to the currently enabled session, so an old buffered command cannot be
applied during an enable transition.

Chunk replacement happens only at the boundary: the client executes every command in the active
chunk, then activates the pending chunk. It does not align or average overlapping predictions, so
the current implementation is double-buffered asynchronous inference, not Temporal Ensemble. A
pending chunk is also not replaced by a newer prediction while it is waiting.

If the active horizon finishes before another chunk is ready, the client keeps publishing the last
command. This includes its locomotion values, so a non-zero velocity command is held until a new
chunk arrives, the observation stream times out, or the robot-side watchdog stops it. If the
observation age exceeds `--observation-timeout`, the client clears both active and pending actions
and stops publishing until observations become fresh again.

### Port binding
The two robot-side ports have opposite directions:

```text
RoboJuDo PUB tcp://*:8561  -> deploy SUB    camera + measured upper joints + task
RoboJuDo SUB deploy:8559   <- deploy PUB    upper targets + velocity/height command
```

Configure RoboJuDo's command `endpoint` with the deploy machine IP when they run on different
hosts. First confirm that observations are fresh and the command subscriber is connected, then
enter `RL_DEFAULT` and enable upper-body takeover. Enabling creates a new control session; inference
and command publishing begin from its first fresh observation. The controller rejects incomplete,
replayed, non-finite, or wrong-session commands atomically and stops locomotion when the command
watchdog expires.

### Deployment health logs

The deploy client prints throttled health reports every `--status-interval` seconds. The default is
5 seconds; use a shorter interval while diagnosing a connection:

```bash
uv run python examples/RoboJuDo/run_robojudo_client.py \
  --profile x2 \
  --robot-endpoint tcp://127.0.0.1:8561 \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --command-endpoint tcp://127.0.0.1:8559 \
  --execution-horizon 16 \
  --status-interval 2
```

The reports cover each stage of the deployment loop:

```text
[observation] rate=29.8Hz, received=60, last_sequence=412, sequence_gaps=2
[command] subscriber connected: tcp://127.0.0.1:8559
[control] takeover enabled for session 0123abcd:1; waiting for a fresh chunk
[inference] chunk ready: observation_sequence=412, session=1, actions=16, latency=0.184s
[command] activated chunk: observation_sequence=412, session=1, actions=16, age=0.190s, inference=0.184s
[command] rate=30.0Hz, published=60, next_sequence=900, subscriber_connected=True, ...
```

`sequence_gaps` counts publisher sequence numbers skipped by the latest-frame subscriber. Some gaps
are expected because the subscriber intentionally drains queued observations before decoding the
newest frame; sustained growth together with a low observation rate indicates that transport or
JPEG decoding cannot keep up.

`subscriber_connected=True` is a ZeroMQ transport connection signal, not an application-level
acknowledgement that RoboJuDo applied a command. Command application and watchdog state remain
visible in the RoboJuDo process logs. If observations stop, the client reports their age, clears
pending actions, and stops publishing commands after `--observation-timeout`. If inference is late,
the client reports that the action horizon is exhausted and holds the last command until a fresh
chunk arrives.
