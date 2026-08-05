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

Use `RoboJuDoPolicyAdapter` on the robot-side client:

```python
from gr00t.policy.server_client import PolicyClient
from examples.RoboJuDo.deploy_adapter import RoboJuDoPolicyAdapter
import time
import zmq

client = PolicyClient(host="<policy-server-ip>", port=5555)
adapter = RoboJuDoPolicyAdapter(client, profile="g1_23dof")  # or profile="x2"

commands = adapter.get_action(
    image=head_rgb,
    joint_positions=upper_body_joint_positions,
    instruction="pick up the red cup",
    execution_horizon=8,
)

context = zmq.Context.instance()
upper_body = context.socket(zmq.PUB)
upper_body.bind("tcp://*:8559")
time.sleep(0.5)  # allow RoboJuDo's subscriber to connect
for command in commands:
    upper_body.send_json({"positions": command["positions"]})
    time.sleep(1.0 / 30.0)
```

Each command contains:

```python
{
    "positions": {"left_shoulder_pitch_joint": 0.1, "...": 0.0},
    "locomotion_command": np.array([vx, vy, yaw_rate, height], dtype=np.float32),
}
```

`positions` is directly compatible with RoboJuDo's existing `UpperBodyZmqCtrl` JSON protocol at
`tcp://127.0.0.1:8559`. The current RoboJuDo pipeline does not expose a ZMQ input for locomotion
commands; route `locomotion_command` into its locomotion policy command source before enabling
autonomous base motion. Until that control-side input exists, keep base control on the joystick
and publish only `positions`.
