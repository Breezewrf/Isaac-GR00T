from collections import deque
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


ROBOJUDO_EXAMPLE = Path(__file__).parents[1] / "examples" / "RoboJuDo"
sys.path.insert(0, str(ROBOJUDO_EXAMPLE))

import deploy_adapter as adapter_module  # noqa: E402
import run_robojudo_client as client_module  # noqa: E402


def _observation(*, session: int, enabled: bool, sequence: int):
    return client_module.Observation(
        stream_id="test-stream",
        control_session=session,
        takeover_enabled=enabled,
        sequence=sequence,
        images={"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
        joint_positions={},
        task="test task",
    )


def _command(joint_name: str, value: float):
    return {
        "positions": {joint_name: value},
        "locomotion_command": np.full(4, value, dtype=np.float32),
    }


def _chunk(*, start_tick: int, values: list[float], session: int = 1):
    return client_module.ActionChunk(
        stream_id="test-stream",
        control_session=session,
        observation_sequence=start_tick,
        observation_received_at=0.0,
        inference_seconds=0.1,
        start_tick=start_tick,
        commands=[_command("joint", value) for value in values],
    )


def _rtc_chunk(*, values: list[float], task: str = "test task"):
    actions = np.asarray(values, dtype=np.float32)[None, :, None]
    return client_module.ActionChunk(
        stream_id="test-stream",
        control_session=1,
        observation_sequence=1,
        observation_received_at=0.0,
        inference_seconds=0.1,
        start_tick=0,
        commands=[_command("joint", value) for value in values],
        physical_actions={"joint": actions},
        task=task,
    )


def test_g1_adapter_splits_and_decodes_dexterous_hand_joints():
    profile = adapter_module.PROFILES["g1_23dof"]
    adapter = adapter_module.RoboJuDoPolicyAdapter(object(), "g1_23dof")
    positions = np.arange(30, dtype=np.float32)

    observation = adapter.build_observation(
        np.zeros((4, 5, 3), dtype=np.uint8),
        positions,
        "pick up the bag",
    )

    assert tuple(observation["state"]) == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    np.testing.assert_array_equal(observation["state"]["left_arm"], positions[0:5][None, None])
    np.testing.assert_array_equal(observation["state"]["right_arm"], positions[5:10][None, None])
    np.testing.assert_array_equal(observation["state"]["left_hand"], positions[10:20][None, None])
    np.testing.assert_array_equal(observation["state"]["right_hand"], positions[20:30][None, None])

    action_chunk = {
        "left_arm": positions[0:5][None, None],
        "right_arm": positions[5:10][None, None],
        "left_hand": positions[10:20][None, None],
        "right_hand": positions[20:30][None, None],
        "navigate_command": np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float32),
        "base_height_command": np.asarray([[[0.75]]], dtype=np.float32),
    }
    command = adapter.decode_action_chunk(action_chunk, execution_horizon=1)[0]

    assert tuple(command["positions"]) == profile.joint_names
    np.testing.assert_array_equal(list(command["positions"].values()), positions)
    np.testing.assert_allclose(command["locomotion_command"], [0.1, 0.2, 0.3, 0.75])


def test_g1_mulcam_adapter_builds_all_video_modalities():
    video_keys = adapter_module.CAMERA_LAYOUTS["mulcam"]
    adapter = adapter_module.RoboJuDoPolicyAdapter(object(), "g1_23dof", video_keys)
    images = {
        key: np.full((4, 5, 3), index, dtype=np.uint8) for index, key in enumerate(video_keys)
    }

    observation = adapter.build_observation(
        images,
        np.arange(30, dtype=np.float32),
        "pick up the bag",
    )

    assert tuple(observation["video"]) == video_keys
    for key in video_keys:
        assert observation["video"][key].shape == (1, 1, 4, 5, 3)
        np.testing.assert_array_equal(observation["video"][key][0, 0], images[key])


def _encoded_observation_parts(protocol_version: int, image_keys: tuple[str, ...]):
    profile = "x2"
    joint_names = adapter_module.PROFILES[profile].joint_names
    images = {
        key: np.full((6, 8, 3), index * 20, dtype=np.uint8) for index, key in enumerate(image_keys)
    }
    header = {
        "protocol_version": protocol_version,
        "profile": profile,
        "joint_names": list(joint_names),
        "joint_positions": [0.0] * len(joint_names),
        "task": "test task",
        "stream_id": "test-stream",
        "control_session": 1,
        "takeover_enabled": True,
        "sequence": 7,
    }
    if protocol_version == 1:
        header["shape"] = list(images["ego_view"].shape)
    else:
        header["image_keys"] = list(image_keys)
        header["image_shapes"] = {key: list(image.shape) for key, image in images.items()}
    parts = [client_module.msgpack.packb(header, use_bin_type=True)]
    for key in image_keys:
        ok, jpeg = client_module.cv2.imencode(".jpg", images[key])
        assert ok
        parts.append(jpeg.tobytes())
    return profile, parts


def test_observation_subscriber_decodes_protocol_v1_single_camera():
    profile, parts = _encoded_observation_parts(1, ("ego_view",))
    subscriber = client_module.ObservationSubscriber.__new__(client_module.ObservationSubscriber)
    subscriber.profile = profile
    subscriber.expected_joint_names = adapter_module.PROFILES[profile].joint_names
    subscriber.expected_image_keys = adapter_module.CAMERA_LAYOUTS["single"]

    observation = subscriber._decode_observation(parts)

    assert tuple(observation.images) == ("ego_view",)
    assert observation.images["ego_view"].shape == (6, 8, 3)


def test_observation_subscriber_decodes_protocol_v2_mulcam():
    image_keys = adapter_module.CAMERA_LAYOUTS["mulcam"]
    profile, parts = _encoded_observation_parts(2, image_keys)
    subscriber = client_module.ObservationSubscriber.__new__(client_module.ObservationSubscriber)
    subscriber.profile = profile
    subscriber.expected_joint_names = adapter_module.PROFILES[profile].joint_names
    subscriber.expected_image_keys = image_keys

    observation = subscriber._decode_observation(parts)

    assert tuple(observation.images) == image_keys
    assert all(image.shape == (6, 8, 3) for image in observation.images.values())


def test_observation_subscriber_rejects_missing_mulcam_part():
    image_keys = adapter_module.CAMERA_LAYOUTS["mulcam"]
    profile, parts = _encoded_observation_parts(2, image_keys)
    subscriber = client_module.ObservationSubscriber.__new__(client_module.ObservationSubscriber)
    subscriber.profile = profile
    subscriber.expected_joint_names = adapter_module.PROFILES[profile].joint_names
    subscriber.expected_image_keys = image_keys

    with pytest.raises(ValueError, match="has 3 parts, expected 4"):
        subscriber._decode_observation(parts[:-1])


def test_x2_adapter_reserves_hands_without_requiring_them():
    profile = adapter_module.PROFILES["x2"]
    assert profile.left_hand_joint_names == ()
    assert profile.right_hand_joint_names == ()
    assert tuple(key for key, _ in profile.joint_groups) == ("left_arm", "right_arm")

    adapter = adapter_module.RoboJuDoPolicyAdapter(object(), "x2")
    observation = adapter.build_observation(
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.arange(14, dtype=np.float32),
        "test x2",
    )
    assert tuple(observation["state"]) == ("left_arm", "right_arm")


def test_rtc_queue_replaces_using_actual_delay_and_keeps_prefix_in_lockstep():
    queue = client_module.RTCActionQueue()
    chunk = _rtc_chunk(values=[10.0, 11.0, 12.0, 13.0])

    assert queue.replace(chunk, skipped_steps=2)
    assert queue.qsize() == 2
    left_over = queue.get_left_over(("test-stream", 1), "test task")
    np.testing.assert_allclose(left_over["joint"], [[[12.0], [13.0]]])
    assert queue.pop()["positions"]["joint"] == 12.0
    np.testing.assert_allclose(
        queue.get_left_over(("test-stream", 1), "test task")["joint"],
        [[[13.0]]],
    )


def test_rtc_queue_rejects_cross_task_prefix_and_expires_old_chunk():
    queue = client_module.RTCActionQueue()
    chunk = _rtc_chunk(values=[1.0, 2.0])
    assert queue.replace(chunk, skipped_steps=0)
    assert queue.get_left_over(("test-stream", 1), "different task") is None

    assert not queue.replace(chunk, skipped_steps=2)
    assert queue.qsize() == 0
    assert queue.pop() is None


def test_parse_args_accepts_horizon_above_previous_fixed_limit():
    argv = [
        "run_robojudo_client.py",
        "--profile",
        "x2",
        "--robot-endpoint",
        "tcp://127.0.0.1:8561",
        "--execution-horizon",
        "32",
    ]
    with patch.object(sys, "argv", argv):
        assert client_module.parse_args().execution_horizon == 32


def test_parse_args_accepts_sync_execution_mode():
    argv = [
        "run_robojudo_client.py",
        "--profile",
        "x2",
        "--robot-endpoint",
        "tcp://127.0.0.1:8561",
        "--execution-mode",
        "sync",
    ]
    with patch.object(sys, "argv", argv):
        assert client_module.parse_args().execution_mode == "sync"


def test_parse_args_rejects_non_positive_horizon():
    argv = [
        "run_robojudo_client.py",
        "--profile",
        "x2",
        "--robot-endpoint",
        "tcp://127.0.0.1:8561",
        "--execution-horizon",
        "0",
    ]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        client_module.parse_args()


def test_parse_args_accepts_g1_mulcam_layout():
    argv = [
        "run_robojudo_client.py",
        "--profile",
        "g1_23dof",
        "--camera-layout",
        "mulcam",
        "--robot-endpoint",
        "tcp://127.0.0.1:8561",
    ]
    with patch.object(sys, "argv", argv):
        assert client_module.parse_args().camera_layout == "mulcam"


def test_temporal_ensemble_equal_average_uses_aligned_chunk_diagonal():
    ensembler = client_module.ACTTemporalEnsembler(("joint",), temporal_ensemble_coeff=0.0)
    ensembler.add_chunk(_chunk(start_tick=0, values=[0.0, 1.0, 2.0, 3.0]))
    ensembler.add_chunk(_chunk(start_tick=1, values=[10.0, 11.0, 12.0, 13.0]))

    action, contributors = ensembler.get_action(current_tick=2)

    assert contributors == 2
    assert action["positions"]["joint"] == 6.5
    np.testing.assert_allclose(action["locomotion_command"], np.full(4, 6.5))


def test_temporal_ensemble_matches_act_exponential_weighting_for_sparse_queries():
    coeff = 0.01
    ensembler = client_module.ACTTemporalEnsembler(("joint",), coeff)
    ensembler.add_chunk(_chunk(start_tick=0, values=[0.0, 1.0, 2.0, 3.0]))
    ensembler.add_chunk(_chunk(start_tick=3, values=[9.0, 10.0, 11.0, 12.0]))

    action, contributors = ensembler.get_action(current_tick=3)

    newer_weight = np.exp(-coeff * 3)
    expected = (3.0 + 9.0 * newer_weight) / (1.0 + newer_weight)
    assert contributors == 2
    np.testing.assert_allclose(action["positions"]["joint"], expected)


def test_temporal_ensemble_skips_elapsed_prefix_and_prunes_expired_chunks():
    ensembler = client_module.ACTTemporalEnsembler(("joint",), temporal_ensemble_coeff=0.0)
    ensembler.add_chunk(_chunk(start_tick=4, values=[4.0, 5.0, 6.0, 7.0]))

    action, contributors = ensembler.get_action(current_tick=7)
    assert contributors == 1
    assert action["positions"]["joint"] == 7.0

    action, contributors = ensembler.get_action(current_tick=8)
    assert action is None
    assert contributors == 0
    assert ensembler.active_chunk_count == 0


def test_temporal_ensemble_reset_and_safe_hold():
    ensembler = client_module.ACTTemporalEnsembler(("joint",), temporal_ensemble_coeff=0.0)
    ensembler.add_chunk(_chunk(start_tick=0, values=[1.0]))
    ensembler.reset()
    assert ensembler.get_action(0) == (None, 0)

    command = {
        "positions": {"joint": 1.5},
        "locomotion_command": np.asarray([0.2, -0.3, 0.4, 0.65], dtype=np.float32),
    }
    held = client_module.make_safe_hold_command(command)
    assert held["positions"] == command["positions"]
    np.testing.assert_allclose(held["locomotion_command"], [0.0, 0.0, 0.0, 0.65])


def test_inference_discards_disabled_session_and_uses_reenabled_session():
    first_inference_started = threading.Event()
    release_first_inference = threading.Event()
    inference_calls = 0

    class FakePolicyClient:
        def __init__(self, host, port):
            del host, port

        def ping(self):
            return True

        def close(self):
            return None

    class FakeAdapter:
        def __init__(self, policy_client, profile, video_keys):
            del policy_client, profile, video_keys

        def get_action_chunk(self, **kwargs):
            nonlocal inference_calls
            del kwargs
            inference_calls += 1
            if inference_calls == 1:
                first_inference_started.set()
                assert release_first_inference.wait(timeout=1)
            return SimpleNamespace(
                commands=[
                    {
                        "positions": {},
                        "locomotion_command": np.zeros(4, dtype=np.float32),
                    }
                ]
            )

    runner = client_module.DoubleBufferedPolicyRunner.__new__(
        client_module.DoubleBufferedPolicyRunner
    )
    runner.policy_host = "test"
    runner.policy_port = 0
    runner.profile = "x2"
    runner.task_override = None
    runner.execution_horizon = 1
    runner.execution_mode = "double_buffer"
    runner.video_keys = adapter_module.CAMERA_LAYOUTS["single"]
    runner._condition = threading.Condition()
    runner._stopping = False
    runner._error = None
    runner._latest_observation = _observation(session=1, enabled=True, sequence=1)
    runner._latest_observation_at = 1.0
    runner._last_inferred_session = None
    runner._last_inferred_sequence = -1
    runner._pending_commands = None
    runner._ready_chunks = deque()
    runner._control_tick = 0
    runner._control_tick_session = None

    with (
        patch.object(client_module, "PolicyClient", FakePolicyClient),
        patch.object(client_module, "RoboJuDoPolicyAdapter", FakeAdapter),
    ):
        inference_thread = threading.Thread(target=runner._inference_loop)
        inference_thread.start()
        assert first_inference_started.wait(timeout=1)

        with runner._condition:
            runner._latest_observation = _observation(session=1, enabled=False, sequence=2)
            runner._latest_observation_at = 2.0
            runner._condition.notify_all()
        release_first_inference.set()

        with runner._condition:
            runner._latest_observation = _observation(session=2, enabled=True, sequence=3)
            runner._latest_observation_at = 3.0
            runner._condition.notify_all()
            assert runner._condition.wait_for(
                lambda: runner._pending_commands is not None,
                timeout=1,
            )
            assert runner._pending_commands.control_session == 2
            assert runner._pending_commands.observation_sequence == 3
            runner._stopping = True
            runner._condition.notify_all()

        inference_thread.join(timeout=1)

    assert not inference_thread.is_alive()
    assert inference_calls == 2


def test_sync_inference_waits_for_active_chunk_to_finish():
    inference_called = threading.Event()

    class FakePolicyClient:
        def __init__(self, host, port):
            del host, port

        def ping(self):
            return True

        def close(self):
            return None

    class FakeAdapter:
        def __init__(self, policy_client, profile, video_keys):
            del policy_client, profile, video_keys

        def get_action_chunk(self, **kwargs):
            del kwargs
            inference_called.set()
            return SimpleNamespace(commands=[_command("joint", 1.0)])

    runner = client_module.DoubleBufferedPolicyRunner.__new__(
        client_module.DoubleBufferedPolicyRunner
    )
    runner.policy_host = "test"
    runner.policy_port = 0
    runner.profile = "x2"
    runner.task_override = None
    runner.execution_horizon = 1
    runner.execution_mode = "sync"
    runner.video_keys = adapter_module.CAMERA_LAYOUTS["single"]
    runner._condition = threading.Condition()
    runner._stopping = False
    runner._error = None
    runner._latest_observation = _observation(session=1, enabled=True, sequence=1)
    runner._latest_observation_at = time.monotonic()
    runner._last_inferred_session = None
    runner._last_inferred_sequence = -1
    runner._pending_commands = None
    runner._sync_chunk_active = True
    runner._ready_chunks = deque()
    runner._control_tick = 0
    runner._control_tick_session = ("test-stream", 1)

    with (
        patch.object(client_module, "PolicyClient", FakePolicyClient),
        patch.object(client_module, "RoboJuDoPolicyAdapter", FakeAdapter),
    ):
        inference_thread = threading.Thread(target=runner._inference_loop)
        inference_thread.start()
        assert not inference_called.wait(timeout=0.05)

        with runner._condition:
            runner._sync_chunk_active = False
            runner._condition.notify_all()
        assert inference_called.wait(timeout=1)

        with runner._condition:
            assert runner._condition.wait_for(
                lambda: runner._pending_commands is not None,
                timeout=1,
            )
            runner._stopping = True
            runner._condition.notify_all()
        inference_thread.join(timeout=1)

    assert not inference_thread.is_alive()


def test_temporal_ensemble_inference_does_not_wait_for_ready_queue_to_drain():
    inference_calls = 0

    class FakePolicyClient:
        def __init__(self, host, port):
            del host, port

        def ping(self):
            return True

        def close(self):
            return None

    class FakeAdapter:
        def __init__(self, policy_client, profile, video_keys):
            del policy_client, profile, video_keys

        def get_action_chunk(self, **kwargs):
            nonlocal inference_calls
            del kwargs
            inference_calls += 1
            return SimpleNamespace(commands=[_command("joint", float(inference_calls))])

    runner = client_module.DoubleBufferedPolicyRunner.__new__(
        client_module.DoubleBufferedPolicyRunner
    )
    runner.policy_host = "test"
    runner.policy_port = 0
    runner.profile = "x2"
    runner.task_override = None
    runner.execution_horizon = 1
    runner.execution_mode = "temporal_ensemble"
    runner.video_keys = adapter_module.CAMERA_LAYOUTS["single"]
    runner._condition = threading.Condition()
    runner._stopping = False
    runner._error = None
    runner._latest_observation = _observation(session=1, enabled=True, sequence=1)
    runner._latest_observation_at = 1.0
    runner._last_inferred_session = None
    runner._last_inferred_sequence = -1
    runner._pending_commands = None
    runner._ready_chunks = deque()
    runner._control_tick = 0
    runner._control_tick_session = ("test-stream", 1)

    with (
        patch.object(client_module, "PolicyClient", FakePolicyClient),
        patch.object(client_module, "RoboJuDoPolicyAdapter", FakeAdapter),
    ):
        inference_thread = threading.Thread(target=runner._inference_loop)
        inference_thread.start()
        with runner._condition:
            assert runner._condition.wait_for(lambda: len(runner._ready_chunks) == 1, timeout=1)
            runner._latest_observation = _observation(session=1, enabled=True, sequence=2)
            runner._latest_observation_at = 2.0
            runner._condition.notify_all()
            assert runner._condition.wait_for(lambda: len(runner._ready_chunks) == 2, timeout=1)
            runner._stopping = True
            runner._condition.notify_all()
        inference_thread.join(timeout=1)

    assert not inference_thread.is_alive()
    assert inference_calls == 2


def test_rtc_inference_sends_leftover_prefix_and_uses_actual_delay_on_replace():
    inference_started = threading.Event()
    release_inference = threading.Event()
    received_options = None

    class FakePolicyClient:
        def __init__(self, host, port):
            del host, port

        def ping(self):
            return True

        def close(self):
            return None

    class FakeAdapter:
        def __init__(self, policy_client, profile, video_keys):
            del policy_client, profile, video_keys

        def get_action_chunk(self, **kwargs):
            nonlocal received_options
            received_options = kwargs["options"]
            inference_started.set()
            assert release_inference.wait(timeout=1)
            values = [100.0, 101.0, 102.0, 103.0]
            return SimpleNamespace(
                commands=[_command("joint", value) for value in values],
                actions={"joint": np.asarray(values, dtype=np.float32)[None, :, None]},
            )

    runner = client_module.DoubleBufferedPolicyRunner.__new__(
        client_module.DoubleBufferedPolicyRunner
    )
    runner.policy_host = "test"
    runner.policy_port = 0
    runner.profile = "x2"
    runner.task_override = None
    runner.execution_horizon = 3
    runner.execution_mode = "rtc"
    runner.video_keys = adapter_module.CAMERA_LAYOUTS["single"]
    runner.rtc_prefix_schedule = "exp"
    runner.rtc_max_guidance_weight = 10.0
    runner.command_period = 0.1
    runner.observation_timeout = 10.0
    runner._condition = threading.Condition()
    runner._stopping = False
    runner._error = None
    runner._latest_observation = _observation(session=1, enabled=True, sequence=1)
    runner._latest_observation_at = time.monotonic()
    runner._last_inferred_session = None
    runner._last_inferred_sequence = -1
    runner._pending_commands = None
    runner._ready_chunks = deque()
    runner._rtc_queue = client_module.RTCActionQueue()
    runner._rtc_queue.replace(_rtc_chunk(values=[10.0, 11.0, 12.0, 13.0]), 1)
    runner._rtc_inference_latencies = deque([0.05], maxlen=10)
    runner._control_tick = 3
    runner._control_tick_session = ("test-stream", 1)

    with (
        patch.object(client_module, "PolicyClient", FakePolicyClient),
        patch.object(client_module, "RoboJuDoPolicyAdapter", FakeAdapter),
    ):
        inference_thread = threading.Thread(target=runner._inference_loop)
        inference_thread.start()
        assert inference_started.wait(timeout=1)
        with runner._condition:
            runner._control_tick = 5
        release_inference.set()
        with runner._condition:
            assert runner._condition.wait_for(
                lambda: (
                    runner._rtc_queue.chunk is not None
                    and runner._rtc_queue.chunk.observation_sequence == 1
                    and runner._rtc_queue.chunk.commands[0]["positions"]["joint"] == 100.0
                ),
                timeout=1,
            )
            runner._stopping = True
            runner._condition.notify_all()
        inference_thread.join(timeout=1)

    rtc = received_options["rtc"]
    np.testing.assert_allclose(rtc["prefix_actions"]["joint"], [[[11.0], [12.0], [13.0]]])
    assert rtc["estimated_delay_steps"] == 1
    assert runner._policy_action_horizon == 4
    assert runner._rtc_queue.qsize() == 2
    assert runner._rtc_queue.pop()["positions"]["joint"] == 102.0
    assert not inference_thread.is_alive()
