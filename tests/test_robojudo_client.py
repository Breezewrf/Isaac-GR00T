from collections import deque
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import numpy as np


ROBOJUDO_EXAMPLE = Path(__file__).parents[1] / "examples" / "RoboJuDo"
sys.path.insert(0, str(ROBOJUDO_EXAMPLE))

import run_robojudo_client as client_module  # noqa: E402


def _observation(*, session: int, enabled: bool, sequence: int):
    return client_module.Observation(
        stream_id="test-stream",
        control_session=session,
        takeover_enabled=enabled,
        sequence=sequence,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
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
        def __init__(self, policy_client, profile):
            del policy_client, profile

        def get_action(self, **kwargs):
            nonlocal inference_calls
            del kwargs
            inference_calls += 1
            if inference_calls == 1:
                first_inference_started.set()
                assert release_first_inference.wait(timeout=1)
            return [
                {
                    "positions": {},
                    "locomotion_command": np.zeros(4, dtype=np.float32),
                }
            ]

    runner = client_module.DoubleBufferedPolicyRunner.__new__(
        client_module.DoubleBufferedPolicyRunner
    )
    runner.policy_host = "test"
    runner.policy_port = 0
    runner.profile = "x2"
    runner.task_override = None
    runner.execution_horizon = 1
    runner.execution_mode = "double_buffer"
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
        def __init__(self, policy_client, profile):
            del policy_client, profile

        def get_action(self, **kwargs):
            nonlocal inference_calls
            del kwargs
            inference_calls += 1
            return [_command("joint", float(inference_calls))]

    runner = client_module.DoubleBufferedPolicyRunner.__new__(
        client_module.DoubleBufferedPolicyRunner
    )
    runner.policy_host = "test"
    runner.policy_port = 0
    runner.profile = "x2"
    runner.task_override = None
    runner.execution_horizon = 1
    runner.execution_mode = "temporal_ensemble"
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
