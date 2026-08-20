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
    runner._condition = threading.Condition()
    runner._stopping = False
    runner._error = None
    runner._latest_observation = _observation(session=1, enabled=True, sequence=1)
    runner._latest_observation_at = 1.0
    runner._last_inferred_session = None
    runner._last_inferred_sequence = -1
    runner._pending_commands = None

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
