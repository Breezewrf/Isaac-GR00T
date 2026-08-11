# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the double-buffered RoboJuDo observation-to-command deployment loop."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

from deploy_adapter import PROFILES, RoboJuDoPolicyAdapter
from gr00t.policy.server_client import PolicyClient


@dataclass(frozen=True)
class Observation:
    sequence: int
    image: np.ndarray
    joint_positions: dict[str, float]
    task: str


class ObservationSubscriber:
    def __init__(self, endpoint: str, profile: str):
        self.endpoint = endpoint
        self.profile = profile
        self.expected_joint_names = PROFILES[profile].joint_names
        self._context = zmq.Context()
        self._socket = None

    def connect(self):
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(self.endpoint)

    def receive(self, timeout_ms: int = 100) -> Observation | None:
        if self._socket is None:
            raise RuntimeError("RoboJuDo observation subscriber is not connected")
        if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
            return None
        parts = self._socket.recv_multipart()
        while self._socket.poll(0, zmq.POLLIN):
            parts = self._socket.recv_multipart()
        if len(parts) != 2:
            raise ValueError(f"RoboJuDo observation has {len(parts)} parts, expected 2")
        header = msgpack.unpackb(parts[0], raw=False)
        if header.get("protocol_version") != 1:
            raise ValueError(f"unsupported RoboJuDo protocol version {header.get('protocol_version')!r}")
        if header.get("profile") != self.profile:
            raise ValueError(
                f"RoboJuDo profile {header.get('profile')!r} does not match {self.profile!r}"
            )
        joint_names = tuple(header.get("joint_names", ()))
        if joint_names != self.expected_joint_names:
            raise ValueError("RoboJuDo observation joint order does not match the deployment profile")
        positions = np.asarray(header.get("joint_positions"), dtype=np.float32)
        if positions.shape != (len(joint_names),) or not np.isfinite(positions).all():
            raise ValueError("RoboJuDo observation contains invalid joint positions")
        bgr = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("failed to decode RoboJuDo observation JPEG")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        expected_shape = tuple(header.get("shape", ()))
        if expected_shape and image.shape != expected_shape:
            raise ValueError(f"RoboJuDo image shape {image.shape} does not match {expected_shape}")
        task = str(header.get("task", "")).strip()
        if not task:
            raise ValueError("RoboJuDo observation task must not be empty")
        return Observation(
            sequence=int(header["sequence"]),
            image=image,
            joint_positions=dict(zip(joint_names, positions.tolist(), strict=True)),
            task=task,
        )

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._context.term()


class DoubleBufferedPolicyRunner:
    def __init__(
        self,
        profile: str,
        policy_host: str,
        policy_port: int,
        subscriber: ObservationSubscriber,
        command_endpoint: str,
        execution_horizon: int,
        command_fps: float,
        observation_timeout: float,
        task_override: str | None,
    ):
        self.profile = profile
        self.policy_host = policy_host
        self.policy_port = policy_port
        self.subscriber = subscriber
        self.execution_horizon = execution_horizon
        self.command_period = 1.0 / command_fps
        self.observation_timeout = observation_timeout
        self.task_override = task_override
        self._condition = threading.Condition()
        self._stopping = False
        self._latest_observation: Observation | None = None
        self._latest_observation_at = float("-inf")
        self._last_inferred_sequence = -1
        self._pending_commands: tuple[float, list[dict]] | None = None
        self._error: Exception | None = None
        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        self._publisher.setsockopt(zmq.LINGER, 0)
        self._publisher.setsockopt(zmq.SNDHWM, 16)
        self._publisher.bind(command_endpoint)
        self._observation_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)

    def _receive_loop(self):
        first_observation = True
        try:
            self.subscriber.connect()
            print(f"Connected to RoboJuDo observations at {self.subscriber.endpoint}", flush=True)
            while not self._stopping:
                observation = self.subscriber.receive()
                if observation is None:
                    continue
                if first_observation:
                    print(
                        f"Received first observation: sequence={observation.sequence}, "
                        f"image={observation.image.shape}, joints={len(observation.joint_positions)}",
                        flush=True,
                    )
                    first_observation = False
                with self._condition:
                    self._latest_observation = observation
                    self._latest_observation_at = time.monotonic()
                    self._condition.notify_all()
        except Exception as exc:
            self._set_error(exc)
        finally:
            self.subscriber.close()

    def _inference_loop(self):
        client = PolicyClient(host=self.policy_host, port=self.policy_port)
        adapter = RoboJuDoPolicyAdapter(client, self.profile)
        try:
            if not client.ping():
                raise ConnectionError(
                    f"GR00T policy server is unavailable at {self.policy_host}:{self.policy_port}"
                )
            print(
                f"Connected to GR00T policy server at {self.policy_host}:{self.policy_port}",
                flush=True,
            )
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopping
                        or self._error is not None
                        or (
                            self._pending_commands is None
                            and self._latest_observation is not None
                            and self._latest_observation.sequence > self._last_inferred_sequence
                        )
                    )
                    if self._stopping or self._error is not None:
                        return
                    observation = self._latest_observation
                    observation_received_at = self._latest_observation_at
                    self._last_inferred_sequence = observation.sequence
                commands = adapter.get_action(
                    image=observation.image,
                    joint_positions=observation.joint_positions,
                    instruction=self.task_override or observation.task,
                    execution_horizon=self.execution_horizon,
                )
                with self._condition:
                    self._pending_commands = (observation_received_at, commands)
                    self._condition.notify_all()
        except Exception as exc:
            self._set_error(exc)
        finally:
            client.close()

    def _set_error(self, exc: Exception):
        with self._condition:
            self._error = exc
            self._condition.notify_all()

    def run(self):
        print("Waiting for the first RoboJuDo observation and GR00T action chunk...", flush=True)
        self._observation_thread.start()
        self._inference_thread.start()
        active_commands: deque[dict] = deque()
        last_command = None
        command_sequence = 0
        command_stream_started = False
        observation_was_fresh = False
        next_command_at = time.monotonic()
        while True:
            with self._condition:
                if self._error is not None:
                    raise RuntimeError("RoboJuDo deployment worker failed") from self._error
                now = time.monotonic()
                observation_fresh = now - self._latest_observation_at <= self.observation_timeout
                if not observation_fresh:
                    if observation_was_fresh:
                        print("RoboJuDo observation timed out; command publishing stopped", flush=True)
                    active_commands.clear()
                    last_command = None
                    self._pending_commands = None
                elif not active_commands and self._pending_commands is not None:
                    observation_received_at, commands = self._pending_commands
                    self._pending_commands = None
                    self._condition.notify_all()
                    if now - observation_received_at <= self.observation_timeout:
                        active_commands.extend(commands)
                observation_was_fresh = observation_fresh
            if active_commands:
                last_command = active_commands.popleft()
            if last_command is not None:
                self._publisher.send_json(
                    {
                        "sequence": command_sequence,
                        "positions": last_command["positions"],
                        "locomotion_command": last_command["locomotion_command"].tolist(),
                    }
                )
                if not command_stream_started:
                    print(
                        f"Publishing GR00T commands on {self._publisher.getsockopt_string(zmq.LAST_ENDPOINT)}",
                        flush=True,
                    )
                    command_stream_started = True
                command_sequence += 1
            next_command_at += self.command_period
            time.sleep(max(0.0, next_command_at - time.monotonic()))

    def close(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._observation_thread.join(timeout=2)
        self._inference_thread.join(timeout=2)
        self._publisher.close(linger=0)
        self._context.term()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--robot-endpoint", required=True, help="RoboJuDo observation endpoint")
    parser.add_argument("--command-endpoint", default="tcp://*:8559")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--command-fps", type=float, default=30.0)
    parser.add_argument("--observation-timeout", type=float, default=1.0)
    parser.add_argument("--task", default=None, help="Override the task sent by RoboJuDo")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"Starting RoboJuDo deploy client: profile={args.profile}, "
        f"observations={args.robot_endpoint}, commands={args.command_endpoint}",
        flush=True,
    )
    subscriber = ObservationSubscriber(args.robot_endpoint, args.profile)
    runner = DoubleBufferedPolicyRunner(
        profile=args.profile,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        subscriber=subscriber,
        command_endpoint=args.command_endpoint,
        execution_horizon=args.execution_horizon,
        command_fps=args.command_fps,
        observation_timeout=args.observation_timeout,
        task_override=args.task,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
