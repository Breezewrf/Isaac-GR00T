# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a selectable asynchronous RoboJuDo observation-to-command deployment loop."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import cv2
from deploy_adapter import CAMERA_LAYOUTS, PROFILES, RoboJuDoPolicyAdapter
from gr00t.policy.server_client import PolicyClient
import msgpack
import numpy as np
import zmq
from zmq.utils.monitor import recv_monitor_message


EXECUTION_MODES = ("double_buffer", "temporal_ensemble", "rtc")
RTC_PREFIX_SCHEDULES = ("zeros", "ones", "linear", "exp")


@dataclass(frozen=True)
class Observation:
    stream_id: str
    control_session: int
    takeover_enabled: bool
    sequence: int
    images: dict[str, np.ndarray]
    joint_positions: dict[str, float]
    task: str


@dataclass(frozen=True)
class ActionChunk:
    stream_id: str
    control_session: int
    observation_sequence: int
    observation_received_at: float
    inference_seconds: float
    start_tick: int  # The control tick at the start of inference
    commands: list[dict]
    physical_actions: dict[str, np.ndarray] | None = None  # Provide Prefix action for RTC mode
    task: str = ""
    estimated_delay_steps: int = 0


class RTCActionQueue:
    """Lockstep physical-action/command queue; caller supplies synchronization."""

    def __init__(self):
        self._chunk: ActionChunk | None = None
        self._next_index = 0  # Index of the next command and next physical action simultaneously

    @property
    def chunk(self) -> ActionChunk | None:
        return self._chunk

    def clear(self):
        self._chunk = None
        self._next_index = 0

    def qsize(self) -> int:
        if self._chunk is None:
            return 0
        return max(0, len(self._chunk.commands) - self._next_index)

    def pop(self) -> dict | None:
        if self._chunk is None or self._next_index >= len(self._chunk.commands):
            return None
        command = self._chunk.commands[self._next_index]
        self._next_index += 1
        return command

    def get_left_over(self, session: tuple[str, int], task: str) -> dict[str, np.ndarray] | None:
        # Get remaining physical action as prefix actions for RTC reference
        chunk = self._chunk
        if (
            chunk is None
            or chunk.physical_actions is None
            or (chunk.stream_id, chunk.control_session) != session
            or chunk.task != task
            or self._next_index >= len(chunk.commands)
        ):
            return None
        return {
            key: value[:, self._next_index :].copy()
            for key, value in chunk.physical_actions.items()
        }

    def replace(self, chunk: ActionChunk, skipped_steps: int) -> bool:
        # Start at the skipped steps rather than the 0 of the chunk
        if chunk.physical_actions is None:
            raise ValueError("RTC action chunks must include physical_actions")
        horizon = len(chunk.commands)
        if any(value.shape[1] != horizon for value in chunk.physical_actions.values()):
            raise ValueError("RTC physical actions and commands must have the same horizon")
        if skipped_steps >= horizon:  # delay is too long, the chunk is expired
            self.clear()
            return False
        self._chunk = chunk
        self._next_index = max(0, skipped_steps)
        return True


@dataclass(frozen=True)
class _EnsembleChunk:
    start_tick: int
    actions: np.ndarray


class ACTTemporalEnsembler:
    """ACT-style temporal ensemble generalized to asynchronously returned chunks."""

    def __init__(self, joint_names: tuple[str, ...], temporal_ensemble_coeff: float):
        self.joint_names = joint_names
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self._chunks: deque[_EnsembleChunk] = deque()

    @property
    def active_chunk_count(self) -> int:
        return len(self._chunks)

    def reset(self):
        self._chunks.clear()

    def _pack_command(self, command: dict) -> np.ndarray:
        positions = command.get("positions")
        if not isinstance(positions, dict):
            raise ValueError("Temporal Ensemble command positions must be a dictionary")
        missing = [name for name in self.joint_names if name not in positions]
        if missing:
            raise ValueError(f"Temporal Ensemble command is missing joints: {missing}")
        locomotion = np.asarray(command.get("locomotion_command"), dtype=np.float32)
        if locomotion.shape != (4,):
            raise ValueError(
                f"Temporal Ensemble locomotion command has shape {locomotion.shape}, expected (4,)"
            )
        action = np.asarray(
            [positions[name] for name in self.joint_names] + locomotion.tolist(),
            dtype=np.float32,
        )
        if not np.isfinite(action).all():
            raise ValueError("Temporal Ensemble command contains non-finite values")
        return action

    def _unpack_command(self, action: np.ndarray) -> dict:
        joint_count = len(self.joint_names)
        return {
            "positions": dict(zip(self.joint_names, action[:joint_count].tolist(), strict=True)),
            "locomotion_command": action[joint_count:].astype(np.float32, copy=True),
        }

    def add_chunk(self, chunk: ActionChunk):
        actions = np.stack([self._pack_command(command) for command in chunk.commands])
        self._chunks.append(_EnsembleChunk(start_tick=chunk.start_tick, actions=actions))

    def get_action(self, current_tick: int) -> tuple[dict | None, int]:
        while (
            self._chunks
            and self._chunks[0].start_tick + len(self._chunks[0].actions) <= current_tick
        ):
            self._chunks.popleft()

        predictions = []
        prediction_start_ticks = []
        for chunk in self._chunks:
            action_index = current_tick - chunk.start_tick
            if 0 <= action_index < len(chunk.actions):
                predictions.append(chunk.actions[action_index])
                prediction_start_ticks.append(chunk.start_tick)
        if not predictions:
            return None, 0

        stacked = np.stack(predictions)
        prediction_offsets = np.asarray(prediction_start_ticks, dtype=np.float32)
        prediction_offsets -= prediction_offsets[0]
        weights = np.exp(-self.temporal_ensemble_coeff * prediction_offsets)
        if not np.isfinite(weights).all():
            raise ValueError("Temporal Ensemble produced non-finite weights")
        ensembled = np.average(stacked, axis=0, weights=weights).astype(np.float32)
        return self._unpack_command(ensembled), len(predictions)


def make_safe_hold_command(command: dict) -> dict:
    """Hold arm/height while stopping planar locomotion after prediction exhaustion."""
    locomotion = np.asarray(command["locomotion_command"], dtype=np.float32).copy()
    if locomotion.shape != (4,):
        raise ValueError(f"locomotion command has shape {locomotion.shape}, expected (4,)")
    locomotion[:3] = 0.0
    return {
        "positions": dict(command["positions"]),
        "locomotion_command": locomotion,
    }


class ObservationSubscriber:
    def __init__(self, endpoint: str, profile: str, image_keys: tuple[str, ...]):
        self.endpoint = endpoint
        self.profile = profile
        self.expected_joint_names = PROFILES[profile].joint_names
        self.expected_image_keys = tuple(image_keys)
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
        return self._decode_observation(parts)

    def _decode_observation(self, parts: list[bytes]) -> Observation:
        if not parts:
            raise ValueError("RoboJuDo observation multipart message is empty")
        header = msgpack.unpackb(parts[0], raw=False)
        protocol_version = header.get("protocol_version")
        if protocol_version == 1:
            image_keys = ("ego_view",)
            image_shapes = {"ego_view": header.get("shape", ())}
        elif protocol_version == 2:
            raw_image_keys = header.get("image_keys")
            if not isinstance(raw_image_keys, list) or not all(
                isinstance(key, str) and key for key in raw_image_keys
            ):
                raise ValueError("RoboJuDo protocol v2 image_keys must be a list of names")
            image_keys = tuple(raw_image_keys)
            raw_image_shapes = header.get("image_shapes", {})
            if not isinstance(raw_image_shapes, dict):
                raise ValueError("RoboJuDo protocol v2 image_shapes must be a dictionary")
            if set(raw_image_shapes) != set(image_keys):
                raise ValueError(
                    "RoboJuDo protocol v2 image_shapes keys must exactly match image_keys"
                )
            image_shapes = raw_image_shapes
        else:
            raise ValueError(f"unsupported RoboJuDo protocol version {protocol_version!r}")
        if image_keys != self.expected_image_keys:
            raise ValueError(
                f"RoboJuDo observation image order {image_keys} does not match camera layout "
                f"{self.expected_image_keys}"
            )
        expected_part_count = 1 + len(image_keys)
        if len(parts) != expected_part_count:
            raise ValueError(
                f"RoboJuDo observation has {len(parts)} parts, expected {expected_part_count} "
                f"for images {image_keys}"
            )
        if header.get("profile") != self.profile:
            raise ValueError(
                f"RoboJuDo profile {header.get('profile')!r} does not match {self.profile!r}"
            )
        joint_names = tuple(header.get("joint_names", ()))
        if joint_names != self.expected_joint_names:
            raise ValueError(
                f"RoboJuDo observation joint order {joint_names} does not match the deployment profile {self.expected_joint_names}"
            )
        positions = np.asarray(header.get("joint_positions"), dtype=np.float32)
        if positions.shape != (len(joint_names),) or not np.isfinite(positions).all():
            raise ValueError("RoboJuDo observation contains invalid joint positions")
        images = {}
        for key, jpeg in zip(image_keys, parts[1:], strict=True):
            bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"failed to decode RoboJuDo observation JPEG for {key!r}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            expected_shape = tuple(image_shapes.get(key, ()))
            if expected_shape and image.shape != expected_shape:
                raise ValueError(
                    f"RoboJuDo image {key!r} shape {image.shape} does not match {expected_shape}"
                )
            images[key] = image
        task = str(header.get("task", "")).strip()
        if not task:
            raise ValueError("RoboJuDo observation task must not be empty")
        stream_id = header.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("RoboJuDo observation stream_id must be a non-empty string")
        control_session = header.get("control_session")
        if (
            isinstance(control_session, bool)
            or not isinstance(control_session, int)
            or control_session < 0
        ):
            raise ValueError("RoboJuDo observation control_session must be a non-negative integer")
        takeover_enabled = header.get("takeover_enabled")
        if not isinstance(takeover_enabled, bool):
            raise ValueError("RoboJuDo observation takeover_enabled must be a boolean")
        return Observation(
            stream_id=stream_id,
            control_session=control_session,
            takeover_enabled=takeover_enabled,
            sequence=int(header["sequence"]),
            images=images,
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
        status_interval: float,
        task_override: str | None,
        execution_mode: str = "double_buffer",
        temporal_ensemble_coeff: float = 0.01,
        rtc_prefix_schedule: str = "exp",
        rtc_max_guidance_weight: float = 10.0,
        rtc_latency_window: int = 10,
        video_keys: tuple[str, ...] = CAMERA_LAYOUTS["single"],
    ):
        self.profile = profile
        self.policy_host = policy_host
        self.policy_port = policy_port
        self.subscriber = subscriber
        self.execution_horizon = execution_horizon
        self.command_period = 1.0 / command_fps
        self.observation_timeout = observation_timeout
        self.status_interval = status_interval
        self.task_override = task_override
        self.execution_mode = execution_mode
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.rtc_prefix_schedule = rtc_prefix_schedule
        self.rtc_max_guidance_weight = rtc_max_guidance_weight
        self.rtc_latency_window = rtc_latency_window
        self.video_keys = video_keys
        # Learned from the first full chunk returned by the policy. The action
        # horizon belongs to the checkpoint/embodiment, not to RoboJuDo.
        self._policy_action_horizon: int | None = None
        # Coordinates observations, inference results, and the command-loop tick.
        self._condition = threading.Condition()
        self._stopping = False
        self._latest_observation: Observation | None = None
        self._latest_observation_at = float("-inf")
        self._last_inferred_session: tuple[str, int] | None = None
        self._last_inferred_sequence = -1
        self._pending_commands: ActionChunk | None = None
        self._ready_chunks: deque[ActionChunk] = deque()
        self._rtc_queue = RTCActionQueue()
        # Store the last N inference latencies for RTC mode to estimate the delay steps
        # e.g. D_est = cel(delay_ms/command_period_ms) = ceil(82 / 33.3) = 3 steps
        self._rtc_inference_latencies: deque[float] = deque(maxlen=rtc_latency_window)
        self._control_tick = 0
        self._control_tick_session: tuple[str, int] | None = None
        self._error: Exception | None = None
        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        self._publisher.setsockopt(zmq.LINGER, 0)
        self._publisher.setsockopt(zmq.SNDHWM, 16)
        self._publisher_monitor = self._publisher.get_monitor_socket(
            events=zmq.EVENT_ACCEPTED | zmq.EVENT_DISCONNECTED
        )
        self._publisher.bind(command_endpoint)
        self._command_subscriber_connected = False
        self._observation_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)

    def _receive_loop(self):
        first_observation = True
        connected_at = time.monotonic()
        report_started_at = connected_at
        report_observations = 0
        report_sequence_gaps = 0
        last_stream_id = None
        last_sequence = None

        def report_health(now: float):
            nonlocal report_started_at, report_observations, report_sequence_gaps
            report_elapsed = now - report_started_at
            if report_elapsed < self.status_interval:
                return
            if report_observations:
                print(
                    f"[observation] rate={report_observations / report_elapsed:.1f}Hz, "
                    f"received={report_observations}, last_sequence={last_sequence}, "
                    f"sequence_gaps={report_sequence_gaps}",
                    flush=True,
                )
            elif last_sequence is None:
                print(
                    f"[observation] still waiting for first frame "
                    f"({now - connected_at:.1f}s elapsed)",
                    flush=True,
                )
            else:
                with self._condition:
                    latest_at = self._latest_observation_at
                print(
                    f"[observation] no frame for {now - latest_at:.2f}s; "
                    f"last_sequence={last_sequence}",
                    flush=True,
                )
            report_started_at = now
            report_observations = 0
            report_sequence_gaps = 0

        try:
            self.subscriber.connect()
            print(f"Connected to RoboJuDo observations at {self.subscriber.endpoint}", flush=True)
            while not self._stopping:
                observation = self.subscriber.receive()
                if observation is None:
                    report_health(time.monotonic())
                    continue
                now = time.monotonic()
                if observation.stream_id != last_stream_id:
                    if last_stream_id is not None:
                        print(
                            f"[observation] stream changed: {last_stream_id} -> "
                            f"{observation.stream_id}",
                            flush=True,
                        )
                    last_stream_id = observation.stream_id
                    last_sequence = None
                if last_sequence is not None:
                    if observation.sequence <= last_sequence:
                        print(
                            f"[observation] ignored non-increasing sequence "
                            f"{observation.sequence} after {last_sequence}",
                            flush=True,
                        )
                        continue
                    report_sequence_gaps += observation.sequence - last_sequence - 1
                last_sequence = observation.sequence
                report_observations += 1
                if first_observation:
                    image_shapes = {key: image.shape for key, image in observation.images.items()}
                    print(
                        f"Received first observation: sequence={observation.sequence}, "
                        f"session={observation.control_session}, "
                        f"takeover_enabled={observation.takeover_enabled}, "
                        f"images={image_shapes}, joints={len(observation.joint_positions)}",
                        flush=True,
                    )
                    first_observation = False
                with self._condition:
                    self._latest_observation = observation
                    self._latest_observation_at = time.monotonic()
                    self._condition.notify_all()
                report_health(now)
        except Exception as exc:
            self._set_error(exc)
        finally:
            self.subscriber.close()

    def _inference_loop(self):
        client = PolicyClient(host=self.policy_host, port=self.policy_port)
        adapter = RoboJuDoPolicyAdapter(client, self.profile, self.video_keys)
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
                        lambda: (
                            self._stopping
                            or self._error is not None
                            or (
                                (
                                    self.execution_mode in ("temporal_ensemble", "rtc")
                                    or self._pending_commands is None
                                )
                                and self._latest_observation is not None
                                and self._latest_observation.takeover_enabled
                                and (
                                    (
                                        self._latest_observation.stream_id,
                                        self._latest_observation.control_session,
                                    )
                                    != self._last_inferred_session
                                    or self._latest_observation.sequence
                                    > self._last_inferred_sequence
                                )
                            )
                        )
                    )
                    if self._stopping or self._error is not None:
                        return
                    observation = self._latest_observation
                    observation_received_at = self._latest_observation_at
                    observation_session = (
                        observation.stream_id,
                        observation.control_session,
                    )
                    query_tick = (
                        self._control_tick
                        if self._control_tick_session == observation_session
                        else 0
                    )

                    # RTC mode: get prefix actions related(rtc_prefix, estimated_delay_steps) for RTC reference
                    instruction = self.task_override or observation.task
                    rtc_prefix = None
                    prefix_length = 0
                    estimated_delay_steps = 0
                    if self.execution_mode == "rtc":
                        rtc_prefix = self._rtc_queue.get_left_over(observation_session, instruction)
                        if rtc_prefix is not None:
                            prefix_length = min(
                                value.shape[1] for value in rtc_prefix.values()
                            )  # L
                            if self._rtc_inference_latencies:
                                estimated_delay_steps = math.ceil(
                                    max(self._rtc_inference_latencies) / self.command_period
                                )
                            estimated_delay_steps = min(  # D_est
                                estimated_delay_steps,
                                prefix_length,
                                self.execution_horizon,
                            )

                    self._last_inferred_session = observation_session
                    self._last_inferred_sequence = observation.sequence
                inference_started_at = time.monotonic()

                # Construct RTC options and start inference. Always request the
                # complete policy chunk so its checkpoint-defined horizon can be
                # discovered and validated at runtime.
                physical_actions = None
                if self.execution_mode == "rtc":
                    rtc_options = None
                    if rtc_prefix is not None:
                        rtc_options = {
                            "rtc": {
                                "prefix_actions": rtc_prefix,
                                "prefix_length": prefix_length,  # L, Remaining prefix length of old action
                                "estimated_delay_steps": estimated_delay_steps,  # D_est
                                "guidance_horizon": min(
                                    self.execution_horizon, prefix_length
                                ),  # H, execution_horizon is the max guidance horizon for RTC
                                "prefix_schedule": self.rtc_prefix_schedule,
                                "max_guidance_weight": self.rtc_max_guidance_weight,
                            }
                        }
                    policy_chunk = adapter.get_action_chunk(
                        images=observation.images,
                        joint_positions=observation.joint_positions,
                        instruction=instruction,
                        execution_horizon=None,
                        options=rtc_options,
                    )
                    commands = policy_chunk.commands
                    physical_actions = policy_chunk.actions
                else:
                    policy_chunk = adapter.get_action_chunk(
                        images=observation.images,
                        joint_positions=observation.joint_positions,
                        instruction=instruction,
                        execution_horizon=None,
                    )
                    commands = policy_chunk.commands[: self.execution_horizon]
                policy_action_horizon = len(policy_chunk.commands)
                known_action_horizon = getattr(self, "_policy_action_horizon", None)
                if known_action_horizon is None:
                    if self.execution_horizon > policy_action_horizon:
                        raise ValueError(
                            f"--execution-horizon={self.execution_horizon} exceeds the policy "
                            f"action horizon {policy_action_horizon} discovered from its first chunk"
                        )
                    self._policy_action_horizon = policy_action_horizon
                    print(
                        f"[inference] detected policy action horizon: {policy_action_horizon}",
                        flush=True,
                    )
                elif policy_action_horizon != known_action_horizon:
                    raise ValueError(
                        "Policy action horizon changed between chunks: "
                        f"expected {known_action_horizon}, got {policy_action_horizon}"
                    )
                if not commands:
                    raise ValueError(
                        f"GR00T returned an empty action chunk for observation {observation.sequence}"
                    )
                inference_seconds = time.monotonic() - inference_started_at
                with self._condition:
                    # After Inference, check if the observation is still valid for the current session and task
                    if self.execution_mode == "rtc":
                        self._rtc_inference_latencies.append(inference_seconds)
                    latest = self._latest_observation
                    latest_instruction = (
                        None if latest is None else self.task_override or latest.task
                    )
                    if (
                        latest is None
                        or not latest.takeover_enabled
                        or (latest.stream_id, latest.control_session) != observation_session
                        or latest_instruction != instruction
                        or (
                            self.execution_mode == "rtc"
                            and time.monotonic() - self._latest_observation_at
                            > self.observation_timeout
                        )
                    ):
                        print(
                            f"[inference] discarded chunk from inactive session "
                            f"{observation.stream_id}:{observation.control_session}",
                            flush=True,
                        )
                        self._condition.notify_all()
                        continue
                    chunk = ActionChunk(
                        stream_id=observation.stream_id,
                        control_session=observation.control_session,
                        observation_sequence=observation.sequence,
                        observation_received_at=observation_received_at,
                        inference_seconds=inference_seconds,
                        start_tick=query_tick,
                        commands=commands,
                        physical_actions=physical_actions,
                        task=instruction,
                        estimated_delay_steps=estimated_delay_steps,
                    )
                    ready_tick = (
                        self._control_tick
                        if self._control_tick_session == observation_session
                        else query_tick
                    )
                    skipped_steps = (
                        max(0, ready_tick - query_tick) if rtc_prefix is not None else 0
                    )  # D_actual
                    if self.execution_mode == "temporal_ensemble":
                        self._ready_chunks.append(chunk)
                    elif self.execution_mode == "rtc":
                        activated = self._rtc_queue.replace(
                            chunk, skipped_steps
                        )  # Skipped the expired steps due to the inference delay
                        if not activated:
                            print(
                                f"[inference] RTC chunk expired before activation: "
                                f"actual_delay={skipped_steps}, horizon={len(commands)}; "
                                "waiting for an unguided refresh",
                                flush=True,
                            )
                    else:
                        self._pending_commands = chunk
                    self._condition.notify_all()
                print(
                    f"[inference] chunk ready: observation_sequence={observation.sequence}, "
                    f"session={observation.control_session}, actions={len(commands)}, "
                    f"latency={inference_seconds:.3f}s, query_tick={query_tick}, "
                    f"ready_tick={ready_tick}, skipped_steps={skipped_steps}, "
                    f"rtc_prefix={prefix_length}, "
                    f"rtc_estimated_delay={estimated_delay_steps}",
                    flush=True,
                )
        except Exception as exc:
            self._set_error(exc)
        finally:
            client.close()

    def _set_error(self, exc: Exception):
        with self._condition:
            self._error = exc
            self._condition.notify_all()

    def _poll_command_subscriber(self):
        while self._publisher_monitor.poll(0, zmq.POLLIN):
            event = recv_monitor_message(self._publisher_monitor)
            endpoint = event.get("endpoint", b"")
            if isinstance(endpoint, bytes):
                endpoint = endpoint.decode(errors="replace")
            if event["event"] == zmq.EVENT_ACCEPTED:
                self._command_subscriber_connected = True
                print(f"[command] subscriber connected: {endpoint}", flush=True)
            elif event["event"] == zmq.EVENT_DISCONNECTED:
                self._command_subscriber_connected = False
                print(f"[command] subscriber disconnected: {endpoint}", flush=True)

    def run(self):
        print("Waiting for the first RoboJuDo observation and GR00T action chunk...", flush=True)
        self._observation_thread.start()
        self._inference_thread.start()
        active_commands: deque[dict] = deque()
        temporal_ensembler = (
            ACTTemporalEnsembler(
                PROFILES[self.profile].joint_names,
                self.temporal_ensemble_coeff,
            )
            if self.execution_mode == "temporal_ensemble"
            else None
        )
        last_command = None
        command_sequence = 0
        command_stream_started = False
        observation_was_fresh = False
        control_was_enabled = False
        active_session: tuple[str, int] | None = None
        active_task: str | None = None
        holding_last_command = False
        report_started_at = time.monotonic()
        report_commands = 0
        next_command_at = time.monotonic()
        ensemble_contributors = 0
        while True:
            self._poll_command_subscriber()
            ready_chunks = []
            rtc_command = None
            current_tick = 0
            with self._condition:
                if self._error is not None:
                    raise RuntimeError("RoboJuDo deployment worker failed") from self._error
                now = time.monotonic()
                observation_fresh = now - self._latest_observation_at <= self.observation_timeout
                latest = self._latest_observation
                control_enabled = bool(
                    observation_fresh and latest is not None and latest.takeover_enabled
                )
                current_session = (
                    (latest.stream_id, latest.control_session) if latest is not None else None
                )
                current_task = None if latest is None else self.task_override or latest.task

                # Handle observation timeout, control takeover, and session/task changes
                if not observation_fresh:
                    if observation_was_fresh:
                        if last_command is None or active_session is None:
                            print(
                                "RoboJuDo observation timed out; no previous command to hold",
                                flush=True,
                            )
                        else:
                            print(
                                "RoboJuDo observation timed out; holding arm/height and "
                                "setting vx/vy/yaw_rate to zero",
                                flush=True,
                            )
                    active_commands.clear()
                    self._pending_commands = None
                    self._ready_chunks.clear()
                    self._rtc_queue.clear()
                    self._rtc_inference_latencies.clear()
                    if temporal_ensembler is not None:
                        temporal_ensembler.reset()
                    self._control_tick = 0
                    if last_command is not None and active_session is not None:
                        last_command = make_safe_hold_command(last_command)
                        self._control_tick_session = active_session
                        holding_last_command = True
                    else:
                        last_command = None
                        self._control_tick_session = None
                        holding_last_command = False
                        active_session = None
                        active_task = None
                elif not observation_was_fresh:
                    print("RoboJuDo observation stream is fresh", flush=True)
                if observation_fresh and not control_enabled:
                    cleared_pending = self._pending_commands is not None
                    if control_was_enabled:
                        print(
                            "[control] takeover disabled; cleared active and pending commands",
                            flush=True,
                        )
                    active_commands.clear()
                    last_command = None
                    self._pending_commands = None
                    self._ready_chunks.clear()
                    self._rtc_queue.clear()
                    self._rtc_inference_latencies.clear()
                    if temporal_ensembler is not None:
                        temporal_ensembler.reset()
                    self._control_tick = 0
                    self._control_tick_session = None
                    holding_last_command = False
                    active_session = None
                    active_task = None
                    if control_was_enabled or cleared_pending:
                        self._condition.notify_all()
                elif control_enabled and (
                    current_session != active_session or current_task != active_task
                ):
                    active_commands.clear()
                    last_command = None
                    holding_last_command = False
                    active_session = current_session
                    active_task = current_task
                    self._control_tick = 0
                    self._control_tick_session = current_session
                    if temporal_ensembler is not None:
                        temporal_ensembler.reset()
                    rtc_chunk = self._rtc_queue.chunk
                    if (
                        rtc_chunk is None
                        or (rtc_chunk.stream_id, rtc_chunk.control_session) != current_session
                        or rtc_chunk.task != current_task
                    ):
                        self._rtc_queue.clear()
                        self._rtc_inference_latencies.clear()
                    pending_session = (
                        (
                            self._pending_commands.stream_id,
                            self._pending_commands.control_session,
                        )
                        if self._pending_commands is not None
                        else None
                    )
                    if self._pending_commands is not None and pending_session != current_session:
                        self._pending_commands = None
                    self._ready_chunks = deque(
                        chunk
                        for chunk in self._ready_chunks
                        if (chunk.stream_id, chunk.control_session) == current_session
                    )
                    self._condition.notify_all()
                    print(
                        f"[control] takeover enabled for session "
                        f"{current_session[0]}:{current_session[1]}; waiting for a fresh chunk",
                        flush=True,
                    )
                if (
                    self.execution_mode == "double_buffer"
                    and control_enabled
                    and not active_commands
                    and self._pending_commands is not None
                ):
                    chunk = self._pending_commands
                    self._pending_commands = None
                    self._condition.notify_all()
                    chunk_age = now - chunk.observation_received_at
                    chunk_session = (chunk.stream_id, chunk.control_session)
                    if chunk_session != current_session:
                        print(
                            f"[command] discarded chunk from inactive session "
                            f"{chunk.stream_id}:{chunk.control_session}",
                            flush=True,
                        )
                    elif chunk_age <= self.observation_timeout:
                        active_commands.extend(chunk.commands)
                        holding_last_command = False
                        print(
                            f"[command] activated chunk: observation_sequence={chunk.observation_sequence}, "
                            f"session={chunk.control_session}, "
                            f"actions={len(chunk.commands)}, age={chunk_age:.3f}s, "
                            f"inference={chunk.inference_seconds:.3f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"[command] discarded stale chunk for observation_sequence="
                            f"{chunk.observation_sequence} (age={chunk_age:.3f}s)",
                            flush=True,
                        )
                if self.execution_mode == "temporal_ensemble" and control_enabled:
                    while self._ready_chunks:
                        chunk = self._ready_chunks.popleft()
                        chunk_session = (chunk.stream_id, chunk.control_session)
                        chunk_age = now - chunk.observation_received_at
                        if chunk_session != current_session:
                            print(
                                f"[command] discarded chunk from inactive session "
                                f"{chunk.stream_id}:{chunk.control_session}",
                                flush=True,
                            )
                        elif chunk_age <= self.observation_timeout:
                            ready_chunks.append(chunk)
                        else:
                            print(
                                f"[command] discarded stale chunk for observation_sequence="
                                f"{chunk.observation_sequence} (age={chunk_age:.3f}s)",
                                flush=True,
                            )
                    current_tick = self._control_tick
                if self.execution_mode == "rtc" and control_enabled:
                    rtc_command = self._rtc_queue.pop()
                    current_tick = self._control_tick
                observation_was_fresh = observation_fresh
                control_was_enabled = control_enabled
            if self.execution_mode == "temporal_ensemble":
                for chunk in ready_chunks:
                    temporal_ensembler.add_chunk(chunk)
                    print(
                        f"[command] added ensemble chunk: "
                        f"observation_sequence={chunk.observation_sequence}, "
                        f"session={chunk.control_session}, start_tick={chunk.start_tick}, "
                        f"actions={len(chunk.commands)}",
                        flush=True,
                    )
                ensembled_command, ensemble_contributors = temporal_ensembler.get_action(
                    current_tick
                )
                if ensembled_command is not None:
                    last_command = ensembled_command
                    holding_last_command = False
                elif last_command is not None:
                    if not holding_last_command:
                        print(
                            "[command] Temporal Ensemble horizon exhausted; holding arm/height "
                            "and setting vx/vy/yaw_rate to zero",
                            flush=True,
                        )
                    last_command = make_safe_hold_command(last_command)
                    holding_last_command = True
            elif self.execution_mode == "rtc":
                if rtc_command is not None:
                    last_command = rtc_command
                    holding_last_command = False
                elif last_command is not None:
                    if not holding_last_command:
                        print(
                            "[command] RTC queue exhausted; holding arm/height "
                            "and setting vx/vy/yaw_rate to zero",
                            flush=True,
                        )
                    last_command = make_safe_hold_command(last_command)
                    holding_last_command = True
            else:
                if active_commands:
                    last_command = active_commands.popleft()
                    holding_last_command = False
                elif last_command is not None and not holding_last_command:
                    print(
                        "[command] action horizon exhausted; holding the last command "
                        "until the next chunk is ready",
                        flush=True,
                    )
                    holding_last_command = True
            if last_command is not None:
                if active_session is None:
                    raise RuntimeError("cannot publish a command without an active control session")
                self._publisher.send_json(
                    {
                        "sequence": command_sequence,
                        "stream_id": active_session[0],
                        "control_session": active_session[1],
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
                report_commands += 1
            if control_enabled and active_session is not None:
                with self._condition:
                    if self._control_tick_session == active_session:
                        self._control_tick += 1
            report_now = time.monotonic()
            report_elapsed = report_now - report_started_at
            if report_elapsed >= self.status_interval:
                with self._condition:
                    observation_age = report_now - self._latest_observation_at
                    ready_chunk_count = len(self._ready_chunks)
                    inference_pending = (
                        self._pending_commands is not None
                        if self.execution_mode == "double_buffer"
                        else bool(ready_chunk_count)
                    )
                    latest = self._latest_observation
                    takeover_enabled = bool(latest and latest.takeover_enabled)
                    control_session = None if latest is None else latest.control_session
                    rtc_queue_size = self._rtc_queue.qsize()
                if self.execution_mode == "temporal_ensemble":
                    mode_status = (
                        f"control_tick={current_tick}, "
                        f"ensemble_chunks={temporal_ensembler.active_chunk_count}, "
                        f"ensemble_contributors={ensemble_contributors}, "
                        f"ready_chunks={ready_chunk_count}"
                    )
                elif self.execution_mode == "rtc":
                    mode_status = (
                        f"control_tick={current_tick}, rtc_queue_remaining={rtc_queue_size}, "
                        f"holding={holding_last_command}, "
                        f"latency_samples={len(self._rtc_inference_latencies)}"
                    )
                else:
                    mode_status = (
                        f"chunk_remaining={len(active_commands)}, "
                        f"holding={holding_last_command}, "
                        f"inference_chunk_pending={inference_pending}"
                    )
                print(
                    f"[command] rate={report_commands / report_elapsed:.1f}Hz, "
                    f"published={report_commands}, next_sequence={command_sequence}, "
                    f"subscriber_connected={self._command_subscriber_connected}, "
                    f"mode={self.execution_mode}, {mode_status}, "
                    f"observation_age={observation_age:.3f}s, "
                    f"takeover_enabled={takeover_enabled}, control_session={control_session}",
                    flush=True,
                )
                report_started_at = report_now
                report_commands = 0
            next_command_at += self.command_period
            time.sleep(max(0.0, next_command_at - time.monotonic()))

    def close(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._observation_thread.join(timeout=2)
        self._inference_thread.join(timeout=2)
        self._publisher.disable_monitor()
        self._publisher_monitor.close(linger=0)
        self._publisher.close(linger=0)
        self._context.term()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument(
        "--camera-layout",
        choices=sorted(CAMERA_LAYOUTS),
        default="single",
        help="Expected observation image layout; mulcam requires protocol v2",
    )
    parser.add_argument("--robot-endpoint", required=True, help="RoboJuDo observation endpoint")
    parser.add_argument("--command-endpoint", default="tcp://*:8559")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="double_buffer",
        help="Action execution strategy; RTC continuously replaces a guided action queue",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help=(
            "Execution/guidance horizon; its upper bound is discovered from the first "
            "action chunk returned by the policy"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="ACT exponential weight coefficient; 0 gives an equal average",
    )
    parser.add_argument(
        "--rtc-prefix-schedule",
        choices=RTC_PREFIX_SCHEDULES,
        default="exp",
        help="RTC prefix attention schedule between estimated delay and execution horizon",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
        help="Maximum per-denoising-step RTC correction gain",
    )
    parser.add_argument(
        "--rtc-latency-window",
        type=int,
        default=10,
        help="Number of recent inference latencies used by RTC's rolling maximum",
    )
    parser.add_argument("--command-fps", type=float, default=30.0)
    parser.add_argument("--observation-timeout", type=float, default=1.0)
    parser.add_argument("--status-interval", type=float, default=5.0)
    parser.add_argument("--task", default=None, help="Override the task sent by RoboJuDo")
    args = parser.parse_args()
    if args.camera_layout == "mulcam" and args.profile != "g1_23dof":
        parser.error("--camera-layout=mulcam is currently supported only for --profile=g1_23dof")
    if args.execution_horizon < 1:
        parser.error("--execution-horizon must be positive")
    if args.command_fps <= 0:
        parser.error("--command-fps must be positive")
    if args.observation_timeout <= 0:
        parser.error("--observation-timeout must be positive")
    if args.status_interval <= 0:
        parser.error("--status-interval must be positive")
    if not np.isfinite(args.temporal_ensemble_coeff):
        parser.error("--temporal-ensemble-coeff must be finite")
    if not np.isfinite(args.rtc_max_guidance_weight) or args.rtc_max_guidance_weight < 0:
        parser.error("--rtc-max-guidance-weight must be finite and non-negative")
    if args.rtc_latency_window <= 0:
        parser.error("--rtc-latency-window must be positive")
    return args


def main():
    args = parse_args()
    print(
        f"Starting RoboJuDo deploy client: profile={args.profile}, "
        f"camera_layout={args.camera_layout}, "
        f"mode={args.execution_mode}, observations={args.robot_endpoint}, "
        f"commands={args.command_endpoint}",
        flush=True,
    )
    video_keys = CAMERA_LAYOUTS[args.camera_layout]
    subscriber = ObservationSubscriber(args.robot_endpoint, args.profile, video_keys)
    runner = DoubleBufferedPolicyRunner(
        profile=args.profile,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        subscriber=subscriber,
        command_endpoint=args.command_endpoint,
        execution_horizon=args.execution_horizon,
        command_fps=args.command_fps,
        observation_timeout=args.observation_timeout,
        status_interval=args.status_interval,
        task_override=args.task,
        execution_mode=args.execution_mode,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        rtc_prefix_schedule=args.rtc_prefix_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_latency_window=args.rtc_latency_window,
        video_keys=video_keys,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
