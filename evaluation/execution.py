"""Deterministic control-clock replay using the deployment TE and RTC containers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np


MODES = (
    "synchronous",
    "double_buffer",
    "rtc_schedule_no_guidance",
    "temporal_ensemble",
    "rtc",
)


@lru_cache(maxsize=1)
def deployment_types():
    # The deployment entry point imports deploy_adapter as a sibling script.
    # Import its existing containers without starting any transport or threads.
    path = str(Path(__file__).resolve().parents[1] / "examples" / "RoboJuDo")
    sys.path.insert(0, path)
    try:
        module = importlib.import_module("run_robojudo_client")
    finally:
        sys.path.remove(path)
    return module.ActionChunk, module.ACTTemporalEnsembler, module.RTCActionQueue


@dataclass(frozen=True)
class ReplayConfig:
    mode: str
    fps: float
    action_horizon: int
    execution_horizon: int = 8
    ensemble_horizon: int | None = None
    query_interval: int = 1
    temporal_ensemble_coeff: float = 0.01
    rtc_guidance_horizon: int = 8
    rtc_prefix_schedule: str = "exp"
    rtc_max_guidance_weight: float = 10.0
    rtc_latency_window: int = 10
    latency_ticks: tuple[int, ...] = (0,)
    measured_latency: bool = False
    replay_clock: str = "wall_clock"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown execution mode: {self.mode}")
        if self.replay_clock not in ("wall_clock", "execution_clock"):
            raise ValueError("replay_clock must be 'wall_clock' or 'execution_clock'")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        for name in ("action_horizon", "query_interval", "rtc_latency_window"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("execution_horizon", "rtc_guidance_horizon", "ensemble_horizon"):
            value = getattr(self, name)
            if value is not None and not 1 <= value <= self.action_horizon:
                raise ValueError(f"{name} must be in [1, action_horizon]")
        if self.rtc_prefix_schedule not in ("zeros", "ones", "linear", "exp"):
            raise ValueError("Invalid RTC prefix schedule")
        if not np.isfinite(self.temporal_ensemble_coeff):
            raise ValueError("temporal_ensemble_coeff must be finite")
        if not np.isfinite(self.rtc_max_guidance_weight) or self.rtc_max_guidance_weight < 0:
            raise ValueError("rtc_max_guidance_weight must be finite and nonnegative")
        if not self.latency_ticks or any(
            isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in self.latency_ticks
        ):
            raise ValueError("latency_ticks must contain nonnegative integers")


class ActionLayout:
    """Lossless physical-action mapping into the existing TE command interface.

    All dimensions use synthetic position names; dummy locomotion values are
    ignored. Averaging remains identical, including for velocity commands.
    Explicit velocity groups control the safe-hold behavior and metric labels.
    """

    def __init__(self, widths: dict[str, int], velocity_groups=()):
        if not widths or any(width < 1 for width in widths.values()):
            raise ValueError("Action groups must have positive widths")
        self.widths = dict(widths)
        self.slices = {}
        offset = 0
        for key, width in widths.items():
            self.slices[key] = slice(offset, offset + width)
            offset += width
        self.size = offset
        self.names = tuple(f"dimension_{i}" for i in range(offset))
        self.velocity_groups = tuple(velocity_groups)
        if set(self.velocity_groups) - widths.keys():
            raise ValueError("Unknown velocity action group")

    def validate_chunk(self, actions, horizon):
        if set(actions) != set(self.widths):
            raise ValueError("Policy action groups do not match dataset action groups")
        arrays = {}
        for key, width in self.widths.items():
            array = np.asarray(actions[key], dtype=np.float32)
            if array.shape != (1, horizon, width) or not np.isfinite(array).all():
                raise ValueError(f"Invalid physical action chunk {key}: {array.shape}")
            arrays[key] = array.copy()
        return arrays

    def commands(self, actions):
        flat = np.concatenate([actions[key][0] for key in self.widths], axis=-1)
        return [self.command(row) for row in flat]

    def command(self, row):
        return {
            "positions": dict(zip(self.names, np.asarray(row).tolist(), strict=True)),
            "locomotion_command": np.zeros(4, dtype=np.float32),
        }

    def flatten(self, command):
        return np.asarray([command["positions"][key] for key in self.names], dtype=np.float64)

    def safe_hold(self, command):
        row = self.flatten(command)
        for group in self.velocity_groups:
            row[self.slices[group]] = 0
        return self.command(row)


@dataclass
class ReplayResult:
    actions: np.ndarray
    status: np.ndarray  # 0: no command, 1: prediction, 2: held command
    boundaries: np.ndarray
    contributors: np.ndarray
    segments: np.ndarray
    events: list[dict]
    chunks: list[dict[str, np.ndarray]]
    dataset_ticks: np.ndarray | None = None


def replay(
    infer: Callable,
    observation_at: Callable,
    task_at: Callable,
    steps: int,
    layout: ActionLayout,
    config: ReplayConfig,
) -> ReplayResult:
    """Replay one episode; GT actions are deliberately absent from this API.

    At a tick: reset on task change, deliver finished work, activate a buffer,
    launch at most one query, deliver zero-delay work, then emit a command.
    One request is in flight. Delay traces are indexed by query *tick*, so
    policies see the same latency environment even if their query counts differ.
    """
    if steps < 1:
        raise ValueError("An episode must contain at least one step")
    ActionChunk, Ensembler, Queue = deployment_types()
    ensemble = Ensembler(layout.names, config.temporal_ensemble_coeff)
    queue = Queue()
    active = deque()
    pending = None
    flight = None
    last = None
    task = None
    session = 0
    next_query = 0
    latency_history = deque(maxlen=config.rtc_latency_window)
    result = ReplayResult(
        actions=np.full((steps, layout.size), np.nan),
        status=np.zeros(steps, dtype=np.int8),
        boundaries=np.zeros(steps, dtype=bool),
        contributors=np.zeros(steps, dtype=np.int32),
        segments=np.zeros(steps, dtype=np.int32),
        events=[],
        chunks=[],
        dataset_ticks=np.full(steps, -1, dtype=np.int64),
    )

    def ensure_capacity(required):
        current = len(result.status)
        if required <= current:
            return
        capacity = max(required, current * 2)

        def grow(array, fill):
            shape = (capacity, *array.shape[1:])
            expanded = np.full(shape, fill, dtype=array.dtype)
            expanded[:current] = array
            return expanded

        result.actions = grow(result.actions, np.nan)
        result.status = grow(result.status, 0)
        result.boundaries = grow(result.boundaries, False)
        result.contributors = grow(result.contributors, 0)
        result.segments = grow(result.segments, 0)
        result.dataset_ticks = grow(result.dataset_ticks, -1)

    def deliver(tick):
        nonlocal flight, pending
        if flight is None or flight[0] > tick:
            return
        _, chunk, event, had_prefix = flight
        flight = None
        if chunk.control_session != session:
            event["discarded_task_change"] = True
            return
        latency_history.append(event["effective_latency_seconds"])
        event["delivered_tick"] = tick
        if config.mode in ("synchronous", "double_buffer"):
            pending = (chunk, event)
        elif config.mode == "temporal_ensemble":
            ensemble.add_chunk(chunk)
            current_action_tick = dataset_tick if config.replay_clock == "execution_clock" else tick
            event["expired"] = current_action_tick - chunk.start_tick >= len(chunk.commands)
            result.boundaries[tick] = not event["expired"]
            event["activated_tick"] = tick if not event["expired"] else None
        else:
            # RTC's unguided startup/recovery begins at index 0. The no-guidance
            # ablation keeps exactly this scheduling but omits model guidance.
            current_action_tick = dataset_tick if config.replay_clock == "execution_clock" else tick
            skipped = current_action_tick - chunk.start_tick if had_prefix else 0
            event["skipped_steps"] = skipped
            event["expired"] = not queue.replace(chunk, skipped)
            result.boundaries[tick] = not event["expired"]
            event["activated_tick"] = tick if not event["expired"] else None

    def activate(tick):
        nonlocal pending
        if not active and pending is not None:
            chunk, event = pending
            pending = None
            active.extend(chunk.commands)
            event["activated_tick"] = tick
            result.boundaries[tick] = True

    tick = 0
    dataset_tick = 0
    stalled_ticks = 0
    while tick < steps if config.replay_clock == "wall_clock" else dataset_tick < steps:
        ensure_capacity(tick + 1)
        result.dataset_ticks[tick] = dataset_tick
        current_task = str(task_at(dataset_tick))
        if task != current_task:
            task = current_task
            session += 1
            ensemble.reset()
            queue.clear()
            active.clear()
            pending = None
            last = None
            latency_history.clear()
            next_query = tick
        result.segments[tick] = session
        deliver(tick)
        if config.mode in ("synchronous", "double_buffer"):
            activate(tick)

        if (
            flight is None
            and tick >= next_query
            and (config.mode != "double_buffer" or pending is None)
            and (config.mode != "synchronous" or (not active and pending is None))
        ):
            prefix = queue.get_left_over(("offline", session), task)
            prefix_length = 0 if prefix is None else min(x.shape[1] for x in prefix.values())
            delay_estimate = min(
                math.ceil(max(latency_history, default=0) * config.fps),
                prefix_length,
                config.rtc_guidance_horizon,
            )
            options = None
            if config.mode == "rtc" and prefix is not None:
                options = {
                    "rtc": {
                        "prefix_actions": prefix,
                        "prefix_length": prefix_length,
                        "estimated_delay_steps": delay_estimate,
                        "guidance_horizon": min(config.rtc_guidance_horizon, prefix_length),
                        "prefix_schedule": config.rtc_prefix_schedule,
                        "max_guidance_weight": config.rtc_max_guidance_weight,
                    }
                }
            observation = observation_at(dataset_tick)
            started = time.perf_counter()
            actions, _ = infer(observation, options, dataset_tick)
            elapsed = time.perf_counter() - started
            actions = layout.validate_chunk(actions, config.action_horizon)
            delay = (
                math.ceil(elapsed * config.fps)
                if config.measured_latency
                else config.latency_ticks[tick % len(config.latency_ticks)]
            )
            commands = layout.commands(actions)
            if config.mode in ("synchronous", "double_buffer"):
                commands = commands[: config.execution_horizon]
            elif config.mode == "temporal_ensemble":
                commands = commands[: config.ensemble_horizon or config.action_horizon]
            action_tick = dataset_tick if config.replay_clock == "execution_clock" else tick
            chunk = ActionChunk(
                stream_id="offline",
                control_session=session,
                observation_sequence=tick,
                observation_received_at=tick / config.fps,
                inference_seconds=elapsed,
                start_tick=action_tick,
                commands=commands,
                physical_actions=actions,
                task=task,
                estimated_delay_steps=delay_estimate,
            )
            event = {
                "query_tick": tick,
                "query_dataset_tick": dataset_tick,
                "ready_tick": tick + delay,
                "latency_ticks": delay,
                "inference_seconds": elapsed,
                "effective_latency_seconds": elapsed
                if config.measured_latency
                else delay / config.fps,
                "prefix_length": prefix_length,
                "estimated_delay_steps": delay_estimate,
                "guidance": options is not None,
                "session": session,
            }
            result.events.append(event)
            result.chunks.append(actions)
            flight = (tick + delay, chunk, event, prefix is not None)
            next_query = tick + config.query_interval
            deliver(tick)
            if config.mode in ("synchronous", "double_buffer"):
                activate(tick)

        if config.mode == "temporal_ensemble":
            action_tick = dataset_tick if config.replay_clock == "execution_clock" else tick
            command, count = ensemble.get_action(action_tick)
        elif config.mode in ("synchronous", "double_buffer"):
            command = active.popleft() if active else None
            count = int(command is not None)
        else:
            command = queue.pop()
            count = int(command is not None)
        result.contributors[tick] = count
        if command is not None:
            last = command
            result.status[tick] = 1
        elif last is not None:
            if config.mode != "double_buffer":
                last = layout.safe_hold(last)
            result.status[tick] = 2
        if last is not None:
            result.actions[tick] = layout.flatten(last)
        if config.replay_clock == "wall_clock":
            dataset_tick = tick + 1
        elif result.status[tick] == 1:
            dataset_tick += 1
            stalled_ticks = 0
        else:
            stalled_ticks += 1
            max_stall = max(1000, 10 * (config.action_horizon + max(config.latency_ticks)))
            if stalled_ticks > max_stall:
                raise RuntimeError(
                    "Execution-clock replay made no dataset progress; the strategy cannot "
                    "produce an executable chunk under this latency/horizon configuration"
                )
        tick += 1

    result.actions = result.actions[:tick]
    result.status = result.status[:tick]
    result.boundaries = result.boundaries[:tick]
    result.contributors = result.contributors[:tick]
    result.segments = result.segments[:tick]
    result.dataset_ticks = result.dataset_ticks[:tick]
    return result
