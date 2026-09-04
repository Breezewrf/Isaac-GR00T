# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy client adapter for RoboJuDo X2 and G1 23-DoF control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gr00t.policy.server_client import PolicyClient
import numpy as np


@dataclass(frozen=True)
class RobotProfile:
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    # Reserved as empty tuples for robots whose dexterous hands are not wired yet.
    left_hand_joint_names: tuple[str, ...] = ()
    right_hand_joint_names: tuple[str, ...] = ()

    @property
    def joint_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Policy joint modalities that are active for this deployment profile."""
        groups = (
            ("left_arm", self.left_arm_joint_names),
            ("right_arm", self.right_arm_joint_names),
            ("left_hand", self.left_hand_joint_names),
            ("right_hand", self.right_hand_joint_names),
        )
        return tuple((key, names) for key, names in groups if names)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(name for _, names in self.joint_groups for name in names)


@dataclass(frozen=True)
class PolicyActionChunk:
    """Physical policy actions together with their RoboJuDo command encoding."""

    actions: dict[str, np.ndarray]  # For RTC prefix guidance
    commands: list[dict[str, Any]]  # For loop execution
    info: dict[str, Any]


PROFILES = {
    "g1_23dof": RobotProfile(
        left_arm_joint_names=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ),
        right_arm_joint_names=(
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ),
        left_hand_joint_names=(
            "left_thumb_proximal",
            "left_thumb_intermediate",
            "left_index_proximal",
            "left_middle_proximal",
            "left_ring_proximal",
            "left_pinky_proximal",
            "left_index_intermediate",
            "left_middle_intermediate",
            "left_ring_intermediate",
            "left_pinky_intermediate",
        ),
        right_hand_joint_names=(
            "right_thumb_proximal",
            "right_thumb_intermediate",
            "right_index_proximal",
            "right_middle_proximal",
            "right_ring_proximal",
            "right_pinky_proximal",
            "right_index_intermediate",
            "right_middle_intermediate",
            "right_ring_intermediate",
            "right_pinky_intermediate",
        ),
    ),
    "x2": RobotProfile(
        left_arm_joint_names=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
        ),
        right_arm_joint_names=(
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ),
        # X2 hand names stay empty until its observation/command transport is connected.
        left_hand_joint_names=(),
        right_hand_joint_names=(),
    ),
}


class RoboJuDoPolicyAdapter:
    """Translate RoboJuDo observations and GR00T action chunks without changing units."""

    def __init__(self, policy_client: PolicyClient, profile: str):
        self.policy_client = policy_client
        self.profile = PROFILES[profile]

    def _ordered_joint_positions(
        self, joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray
    ) -> np.ndarray:
        if isinstance(joint_positions, Mapping):
            missing = [name for name in self.profile.joint_names if name not in joint_positions]
            if missing:
                raise ValueError(f"Missing RoboJuDo joint positions: {missing}")
            values = [joint_positions[name] for name in self.profile.joint_names]
        else:
            values = joint_positions
        positions = np.asarray(values, dtype=np.float32)
        expected_shape = (len(self.profile.joint_names),)
        if positions.shape != expected_shape:
            raise ValueError(
                f"Joint positions have shape {positions.shape}, expected {expected_shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("Joint positions contain non-finite values")
        return positions

    def build_observation(
        self,
        image: np.ndarray,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        instruction: str,
    ) -> dict[str, Any]:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("image must be an HWC uint8 RGB array")
        if not instruction:
            raise ValueError("instruction must not be empty")
        positions = self._ordered_joint_positions(joint_positions)
        return {
            "video": {"ego_view": image[None, None]},
            "state": self._split_joint_groups(positions),
            "language": {"task": [[instruction]]},
        }

    def _split_joint_groups(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        groups = {}
        start = 0
        for key, names in self.profile.joint_groups:
            end = start + len(names)
            groups[key] = positions[start:end][None, None]
            start = end
        assert start == len(positions)
        return groups

    def _validate_action_chunk(
        self, action_chunk: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], int]:
        required = {
            **{key: len(names) for key, names in self.profile.joint_groups},
            "navigate_command": 3,
            "base_height_command": 1,
        }
        arrays = {}
        for key, width in required.items():
            if key not in action_chunk:
                raise ValueError(f"Policy response is missing action group {key!r}")
            value = np.asarray(action_chunk[key], dtype=np.float32)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
                raise ValueError(
                    f"Action group {key!r} has shape {value.shape}, expected (1, T, {width})"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"Action group {key!r} contains non-finite values")
            arrays[key] = value

        available_horizon = min(value.shape[1] for value in arrays.values())
        return arrays, available_horizon

    def decode_action_chunk(
        self, action_chunk: Mapping[str, np.ndarray], execution_horizon: int
    ) -> list[dict[str, Any]]:
        arrays, available_horizon = self._validate_action_chunk(action_chunk)
        if not 1 <= execution_horizon <= available_horizon:
            raise ValueError(
                f"execution_horizon must be in [1, {available_horizon}], got {execution_horizon}"
            )

        commands = []
        for step in range(execution_horizon):
            joint_positions = np.concatenate(
                [arrays[key][0, step] for key, _ in self.profile.joint_groups]
            )
            locomotion_command = np.concatenate(
                (
                    arrays["navigate_command"][0, step],
                    arrays["base_height_command"][0, step],
                )
            )
            commands.append(
                {
                    "positions": dict(
                        zip(self.profile.joint_names, joint_positions.tolist(), strict=True)
                    ),
                    "locomotion_command": locomotion_command,
                }
            )
        return commands

    def get_action_chunk(
        self,
        image: np.ndarray,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        instruction: str,
        *,
        execution_horizon: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PolicyActionChunk:
        """Return the physical chunk and commands; RTC uses all available steps."""
        observation = self.build_observation(image, joint_positions, instruction)

        # Inference, pass the options to the Policy client
        action_chunk, info = self.policy_client.get_action(observation, options=options)

        arrays, available_horizon = self._validate_action_chunk(action_chunk)
        # When using RTC, the execution_horizon is None, and we use all available steps.
        horizon = available_horizon if execution_horizon is None else execution_horizon
        actions = {key: value[:, :horizon].copy() for key, value in arrays.items()}
        return PolicyActionChunk(
            actions=actions,
            commands=self.decode_action_chunk(actions, horizon),
            info=info,
        )

    def get_action(
        self,
        image: np.ndarray,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        instruction: str,
        *,
        execution_horizon: int = 8,
    ) -> list[dict[str, Any]]:
        return self.get_action_chunk(  # Only need the commands except for RTC, which uses get_action_chunk() to get the actions for prefix guidance
            image,
            joint_positions,
            instruction,
            execution_horizon=execution_horizon,
        ).commands
