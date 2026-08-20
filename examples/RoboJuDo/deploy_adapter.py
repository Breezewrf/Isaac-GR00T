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
    joint_names: tuple[str, ...]
    arm_width: int


PROFILES = {
    "g1_23dof": RobotProfile(
        joint_names=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ),
        arm_width=5,
    ),
    "x2": RobotProfile(
        joint_names=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ),
        arm_width=7,
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
        split = self.profile.arm_width
        return {
            "video": {"head_view": image[None, None]},
            "state": {
                "left_arm": positions[:split][None, None],
                "right_arm": positions[split:][None, None],
            },
            "language": {"task": [[instruction]]},
        }

    def decode_action_chunk(
        self, action_chunk: Mapping[str, np.ndarray], execution_horizon: int
    ) -> list[dict[str, Any]]:
        required = {
            "left_arm": self.profile.arm_width,
            "right_arm": self.profile.arm_width,
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
        if not 1 <= execution_horizon <= available_horizon:
            raise ValueError(
                f"execution_horizon must be in [1, {available_horizon}], got {execution_horizon}"
            )

        commands = []
        for step in range(execution_horizon):
            arm_positions = np.concatenate(
                (arrays["left_arm"][0, step], arrays["right_arm"][0, step])
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
                        zip(self.profile.joint_names, arm_positions.tolist(), strict=True)
                    ),
                    "locomotion_command": locomotion_command,
                }
            )
        return commands

    def get_action(
        self,
        image: np.ndarray,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        instruction: str,
        *,
        execution_horizon: int = 8,
    ) -> list[dict[str, Any]]:
        observation = self.build_observation(image, joint_positions, instruction)
        action_chunk, _ = self.policy_client.get_action(observation)
        return self.decode_action_chunk(action_chunk, execution_horizon)
