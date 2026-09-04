# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robojudo_x2_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    # left_hand/right_hand are intentionally omitted until X2 hand telemetry and
    # commands are connected. The deployment profile reserves those group names.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm"],
        sin_cos_embedding_keys=["left_arm", "right_arm"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "navigate_command",
            "base_height_command",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
}

register_modality_config(
    robojudo_x2_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
