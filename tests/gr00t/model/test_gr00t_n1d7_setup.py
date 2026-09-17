# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from gr00t.configs.base_config import Config
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
import torch


class _FakeModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(1))


def test_checkpoint_loading_overrides_rtc_config(tmp_path, monkeypatch):
    config = Config()
    config.training.start_from_checkpoint = "base-checkpoint"
    config.model.training_time_rtc = True
    config.model.rtc_max_delay_steps = 6
    config.model.rtc_condition_prob = 0.25

    fake_model = _FakeModel(config.model)
    from_pretrained = Mock(return_value=(fake_model, {}))
    monkeypatch.setattr("gr00t.model.gr00t_n1d7.setup.AutoModel.from_pretrained", from_pretrained)

    pipeline = Gr00tN1d7Pipeline(config, tmp_path)
    assert pipeline._create_model() is fake_model

    loading_kwargs = from_pretrained.call_args.kwargs
    assert loading_kwargs["training_time_rtc"] is True
    assert loading_kwargs["rtc_max_delay_steps"] == 6
    assert loading_kwargs["rtc_condition_prob"] == 0.25
