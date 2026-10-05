# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test DatasetFactory: dataset construction and statistics generation.

DatasetFactory.build() depends heavily on Config, distributed utilities, and
real LeRobot datasets. We test the parts that can be isolated:
- merge_statistics (already tested in test_sharded_datasets.py)
- Factory instantiation with mock config
- Build pipeline with mocked dependencies
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_config():
    """Create a minimal mock training Config."""
    config = MagicMock()
    config.training.eval_strategy = "no"
    config.data.mode = "single_turn"
    config.data.shard_size = 128
    config.data.episode_sampling_rate = 0.5
    config.data.seed = 42
    config.data.allow_padding = False
    config.data.dagger_expert_only = False
    config.data.reuse_pretraining_statistics = False
    config.data.explicit_mixture_weights = False
    config.data.ds_weights_alpha = None
    config.data.num_shards_per_epoch = 100
    config.data.override_pretraining_statistics = False

    # Single dataset spec
    dataset_spec = MagicMock()
    dataset_spec.dataset_paths = ["/fake/dataset_path"]
    dataset_spec.embodiment_tag = "new_embodiment"
    dataset_spec.mix_ratio = 1.0
    dataset_spec.repo_id = None
    dataset_spec.dagger_expert_only = False
    config.data.datasets = [dataset_spec]

    config.data.modality_configs = {
        "new_embodiment": {
            "video": MagicMock(delta_indices=[0], modality_keys=["cam"]),
            "state": MagicMock(delta_indices=[0], modality_keys=["x"]),
            "action": MagicMock(delta_indices=list(range(4)), modality_keys=["x"]),
            "language": MagicMock(delta_indices=[0], modality_keys=["task"]),
        }
    }
    return config


class TestDatasetFactory:
    """Test factory construction."""

    def test_init(self):
        from gr00t.data.dataset.factory import DatasetFactory

        config = _make_mock_config()
        factory = DatasetFactory(config)
        assert factory.config is config

    def test_build_creates_mixture_dataset(self):
        from gr00t.data.dataset.factory import DatasetFactory

        config = _make_mock_config()
        factory = DatasetFactory(config)
        mock_processor = MagicMock()
        mock_processor.set_statistics = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.shard_lengths = np.full(10, 100)
        mock_dataset.get_shard_length = MagicMock(return_value=100)
        mock_dataset.embodiment_tag = type("ET", (), {"value": "new_embodiment"})()
        mock_dataset.get_dataset_statistics.return_value = {
            "state": {
                "x": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.2],
                    "q01": [0.05],
                    "q99": [0.95],
                }
            },
            "action": {
                "x": {
                    "min": [-1.0],
                    "max": [1.0],
                    "mean": [0.0],
                    "std": [0.3],
                    "q01": [-0.9],
                    "q99": [0.9],
                }
            },
        }

        with (
            patch("gr00t.data.dataset.factory.generate_stats"),
            patch("gr00t.data.dataset.factory.generate_rel_stats"),
            patch("gr00t.data.dataset.factory.ShardedSingleStepDataset", return_value=mock_dataset),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            train_ds, eval_ds = factory.build(mock_processor)

        assert train_ds is not None
        assert eval_ds is None

    def test_build_rejects_eval_strategy(self):
        from gr00t.data.dataset.factory import DatasetFactory

        config = _make_mock_config()
        config.training.eval_strategy = "steps"
        factory = DatasetFactory(config)
        with pytest.raises(AssertionError, match="does not support evaluation"):
            factory.build(MagicMock())

    def test_dagger_build_skips_recomputing_statistics(self):
        from gr00t.data.dataset.factory import DatasetFactory

        config = _make_mock_config()
        config.data.dagger_expert_only = True
        config.data.override_pretraining_statistics = False
        dataset = MagicMock()
        dataset.__len__.return_value = 1
        with (
            patch("gr00t.data.dataset.factory.generate_stats") as generate_stats,
            patch("gr00t.data.dataset.factory.generate_rel_stats") as generate_rel_stats,
            patch(
                "gr00t.data.dataset.factory.ShardedSingleStepDataset", return_value=dataset
            ) as dataset_cls,
            patch("gr00t.data.dataset.factory.ShardedMixtureDataset") as mixture_cls,
        ):
            DatasetFactory(config).build(MagicMock())

        generate_stats.assert_not_called()
        generate_rel_stats.assert_not_called()
        assert dataset_cls.call_args.kwargs["dagger_expert_only"] is True
        assert mixture_cls.call_args.kwargs["reuse_pretraining_statistics"] is True
