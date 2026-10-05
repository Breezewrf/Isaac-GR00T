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

import numpy as np
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.finetune_data_config import (
    resolve_hub_source,
    validate_lerobot_source,
    validate_mixture_inputs,
)
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.utils.dist_utils import run_or_wait_on_rank0


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        all_datasets = []
        all_weights = []
        dagger_expert_only = self.config.data.dagger_expert_only
        reuse_statistics = (
            dagger_expert_only or self.config.data.reuse_pretraining_statistics is True
        )
        explicit_weights = self.config.data.explicit_mixture_weights is True
        if explicit_weights and self.config.data.ds_weights_alpha is not None:
            raise ValueError("Explicit mixture weights cannot be combined with ds_weights_alpha")
        if (
            dagger_expert_only
            and sum(len(spec.dataset_paths) for spec in self.config.data.datasets) != 1
        ):
            raise ValueError("DAgger expert-only training requires exactly one dataset")
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            dataset_paths = list(dataset_spec.dataset_paths)
            if getattr(dataset_spec, "repo_id", None) and not dataset_paths:
                with run_or_wait_on_rank0(label=f"download({dataset_spec.repo_id})") as is_rank0:
                    if is_rank0:
                        resolve_hub_source(dataset_spec.repo_id, dataset_spec.revision)
                dataset_paths = [
                    resolve_hub_source(
                        dataset_spec.repo_id, dataset_spec.revision, local_files_only=True
                    )
                ]
            expert_only = dagger_expert_only or dataset_spec.dagger_expert_only is True
            for dataset_path in dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                # rank-0 writes stats; helper barriers before peers read them.
                if explicit_weights:
                    validate_lerobot_source(dataset_path, require_statistics=reuse_statistics)
                if not reuse_statistics:
                    with run_or_wait_on_rank0(label=f"generate_stats({dataset_path})") as is_rank0:
                        if is_rank0:
                            generate_stats(dataset_path)
                            generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                    dagger_expert_only=expert_only,
                )
                datasets.append(dataset)
            dataset_lengths = np.array(
                [
                    sum(dataset.shard_lengths) if explicit_weights else len(dataset)
                    for dataset in datasets
                ]
            )
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        if explicit_weights:
            validate_mixture_inputs(all_datasets, processor if reuse_statistics else None)
            total_weight = sum(all_weights)
            for index, (dataset, weight) in enumerate(zip(all_datasets, all_weights)):
                print(
                    f"Mixture source {index}: {dataset.dataset_path}; "
                    f"expert_only={dataset.dagger_expert_only}; "
                    f"valid_windows={sum(dataset.shard_lengths)}; target_fraction={weight / total_weight:.6f}"
                )

        alpha = self.config.data.ds_weights_alpha
        if alpha is not None and len(all_datasets) > 1:
            ds_lengths = np.array([len(dataset) for dataset in all_datasets], dtype=np.float64)
            all_weights = (np.power(ds_lengths, alpha) / np.power(ds_lengths[0], alpha)).tolist()
            print(
                f"Applied ds_weights_alpha={alpha} across {len(all_datasets)} datasets; "
                "this overrides per-dataset mix_ratio sampling weights."
            )

        train_dataset = ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            reuse_pretraining_statistics=reuse_statistics,
        )
        if explicit_weights:
            scheduled_windows = np.zeros(len(all_datasets), dtype=np.int64)
            for dataset_index, shard_index in train_dataset.shard_sampling_schedule:
                scheduled_windows[dataset_index] += all_datasets[dataset_index].get_shard_length(
                    shard_index
                )
            fractions = scheduled_windows / scheduled_windows.sum()
            for index, fraction in enumerate(fractions):
                print(f"Mixture source {index}: scheduled_sample_fraction={fraction:.6f}")
        return train_dataset, None
