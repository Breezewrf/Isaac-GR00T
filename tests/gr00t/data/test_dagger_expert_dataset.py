"""DAgger sampling keeps only starts with a fully expert-executed action chunk."""

from types import SimpleNamespace
from unittest.mock import patch

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
import numpy as np
import pandas as pd
import pytest


def _dataset(tmp_path, episodes, *, action_indices=(0, 1, 2)):
    for episode_id, flags in enumerate(episodes):
        path = tmp_path / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "expert_applied": flags,
                "action_source": ["expert" if flag else "policy" for flag in flags],
            }
        ).to_parquet(path)
    loader = SimpleNamespace(
        episodes_metadata=[
            {"episode_index": i, "length": len(flags)} for i, flags in enumerate(episodes)
        ],
        episode_lengths=[len(flags) for flags in episodes],
        data_path_pattern="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        chunk_size=1000,
        get_episode_length=lambda index: len(episodes[index]),
    )
    modalities = {"action": SimpleNamespace(delta_indices=list(action_indices))}
    with patch(
        "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader", return_value=loader
    ):
        return ShardedSingleStepDataset(
            dataset_path=tmp_path,
            embodiment_tag=SimpleNamespace(value="new_embodiment"),
            modality_configs=modalities,
            episode_sampling_rate=1.0,
            dagger_expert_only=True,
        )


def test_dagger_starts_require_all_expert_actions_and_stay_in_episode(tmp_path):
    dataset = _dataset(
        tmp_path, [[False, True, True, True, False, True, True], [True, True, True, True]]
    )

    np.testing.assert_array_equal(dataset.get_expert_step_indices(0), [1])
    np.testing.assert_array_equal(dataset.get_expert_step_indices(1), [0, 1])
    assert sum(dataset.shard_lengths) == 3


def test_dagger_starts_follow_configured_action_indices(tmp_path):
    dataset = _dataset(tmp_path, [[True, False, True, True]], action_indices=(0, 2))

    np.testing.assert_array_equal(dataset.get_expert_step_indices(0), [0])


def test_dagger_rejects_dataset_without_full_expert_chunk(tmp_path):
    with pytest.raises(AssertionError, match="No valid expert action chunks"):
        _dataset(tmp_path, [[True, True, False, True]])


def test_dagger_rejects_mismatched_action_labels(tmp_path):
    dataset = _dataset(tmp_path, [[True, True, True]])
    path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "action_source"] = "policy"
    frame.to_parquet(path)

    with pytest.raises(ValueError, match="disagree"):
        dataset.get_expert_step_indices(0)
