"""Source isolation, expert supervision, and configured sample fractions."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.dataset.finetune_data_config import (
    load_finetune_data_config,
    resolve_hub_source,
    validate_lerobot_source,
    validate_mixture_inputs,
)
import numpy as np
import pandas as pd
import pytest
import yaml


def _write_config(tmp_path, sources, *, normalization="inherit"):
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"normalization": normalization, "datasets": sources}))
    return path


def test_yaml_resolves_paths_and_preserves_hub_source(tmp_path):
    path = _write_config(
        tmp_path,
        [
            {"kind": "sft", "path": "sft", "weight": 8},
            {"kind": "dagger", "repo_id": "owner/dagger", "revision": "v2.1", "weight": 2},
        ],
    )
    data = load_finetune_data_config(str(path), "new_embodiment")
    sft, dagger = data["datasets"]
    assert sft["dataset_paths"] == [str(tmp_path / "sft")]
    assert not sft["dagger_expert_only"]
    assert dagger["dagger_expert_only"]
    assert dagger["revision"] == "v2.1"
    assert [sft["mix_ratio"], dagger["mix_ratio"]] == [0.8, 0.2]
    with patch("huggingface_hub.snapshot_download", return_value="snapshot") as download:
        assert resolve_hub_source("owner/dagger", "v2.1", local_files_only=True) == "snapshot"
    download.assert_called_once_with(
        repo_id="owner/dagger", repo_type="dataset", revision="v2.1", local_files_only=True
    )


@pytest.mark.parametrize(
    "change",
    [
        {"weight": 0},
        {"weight": -1},
        {"weight": float("nan")},
        {"weight": True},
        {"kind": "policy"},
        {"repo_id": "owner/dataset"},
        {"revision": "v2.1"},
        {"weigth": 1},
    ],
)
def test_yaml_rejects_ambiguous_sources_and_bad_weights(tmp_path, change):
    source = {"kind": "sft", "path": "sft", "weight": 1, **change}
    with pytest.raises(ValueError):
        load_finetune_data_config(str(_write_config(tmp_path, [source])), "new_embodiment")


def _make_source(root, *, flags=None, length=202):
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "robot_type": "g1",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (meta / "info.json").write_text(json.dumps(info))
    for name in ("episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json"):
        (meta / name).write_text("{}")
    if flags is not None:
        length = len(flags)
        data_path = root / "data/chunk-000/episode_000000.parquet"
        data_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {"expert_applied": flags, "action_source": ["expert" if f else "policy" for f in flags]}
        ).to_parquet(data_path)
    return SimpleNamespace(
        fps=10,
        info_meta=info,
        modality_meta={
            "state": {"x": {"start": 0, "end": 1}},
            "action": {"x": {"start": 0, "end": 1}},
        },
        episodes_metadata=[{"episode_index": 0, "length": length}],
        episode_lengths=[length],
        data_path_pattern=info["data_path"],
        chunk_size=1000,
        get_episode_length=lambda _: length,
    )


@pytest.mark.parametrize("hub", [False, True])
def test_runtime_mixture_filters_only_dagger_and_keeps_checkpoint_statistics(tmp_path, hub):
    sft = tmp_path / "sft"
    dagger = tmp_path / "dagger"
    loaders = {
        str(sft): _make_source(sft),
        str(dagger): _make_source(
            dagger, flags=[False, True, True, True, False, True, True, True, True]
        ),
    }
    path = _write_config(
        tmp_path,
        [
            {
                "kind": "sft",
                "weight": 0.8,
                **({"repo_id": "owner/sft", "revision": "v2.1"} if hub else {"path": "sft"}),
            },
            {"kind": "dagger", "path": "dagger", "weight": 0.2},
        ],
    )
    config = get_default_config().load_dict(
        {"data": load_finetune_data_config(str(path), "new_embodiment")}
    )
    modalities = {
        "state": SimpleNamespace(modality_keys=["x"], delta_indices=[0]),
        "action": SimpleNamespace(modality_keys=["x"], delta_indices=[0, 1, 2]),
    }
    config.data.modality_configs = {"new_embodiment": modalities}
    config.data.shard_size = 40
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 20000
    processor = MagicMock()
    processor.use_relative_action = False
    stats = {"new_embodiment": {"state": {"x": {"mean": [0.0]}}, "action": {"x": {"mean": [0.0]}}}}
    processor.state_action_processor.statistics = stats
    with (
        patch(
            "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader",
            side_effect=lambda dataset_path, **_: loaders[str(dataset_path)],
        ),
        patch("gr00t.data.dataset.factory.generate_stats") as generate_stats,
        patch("gr00t.data.dataset.factory.generate_rel_stats") as generate_rel_stats,
        patch(
            "gr00t.data.dataset.factory.resolve_hub_source", return_value=str(sft)
        ) as resolve_source,
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        mixture, _ = DatasetFactory(config).build(processor)
    assert not mixture.datasets[0].dagger_expert_only
    assert mixture.datasets[1].dagger_expert_only
    np.testing.assert_array_equal(mixture.datasets[1].get_expert_step_indices(0), [1, 5, 6])
    assert sum(mixture.datasets[0].shard_lengths) == 200
    assert sum(mixture.datasets[1].shard_lengths) == 3
    assert mixture.weights == [0.8, 0.2]
    assert mixture.global_stats["new_embodiment"] is stats["new_embodiment"]
    processor.set_statistics.assert_not_called()
    generate_stats.assert_not_called()
    generate_rel_stats.assert_not_called()
    if hub:
        assert resolve_source.call_count == 2
        resolve_source.assert_any_call("owner/sft", "v2.1", local_files_only=True)
    else:
        resolve_source.assert_not_called()
    counts = np.zeros(2)
    for index, shard in mixture.shard_sampling_schedule:
        counts[index] += mixture.datasets[index].get_shard_length(shard)
    assert abs(counts[1] / counts.sum() - 0.2) < 0.02
    # Both independent loaders retain episode zero; SFT has no intervention columns.
    assert all(
        ds.episode_loader.episodes_metadata[0]["episode_index"] == 0 for ds in mixture.datasets
    )


def test_v3_tag_or_layout_cannot_replace_actual_conversion(tmp_path):
    loader = _make_source(tmp_path / "source")
    path = tmp_path / "source/meta/info.json"
    loader.info_meta["codebase_version"] = "v3.0"
    path.write_text(json.dumps(loader.info_meta))
    with pytest.raises(ValueError, match="convert to v2.1"):
        validate_lerobot_source(str(tmp_path / "source"))


def test_incompatible_action_frequency_is_rejected(tmp_path):
    loader_a = _make_source(tmp_path / "a")
    loader_b = _make_source(tmp_path / "b")
    loader_b.fps = 30
    modalities = {
        "state": SimpleNamespace(modality_keys=["x"]),
        "action": SimpleNamespace(modality_keys=["x"]),
    }
    datasets = [
        SimpleNamespace(dataset_path=path, episode_loader=loader, modality_configs=modalities)
        for path, loader in (("a", loader_a), ("b", loader_b))
    ]
    with pytest.raises(ValueError, match="Mixture inputs differ"):
        validate_mixture_inputs(datasets)


def test_checkpoint_statistics_must_match_joint_group_dimensions(tmp_path):
    loader = _make_source(tmp_path / "source")
    modalities = {
        "state": SimpleNamespace(modality_keys=["x"]),
        "action": SimpleNamespace(modality_keys=["x"]),
    }
    dataset = SimpleNamespace(
        dataset_path="source",
        episode_loader=loader,
        modality_configs=modalities,
        embodiment_tag=SimpleNamespace(value="new_embodiment"),
    )
    processor = SimpleNamespace(
        state_action_processor=SimpleNamespace(
            statistics={"new_embodiment": {"state": {"x": {"mean": [0, 0]}}}}
        )
    )
    with pytest.raises(ValueError, match="Checkpoint statistics do not match"):
        validate_mixture_inputs([dataset], processor)


def test_missing_configured_camera_is_rejected(tmp_path):
    loader = _make_source(tmp_path / "source")
    loader._video_key_mapping = {}
    modalities = {
        "state": SimpleNamespace(modality_keys=["x"]),
        "action": SimpleNamespace(modality_keys=["x"]),
        "video": SimpleNamespace(modality_keys=["wrist"]),
    }
    dataset = SimpleNamespace(
        dataset_path="source", episode_loader=loader, modality_configs=modalities
    )
    with pytest.raises(ValueError, match="missing camera wrist"):
        validate_mixture_inputs([dataset])


@pytest.mark.parametrize("normalization", ["unknown", None, True])
def test_yaml_rejects_unknown_normalization(tmp_path, normalization):
    path = _write_config(
        tmp_path, [{"kind": "sft", "path": "demo", "weight": 1}], normalization=normalization
    )
    with pytest.raises(ValueError, match="normalization must be"):
        load_finetune_data_config(str(path), "new_embodiment")


def test_dataset_normalization_cannot_include_unfiltered_policy_statistics(tmp_path):
    path = _write_config(
        tmp_path, [{"kind": "dagger", "path": "rollout", "weight": 1}], normalization="recalculate"
    )
    with pytest.raises(ValueError, match="requires all sources to be kind: sft"):
        load_finetune_data_config(str(path), "new_embodiment")


def test_sft_mixture_generates_and_merges_statistics_by_sample_weights(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    loaders = {
        str(root): _make_source(root, length=length) for root, length in zip(roots, (202, 12))
    }

    def stats(mean, std):
        return {
            "mean": [mean],
            "std": [std],
            "min": [mean - 2 * std],
            "max": [mean + 2 * std],
            "q01": [mean - std],
            "q99": [mean + std],
        }

    for root, mean, std in zip(roots, (0, 10), (1, 2)):
        (root / "meta/stats.json").unlink()
        group = stats(mean, std)
        relative_group = {key: [value] * 3 for key, value in group.items()}
        loader_stats = {
            "state": {"x": group},
            "action": {"x": group},
            "relative_action": {"x": relative_group},
        }
        loaders[str(root)].get_dataset_statistics = lambda data=loader_stats: data
    path = _write_config(
        tmp_path,
        [
            {"kind": "sft", "path": "a", "weight": 0.6},
            {"kind": "sft", "path": "b", "weight": 0.4},
        ],
        normalization="recalculate",
    )
    config = get_default_config().load_dict(
        {"data": load_finetune_data_config(str(path), "new_embodiment")}
    )
    config.data.modality_configs = {
        "new_embodiment": {
            "state": SimpleNamespace(modality_keys=["x"], delta_indices=[0]),
            "action": SimpleNamespace(modality_keys=["x"], delta_indices=[0, 1, 2]),
        }
    }
    config.data.shard_size = 40
    config.data.num_shards_per_epoch = 10000
    processor = MagicMock()
    processor.state_action_processor.statistics = {}
    with (
        patch(
            "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader",
            side_effect=lambda dataset_path, **_: loaders[str(dataset_path)],
        ),
        patch("gr00t.data.dataset.factory.generate_stats") as generate_stats,
        patch("gr00t.data.dataset.factory.generate_rel_stats") as generate_rel_stats,
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        mixture, _ = DatasetFactory(config).build(processor)
    assert not mixture.reuse_pretraining_statistics
    assert mixture.weights == [0.6, 0.4]
    assert all(not dataset.dagger_expert_only for dataset in mixture.datasets)
    assert generate_stats.call_count == 2
    assert generate_rel_stats.call_count == 2
    for root in roots:
        generate_stats.assert_any_call(str(root))
    processor.set_statistics.assert_called_once_with(mixture.global_stats, override=True)
    actual = mixture.global_stats["new_embodiment"]
    for modality in ("state", "action"):
        np.testing.assert_allclose(actual[modality]["x"]["mean"], [4])
        np.testing.assert_allclose(actual[modality]["x"]["std"], [np.sqrt(26.2)])
    np.testing.assert_allclose(actual["relative_action"]["x"]["mean"], [[4]] * 3)
    np.testing.assert_allclose(actual["relative_action"]["x"]["std"], [[np.sqrt(26.2)]] * 3)
    counts = np.zeros(2)
    for index, shard in mixture.shard_sampling_schedule:
        counts[index] += mixture.datasets[index].get_shard_length(shard)
    assert abs(counts[1] / counts.sum() - 0.4) < 0.02
