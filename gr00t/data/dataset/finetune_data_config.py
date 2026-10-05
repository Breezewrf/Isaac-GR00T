"""Opt-in SFT/DAgger mixtures without rewriting source datasets."""

import json
import math
from pathlib import Path

import yaml


def load_finetune_data_config(config_path: str, embodiment_tag: str) -> dict:
    """Read safe YAML; local paths are relative to the YAML file's directory."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {"normalization", "datasets"}:
        raise ValueError("Data YAML must contain only normalization and datasets")
    normalization = config["normalization"]
    if normalization not in ("inherit", "recalculate"):
        raise ValueError("normalization must be inherit or recalculate")
    sources = config["datasets"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("datasets must be a non-empty list")
    datasets = []
    for index, source in enumerate(sources):
        label = f"datasets[{index}]"
        if not isinstance(source, dict) or set(source) - {
            "kind",
            "path",
            "repo_id",
            "revision",
            "weight",
        }:
            raise ValueError(f"{label}: unknown fields or invalid source mapping")
        if source.get("kind") not in ("sft", "dagger"):
            raise ValueError(f"{label}: kind must be sft or dagger")
        if normalization == "recalculate" and source["kind"] == "dagger":
            raise ValueError(
                "normalization: recalculate requires all sources to be kind: sft; "
                "DAgger mixtures must use checkpoint statistics to exclude policy actions from normalization"
            )
        weight = source.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"{label}: weight must be a finite positive number")
        if ("path" in source) == ("repo_id" in source):
            raise ValueError(f"{label}: provide exactly one of path or repo_id")
        dataset = {
            "dataset_paths": [],
            "embodiment_tag": embodiment_tag,
            "mix_ratio": float(weight),
            "dagger_expert_only": source["kind"] == "dagger",
        }
        if "path" in source:
            if "revision" in source:
                raise ValueError(f"{label}: revision is only valid with repo_id")
            if not isinstance(source["path"], str) or not source["path"].strip():
                raise ValueError(f"{label}: path must be a non-empty string")
            path = Path(source["path"]).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            dataset["dataset_paths"] = [str(path.resolve())]
        else:
            repo_id = source["repo_id"]
            if (
                not isinstance(repo_id, str)
                or len(repo_id.split("/")) != 2
                or not all(repo_id.split("/"))
            ):
                raise ValueError(f"{label}: repo_id must be owner/dataset")
            revision = source.get("revision")
            if revision is not None and (not isinstance(revision, str) or not revision.strip()):
                raise ValueError(f"{label}: revision must be a non-empty string")
            dataset.update(repo_id=repo_id, revision=revision)
        datasets.append(dataset)
    total = sum(dataset["mix_ratio"] for dataset in datasets)
    if not math.isfinite(total):
        raise ValueError("Total dataset weight must be finite")
    for dataset in datasets:
        dataset["mix_ratio"] /= total
    return {
        "datasets": datasets,
        "download_cache": False,
        "explicit_mixture_weights": True,
        "reuse_pretraining_statistics": normalization == "inherit",
        "override_pretraining_statistics": normalization == "recalculate",
    }


def validate_lerobot_source(path: str, *, require_statistics: bool = True) -> None:
    """Reject v3 layouts instead of treating a Hub revision tag as conversion."""
    root = Path(path)
    filenames = ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"]
    if require_statistics:
        filenames.append("stats.json")
    for name in filenames:
        if not (root / "meta" / name).is_file():
            raise ValueError(
                f"{root}: missing meta/{name}; prepare a GR00T-compatible LeRobot v2.1 dataset"
            )
    with (root / "meta/info.json").open() as stream:
        info = json.load(stream)
    if info.get("codebase_version") not in ("v2.0", "v2.1"):
        raise ValueError(
            f"{root}: unsupported codebase_version {info.get('codebase_version')!r}; convert to v2.1 first"
        )
    if "{episode_index" not in info.get("data_path", ""):
        raise ValueError(
            f"{root}: data_path must address individual episodes; convert to v2.1 first"
        )


def resolve_hub_source(
    repo_id: str, revision: str | None, *, local_files_only: bool = False
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision, local_files_only=local_files_only
    )


def validate_mixture_inputs(datasets: list, processor=None) -> None:
    """Compare model-used inputs; DAgger bookkeeping columns may differ."""
    reference = None
    reference_path = None
    for dataset in datasets:
        loader = dataset.episode_loader
        signature = {"fps": loader.fps, "robot_type": loader.info_meta.get("robot_type")}
        for modality in ("state", "action"):
            groups = {}
            for key in dataset.modality_configs[modality].modality_keys:
                mapping = loader.modality_meta.get(modality, {}).get(key)
                if mapping is None:
                    raise ValueError(f"{dataset.dataset_path}: missing {modality} group {key}")
                groups[key] = mapping["end"] - mapping["start"]
            signature[modality] = groups
        if processor is not None:
            checkpoint_stats = getattr(processor.state_action_processor, "statistics", {})
            stats = checkpoint_stats.get(dataset.embodiment_tag.value, {})
            for modality in ("state", "action"):
                for key, width in signature[modality].items():
                    mean = stats.get(modality, {}).get(key, {}).get("mean")
                    if mean is None or len(mean) != width:
                        raise ValueError(
                            f"Checkpoint statistics do not match {dataset.dataset_path}: "
                            f"{modality}.{key} requires {width} dimensions"
                        )
        video_config = dataset.modality_configs.get("video")
        for key in video_config.modality_keys if video_config else ():
            meta_key = loader._video_key_mapping.get(key, key)
            mapping = loader.modality_meta.get("video", {}).get(meta_key)
            if mapping is None:
                raise ValueError(f"{dataset.dataset_path}: missing camera {key}")
            original_key = mapping.get("original_key", f"observation.images.{meta_key}")
            if original_key not in loader.feature_config:
                raise ValueError(f"{dataset.dataset_path}: missing camera feature {original_key}")
        if reference is not None and signature != reference:
            raise ValueError(
                f"Mixture inputs differ between {reference_path} and {dataset.dataset_path}: {reference} vs {signature}"
            )
        reference, reference_path = signature, dataset.dataset_path
