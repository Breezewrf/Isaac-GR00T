#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure frame-to-frame action smoothness in a LeRobot v2.x dataset.

The script computes ``delta_a[t] = action[t] - action[t - 1]`` independently
inside each episode, so episode boundaries never create artificial jumps.
Thresholds are scale-dependent and should be interpreted in the action units.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeMovement:
    episode_index: int
    frame_count: int
    average_delta_l2: float
    mean_abs_delta: float


@dataclass(frozen=True)
class DimensionSmoothness:
    index: int
    name: str
    sigma: float
    mean_abs_delta: float
    p95_abs_delta: float
    max_abs_delta: float
    classification: str


@dataclass(frozen=True)
class JumpEvent:
    episode_index: int
    from_frame: int
    to_frame: int
    delta_l2: float
    largest_dimension: str
    largest_abs_delta: float


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _action_names(info: dict[str, Any], action_dim: int) -> list[str]:
    names = info.get("features", {}).get("action", {}).get("names")
    if names is None:
        return [f"action[{index}]" for index in range(action_dim)]
    if len(names) != action_dim:
        raise ValueError(
            f"info.json declares {len(names)} action names, but action dimension is {action_dim}"
        )
    return [str(name) for name in names]


def _episode_path(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    chunk_size = int(info["chunks_size"])
    return dataset_path / info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        chunk_index=episode_index // chunk_size,
        file_index=episode_index,
    )


def _load_actions(path: Path) -> np.ndarray:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required; run this script through the project environment with `uv run`"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Episode parquet does not exist: {path}")
    frame = pd.read_parquet(path, columns=["action"])
    if frame.empty:
        raise ValueError(f"Episode parquet has no frames: {path}")
    actions = np.stack(frame["action"].to_numpy()).astype(np.float64, copy=False)
    if actions.ndim != 2:
        raise ValueError(
            f"Action array in {path} has shape {actions.shape}, expected (frames, dim)"
        )
    if not np.isfinite(actions).all():
        bad_count = int((~np.isfinite(actions)).sum())
        raise ValueError(f"Action array in {path} contains {bad_count} non-finite values")
    return actions


def classify_dimension(
    sigma: float,
    max_abs_delta: float,
    *,
    smooth_threshold: float,
    jerky_threshold: float,
    inactive_epsilon: float,
) -> str:
    if max_abs_delta <= inactive_epsilon:
        return "inactive/discrete"
    if sigma < smooth_threshold:
        return "smooth"
    if sigma < jerky_threshold:
        return "moderate"
    return "jerky"


def _sample_episode_indices(
    episode_indices: list[int],
    max_episodes: int,
    seed: int,
) -> list[int]:
    if max_episodes < 0 or max_episodes >= len(episode_indices):
        return episode_indices
    if max_episodes == 0:
        raise ValueError("--max-episodes must be positive or -1 for all episodes")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(episode_indices, size=max_episodes, replace=False)
    return sorted(int(index) for index in chosen)


def analyze_dataset(
    dataset_path: Path,
    *,
    max_episodes: int,
    seed: int,
    smooth_threshold: float,
    jerky_threshold: float,
    inactive_epsilon: float,
    top_jumps: int,
) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    info = _read_json(info_path)
    episode_records = _read_jsonl(episodes_path)
    all_episode_indices = [int(record["episode_index"]) for record in episode_records]
    episode_indices = _sample_episode_indices(all_episode_indices, max_episodes, seed)

    all_deltas: list[np.ndarray] = []
    movements: list[EpisodeMovement] = []
    jump_candidates: list[JumpEvent] = []
    names: list[str] | None = None
    action_dim: int | None = None

    for episode_index in episode_indices:
        actions = _load_actions(_episode_path(dataset_path, info, episode_index))
        if action_dim is None:
            action_dim = actions.shape[1]
            names = _action_names(info, action_dim)
        elif actions.shape[1] != action_dim:
            raise ValueError(
                f"Episode {episode_index} action dimension {actions.shape[1]} does not match "
                f"the first sampled episode dimension {action_dim}"
            )

        if len(actions) < 2:
            movements.append(EpisodeMovement(episode_index, len(actions), 0.0, 0.0))
            continue

        deltas = np.diff(actions, axis=0)
        all_deltas.append(deltas)
        delta_l2 = np.linalg.norm(deltas, axis=1)
        movements.append(
            EpisodeMovement(
                episode_index=episode_index,
                frame_count=len(actions),
                average_delta_l2=float(delta_l2.mean()),
                mean_abs_delta=float(np.abs(deltas).mean()),
            )
        )

        if top_jumps > 0:
            candidate_count = min(top_jumps, len(delta_l2))
            candidate_rows = np.argpartition(delta_l2, -candidate_count)[-candidate_count:]
            for row in candidate_rows:
                abs_delta = np.abs(deltas[row])
                dimension = int(abs_delta.argmax())
                jump_candidates.append(
                    JumpEvent(
                        episode_index=episode_index,
                        from_frame=int(row),
                        to_frame=int(row + 1),
                        delta_l2=float(delta_l2[row]),
                        largest_dimension=names[dimension],
                        largest_abs_delta=float(abs_delta[dimension]),
                    )
                )

    if not all_deltas or action_dim is None or names is None:
        raise ValueError("No sampled episode contains at least two action frames")

    combined = np.concatenate(all_deltas, axis=0)
    abs_combined = np.abs(combined)
    sigmas = combined.std(axis=0)
    max_abs = abs_combined.max(axis=0)
    dimensions = [
        DimensionSmoothness(
            index=index,
            name=names[index],
            sigma=float(sigmas[index]),
            mean_abs_delta=float(abs_combined[:, index].mean()),
            p95_abs_delta=float(np.quantile(abs_combined[:, index], 0.95)),
            max_abs_delta=float(max_abs[index]),
            classification=classify_dimension(
                float(sigmas[index]),
                float(max_abs[index]),
                smooth_threshold=smooth_threshold,
                jerky_threshold=jerky_threshold,
                inactive_epsilon=inactive_epsilon,
            ),
        )
        for index in range(action_dim)
    ]
    counts = {
        label: sum(item.classification == label for item in dimensions)
        for label in ("smooth", "moderate", "jerky", "inactive/discrete")
    }
    active_verdict = "jerky" if counts["jerky"] else "moderate" if counts["moderate"] else "smooth"
    lowest = sorted(movements, key=lambda item: item.average_delta_l2)
    jumps = sorted(jump_candidates, key=lambda item: item.delta_l2, reverse=True)[:top_jumps]

    return {
        "dataset_path": str(dataset_path),
        "sampled_episode_count": len(episode_indices),
        "sampled_episode_indices": episode_indices,
        "action_dimension": action_dim,
        "delta_frame_count": len(combined),
        "thresholds": {
            "smooth_sigma_below": smooth_threshold,
            "jerky_sigma_at_or_above": jerky_threshold,
            "inactive_max_abs_delta_at_or_below": inactive_epsilon,
        },
        "overall": active_verdict,
        "classification_counts": counts,
        "episode_movement": [asdict(item) for item in movements],
        "lowest_movement_episodes": [asdict(item) for item in lowest],
        "dimensions": [asdict(item) for item in dimensions],
        "largest_jump_events": [asdict(item) for item in jumps],
    }


def _print_report(report: dict[str, Any], lowest_count: int) -> None:
    print(f"Action Smoothness ({report['sampled_episode_count']} episodes sampled)")
    print(f"Dataset: {report['dataset_path']}")
    print(f"Action dimensions: {report['action_dimension']}")
    print(f"Frame-to-frame deltas: {report['delta_frame_count']}")

    print("\nLowest-Movement Episodes")
    print("average_delta_l2 is the mean L2 norm of action[t] - action[t-1].")
    for episode in report["lowest_movement_episodes"][:lowest_count]:
        print(
            f"  ep {episode['episode_index']:>4}: "
            f"avg ||delta_a||2={episode['average_delta_l2']:.6f}, "
            f"mean |delta_a|={episode['mean_abs_delta']:.6f}, "
            f"frames={episode['frame_count']}"
        )

    print("\nAction Velocity (delta_a) -- Smoothness Proxy")
    for item in report["dimensions"]:
        print(
            f"  [{item['index']:>2}] {item['name']}: "
            f"sigma={item['sigma']:.6f}, "
            f"mean|delta|={item['mean_abs_delta']:.6f}, "
            f"p95|delta|={item['p95_abs_delta']:.6f}, "
            f"max|delta|={item['max_abs_delta']:.6f} "
            f"[{item['classification']}]"
        )

    counts = report["classification_counts"]
    print(f"\nOverall: {report['overall'].upper()}")
    print(
        f"  {counts['smooth']} smooth, {counts['moderate']} moderate, "
        f"{counts['jerky']} jerky, {counts['inactive/discrete']} inactive/discrete"
    )

    if report["largest_jump_events"]:
        print("\nLargest frame-to-frame jumps")
        for event in report["largest_jump_events"]:
            print(
                f"  ep {event['episode_index']} frame {event['from_frame']}->{event['to_frame']}: "
                f"||delta_a||2={event['delta_l2']:.6f}, "
                f"largest={event['largest_dimension']} "
                f"({event['largest_abs_delta']:.6f})"
            )

    print(
        "\nNote: thresholds are action-unit dependent. Inspect the reported jump frames in "
        "video before flagging or removing an episode."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=-1,
        help="Number of randomly sampled episodes; -1 analyzes all episodes (default: -1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lowest-count", type=int, default=10)
    parser.add_argument("--top-jumps", type=int, default=10)
    parser.add_argument(
        "--smooth-threshold",
        type=float,
        default=0.005,
        help="A dimension is smooth when std(delta_a) is below this value.",
    )
    parser.add_argument(
        "--jerky-threshold",
        type=float,
        default=0.01,
        help="A dimension is jerky when std(delta_a) is at or above this value.",
    )
    parser.add_argument("--inactive-epsilon", type=float, default=1e-8)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.smooth_threshold < 0:
        parser.error("--smooth-threshold must be non-negative")
    if args.jerky_threshold <= args.smooth_threshold:
        parser.error("--jerky-threshold must be greater than --smooth-threshold")
    if args.inactive_epsilon < 0:
        parser.error("--inactive-epsilon must be non-negative")
    if args.lowest_count < 0 or args.top_jumps < 0:
        parser.error("--lowest-count and --top-jumps must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = analyze_dataset(
        args.dataset_path,
        max_episodes=args.max_episodes,
        seed=args.seed,
        smooth_threshold=args.smooth_threshold,
        jerky_threshold=args.jerky_threshold,
        inactive_epsilon=args.inactive_epsilon,
        top_jumps=args.top_jumps,
    )
    _print_report(report, args.lowest_count)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nSaved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
