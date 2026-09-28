"""Evaluate synchronous and asynchronous execution strategies on LeRobot episodes.

Run from the repository root: python -m evaluation.execution_strategy_eval --help
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np

from evaluation.execution import MODES, ActionLayout, ReplayConfig, replay
from evaluation.metrics import summarize, trajectory_metrics
from evaluation.reporting import save_comparison_plot, save_summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-path", type=Path, required=True)
    result.add_argument(
        "--model-path", help="Local GR00T checkpoint; otherwise use a policy server"
    )
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=5555)
    result.add_argument("--embodiment-tag", default="new_embodiment")
    result.add_argument("--device", default="cuda")
    result.add_argument("--denoising-steps", type=int, default=4, help="Local policy only")
    result.add_argument(
        "--episode-ids", type=int, nargs="+", help="Metadata episode IDs; default: all"
    )
    result.add_argument(
        "--steps", type=int, default=0, help="Maximum ticks per episode; 0: full episode"
    )
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument(
        "--execution-horizon",
        type=int,
        help="Synchronous/double-buffer execution length; default: min(8, chunk)",
    )
    result.add_argument(
        "--ensemble-horizon", type=int, help="TE valid chunk length; default: full chunk"
    )
    result.add_argument(
        "--rtc-guidance-horizon", type=int, help="RTC guidance window; default: min(8, chunk)"
    )
    result.add_argument(
        "--query-interval", type=int, default=1, help="Minimum ticks between queries"
    )
    result.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    result.add_argument(
        "--rtc-prefix-schedule", choices=("zeros", "ones", "linear", "exp"), default="exp"
    )
    result.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    result.add_argument("--rtc-latency-window", type=int, default=10)
    latency = result.add_mutually_exclusive_group()
    latency.add_argument(
        "--latency-ticks", type=int, nargs="+", help="Fixed-delay sweep; default: 0 1 2 4 8"
    )
    latency.add_argument(
        "--latency-trace",
        type=Path,
        help="JSON list of delay ticks, cyclically indexed by query tick",
    )
    latency.add_argument(
        "--measured-latency",
        action="store_true",
        help="Use each method's measured get_action duration",
    )
    result.add_argument(
        "--control-fps", type=float, help="Must equal dataset fps; no implicit resampling"
    )
    result.add_argument(
        "--replay-clock",
        choices=("wall_clock", "execution_clock"),
        default="wall_clock",
        help="Advance dataset every wall tick, or only when a predicted action executes",
    )
    result.add_argument(
        "--velocity-groups",
        nargs="*",
        help="Velocity commands zeroed on TE/RTC starvation; default: navigate_command if present",
    )
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--warmup-ticks",
        type=int,
        default=0,
        help="Exclude this initial interval from accuracy/smoothness only",
    )
    result.add_argument("--max-lag-ticks", type=int, default=10)
    result.add_argument(
        "--output-dir", type=Path, help="New output directory (must not already exist)"
    )
    result.add_argument("--no-plots", action="store_true")
    return result


def validate_args(args):
    if args.steps < 0 or args.warmup_ticks < 0 or args.max_lag_ticks < 0:
        raise ValueError("steps, warmup-ticks and max-lag-ticks must be nonnegative")
    if args.repeats < 1 or args.denoising_steps < 1 or args.seed < 0:
        raise ValueError("repeats/denoising-steps must be positive and seed nonnegative")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("Duplicate execution modes")
    if args.episode_ids is not None and len(args.episode_ids) != len(set(args.episode_ids)):
        raise ValueError("Duplicate episode IDs")


def latency_scenarios(args):
    if args.measured_latency:
        return {"measured": (0,)}
    if args.latency_trace is not None:
        trace = json.loads(args.latency_trace.read_text())
        if not isinstance(trace, list):
            raise ValueError("Latency trace must be a JSON list of nonnegative integer ticks")
        return {"trace": tuple(trace)}
    values = args.latency_ticks if args.latency_ticks is not None else [0, 1, 2, 4, 8]
    if len(values) != len(set(values)):
        raise ValueError("Duplicate fixed latency values")
    return {f"delay_{delay}": (delay,) for delay in values}


def validate_timebase(frame, fps):
    if "timestamp" in frame:
        timestamps = frame["timestamp"].to_numpy(dtype=float)
        if not np.isfinite(timestamps).all() or not np.allclose(
            np.diff(timestamps), 1 / fps, rtol=0.01, atol=1e-6
        ):
            raise ValueError("Episode timestamps are not uniformly sampled at dataset fps")


def observation_reader(frame, modalities, embodiment):
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    observation_modalities = {key: value for key, value in modalities.items() if key != "action"}
    language_keys = observation_modalities["language"].modality_keys
    if len(language_keys) != 1 or list(observation_modalities["language"].delta_indices) != [0]:
        raise ValueError("Replay requires one language modality sampled at the current tick")
    for key, config in observation_modalities.items():
        if any(delta > 0 for delta in config.delta_indices):
            raise ValueError(f"Future {key} observations would leak information into replay")

    def read(tick):
        # Explicit left-edge padding prevents negative iloc indices from reading
        # the *end* of the episode when a checkpoint uses observation history.
        step = extract_step_data(
            frame, tick, observation_modalities, embodiment, allow_padding=True
        )
        observation = {
            "video": {key: np.stack(images)[None] for key, images in step.images.items()},
            "state": {key: values[None] for key, values in step.states.items()},
            "language": {language_keys[0]: [[step.text]]},
        }
        if step.masks:
            observation["mask"] = {
                key: np.stack(values)[None] for key, values in step.masks.items()
            }
        return observation

    tasks = frame[f"language.{language_keys[0]}"].tolist()
    return read, lambda tick: tasks[tick]


def seeded_inference(policy, local, seed, episode_id):
    def infer(observation, options, tick):
        if local:
            import torch

            # Common noise for the same episode/repeat/query tick across methods.
            request_seed = int(
                np.random.SeedSequence([seed, episode_id, tick]).generate_state(1)[0]
            )
            random.seed(request_seed)
            np.random.seed(request_seed)
            torch.manual_seed(request_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request_seed)
        return policy.get_action(observation, options=options)

    return infer


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_evaluation(args, policy, dataset, embodiment, *, local=False):
    """Run with a local/remote policy and an existing LeRobotEpisodeLoader."""
    from gr00t.eval._horizon_contract import PolicyHorizonSpec

    validate_args(args)
    fps = float(dataset.info_meta["fps"])
    if args.control_fps is not None and not np.isclose(args.control_fps, fps):
        raise ValueError(
            "control-fps must equal dataset fps; resample the dataset explicitly first"
        )
    modalities = dataset.modality_configs
    horizon = PolicyHorizonSpec.from_modality_config(modalities).action_horizon
    scenarios = latency_scenarios(args)
    configs = {
        (scenario, mode): ReplayConfig(
            mode=mode,
            fps=fps,
            action_horizon=horizon,
            execution_horizon=args.execution_horizon
            if args.execution_horizon is not None
            else min(8, horizon),
            ensemble_horizon=args.ensemble_horizon,
            rtc_guidance_horizon=args.rtc_guidance_horizon
            if args.rtc_guidance_horizon is not None
            else min(8, horizon),
            query_interval=args.query_interval,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            rtc_prefix_schedule=args.rtc_prefix_schedule,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_latency_window=args.rtc_latency_window,
            latency_ticks=latencies,
            measured_latency=args.measured_latency,
            replay_clock=args.replay_clock,
        )
        for scenario, latencies in scenarios.items()
        for mode in args.modes
    }
    episode_indices = {meta["episode_index"]: i for i, meta in enumerate(dataset.episodes_metadata)}
    selected = list(episode_indices) if args.episode_ids is None else args.episode_ids
    if not selected or set(selected) - episode_indices.keys():
        raise ValueError(
            "No episodes selected or requested episode IDs are absent from the dataset"
        )
    output = args.output_dir or Path("evaluation/results") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "fps": fps,
        "episode_ids": selected,
        "local_policy_seeding": local,
        "configs": {
            f"{scenario}/{mode}": asdict(config) for (scenario, mode), config in configs.items()
        },
        "scope": "Recorded-observation replay of physical commands; no dynamics or closed-loop success measurement.",
        "replay_clock": args.replay_clock,
        "startup": "No command before the first prediction. Missing ticks remain NaN in NPZ and count as unavailable.",
    }
    write_json(output / "manifest.json", manifest)
    records = []
    for episode_id in selected:
        frame = dataset[episode_indices[episode_id]]
        if args.steps:
            frame = frame.iloc[: args.steps]
        if frame.empty:
            raise ValueError(f"Episode {episode_id} is empty")
        validate_timebase(frame, fps)
        truth_groups = {
            key: np.vstack(frame[f"action.{key}"].to_numpy()).astype(np.float64)
            for key in modalities["action"].modality_keys
        }
        truth = np.concatenate(list(truth_groups.values()), axis=-1)
        if not np.isfinite(truth).all():
            raise ValueError(f"Episode {episode_id} contains non-finite GT actions")
        velocity_groups = args.velocity_groups
        if velocity_groups is None:
            velocity_groups = ["navigate_command"] if "navigate_command" in truth_groups else []
        layout = ActionLayout(
            {key: value.shape[1] for key, value in truth_groups.items()}, velocity_groups
        )
        read_observation, read_task = observation_reader(frame, modalities, embodiment)
        episode_dir = output / f"episode_{episode_id:06d}"
        episode_dir.mkdir()
        write_json(
            episode_dir / "layout.json",
            {
                "groups": layout.widths,
                "velocity_groups": layout.velocity_groups,
                "derivative_units": "dN = action_units / seconds**N; velocity commands have acceleration at d1 and jerk at d2",
            },
        )
        np.savez_compressed(episode_dir / "gt.npz", actions=truth, time=np.arange(len(frame)) / fps)
        for scenario in scenarios:
            for repeat in range(args.repeats):
                results = {}
                run_dir = episode_dir / scenario / f"repeat_{repeat:03d}"
                run_dir.mkdir(parents=True)
                for mode in args.modes:
                    logging.info(
                        "Episode %s | %s | repeat %s | %s", episode_id, scenario, repeat, mode
                    )
                    policy.reset()
                    result = replay(
                        seeded_inference(policy, local, args.seed + repeat, episode_id),
                        read_observation,
                        read_task,
                        len(frame),
                        layout,
                        configs[scenario, mode],
                    )
                    results[mode] = result
                    np.savez_compressed(
                        run_dir / f"{mode}.npz",
                        actions=result.actions,
                        status=result.status,
                        boundaries=result.boundaries,
                        contributors=result.contributors,
                        segments=result.segments,
                        dataset_ticks=result.dataset_ticks,
                        query_ticks=np.asarray([event["query_tick"] for event in result.events]),
                        **{
                            f"chunk.{key}": np.stack([chunk[key][0] for chunk in result.chunks])
                            for key in layout.widths
                        },
                    )
                    write_json(run_dir / f"{mode}.events.json", result.events)
                if args.replay_clock == "wall_clock":
                    common = np.logical_and.reduce(
                        [np.isfinite(result.actions).all(axis=1) for result in results.values()]
                    )
                    np.save(run_dir / "common_mask.npy", common)
                else:
                    common = None
                for mode, result in results.items():
                    kwargs = {"warmup_ticks": args.warmup_ticks, "max_lag": args.max_lag_ticks}
                    if args.replay_clock == "execution_clock":
                        executed = result.status == 1
                        progress_ticks = result.dataset_ticks[executed]
                        if not np.array_equal(progress_ticks, np.arange(len(frame))):
                            raise RuntimeError(
                                f"{mode} did not execute exactly one action per dataset tick"
                            )
                        progress_result = type(result)(
                            actions=result.actions[executed],
                            status=result.status[executed],
                            boundaries=result.boundaries[executed],
                            contributors=result.contributors[executed],
                            segments=result.segments[executed],
                            events=result.events,
                            chunks=result.chunks,
                            dataset_ticks=progress_ticks,
                        )
                    else:
                        progress_result = result
                    available_metrics = trajectory_metrics(result, truth, layout, fps, **kwargs)
                    common_metrics = trajectory_metrics(
                        progress_result, truth, layout, fps, mask=common, **kwargs
                    )
                    # Execution availability and completion time always belong to
                    # the full wall-time replay, even when common task metrics are
                    # projected to one executed action per dataset tick.
                    common_metrics["__execution__"] = available_metrics["__execution__"]
                    record = {
                        "episode_id": episode_id,
                        "scenario": scenario,
                        "repeat": repeat,
                        "seed": args.seed + repeat if local else None,
                        "mode": mode,
                        "available": available_metrics,
                        "common": common_metrics,
                    }
                    records.append(record)
                    write_json(run_dir / f"{mode}.metrics.json", record)
                if not args.no_plots:
                    save_comparison_plot(run_dir, truth, results, layout, fps)
                write_json(output / "metrics.json", records)
                save_summary(output, summarize(records, seed=args.seed), plots=not args.no_plots)
    logging.info("Evaluation complete: %s", output)
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment = EmbodimentTag.resolve(args.embodiment_tag)
    local = args.model_path is not None
    if local:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=embodiment, model_path=args.model_path, device=args.device
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
    else:
        from gr00t.policy.server_client import PolicyClient

        logging.warning(
            "Remote policy RNG cannot be seeded through the current API; repeats are independent draws. Configure denoising steps on the server."
        )
        policy = PolicyClient(host=args.host, port=args.port)
    try:
        dataset = LeRobotEpisodeLoader(
            dataset_path=args.dataset_path, modality_configs=policy.get_modality_config()
        )
        run_evaluation(args, policy, dataset, embodiment, local=local)
    finally:
        if not local:
            policy.close()


if __name__ == "__main__":
    main()
