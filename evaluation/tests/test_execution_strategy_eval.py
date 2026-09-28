from dataclasses import replace
import json
from types import SimpleNamespace

from evaluation.execution import ActionLayout, ReplayConfig, ReplayResult, replay
from evaluation.execution_strategy_eval import (
    observation_reader,
    parser,
    run_evaluation,
    validate_timebase,
)
from evaluation.metrics import lag_diagnostic, summarize, trajectory_metrics
import numpy as np
import pandas as pd
import pytest


def config(mode, **kwargs):
    return ReplayConfig(
        mode, fps=10, action_horizon=6, execution_horizon=3, rtc_guidance_horizon=4, **kwargs
    )


def run(mode, *, steps=14, settings=None, task_at=None, infer=None):
    layout = ActionLayout({"joint": 1, "navigate_command": 1}, ["navigate_command"])
    calls = []

    def predict(obs, options, tick):
        calls.append((tick, options))
        values = np.arange(tick, tick + 6, dtype=np.float32)[None, :, None]
        return {"joint": values, "navigate_command": np.ones_like(values)}, {}

    result = replay(
        infer or predict,
        lambda t: {"tick": t},
        task_at or (lambda t: "task"),
        steps,
        layout,
        settings or config(mode),
    )
    return result, layout, calls


def test_te_aligns_predictions_to_query_tick_and_excludes_future_returns():
    result, _, calls = run(
        "temporal_ensemble", settings=config("temporal_ensemble", latency_ticks=(2,))
    )
    assert np.isnan(result.actions[:2]).all()
    np.testing.assert_allclose(result.actions[2:, 0], np.arange(2, 14), atol=2e-6)
    assert [tick for tick, _ in calls] == list(range(0, 14, 2))
    assert result.contributors[2] == 1
    assert result.contributors[4] == 2


def test_rtc_prefix_is_unexecuted_physical_prediction_and_delays_are_skipped():
    result, _, calls = run("rtc", settings=config("rtc", latency_ticks=(2,)))
    assert calls[0] == (0, None)
    tick, options = calls[1]
    assert tick == 2
    rtc = options["rtc"]
    # At the start of tick 2 the newly received startup chunk is still unconsumed.
    np.testing.assert_array_equal(rtc["prefix_actions"]["joint"], np.arange(6)[None, :, None])
    assert rtc["prefix_length"] == 6
    assert rtc["estimated_delay_steps"] == 2
    assert rtc["guidance_horizon"] == 4
    np.testing.assert_allclose(result.actions[2:6, 0], [0, 1, 4, 5])
    assert result.events[1]["skipped_steps"] == 2
    assert result.events[0]["skipped_steps"] == 0


def test_rtc_schedule_no_guidance_has_rtc_schedule_without_guidance():
    rtc, _, _ = run("rtc", settings=config("rtc", latency_ticks=(2,)))
    direct, _, calls = run(
        "rtc_schedule_no_guidance",
        settings=config("rtc_schedule_no_guidance", latency_ticks=(2,)),
    )
    np.testing.assert_allclose(rtc.actions, direct.actions)
    assert all(options is None for _, options in calls)
    assert [event["query_tick"] for event in rtc.events] == [
        event["query_tick"] for event in direct.events
    ]


def test_double_buffer_keeps_current_chunk_and_one_pending_chunk():
    result, _, calls = run("double_buffer", steps=8)
    np.testing.assert_allclose(result.actions[:, 0], [0, 1, 2, 1, 2, 3, 3, 4])
    assert [tick for tick, _ in calls] == [0, 1, 3, 6]
    assert result.events[1]["activated_tick"] == 3


def test_synchronous_waits_until_chunk_finishes_before_next_inference():
    result, _, calls = run(
        "synchronous", steps=12, settings=config("synchronous", latency_ticks=(2,))
    )
    assert [tick for tick, _ in calls] == [0, 5, 10]
    np.testing.assert_array_equal(result.status, [0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2])
    np.testing.assert_allclose(
        result.actions[:, 0], [np.nan, np.nan, 0, 1, 2, 2, 2, 5, 6, 7, 7, 7], equal_nan=True
    )
    assert [(event["query_tick"], event.get("activated_tick")) for event in result.events] == [
        (0, 2),
        (5, 7),
        (10, None),
    ]


def test_zero_latency_synchronous_has_no_hold_gap():
    result, _, calls = run(
        "synchronous", steps=8, settings=config("synchronous", latency_ticks=(0,))
    )
    assert [tick for tick, _ in calls] == [0, 3, 6]
    np.testing.assert_array_equal(result.status, np.ones(8, dtype=np.int8))
    np.testing.assert_allclose(result.actions[:, 0], [0, 1, 2, 3, 4, 5, 6, 7])


def test_execution_clock_pauses_dataset_during_sync_waits_and_finishes_episode():
    result, layout, calls = run(
        "synchronous",
        steps=8,
        settings=config("synchronous", latency_ticks=(2,), replay_clock="execution_clock"),
    )
    assert [tick for tick, _ in calls] == [0, 3, 6]
    np.testing.assert_array_equal(result.dataset_ticks, [0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7])
    np.testing.assert_array_equal(result.status, [0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1])
    executed = result.status == 1
    np.testing.assert_allclose(result.actions[executed, 0], np.arange(8))
    truth = np.column_stack([np.arange(8), np.ones(8)])
    metrics = trajectory_metrics(result, truth, layout, 10)
    assert metrics["__execution__"]["wall_ticks"] == 14
    assert metrics["__execution__"]["dataset_steps_reached"] == 8
    assert metrics["joint"]["samples"] == 12  # startup unavailable; holds remain wall samples


def test_execution_clock_async_advances_after_startup_without_holds():
    result, _, calls = run(
        "temporal_ensemble",
        steps=8,
        settings=config("temporal_ensemble", latency_ticks=(2,), replay_clock="execution_clock"),
    )
    assert len(result.actions) == 10
    np.testing.assert_array_equal(result.dataset_ticks, [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(result.status, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    assert [tick for tick, _ in calls] == [0, 0, 2, 4, 6]


@pytest.mark.parametrize("mode", ["rtc", "rtc_schedule_no_guidance", "temporal_ensemble"])
def test_starvation_holds_position_but_stops_velocity(mode):
    result, _, _ = run(mode, settings=config(mode, query_interval=20))
    np.testing.assert_array_equal(result.status[6:], 2)
    np.testing.assert_allclose(result.actions[6:, 0], 5)
    np.testing.assert_allclose(result.actions[6:, 1], 0)


def test_double_buffer_starvation_keeps_velocity_like_deployment():
    result, _, _ = run("double_buffer", settings=config("double_buffer", query_interval=20))
    np.testing.assert_array_equal(result.status[3:], 2)
    np.testing.assert_allclose(result.actions[3:, 1], 1)


def test_expired_rtc_chunk_clears_queue_and_recovery_has_no_prefix():
    delays = (0, 7, 0, 0, 0, 0, 0, 0, 0)
    result, _, calls = run("rtc", settings=config("rtc", latency_ticks=delays))
    assert result.events[1]["expired"] is True
    assert result.events[1]["skipped_steps"] == 7
    assert calls[2] == (8, None)
    assert result.status[6] == 2
    assert result.events[2]["skipped_steps"] == 0


def test_task_change_discards_inflight_prediction_and_old_hold():
    result, _, calls = run(
        "rtc",
        settings=config("rtc", latency_ticks=(3,)),
        task_at=lambda t: "old" if t < 2 else "new",
    )
    assert result.events[0]["discarded_task_change"] is True
    assert np.isnan(result.actions[:6]).all()
    assert calls[1] == (3, None)
    assert result.segments[1] != result.segments[2]


def test_measured_latency_uses_policy_call_duration(monkeypatch):
    times = iter([0, 0.21, 1, 1.21])
    monkeypatch.setattr("evaluation.execution.time.perf_counter", lambda: next(times))
    result, _, _ = run("rtc", steps=4, settings=config("rtc", measured_latency=True))
    assert result.events[0]["ready_tick"] == 3
    assert result.events[1]["estimated_delay_steps"] == 3
    np.testing.assert_array_equal(result.status, [0, 0, 0, 1])


def test_invalid_policy_chunk_fails_before_execution():
    def bad(obs, options, tick):
        return {"joint": np.zeros((1, 5, 1))}, {}

    with pytest.raises(ValueError, match="action groups"):
        run("rtc", infer=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency_ticks": ()},
        {"latency_ticks": (-1,)},
        {"latency_ticks": (1.5,)},
        {"latency_ticks": (True,)},
        {"fps": 0},
        {"rtc_guidance_horizon": 7},
        {"query_interval": 0},
        {"temporal_ensemble_coeff": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(config("rtc"), **kwargs)


def metric_result(actions, segments=None):
    actions = np.asarray(actions, dtype=float).reshape(-1, 1)
    length = len(actions)
    return ReplayResult(
        actions,
        np.ones(length, dtype=np.int8),
        np.ones(length, dtype=bool),
        np.ones(length, dtype=int),
        np.ones(length, dtype=int) if segments is None else np.asarray(segments),
        [],
        [],
    )


def test_physical_derivatives_and_gt_reference():
    time = np.arange(12) / 10
    result = metric_result(time**3)
    metrics = trajectory_metrics(result, result.actions.copy(), ActionLayout({"joint": 1}), 10)[
        "joint"
    ]
    assert metrics["rmse"] == 0
    assert metrics["d1_rmse"] == 0
    assert metrics["d3_rms"] == pytest.approx(6)
    assert metrics["gt_d3_rms"] == pytest.approx(6)


def test_derivatives_do_not_bridge_missing_ticks_or_task_changes():
    result = metric_result([0, 1, np.nan, 100, 101, 200, 201], [1, 1, 1, 1, 1, 2, 2])
    truth = np.array([0, 1, 50, 100, 101, 200, 201], dtype=float)[:, None]
    metrics = trajectory_metrics(result, truth, ActionLayout({"joint": 1}), 1)["joint"]
    assert metrics["samples"] == 6
    assert metrics["d1_samples"] == 3
    assert metrics["d1_rms"] == 1
    assert metrics["d2_samples"] == 0
    assert metrics["d2_rms"] is None
    assert metrics["boundary_jump_rms"] == 1


def test_common_mask_and_warmup_apply_without_time_compression():
    result = metric_result([0, 1, 2, 3, 4, 5])
    mask = np.array([True, True, True, False, True, True])
    metrics = trajectory_metrics(
        result, result.actions.copy(), ActionLayout({"joint": 1}), 1, mask=mask, warmup_ticks=2
    )["joint"]
    assert metrics["samples"] == 3
    assert metrics["d1_samples"] == 1


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_short_unavailable_episode_has_null_metrics_and_full_unavailability(steps):
    result, layout, _ = run("rtc", steps=steps, settings=config("rtc", latency_ticks=(20,)))
    metrics = trajectory_metrics(result, np.zeros_like(result.actions), layout, 10)
    assert metrics["joint"]["samples"] == 0
    assert metrics["joint"]["rmse"] is None
    assert metrics["joint"]["d3_rms"] is None
    assert metrics["__execution__"]["unavailable_fraction"] == 1
    assert metrics["__execution__"]["first_command_tick"] is None
    assert metrics["__execution__"]["delivered_chunks"] == 0


def test_lag_sign_and_static_trajectory():
    gt = np.random.default_rng(7).normal(size=(100, 1)).cumsum(axis=0)
    pred = np.roll(gt, 3, axis=0)
    valid = np.ones(100, dtype=bool)
    segments = np.ones(100, dtype=int)
    lag, correlation = lag_diagnostic(pred, gt, valid, segments, 6)
    assert lag == 3
    assert correlation == pytest.approx(1)
    assert lag_diagnostic(np.ones_like(gt), np.ones_like(gt), valid, segments, 6) == (None, None)


def test_bootstrap_uses_paired_episode_means_not_repeat_count():
    records = []
    for episode, baseline in [(0, 10), (1, 20)]:
        for repeat in range(3):
            for mode, value in [("double_buffer", baseline), ("rtc", baseline - 2)]:
                records.append(
                    {
                        "scenario": "d1",
                        "episode_id": episode,
                        "repeat": repeat,
                        "mode": mode,
                        "common": {"joint": {"rmse": value}},
                    }
                )
    row = next(row for row in summarize(records) if row["mode"] == "rtc")
    assert row["episodes"] == 2
    assert row["paired_delta"] == -2
    assert row["paired_ci95_low"] == row["paired_ci95_high"] == -2


def modalities():
    from gr00t.data.types import ModalityConfig

    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
        "state": ModalityConfig(delta_indices=[-1, 0], modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=list(range(6)), modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def frame():
    return pd.DataFrame(
        {
            "timestamp": np.arange(12) / 10,
            "state.joint": [np.array([tick], dtype=np.float32) for tick in range(12)],
            "action.joint": [np.array([tick], dtype=np.float32) for tick in range(12)],
            "video.ego_view": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(12)],
            "language.task": ["pick up"] * 12,
        }
    )


def test_observation_history_pads_start_and_never_reads_gt():
    from gr00t.data.embodiment_tags import EmbodimentTag

    read, _ = observation_reader(frame(), modalities(), EmbodimentTag.NEW_EMBODIMENT)
    obs = read(0)
    np.testing.assert_array_equal(obs["state"]["joint"], [[[0], [0]]])
    assert "action" not in obs
    assert obs["video"]["ego_view"].shape == (1, 1, 4, 4, 3)


def test_future_observations_are_rejected():
    from gr00t.data.embodiment_tags import EmbodimentTag

    configs = modalities()
    configs["state"].delta_indices = [1]
    with pytest.raises(ValueError, match="Future state"):
        observation_reader(frame(), configs, EmbodimentTag.NEW_EMBODIMENT)


def test_nonuniform_timestamps_are_rejected():
    data = frame()
    data.loc[3, "timestamp"] = 0.4
    with pytest.raises(ValueError, match="timestamps"):
        validate_timebase(data, 10)


def test_end_to_end_report_with_fake_policy(tmp_path):
    from gr00t.data.embodiment_tags import EmbodimentTag

    class Dataset:
        info_meta = {"fps": 10}
        modality_configs = modalities()
        episodes_metadata = [{"episode_index": 42}]

        def __getitem__(self, index):
            assert index == 0
            return frame()

    resets = []

    def predict(obs, options=None):
        assert "action" not in obs
        tick = obs["state"]["joint"][0, -1, 0]
        return {"joint": np.arange(tick, tick + 6, dtype=np.float32)[None, :, None]}, {}

    policy = SimpleNamespace(reset=lambda: resets.append(True), get_action=predict)
    args = parser().parse_args(
        [
            "--dataset-path",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "report"),
            "--episode-ids",
            "42",
            "--latency-ticks",
            "0",
            "2",
            "--max-lag-ticks",
            "2",
        ]
    )
    output = run_evaluation(args, policy, Dataset(), EmbodimentTag.NEW_EMBODIMENT)
    records = json.loads((output / "metrics.json").read_text())
    assert len(records) == len(resets) == 10
    te = next(
        record
        for record in records
        if record["mode"] == "temporal_ensemble" and record["scenario"] == "delay_2"
    )
    assert te["common"]["joint"]["rmse"] == pytest.approx(0, abs=2e-6)
    assert te["common"]["joint"]["samples"] == 10
    arrays = np.load(output / "episode_000042/delay_2/repeat_000/rtc.npz")
    assert arrays["chunk.joint"].shape[1:] == (6, 1)
    assert arrays["actions"].shape == (12, 1)
    assert (output / "summary.csv").is_file()
    assert (output / "tradeoff_0.png").is_file()
    assert len(list(output.rglob("trajectory_*.png"))) == 2
    with pytest.raises(FileExistsError):
        run_evaluation(args, policy, Dataset(), EmbodimentTag.NEW_EMBODIMENT)

    clock_args = parser().parse_args(
        [
            "--dataset-path",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "execution-clock-report"),
            "--episode-ids",
            "42",
            "--latency-ticks",
            "2",
            "--modes",
            "synchronous",
            "--replay-clock",
            "execution_clock",
            "--no-plots",
        ]
    )
    clock_output = run_evaluation(clock_args, policy, Dataset(), EmbodimentTag.NEW_EMBODIMENT)
    clock_record = json.loads((clock_output / "metrics.json").read_text())[0]
    assert clock_record["common"]["joint"]["samples"] == 12
    assert clock_record["common"]["joint"]["rmse"] == 0
    assert clock_record["common"]["__execution__"]["wall_ticks"] == 16
    arrays = np.load(clock_output / "episode_000042/delay_2/repeat_000/synchronous.npz")
    np.testing.assert_array_equal(
        arrays["dataset_ticks"], [0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11]
    )
    assert not (clock_output / "episode_000042/delay_2/repeat_000/common_mask.npy").exists()
