"""Physical-unit trajectory metrics; differences never bridge missing ticks/tasks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from evaluation.execution import ActionLayout, ReplayResult


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def percentile_abs(values, percentile=95):
    return float(np.percentile(np.abs(values), percentile)) if values.size else None


def difference_mask(valid, segments, order):
    mask = valid.copy()
    for _ in range(order):
        mask = mask[1:] & mask[:-1]
    if order:
        mask &= segments[order:] == segments[:-order]
    return mask


def lag_diagnostic(prediction, truth, valid, segments, max_lag):
    """Velocity correlation on a common support for every candidate lag.

    Positive lag means prediction follows GT late. This diagnostic never changes
    the alignment used for the accuracy metrics. Flat trajectories return null.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    pred = np.diff(prediction, axis=0)
    gt = np.diff(truth, axis=0)
    derivative_valid = difference_mask(valid, segments, 1)
    indices = np.arange(max_lag, len(pred) - max_lag)
    if not len(indices):
        return None, None
    support = derivative_valid[indices].copy()
    for lag in range(-max_lag, max_lag + 1):
        support &= derivative_valid[indices + lag]
        support &= segments[indices] == segments[indices + lag]
    indices = indices[support]
    if len(indices) < 3:
        return None, None
    reference = gt[indices] - gt[indices].mean(axis=0)
    if np.linalg.norm(reference) < 1e-10:
        return None, None
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = pred[indices + lag]
        shifted = shifted - shifted.mean(axis=0)
        denominator = np.linalg.norm(reference) * np.linalg.norm(shifted)
        if denominator > 1e-10:
            candidates.append((float(np.sum(reference * shifted) / denominator), lag))
    if not candidates:
        return None, None
    correlation, lag = max(candidates, key=lambda item: (item[0], -abs(item[1])))
    return lag, correlation


def trajectory_metrics(
    result: ReplayResult,
    truth: np.ndarray,
    layout: ActionLayout,
    fps: float,
    *,
    mask: np.ndarray | None = None,
    warmup_ticks: int = 0,
    max_lag: int = 10,
):
    if (
        truth.ndim != 2
        or truth.shape[1:] != result.actions.shape[1:]
        or not np.isfinite(truth).all()
    ):
        raise ValueError("GT must be finite and match the executed action dimensions")
    dataset_ticks = (
        np.arange(len(result.actions), dtype=np.int64)
        if result.dataset_ticks is None
        else np.asarray(result.dataset_ticks)
    )
    if (
        dataset_ticks.shape != (len(result.actions),)
        or np.any(dataset_ticks < 0)
        or np.any(dataset_ticks >= len(truth))
    ):
        raise ValueError("Replay dataset ticks are missing or outside the GT trajectory")
    truth = truth[dataset_ticks]
    if not np.isfinite(fps) or fps <= 0 or warmup_ticks < 0:
        raise ValueError("Invalid fps or warmup_ticks")
    valid = np.isfinite(result.actions).all(axis=1)
    if mask is not None:
        if mask.shape != valid.shape:
            raise ValueError("Metric mask has the wrong shape")
        valid &= mask
    valid[:warmup_ticks] = False
    groups = {}
    for group, columns in layout.slices.items():
        pred = result.actions[:, columns]
        gt = truth[:, columns]
        error = (pred - gt)[valid]
        values = {
            "samples": int(valid.sum()),
            "mae": float(np.mean(np.abs(error))) if error.size else None,
            "rmse": rms(error),
            "p95_abs_error": percentile_abs(error),
        }
        for order in (1, 2, 3):
            pd = np.diff(pred, n=order, axis=0) * fps**order
            gd = np.diff(gt, n=order, axis=0) * fps**order
            dv = difference_mask(valid, result.segments, order)
            values[f"d{order}_samples"] = int(dv.sum())
            values[f"d{order}_rmse"] = rms((pd - gd)[dv])
            values[f"d{order}_rms"] = rms(pd[dv])
            values[f"gt_d{order}_rms"] = rms(gd[dv])
            values[f"d{order}_p95"] = percentile_abs(pd[dv])
            values[f"gt_d{order}_p95"] = percentile_abs(gd[dv])
        boundary = difference_mask(valid, result.segments, 1) & result.boundaries[1:]
        delta = np.diff(pred, axis=0)
        gt_delta = np.diff(gt, axis=0)
        values["boundary_samples"] = int(boundary.sum())
        values["boundary_jump_rms"] = rms(delta[boundary])
        values["boundary_jump_p95"] = percentile_abs(delta[boundary])
        values["gt_boundary_jump_rms"] = rms(gt_delta[boundary])
        values["boundary_delta_rmse"] = rms((delta - gt_delta)[boundary])
        lag, correlation = lag_diagnostic(pred, gt, valid, result.segments, max_lag)
        values["lag_ticks"] = lag
        values["lag_seconds"] = None if lag is None else lag / fps
        values["lag_correlation"] = correlation
        groups[group] = values

    events = result.events
    delivered = [event for event in events if "delivered_tick" in event]
    first = np.flatnonzero(result.status)
    groups["__execution__"] = {
        "wall_ticks": len(result.actions),
        "wall_seconds": len(result.actions) / fps,
        "dataset_steps_reached": int(np.unique(dataset_ticks[result.status == 1]).size),
        "unavailable_fraction": float(np.mean(result.status == 0)),
        "hold_fraction": float(np.mean(result.status == 2)),
        "prediction_fraction": float(np.mean(result.status == 1)),
        "first_command_tick": int(first[0]) if first.size else None,
        "queries": len(events),
        "delivered_chunks": len(delivered),
        "expired_fraction": (
            float(np.mean([event.get("expired", False) for event in delivered]))
            if delivered
            else None
        ),
        "inference_seconds_mean": (
            float(np.mean([event["inference_seconds"] for event in events])) if events else None
        ),
        "inference_seconds_p95": (
            float(np.percentile([event["inference_seconds"] for event in events], 95))
            if events
            else None
        ),
        "latency_ticks_mean": (
            float(np.mean([event["latency_ticks"] for event in events])) if events else None
        ),
    }
    return groups


def summarize(records, baseline="double_buffer", seed=0, bootstrap_samples=2000):
    """Average repeats per episode, then bootstrap paired *episodes*, not frames."""
    buckets = defaultdict(lambda: defaultdict(list))
    for record in records:
        for group, metrics in record["common"].items():
            for metric, value in metrics.items():
                if value is not None:
                    key = (record["scenario"], record["mode"], group, metric)
                    buckets[key][record["episode_id"]].append(float(value))
    means = {
        key: {episode: float(np.mean(values)) for episode, values in episodes.items()}
        for key, episodes in buckets.items()
    }
    rng = np.random.default_rng(seed)

    def ci(values):
        if len(values) < 2:
            return None, None
        draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
        return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))

    rows = []
    for (scenario, mode, group, metric), episodes in sorted(means.items()):
        values = list(episodes.values())
        low, high = ci(values)
        base = means.get((scenario, baseline, group, metric), {})
        delta = [value - base[episode] for episode, value in episodes.items() if episode in base]
        delta_low, delta_high = ci(delta)
        rows.append(
            {
                "scenario": scenario,
                "mode": mode,
                "group": group,
                "metric": metric,
                "episodes": len(values),
                "mean": float(np.mean(values)),
                "ci95_low": low,
                "ci95_high": high,
                "baseline": baseline,
                "paired_episodes": len(delta),
                "paired_delta": float(np.mean(delta)) if delta else None,
                "paired_ci95_low": delta_low,
                "paired_ci95_high": delta_high,
            }
        )
    return rows
