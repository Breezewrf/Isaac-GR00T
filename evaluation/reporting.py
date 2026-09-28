"""CSV summaries and static plots of commands, derivatives and tradeoffs."""

import csv
import re

import numpy as np


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def save_comparison_plot(output, truth, results, layout, fps):
    plt = pyplot()
    gt_time = np.arange(len(truth)) / fps
    for group_index, (group, columns) in enumerate(layout.slices.items()):
        # A page per dimension keeps multi-DoF arms readable without dropping channels.
        for dimension in range(columns.start, columns.stop):
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for order, ax in enumerate(axes):
                ax.plot(
                    gt_time[order:],
                    np.diff(truth[:, dimension], n=order) * fps**order,
                    color="black",
                    label="GT",
                    linewidth=1.5,
                )
                for mode, result in results.items():
                    time = np.arange(len(result.actions)) / fps
                    values = np.diff(result.actions[:, dimension], n=order) * fps**order
                    if order:
                        values[result.segments[order:] != result.segments[:-order]] = np.nan
                    (line,) = ax.plot(time[order:], values, label=mode, alpha=0.8)
                    if order == 0:
                        updates = result.boundaries & np.isfinite(result.actions[:, dimension])
                        ax.scatter(
                            time[updates],
                            result.actions[updates, dimension],
                            color=line.get_color(),
                            s=12,
                        )
                        holds = result.status == 2
                        ax.scatter(
                            time[holds],
                            result.actions[holds, dimension],
                            color=line.get_color(),
                            marker="x",
                            s=18,
                        )
                ax.set_ylabel("command" if order == 0 else f"d{order} / s^{order}")
                ax.grid(alpha=0.2)
            axes[0].legend(ncol=3)
            axes[-1].set_xlabel("Dataset time (s); dots: chunk updates, crosses: held commands")
            fig.suptitle(f"{group}[{dimension - columns.start}] — recorded-observation replay")
            fig.tight_layout()
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", group)
            fig.savefig(
                output / f"trajectory_{group_index}_{safe_name}_{dimension - columns.start}.png",
                dpi=130,
            )
            plt.close(fig)


def save_summary(output, rows, *, plots=True):
    if not rows:
        return
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not plots:
        return
    plt = pyplot()
    from matplotlib.lines import Line2D

    lookup = {
        (row["scenario"], row["mode"], row["group"], row["metric"]): row["mean"] for row in rows
    }
    groups = sorted({row["group"] for row in rows} - {"__execution__"})
    modes = sorted({row["mode"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    marker_choices = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    markers = {mode: marker_choices[i % len(marker_choices)] for i, mode in enumerate(modes)}
    colors = {scenario: plt.get_cmap("tab10")(i % 10) for i, scenario in enumerate(scenarios)}
    legend = [
        Line2D(
            [],
            [],
            marker=markers[mode],
            color="black",
            markerfacecolor="none",
            linestyle="",
            label=mode,
        )
        for mode in modes
    ] + [
        Line2D([], [], marker="o", color=colors[scenario], linestyle="", label=scenario)
        for scenario in scenarios
    ]
    # Generic derivative labels avoid calling d3 of a velocity command "jerk".
    for index, group in enumerate(groups):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, derivative in zip(axes, ("d2_rms", "d3_rms"), strict=True):
            for scenario, mode, candidate_group, metric in lookup:
                if candidate_group != group or metric != "rmse":
                    continue
                y = lookup.get((scenario, mode, group, derivative))
                if y is not None:
                    x = lookup[scenario, mode, group, metric]
                    ax.scatter(
                        x,
                        y,
                        marker=markers[mode],
                        edgecolors=colors[scenario],
                        facecolors="none",
                        s=65,
                        linewidths=1.5,
                    )
            ax.set_xlabel("Action RMSE (common valid ticks)")
            ax.set_ylabel(derivative)
            ax.grid(alpha=0.2)
        fig.suptitle(f"{group}: accuracy vs command derivatives (episode means)")
        fig.legend(
            handles=legend, loc="center right", title="Shape: method / Color: delay", fontsize=9
        )
        fig.tight_layout(rect=(0, 0, 0.79, 0.95))
        fig.savefig(output / f"tradeoff_{index}.png", dpi=140)
        plt.close(fig)
