"""Plot reward curves from training run logs.

The script searches a runs directory recursively for metrics.jsonl files and
plots reward versus training step for each run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_REWARD_KEYS = (
    "episode_return_mean",
    "env_episode_return_mean",
    "shaped_rewards_mean",
    "raw_rewards_mean",
    "scaled_rewards_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward versus training step from runs/*/metrics.jsonl."
    )
    parser.add_argument(
        "runs_dirs",
        nargs="*",
        type=Path,
        help="Root runs directories, single run directories, or metrics.jsonl files. Default: runs",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Reward metric to plot. Default: first available of "
            f"{', '.join(DEFAULT_REWARD_KEYS)}"
        ),
    )
    parser.add_argument(
        "--step-key",
        default="global_step",
        help="Metric key to use for the x-axis. Default: global_step",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window size. Use 1 to disable smoothing. Default: 1",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reward_curve.png"),
        help="Output image path. Default: reward_curve.png",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom legend labels. Must match the number of plotted curves.",
    )
    parser.add_argument(
        "--common-x-max",
        action="store_true",
        help=(
            "Limit the x-axis maximum to the smallest final x value among "
            "the plotted curves."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive matplotlib window after saving the plot.",
    )
    return parser.parse_args()


def find_metric_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    direct_metrics = path / "metrics.jsonl"
    if direct_metrics.is_file():
        return [direct_metrics]

    return sorted(path.rglob("metrics.jsonl"))


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
            if isinstance(record, dict):
                yield record


def to_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_reward_key(records: list[dict[str, object]], metric: str | None) -> str:
    if metric is not None:
        return metric

    for key in DEFAULT_REWARD_KEYS:
        if any(to_float(record.get(key)) is not None for record in records):
            return key

    available_reward_keys = sorted(
        {
            key
            for record in records
            for key in record
            if "reward" in key.lower() or "return" in key.lower()
        }
    )
    if available_reward_keys:
        return available_reward_keys[0]

    raise ValueError("No reward/return metric found in records.")


def load_curve(
    metrics_path: Path,
    *,
    metric: str | None,
    step_key: str,
) -> tuple[list[float], list[float], str]:
    records = list(iter_jsonl(metrics_path))
    if not records:
        raise ValueError(f"{metrics_path} is empty.")

    reward_key = choose_reward_key(records, metric)
    steps: list[float] = []
    rewards: list[float] = []

    for index, record in enumerate(records, start=1):
        reward = to_float(record.get(reward_key))
        if reward is None:
            continue

        step = to_float(record.get(step_key))
        if step is None:
            step = to_float(record.get("update"))
        if step is None:
            step = float(index)

        steps.append(step)
        rewards.append(reward)

    if not steps:
        raise ValueError(f"{metrics_path} has no plottable values for {reward_key!r}.")

    return steps, rewards, reward_key


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        smoothed.append(running_sum / count)
    return smoothed


def run_label(metrics_path: Path, runs_dir: Path) -> str:
    try:
        run_dir = metrics_path.parent.relative_to(runs_dir)
    except ValueError:
        run_dir = metrics_path.parent
    return str(run_dir)


def main() -> None:
    args = parse_args()
    runs_dirs = args.runs_dirs or [Path("runs")]
    metrics_files = [
        (metrics_path, runs_dir)
        for runs_dir in runs_dirs
        for metrics_path in find_metric_files(runs_dir)
    ]
    if not metrics_files:
        paths = ", ".join(str(path) for path in runs_dirs)
        raise SystemExit(f"No metrics.jsonl files found under {paths}.")

    if args.labels is not None and len(args.labels) != len(metrics_files):
        raise SystemExit(
            f"--labels received {len(args.labels)} label(s), but "
            f"{len(metrics_files)} curve(s) would be plotted."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to plot curves. Install it with: "
            "pip install matplotlib"
        ) from exc

    plotted = 0
    used_metric: str | None = None
    curve_max_steps: list[float] = []
    plt.figure(figsize=(11, 6))

    for curve_index, (metrics_path, runs_dir) in enumerate(metrics_files):
        try:
            steps, rewards, reward_key = load_curve(
                metrics_path,
                metric=args.metric,
                step_key=args.step_key,
            )
        except ValueError as exc:
            print(f"Skipping {metrics_path}: {exc}")
            continue

        used_metric = used_metric or reward_key
        rewards = moving_average(rewards, args.smooth)
        label = args.labels[curve_index] if args.labels is not None else run_label(metrics_path, runs_dir)
        plt.plot(steps, rewards, linewidth=1.8, label=label)
        curve_max_steps.append(max(steps))
        plotted += 1

    if plotted == 0:
        raise SystemExit("No plottable reward curves found.")

    metric_label = args.metric or used_metric or "reward"
    title = f"{metric_label} vs {args.step_key}"
    if args.smooth > 1:
        title += f" (moving average {args.smooth})"

    plt.title(title)
    plt.xlabel(args.step_key)
    plt.ylabel(metric_label)
    plt.grid(True, alpha=0.25)
    if args.common_x_max and curve_max_steps:
        plt.xlim(right=min(curve_max_steps))
    if plotted <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print(f"Saved reward curve to {args.output}")
    print(f"Plotted {plotted} run(s)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
