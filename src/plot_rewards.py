"""
Plot mean reward vs. timestep from metrics.jsonl files.

Usage:
    python plot_rewards.py                         # auto-discover all algos
    python plot_rewards.py --out reward_curve.png  # custom output path
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive; avoids blocking on plt.show()
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_metrics(path: Path) -> tuple[list[int], list[float], list[float]]:
    timesteps, rewards, alphas = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            timesteps.append(d["timestep"])
            rewards.append(d["mean_reward_10ep"])
            alphas.append(d["alpha"])
    return timesteps, rewards, alphas


def smooth(values: list[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def latest_run_per_algo(runs_dir: Path) -> dict[str, Path]:
    """
    Walk runs/{algo}/seed_*/{timestamp}/metrics.jsonl and return
    the metrics.jsonl of the LATEST timestamp for each algo.
    Directory structure: runs_dir / algo / seed_N / YYYYMMDD_HHMMSS / metrics.jsonl
    """
    by_algo: dict[str, list[Path]] = defaultdict(list)
    for mfile in runs_dir.rglob("metrics.jsonl"):
        algo = mfile.parts[len(runs_dir.parts)]   # first component after runs_dir
        by_algo[algo].append(mfile)
    # sort by timestamp directory name (lexicographic == chronological for YYYYMMDD_HHMMSS)
    return {
        algo: sorted(files, key=lambda p: p.parent.name)[-1]
        for algo, files in sorted(by_algo.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir", type=Path,
        default=Path(__file__).parent.parent / "runs",
        help="Root directory that contains all run folders",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent.parent / "reward_curve.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--smooth", type=int, default=20,
        help="Moving-average window (set to 1 to disable)",
    )
    return parser.parse_args()


ALGO_STYLE: dict[str, dict] = {
    "sac":  {"label": "SAC (improved)",  "color": "#F44336"},
    "sac0": {"label": "SAC0 (baseline)", "color": "#2196F3"},
}
_FALLBACK_COLORS = ["#4CAF50", "#FF9800", "#9C27B0", "#795548"]


def main() -> None:
    args = parse_args()

    algo_to_file = latest_run_per_algo(args.runs_dir)
    if not algo_to_file:
        print(f"No metrics.jsonl files found under {args.runs_dir}")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=False
    )

    fallback_idx = 0
    for algo, mfile in algo_to_file.items():
        style = ALGO_STYLE.get(algo)
        if style:
            label = style["label"]
            color = style["color"]
        else:
            label = algo
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1

        ts, rw, alpha_vals = load_metrics(mfile)
        ts_arr    = np.asarray(ts) / 1_000_000
        rw_arr    = np.asarray(rw)
        alpha_arr = np.asarray(alpha_vals)

        ax1.plot(ts_arr, rw_arr, color=color, alpha=0.15, linewidth=0.7)
        ax1.plot(ts_arr, smooth(rw_arr.tolist(), args.smooth),
                 color=color, linewidth=1.8, label=label)

        ax2.plot(ts_arr, smooth(alpha_arr.tolist(), args.smooth),
                 color=color, linewidth=1.5, label=label)

        print(
            f"{label}: {len(ts)} data points | "
            f"t=[{ts[0]/1e6:.2f}M, {ts[-1]/1e6:.2f}M] | "
            f"run={mfile.parent.name} | "
            f"peak reward={max(rw):.0f}"
        )


    ax1.set_ylabel("Mean reward (10-ep window)", fontsize=12)
    ax1.set_title(
        "HumanoidStandup-v5 · SAC Improved vs. SAC Baseline",
        fontsize=14, fontweight="bold",
    )
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)


    ax2.set_xlabel("Timesteps (millions)", fontsize=12)
    ax2.set_ylabel("Entropy coeff. α", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.out.resolve()}")


if __name__ == "__main__":
    main()
