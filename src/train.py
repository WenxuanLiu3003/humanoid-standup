from __future__ import annotations

"""Training entry point.

Fresh training:
    .venv/bin/python src/train.py --algo ppo
    .venv/bin/python src/train.py --algo td3

Resume training from a checkpoint:
    .venv/bin/python src/train.py --algo td3 \
        --checkpoint runs/td3/seed_0/<run_id>/checkpoints/final.pt

Resume writes to a new run directory; it does not overwrite the old run.
The checkpoint network architecture must match the current algorithm config.
"""

import argparse
from datetime import datetime
from pathlib import Path

from algorithms import get_algorithm
from config import ROOT, available_algorithms, load_algorithm_config, load_env_config
from env import make_env, make_vector_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        required=True,
        choices=available_algorithms(),
        help="Algorithm config name under configs/algorithms/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional .pt checkpoint to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_env_config()
    algo_config = load_algorithm_config(args.algo)
    run_dir = make_run_dir(args.algo, int(env_config.get("seed", 0)))
    print(f"run_dir: {run_dir}", flush=True)

    num_envs = int(algo_config.get("collection", {}).get("num_envs", 1))
    env = (
        make_vector_env(env_config, num_envs=num_envs)
        if num_envs > 1
        else make_env(env_config)
    )
    try:
        algorithm_cls = get_algorithm(args.algo)
        algorithm = algorithm_cls(
            env=env,
            env_config=env_config,
            algo_config=algo_config,
            run_dir=run_dir,
        )
        if args.checkpoint is not None:
            algorithm.load(args.checkpoint.expanduser().resolve())
        algorithm.train()
    finally:
        env.close()


def make_run_dir(algo: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / algo / f"seed_{seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    main()
