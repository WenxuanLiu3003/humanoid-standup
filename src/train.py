from __future__ import annotations

"""Training entry point.

Fresh training:
    .venv/bin/python src/train.py --algo ppo
    .venv/bin/python src/train.py --algo sac
    .venv/bin/python src/train.py --algo td3
    .venv/bin/python src/train.py --algo trpo

Resume training from a checkpoint:
    .venv/bin/python src/train.py --algo td3 \
        --checkpoint runs/td3/seed_0/<run_id>/checkpoints/final.pt

Resume writes to a new run directory; it does not overwrite the old run.
The checkpoint network architecture must match the current algorithm config.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
    run_training(args.algo, checkpoint=args.checkpoint)


def run_training(algo: str, *, checkpoint: Path | None = None) -> Path:
    env_config = load_env_config()
    algo_config = load_algorithm_config(algo)
    run_dir = make_run_dir(algo, int(env_config.get("seed", 0)))
    write_run_configs(run_dir, env_config=env_config, algo_config=algo_config)
    print(f"run_dir: {run_dir}", flush=True)

    num_envs = int(algo_config.get("collection", {}).get("num_envs", 1))
    env = (
        make_vector_env(env_config, num_envs=num_envs)
        if num_envs > 1
        else make_env(env_config)
    )
    try:
        algorithm_cls = get_algorithm(algo)
        algorithm = algorithm_cls(
            env=env,
            env_config=env_config,
            algo_config=algo_config,
            run_dir=run_dir,
        )
        if checkpoint is not None:
            load = getattr(algorithm, "load", None)
            if load is None:
                raise TypeError(f"{algo} does not support checkpoint loading.")
            load(checkpoint.expanduser().resolve())
        algorithm.train()
    finally:
        env.close()

    return run_dir


def make_run_dir(algo: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / algo / f"seed_{seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_configs(
    run_dir: Path,
    *,
    env_config: dict[str, Any],
    algo_config: dict[str, Any],
) -> None:
    with (run_dir / "env_config.json").open("w", encoding="utf-8") as file:
        json.dump(env_config, file, indent=2)
    with (run_dir / "algo_config.json").open("w", encoding="utf-8") as file:
        json.dump(algo_config, file, indent=2)


if __name__ == "__main__":
    main()
