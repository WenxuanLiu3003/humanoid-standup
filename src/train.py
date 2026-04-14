from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from algorithms import get_algorithm
from config import ROOT, available_algorithms, load_algorithm_config, load_env_config
from env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        required=True,
        choices=available_algorithms(),
        help="Algorithm config name under configs/algorithms/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_env_config()
    algo_config = load_algorithm_config(args.algo)
    run_dir = make_run_dir(args.algo, int(env_config.get("seed", 0)))

    env = make_env(env_config)
    try:
        algorithm_cls = get_algorithm(args.algo)
        algorithm = algorithm_cls(env=env, env_config=env_config, algo_config=algo_config, run_dir=run_dir)
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
