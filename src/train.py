from __future__ import annotations

"""Training entry point.

Fresh PPO training:
    .venv/bin/python src/train.py --algo ppo

Resume PPO training from a checkpoint:
    .venv/bin/python src/train.py --algo ppo \
        --checkpoint runs/ppo/seed_0/<run_id>/checkpoints/final.pt

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
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional run directory id under runs/<algo>/seed_<seed>/ to resume from "
            "using its checkpoints/final.pt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_env_config()
    algo_config = load_algorithm_config(args.algo)
    seed = int(env_config.get("seed", 0))
    checkpoint_path = resolve_checkpoint_path(
        algo=args.algo,
        seed=seed,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
    )
    run_dir = make_run_dir(args.algo, seed)

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
        if checkpoint_path is not None:
            algorithm.load(checkpoint_path)
        algorithm.train()
    finally:
        env.close()


def make_run_dir(algo: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / algo / f"seed_{seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_checkpoint_path(
    *,
    algo: str,
    seed: int,
    checkpoint: Path | None,
    run_id: str | None,
) -> Path | None:
    if checkpoint is not None and run_id is not None:
        raise ValueError("Use either --checkpoint or --run-id, not both.")
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    if run_id is None:
        return None

    resolved = ROOT / "runs" / algo / f"seed_{seed}" / run_id / "checkpoints" / "final.pt"
    return resolved.expanduser().resolve()


if __name__ == "__main__":
    main()
