from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from algorithms.ppo import PPO
from config import load_algorithm_config, load_env_config
from env import make_env


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def deterministic_action(agent: PPO, observation: np.ndarray) -> np.ndarray:
    """Compute a deterministic action from the Gaussian policy mean."""
    assert agent.policy_network is not None

    observation_batch = agent._as_observation_batch(observation)
    normalized_observation = agent._normalize_observation_batch(
        observation_batch,
        update=False,
    )
    observation_tensor = torch.as_tensor(
        normalized_observation,
        dtype=torch.float32,
        device=agent.device,
    )

    with torch.no_grad():
        mean_action = agent.policy_network.forward(observation_tensor)
        env_action = agent._env_action_from_raw(mean_action)

    action = env_action.cpu().numpy().astype(np.float32)
    if agent.num_envs == 1:
        return action[0]
    return action


def evaluate(
    agent: PPO,
    env: gym.Env,
    *,
    num_episodes: int,
) -> list[float]:
    episode_returns: list[float] = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action = deterministic_action(agent, observation)
            observation, reward, terminated, truncated, _ = env.step(action)

            episode_return += float(reward)
            episode_length += 1
            done = bool(terminated or truncated)

        episode_returns.append(episode_return)
        print(
            f"episode={episode + 1}/{num_episodes} "
            f"return={episode_return:.2f} "
            f"length={episode_length}"
        )

    return episode_returns


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    env_config = load_config_or_default(run_dir / "env_config.json", load_env_config)
    algo_config = load_config_or_default(
        run_dir / "algo_config.json",
        lambda: load_algorithm_config("ppo"),
    )
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env = make_env(env_config, render_mode=args.render_mode)
    agent = PPO(
        env=env,
        env_config=env_config,
        algo_config=algo_config,
        run_dir=run_dir,
    )
    agent.load(checkpoint_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluating on: {env_config.get('env_id', 'unknown')}")
    print(f"Device: {agent.device}")

    try:
        returns = evaluate(agent, env, num_episodes=args.episodes)
    finally:
        env.close()

    returns_array = np.asarray(returns, dtype=np.float64)
    print()
    print(f"mean_return: {returns_array.mean():.2f}")
    print(f"std_return : {returns_array.std():.2f}")
    print(f"min_return : {returns_array.min():.2f}")
    print(f"max_return : {returns_array.max():.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="PPO run directory. Defaults to the latest runs/ppo/seed_* run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to final.pt.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render-mode", default=None)
    return parser.parse_args()


def resolve_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir

    candidates = sorted(
        (Path("runs") / "ppo").glob("seed_*/*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No PPO runs found under runs/ppo/seed_*.")
    return candidates[0]


def resolve_checkpoint_path(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    return run_dir / "checkpoints" / "final.pt"


def load_config_or_default(
    path: Path,
    default_loader: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if path.exists():
        return load_json(path)
    return default_loader()


if __name__ == "__main__":
    main()
