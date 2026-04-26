from __future__ import annotations

import os
from typing import Any

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv


def make_env(env_config: dict[str, Any], *, render_mode: str | None = None) -> gym.Env:
    env_id = env_config["env_id"]
    seed = int(env_config.get("seed", 0))
    env_kwargs = dict(env_config.get("env_kwargs", {}))

    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_vector_env(
    env_config: dict[str, Any],
    *,
    num_envs: int,
    render_mode: str | None = None,
) -> SyncVectorEnv:
    seed = int(env_config.get("seed", 0))

    def make_factory(seed_offset: int):
        def factory() -> gym.Env:
            env = make_env(env_config, render_mode=render_mode)
            env.reset(seed=seed + seed_offset)
            env.action_space.seed(seed + seed_offset)
            return env

        return factory

    return SyncVectorEnv(
        [make_factory(index) for index in range(num_envs)],
        autoreset_mode=AutoresetMode.DISABLED,
    )
