from __future__ import annotations

from typing import Any

import gymnasium as gym


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
