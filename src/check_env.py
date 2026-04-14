from __future__ import annotations

from config import load_env_config
from env import make_env


def main() -> None:
    env_config = load_env_config()
    env = make_env(env_config)

    try:
        observation, info = env.reset(seed=int(env_config.get("seed", 0)))
        print(f"env_id: {env_config['env_id']}")
        print(f"action_space: {env.action_space}")
        print(f"observation_space: {env.observation_space}")
        print(f"observation_shape: {getattr(observation, 'shape', None)}")
        print(f"info_keys: {sorted(info.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
