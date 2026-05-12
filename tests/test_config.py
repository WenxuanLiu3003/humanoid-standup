from pathlib import Path

from src.config import available_algorithms, load_algorithm_config, load_env_config


def test_env_config() -> None:
    config = load_env_config(Path("configs/env.yaml"))

    assert config["env_id"] == "HumanoidStandup-v5"
    assert config["seed"] == 0


def test_algorithm_configs_exist() -> None:
    assert available_algorithms() == ["a2c", "ddpg", "ppo", "sac", "td3"]
    assert load_algorithm_config("ppo")["name"] == "ppo"
