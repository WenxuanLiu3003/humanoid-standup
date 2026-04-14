from src.config import load_env_config
from src.env import make_env


def test_make_env() -> None:
    env = make_env(load_env_config())

    try:
        observation, info = env.reset(seed=0)

        assert env.spec is not None
        assert env.spec.id == "HumanoidStandup-v5"
        assert env.action_space.shape == (17,)
        assert observation.shape == (348,)
        assert "x_position" in info
    finally:
        env.close()
