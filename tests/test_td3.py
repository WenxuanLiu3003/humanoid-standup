from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from algorithms.td3 import Actor, ReplayBuffer, TD3
from src.config import load_algorithm_config


class TinyContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.step_count = 0
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = np.zeros(3, dtype=np.float32)
        return self.state.copy(), {"z_distance_from_origin": 0.0}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        self.state = np.asarray(
            [
                action[0],
                action[1],
                float(self.step_count),
            ],
            dtype=np.float32,
        )
        reward = float(1.0 - np.square(action).sum())
        terminated = False
        truncated = self.step_count >= 4
        info = {
            "z_distance_from_origin": float(self.state[2]),
            "reward_linup": reward,
        }
        return self.state.copy(), reward, terminated, truncated, info


def tiny_algo_config() -> dict:
    return {
        "name": "td3",
        "collection": {"num_envs": 2},
        "hyperparameters": {
            "total_timesteps": 16,
            "buffer_size": 64,
            "learning_starts": 4,
            "random_steps": 2,
            "batch_size": 4,
            "gradient_steps": 1,
            "actor_learning_rate": 0.001,
            "critic_learning_rate": 0.001,
            "gamma": 0.99,
            "tau": 0.01,
            "policy_delay": 2,
            "target_policy_noise": 0.1,
            "target_noise_clip": 0.2,
            "exploration_noise_initial": 0.1,
            "exploration_noise_final": 0.01,
            "exploration_fraction": 1.0,
        },
        "network": {
            "actor_hidden_sizes": [16],
            "critic_hidden_sizes": [16],
        },
        "normalization": {
            "normalize_observations": True,
            "observation_clip": 10.0,
            "reward_scale": 1.0,
            "epsilon": 1e-8,
        },
        "reward_shaping": {},
        "evaluation": {
            "eval_interval": 0,
            "render_final_video": False,
            "early_stop_on_success": False,
        },
        "logging": {
            "log_interval": 4,
            "checkpoint_interval": 0,
        },
        "device": "cpu",
    }


def make_tiny_vector_env() -> SyncVectorEnv:
    return SyncVectorEnv(
        [TinyContinuousEnv, TinyContinuousEnv],
        autoreset_mode=AutoresetMode.DISABLED,
    )


def test_td3_config_has_required_sections() -> None:
    config = load_algorithm_config("td3")

    assert config["name"] == "td3"
    assert config["collection"]["num_envs"] == 16
    assert config["hyperparameters"]["batch_size"] == 8192
    assert config["hyperparameters"]["target_policy_noise"] == 0.1
    assert config["hyperparameters"]["target_noise_clip"] == 0.25
    assert config["hyperparameters"]["exploration_noise_initial"] == 0.25
    assert config["hyperparameters"]["exploration_noise_final"] == 0.02
    assert config["hyperparameters"]["exploration_fraction"] == 0.5
    assert config["network"]["actor_hidden_sizes"] == [512, 256, 128]
    assert config["network"]["critic_hidden_sizes"] == [1024, 512, 256]
    assert config["evaluation"]["return_threshold"] == 180000.0
    assert config["evaluation"]["height_threshold"] == 1.0
    reward_shaping = config["reward_shaping"]
    assert reward_shaping["environment_reward"] == {
        "initial_scale": 1.0,
        "final_scale": 0.25,
        "decay_steps": 2000000,
    }
    assert reward_shaping["knee_height"]["weight"] == 0.5
    assert reward_shaping["knee_force"]["weight"] == 0.0
    assert reward_shaping["hip_height"]["weight"] == 0.5
    assert reward_shaping["hip_velocity"]["weight"] == 0.2
    assert reward_shaping["torso_upright"]["weight"] == 60.0
    assert reward_shaping["abdomen_force"]["weight"] == 0.0
    assert reward_shaping["leg_vertical_angle"]["weight"] == 4.0
    assert reward_shaping["stand_hold"] == {
        "scale": 1.0,
        "base_reward": 300.0,
        "success_bonus": 100.0,
        "height_min": 0.55,
        "height_target": 1.05,
        "upright_min": 0.0,
        "upright_target": 0.65,
    }
    assert reward_shaping["action_penalty"]["weight"] == 2.0


def test_actor_outputs_bounded_actions() -> None:
    actor = Actor(
        observation_dim=3,
        action_dim=2,
        hidden_sizes=(8,),
        action_low=np.asarray([-0.5, -1.0], dtype=np.float32),
        action_high=np.asarray([0.5, 1.0], dtype=np.float32),
    )

    actions = actor(torch.zeros((5, 3), dtype=torch.float32))

    assert actions.shape == (5, 2)
    assert torch.all(actions[:, 0] >= -0.5)
    assert torch.all(actions[:, 0] <= 0.5)
    assert torch.all(actions[:, 1] >= -1.0)
    assert torch.all(actions[:, 1] <= 1.0)


def test_replay_buffer_samples_expected_shapes() -> None:
    buffer = ReplayBuffer(capacity=10, observation_dim=3, action_dim=2)
    buffer.add_batch(
        observations=np.zeros((4, 3), dtype=np.float32),
        actions=np.zeros((4, 2), dtype=np.float32),
        rewards=np.ones(4, dtype=np.float32),
        next_observations=np.ones((4, 3), dtype=np.float32),
        terminations=np.zeros(4, dtype=np.float32),
        dones=np.zeros(4, dtype=np.float32),
    )

    sample = buffer.sample(6, device=torch.device("cpu"))

    assert sample.observations.shape == (6, 3)
    assert sample.actions.shape == (6, 2)
    assert sample.rewards.shape == (6,)
    assert sample.next_observations.shape == (6, 3)
    assert sample.terminations.shape == (6,)
    assert sample.dones.shape == (6,)


def test_td3_stand_hold_reward_curriculum_and_action_penalty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_algo_config()
    config["reward_shaping"] = {
        "environment_reward": {
            "initial_scale": 1.0,
            "final_scale": 0.25,
            "decay_steps": 2_000_000,
        },
        "stand_hold": {
            "scale": 1.0,
            "base_reward": 300.0,
            "success_bonus": 100.0,
            "height_min": 0.55,
            "height_target": 1.05,
            "upright_min": 0.0,
            "upright_target": 0.65,
        },
        "action_penalty": {"weight": 2.0},
    }
    env = make_tiny_vector_env()
    algorithm = TD3(
        env=env,
        env_config={"env_id": "TinyContinuousEnv-v0", "seed": 0},
        algo_config=config,
        run_dir=tmp_path,
    )
    envs = algorithm._unwrapped_envs()
    metrics_by_env = {
        id(envs[0]): (0.55, 0.0),
        id(envs[1]): (1.05, 0.65),
    }

    def fake_torso_metrics(env) -> tuple[float, float]:
        return metrics_by_env[id(env)]

    monkeypatch.setattr(algorithm, "_torso_metrics", fake_torso_metrics)
    algorithm.global_step = 1_000_000

    try:
        reward = algorithm._shape_reward(
            np.asarray([10.0, 10.0], dtype=np.float32),
            env_action=np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
        )
    finally:
        env.close()

    np.testing.assert_allclose(reward, np.asarray([6.25, 402.25], dtype=np.float32))
    np.testing.assert_allclose(
        algorithm._diagnostics["environment_reward_scale"],
        [0.625, 0.625],
    )
    np.testing.assert_allclose(
        algorithm._diagnostics["stand_height_score"],
        [0.0, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        algorithm._diagnostics["stand_upright_score"],
        [0.0, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        algorithm._diagnostics["reward_stand_hold"],
        [0.0, 400.0],
    )
    np.testing.assert_allclose(
        algorithm._diagnostics["reward_action_penalty"],
        [-0.0, -4.0],
    )


def test_td3_tiny_smoke_run_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    env = make_tiny_vector_env()
    algorithm = TD3(
        env=env,
        env_config={"env_id": "TinyContinuousEnv-v0", "seed": 0},
        algo_config=tiny_algo_config(),
        run_dir=tmp_path,
    )

    try:
        algorithm.train()
    finally:
        env.close()

    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "checkpoints" / "final.pt"

    assert metrics_path.exists()
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0


def test_td3_checkpoint_load_and_continue(tmp_path: Path) -> None:
    env = make_tiny_vector_env()
    first = TD3(
        env=env,
        env_config={"env_id": "TinyContinuousEnv-v0", "seed": 0},
        algo_config=tiny_algo_config(),
        run_dir=tmp_path / "first",
    )

    try:
        first.train()
    finally:
        env.close()

    second_config = tiny_algo_config()
    second_config["hyperparameters"]["total_timesteps"] = 20
    second_env = make_tiny_vector_env()
    second = TD3(
        env=second_env,
        env_config={"env_id": "TinyContinuousEnv-v0", "seed": 0},
        algo_config=second_config,
        run_dir=tmp_path / "second",
    )

    try:
        second.load(tmp_path / "first" / "checkpoints" / "final.pt")
        loaded_step = second.global_step
        second.train()
    finally:
        second_env.close()

    assert loaded_step > 0
    assert second.global_step >= loaded_step
    assert (tmp_path / "second" / "checkpoints" / "final.pt").exists()
