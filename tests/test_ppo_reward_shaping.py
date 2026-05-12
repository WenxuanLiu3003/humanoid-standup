import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.config import load_algorithm_config
from algorithms.ppo import PPO


def test_ppo_reward_shaping_config_uses_delayed_rewards() -> None:
    reward_shaping = load_algorithm_config("ppo")["reward_shaping"]

    assert reward_shaping["knee_height"]["weight"] == 0.75
    assert reward_shaping["knee_force"]["weight"] == 0.0
    assert reward_shaping["knee_symmetry"]["weight"] == 0.15
    assert reward_shaping["hip_height"]["weight"] == 0.25
    assert reward_shaping["hip_velocity"]["weight"] == 0.02
    assert reward_shaping["torso_upright"]["weight"] == 1.8
    assert reward_shaping["torso_upright"]["active_pelvis_height_threshold"] == 0.5
    assert reward_shaping["torso_upright"]["upright_threshold"] == 0.25
    assert reward_shaping["torso_upright"]["positive_only"] is False
    assert reward_shaping["standup_hold"]["weight"] == 1.0
    assert reward_shaping["abdomen_force"]["weight"] == 0.0
    assert reward_shaping["sway_penalty"]["weight"] == 0.15
    assert (
        reward_shaping["sway_penalty"]["active_pelvis_height_threshold"] == 0.55
    )
    assert reward_shaping["support_balance"]["weight"] == 1.5
    assert reward_shaping["support_balance"]["active_pelvis_height_threshold"] == 0.55
    assert reward_shaping["support_balance"]["torso_horizontal_tolerance"] == 0.08
    assert reward_shaping["dual_support_symmetry"]["weight"] == 0.75
    assert (
        reward_shaping["dual_support_symmetry"]["active_pelvis_height_threshold"] == 0.55
    )
    assert reward_shaping["dual_support_symmetry"]["foot_height_tolerance"] == 0.04
    assert reward_shaping["dual_support_symmetry"]["torso_lateral_tolerance"] == 0.06
    assert reward_shaping["dual_support_symmetry"]["foot_height_coef"] == 1.0
    assert reward_shaping["dual_support_symmetry"]["torso_lateral_coef"] == 0.5
    assert reward_shaping["leg_vertical_angle"]["weight"] == 0.0
    assert reward_shaping["delayed_rewards"]["knee_height"]["weight"] == 0.0
    assert reward_shaping["delayed_rewards"]["knee_height"]["delay_seconds"] == 2.0
    assert reward_shaping["delayed_rewards"]["knee_height"]["clip"] == 0.45
    assert reward_shaping["delayed_rewards"]["hip_height"]["weight"] == 0.05
    assert reward_shaping["delayed_rewards"]["hip_height"]["delay_seconds"] == 2.0
    assert reward_shaping["delayed_rewards"]["hip_height"]["clip"] == 0.35
    assert reward_shaping["delayed_rewards"]["leg_vertical_angle"]["weight"] == 0.0
    assert (
        reward_shaping["delayed_rewards"]["leg_vertical_angle"]["delay_seconds"]
        == 2.0
    )
    assert reward_shaping["standup_success"]["pelvis_height_threshold"] == 0.72
    assert reward_shaping["standup_success"]["torso_upright_threshold"] == 0.65
    assert reward_shaping["standup_success"]["sustain_seconds"] == 0.4


def test_reward_delay_gate_batch_enables_rewards_after_two_seconds() -> None:
    ppo = PPO.__new__(PPO)
    ppo.num_envs = 3
    ppo.env_dt_seconds = 0.015
    ppo._current_episode_length = np.asarray([132, 133, 140], dtype=np.int64)

    gate = ppo._reward_delay_gate_batch(delay_seconds=2.0)

    np.testing.assert_array_equal(
        gate,
        np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    )


def test_update_standup_success_tracking_requires_sustain_steps() -> None:
    ppo = PPO.__new__(PPO)
    ppo.num_envs = 2
    ppo.standup_success_sustain_steps = 3
    ppo._current_success_streak = np.zeros(2, dtype=np.int64)
    ppo._current_episode_success = np.zeros(2, dtype=np.bool_)

    sustained = ppo._update_standup_success_tracking(np.asarray([True, False]))
    np.testing.assert_array_equal(sustained, np.asarray([0.0, 0.0], dtype=np.float32))

    sustained = ppo._update_standup_success_tracking(np.asarray([True, True]))
    np.testing.assert_array_equal(sustained, np.asarray([0.0, 0.0], dtype=np.float32))

    sustained = ppo._update_standup_success_tracking(np.asarray([True, True]))
    np.testing.assert_array_equal(sustained, np.asarray([1.0, 0.0], dtype=np.float32))


def test_standing_gate_batch_activates_above_threshold() -> None:
    ppo = PPO.__new__(PPO)
    ppo.num_envs = 3

    gate = ppo._standing_gate_batch(
        hip_z_array=np.asarray([0.74, 0.75, 0.9], dtype=np.float32),
        threshold=0.75,
    )

    np.testing.assert_array_equal(
        gate,
        np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    )


def test_support_balance_reward_penalizes_torso_offset_from_foot_midpoint() -> None:
    class DummyBody:
        def __init__(self, body_id: int) -> None:
            self.id = body_id

    class DummyModel:
        def __init__(self) -> None:
            self._bodies = {
                "torso": DummyBody(0),
                "right_foot": DummyBody(1),
                "left_foot": DummyBody(2),
            }

        def body(self, name: str) -> DummyBody:
            return self._bodies[name]

    class DummyData:
        def __init__(self) -> None:
            self.xipos = np.asarray(
                [
                    [0.75, 0.0, 1.0],
                    [0.40, 0.0, 0.0],
                    [0.60, 0.0, 0.0],
                ],
                dtype=np.float64,
            )

    class DummyEnv:
        def __init__(self) -> None:
            self.model = DummyModel()
            self.data = DummyData()

    ppo = PPO.__new__(PPO)
    ppo.num_envs = 1
    ppo.support_balance_weight = 1.5
    ppo.support_balance_active_pelvis_height_threshold = 0.55
    ppo.support_balance_torso_horizontal_tolerance = 0.1
    ppo._unwrapped_envs = lambda: [DummyEnv()]  # type: ignore[method-assign]

    offset, gate, reward = ppo._support_balance_reward_batch(
        hip_z_array=np.asarray([0.8], dtype=np.float32)
    )

    np.testing.assert_allclose(offset, np.asarray([0.25], dtype=np.float32))
    np.testing.assert_allclose(gate, np.asarray([1.0], dtype=np.float32))
    np.testing.assert_allclose(reward, np.asarray([-2.25], dtype=np.float32))


def test_dual_support_symmetry_penalizes_single_leg_bias() -> None:
    class DummyBody:
        def __init__(self, body_id: int) -> None:
            self.id = body_id

    class DummyModel:
        def __init__(self) -> None:
            self._bodies = {
                "torso": DummyBody(0),
                "right_foot": DummyBody(1),
                "left_foot": DummyBody(2),
            }

        def body(self, name: str) -> DummyBody:
            return self._bodies[name]

    class DummyData:
        def __init__(self) -> None:
            self.xipos = np.asarray(
                [
                    [0.0, 0.20, 1.0],
                    [0.0, 0.10, 0.18],
                    [0.0, -0.10, 0.00],
                ],
                dtype=np.float64,
            )

    class DummyEnv:
        def __init__(self) -> None:
            self.model = DummyModel()
            self.data = DummyData()

    ppo = PPO.__new__(PPO)
    ppo.num_envs = 1
    ppo.dual_support_symmetry_weight = 0.75
    ppo.dual_support_symmetry_active_pelvis_height_threshold = 0.55
    ppo.dual_support_symmetry_foot_height_tolerance = 0.04
    ppo.dual_support_symmetry_torso_lateral_tolerance = 0.06
    ppo.dual_support_symmetry_foot_height_coef = 1.0
    ppo.dual_support_symmetry_torso_lateral_coef = 0.5
    ppo._unwrapped_envs = lambda: [DummyEnv()]  # type: ignore[method-assign]

    foot_asymmetry, torso_offset, gate, reward = ppo._dual_support_symmetry_reward_batch(
        hip_z_array=np.asarray([0.8], dtype=np.float32)
    )

    np.testing.assert_allclose(foot_asymmetry, np.asarray([0.18], dtype=np.float32))
    np.testing.assert_allclose(torso_offset, np.asarray([0.20], dtype=np.float32))
    np.testing.assert_allclose(gate, np.asarray([1.0], dtype=np.float32))
    np.testing.assert_allclose(reward, np.asarray([-3.5], dtype=np.float32))


def test_load_restores_networks_but_not_optimizer_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "policy_network": {"policy": 1},
            "value_network": {"value": 2},
            "optimizer": {"state": {"should_not": "load"}},
            "global_step": 123,
            "network": {"hidden_sizes": [64, 64]},
            "normalization": {
                "observation_rms": {"mean": [0.0], "var": [1.0], "count": 1.0},
                "return_rms": {"mean": 0.0, "var": 1.0, "count": 1.0},
                "discounted_return": [0.0],
            },
        },
        checkpoint_path,
    )

    class DummyModule:
        def __init__(self) -> None:
            self.loaded_state = None

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.loaded_state = state

    class DummyRMS:
        def __init__(self) -> None:
            self.loaded_state = None

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.loaded_state = state

    ppo = PPO.__new__(PPO)
    ppo.device = torch.device("cpu")
    ppo.hidden_sizes = (64, 64)
    ppo._discounted_return = np.zeros(1, dtype=np.float64)
    ppo.observation_rms = DummyRMS()
    ppo.return_rms = DummyRMS()

    policy = DummyModule()
    value = DummyModule()
    optimizer = SimpleNamespace(state_loaded=False)

    def initialize_networks() -> None:
        ppo.policy_network = policy
        ppo.value_network = value
        ppo.optimizer = optimizer

    ppo._initialize_networks = initialize_networks  # type: ignore[method-assign]

    ppo.load(checkpoint_path)

    assert policy.loaded_state == {"policy": 1}
    assert value.loaded_state == {"value": 2}
    assert ppo.global_step == 123
    assert ppo._resume_loaded is True
    assert getattr(optimizer, "loaded_state", None) is None
