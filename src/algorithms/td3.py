from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from algorithms.base import Algorithm
from algorithms.ppo import RunningMeanStd
from env import make_env


def _build_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
    *,
    output_gain: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_size in hidden_sizes:
        linear = nn.Linear(last_dim, hidden_size)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.constant_(linear.bias, 0.0)
        layers.extend((linear, nn.ReLU()))
        last_dim = hidden_size

    output_layer = nn.Linear(last_dim, output_dim)
    nn.init.orthogonal_(output_layer.weight, gain=output_gain)
    nn.init.constant_(output_layer.bias, 0.0)
    layers.append(output_layer)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic tanh-squashed actor for continuous TD3 control."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_sizes: tuple[int, ...],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.network = _build_mlp(
            observation_dim,
            hidden_sizes,
            action_dim,
            output_gain=0.01,
        )
        low = torch.as_tensor(action_low.reshape(-1), dtype=torch.float32)
        high = torch.as_tensor(action_high.reshape(-1), dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        raw_action = self.network(observations)
        return self.action_bias + self.action_scale * torch.tanh(raw_action)


class Critic(nn.Module):
    """Q-value network used by TD3."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.network = _build_mlp(
            observation_dim + action_dim,
            hidden_sizes,
            1,
            output_gain=1.0,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class ReplaySample:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Simple CPU replay buffer for vectorized continuous-control rollouts."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")

        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.terminations = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add_batch(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        observations = np.asarray(observations, dtype=np.float32).reshape(
            -1,
            self.observations.shape[1],
        )
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.actions.shape[1])
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        next_observations = np.asarray(next_observations, dtype=np.float32).reshape(
            -1,
            self.next_observations.shape[1],
        )
        terminations = np.asarray(terminations, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        batch_size = observations.shape[0]
        if batch_size > self.capacity:
            raise ValueError("Replay batch is larger than replay capacity.")

        indices = (np.arange(batch_size) + self.position) % self.capacity
        self.observations[indices] = observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_observations[indices] = next_observations
        self.terminations[indices] = terminations
        self.dones[indices] = dones
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, *, device: torch.device) -> ReplaySample:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("Replay sample batch_size must be positive.")

        indices = np.random.randint(0, self.size, size=int(batch_size))
        return ReplaySample(
            observations=torch.as_tensor(
                self.observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                self.actions[indices],
                dtype=torch.float32,
                device=device,
            ),
            rewards=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            terminations=torch.as_tensor(
                self.terminations[indices],
                dtype=torch.float32,
                device=device,
            ),
            dones=torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminations": self.terminations,
            "dones": self.dones,
            "position": self.position,
            "size": self.size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        capacity = int(state["capacity"])
        if capacity != self.capacity:
            raise ValueError(
                "Replay buffer capacity in checkpoint does not match current config: "
                f"{capacity} != {self.capacity}"
            )

        self.observations[...] = np.asarray(state["observations"], dtype=np.float32)
        self.actions[...] = np.asarray(state["actions"], dtype=np.float32)
        self.rewards[...] = np.asarray(state["rewards"], dtype=np.float32)
        self.next_observations[...] = np.asarray(
            state["next_observations"],
            dtype=np.float32,
        )
        self.terminations[...] = np.asarray(state["terminations"], dtype=np.float32)
        self.dones[...] = np.asarray(state["dones"], dtype=np.float32)
        self.position = int(state["position"])
        self.size = int(state["size"])


class TD3(Algorithm):
    """FastTD3-style TD3 trainer for Gymnasium HumanoidStandup."""

    def __init__(
        self,
        *,
        env: gym.Env,
        env_config: dict[str, Any],
        algo_config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        super().__init__(
            env=env,
            env_config=env_config,
            algo_config=algo_config,
            run_dir=run_dir,
        )

        self.hyperparameters = self.algo_config.get("hyperparameters", {})
        self.collection_config = self.algo_config.get("collection", {})
        self.network_config = self.algo_config.get("network", {})
        self.normalization_config = self.algo_config.get("normalization", {})
        self.reward_shaping_config = self.algo_config.get("reward_shaping", {})
        self.evaluation_config = self.algo_config.get("evaluation", {})
        self.logging_config = self.algo_config.get("logging", {})

        self.is_vector_env = hasattr(self.env, "num_envs")
        self.num_envs = int(getattr(self.env, "num_envs", 1))
        self.observation_space = getattr(
            self.env,
            "single_observation_space",
            self.env.observation_space,
        )
        self.action_space = getattr(
            self.env,
            "single_action_space",
            self.env.action_space,
        )
        if not isinstance(self.observation_space, spaces.Box):
            raise TypeError("TD3 only supports Box observation spaces.")
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("TD3 only supports Box action spaces.")

        self.seed = int(self.env_config.get("seed", 0))
        self.total_timesteps = int(self.hyperparameters.get("total_timesteps", 1_000_000))
        self.learning_starts = int(self.hyperparameters.get("learning_starts", 1024))
        self.random_steps = int(self.hyperparameters.get("random_steps", self.learning_starts))
        self.batch_size = int(self.hyperparameters.get("batch_size", 256))
        self.gradient_steps = int(self.hyperparameters.get("gradient_steps", 1))
        self.gamma = float(self.hyperparameters.get("gamma", 0.99))
        self.tau = float(self.hyperparameters.get("tau", 0.005))
        self.policy_delay = int(self.hyperparameters.get("policy_delay", 2))
        self.actor_learning_rate = float(
            self.hyperparameters.get("actor_learning_rate", 3e-4)
        )
        self.critic_learning_rate = float(
            self.hyperparameters.get("critic_learning_rate", 3e-4)
        )
        self.target_policy_noise = float(
            self.hyperparameters.get("target_policy_noise", 0.2)
        )
        self.target_noise_clip = float(self.hyperparameters.get("target_noise_clip", 0.5))
        self.exploration_noise_initial = float(
            self.hyperparameters.get("exploration_noise_initial", 0.2)
        )
        self.exploration_noise_final = float(
            self.hyperparameters.get("exploration_noise_final", 0.05)
        )
        self.exploration_fraction = float(
            self.hyperparameters.get("exploration_fraction", 0.5)
        )
        self.reward_scale = float(self.normalization_config.get("reward_scale", 0.01))

        environment_reward_config = self.reward_shaping_config.get(
            "environment_reward",
            {},
        )
        self.environment_reward_initial_scale = float(
            environment_reward_config.get("initial_scale", 1.0)
        )
        self.environment_reward_final_scale = float(
            environment_reward_config.get(
                "final_scale",
                self.environment_reward_initial_scale,
            )
        )
        self.environment_reward_decay_steps = int(
            environment_reward_config.get("decay_steps", 0)
        )
        if self.environment_reward_decay_steps < 0:
            raise ValueError("TD3 environment reward decay_steps must be non-negative.")

        self.actor_hidden_sizes = tuple(
            int(hidden_size)
            for hidden_size in self.network_config.get("actor_hidden_sizes", (256, 256))
        )
        self.critic_hidden_sizes = tuple(
            int(hidden_size)
            for hidden_size in self.network_config.get("critic_hidden_sizes", (256, 256))
        )
        self.normalize_observations = bool(
            self.normalization_config.get("normalize_observations", True)
        )
        self.observation_clip = float(self.normalization_config.get("observation_clip", 10.0))
        self.normalization_epsilon = float(self.normalization_config.get("epsilon", 1e-8))

        self.log_interval = int(self.logging_config.get("log_interval", 10_000))
        self.checkpoint_interval = int(
            self.logging_config.get("checkpoint_interval", 100_000)
        )
        self.save_replay_buffer = bool(
            self.logging_config.get("save_replay_buffer", False)
        )

        self.eval_interval = int(self.evaluation_config.get("eval_interval", 100_000))
        self.eval_episodes = int(self.evaluation_config.get("episodes", 1))
        self.eval_max_steps = int(self.evaluation_config.get("max_steps", 1000))
        self.eval_final_window = int(self.evaluation_config.get("final_window", 200))
        self.height_threshold = float(
            self.evaluation_config.get("height_threshold", 1.0)
        )
        self.upright_threshold = float(
            self.evaluation_config.get("upright_threshold", 0.65)
        )
        self.max_height_threshold = float(
            self.evaluation_config.get("max_height_threshold", 1.2)
        )
        return_threshold = self.evaluation_config.get("return_threshold")
        self.return_threshold = (
            None if return_threshold is None else float(return_threshold)
        )
        self.required_consecutive_successes = int(
            self.evaluation_config.get("consecutive_successes", 2)
        )
        self.early_stop_on_success = bool(
            self.evaluation_config.get("early_stop_on_success", True)
        )
        self.render_final_video = bool(
            self.evaluation_config.get("render_final_video", True)
        )
        self.video_fps = int(self.evaluation_config.get("video_fps", 30))

        knee_height_config = self.reward_shaping_config.get("knee_height", {})
        self.knee_height_weight = float(knee_height_config.get("weight", 0.0))
        self.knee_height_dt = float(knee_height_config.get("dt", 0.05))
        if self.knee_height_dt <= 0.0:
            raise ValueError("TD3 knee height reward shaping dt must be positive.")
        knee_force_config = self.reward_shaping_config.get("knee_force", {})
        self.knee_force_weight = float(knee_force_config.get("weight", 0.0))
        knee_force_clip = knee_force_config.get("clip", 200.0)
        self.knee_force_clip = (
            None if knee_force_clip is None else float(knee_force_clip)
        )
        if self.knee_force_clip is not None and self.knee_force_clip <= 0.0:
            raise ValueError("TD3 knee force reward shaping clip must be positive.")
        knee_symmetry_config = self.reward_shaping_config.get("knee_symmetry", {})
        self.knee_symmetry_weight = float(
            knee_symmetry_config.get("weight", 0.0)
        )
        hip_height_config = self.reward_shaping_config.get("hip_height", {})
        self.hip_height_weight = float(hip_height_config.get("weight", 0.0))
        self.hip_height_dt = float(hip_height_config.get("dt", 0.05))
        if self.hip_height_dt <= 0.0:
            raise ValueError("TD3 hip height reward shaping dt must be positive.")
        hip_velocity_config = self.reward_shaping_config.get("hip_velocity", {})
        self.hip_velocity_weight = float(hip_velocity_config.get("weight", 0.0))
        torso_upright_config = self.reward_shaping_config.get("torso_upright", {})
        self.torso_upright_weight = float(
            torso_upright_config.get("weight", 0.0)
        )
        abdomen_force_config = self.reward_shaping_config.get("abdomen_force", {})
        self.abdomen_force_weight = float(abdomen_force_config.get("weight", 0.0))
        abdomen_force_clip = abdomen_force_config.get("clip", 100.0)
        self.abdomen_force_clip = (
            None if abdomen_force_clip is None else float(abdomen_force_clip)
        )
        if self.abdomen_force_clip is not None and self.abdomen_force_clip <= 0.0:
            raise ValueError("TD3 abdomen force reward shaping clip must be positive.")
        self.abdomen_force_torque_sign = float(
            abdomen_force_config.get("torque_sign", 1.0)
        )
        if self.abdomen_force_torque_sign == 0.0:
            raise ValueError("TD3 abdomen force torque_sign must be non-zero.")
        self.abdomen_force_upright_threshold = float(
            abdomen_force_config.get("torso_upright_threshold", 0.8)
        )
        leg_vertical_angle_config = self.reward_shaping_config.get("leg_vertical_angle", {})
        self.leg_vertical_angle_weight = float(
            leg_vertical_angle_config.get("weight", 0.0)
        )
        stand_hold_config = self.reward_shaping_config.get("stand_hold", {})
        self.stand_hold_scale = float(stand_hold_config.get("scale", 0.0))
        self.stand_hold_base_reward = float(
            stand_hold_config.get("base_reward", 300.0)
        )
        self.stand_hold_success_bonus = float(
            stand_hold_config.get("success_bonus", 0.0)
        )
        self.stand_hold_height_min = float(stand_hold_config.get("height_min", 0.55))
        self.stand_hold_height_target = float(
            stand_hold_config.get("height_target", 1.05)
        )
        if self.stand_hold_height_target <= self.stand_hold_height_min:
            raise ValueError("TD3 stand_hold height_target must exceed height_min.")
        self.stand_hold_upright_min = float(stand_hold_config.get("upright_min", 0.0))
        self.stand_hold_upright_target = float(
            stand_hold_config.get("upright_target", 0.65)
        )
        if self.stand_hold_upright_target <= self.stand_hold_upright_min:
            raise ValueError("TD3 stand_hold upright_target must exceed upright_min.")
        action_penalty_config = self.reward_shaping_config.get("action_penalty", {})
        self.action_penalty_weight = float(action_penalty_config.get("weight", 0.0))
        if self.action_penalty_weight < 0.0:
            raise ValueError("TD3 action penalty weight must be non-negative.")

        requested_device = str(self.algo_config.get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        self.observation_dim = int(np.prod(self.observation_space.shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        self.action_low_np = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        self.action_high_np = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        self.action_scale_np = (self.action_high_np - self.action_low_np) / 2.0
        self.action_low = torch.as_tensor(
            self.action_low_np,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_high = torch.as_tensor(
            self.action_high_np,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_scale = torch.as_tensor(
            self.action_scale_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.actor: Actor | None = None
        self.actor_target: Actor | None = None
        self.critic_1: Critic | None = None
        self.critic_2: Critic | None = None
        self.critic_1_target: Critic | None = None
        self.critic_2_target: Critic | None = None
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer = ReplayBuffer(
            capacity=int(self.hyperparameters.get("buffer_size", 250_000)),
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        )
        self.observation_rms = RunningMeanStd(shape=(self.observation_dim,))

        self.global_step = 0
        self.gradient_update = 0
        self._resume_loaded = False
        self._last_observation: np.ndarray | None = None
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_returns_since_log: list[float] = []
        self._episode_lengths_since_log: list[int] = []
        self._recent_episode_returns: deque[float] = deque(maxlen=100)
        self._recent_episode_lengths: deque[int] = deque(maxlen=100)
        self._diagnostics = self._empty_diagnostics()
        self.metrics: dict[str, Any] = {}
        self._last_actor_loss: float | None = None
        self.consecutive_successes = 0
        self.success_reached = False
        self._last_log_step = 0
        self._last_checkpoint_step = 0
        self._last_eval_step = 0
        self._latest_eval: dict[str, Any] | None = None

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.video_dir = self.run_dir / "videos"
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def train(self) -> None:
        self._before_training()

        while self.global_step < self.total_timesteps:
            raw_observation = self._as_observation_batch(self._last_observation)
            self.observation_rms.update(raw_observation)
            normalized_observation = self._normalize_observation_batch(
                raw_observation,
                update=False,
            )
            action = self._select_action(normalized_observation)
            env_action = action if self.is_vector_env else action[0]

            next_observation, reward, terminated, truncated, info = self.env.step(env_action)
            next_observation_batch = self._as_observation_batch(next_observation)
            reward_array = np.asarray(reward, dtype=np.float32).reshape(self.num_envs)
            reward_array = self._shape_reward(reward_array, env_action=action)

            terminated_array = np.asarray(terminated, dtype=np.bool_).reshape(self.num_envs)
            truncated_array = np.asarray(truncated, dtype=np.bool_).reshape(self.num_envs)
            done_array = np.logical_or(terminated_array, truncated_array)
            replay_next_observation = next_observation_batch.copy()

            self.replay_buffer.add_batch(
                observations=raw_observation,
                actions=action,
                rewards=(reward_array * self.reward_scale).astype(np.float32),
                next_observations=replay_next_observation,
                terminations=terminated_array.astype(np.float32),
                dones=done_array.astype(np.float32),
            )

            self._record_step_diagnostics(info=info, env_action=action)
            self._current_episode_return += reward_array
            self._current_episode_length += 1
            self.global_step += self.num_envs

            if np.any(done_array):
                self._record_completed_episodes(done_array)
                next_observation_batch = self._reset_done_envs(done_array)

            self._last_observation = next_observation_batch

            if self.replay_buffer.size >= self.learning_starts:
                for _ in range(self.gradient_steps):
                    self._update_policy()

            self._maybe_log()
            self._maybe_save_checkpoint()
            self._maybe_evaluate()

            if self.success_reached and self.early_stop_on_success:
                break

        self._after_training()

    def _before_training(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_networks()
        if self._last_observation is None:
            self._last_observation, _ = self.env.reset(seed=self.seed)
            self._last_observation = self._as_observation_batch(self._last_observation)
        if not self._resume_loaded:
            self.global_step = 0
            self.gradient_update = 0
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _initialize_networks(self) -> None:
        if self.actor is not None:
            return

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.actor = Actor(
            self.observation_dim,
            self.action_dim,
            hidden_sizes=self.actor_hidden_sizes,
            action_low=self.action_low_np,
            action_high=self.action_high_np,
        ).to(self.device)
        self.actor_target = Actor(
            self.observation_dim,
            self.action_dim,
            hidden_sizes=self.actor_hidden_sizes,
            action_low=self.action_low_np,
            action_high=self.action_high_np,
        ).to(self.device)
        self.critic_1 = Critic(
            self.observation_dim,
            self.action_dim,
            hidden_sizes=self.critic_hidden_sizes,
        ).to(self.device)
        self.critic_2 = Critic(
            self.observation_dim,
            self.action_dim,
            hidden_sizes=self.critic_hidden_sizes,
        ).to(self.device)
        self.critic_1_target = Critic(
            self.observation_dim,
            self.action_dim,
            hidden_sizes=self.critic_hidden_sizes,
        ).to(self.device)
        self.critic_2_target = Critic(
            self.observation_dim,
            self.action_dim,
            hidden_sizes=self.critic_hidden_sizes,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            [*self.critic_1.parameters(), *self.critic_2.parameters()],
            lr=self.critic_learning_rate,
        )

    def _select_action(self, observations: np.ndarray) -> np.ndarray:
        if self.global_step < self.random_steps:
            return np.asarray(
                [self.action_space.sample() for _ in range(self.num_envs)],
                dtype=np.float32,
            ).reshape(self.num_envs, self.action_dim)

        if self.actor is None:
            self._initialize_networks()
        assert self.actor is not None

        observation_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            action_tensor = self.actor(observation_tensor)

        action = action_tensor.cpu().numpy()
        noise_scale = self._exploration_noise_scale()
        if noise_scale > 0.0:
            noise = np.random.normal(
                loc=0.0,
                scale=noise_scale * self.action_scale_np,
                size=action.shape,
            ).astype(np.float32)
            action = action + noise

        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32)

    def _exploration_noise_scale(self) -> float:
        if self.exploration_fraction <= 0.0:
            return self.exploration_noise_final
        decay_steps = max(int(self.total_timesteps * self.exploration_fraction), 1)
        progress = min(max(self.global_step / decay_steps, 0.0), 1.0)
        return (
            self.exploration_noise_initial
            + progress * (self.exploration_noise_final - self.exploration_noise_initial)
        )

    def _update_policy(self) -> None:
        if (
            self.actor is None
            or self.actor_target is None
            or self.critic_1 is None
            or self.critic_2 is None
            or self.critic_1_target is None
            or self.critic_2_target is None
            or self.actor_optimizer is None
            or self.critic_optimizer is None
        ):
            self._initialize_networks()

        assert self.actor is not None
        assert self.actor_target is not None
        assert self.critic_1 is not None
        assert self.critic_2 is not None
        assert self.critic_1_target is not None
        assert self.critic_2_target is not None
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None

        batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        observations = self._normalize_observation_tensor(batch.observations)
        next_observations = self._normalize_observation_tensor(batch.next_observations)

        with torch.no_grad():
            next_actions = self.actor_target(next_observations)
            target_noise = torch.randn_like(next_actions) * (
                self.target_policy_noise * self.action_scale
            )
            target_noise = torch.clamp(
                target_noise,
                -self.target_noise_clip * self.action_scale,
                self.target_noise_clip * self.action_scale,
            )
            next_actions = torch.clamp(
                next_actions + target_noise,
                self.action_low,
                self.action_high,
            )
            next_q_1 = self.critic_1_target(next_observations, next_actions)
            next_q_2 = self.critic_2_target(next_observations, next_actions)
            next_q = torch.minimum(next_q_1, next_q_2)
            target_q = batch.rewards + self.gamma * (1.0 - batch.terminations) * next_q

        current_q_1 = self.critic_1(observations, batch.actions)
        current_q_2 = self.critic_2(observations, batch.actions)
        critic_loss = F.mse_loss(current_q_1, target_q) + F.mse_loss(
            current_q_2,
            target_q,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value: float | None = None
        if self.gradient_update % self.policy_delay == 0:
            actor_actions = self.actor(observations)
            actor_loss = -self.critic_1(observations, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu().item())
            self._last_actor_loss = actor_loss_value

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)

        self.gradient_update += 1
        self.metrics.update(
            {
                "critic_loss": float(critic_loss.detach().cpu().item()),
                "actor_loss": actor_loss_value,
                "last_actor_loss": self._last_actor_loss,
                "q1_mean": float(current_q_1.detach().mean().cpu().item()),
                "q2_mean": float(current_q_2.detach().mean().cpu().item()),
                "target_q_mean": float(target_q.detach().mean().cpu().item()),
                "gradient_update": self.gradient_update,
            }
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(self.tau * source_param)

    def _maybe_log(self) -> None:
        if self.log_interval <= 0:
            return
        if self.global_step - self._last_log_step < self.log_interval:
            return

        record = self._build_log_record(event="train")
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

        episode_text = ""
        if "episode_return_mean" in record:
            episode_text = f" episode_return={record['episode_return_mean']:.2f}"
        actor_loss = record.get("actor_loss")
        if actor_loss is None:
            actor_loss = record.get("last_actor_loss")
        actor_text = "pending" if actor_loss is None else f"{float(actor_loss):.4f}"
        print(
            f"step={self.global_step}/{self.total_timesteps} "
            f"buffer={self.replay_buffer.size} "
            f"critic_loss={record.get('critic_loss', 0.0):.4f} "
            f"actor_loss={actor_text}"
            f"{episode_text}"
        )

        self._episode_returns_since_log.clear()
        self._episode_lengths_since_log.clear()
        self._diagnostics = self._empty_diagnostics()
        self._last_log_step = self.global_step

    def _build_log_record(self, *, event: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": event,
            "global_step": self.global_step,
            "num_envs": self.num_envs,
            "buffer_size": self.replay_buffer.size,
            "exploration_noise": self._exploration_noise_scale(),
            "consecutive_successes": self.consecutive_successes,
            "success_reached": self.success_reached,
            **self.metrics,
            **self._summarize_diagnostics(self._diagnostics),
        }
        if self._episode_returns_since_log:
            record["episode_return_mean"] = float(np.mean(self._episode_returns_since_log))
            record["episode_return_min"] = float(np.min(self._episode_returns_since_log))
            record["episode_return_max"] = float(np.max(self._episode_returns_since_log))
        if self._episode_lengths_since_log:
            record["episode_length_mean"] = float(np.mean(self._episode_lengths_since_log))
        if self._recent_episode_returns:
            record["recent_episode_return_mean"] = float(
                np.mean(self._recent_episode_returns)
            )
        if self._recent_episode_lengths:
            record["recent_episode_length_mean"] = float(
                np.mean(self._recent_episode_lengths)
            )
        if self._latest_eval is not None:
            record.update({f"eval_{key}": value for key, value in self._latest_eval.items()})
        return record

    def _maybe_save_checkpoint(self) -> None:
        if self.checkpoint_interval <= 0:
            return
        if self.global_step - self._last_checkpoint_step < self.checkpoint_interval:
            return

        self.save(self.checkpoint_dir / f"step_{self.global_step:09d}.pt")
        self._last_checkpoint_step = self.global_step

    def _maybe_evaluate(self) -> None:
        if self.eval_interval <= 0:
            return
        if self.global_step - self._last_eval_step < self.eval_interval:
            return

        eval_metrics = self.evaluate(render_video=False)
        self._latest_eval = eval_metrics
        self._update_success_state(eval_metrics)
        record = self._build_log_record(event="eval")
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

        print(
            f"eval step={self.global_step} "
            f"return={eval_metrics['return_mean']:.1f} "
            f"height={eval_metrics['final_window_torso_height_mean']:.3f} "
            f"upright={eval_metrics['final_window_torso_upright_mean']:.3f} "
            f"max_height={eval_metrics['max_torso_height']:.3f} "
            f"success={eval_metrics['success']}"
        )
        self._last_eval_step = self.global_step

    def _update_success_state(self, eval_metrics: dict[str, Any]) -> None:
        if bool(eval_metrics.get("success", False)):
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        self.success_reached = (
            self.consecutive_successes >= self.required_consecutive_successes
        )

    def _after_training(self) -> None:
        self.save(self.checkpoint_dir / "final.pt")
        if self.eval_interval <= 0 and not self.render_final_video:
            return

        final_eval = self.evaluate(
            render_video=self.render_final_video,
            video_path=self.video_dir / "final.mp4",
        )
        self._latest_eval = final_eval
        self._update_success_state(final_eval)
        record = self._build_log_record(event="final_eval")
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def evaluate(
        self,
        *,
        render_video: bool = False,
        video_path: Path | None = None,
    ) -> dict[str, Any]:
        if self.actor is None:
            self._initialize_networks()
        assert self.actor is not None

        heights: list[float] = []
        uprightness: list[float] = []
        returns: list[float] = []
        lengths: list[int] = []
        frames: list[np.ndarray] = []

        for episode_index in range(self.eval_episodes):
            env = make_env(
                self.env_config,
                render_mode="rgb_array" if render_video and episode_index == 0 else None,
            )
            try:
                observation, _ = env.reset(seed=self.seed + 10_000 + episode_index)
                episode_return = 0.0
                episode_length = 0

                if render_video and episode_index == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))

                for _ in range(self.eval_max_steps):
                    observation_batch = self._as_single_observation_batch(observation)
                    normalized_observation = self._normalize_observation_batch(
                        observation_batch,
                        update=False,
                    )
                    action = self._predict_action(normalized_observation)[0]
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    episode_length += 1

                    torso_height, torso_upright = self._torso_metrics(env.unwrapped)
                    heights.append(torso_height)
                    uprightness.append(torso_upright)

                    if render_video and episode_index == 0:
                        frame = env.render()
                        if frame is not None:
                            frames.append(np.asarray(frame))

                    if terminated or truncated:
                        break

                returns.append(episode_return)
                lengths.append(episode_length)
            finally:
                env.close()

        if render_video and video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=self.video_fps)

        height_array = np.asarray(heights, dtype=np.float64)
        upright_array = np.asarray(uprightness, dtype=np.float64)
        final_window = min(self.eval_final_window, height_array.size)
        if final_window == 0:
            final_height_mean = 0.0
            final_upright_mean = 0.0
            max_height = 0.0
        else:
            final_height_mean = float(height_array[-final_window:].mean())
            final_upright_mean = float(upright_array[-final_window:].mean())
            max_height = float(height_array.max())

        success = (
            final_height_mean >= self.height_threshold
            and final_upright_mean >= self.upright_threshold
            and max_height >= self.max_height_threshold
            and (
                self.return_threshold is None
                or (float(np.mean(returns)) if returns else 0.0) >= self.return_threshold
            )
        )
        return {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "final_window_torso_height_mean": final_height_mean,
            "final_window_torso_upright_mean": final_upright_mean,
            "max_torso_height": max_height,
            "return_threshold": self.return_threshold,
            "success": success,
            "video_path": str(video_path) if render_video and video_path is not None else None,
        }

    def _predict_action(self, observations: np.ndarray) -> np.ndarray:
        if self.actor is None:
            self._initialize_networks()
        assert self.actor is not None

        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=self.device,
            )
            action = self.actor(observation_tensor)
        return action.cpu().numpy().astype(np.float32)

    def save(self, path: Path) -> None:
        if (
            self.actor is None
            or self.actor_target is None
            or self.critic_1 is None
            or self.critic_2 is None
            or self.critic_1_target is None
            or self.critic_2_target is None
            or self.actor_optimizer is None
            or self.critic_optimizer is None
        ):
            self._initialize_networks()

        assert self.actor is not None
        assert self.actor_target is not None
        assert self.critic_1 is not None
        assert self.critic_2 is not None
        assert self.critic_1_target is not None
        assert self.critic_2_target is not None
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None

        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "observation_rms": self.observation_rms.state_dict(),
            "global_step": self.global_step,
            "gradient_update": self.gradient_update,
            "network": {
                "actor_hidden_sizes": list(self.actor_hidden_sizes),
                "critic_hidden_sizes": list(self.critic_hidden_sizes),
            },
            "env_config": self.env_config,
            "algo_config": self.algo_config,
        }
        if self.save_replay_buffer:
            checkpoint["replay_buffer"] = self.replay_buffer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._initialize_networks()

        assert self.actor is not None
        assert self.actor_target is not None
        assert self.critic_1 is not None
        assert self.critic_2 is not None
        assert self.critic_1_target is not None
        assert self.critic_2_target is not None
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None

        network = checkpoint.get("network", {})
        checkpoint_actor_sizes = tuple(
            int(hidden_size)
            for hidden_size in network.get("actor_hidden_sizes", self.actor_hidden_sizes)
        )
        checkpoint_critic_sizes = tuple(
            int(hidden_size)
            for hidden_size in network.get(
                "critic_hidden_sizes",
                self.critic_hidden_sizes,
            )
        )
        if checkpoint_actor_sizes != self.actor_hidden_sizes:
            raise ValueError(
                "Checkpoint actor_hidden_sizes do not match current config: "
                f"{checkpoint_actor_sizes} != {self.actor_hidden_sizes}"
            )
        if checkpoint_critic_sizes != self.critic_hidden_sizes:
            raise ValueError(
                "Checkpoint critic_hidden_sizes do not match current config: "
                f"{checkpoint_critic_sizes} != {self.critic_hidden_sizes}"
            )

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self.observation_rms.load_state_dict(checkpoint["observation_rms"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.gradient_update = int(checkpoint.get("gradient_update", 0))
        self._resume_loaded = True

    def _shape_reward(
        self,
        reward_array: np.ndarray,
        *,
        env_action: np.ndarray | None = None,
    ) -> np.ndarray:
        environment_reward_scale = self._environment_reward_scale()
        environment_reward_array = reward_array * environment_reward_scale
        knee_z_array, knee_reward_array = self._knee_height_reward_batch()
        knee_force_array, knee_force_reward_array = self._knee_force_reward_batch()
        (
            right_knee_angle_array,
            left_knee_angle_array,
            knee_asymmetry_array,
            knee_symmetry_reward_array,
        ) = self._knee_symmetry_reward_batch()
        (
            hip_z_array,
            hip_speed_array,
            hip_height_reward_array,
            hip_velocity_reward_array,
        ) = self._hip_reward_batch()
        torso_upright_array, torso_upright_reward_array = (
            self._torso_upright_reward_batch()
        )
        (
            abdomen_force_array,
            abdomen_force_gate_array,
            abdomen_force_reward_array,
        ) = self._abdomen_force_reward_batch()
        (
            right_leg_angle_array,
            left_leg_angle_array,
            leg_angle_array,
            leg_angle_reward_array,
        ) = self._leg_vertical_angle_reward_batch()
        (
            stand_height_array,
            stand_upright_array,
            stand_height_score_array,
            stand_upright_score_array,
            stand_hold_reward_array,
        ) = self._stand_hold_reward_batch()
        action_penalty_array, normalized_action_l2_array = (
            self._action_penalty_reward_batch(env_action)
        )

        shaped_reward = (
            environment_reward_array
            + knee_reward_array
            + knee_force_reward_array
            + knee_symmetry_reward_array
            + hip_height_reward_array
            + hip_velocity_reward_array
            + torso_upright_reward_array
            + abdomen_force_reward_array
            + leg_angle_reward_array
            + stand_hold_reward_array
            + action_penalty_array
        )
        self._extend_diagnostic("reward_environment", environment_reward_array)
        self._extend_diagnostic(
            "environment_reward_scale",
            np.full(self.num_envs, environment_reward_scale, dtype=np.float32),
        )
        self._extend_diagnostic("knee_z", knee_z_array)
        self._extend_diagnostic("reward_knee", knee_reward_array)
        self._extend_diagnostic("knee_force", knee_force_array)
        self._extend_diagnostic("reward_knee_force", knee_force_reward_array)
        self._extend_diagnostic("right_knee_angle", right_knee_angle_array)
        self._extend_diagnostic("left_knee_angle", left_knee_angle_array)
        self._extend_diagnostic("knee_angle_asymmetry", knee_asymmetry_array)
        self._extend_diagnostic("reward_knee_symmetry", knee_symmetry_reward_array)
        self._extend_diagnostic("hip_z", hip_z_array)
        self._extend_diagnostic("hip_speed", hip_speed_array)
        self._extend_diagnostic("reward_hip_height", hip_height_reward_array)
        self._extend_diagnostic("reward_hip_velocity", hip_velocity_reward_array)
        self._extend_diagnostic("torso_upright", torso_upright_array)
        self._extend_diagnostic("reward_torso_upright", torso_upright_reward_array)
        self._extend_diagnostic("abdomen_force", abdomen_force_array)
        self._extend_diagnostic("abdomen_force_gate", abdomen_force_gate_array)
        self._extend_diagnostic("reward_abdomen_force", abdomen_force_reward_array)
        self._extend_diagnostic("right_leg_vertical_angle", right_leg_angle_array)
        self._extend_diagnostic("left_leg_vertical_angle", left_leg_angle_array)
        self._extend_diagnostic("leg_vertical_angle", leg_angle_array)
        self._extend_diagnostic("reward_leg_vertical_angle", leg_angle_reward_array)
        self._extend_diagnostic("stand_torso_height", stand_height_array)
        self._extend_diagnostic("stand_torso_upright", stand_upright_array)
        self._extend_diagnostic("stand_height_score", stand_height_score_array)
        self._extend_diagnostic("stand_upright_score", stand_upright_score_array)
        self._extend_diagnostic("reward_stand_hold", stand_hold_reward_array)
        self._extend_diagnostic(
            "stand_hold_scale",
            np.full(self.num_envs, self.stand_hold_scale, dtype=np.float32),
        )
        self._extend_diagnostic("reward_action_penalty", action_penalty_array)
        self._extend_diagnostic("normalized_action_l2", normalized_action_l2_array)
        return shaped_reward.astype(np.float32)

    def _environment_reward_scale(self) -> float:
        if self.environment_reward_decay_steps <= 0:
            return self.environment_reward_final_scale

        progress = min(
            max(self.global_step / self.environment_reward_decay_steps, 0.0),
            1.0,
        )
        return (
            self.environment_reward_initial_scale
            + progress
            * (
                self.environment_reward_final_scale
                - self.environment_reward_initial_scale
            )
        )

    def _knee_height_reward_batch(self) -> tuple[np.ndarray, np.ndarray]:
        knee_z = np.zeros(self.num_envs, dtype=np.float32)
        if self.knee_height_weight == 0.0:
            return knee_z, knee_z.copy()

        for env_index, env in enumerate(self._unwrapped_envs()):
            try:
                right_knee_id = env.model.joint("right_knee").id
                left_knee_id = env.model.joint("left_knee").id
            except KeyError as error:
                raise ValueError(
                    "TD3 knee height reward shaping requires MuJoCo joints "
                    "'right_knee' and 'left_knee'."
                ) from error

            right_knee_z = float(env.data.xanchor[right_knee_id, 2])
            left_knee_z = float(env.data.xanchor[left_knee_id, 2])
            knee_z[env_index] = 0.5 * (right_knee_z + left_knee_z)

        knee_reward = self.knee_height_weight * knee_z / self.knee_height_dt
        return knee_z, knee_reward.astype(np.float32, copy=False)

    def _knee_force_reward_batch(self) -> tuple[np.ndarray, np.ndarray]:
        knee_force = np.zeros(self.num_envs, dtype=np.float32)
        if self.knee_force_weight == 0.0:
            return knee_force, knee_force.copy()

        for env_index, env in enumerate(self._unwrapped_envs()):
            try:
                right_knee_id = env.model.joint("right_knee").id
                left_knee_id = env.model.joint("left_knee").id
            except KeyError as error:
                raise ValueError(
                    "TD3 knee force reward shaping requires MuJoCo joints "
                    "'right_knee' and 'left_knee'."
                ) from error

            right_dof = int(env.model.jnt_dofadr[right_knee_id])
            left_dof = int(env.model.jnt_dofadr[left_knee_id])
            right_torque = float(env.data.qfrc_actuator[right_dof])
            left_torque = float(env.data.qfrc_actuator[left_dof])
            knee_force[env_index] = 0.5 * (abs(right_torque) + abs(left_torque))

        if self.knee_force_clip is not None:
            knee_force = np.clip(knee_force, 0.0, self.knee_force_clip)

        knee_force_reward = self.knee_force_weight * knee_force
        return knee_force, knee_force_reward.astype(np.float32, copy=False)

    def _knee_symmetry_reward_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        right_knee_angle = np.zeros(self.num_envs, dtype=np.float32)
        left_knee_angle = np.zeros(self.num_envs, dtype=np.float32)
        knee_asymmetry = np.zeros(self.num_envs, dtype=np.float32)
        if self.knee_symmetry_weight == 0.0:
            return (
                right_knee_angle,
                left_knee_angle,
                knee_asymmetry,
                knee_asymmetry.copy(),
            )

        for env_index, env in enumerate(self._unwrapped_envs()):
            try:
                right_knee_id = env.model.joint("right_knee").id
                left_knee_id = env.model.joint("left_knee").id
            except KeyError as error:
                raise ValueError(
                    "TD3 knee symmetry reward shaping requires MuJoCo joints "
                    "'right_knee' and 'left_knee'."
                ) from error

            right_qpos = int(env.model.jnt_qposadr[right_knee_id])
            left_qpos = int(env.model.jnt_qposadr[left_knee_id])
            right_knee_angle[env_index] = float(env.data.qpos[right_qpos])
            left_knee_angle[env_index] = float(env.data.qpos[left_qpos])
            angle_diff = right_knee_angle[env_index] - left_knee_angle[env_index]
            knee_asymmetry[env_index] = angle_diff * angle_diff

        knee_symmetry_reward = -self.knee_symmetry_weight * knee_asymmetry
        return (
            right_knee_angle,
            left_knee_angle,
            knee_asymmetry,
            knee_symmetry_reward.astype(np.float32, copy=False),
        )

    def _hip_reward_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hip_z = np.zeros(self.num_envs, dtype=np.float32)
        hip_speed = np.zeros(self.num_envs, dtype=np.float32)
        if self.hip_height_weight == 0.0 and self.hip_velocity_weight == 0.0:
            return hip_z, hip_speed, hip_z.copy(), hip_speed.copy()

        for env_index, env in enumerate(self._unwrapped_envs()):
            try:
                pelvis_id = env.model.body("pelvis").id
            except KeyError as error:
                raise ValueError(
                    "TD3 hip reward shaping requires a MuJoCo body named 'pelvis'."
                ) from error

            linear_velocity = np.asarray(env.data.cvel[pelvis_id, 3:6], dtype=np.float64)
            hip_z[env_index] = float(env.data.xipos[pelvis_id, 2])
            hip_speed[env_index] = float(np.linalg.norm(linear_velocity))

        hip_height_reward = self.hip_height_weight * hip_z / self.hip_height_dt
        hip_velocity_reward = -self.hip_velocity_weight * np.square(hip_speed)
        return (
            hip_z,
            hip_speed,
            hip_height_reward.astype(np.float32, copy=False),
            hip_velocity_reward.astype(np.float32, copy=False),
        )

    def _torso_upright_reward_batch(self) -> tuple[np.ndarray, np.ndarray]:
        torso_upright = np.zeros(self.num_envs, dtype=np.float32)
        if self.torso_upright_weight == 0.0:
            return torso_upright, torso_upright.copy()

        for env_index, env in enumerate(self._unwrapped_envs()):
            _, torso_upright[env_index] = self._torso_metrics(env)

        torso_upright_reward = self.torso_upright_weight * torso_upright
        return (
            torso_upright,
            torso_upright_reward.astype(np.float32, copy=False),
        )

    def _abdomen_force_reward_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        abdomen_force = np.zeros(self.num_envs, dtype=np.float32)
        abdomen_gate = np.zeros(self.num_envs, dtype=np.float32)
        if self.abdomen_force_weight == 0.0:
            return abdomen_force, abdomen_gate, abdomen_force.copy()

        for env_index, env in enumerate(self._unwrapped_envs()):
            try:
                abdomen_y_id = env.model.joint("abdomen_y").id
                torso_id = env.model.body("torso").id
            except KeyError as error:
                raise ValueError(
                    "TD3 abdomen force reward shaping requires a MuJoCo joint "
                    "named 'abdomen_y' and a body named 'torso'."
                ) from error

            dof = int(env.model.jnt_dofadr[abdomen_y_id])
            torque = float(env.data.qfrc_actuator[dof])
            directed_torque = self.abdomen_force_torque_sign * torque
            torso_upright = float(env.data.xmat[torso_id, 8])
            abdomen_gate[env_index] = float(
                torso_upright < self.abdomen_force_upright_threshold
            )
            abdomen_force[env_index] = max(directed_torque, 0.0) * abdomen_gate[env_index]

        if self.abdomen_force_clip is not None:
            abdomen_force = np.clip(abdomen_force, 0.0, self.abdomen_force_clip)

        abdomen_force_reward = self.abdomen_force_weight * abdomen_force
        return (
            abdomen_force,
            abdomen_gate,
            abdomen_force_reward.astype(np.float32, copy=False),
        )

    def _leg_vertical_angle_reward_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        right_angle = np.zeros(self.num_envs, dtype=np.float32)
        left_angle = np.zeros(self.num_envs, dtype=np.float32)
        mean_angle = np.zeros(self.num_envs, dtype=np.float32)
        if self.leg_vertical_angle_weight == 0.0:
            return right_angle, left_angle, mean_angle, mean_angle.copy()

        for env_index, env in enumerate(self._unwrapped_envs()):
            right_angle[env_index] = self._leg_vertical_angle(
                env,
                hip_joint_name="right_hip_x",
                foot_body_name="right_foot",
            )
            left_angle[env_index] = self._leg_vertical_angle(
                env,
                hip_joint_name="left_hip_x",
                foot_body_name="left_foot",
            )
            mean_angle[env_index] = 0.5 * (
                right_angle[env_index] + left_angle[env_index]
            )

        angle_reward = -self.leg_vertical_angle_weight * mean_angle
        return (
            right_angle,
            left_angle,
            mean_angle,
            angle_reward.astype(np.float32, copy=False),
        )

    def _leg_vertical_angle(
        self,
        env: gym.Env,
        *,
        hip_joint_name: str,
        foot_body_name: str,
    ) -> float:
        try:
            hip_joint_id = env.model.joint(hip_joint_name).id
            foot_body_id = env.model.body(foot_body_name).id
        except KeyError as error:
            raise ValueError(
                "TD3 leg vertical angle reward shaping requires MuJoCo joints "
                "'right_hip_x' and 'left_hip_x', and bodies 'right_foot' "
                "and 'left_foot'."
            ) from error

        hip_position = np.asarray(env.data.xanchor[hip_joint_id], dtype=np.float64)
        foot_position = np.asarray(env.data.xipos[foot_body_id], dtype=np.float64)
        leg_vector = foot_position - hip_position
        leg_length = float(np.linalg.norm(leg_vector))
        if leg_length <= 1e-8:
            return float(np.pi / 2.0)

        vertical_alignment = abs(float(leg_vector[2])) / leg_length
        vertical_alignment = float(np.clip(vertical_alignment, 0.0, 1.0))
        return float(np.arccos(vertical_alignment))

    def _stand_hold_reward_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        torso_height = np.zeros(self.num_envs, dtype=np.float32)
        torso_upright = np.zeros(self.num_envs, dtype=np.float32)
        height_score = np.zeros(self.num_envs, dtype=np.float32)
        upright_score = np.zeros(self.num_envs, dtype=np.float32)
        if self.stand_hold_scale == 0.0 and self.stand_hold_success_bonus == 0.0:
            return (
                torso_height,
                torso_upright,
                height_score,
                upright_score,
                height_score.copy(),
            )

        for env_index, env in enumerate(self._unwrapped_envs()):
            torso_height[env_index], torso_upright[env_index] = self._torso_metrics(env)

        height_score = np.clip(
            (torso_height - self.stand_hold_height_min)
            / (self.stand_hold_height_target - self.stand_hold_height_min),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        upright_score = np.clip(
            (torso_upright - self.stand_hold_upright_min)
            / (self.stand_hold_upright_target - self.stand_hold_upright_min),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        success_bonus = self.stand_hold_success_bonus * (
            (torso_height >= self.height_threshold)
            & (torso_upright >= self.upright_threshold)
        ).astype(np.float32)
        stand_hold_reward = (
            self.stand_hold_scale
            * self.stand_hold_base_reward
            * height_score
            * upright_score
            + success_bonus
        )
        return (
            torso_height,
            torso_upright,
            height_score,
            upright_score,
            stand_hold_reward.astype(np.float32, copy=False),
        )

    def _action_penalty_reward_batch(
        self,
        env_action: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        reward = np.zeros(self.num_envs, dtype=np.float32)
        normalized_action_l2 = np.zeros(self.num_envs, dtype=np.float32)
        if self.action_penalty_weight == 0.0:
            return reward, normalized_action_l2
        if env_action is None:
            raise ValueError("TD3 action penalty reward shaping requires env_action.")

        action = np.asarray(env_action, dtype=np.float32).reshape(
            self.num_envs,
            self.action_dim,
        )
        action_bias = (self.action_high_np + self.action_low_np) / 2.0
        normalized_action = (action - action_bias) / np.maximum(
            self.action_scale_np,
            1e-8,
        )
        normalized_action_l2 = np.sum(np.square(normalized_action), axis=1).astype(
            np.float32,
            copy=False,
        )
        reward = -self.action_penalty_weight * normalized_action_l2
        return reward.astype(np.float32, copy=False), normalized_action_l2

    def _torso_metrics(self, env: gym.Env) -> tuple[float, float]:
        try:
            torso_id = env.model.body("torso").id
        except KeyError as error:
            raise ValueError(
                "TD3 standing evaluation requires a MuJoCo body named 'torso'."
            ) from error

        torso_height = float(env.data.xpos[torso_id, 2])
        torso_upright = float(env.data.xmat[torso_id, 8])
        return torso_height, torso_upright

    def _reset_done_envs(self, done_array: np.ndarray) -> np.ndarray:
        if self.is_vector_env:
            next_observation, _ = self.env.reset(options={"reset_mask": done_array.copy()})
        else:
            next_observation, _ = self.env.reset()
        self._current_episode_return[done_array] = 0.0
        self._current_episode_length[done_array] = 0
        return self._as_observation_batch(next_observation)

    def _record_completed_episodes(self, done_array: np.ndarray) -> None:
        for env_index in np.flatnonzero(done_array):
            episode_return = float(self._current_episode_return[env_index])
            episode_length = int(self._current_episode_length[env_index])
            self._episode_returns_since_log.append(episode_return)
            self._episode_lengths_since_log.append(episode_length)
            self._recent_episode_returns.append(episode_return)
            self._recent_episode_lengths.append(episode_length)

    def _record_step_diagnostics(
        self,
        *,
        info: dict[str, Any],
        env_action: np.ndarray,
    ) -> None:
        env_action = np.asarray(env_action, dtype=np.float32).reshape(self.num_envs, -1)
        self._diagnostics["env_action_mean"].append(float(env_action.mean()))
        self._diagnostics["env_action_std"].append(float(env_action.std()))
        self._diagnostics["action_abs_mean"].append(float(np.abs(env_action).mean()))
        action_bias = (self.action_high_np + self.action_low_np) / 2.0
        normalized_action = (env_action - action_bias) / np.maximum(
            self.action_scale_np,
            1e-8,
        )
        self._diagnostics["action_saturation_fraction_095"].append(
            float((np.abs(normalized_action) > 0.95).mean())
        )

        for key in (
            "z_distance_from_origin",
            "reward_linup",
            "reward_quadctrl",
            "reward_impact",
        ):
            values = self._info_values(info, key)
            if values is not None:
                self._extend_diagnostic(key, values)

    def _extend_diagnostic(self, key: str, values: np.ndarray) -> None:
        self._diagnostics.setdefault(key, [])
        self._diagnostics[key].extend(float(value) for value in np.asarray(values).reshape(-1))

    def _empty_diagnostics(self) -> dict[str, list[float]]:
        return {
            "z_distance_from_origin": [],
            "reward_linup": [],
            "reward_quadctrl": [],
            "reward_impact": [],
            "reward_environment": [],
            "environment_reward_scale": [],
            "knee_z": [],
            "reward_knee": [],
            "knee_force": [],
            "reward_knee_force": [],
            "right_knee_angle": [],
            "left_knee_angle": [],
            "knee_angle_asymmetry": [],
            "reward_knee_symmetry": [],
            "hip_z": [],
            "hip_speed": [],
            "reward_hip_height": [],
            "reward_hip_velocity": [],
            "torso_upright": [],
            "reward_torso_upright": [],
            "abdomen_force": [],
            "abdomen_force_gate": [],
            "reward_abdomen_force": [],
            "right_leg_vertical_angle": [],
            "left_leg_vertical_angle": [],
            "leg_vertical_angle": [],
            "reward_leg_vertical_angle": [],
            "stand_torso_height": [],
            "stand_torso_upright": [],
            "stand_height_score": [],
            "stand_upright_score": [],
            "reward_stand_hold": [],
            "stand_hold_scale": [],
            "reward_action_penalty": [],
            "normalized_action_l2": [],
            "env_action_mean": [],
            "env_action_std": [],
            "action_abs_mean": [],
            "action_saturation_fraction_095": [],
        }

    def _summarize_diagnostics(
        self,
        diagnostics: dict[str, list[float]],
    ) -> dict[str, float]:
        summary: dict[str, float] = {}
        for key, values in diagnostics.items():
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            summary[f"{key}_mean"] = float(array.mean())
            summary[f"{key}_min"] = float(array.min())
            summary[f"{key}_max"] = float(array.max())
        return summary

    def _info_values(self, info: dict[str, Any], key: str) -> np.ndarray | None:
        if key not in info:
            return None

        values = np.asarray(info[key])
        mask = info.get(f"_{key}")
        if mask is not None:
            mask_array = np.asarray(mask, dtype=np.bool_)
            values = values[mask_array]

        if values.size == 0:
            return None
        return values.astype(np.float64, copy=False)

    def _normalize_observation_batch(
        self,
        observations: np.ndarray,
        *,
        update: bool,
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        else:
            observations = observations.reshape(observations.shape[0], -1)
        if not self.normalize_observations:
            return observations.astype(np.float32)
        if update:
            self.observation_rms.update(observations)

        normalized = (observations - self.observation_rms.mean) / np.sqrt(
            self.observation_rms.var + self.normalization_epsilon
        )
        return np.clip(
            normalized,
            -self.observation_clip,
            self.observation_clip,
        ).astype(np.float32)

    def _normalize_observation_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observations:
            return observations
        mean = torch.as_tensor(
            self.observation_rms.mean,
            dtype=torch.float32,
            device=observations.device,
        )
        var = torch.as_tensor(
            self.observation_rms.var,
            dtype=torch.float32,
            device=observations.device,
        )
        normalized = (observations - mean) / torch.sqrt(var + self.normalization_epsilon)
        return torch.clamp(normalized, -self.observation_clip, self.observation_clip)

    def _as_observation_batch(self, observations: Any) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        return array.reshape(self.num_envs, -1)

    def _as_single_observation_batch(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        return array.reshape(1, -1)

    def _unwrapped_envs(self) -> list[gym.Env]:
        envs = getattr(self.env, "envs", None)
        if envs is None:
            return [self.env.unwrapped]
        return [env.unwrapped for env in envs]
