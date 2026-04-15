from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal

from algorithms.base import Algorithm


def _orthogonal_init(module: nn.Module, *, gain: float = np.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


def _build_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
    *,
    output_gain: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_size in hidden_sizes:
        linear = nn.Linear(last_dim, hidden_size)
        _orthogonal_init(linear)
        layers.extend((linear, nn.Tanh()))
        last_dim = hidden_size

    output_layer = nn.Linear(last_dim, output_dim)
    _orthogonal_init(output_layer, gain=output_gain)
    layers.append(output_layer)
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Gaussian policy for continuous-action PPO."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (64, 64),
        initial_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.mean_network = _build_mlp(
            observation_dim,
            hidden_sizes,
            action_dim,
            output_gain=0.01,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        return self.mean_network(observations)

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.forward(observations)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)


class RunningMeanStd:
    """Online mean/variance tracker used for normalization."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape == self.mean.shape:
            values = values.reshape((1, *self.mean.shape))

        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (mean_a + mean_b + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = total_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ValueNetwork(nn.Module):
    """State-value function for PPO."""

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.value_network = _build_mlp(
            observation_dim,
            hidden_sizes,
            1,
            output_gain=1.0,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        return self.value_network(observations).squeeze(-1)


@dataclass(frozen=True)
class ActionSample:
    action: np.ndarray
    env_action: np.ndarray
    log_prob: np.ndarray
    value: np.ndarray


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    raw_rewards: torch.Tensor
    terminations: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    episode_returns: list[float]
    episode_lengths: list[int]
    diagnostics: dict[str, float]
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None


@dataclass(frozen=True)
class MiniBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class PPO(Algorithm):
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
        self.logging_config = self.algo_config.get("logging", {})

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
        self.seed = int(self.env_config.get("seed", 0))

        self.total_timesteps = int(self.hyperparameters.get("total_timesteps", 1_000_000))
        self.steps_per_env = int(
            self.collection_config.get(
                "steps_per_env",
                self.hyperparameters.get("rollout_steps", 2048),
            )
        )
        self.rollout_steps = self.steps_per_env
        self.rollout_batch_size = self.steps_per_env * self.num_envs
        self.minibatch_size = int(self.hyperparameters.get("minibatch_size", 64))
        self.update_epochs = int(self.hyperparameters.get("update_epochs", 10))

        self.learning_rate = float(self.hyperparameters.get("learning_rate", 3e-4))
        self.gamma = float(self.hyperparameters.get("gamma", 0.99))
        self.gae_lambda = float(self.hyperparameters.get("gae_lambda", 0.95))

        self.clip_ratio = float(self.hyperparameters.get("clip_ratio", 0.2))
        self.value_coef = float(self.hyperparameters.get("value_coef", 0.5))
        self.entropy_coef = float(self.hyperparameters.get("entropy_coef", 0.0))
        self.max_grad_norm = float(self.hyperparameters.get("max_grad_norm", 0.5))

        self.normalize_advantages = bool(
            self.hyperparameters.get("normalize_advantages", True)
        )
        self.target_kl = self.hyperparameters.get("target_kl")
        self.clip_value_loss = bool(self.hyperparameters.get("clip_value_loss", True))
        self.value_clip_range = float(
            self.hyperparameters.get("value_clip_range", self.clip_ratio)
        )

        self.log_interval = int(self.logging_config.get("log_interval", 1))
        self.checkpoint_interval = int(self.logging_config.get("checkpoint_interval", 50))

        self.num_updates = self.total_timesteps // self.rollout_batch_size

        network_config = self.algo_config.get("network", {})
        self.hidden_sizes = tuple(
            int(hidden_size)
            for hidden_size in network_config.get("hidden_sizes", (64, 64))
        )
        self.initial_log_std = float(network_config.get("initial_log_std", 0.0))

        normalization_config = self.algo_config.get("normalization", {})
        self.normalize_observations = bool(
            normalization_config.get("normalize_observations", True)
        )
        self.observation_clip = float(normalization_config.get("observation_clip", 10.0))
        self.normalization_epsilon = float(normalization_config.get("epsilon", 1e-8))
        self.reward_scale = float(normalization_config.get("reward_scale", 0.01))
        self.normalize_rewards = bool(
            normalization_config.get("normalize_rewards", False)
        )
        self.reward_clip = float(normalization_config.get("reward_clip", 10.0))

        requested_device = str(self.algo_config.get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        self.rollout_buffer: RolloutBatch | None = None
        self.policy_network: PolicyNetwork | None = None
        self.value_network: ValueNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.metrics: dict[str, Any] = {}
        self._last_observation: np.ndarray | None = None
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int64)
        self._discounted_return = np.zeros(self.num_envs, dtype=np.float64)
        self.global_step = 0
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._last_approx_kl = 0.0
        self._last_clip_fraction = 0.0
        self._last_value_clip_fraction = 0.0
        self._last_entropy = 0.0
        self._last_value_prediction_mean = 0.0
        self._last_value_prediction_std = 0.0
        self._action_scale: torch.Tensor | None = None
        self._action_bias: torch.Tensor | None = None
        self._action_log_scale_sum: torch.Tensor | None = None

        if not isinstance(self.observation_space, spaces.Box):
            raise TypeError("PPO only supports Box observation spaces.")
        observation_dim = int(np.prod(self.observation_space.shape))
        self.observation_rms = RunningMeanStd(shape=(observation_dim,))
        self.return_rms = RunningMeanStd(shape=())

    def train(self) -> None:
        self._before_training()

        for update_index in range(self.num_updates):
            self._collect_rollout(update_index)
            self._compute_returns_and_advantages()
            self._update_policy(update_index)
            self._log_update(update_index)
            self._maybe_save_checkpoint(update_index)

        self._after_training()

    def _before_training(self) -> None:
        """Run initialization immediately before training starts."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_networks()
        self._last_observation, _ = self.env.reset(seed=self.seed)
        self._last_observation = self._as_observation_batch(self._last_observation)
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int64)
        self._discounted_return = np.zeros(self.num_envs, dtype=np.float64)
        self.global_step = 0

    def _collect_rollout(self, update_index: int) -> RolloutBatch:
        """Interact with the environment and store one PPO rollout."""
        del update_index
        self.rollout_buffer = self.sample_trajectory(self.rollout_steps)
        return self.rollout_buffer

    def _select_action(self, observation: Any) -> ActionSample:
        """Sample an action from the current policy."""
        if self.policy_network is None or self.value_network is None:
            self._initialize_networks()

        assert self.policy_network is not None
        assert self.value_network is not None

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            distribution = self.policy_network.distribution(observation_tensor)
            action_tensor = distribution.sample()
            log_prob = self._squashed_log_prob(distribution, action_tensor)
            value = self.value_network(observation_tensor)

        action = action_tensor.cpu().numpy()
        env_action = self._env_action_from_raw(action_tensor).cpu().numpy()
        return ActionSample(
            action=action.astype(np.float32),
            env_action=env_action.astype(np.float32),
            log_prob=log_prob.cpu().numpy().astype(np.float32),
            value=value.cpu().numpy().astype(np.float32),
        )

    def sample_trajectory(self, num_steps: int | None = None) -> RolloutBatch:
        """Sample a fixed-length trajectory from the current policy."""
        if self.policy_network is None or self.value_network is None:
            self._initialize_networks()
        if self._last_observation is None:
            self._last_observation, _ = self.env.reset(seed=self.seed)

        steps = int(self.rollout_steps if num_steps is None else num_steps)
        if steps <= 0:
            raise ValueError("Trajectory length must be positive.")
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        raw_rewards: list[np.ndarray] = []
        terminations: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        values: list[np.ndarray] = []
        next_values: list[np.ndarray] = []
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        diagnostics: dict[str, list[float]] = {
            "z_distance_from_origin": [],
            "reward_linup": [],
            "reward_quadctrl": [],
            "reward_impact": [],
            "raw_action_mean": [],
            "raw_action_std": [],
            "env_action_mean": [],
            "env_action_std": [],
            "action_abs_mean": [],
            "action_saturation_fraction_095": [],
        }

        for _ in range(steps):
            raw_observation = self._as_observation_batch(self._last_observation)
            observation = self._normalize_observation_batch(raw_observation, update=True)
            action_sample = self._select_action(observation)

            env_action = (
                action_sample.env_action
                if self.num_envs > 1
                else action_sample.env_action[0]
            )
            next_observation, reward, terminated, truncated, info = self.env.step(env_action)

            next_observation_batch = self._as_observation_batch(next_observation)
            reward_array = np.asarray(reward, dtype=np.float32).reshape(self.num_envs)
            terminated_array = np.asarray(terminated, dtype=np.bool_).reshape(self.num_envs)
            truncated_array = np.asarray(truncated, dtype=np.bool_).reshape(self.num_envs)
            done_array = np.logical_or(terminated_array, truncated_array)
            scaled_reward = self._scale_reward_batch(reward_array, done=done_array)
            transition_next_values = self._predict_values(
                self._normalize_observation_batch(next_observation_batch, update=False)
            )

            self.global_step += self.num_envs

            self._record_rollout_diagnostics(
                diagnostics,
                info=info,
                raw_action=action_sample.action,
                env_action=action_sample.env_action,
            )

            observations.append(observation)
            actions.append(action_sample.action)
            log_probs.append(action_sample.log_prob)
            rewards.append(scaled_reward)
            raw_rewards.append(reward_array)
            terminations.append(terminated_array.astype(np.float32))
            dones.append(done_array.astype(np.float32))
            values.append(action_sample.value)
            next_values.append(transition_next_values)

            self._current_episode_return += reward_array
            self._current_episode_length += 1

            if np.any(done_array):
                for env_index in np.flatnonzero(done_array):
                    episode_returns.append(float(self._current_episode_return[env_index]))
                    episode_lengths.append(int(self._current_episode_length[env_index]))

                if self.num_envs > 1:
                    next_observation, _ = self.env.reset(
                        options={"reset_mask": done_array.copy()}
                    )
                    next_observation_batch = self._as_observation_batch(next_observation)
                else:
                    next_observation, _ = self.env.reset()
                    next_observation_batch = self._as_observation_batch(next_observation)

                self._current_episode_return[done_array] = 0.0
                self._current_episode_length[done_array] = 0

            self._last_observation = next_observation_batch

        return RolloutBatch(
            observations=torch.as_tensor(
                np.asarray(observations),
                dtype=torch.float32,
                device=self.device,
            ),
            actions=torch.as_tensor(
                np.asarray(actions),
                dtype=torch.float32,
                device=self.device,
            ),
            log_probs=torch.as_tensor(
                np.asarray(log_probs),
                dtype=torch.float32,
                device=self.device,
            ),
            rewards=torch.as_tensor(
                np.asarray(rewards),
                dtype=torch.float32,
                device=self.device,
            ),
            raw_rewards=torch.as_tensor(
                np.asarray(raw_rewards),
                dtype=torch.float32,
                device=self.device,
            ),
            terminations=torch.as_tensor(
                np.asarray(terminations),
                dtype=torch.float32,
                device=self.device,
            ),
            dones=torch.as_tensor(
                np.asarray(dones),
                dtype=torch.float32,
                device=self.device,
            ),
            values=torch.as_tensor(
                np.asarray(values),
                dtype=torch.float32,
                device=self.device,
            ),
            next_values=torch.as_tensor(
                np.asarray(next_values),
                dtype=torch.float32,
                device=self.device,
            ),
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            diagnostics=self._summarize_diagnostics(diagnostics),
        )

    def _initialize_networks(self) -> None:
        if self.policy_network is not None and self.value_network is not None:
            return

        if not isinstance(self.observation_space, spaces.Box):
            raise TypeError("PPO only supports Box observation spaces.")
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("PPO only supports Box action spaces.")

        observation_dim = int(np.prod(self.observation_space.shape))
        action_dim = int(np.prod(self.action_space.shape))
        action_low = torch.as_tensor(
            self.action_space.low.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        action_high = torch.as_tensor(
            self.action_space.high.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        self._action_scale = (action_high - action_low) / 2.0
        self._action_bias = (action_high + action_low) / 2.0
        self._action_log_scale_sum = torch.log(self._action_scale).sum()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.policy_network = PolicyNetwork(
            observation_dim,
            action_dim,
            hidden_sizes=self.hidden_sizes,
            initial_log_std=self.initial_log_std,
        ).to(self.device)
        self.value_network = ValueNetwork(
            observation_dim,
            hidden_sizes=self.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                *self.policy_network.parameters(),
                *self.value_network.parameters(),
            ],
            lr=self.learning_rate,
        )

    def _compute_returns_and_advantages(self) -> None:
        """Compute discounted returns and advantage estimates."""
        if self.rollout_buffer is None:
            raise RuntimeError("Cannot compute advantages before collecting a rollout.")
        if self.value_network is None:
            self._initialize_networks()

        assert self.value_network is not None

        buffer = self.rollout_buffer
        rewards = buffer.rewards
        terminations = buffer.terminations
        dones = buffer.dones
        values = buffer.values
        next_values = buffer.next_values

        with torch.no_grad():
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

            for step in reversed(range(rewards.shape[0])):
                bootstrap_non_terminal = 1.0 - terminations[step]
                gae_non_terminal = 1.0 - dones[step]

                delta = (
                    rewards[step]
                    + self.gamma * next_values[step] * bootstrap_non_terminal
                    - values[step]
                )
                last_gae = (
                    delta
                    + self.gamma * self.gae_lambda * gae_non_terminal * last_gae
                )
                advantages[step] = last_gae

            returns = advantages + values

            if self.normalize_advantages:
                advantage_std = advantages.std(unbiased=False)
                advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)

        buffer.advantages = advantages
        buffer.returns = returns
        self.metrics.update(
            {
                "advantages_mean": float(advantages.mean().item()),
                "advantages_std": float(advantages.std(unbiased=False).item()),
                "raw_rewards_mean": float(buffer.raw_rewards.mean().item()),
                "raw_rewards_std": float(buffer.raw_rewards.std(unbiased=False).item()),
                "scaled_rewards_mean": float(rewards.mean().item()),
                "scaled_rewards_std": float(rewards.std(unbiased=False).item()),
                "termination_fraction": float(terminations.mean().item()),
                "truncation_fraction": float((dones - terminations).mean().item()),
                "returns_mean": float(returns.mean().item()),
                "returns_std": float(returns.std(unbiased=False).item()),
                "reward_scale": self.reward_scale,
                **buffer.diagnostics,
            }
        )

    def _update_policy(self, update_index: int) -> None:
        """Run PPO minibatch optimization."""
        del update_index
        if self.rollout_buffer is None:
            raise RuntimeError("Cannot update PPO before collecting a rollout.")
        if self.rollout_buffer.advantages is None or self.rollout_buffer.returns is None:
            raise RuntimeError("Cannot update PPO before computing advantages.")
        if (
            self.policy_network is None
            or self.value_network is None
            or self.optimizer is None
        ):
            self._initialize_networks()

        assert self.optimizer is not None

        buffer = self.rollout_buffer
        assert buffer.advantages is not None
        assert buffer.returns is not None

        observations = self._flatten_rollout_tensor(buffer.observations)
        actions = self._flatten_rollout_tensor(buffer.actions)
        log_probs = self._flatten_rollout_tensor(buffer.log_probs)
        values = self._flatten_rollout_tensor(buffer.values)
        advantages = self._flatten_rollout_tensor(buffer.advantages)
        returns = self._flatten_rollout_tensor(buffer.returns)

        batch_size = observations.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        value_clip_fractions: list[float] = []
        grad_norms: list[float] = []
        early_stopped = False
        early_stop_epoch: int | None = None
        minibatches_used = 0

        for epoch_index in range(self.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                minibatch_indices = indices[start : start + minibatch_size]
                minibatch = MiniBatch(
                    observations=observations[minibatch_indices],
                    actions=actions[minibatch_indices],
                    old_log_probs=log_probs[minibatch_indices],
                    old_values=values[minibatch_indices],
                    advantages=advantages[minibatch_indices],
                    returns=returns[minibatch_indices],
                )

                policy_loss = self._compute_policy_loss(minibatch)
                value_loss = self._compute_value_loss(minibatch)
                entropy = self._compute_entropy_bonus(minibatch)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                parameters = self._trainable_parameters()
                grad_norm = nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
                approx_kls.append(self._last_approx_kl)
                clip_fractions.append(self._last_clip_fraction)
                value_clip_fractions.append(self._last_value_clip_fraction)
                grad_norms.append(float(grad_norm.detach().cpu().item()))
                minibatches_used += 1

                if (
                    self.target_kl is not None
                    and self._last_approx_kl > float(self.target_kl)
                ):
                    early_stopped = True
                    early_stop_epoch = epoch_index + 1
                    break

            if early_stopped:
                break

        self.metrics.update(
            {
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                "approx_kl_max": float(np.max(approx_kls)) if approx_kls else 0.0,
                "clip_fraction": (
                    float(np.mean(clip_fractions)) if clip_fractions else 0.0
                ),
                "value_clip_fraction": (
                    float(np.mean(value_clip_fractions))
                    if value_clip_fractions
                    else 0.0
                ),
                "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "early_stopped": early_stopped,
                "early_stop_epoch": early_stop_epoch,
                "minibatches_used": minibatches_used,
                "minibatches_possible": self.update_epochs
                * int(np.ceil(batch_size / minibatch_size)),
                **self._policy_diagnostics(),
            }
        )
        self._update_value_metrics(buffer)

    def _compute_policy_loss(self, batch: MiniBatch) -> torch.Tensor:
        """Compute clipped PPO policy loss."""
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None

        distribution = self.policy_network.distribution(batch.observations)
        new_log_probs = self._squashed_log_prob(distribution, batch.actions)
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = log_ratio.exp()

        unclipped_objective = ratio * batch.advantages
        clipped_objective = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * batch.advantages
        )
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()
            )
            self._last_approx_kl = float(approx_kl.cpu().item())
            self._last_clip_fraction = float(clip_fraction.cpu().item())

        return policy_loss

    def _compute_value_loss(self, batch: MiniBatch) -> torch.Tensor:
        """Compute value-function loss."""
        if self.value_network is None:
            self._initialize_networks()
        assert self.value_network is not None

        values = self.value_network(batch.observations)
        unclipped_loss = torch.square(values - batch.returns)
        if self.clip_value_loss:
            clipped_values = batch.old_values + torch.clamp(
                values - batch.old_values,
                -self.value_clip_range,
                self.value_clip_range,
            )
            clipped_loss = torch.square(clipped_values - batch.returns)
            value_loss = 0.5 * torch.maximum(unclipped_loss, clipped_loss).mean()
            value_clip_fraction = (
                (torch.abs(values - batch.old_values) > self.value_clip_range)
                .float()
                .mean()
            )
        else:
            value_loss = 0.5 * unclipped_loss.mean()
            value_clip_fraction = torch.zeros((), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            self._last_value_clip_fraction = float(value_clip_fraction.cpu().item())
            self._last_value_prediction_mean = float(values.mean().cpu().item())
            self._last_value_prediction_std = float(values.std(unbiased=False).cpu().item())

        return value_loss

    def _compute_entropy_bonus(self, batch: MiniBatch) -> torch.Tensor:
        """Compute entropy regularization term."""
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None

        distribution = self.policy_network.distribution(batch.observations)
        entropy = -self._squashed_log_prob(distribution, batch.actions).mean()
        with torch.no_grad():
            self._last_entropy = float(entropy.cpu().item())
        return entropy

    def _log_update(self, update_index: int) -> None:
        """Write metrics for the current update."""
        if self.rollout_buffer is None:
            raise RuntimeError("Cannot log PPO update before collecting a rollout.")

        update = update_index + 1
        buffer = self.rollout_buffer
        complete_episode_returns = buffer.episode_returns
        complete_episode_lengths = buffer.episode_lengths

        record: dict[str, Any] = {
            "update": update,
            "global_step": self.global_step,
            "num_envs": self.num_envs,
            "steps_per_env": int(buffer.rewards.shape[0]),
            "rollout_steps": int(buffer.rewards.numel()),
            **self.metrics,
        }
        if complete_episode_returns:
            record["episode_return_mean"] = float(np.mean(complete_episode_returns))
            record["episode_return_min"] = float(np.min(complete_episode_returns))
            record["episode_return_max"] = float(np.max(complete_episode_returns))
        if complete_episode_lengths:
            record["episode_length_mean"] = float(np.mean(complete_episode_lengths))

        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

        if update % self.log_interval == 0:
            episode_text = ""
            if complete_episode_returns:
                episode_text = (
                    f" episode_return={record['episode_return_mean']:.2f}"
                )
            print(
                f"update={update}/{self.num_updates} "
                f"step={self.global_step} "
                f"policy_loss={record.get('policy_loss', 0.0):.4f} "
                f"value_loss={record.get('value_loss', 0.0):.4f} "
                f"entropy={record.get('entropy', 0.0):.4f} "
                f"kl={record.get('approx_kl', 0.0):.6f}"
                f"{episode_text}"
            )

    def _maybe_save_checkpoint(self, update_index: int) -> None:
        """Save model state at configured intervals."""
        update = update_index + 1
        if self.checkpoint_interval <= 0:
            return
        if update % self.checkpoint_interval != 0:
            return

        self.save(self.checkpoint_dir / f"update_{update:06d}.pt")

    def _after_training(self) -> None:
        """Clean up and save final artifacts."""
        self.save(self.checkpoint_dir / "final.pt")

    def save(self, path: Path) -> None:
        """Save PPO model, optimizer, and metadata."""
        if (
            self.policy_network is None
            or self.value_network is None
            or self.optimizer is None
        ):
            self._initialize_networks()

        assert self.policy_network is not None
        assert self.value_network is not None
        assert self.optimizer is not None

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "value_network": self.value_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "hyperparameters": self.hyperparameters,
                "network": {
                    "hidden_sizes": list(self.hidden_sizes),
                    "initial_log_std": self.initial_log_std,
                },
                "normalization": {
                    "observation_rms": self.observation_rms.state_dict(),
                    "return_rms": self.return_rms.state_dict(),
                    "discounted_return": self._discounted_return,
                },
                "env_config": self.env_config,
                "algo_config": self.algo_config,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load PPO model, optimizer, and metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        if self.policy_network is None and self.value_network is None:
            network_config = checkpoint.get("network", {})
            self.hidden_sizes = tuple(
                int(hidden_size)
                for hidden_size in network_config.get("hidden_sizes", self.hidden_sizes)
            )
            self.initial_log_std = float(
                network_config.get("initial_log_std", self.initial_log_std)
            )
        self._initialize_networks()

        assert self.policy_network is not None
        assert self.value_network is not None
        assert self.optimizer is not None

        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        normalization_state = checkpoint.get("normalization", {})
        if "observation_rms" in normalization_state:
            self.observation_rms.load_state_dict(normalization_state["observation_rms"])
        if "return_rms" in normalization_state:
            self.return_rms.load_state_dict(normalization_state["return_rms"])
        self._discounted_return = np.asarray(
            normalization_state.get("discounted_return", self._discounted_return),
            dtype=np.float64,
        )

    def _normalize_observation(self, observation: np.ndarray, *, update: bool) -> np.ndarray:
        return self._normalize_observation_batch(observation, update=update).reshape(-1)

    def _normalize_observation_batch(
        self,
        observations: np.ndarray,
        *,
        update: bool,
    ) -> np.ndarray:
        observations = self._as_observation_batch(observations)
        if not self.normalize_observations:
            return observations

        if update:
            self.observation_rms.update(observations)

        normalized = (observations - self.observation_rms.mean) / np.sqrt(
            self.observation_rms.var + self.normalization_epsilon
        )
        normalized = np.clip(
            normalized,
            -self.observation_clip,
            self.observation_clip,
        )
        return normalized.astype(np.float32)

    def _scale_reward(self, reward: float, *, done: bool) -> float:
        return float(self._scale_reward_batch(np.asarray([reward]), done=np.asarray([done]))[0])

    def _scale_reward_batch(self, rewards: np.ndarray, *, done: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64).reshape(self.num_envs)
        done = np.asarray(done, dtype=np.bool_).reshape(self.num_envs)
        scaled_rewards = rewards * self.reward_scale

        if self.normalize_rewards:
            self._discounted_return = self.gamma * self._discounted_return + scaled_rewards
            self.return_rms.update(self._discounted_return)
            scaled_rewards = scaled_rewards / float(
                np.sqrt(self.return_rms.var + self.normalization_epsilon)
            )

        scaled_rewards = np.clip(scaled_rewards, -self.reward_clip, self.reward_clip)
        self._discounted_return[done] = 0.0
        return scaled_rewards.astype(np.float32)

    def _env_action_from_raw(self, raw_action: torch.Tensor) -> torch.Tensor:
        if self._action_scale is None or self._action_bias is None:
            self._initialize_networks()
        assert self._action_scale is not None
        assert self._action_bias is not None

        if raw_action.ndim == 1:
            raw_action = raw_action.unsqueeze(0)
        return self._action_bias + self._action_scale * torch.tanh(raw_action)

    def _squashed_log_prob(
        self,
        distribution: Normal,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        if self._action_log_scale_sum is None:
            self._initialize_networks()
        assert self._action_log_scale_sum is not None

        if raw_action.ndim == 1:
            raw_action = raw_action.unsqueeze(0)

        squashed_action = torch.tanh(raw_action)
        base_log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        tanh_log_det = torch.log(
            torch.clamp(1.0 - squashed_action.pow(2), min=1e-6)
        ).sum(dim=-1)
        return base_log_prob - tanh_log_det - self._action_log_scale_sum

    def _record_rollout_diagnostics(
        self,
        diagnostics: dict[str, list[float]],
        *,
        info: dict[str, Any],
        raw_action: np.ndarray,
        env_action: np.ndarray,
    ) -> None:
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(self.num_envs, -1)
        env_action = np.asarray(env_action, dtype=np.float32).reshape(self.num_envs, -1)
        squashed_action = np.tanh(raw_action)

        diagnostics["raw_action_mean"].append(float(raw_action.mean()))
        diagnostics["raw_action_std"].append(float(raw_action.std()))
        diagnostics["env_action_mean"].append(float(env_action.mean()))
        diagnostics["env_action_std"].append(float(env_action.std()))
        diagnostics["action_abs_mean"].append(float(np.abs(env_action).mean()))
        diagnostics["action_saturation_fraction_095"].append(
            float((np.abs(squashed_action) > 0.95).mean())
        )

        for key in (
            "z_distance_from_origin",
            "reward_linup",
            "reward_quadctrl",
            "reward_impact",
        ):
            values = self._info_values(info, key)
            if values is not None:
                diagnostics[key].extend(float(value) for value in values.reshape(-1))

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
            if key in {
                "z_distance_from_origin",
                "reward_linup",
                "raw_action_std",
                "env_action_std",
                "action_abs_mean",
                "action_saturation_fraction_095",
            }:
                summary[f"{key}_max"] = float(array.max())
            if key in {
                "z_distance_from_origin",
                "reward_linup",
                "reward_quadctrl",
                "reward_impact",
            }:
                summary[f"{key}_min"] = float(array.min())
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

    def _policy_diagnostics(self) -> dict[str, float]:
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None

        with torch.no_grad():
            log_std = self.policy_network.log_std.clamp(-5.0, 2.0)
            std = log_std.exp()

        return {
            "policy_log_std_mean": float(log_std.mean().cpu().item()),
            "policy_log_std_min": float(log_std.min().cpu().item()),
            "policy_log_std_max": float(log_std.max().cpu().item()),
            "policy_std_mean": float(std.mean().cpu().item()),
            "policy_std_min": float(std.min().cpu().item()),
            "policy_std_max": float(std.max().cpu().item()),
        }

    def _update_value_metrics(self, buffer: RolloutBatch) -> None:
        if self.value_network is None:
            self._initialize_networks()
        assert self.value_network is not None

        with torch.no_grad():
            current_values = self.value_network(
                self._flatten_rollout_tensor(buffer.observations)
            )
            returns = self._flatten_rollout_tensor(buffer.returns)
            old_values = self._flatten_rollout_tensor(buffer.values)

        self.metrics.update(
            {
                "explained_variance": self._explained_variance(
                    current_values,
                    returns,
                ),
                "old_explained_variance": self._explained_variance(
                    old_values,
                    returns,
                ),
                "value_prediction_mean": float(current_values.mean().cpu().item()),
                "value_prediction_std": float(
                    current_values.std(unbiased=False).cpu().item()
                ),
                "old_value_prediction_mean": float(old_values.mean().cpu().item()),
                "old_value_prediction_std": float(
                    old_values.std(unbiased=False).cpu().item()
                ),
                "value_target_mean": float(returns.mean().cpu().item()),
                "value_target_std": float(
                    returns.std(unbiased=False).cpu().item()
                ),
            }
        )

    def _as_observation_batch(self, observations: Any) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        return array.reshape(self.num_envs, -1)

    def _predict_values(self, observations: np.ndarray) -> np.ndarray:
        if self.value_network is None:
            self._initialize_networks()
        assert self.value_network is not None

        observation_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            values = self.value_network(observation_tensor)
        return values.cpu().numpy().astype(np.float32)

    def _flatten_rollout_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError("Cannot flatten an empty rollout tensor.")
        if tensor.ndim <= 2:
            return tensor.reshape(-1)
        return tensor.reshape(-1, *tensor.shape[2:])

    def _explained_variance(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        target_variance = torch.var(targets, unbiased=False)
        if target_variance.item() == 0.0:
            return 0.0
        residual_variance = torch.var(targets - predictions, unbiased=False)
        return float((1.0 - residual_variance / target_variance).item())

    def _trainable_parameters(self) -> list[nn.Parameter]:
        if self.policy_network is None or self.value_network is None:
            self._initialize_networks()
        assert self.policy_network is not None
        assert self.value_network is not None

        return [
            *self.policy_network.parameters(),
            *self.value_network.parameters(),
        ]
