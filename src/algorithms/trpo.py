from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([param.data.reshape(-1) for param in module.parameters()])


def set_flat_params(module: nn.Module, flat_vector: torch.Tensor) -> None:
    pointer = 0
    for param in module.parameters():
        numel = param.numel()
        param.data.copy_(flat_vector[pointer:pointer + numel].view_as(param))
        pointer += numel


def flat_grad(
    output: torch.Tensor,
    params: list[nn.Parameter],
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        output,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=False,
    )
    return torch.cat([grad.reshape(-1) for grad in grads])


def conjugate_gradient(
    Avp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot_r = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = Avp(p)
        alpha = r_dot_r / (torch.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        new_r_dot_r = torch.dot(r, r)
        if new_r_dot_r.item() < tol:
            break
        beta = new_r_dot_r / (r_dot_r + 1e-8)
        p = r + beta * p
        r_dot_r = new_r_dot_r

    return x


class RunningMeanStd:
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


class PolicyNetwork(nn.Module):
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


class ValueNetwork(nn.Module):
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
    raw_action: np.ndarray
    env_action: np.ndarray
    log_prob: np.ndarray
    value: np.ndarray


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    raw_rewards: torch.Tensor
    terminations: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    episode_returns: list[float] | None = None
    episode_lengths: list[int] | None = None
    diagnostics: dict[str, float] | None = None


class TRPO(Algorithm):
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

        # keep the same config structure as PPO
        self.hyperparameters = self.algo_config.get("hyperparameters", {})
        self.collection_config = self.algo_config.get("collection", {})
        self.logging_config = self.algo_config.get("logging", {})
        self.network_config = self.algo_config.get("network", {})
        self.normalization_config = self.algo_config.get("normalization", {})

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

        # rollout / training budget
        self.total_timesteps = int(self.hyperparameters.get("total_timesteps", 1_000_000))
        self.steps_per_env = int(
            self.collection_config.get(
                "steps_per_env",
                self.hyperparameters.get("rollout_steps", 2048),
            )
        )
        self.rollout_steps = self.steps_per_env
        self.rollout_batch_size = self.rollout_steps * self.num_envs
        self.num_updates = self.total_timesteps // self.rollout_batch_size

        # TRPO hyperparameters
        self.gamma = float(self.hyperparameters.get("gamma", 0.99))
        self.gae_lambda = float(self.hyperparameters.get("gae_lambda", 0.95))
        self.target_kl = float(self.hyperparameters.get("target_kl", 0.01))
        self.cg_iters = int(self.hyperparameters.get("cg_iters", 10))
        self.cg_damping = float(self.hyperparameters.get("cg_damping", 0.1))
        self.backtrack_iters = int(self.hyperparameters.get("backtrack_iters", 10))
        self.backtrack_coeff = float(self.hyperparameters.get("backtrack_coeff", 0.8))
        self.accept_ratio = float(self.hyperparameters.get("accept_ratio", 0.1))

        # critic optimization
        self.vf_lr = float(self.hyperparameters.get("vf_lr", 1e-3))
        self.vf_epochs = int(self.hyperparameters.get("vf_epochs", 5))
        self.vf_minibatch_size = int(self.hyperparameters.get("vf_minibatch_size", 256))
        self.max_grad_norm = float(self.hyperparameters.get("max_grad_norm", 0.5))

        # logging / checkpoint
        self.log_interval = int(self.logging_config.get("log_interval", 1))
        self.checkpoint_interval = int(self.logging_config.get("checkpoint_interval", 50))

        # networks
        self.hidden_sizes = tuple(
            int(hidden_size)
            for hidden_size in self.network_config.get("hidden_sizes", (64, 64))
        )
        self.initial_log_std = float(self.network_config.get("initial_log_std", 0.0))

        # normalization
        self.normalize_observations = bool(
            self.normalization_config.get("normalize_observations", True)
        )
        self.observation_clip = float(self.normalization_config.get("observation_clip", 10.0))
        self.normalization_epsilon = float(self.normalization_config.get("epsilon", 1e-8))
        self.reward_scale = float(self.normalization_config.get("reward_scale", 0.01))
        self.normalize_rewards = bool(
            self.normalization_config.get("normalize_rewards", False)
        )
        self.reward_clip = float(self.normalization_config.get("reward_clip", 10.0))

        requested_device = str(self.algo_config.get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        # runtime state
        self.rollout_buffer: RolloutBatch | None = None
        self.policy_network: PolicyNetwork | None = None
        self.value_network: ValueNetwork | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self.metrics: dict[str, Any] = {}

        self._last_observation: np.ndarray | None = None
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int64)
        self._discounted_return = np.zeros(self.num_envs, dtype=np.float64)

        self.global_step = 0
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"

        # diagnostics we will fill later
        self._last_policy_loss = 0.0
        self._last_value_loss = 0.0
        self._last_kl = 0.0
        self._last_entropy = 0.0
        self._last_step_size = 0.0
        self._last_line_search_success = False

        # action transform buffers
        self._action_scale: torch.Tensor | None = None
        self._action_bias: torch.Tensor | None = None
        self._action_log_scale_sum: torch.Tensor | None = None

        if not isinstance(self.observation_space, spaces.Box):
            raise TypeError("TRPO only supports Box observation spaces.")
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("TRPO only supports Box action spaces.")

        observation_dim = int(np.prod(self.observation_space.shape))
        self.observation_rms = RunningMeanStd(shape=(observation_dim,))
        self.return_rms = RunningMeanStd(shape=())

        self.normalize_advantages = bool(
            self.hyperparameters.get("normalize_advantages", True)
            )
        self.best_episode_return_mean = -float("inf")
        self.best_update = 0

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
        del update_index
        self.rollout_buffer = self.sample_trajectory(self.rollout_steps)
        return self.rollout_buffer

    def _initialize_networks(self) -> None:
        if self.policy_network is not None and self.value_network is not None:
            return

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

        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.vf_lr,
        )

    def _select_action(self, observation: Any) -> ActionSample:
        """Sample one action from the current policy and evaluate its log-prob/value."""
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
            raw_action_tensor = distribution.sample()
            log_prob = self._squashed_log_prob(distribution, raw_action_tensor)
            value = self.value_network(observation_tensor)

        raw_action = raw_action_tensor.cpu().numpy()
        env_action = self._env_action_from_raw(raw_action_tensor).cpu().numpy()

        return ActionSample(
            raw_action=raw_action.astype(np.float32),
            env_action=env_action.astype(np.float32),
            log_prob=log_prob.cpu().numpy().astype(np.float32),
            value=value.cpu().numpy().astype(np.float32),
        )


    def sample_trajectory(self, num_steps: int | None = None) -> RolloutBatch:
        """Collect one fixed-length rollout from the current policy."""
        if self.policy_network is None or self.value_network is None:
            self._initialize_networks()
        if self._last_observation is None:
            self._last_observation, _ = self.env.reset(seed=self.seed)
            self._last_observation = self._as_observation_batch(self._last_observation)

        steps = int(self.rollout_steps if num_steps is None else num_steps)
        if steps <= 0:
            raise ValueError("Trajectory length must be positive.")

        observations: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
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
                raw_action=action_sample.raw_action,
                env_action=action_sample.env_action,
            )

            observations.append(observation)
            raw_actions.append(action_sample.raw_action)
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
            raw_actions=torch.as_tensor(
                np.asarray(raw_actions),
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


    def _compute_returns_and_advantages(self) -> None:
        """Compute GAE advantages and bootstrap returns."""
        if self.rollout_buffer is None:
            raise RuntimeError("Cannot compute advantages before collecting a rollout.")
        if self.value_network is None:
            self._initialize_networks()

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
                **(buffer.diagnostics or {}),
            }
        )

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


    def _policy_parameters(self) -> list[nn.Parameter]:
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None
        return [param for param in self.policy_network.parameters() if param.requires_grad]


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


    def _surrogate_objective(
        self,
        observations: torch.Tensor,
        raw_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None

        distribution = self.policy_network.distribution(observations)
        new_log_probs = self._squashed_log_prob(distribution, raw_actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        return (ratio * advantages).mean()


    def _mean_kl(
        self,
        observations: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
    ) -> torch.Tensor:
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None

        new_distribution = self.policy_network.distribution(observations)
        old_distribution = Normal(old_mean, old_std)

        # KL is computed on the pre-tanh Gaussian.
        # Because old and new policies use the same tanh squashing map,
        # this is the cleanest way to impose the trust region.
        kl = torch.distributions.kl.kl_divergence(
            old_distribution,
            new_distribution,
        ).sum(dim=-1).mean()
        return kl


    def _fisher_vector_product(
        self,
        observations: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        params = self._policy_parameters()

        mean_kl = self._mean_kl(observations, old_mean, old_std)
        kl_grad = flat_grad(
            mean_kl,
            params,
            retain_graph=True,
            create_graph=True,
        )
        kl_grad_vector = torch.dot(kl_grad, vector)
        hessian_vector = flat_grad(
            kl_grad_vector,
            params,
            retain_graph=False,
            create_graph=False,
        )

        return hessian_vector + self.cg_damping * vector


    def _line_search(
        self,
        observations: torch.Tensor,
        raw_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        old_params: torch.Tensor,
        full_step: torch.Tensor,
        old_surrogate: float,
        expected_improve_rate: float,
    ) -> tuple[bool, float, float, float]:
        if self.policy_network is None:
            self._initialize_networks()
        assert self.policy_network is not None

        for step_index in range(self.backtrack_iters):
            step_fraction = self.backtrack_coeff ** step_index
            candidate_params = old_params + step_fraction * full_step
            set_flat_params(self.policy_network, candidate_params)

            with torch.no_grad():
                new_surrogate = float(
                    self._surrogate_objective(
                        observations,
                        raw_actions,
                        old_log_probs,
                        advantages,
                    ).item()
                )
                new_kl = float(
                    self._mean_kl(
                        observations,
                        old_mean,
                        old_std,
                    ).item()
                )

            actual_improve = new_surrogate - old_surrogate
            expected_improve = step_fraction * expected_improve_rate

            if not np.isfinite(new_surrogate) or not np.isfinite(new_kl):
                continue

            if (
                actual_improve > 0.0
                and new_kl <= self.target_kl
                and actual_improve >= self.accept_ratio * expected_improve
            ):
                return True, step_fraction, new_surrogate, new_kl

        set_flat_params(self.policy_network, old_params)
        return False, 0.0, old_surrogate, 0.0


    def _update_value_function(
        self,
        observations: torch.Tensor,
        returns: torch.Tensor,
    ) -> float:
        if self.value_network is None or self.value_optimizer is None:
            self._initialize_networks()

        assert self.value_network is not None
        assert self.value_optimizer is not None

        losses: list[float] = []
        batch_size = observations.shape[0]
        minibatch_size = min(self.vf_minibatch_size, batch_size)

        for _ in range(self.vf_epochs):
            permutation = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]

                predicted_values = self.value_network(observations[indices])
                value_loss = 0.5 * torch.square(
                    predicted_values - returns[indices]
                ).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_network.parameters(),
                    self.max_grad_norm,
                )
                self.value_optimizer.step()

                losses.append(float(value_loss.detach().cpu().item()))

        return float(np.mean(losses)) if losses else 0.0

    def _update_policy(self, update_index: int) -> None:
        del update_index

        if self.rollout_buffer is None:
            raise RuntimeError("Cannot update TRPO before collecting a rollout.")
        if self.rollout_buffer.advantages is None or self.rollout_buffer.returns is None:
            raise RuntimeError("Cannot update TRPO before computing advantages.")
        if self.policy_network is None or self.value_network is None:
            self._initialize_networks()

        assert self.policy_network is not None
        assert self.value_network is not None

        buffer = self.rollout_buffer
        assert buffer.advantages is not None
        assert buffer.returns is not None

        observations = self._flatten_rollout_tensor(buffer.observations)
        raw_actions = self._flatten_rollout_tensor(buffer.raw_actions)
        old_log_probs = self._flatten_rollout_tensor(buffer.log_probs)
        advantages = self._flatten_rollout_tensor(buffer.advantages)
        returns = self._flatten_rollout_tensor(buffer.returns)
        old_values = self._flatten_rollout_tensor(buffer.values)

        # Save old policy statistics for the trust-region constraint
        with torch.no_grad():
            old_distribution = self.policy_network.distribution(observations)
            old_mean = old_distribution.loc.detach()
            old_std = old_distribution.scale.detach()
            old_entropy = float(old_distribution.entropy().sum(dim=-1).mean().cpu().item())

        # Compute policy gradient of the surrogate objective
        surrogate = self._surrogate_objective(
            observations,
            raw_actions,
            old_log_probs,
            advantages,
        )
        old_surrogate = float(surrogate.detach().cpu().item())

        params = self._policy_parameters()
        policy_grad = flat_grad(
            surrogate,
            params,
            retain_graph=True,
            create_graph=False,
        ).detach()

        grad_norm = float(torch.norm(policy_grad).cpu().item())
        line_search_success = False
        line_search_fraction = 0.0
        final_surrogate = old_surrogate
        final_kl = 0.0
        full_step_norm = 0.0

        # Only try a TRPO step if the gradient is nontrivial
        if torch.isfinite(policy_grad).all() and torch.norm(policy_grad).item() > 1e-12:
            def fisher_vector_product_fn(vector: torch.Tensor) -> torch.Tensor:
                return self._fisher_vector_product(
                    observations,
                    old_mean,
                    old_std,
                    vector,
                )

            step_direction = conjugate_gradient(
                fisher_vector_product_fn,
                policy_grad,
                max_iter=self.cg_iters,
            )

            fisher_step = fisher_vector_product_fn(step_direction)
            shs = 0.5 * torch.dot(step_direction, fisher_step)

            if torch.isfinite(shs).item() and shs.item() > 0.0:
                lagrange_multiplier = torch.sqrt(shs / self.target_kl + 1e-8)
                full_step = step_direction / (lagrange_multiplier + 1e-8)
                full_step_norm = float(torch.norm(full_step).cpu().item())

                expected_improve_rate = float(torch.dot(policy_grad, full_step).cpu().item())
                old_params = flat_params(self.policy_network).detach().clone()

                (
                    line_search_success,
                    line_search_fraction,
                    final_surrogate,
                    final_kl,
                ) = self._line_search(
                    observations,
                    raw_actions,
                    old_log_probs,
                    advantages,
                    old_mean,
                    old_std,
                    old_params,
                    full_step,
                    old_surrogate,
                    expected_improve_rate,
                )
            else:
                final_kl = 0.0

        # Critic update happens regardless of whether the policy step succeeded
        value_loss = self._update_value_function(observations, returns)

        with torch.no_grad():
            current_values = self.value_network(observations)
            current_distribution = self.policy_network.distribution(observations)
            current_entropy = float(
                current_distribution.entropy().sum(dim=-1).mean().cpu().item()
            )
            if final_kl == 0.0:
                final_kl = float(
                    self._mean_kl(
                        observations,
                        old_mean,
                        old_std,
                    ).cpu().item()
                )

        self._last_policy_loss = -final_surrogate
        self._last_value_loss = value_loss
        self._last_kl = final_kl
        self._last_entropy = current_entropy
        self._last_step_size = line_search_fraction
        self._last_line_search_success = line_search_success

        self.metrics.update(
            {
                "policy_loss": float(-final_surrogate),
                "value_loss": float(value_loss),
                "surrogate_before": float(old_surrogate),
                "surrogate_after": float(final_surrogate),
                "kl": float(final_kl),
                "policy_grad_norm": float(grad_norm),
                "policy_step_norm": float(full_step_norm),
                "line_search_success": bool(line_search_success),
                "line_search_fraction": float(line_search_fraction),
                "gaussian_entropy_old": float(old_entropy),
                "gaussian_entropy_new": float(current_entropy),
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
                "value_target_mean": float(returns.mean().cpu().item()),
                "value_target_std": float(
                    returns.std(unbiased=False).cpu().item()
                ),
            }
        )

    def _log_update(self, update_index: int) -> None:
        """Write metrics for the current TRPO update."""
        if self.rollout_buffer is None:
            raise RuntimeError("Cannot log TRPO update before collecting a rollout.")

        update = update_index + 1
        buffer = self.rollout_buffer
        complete_episode_returns = buffer.episode_returns or []
        complete_episode_lengths = buffer.episode_lengths or []

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

        # Save best checkpoint based on training episode return mean
        if "episode_return_mean" in record:
            current_score = float(record["episode_return_mean"])
            if current_score > self.best_episode_return_mean:
                self.best_episode_return_mean = current_score
                self.best_update = update
                self.save(self.checkpoint_dir / "best.pt")

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
                f"kl={record.get('kl', 0.0):.6f} "
                f"step_norm={record.get('policy_step_norm', 0.0):.4f} "
                f"line_search={record.get('line_search_success', False)} "
                f"ev={record.get('explained_variance', 0.0):.4f}"
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
        """Save the final checkpoint."""
        self.save(self.checkpoint_dir / "final.pt")


    def save(self, path: Path) -> None:
        """Save TRPO model, optimizer, normalization state, and metadata."""
        if (
            self.policy_network is None
            or self.value_network is None
            or self.value_optimizer is None
        ):
            self._initialize_networks()

        assert self.policy_network is not None
        assert self.value_network is not None
        assert self.value_optimizer is not None

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "value_network": self.value_network.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "global_step": self.global_step,
                "hyperparameters": self.hyperparameters,
                "network": {
                    "hidden_sizes": list(self.hidden_sizes),
                    "initial_log_std": self.initial_log_std,
                },
                "normalization": {
                    "observation_rms": {
                        "mean": self.observation_rms.mean,
                        "var": self.observation_rms.var,
                        "count": self.observation_rms.count,
                    },
                    "return_rms": {
                        "mean": self.return_rms.mean,
                        "var": self.return_rms.var,
                        "count": self.return_rms.count,
                    },
                    "discounted_return": self._discounted_return,
                },
                "env_config": self.env_config,
                "algo_config": self.algo_config,
                "best_episode_return_mean": self.best_episode_return_mean,
                "best_update": self.best_update,
            },
            path,
        )


    def load(self, path: Path) -> None:
        """Load TRPO model, optimizer, and normalization state."""
        # checkpoint = torch.load(path, map_location=self.device)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
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
        assert self.value_optimizer is not None

        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))

        normalization_state = checkpoint.get("normalization", {})

        observation_rms_state = normalization_state.get("observation_rms")
        if observation_rms_state is not None:
            self.observation_rms.mean = np.asarray(observation_rms_state["mean"], dtype=np.float64)
            self.observation_rms.var = np.asarray(observation_rms_state["var"], dtype=np.float64)
            self.observation_rms.count = float(observation_rms_state["count"])

        return_rms_state = normalization_state.get("return_rms")
        if return_rms_state is not None:
            self.return_rms.mean = np.asarray(return_rms_state["mean"], dtype=np.float64)
            self.return_rms.var = np.asarray(return_rms_state["var"], dtype=np.float64)
            self.return_rms.count = float(return_rms_state["count"])

        self._discounted_return = np.asarray(
            normalization_state.get("discounted_return", self._discounted_return),
            dtype=np.float64,
        )

        self.best_episode_return_mean = float(
            checkpoint.get("best_episode_return_mean", -float("inf"))
            )
        self.best_update = int(checkpoint.get("best_update", 0))

    def load_policy_only(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._initialize_networks()
        self.policy_network.load_state_dict(checkpoint["policy_network"])

    def _as_observation_batch(self, observations: Any) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        return array.reshape(self.num_envs, -1)
