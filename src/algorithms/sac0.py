"""
SAC v1 baseline — Algorithm 1 from Haarnoja et al. (2018).
"""
from __future__ import annotations

import copy
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from algorithms.base import Algorithm


class SAC0(Algorithm):

    def __init__(
        self,
        *,
        env: gym.Env,
        env_config: dict[str, Any],
        algo_config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        super().__init__(env=env, env_config=env_config, algo_config=algo_config, run_dir=run_dir)

        hp        = algo_config.get("hyperparameters", {})
        self.hp   = hp
        self.seed = int(env_config.get("seed", 0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space",      env.action_space)
        obs_dim   = obs_space.shape[0]
        act_dim   = act_space.shape[0]

        self.action_scale = float(env_config.get("action_scale", 0.4))
        self.replay_buffer = ReplayBuffer(hp, obs_dim, act_dim)
        self.V = ValueNetwork(obs_dim).to(self.device)
        self.V_target = copy.deepcopy(self.V)
        self.Q1  = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q2 = QNetwork(obs_dim, act_dim).to(self.device)
        self.policy = PolicyNetwork(obs_dim, act_dim, self.action_scale).to(self.device)

        for p in self.V_target.parameters():
            p.requires_grad = False

        lr         = float(hp.get("learning_rate", 3e-4))
        self.alpha = float(hp.get("alpha", 0.2))

        self.V_opt = torch.optim.Adam(self.V.parameters(), lr=lr)
        self.Q1_opt = torch.optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_opt= torch.optim.Adam(self.Q2.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _update(self, gamma: float, tau: float) -> None:
        s, a, r, s_, d = self.replay_buffer.sample(self.device)

        reward_scale = float(self.hp.get("reward_scale", 20.0))

        # Update V
        with torch.no_grad():
            a_pi, log_pi = self.policy.sample(s)
            v_tgt = (
                torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
                - self.alpha * log_pi.unsqueeze(1)
            )

        loss_V = 0.5 * (self.V(s) - v_tgt).pow(2).mean()
        self.V_opt.zero_grad()
        loss_V.backward()
        nn.utils.clip_grad_norm_(self.V.parameters(), max_norm=10.0)
        self.V_opt.step()

        # Update Q
        with torch.no_grad():
            q_hat = reward_scale * r + gamma * (1.0 - d) * self.V_target(s_)

        loss_Q1 = 0.5 * (self.Q1(s, a) - q_hat).pow(2).mean()
        self.Q1_opt.zero_grad()
        loss_Q1.backward()
        nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=10.0)
        self.Q1_opt.step()

        loss_Q2 = 0.5 * (self.Q2(s, a) - q_hat).pow(2).mean()
        self.Q2_opt.zero_grad()
        loss_Q2.backward()
        nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=10.0)
        self.Q2_opt.step()

        # Update π
        a_pi, log_pi = self.policy.sample(s)
        loss_policy = (
            self.alpha * log_pi.unsqueeze(1)
            - torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
        ).mean()
        self.policy_opt.zero_grad()
        loss_policy.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.policy_opt.step()

        # Soft update of target V
        with torch.no_grad():
            for p, p_t in zip(self.V.parameters(), self.V_target.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)


    def train(self) -> None:
        hp = self.hp
        total_timesteps = int(hp.get("total_timesteps", 3_000_000))
        learning_starts = int(hp.get("learning_starts", 10_000))
        gradient_steps  = int(hp.get("gradient_steps", 1))
        gamma = float(hp.get("gamma", 0.99))
        tau = float(hp.get("tau", 0.005))
        log_interval = int(hp.get("log_interval", 5_000))
        save_interval = int(hp.get("save_interval", 100_000))

        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path   = self.run_dir / "metrics.jsonl"
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_count  = 0
        recent_rewards: deque = deque(maxlen=10)

        for t in range(1, total_timesteps + 1):
            if t < learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).to(self.device)
                    action, _ = self.policy.sample(s_t)
                    action = action.cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated


            self.replay_buffer.add(state, action, reward, next_state, float(terminated))
            episode_reward += reward
            state = next_state

            if done:
                episode_count += 1
                recent_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0.0


            if t >= learning_starts and len(self.replay_buffer) >= learning_starts:
                for _ in range(gradient_steps):
                    self._update(gamma, tau)

            if t % log_interval == 0 and recent_rewards:
                mean_r = sum(recent_rewards) / len(recent_rewards)
                metrics = {
                    "timestep": t,
                    "episodes": episode_count,
                    "mean_reward_10ep": mean_r,
                    "alpha": self.alpha,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(
                    f"[t={t}] episodes={episode_count} "
                    f"mean_reward(10ep)={mean_r:.1f} "
                    f"alpha={self.alpha:.4f}",
                    flush=True,
                )

            if t % save_interval == 0:
                self._save_checkpoint(checkpoint_dir / f"step_{t}.pt")

        self._save_checkpoint(checkpoint_dir / "final.pt")
        print("Training complete. Checkpoint saved.")

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "V": self.V.state_dict(),
                "Q1": self.Q1.state_dict(),
                "Q2": self.Q2.state_dict(),
                "action_scale": self.action_scale,
            },
            path,
        )



class ReplayBuffer:
    """
    Numpy circular-array buffer — O(1) add and O(1) sample regardless of buffer size.
    (Replaces the deque-based version which had O(N) random-access sampling.)
    """

    def __init__(self, hp: dict, obs_dim: int, act_dim: int):
        maxlen = int(hp.get("buffer_maxlength",  1_000_000))
        self._batchsize = int(hp.get("buffer_batch_size", 256))

        self._obs  = np.zeros((maxlen, obs_dim), dtype=np.float32)
        self._acts = np.zeros((maxlen, act_dim), dtype=np.float32)
        self._rews = np.zeros((maxlen, 1), dtype=np.float32)
        self._next = np.zeros((maxlen, obs_dim), dtype=np.float32)
        self._term = np.zeros((maxlen, 1), dtype=np.float32)
        self._maxlen = maxlen
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, terminated: float) -> None:
        self._obs[self._ptr]  = obs
        self._acts[self._ptr] = action
        self._rews[self._ptr] = reward
        self._next[self._ptr] = next_obs
        self._term[self._ptr] = terminated
        self._ptr  = (self._ptr + 1) % self._maxlen
        self._size = min(self._size + 1, self._maxlen)

    def sample(self, device: torch.device):
        idx   = np.random.randint(0, self._size, size=self._batchsize)
        to_t  = lambda x: torch.FloatTensor(x[idx]).to(device)
        return to_t(self._obs), to_t(self._acts), to_t(self._rews), to_t(self._next), to_t(self._term)

    def __len__(self) -> int:
        return self._size


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PolicyNetwork(nn.Module):
    """
    Squashed-Gaussian policy — Eq. 11:  a_t = tanh(f_φ(ε_t; s_t)) * scale
    Samples via rsample() (reparameterisation trick).
    """

    def __init__(self, obs_dim: int, act_dim: int, action_scale: float = 0.4, hidden: int = 256):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def sample(self, state: torch.Tensor):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-20, 2)
        std = log_std.exp()

        dist = Normal(mu, std)
        u = dist.rsample()                          # reparameterised sample (Eq. 11)
        action = torch.tanh(u) * self.action_scale

        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= (
            math.log(self.action_scale)
            + torch.log(1.0 - (action / self.action_scale).pow(2) + 1e-6)
        ).sum(-1)

        return action, log_prob
