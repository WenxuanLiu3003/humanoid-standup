from __future__ import annotations

from algorithms.base import Algorithm
from pathlib import Path
from typing import Any
from collections import deque
import gymnasium as gym
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy


class RunningMeanStd:

    def __init__(self, shape: tuple):
        self.mean= np.zeros(shape, dtype=np.float64)
        self.var= np.ones(shape,  dtype=np.float64)
        self.count= 1e-4

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis]
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean= np.asarray(d["mean"],  dtype=np.float64)
        self.var= np.asarray(d["var"],   dtype=np.float64)
        self.count= float(d["count"])


class SAC(Algorithm):
    def __init__(self, *, env: gym.Env, env_config: dict[str, Any], algo_config: dict[str, Any], run_dir: Path) -> None:
        super().__init__(
            env=env,
            env_config=env_config,
            algo_config=algo_config,
            run_dir=run_dir,
        )

        self.hp = algo_config.get("hyperparameters", {})
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.observation_space = getattr(env, "single_observation_space", env.observation_space)
        self.action_space = getattr(env, "single_action_space", env.action_space)
        self.seed = int(env_config.get("seed", 0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        action_scale = float(self.action_space.high[0])

        lr = float(self.hp.get("learning_rate", 3e-4))

        self.Q1 = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q2 = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2_target = copy.deepcopy(self.Q2)
        self.policy = PolicyNetwork(obs_dim, act_dim, action_scale=action_scale).to(self.device)

        for p in self.Q1_target.parameters():
            p.requires_grad = False
        for p in self.Q2_target.parameters():
            p.requires_grad = False

        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Automatic entropy tuning
        raw_te = self.hp.get("target_entropy", "auto")
        self.target_entropy = -float(act_dim) if raw_te == "auto" else float(raw_te)
        init_alpha = self.hp.get("alpha", 0.2)
        self.log_alpha = torch.tensor(
            math.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, hp=self.hp, device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
        std  = torch.as_tensor(np.sqrt(self.obs_rms.var) + 1e-8, dtype=torch.float32, device=self.device)
        return (obs - mean) / std

    def _update(self, gamma_n: float, tau: float) -> dict:
        """gamma_n = gamma ** n_steps — the effective bootstrap discount."""
        s_raw, a, r, s_next_raw, terminated = self.replay_buffer.sample()
        s  = self._normalize(s_raw)
        s_ = self._normalize(s_next_raw)

        alpha = self.alpha.detach()

        # Critic update
        with torch.no_grad():
            a_next, log_prob_next = self.policy.sample(s_)
            q_next = torch.min(self.Q1_target(s_, a_next), self.Q2_target(s_, a_next))
            q_hat = r + gamma_n * (1.0 - terminated) * (q_next - alpha * log_prob_next)

        loss_Q1 = 0.5 * (self.Q1(s, a) - q_hat).pow(2).mean()
        loss_Q2 = 0.5 * (self.Q2(s, a) - q_hat).pow(2).mean()

        self.Q1_optimizer.zero_grad()
        loss_Q1.backward()
        nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=10.0)
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        loss_Q2.backward()
        nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=10.0)
        self.Q2_optimizer.step()

        # Actor update
        a_new, log_prob = self.policy.sample(s)
        q_pi = torch.min(self.Q1(s, a_new), self.Q2(s, a_new))
        loss_policy = (alpha * log_prob - q_pi).mean()

        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.policy_optimizer.step()

        loss_alpha = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()

        for p, p_tgt in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            p_tgt.data.mul_(1 - tau).add_(tau * p.data)
        for p, p_tgt in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            p_tgt.data.mul_(1 - tau).add_(tau * p.data)

        return {
            "loss_Q1": loss_Q1.item(),
            "loss_Q2": loss_Q2.item(),
            "loss_policy": loss_policy.item(),
            "alpha": self.alpha.item(),
        }

    def train(self):
        hp = self.hp
        total_timesteps = hp.get("total_timesteps", 3_000_000)
        learning_starts = hp.get("learning_starts", 10_000)
        gradient_steps = int(hp.get("gradient_steps", 1))
        gamma = float(hp.get("gamma", 0.99))
        tau = float(hp.get("tau", 0.005))
        n_steps= int(hp.get("n_steps", 1))
        log_interval= hp.get("log_interval", 5_000)
        save_interval = hp.get("save_interval", 100_000)

        gamma_n = gamma ** n_steps  # effective bootstrap discount for n-step targets
        nstep_buf = NStepBuffer(n=n_steps, gamma=gamma)

        metrics_path = self.run_dir / "metrics.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_count  = 0
        recent_rewards: list[float] = []

        for t in range(1, total_timesteps + 1):
            self.obs_rms.update(state)

            if t < learning_starts:
                action = self.action_space.sample()
            else:
                with torch.no_grad():
                    s_norm = self.obs_rms.normalize(state)
                    s_t = torch.FloatTensor(s_norm).to(self.device)
                    action, _ = self.policy.sample(s_t)
                    action = action.cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            for (s, a, R, sn, term) in nstep_buf.add(state, action, reward, next_state,
                                                       float(terminated), float(truncated)):
                self.replay_buffer.add(s, a, R, sn, term)

            episode_reward += reward
            state = next_state

            if done:
                episode_count += 1
                recent_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0.0

            if len(self.replay_buffer) < learning_starts:
                continue

            for _ in range(gradient_steps):
                self._update(gamma_n, tau)

            if t % 1000 == 0 and t % log_interval != 0:
                print(f"[t={t}] running...", flush=True)

            if t % log_interval == 0:
                mean_r = float(np.mean(recent_rewards[-10:])) if recent_rewards else 0.0
                metrics = {
                    "timestep": t,
                    "episodes": episode_count,
                    "mean_reward_10ep": mean_r,
                    "alpha": self.alpha.item(),
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(f"[t={t}] episodes={episode_count} mean_reward(10ep)={mean_r:.1f} alpha={self.alpha.item():.4f}")

            if t % save_interval == 0:
                self._save_checkpoint(checkpoint_dir / f"step_{t}.pt")

        self._save_checkpoint(checkpoint_dir / "final.pt")
        print("Training complete. Checkpoint saved.")

    def _save_checkpoint(self, path: Path):
        rms = self.obs_rms.state_dict()
        torch.save({
            "policy":self.policy.state_dict(),
            "Q1": self.Q1.state_dict(),
            "Q2": self.Q2.state_dict(),
            "log_alpha":self.log_alpha.detach().cpu(),
            "obs_rms":{k: v.tolist() for k, v in rms.items()},
        }, path)


class NStepBuffer:

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self._window: deque = deque()

    def add(self, obs, action, reward, next_obs,
            terminated: float, truncated: float = 0.0) -> list:
        self._window.append((obs, action, float(reward), next_obs, terminated, truncated))

        ready = []
        if terminated or truncated:
            while self._window:
                ready.append(self._compute())
                self._window.popleft()
        elif len(self._window) >= self.n:
            ready.append(self._compute())
            self._window.popleft()
        return ready

    def _compute(self) -> tuple:
        s0, a0 = self._window[0][0], self._window[0][1]
        R = 0.0
        for i, (_, _, r, s_next, term, trunc) in enumerate(self._window):
            R += (self.gamma ** i) * r
            if term:
                return s0, a0, R, s_next, 1.0
            if trunc:
                return s0, a0, R, s_next, 0.0
        _, _, _, s_n, term_n, _ = self._window[-1]
        return s0, a0, R, s_n, term_n


class ReplayBuffer:
    def __init__(self, *, obs_dim: int, act_dim: int, hp: dict[str, Any], device: torch.device):
        self.maxlength = hp.get("buffer_maxlength", 1_000_000)
        self.batchsize = hp.get("buffer_batch_size", 256)
        self.device = device

        self._obs = np.zeros((self.maxlength, obs_dim), dtype=np.float32)
        self._acts = np.zeros((self.maxlength, act_dim), dtype=np.float32)
        self._rews = np.zeros((self.maxlength, 1), dtype=np.float32)
        self._next = np.zeros((self.maxlength, obs_dim), dtype=np.float32)
        self._term = np.zeros((self.maxlength, 1),dtype=np.float32)

        self._ptr  = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, terminated: float):
        self._obs[self._ptr] = obs
        self._acts[self._ptr] = action
        self._rews[self._ptr] = reward
        self._next[self._ptr] = next_obs
        self._term[self._ptr] = terminated
        self._ptr = (self._ptr + 1) % self.maxlength
        self._size = min(self._size + 1, self.maxlength)

    def sample(self):
        idx = np.random.randint(0, self._size, size=self.batchsize)
        to_t = lambda x: torch.FloatTensor(x[idx]).to(self.device)
        return to_t(self._obs), to_t(self._acts), to_t(self._rews), to_t(self._next), to_t(self._term)

    def __len__(self):
        return self._size


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class PolicyNetwork(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256, action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mu_head= nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def sample(self, state: torch.Tensor):
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)
        u = dist.rsample() 
        action = torch.tanh(u) * self.action_scale

        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        log_prob -= (
            math.log(self.action_scale)
            + 2 * (math.log(2) - u - F.softplus(-2 * u))
        ).sum(-1, keepdim=True)

        return action, log_prob
