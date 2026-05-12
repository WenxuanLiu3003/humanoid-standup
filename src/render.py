"""
Visualise or record a trained SAC policy.

Usage:
    # Live window
    python render.py --checkpoint ..\runs\sac\seed_0\<ts>\checkpoints\final.pt

    # Save MP4 video
    python render.py --checkpoint ..\runs\sac\seed_0\<ts>\checkpoints\final.pt --video
    python render.py --checkpoint ..\runs\sac\seed_0\<ts>\checkpoints\final.pt --video --out my_video
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from algorithms.sac import PolicyNetwork, RunningMeanStd
from config import load_env_config
from env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes",   type=int,  default=3)
    parser.add_argument("--video",      action="store_true", help="Save MP4 instead of opening a window")
    parser.add_argument("--out",        type=str,  default="videos", help="Output folder for video files")
    return parser.parse_args()


def make_policy(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Infer dims from saved weights
    first_layer = ckpt["policy"]["net.0.weight"]
    last_layer  = ckpt["policy"]["mu_head.weight"]
    obs_dim     = first_layer.shape[1]
    act_dim     = last_layer.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    # Restore action scale from checkpoint if available, else default 0.4
    action_scale = float(ckpt.get("action_scale", 0.4))
    policy.action_scale = action_scale

    return policy, obs_rms


def run_episodes(env, policy, obs_rms, n_episodes: int):
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(obs_rms.normalize(state))
                action, _ = policy.sample(s)
                action = action.numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep}: reward = {total_reward:.1f}")


def main() -> None:
    args  = parse_args()
    policy, obs_rms = make_policy(args.checkpoint)
    env_config = load_env_config()

    if args.video:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = make_env(env_config, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=str(out_dir),
            episode_trigger=lambda ep: True,   # record every episode
            name_prefix=args.checkpoint.stem,
        )
        print(f"Recording {args.episodes} episode(s) → {out_dir.resolve()}/")
    else:
        env = make_env(env_config, render_mode="human")

    run_episodes(env, policy, obs_rms, args.episodes)
    env.close()

    if args.video:
        files = sorted(out_dir.glob("*.mp4"))
        print(f"Saved {len(files)} video(s):")
        for f in files:
            print(f"  {f.resolve()}")


if __name__ == "__main__":
    main()
