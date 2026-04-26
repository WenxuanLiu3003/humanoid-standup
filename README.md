# Humanoid Standup

This is a small scaffold for trying several algorithms on Gymnasium's MuJoCo
`HumanoidStandup-v5` task.

The project is intentionally simple:

- environment settings live in `configs/env.yaml`
- each algorithm has one small config file in `configs/algorithms/`
- PPO and TD3 have native PyTorch implementations
- run commands use plain Python files, not package entry points

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Check The Environment

```bash
python src/check_env.py
```

## Start A Training Run

TD3 is a FastTD3-inspired off-policy baseline tuned for local GPU experiments:

```bash
python src/train.py --algo ppo
python src/train.py --algo td3
```

Training writes JSONL metrics and checkpoints under `runs/<algo>/seed_<seed>/...`.
TD3 also runs deterministic evaluation and writes a final rollout video to
`runs/td3/.../videos/final.mp4`.

Resume from a checkpoint:

```bash
python src/train.py --algo td3 --checkpoint runs/td3/seed_0/<run_id>/checkpoints/final.pt
```
