# Humanoid Standup

This is a small scaffold for trying several algorithms on Gymnasium's MuJoCo
`HumanoidStandup-v5` task.

The project is intentionally simple:

- environment settings live in `configs/env.yaml`
- each algorithm has one small config file in `configs/algorithms/`
- algorithm implementations are placeholders for now
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

## Start A Future Training Run

The command shape is already fixed, but algorithm details are not implemented yet:

```bash
python src/train.py --algo ppo
python src/train.py --algo sac
python src/train.py --algo td3
```

At this stage these commands load the environment config and algorithm config, then
raise a clear `NotImplementedError`.
