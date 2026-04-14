from __future__ import annotations

from algorithms.a2c import A2C
from algorithms.base import Algorithm
from algorithms.ddpg import DDPG
from algorithms.ppo import PPO
from algorithms.sac import SAC
from algorithms.td3 import TD3


ALGORITHMS: dict[str, type[Algorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def get_algorithm(name: str) -> type[Algorithm]:
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}") from exc
