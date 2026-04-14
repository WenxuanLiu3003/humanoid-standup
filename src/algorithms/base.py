from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym


class Algorithm(ABC):
    def __init__(
        self,
        *,
        env: gym.Env,
        env_config: dict[str, Any],
        algo_config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        self.env = env
        self.env_config = env_config
        self.algo_config = algo_config
        self.run_dir = run_dir

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
