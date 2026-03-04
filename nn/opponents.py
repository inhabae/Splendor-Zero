from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .checkpoints import load_checkpoint
from .mcts import MCTSConfig, run_mcts


class OpponentPolicy(Protocol):
    name: str

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        ...


@dataclass
class RandomOpponent:
    name: str = "random"

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        legal = np.flatnonzero(state.mask)
        if legal.size == 0:
            raise RuntimeError("RandomOpponent: no legal actions")
        return int(rng.choice(legal.tolist()))


@dataclass
class GreedyHeuristicOpponent:
    name: str = "heuristic"

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        if not hasattr(env, "heuristic_action"):
            raise RuntimeError("GreedyHeuristicOpponent requires native env.heuristic_action()")
        return int(env.heuristic_action())


@dataclass
class ModelMCTSOpponent:
    model: Any
    mcts_config: MCTSConfig
    device: str = "cpu"
    name: str = "mcts_model"

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        result = run_mcts(
            env,
            self.model,
            state,
            turns_taken=turns_taken,
            device=self.device,
            config=self.mcts_config,
            rng=rng,
        )
        return int(result.chosen_action_idx)


@dataclass
class CheckpointMCTSOpponent:
    checkpoint_path: str
    mcts_config: MCTSConfig
    device: str = "cpu"
    name: str = "checkpoint_mcts"

    def __post_init__(self) -> None:
        self._model = load_checkpoint(self.checkpoint_path, device=self.device)

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        result = run_mcts(
            env,
            self._model,
            state,
            turns_taken=turns_taken,
            device=self.device,
            config=self.mcts_config,
            rng=rng,
        )
        return int(result.chosen_action_idx)
