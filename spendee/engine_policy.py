from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from nn.checkpoints import load_checkpoint
from nn.ismcts import ISMCTSConfig, run_ismcts
from nn.mcts import MCTSConfig, run_mcts
from nn.native_env import SplendorNativeEnv

from .determinize import build_root_determinized_payload
from .shadow_state import ShadowState


@dataclass
class DeterminizedPolicyResult:
    action_idx: int
    visit_probs: np.ndarray
    root_best_value_mean: float
    num_determinizations: int


@dataclass
class DeterminizedMCTSPolicy:
    checkpoint_path: str
    mcts_config: MCTSConfig
    device: str = "cpu"
    determinization_samples: int = 1
    search_type: Literal["mcts", "ismcts"] = "mcts"

    def __post_init__(self) -> None:
        self._model = load_checkpoint(self.checkpoint_path, device=self.device)

    def _run_search(
        self,
        env: SplendorNativeEnv,
        state,
        *,
        turns_taken: int,
        rng: random.Random,
    ):
        if self.search_type == "ismcts":
            return run_ismcts(
                env,
                self._model,
                state=state,
                turns_taken=int(turns_taken),
                device=self.device,
                config=ISMCTSConfig(
                    num_simulations=int(self.mcts_config.num_simulations),
                    c_puct=float(self.mcts_config.c_puct),
                    eval_batch_size=1,
                ),
                rng=rng,
            )
        if self.search_type == "mcts":
            return run_mcts(
                env,
                self._model,
                state,
                turns_taken=int(turns_taken),
                device=self.device,
                config=self.mcts_config,
                rng=rng,
            )
        raise ValueError(f"Unsupported search_type: {self.search_type}")

    def _choose_action_from_payload(
        self,
        payload: dict[str, object],
        *,
        turns_taken: int,
        rng: random.Random,
    ) -> DeterminizedPolicyResult:
        with SplendorNativeEnv() as env:
            state = env.load_state(payload)
            result = self._run_search(env, state, turns_taken=int(turns_taken), rng=rng)

        visit_probs = np.asarray(result.visit_probs, dtype=np.float32)
        action_idx = int(np.argmax(visit_probs))
        return DeterminizedPolicyResult(
            action_idx=action_idx,
            visit_probs=visit_probs.astype(np.float32, copy=False),
            root_best_value_mean=float(result.root_best_value),
            num_determinizations=1,
        )

    def choose_action(self, shadow: ShadowState, *, rng: random.Random | None = None) -> DeterminizedPolicyResult:
        random_source = rng or random.Random()
        payload = build_root_determinized_payload(shadow, rng=random_source)
        return self._choose_action_from_payload(
            payload,
            turns_taken=int(payload.get("move_number", 0)),
            rng=random_source,
        )

    def choose_return_actions(
        self,
        shadow: ShadowState,
        *,
        rng: random.Random | None = None,
    ) -> tuple[DeterminizedPolicyResult, list[int]]:
        random_source = rng or random.Random()
        payload = build_root_determinized_payload(shadow, rng=random_source)
        turns_taken = int(payload.get("move_number", 0))
        chosen: list[int] = []
        first_result: DeterminizedPolicyResult | None = None

        with SplendorNativeEnv() as env:
            state = env.load_state(payload)
            while True:
                exported = env.export_state()
                phase_flags = dict(exported.get("phase_flags", {}))
                if not bool(phase_flags.get("is_return_phase")):
                    break

                result = self._run_search(env, state, turns_taken=turns_taken, rng=random_source)
                visit_probs = np.asarray(result.visit_probs, dtype=np.float32)
                action_idx = int(np.argmax(visit_probs))
                if action_idx < 61 or action_idx > 65:
                    raise RuntimeError(f"Expected return-phase action in [61, 65], got {action_idx}")
                current_result = DeterminizedPolicyResult(
                    action_idx=action_idx,
                    visit_probs=visit_probs.astype(np.float32, copy=False),
                    root_best_value_mean=float(result.root_best_value),
                    num_determinizations=1,
                )
                if first_result is None:
                    first_result = current_result
                chosen.append(action_idx)
                state = env.step(action_idx)
                turns_taken += 1
        if first_result is None:
            raise RuntimeError("Return phase produced no return actions")
        return first_result, chosen
