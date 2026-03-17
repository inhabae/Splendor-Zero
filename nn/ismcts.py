from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .mcts import MCTSResult
from .native_env import SplendorNativeEnv, StepState
from .state_schema import ACTION_DIM, STATE_DIM


@dataclass
class ISMCTSConfig:
    num_simulations: int = 128
    c_puct: float = 1.25
    eps: float = 1e-8
    eval_batch_size: int = 32
    root_parallel_workers: int = 1


def _batch_evaluator(model: Any, states_np: np.ndarray, masks_np: np.ndarray, *, device: str):
    states_np = np.asarray(states_np, dtype=np.float32)
    masks_np = np.asarray(masks_np)
    if states_np.ndim != 2 or states_np.shape[1] != STATE_DIM:
        raise ValueError(f"evaluator states shape must be (B,{STATE_DIM}), got {states_np.shape}")
    if masks_np.ndim != 2 or masks_np.shape[1] != ACTION_DIM:
        raise ValueError(f"evaluator masks shape must be (B,{ACTION_DIM}), got {masks_np.shape}")
    if states_np.shape[0] != masks_np.shape[0]:
        raise ValueError("evaluator batch size mismatch between states and masks")
    if states_np.shape[0] == 0:
        raise ValueError("evaluator requires non-empty batch")

    state_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, value_t = model(state_t)
    if tuple(logits.shape) != (states_np.shape[0], ACTION_DIM):
        raise ValueError(f"Model logits shape must be (B,{ACTION_DIM}), got {tuple(logits.shape)}")
    if value_t.ndim == 2 and value_t.shape[1] == 1:
        value_t = value_t.squeeze(1)
    if value_t.ndim != 1 or value_t.shape[0] != states_np.shape[0]:
        raise ValueError(f"Model values shape must be (B,), got {tuple(value_t.shape)}")

    policy_scores = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    values = value_t.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(policy_scores).all():
        raise ValueError("Model returned non-finite policy scores")
    if not np.isfinite(values).all():
        raise ValueError("Model returned non-finite values")
    return policy_scores, values


def run_ismcts(
    env: Any,
    model: Any,
    state: StepState,
    *,
    turns_taken: int,
    device: str = "cpu",
    config: ISMCTSConfig | None = None,
    rng: random.Random | None = None,
) -> MCTSResult:
    del turns_taken
    cfg = config or ISMCTSConfig()
    if bool(getattr(model, "training", False)):
        model.eval()
    if not isinstance(env, SplendorNativeEnv):
        raise TypeError("run_ismcts requires nn.native_env.SplendorNativeEnv")
    if state.is_terminal:
        raise ValueError("run_ismcts called on terminal state")
    if int(cfg.num_simulations) <= 0:
        raise ValueError("num_simulations must be positive")
    if int(cfg.eval_batch_size) <= 0:
        raise ValueError("eval_batch_size must be positive")
    if int(cfg.root_parallel_workers) <= 0:
        raise ValueError("root_parallel_workers must be positive")

    py_rng = rng if rng is not None else random
    native_result = env.run_ismcts_native(
        evaluator=lambda states_np, masks_np: _batch_evaluator(model, states_np, masks_np, device=device),
        num_simulations=int(cfg.num_simulations),
        c_puct=float(cfg.c_puct),
        eps=float(cfg.eps),
        eval_batch_size=int(cfg.eval_batch_size),
        rng_seed=int(py_rng.getrandbits(64)),
        root_parallel_workers=int(cfg.root_parallel_workers),
    )

    visit_probs = np.asarray(native_result.visit_probs, dtype=np.float32)
    q_values = np.asarray(native_result.q_values, dtype=np.float32)
    if visit_probs.shape != (ACTION_DIM,):
        raise RuntimeError(f"Unexpected native ISMCTS visit_probs shape {visit_probs.shape}")
    if q_values.shape != (ACTION_DIM,):
        raise RuntimeError(f"Unexpected native ISMCTS q_values shape {q_values.shape}")

    return MCTSResult(
        chosen_action_idx=int(native_result.chosen_action_idx),
        visit_probs=visit_probs,
        q_values=q_values,
        root_best_value=float(native_result.root_best_value),
        search_slots_requested=int(native_result.search_slots_requested),
        search_slots_evaluated=int(native_result.search_slots_evaluated),
        search_slots_drop_pending_eval=int(getattr(native_result, "search_slots_drop_pending_eval", 0)),
        search_slots_drop_no_action=int(getattr(native_result, "search_slots_drop_no_action", 0)),
    )
