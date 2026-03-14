from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .native_env import SplendorNativeEnv, StepState
from .state_schema import ACTION_DIM, STATE_DIM


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.25
    temperature_moves: int = 10
    temperature: float = 1.0
    eps: float = 1e-8
    root_dirichlet_noise: bool = False
    root_dirichlet_epsilon: float = 0.25
    root_dirichlet_alpha_total: float = 10.0
    eval_batch_size: int = 32
    use_forced_playouts: bool = False
    forced_playouts_k: float = 2.0


@dataclass
class MCTSResult:
    chosen_action_idx: int
    visit_probs: np.ndarray  # (69,) float32
    root_best_value: float


def run_mcts(
    env: Any,
    model: Any,
    state: StepState,
    *,
    turns_taken: int,
    device: str = "cpu",
    config: MCTSConfig | None = None,
    rng: random.Random | None = None,
) -> MCTSResult:
    cfg = config or MCTSConfig()
    if bool(getattr(model, "training", False)):
        model.eval()

    if not isinstance(env, SplendorNativeEnv):
        raise TypeError("run_mcts requires nn.native_env.SplendorNativeEnv (native-env-only implementation)")
    if cfg.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if not (0.0 <= float(cfg.root_dirichlet_epsilon) <= 1.0):
        raise ValueError("root_dirichlet_epsilon must be in [0,1]")
    if float(cfg.root_dirichlet_alpha_total) <= 0.0:
        raise ValueError("root_dirichlet_alpha_total must be positive")
    if int(cfg.eval_batch_size) <= 0:
        raise ValueError("eval_batch_size must be positive")
    if float(cfg.forced_playouts_k) <= 0.0:
        raise ValueError("forced_playouts_k must be positive")
    if state.is_terminal:
        raise ValueError("run_mcts called on terminal state")

    def evaluator(states_np: np.ndarray, masks_np: np.ndarray):
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

        if logits.device.type == "cpu":
            policy_scores = logits.detach().numpy().astype(np.float32, copy=False)
        else:
            policy_scores = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        if value_t.device.type == "cpu":
            values = value_t.detach().numpy().astype(np.float32, copy=False)
        else:
            values = value_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(policy_scores).all():
            raise ValueError("Model returned non-finite policy scores")
        if not np.isfinite(values).all():
            raise ValueError("Model returned non-finite values")
        return policy_scores, values

    py_rng = rng if rng is not None else random
    rng_seed = int(py_rng.getrandbits(64))

    native_kwargs = dict(
        evaluator=evaluator,
        turns_taken=int(turns_taken),
        num_simulations=int(cfg.num_simulations),
        c_puct=float(cfg.c_puct),
        temperature_moves=int(cfg.temperature_moves),
        temperature=float(cfg.temperature),
        eps=float(cfg.eps),
        root_dirichlet_noise=bool(cfg.root_dirichlet_noise),
        root_dirichlet_epsilon=float(cfg.root_dirichlet_epsilon),
        eval_batch_size=int(cfg.eval_batch_size),
        rng_seed=rng_seed,
        use_forced_playouts=bool(cfg.use_forced_playouts),
        forced_playouts_k=float(cfg.forced_playouts_k),
    )
    native_result = env.run_mcts_native(
        **native_kwargs,
        root_dirichlet_alpha_total=float(cfg.root_dirichlet_alpha_total),
    )

    visit_probs = np.asarray(native_result.visit_probs, dtype=np.float32)
    if visit_probs.shape != (ACTION_DIM,):
        raise RuntimeError(f"Unexpected native visit_probs shape {visit_probs.shape}")

    return MCTSResult(
        chosen_action_idx=int(native_result.chosen_action_idx),
        visit_probs=visit_probs,
        root_best_value=float(native_result.root_best_value),
    )
