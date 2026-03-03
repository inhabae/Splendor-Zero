from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nn.native_env import StepState
from nn.state_schema import ACTION_DIM


@dataclass
class ReferenceMCTSResult:
    chosen_action_idx: int
    visit_probs: np.ndarray
    root_value: float
    debug_counts: dict[str, int]


@dataclass
class _Node:
    priors: np.ndarray = field(default_factory=lambda: np.zeros((ACTION_DIM,), dtype=np.float32))
    visit_count: np.ndarray = field(default_factory=lambda: np.zeros((ACTION_DIM,), dtype=np.int32))
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros((ACTION_DIM,), dtype=np.float32))
    child_index: np.ndarray = field(default_factory=lambda: np.full((ACTION_DIM,), -1, dtype=np.int32))
    expanded: bool = False
    pending_eval: bool = False


@dataclass
class _PathStep:
    node_index: int
    action: int
    same_player: bool


@dataclass
class _PendingLeafEval:
    node_index: int
    state: np.ndarray
    mask: np.ndarray
    path: list[_PathStep]


@dataclass
class _ReadyBackup:
    value: float
    path: list[_PathStep]


def _winner_to_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    if winner not in (0, 1):
        raise RuntimeError("Unexpected winner value in reference MCTS")
    return 1.0 if winner == player_id else -1.0


def _apply_dirichlet_root_noise(
    priors: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    alpha_total: float,
    rng: random.Random,
) -> None:
    if epsilon <= 0.0:
        return
    legal = np.flatnonzero(legal_mask)
    priors[~legal_mask] = 0.0
    if legal.size < 2:
        return
    alpha = float(alpha_total) / float(legal.size)
    noise = np.asarray([rng.gammavariate(alpha, 1.0) for _ in range(int(legal.size))], dtype=np.float64)
    noise_sum = float(noise.sum())
    if not (noise_sum > 0.0) or not np.isfinite(noise_sum):
        noise = np.full((legal.size,), 1.0 / float(legal.size), dtype=np.float64)
    else:
        noise /= noise_sum
    mixed = ((1.0 - float(epsilon)) * priors[legal].astype(np.float64, copy=False)) + (float(epsilon) * noise)
    mixed_sum = float(mixed.sum())
    if not (mixed_sum > 0.0) or not np.isfinite(mixed_sum):
        priors[legal] = np.float32(1.0 / float(legal.size))
        return
    priors[legal] = (mixed / mixed_sum).astype(np.float32)


def _select_puct_action(
    nodes: list[_Node],
    node_index: int,
    legal_mask: np.ndarray,
    c_puct: float,
    eps: float,
) -> int:
    node = nodes[node_index]
    parent_n = float(node.visit_count.sum())
    sqrt_parent = math.sqrt(parent_n + float(eps))

    best_action = -1
    best_score = -float("inf")
    for action in range(ACTION_DIM):
        if not bool(legal_mask[action]):
            continue
        child_idx = int(node.child_index[action])
        if child_idx >= 0 and nodes[child_idx].pending_eval:
            continue
        n = float(node.visit_count[action])
        q = 0.0 if n <= 0.0 else float(node.value_sum[action] / n)
        u = float(c_puct) * float(node.priors[action]) * sqrt_parent / (1.0 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _normalize_priors(policy_scores: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    priors = np.zeros((ACTION_DIM,), dtype=np.float32)
    legal = np.flatnonzero(legal_mask)
    if legal.size == 0:
        return priors

    legal_scores = policy_scores[legal]
    finite = np.isfinite(legal_scores)
    if not bool(np.any(finite)):
        priors[legal] = np.float32(1.0 / float(legal.size))
        return priors

    max_score = float(np.max(legal_scores[finite]))
    weights = np.zeros((legal.size,), dtype=np.float64)
    for i, score in enumerate(legal_scores):
        if np.isfinite(score):
            weights[i] = math.exp(float(score) - max_score)
    wsum = float(np.sum(weights))
    if not (wsum > 0.0) or not np.isfinite(wsum):
        priors[legal] = np.float32(1.0 / float(legal.size))
    else:
        priors[legal] = (weights / wsum).astype(np.float32)
    return priors


def _sample_action_from_visits(
    visit_probs: np.ndarray,
    legal_mask: np.ndarray,
    turns_taken: int,
    temperature_moves: int,
    temperature: float,
    rng: random.Random,
) -> int:
    legal = np.flatnonzero(legal_mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions for final reference MCTS action selection")
    if int(turns_taken) >= int(temperature_moves):
        best_action = int(legal[0])
        best_prob = float(visit_probs[best_action])
        for a in legal.tolist():
            p = float(visit_probs[a])
            if p > best_prob:
                best_prob = p
                best_action = int(a)
        return best_action

    if float(temperature) <= 0.0:
        weights = [float(visit_probs[a]) for a in legal.tolist()]
    else:
        inv_temp = 1.0 / float(temperature)
        weights = [float(visit_probs[a]) ** inv_temp for a in legal.tolist()]
    wsum = float(sum(weights))
    if not (wsum > 0.0) or not np.isfinite(wsum):
        weights = [1.0 for _ in legal.tolist()]
    return int(rng.choices(legal.tolist(), weights=weights, k=1)[0])


def run_reference_mcts(
    *,
    root_state: StepState,
    transition_from_root: Callable[[list[int]], StepState],
    evaluator: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    turns_taken: int,
    num_simulations: int = 64,
    c_puct: float = 1.25,
    temperature_moves: int = 10,
    temperature: float = 1.0,
    eps: float = 1e-8,
    root_dirichlet_noise: bool = False,
    root_dirichlet_epsilon: float = 0.25,
    root_dirichlet_alpha_total: float = 10.0,
    eval_batch_size: int = 32,
    rng_seed: int = 0,
) -> ReferenceMCTSResult:
    if int(num_simulations) <= 0:
        raise ValueError("num_simulations must be positive")
    if root_state.is_terminal:
        raise ValueError("run_reference_mcts called on terminal state")
    if not bool(np.any(root_state.mask)):
        raise ValueError("MCTS root has no legal actions")
    if int(eval_batch_size) <= 0:
        raise ValueError("eval_batch_size must be positive")

    nodes: list[_Node] = [_Node()]
    rng = random.Random(int(rng_seed))
    root_noise_applied = False
    completed = 0
    total_evaluated = 0

    def _evaluate_pending(pending: list[_PendingLeafEval], backups: list[_ReadyBackup]) -> None:
        nonlocal root_noise_applied, total_evaluated
        if not pending:
            return

        batch_states = np.stack([req.state for req in pending], axis=0).astype(np.float32, copy=False)
        batch_masks = np.stack([req.mask for req in pending], axis=0).astype(np.bool_, copy=False)
        priors_raw, values_raw = evaluator(batch_states, batch_masks)
        priors_raw = np.asarray(priors_raw, dtype=np.float32)
        values_raw = np.asarray(values_raw, dtype=np.float32)

        if priors_raw.shape != (len(pending), ACTION_DIM):
            raise RuntimeError("Reference evaluator priors must have shape (B, ACTION_DIM)")
        if values_raw.shape != (len(pending),):
            raise RuntimeError("Reference evaluator values must have shape (B,)")

        for i, req in enumerate(pending):
            node = nodes[req.node_index]
            node.priors = _normalize_priors(priors_raw[i], req.mask)
            value = float(values_raw[i])
            if not np.isfinite(value):
                raise RuntimeError("Reference evaluator values contain non-finite entries")
            node.expanded = True
            node.pending_eval = False

            if req.node_index == 0 and (not root_noise_applied) and bool(root_dirichlet_noise):
                _apply_dirichlet_root_noise(
                    node.priors,
                    req.mask,
                    float(root_dirichlet_epsilon),
                    float(root_dirichlet_alpha_total),
                    rng,
                )
                root_noise_applied = True

            backups.append(_ReadyBackup(value=value, path=req.path))
            total_evaluated += 1

    # Pre-expand root to match native implementation semantics.
    root = nodes[0]
    root.pending_eval = True
    _evaluate_pending(
        [
            _PendingLeafEval(
                node_index=0,
                state=np.asarray(root_state.state, dtype=np.float32, copy=True),
                mask=np.asarray(root_state.mask, dtype=np.bool_, copy=True),
                path=[],
            )
        ],
        [],
    )

    while completed < int(num_simulations):
        target_batch = min(int(eval_batch_size), int(num_simulations) - completed)
        pending: list[_PendingLeafEval] = []
        backups: list[_ReadyBackup] = []

        for _ in range(target_batch):
            node_index = 0
            sim_actions: list[int] = []
            sim_state = root_state
            path: list[_PathStep] = []

            while True:
                node = nodes[node_index]
                if sim_state.is_terminal:
                    backups.append(
                        _ReadyBackup(
                            value=_winner_to_value_for_player(int(sim_state.winner), int(sim_state.current_player_id)),
                            path=path,
                        )
                    )
                    break
                if not node.expanded:
                    if node.pending_eval:
                        break
                    node.pending_eval = True
                    pending.append(
                        _PendingLeafEval(
                            node_index=node_index,
                            state=np.asarray(sim_state.state, dtype=np.float32, copy=True),
                            mask=np.asarray(sim_state.mask, dtype=np.bool_, copy=True),
                            path=path,
                        )
                    )
                    break

                action = _select_puct_action(nodes, node_index, sim_state.mask, float(c_puct), float(eps))
                if action < 0:
                    break

                child_idx = int(node.child_index[action])
                if child_idx < 0:
                    child_idx = len(nodes)
                    nodes.append(_Node())
                    node.child_index[action] = child_idx

                parent_to_play = int(sim_state.current_player_id)
                sim_actions.append(int(action))
                sim_state = transition_from_root(sim_actions)
                same_player = int(sim_state.current_player_id) == parent_to_play
                path.append(_PathStep(node_index=node_index, action=int(action), same_player=bool(same_player)))
                node_index = child_idx

        _evaluate_pending(pending, backups)

        if not backups:
            raise RuntimeError("Reference MCTS made no progress while gathering/evaluating leaves")

        for ready in backups:
            value = float(ready.value)
            for step in reversed(ready.path):
                backed = value if step.same_player else -value
                parent = nodes[step.node_index]
                parent.visit_count[step.action] += 1
                parent.value_sum[step.action] += np.float32(backed)
                value = backed

        completed += len(backups)

    root = nodes[0]
    visit_probs = np.zeros((ACTION_DIM,), dtype=np.float32)
    total_visits = int(np.sum(root.visit_count))
    legal = np.flatnonzero(root_state.mask)
    if total_visits > 0:
        visit_probs = (root.visit_count.astype(np.float64) / float(total_visits)).astype(np.float32)
    elif legal.size > 0:
        visit_probs[legal] = np.float32(1.0 / float(legal.size))
    visit_probs[~root_state.mask] = 0.0
    psum = float(np.sum(visit_probs))
    if psum > 0.0 and np.isfinite(psum):
        visit_probs = (visit_probs / psum).astype(np.float32)

    chosen_action_idx = _sample_action_from_visits(
        visit_probs=visit_probs,
        legal_mask=root_state.mask,
        turns_taken=int(turns_taken),
        temperature_moves=int(temperature_moves),
        temperature=float(temperature),
        rng=rng,
    )

    q_vals = []
    for action in legal.tolist():
        n = int(root.visit_count[action])
        q_vals.append(0.0 if n <= 0 else float(root.value_sum[action] / float(n)))
    root_value = float(sum(q_vals) / float(len(q_vals))) if q_vals else 0.0

    return ReferenceMCTSResult(
        chosen_action_idx=int(chosen_action_idx),
        visit_probs=visit_probs,
        root_value=float(root_value),
        debug_counts={"evaluated_leaves": int(total_evaluated), "total_nodes": int(len(nodes))},
    )
