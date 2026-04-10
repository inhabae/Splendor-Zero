from __future__ import annotations

import random
from dataclasses import dataclass, field
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
    q_values: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Alpha-Beta search with NN value head as leaf evaluator
# ---------------------------------------------------------------------------

@dataclass
class AlphaBetaConfig:
    depth: int = 3
    """Maximum search depth (in plies). Same-player sub-turns count as a ply
    but do NOT switch perspective."""


def _nn_evaluate(model, env: SplendorNativeEnv, state, device: str) -> tuple[np.ndarray, float]:
    """Return (policy_logits, value) from the NN for the current state.

    Value is always from the perspective of the current player-to-move.
    """
    import torch

    from nn.state_schema import ACTION_DIM, STATE_DIM

    state_np = np.asarray(state.state, dtype=np.float32).reshape(1, STATE_DIM)
    mask_np = np.asarray(state.mask, dtype=bool).reshape(1, ACTION_DIM)

    state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, value_t = model(state_t)

    policy = logits[0].cpu().numpy().astype(np.float32)
    value = float(value_t[0].cpu().numpy()) if value_t.ndim == 1 else float(value_t[0, 0].cpu().numpy())
    return policy, value


def _alphabeta(
    env: SplendorNativeEnv,
    state,
    depth: int,
    alpha: float,
    beta: float,
    model,
    device: str,
) -> float:
    """Negamax alpha-beta.

    Returns the value from the perspective of the player-to-move at this node.

    Handles same-player sub-turns (gem return, noble choice) correctly:
    when current_player does not change after applying a move, we do NOT negate.
    """
    if state.is_terminal:
        # Terminal: winner == current_player_id => +1, else -1, draw => 0
        w = state.winner
        cp = state.current_player_id
        if w == -1:
            return 0.0
        return 1.0 if w == cp else -1.0

    if depth == 0:
        _, value = _nn_evaluate(model, env, state, device)
        return value

    mask = np.asarray(state.mask, dtype=bool)
    legal_actions = np.where(mask)[0].tolist()
    if not legal_actions:
        return 0.0

    # Use NN policy to order moves (higher prior = try first => more cutoffs)
    policy, _ = _nn_evaluate(model, env, state, device)
    legal_actions.sort(key=lambda a: -policy[a])

    current_player = state.current_player_id
    best = -2.0  # below any valid value

    for action in legal_actions:
        child_state = env.step(action)  # advances the env in-place
        same_player = (child_state.current_player_id == current_player)

        if same_player:
            # Same player's turn continues — no perspective flip
            child_value = _alphabeta(env, child_state, depth - 1, alpha, beta, model, device)
        else:
            # Opponent to move — flip perspective
            child_value = -_alphabeta(env, child_state, depth - 1, -beta, -alpha, model, device)

        # Undo: restore env to parent state by reloading (env.clone was used outside)
        # NOTE: we use a cloned env per call level — see run_alphabeta_search below

        best = max(best, child_value)
        alpha = max(alpha, best)
        if alpha >= beta:
            break  # beta cutoff

    return best


def _alphabeta_with_clone(
    parent_env: SplendorNativeEnv,
    parent_state,
    depth: int,
    alpha: float,
    beta: float,
    model,
    device: str,
) -> float:
    """Wrapper that clones the env before each recursive call so moves are
    automatically 'undone' when the clone goes out of scope."""
    if parent_state.is_terminal:
        w = parent_state.winner
        cp = parent_state.current_player_id
        if w == -1:
            return 0.0
        return 1.0 if w == cp else -1.0

    if depth <= 0:
        _, value = _nn_evaluate(model, parent_env, parent_state, device)
        return value

    mask = np.asarray(parent_state.mask, dtype=bool)
    legal_actions = np.where(mask)[0].tolist()
    if not legal_actions:
        return 0.0

    policy, _ = _nn_evaluate(model, parent_env, parent_state, device)
    legal_actions.sort(key=lambda a: -policy[a])

    current_player = parent_state.current_player_id
    best = -2.0

    for action in legal_actions:
        child_env = parent_env.clone()
        child_state = child_env.step(action)
        same_player = (child_state.current_player_id == current_player)

        if same_player:
            child_value = _alphabeta_with_clone(
                child_env, child_state, depth - 1, alpha, beta, model, device
            )
        else:
            child_value = -_alphabeta_with_clone(
                child_env, child_state, depth - 1, -beta, -alpha, model, device
            )

        best = max(best, child_value)
        alpha = max(alpha, best)
        if alpha >= beta:
            break

    return best


def run_alphabeta_search(
    env: SplendorNativeEnv,
    model,
    state,
    *,
    device: str = "cpu",
    config: AlphaBetaConfig | None = None,
) -> DeterminizedPolicyResult:
    """Run alpha-beta search from the root and return the best action."""
    cfg = config or AlphaBetaConfig()
    if bool(getattr(model, "training", False)):
        model.eval()

    mask = np.asarray(state.mask, dtype=bool)
    legal_actions = np.where(mask)[0].tolist()
    if not legal_actions:
        raise ValueError("run_alphabeta_search called with no legal actions")

    # Order root moves by NN policy
    policy, root_value = _nn_evaluate(model, env, state, device)
    legal_actions.sort(key=lambda a: -policy[a])

    current_player = state.current_player_id
    best_action = legal_actions[0]
    best_value = -2.0
    alpha = -2.0
    beta = 2.0

    visit_probs = np.zeros(len(mask), dtype=np.float32)

    for action in legal_actions:
        child_env = env.clone()
        child_state = child_env.step(action)
        same_player = (child_state.current_player_id == current_player)

        if same_player:
            child_value = _alphabeta_with_clone(
                child_env, child_state, cfg.depth - 1, alpha, beta, model, device
            )
        else:
            child_value = -_alphabeta_with_clone(
                child_env, child_state, cfg.depth - 1, -beta, -alpha, model, device
            )

        if child_value > best_value:
            best_value = child_value
            best_action = action

        alpha = max(alpha, best_value)

    visit_probs[best_action] = 1.0
    return DeterminizedPolicyResult(
        action_idx=best_action,
        visit_probs=visit_probs,
        root_best_value_mean=best_value,
        num_determinizations=1,
        q_values=None,
    )


# ---------------------------------------------------------------------------
# Forced child search: run MCTS rooted at each child, pick best Q
# ---------------------------------------------------------------------------

@dataclass
class ForcedChildSearchConfig:
    simulations_per_child: int = 2000
    c_puct: float = 1.25
    eval_batch_size: int = 1


def run_forced_child_search(
    env: SplendorNativeEnv,
    model,
    state,
    *,
    turns_taken: int,
    device: str = "cpu",
    config: ForcedChildSearchConfig | None = None,
    rng: random.Random | None = None,
) -> DeterminizedPolicyResult:
    """For each legal root action, run a separate MCTS rooted at the resulting
    child state. Select the action whose child MCTS returns the best Q-value
    (negated back to the root player's perspective).

    Same-player sub-turns are handled correctly: if the child is still the same
    player (gem return, noble choice), we do NOT negate the value.
    """
    cfg = config or ForcedChildSearchConfig()
    if bool(getattr(model, "training", False)):
        model.eval()
    py_rng = rng or random.Random()

    mask = np.asarray(state.mask, dtype=bool)
    legal_actions = np.where(mask)[0].tolist()
    if not legal_actions:
        raise ValueError("run_forced_child_search called with no legal actions")

    current_player = state.current_player_id

    best_action = legal_actions[0]
    best_value = -float("inf")
    visit_probs = np.zeros(len(mask), dtype=np.float32)
    q_values = np.full(len(mask), np.nan, dtype=np.float32)
    child_values: dict[int, float] = {}

    mcts_cfg = MCTSConfig(
        num_simulations=int(cfg.simulations_per_child),
        c_puct=float(cfg.c_puct),
        temperature_moves=0,
        temperature=0.0,
        root_dirichlet_noise=False,
        eval_batch_size=int(cfg.eval_batch_size),
    )

    for action in legal_actions:
        child_env = env.clone()
        child_state = child_env.step(action)

        if child_state.is_terminal:
            # Immediate terminal — use exact outcome
            w = child_state.winner
            cp_at_child = child_state.current_player_id
            if w == -1:
                raw_value = 0.0
            else:
                raw_value = 1.0 if w == cp_at_child else -1.0
            same_player = (cp_at_child == current_player)
        else:
            same_player = (child_state.current_player_id == current_player)
            child_turns_taken = turns_taken if same_player else turns_taken + 1
            child_rng = random.Random(py_rng.getrandbits(64))
            child_result = run_mcts(
                child_env,
                model,
                child_state,
                turns_taken=child_turns_taken,
                device=device,
                config=mcts_cfg,
                rng=child_rng,
            )
            # root_best_value is from the perspective of the player-to-move
            # at the child root (i.e. child_state.current_player_id)
            raw_value = float(child_result.root_best_value)

        # Convert to root player's perspective
        value_from_root = raw_value if same_player else -raw_value

        child_values[action] = value_from_root
        q_values[action] = np.float32(value_from_root)
        if value_from_root > best_value:
            best_value = value_from_root
            best_action = action

    # Build visit_probs proportional to shifted values so callers can inspect
    min_val = min(child_values.values())
    total = sum(v - min_val for v in child_values.values())
    if total > 0:
        for a, v in child_values.items():
            visit_probs[a] = (v - min_val) / total
    else:
        for a in legal_actions:
            visit_probs[a] = 1.0 / len(legal_actions)

    return DeterminizedPolicyResult(
        action_idx=best_action,
        visit_probs=visit_probs,
        root_best_value_mean=best_value,
        num_determinizations=1,
        q_values=q_values,
    )


# ---------------------------------------------------------------------------
# Main policy class
# ---------------------------------------------------------------------------

@dataclass
class DeterminizedMCTSPolicy:
    checkpoint_path: str
    mcts_config: MCTSConfig
    device: str = "cpu"
    determinization_samples: int = 1
    search_type: Literal["mcts", "ismcts", "alphabeta", "forced_child"] = "mcts"
    gpu_batching_enabled: bool = False
    alphabeta_config: AlphaBetaConfig = field(default_factory=AlphaBetaConfig)
    forced_child_config: ForcedChildSearchConfig = field(default_factory=ForcedChildSearchConfig)

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
        def _run_mcts_fallback():
            mcts_cfg = MCTSConfig(
                num_simulations=int(self.mcts_config.num_simulations),
                c_puct=float(self.mcts_config.c_puct),
                temperature_moves=int(self.mcts_config.temperature_moves),
                temperature=float(self.mcts_config.temperature),
                eps=float(self.mcts_config.eps),
                root_dirichlet_noise=bool(self.mcts_config.root_dirichlet_noise),
                root_dirichlet_epsilon=float(self.mcts_config.root_dirichlet_epsilon),
                root_dirichlet_alpha_total=float(self.mcts_config.root_dirichlet_alpha_total),
                eval_batch_size=(32 if self.gpu_batching_enabled else 1),
                use_forced_playouts=bool(self.mcts_config.use_forced_playouts),
                forced_playouts_k=float(self.mcts_config.forced_playouts_k),
            )
            return run_mcts(
                env,
                self._model,
                state,
                turns_taken=int(turns_taken),
                device=self.device,
                config=mcts_cfg,
                rng=rng,
            )

        if self.search_type == "alphabeta":
            return run_alphabeta_search(
                env,
                self._model,
                state,
                device=self.device,
                config=self.alphabeta_config,
            )

        if self.search_type == "forced_child":
            return run_forced_child_search(
                env,
                self._model,
                state,
                turns_taken=int(turns_taken),
                device=self.device,
                config=self.forced_child_config,
                rng=rng,
            )

        if self.search_type == "ismcts":
            eval_batch_size = 32 if self.gpu_batching_enabled else 1
            try:
                return run_ismcts(
                    env,
                    self._model,
                    state=state,
                    turns_taken=int(turns_taken),
                    device=self.device,
                    config=ISMCTSConfig(
                        num_simulations=int(self.mcts_config.num_simulations),
                        c_puct=float(self.mcts_config.c_puct),
                        eval_batch_size=eval_batch_size,
                    ),
                    rng=rng,
                )
            except RuntimeError as exc:
                if "Native ISMCTS made no progress while gathering leaves" not in str(exc):
                    raise
                return _run_mcts_fallback()

        if self.search_type == "mcts":
            return _run_mcts_fallback()

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

        if isinstance(result, DeterminizedPolicyResult):
            return result

        visit_probs = np.asarray(result.visit_probs, dtype=np.float32)
        action_idx = int(np.argmax(visit_probs))
        return DeterminizedPolicyResult(
            action_idx=action_idx,
            visit_probs=visit_probs.astype(np.float32, copy=False),
            root_best_value_mean=float(result.root_best_value),
            num_determinizations=1,
            q_values=np.asarray(result.q_values, dtype=np.float32),
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

                if isinstance(result, DeterminizedPolicyResult):
                    visit_probs = result.visit_probs
                    action_idx = result.action_idx
                    root_best_value = result.root_best_value_mean
                    q_values = result.q_values
                else:
                    visit_probs = np.asarray(result.visit_probs, dtype=np.float32)
                    action_idx = int(np.argmax(visit_probs))
                    root_best_value = float(result.root_best_value)
                    q_values = np.asarray(result.q_values, dtype=np.float32)

                if action_idx < 61 or action_idx > 65:
                    raise RuntimeError(f"Expected return-phase action in [61, 65], got {action_idx}")

                current_result = DeterminizedPolicyResult(
                    action_idx=action_idx,
                    visit_probs=np.asarray(visit_probs, dtype=np.float32),
                    root_best_value_mean=root_best_value,
                    num_determinizations=1,
                    q_values=q_values,
                )
                if first_result is None:
                    first_result = current_result
                chosen.append(action_idx)
                state = env.step(action_idx)
                turns_taken += 1

        if first_result is None:
            raise RuntimeError("Return phase produced no return actions")
        return first_result, chosen
