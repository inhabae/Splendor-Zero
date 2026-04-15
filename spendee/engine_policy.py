from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nn.checkpoints import load_checkpoint
from nn.ismcts import ISMCTSConfig, run_ismcts
from nn.mcts import MCTSConfig, create_mcts_session, run_mcts
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
    """Maximum search depth (in opponent-changing plies).

    Same-player modal sub-turns such as gem returns and noble choices are
    resolved within the same ply so search never stops on those states.
    """
    chance_node_samples: int = 4
    """Number of card draws to sample when a deck draw occurs mid-search.
    Set to 1 to disable chance nodes (reverts to single-determinization behaviour).
    Capped automatically at the number of cards remaining in the affected tier."""


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


def _terminal_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    return 1.0 if winner == player_id else -1.0


def _is_modal_subturn_state(state) -> bool:
    return bool(
        getattr(state, "is_return_phase", False)
        or getattr(state, "is_noble_choice_phase", False)
    )



# ---------------------------------------------------------------------------
# Chance node helpers
# ---------------------------------------------------------------------------

def _tier_for_action(action_idx: int) -> int | None:
    """Return the deck tier a draw event is associated with, or None if the
    action cannot trigger a deck draw."""
    if 0 <= action_idx <= 11:           # buy face-up: tier = row + 1
        return action_idx // 4 + 1
    if 15 <= action_idx <= 26:          # reserve face-up: same layout
        return (action_idx - 15) // 4 + 1
    if 27 <= action_idx <= 29:          # reserve from deck: explicit tier
        return action_idx - 27 + 1
    return None


def _slot_for_action(action_idx: int) -> int | None:
    """Return the face-up slot index (0–3) for buy/reserve-faceup actions."""
    if 0 <= action_idx <= 11:
        return action_idx % 4
    if 15 <= action_idx <= 26:
        return (action_idx - 15) % 4
    return None


def _find_drawn_card(
    pre_deck: list[int],
    post_deck: list[int],
) -> int | None:
    """Given the deck contents before and after a step, return the card id
    that was drawn (i.e. present in pre but absent in post), or None."""
    pre_set = set(pre_deck)
    post_set = set(post_deck)
    diff = pre_set - post_set
    if len(diff) == 1:
        return next(iter(diff))
    return None


def _build_chance_outcomes(
    pre_export: dict,
    post_export: dict,
    action_idx: int,
    actor_player_index: int,
    *,
    samples: int,
    rng: random.Random,
) -> list[dict] | None:
    """Return a list of modified post-step exported states, each representing
    one sampled deck draw outcome.  Returns None if no chance event occurred
    (deck was empty, only one card possible, or action doesn't draw).

    Each returned dict is a ready-to-load payload for env.load_state().
    """
    tier = _tier_for_action(action_idx)
    if tier is None:
        return None

    pre_deck: list[int] = list(pre_export.get("deck_card_ids_by_tier", [[], [], []])[tier - 1])
    post_deck: list[int] = list(post_export.get("deck_card_ids_by_tier", [[], [], []])[tier - 1])

    # No draw happened (deck was already empty before the action)
    if len(pre_deck) == len(post_deck):
        return None

    drawn_card = _find_drawn_card(pre_deck, post_deck)
    if drawn_card is None:
        return None

    # All cards that could have been drawn: the pre-deck contents
    possible_cards: list[int] = list(pre_deck)
    if len(possible_cards) <= 1:
        # Only one outcome — no branching needed
        return None

    n = min(int(samples), len(possible_cards))
    sampled_cards: list[int] = rng.sample(possible_cards, n)

    slot = _slot_for_action(action_idx)
    is_reserve_deck = 27 <= action_idx <= 29

    outcomes: list[dict] = []
    for card_id in sampled_cards:
        state = copy.deepcopy(post_export)

        # Replace drawn_card with card_id in the deck (deck shrinks by 1 regardless,
        # but we need to represent the correct remaining deck contents)
        new_deck = [c for c in pre_deck if c != card_id]
        state["deck_card_ids_by_tier"][tier - 1] = new_deck

        if is_reserve_deck:
            # The drawn card goes into the acting player's reserved hand.
            # Find it in the reserved list and swap it.
            players = state.get("players", [])
            if actor_player_index < len(players):
                player = players[actor_player_index]
                reserved = player.get("reserved", [])
                for entry in reserved:
                    if entry.get("card_id") == drawn_card:
                        entry["card_id"] = card_id
                        break
        else:
            # The drawn card appears face-up in the vacated slot.
            faceup = state.get("faceup_card_ids", [[], [], [], []])
            if tier - 1 < len(faceup) and slot is not None and slot < len(faceup[tier - 1]):
                row = faceup[tier - 1]
                row[slot] = card_id

        outcomes.append(state)

    return outcomes


def _alphabeta_chance(
    parent_env: "SplendorNativeEnv",
    post_export: dict,
    pre_export: dict,
    action_idx: int,
    actor_player_index: int,
    depth: int,
    alpha: float,
    beta: float,
    model,
    device: str,
    root_player: int,
    config: "AlphaBetaConfig",
    rng: random.Random,
) -> float:
    """Evaluate a chance node by averaging over sampled deck-draw outcomes.

    If no chance event is detected, falls through immediately to
    _alphabeta_with_clone on the already-stepped state.
    """
    outcomes = _build_chance_outcomes(
        pre_export,
        post_export,
        action_idx,
        actor_player_index,
        samples=config.chance_node_samples,
        rng=rng,
    )

    if not outcomes:
        # No chance event — evaluate the single post-step state directly.
        child_env = parent_env.clone()
        child_state = child_env.load_state(post_export)
        return _alphabeta_with_clone(
            child_env, child_state, depth, alpha, beta, model, device,
            root_player=root_player, config=config, rng=rng,
        )

    # Chance node: average over sampled outcomes (no pruning at this level).
    total = 0.0
    for outcome_export in outcomes:
        child_env = parent_env.clone()
        child_state = child_env.load_state(outcome_export)
        # Depth is NOT decremented at the chance node itself.
        v = _alphabeta_with_clone(
            child_env, child_state, depth, alpha, beta, model, device,
            root_player=root_player, config=config, rng=rng,
        )
        total += v

    return total / len(outcomes)


def _alphabeta_with_clone(
    parent_env: "SplendorNativeEnv",
    parent_state,
    depth: int,
    alpha: float,
    beta: float,
    model,
    device: str,
    *,
    root_player: int,
    config: "AlphaBetaConfig",
    rng: random.Random,
) -> float:
    """Negamax alpha-beta with clone-based undo and explicit chance nodes.

    Values are always from the perspective of the player-to-move at this node.
    """
    if parent_state.is_terminal:
        w = parent_state.winner
        cp = parent_state.current_player_id
        return _terminal_value_for_player(w, cp)

    if depth <= 0 and not _is_modal_subturn_state(parent_state):
        _, value = _nn_evaluate(model, parent_env, parent_state, device)
        return value

    mask = np.asarray(parent_state.mask, dtype=bool)
    legal_actions = np.where(mask)[0].tolist()
    if not legal_actions:
        return 0.0

    if not _is_modal_subturn_state(parent_state):
        policy, _ = _nn_evaluate(model, parent_env, parent_state, device)
        legal_actions.sort(key=lambda a: -policy[a])

    current_player = parent_state.current_player_id
    pre_export = parent_env.export_state()
    best = -2.0

    for action in legal_actions:
        # Step in a clone to get the post-action state
        step_env = parent_env.clone()
        child_state = step_env.step(action)
        post_export = step_env.export_state()

        same_player = (child_state.current_player_id == current_player)
        next_depth = depth if same_player else depth - 1

        # Determine whether this action may have triggered a deck draw
        # and route through the chance node evaluator.
        if same_player:
            child_value = _alphabeta_chance(
                step_env, post_export, pre_export, action,
                current_player, next_depth,
                alpha, beta, model, device,
                root_player=root_player, config=config, rng=rng,
            )
        else:
            child_value = -_alphabeta_chance(
                step_env, post_export, pre_export, action,
                current_player, next_depth,
                -beta, -alpha, model, device,
                root_player=root_player, config=config, rng=rng,
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
    rng: random.Random | None = None,
) -> DeterminizedPolicyResult:
    """Run alpha-beta search from the root and return the best action."""
    cfg = config or AlphaBetaConfig()
    py_rng = rng or random.Random()
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
    pre_export = env.export_state()
    best_action = legal_actions[0]
    best_value = -2.0
    alpha = -2.0
    beta = 2.0

    visit_probs = np.zeros(len(mask), dtype=np.float32)
    q_values = np.full(len(mask), np.nan, dtype=np.float32)

    for action in legal_actions:
        step_env = env.clone()
        child_state = step_env.step(action)
        post_export = step_env.export_state()
        same_player = (child_state.current_player_id == current_player)

        if same_player:
            child_value = _alphabeta_chance(
                step_env, post_export, pre_export, action,
                current_player, cfg.depth - 1,
                alpha, beta, model, device,
                root_player=current_player, config=cfg, rng=py_rng,
            )
        else:
            child_value = -_alphabeta_chance(
                step_env, post_export, pre_export, action,
                current_player, cfg.depth - 1,
                -beta, -alpha, model, device,
                root_player=current_player, config=cfg, rng=py_rng,
            )

        q_values[action] = np.float32(child_value)
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
        q_values=q_values,
    )


# ---------------------------------------------------------------------------
# Forced child search: run MCTS rooted at each child, pick best Q
# ---------------------------------------------------------------------------

@dataclass
class ForcedChildSearchConfig:
    simulations_per_child: int = 2000
    c_puct: float = 1.25
    eval_batch_size: int = 1


@dataclass
class BootstrapMCTSConfig:
    bootstrap_simulations_per_action: int = 0
    eval_batch_size: int = 1


_RETURN_ACTION_LO = 61
_RETURN_ACTION_HI = 65
_NOBLE_ACTION_LO = 66
_NOBLE_ACTION_HI = 68


@dataclass(frozen=True)
class _ForcedChildResolvedOutcome:
    state_key: str
    state_payload: dict
    turns_taken: int
    is_terminal: bool
    winner: int
    current_player_id: int


def _legal_actions_from_state(state) -> list[int]:
    mask = np.asarray(state.mask, dtype=bool)
    return np.where(mask)[0].tolist()


def _is_forced_child_modal_subturn(state) -> bool:
    legal_actions = _legal_actions_from_state(state)
    if not legal_actions:
        return False
    legal = np.asarray(legal_actions, dtype=np.int32)
    return bool(
        np.all((legal >= _RETURN_ACTION_LO) & (legal <= _RETURN_ACTION_HI))
        or np.all((legal >= _NOBLE_ACTION_LO) & (legal <= _NOBLE_ACTION_HI))
    )


def _canonical_exported_state_key(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _collect_forced_child_resolved_outcomes(
    env: SplendorNativeEnv,
    state,
    *,
    turns_taken: int,
    out: dict[str, _ForcedChildResolvedOutcome],
) -> None:
    if state.is_terminal or not _is_forced_child_modal_subturn(state):
        payload = env.export_state()
        key = _canonical_exported_state_key(payload)
        if key not in out:
            out[key] = _ForcedChildResolvedOutcome(
                state_key=key,
                state_payload=payload,
                turns_taken=int(turns_taken),
                is_terminal=bool(state.is_terminal),
                winner=int(state.winner),
                current_player_id=int(state.current_player_id),
            )
        return

    current_player = int(state.current_player_id)
    for action in sorted(_legal_actions_from_state(state)):
        child_env = env.clone()
        child_state = child_env.step(action)
        next_turns_taken = turns_taken if int(child_state.current_player_id) == current_player else turns_taken + 1
        _collect_forced_child_resolved_outcomes(
            child_env,
            child_state,
            turns_taken=next_turns_taken,
            out=out,
        )


def _sorted_forced_child_outcomes(
    env: SplendorNativeEnv,
    state,
    *,
    turns_taken: int,
) -> list[_ForcedChildResolvedOutcome]:
    outcomes: dict[str, _ForcedChildResolvedOutcome] = {}
    _collect_forced_child_resolved_outcomes(
        env,
        state,
        turns_taken=turns_taken,
        out=outcomes,
    )
    return sorted(
        outcomes.values(),
        key=lambda outcome: (0 if outcome.is_terminal else 1, outcome.state_key),
    )


def _split_forced_child_budget(total_budget: int, num_states: int) -> list[int]:
    if num_states <= 0:
        return []
    if total_budget <= 0:
        return [0] * num_states
    if total_budget >= num_states:
        base = total_budget // num_states
        remainder = total_budget % num_states
        return [base + (1 if idx < remainder else 0) for idx in range(num_states)]
    return [1 if idx < total_budget else 0 for idx in range(num_states)]


def _split_bootstrap_budget(total_budget: int, num_actions: int) -> list[int]:
    if num_actions <= 0:
        return []
    if total_budget <= 0:
        return [0] * num_actions
    base = total_budget // num_actions
    remainder = total_budget % num_actions
    return [base + (1 if idx < remainder else 0) for idx in range(num_actions)]


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
        same_player = (int(child_state.current_player_id) == current_player)
        child_turns_taken = turns_taken if same_player else turns_taken + 1

        resolved_outcomes = _sorted_forced_child_outcomes(
            child_env,
            child_state,
            turns_taken=child_turns_taken,
        )

        best_outcome_value = -float("inf")
        nonterminal_outcomes = [outcome for outcome in resolved_outcomes if not outcome.is_terminal]
        nonterminal_budgets = _split_forced_child_budget(
            int(cfg.simulations_per_child),
            len(nonterminal_outcomes),
        )

        for outcome in resolved_outcomes:
            if outcome.is_terminal:
                value_from_root = _terminal_value_for_player(outcome.winner, current_player)
            else:
                budget = nonterminal_budgets.pop(0)
                if budget <= 0:
                    continue
                outcome_env = env.clone()
                outcome_state = outcome_env.load_state(outcome.state_payload)
                outcome_rng = random.Random(py_rng.getrandbits(64))
                outcome_result = run_mcts(
                    outcome_env,
                    model,
                    outcome_state,
                    turns_taken=int(outcome.turns_taken),
                    device=device,
                    config=MCTSConfig(
                        num_simulations=int(budget),
                        c_puct=float(cfg.c_puct),
                        temperature_moves=0,
                        temperature=0.0,
                        root_dirichlet_noise=False,
                        eval_batch_size=int(cfg.eval_batch_size),
                    ),
                    rng=outcome_rng,
                )
                raw_value = float(outcome_result.root_best_value)
                value_from_root = (
                    raw_value
                    if int(outcome.current_player_id) == current_player
                    else -raw_value
                )

            if value_from_root > best_outcome_value:
                best_outcome_value = value_from_root

        if best_outcome_value == -float("inf"):
            best_outcome_value = 0.0

        child_values[action] = best_outcome_value
        q_values[action] = np.float32(best_outcome_value)
        if best_outcome_value > best_value:
            best_value = best_outcome_value
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


def run_bootstrap_mcts_search(
    env: SplendorNativeEnv,
    model,
    state,
    *,
    turns_taken: int,
    device: str = "cpu",
    mcts_config: MCTSConfig,
    bootstrap_config: BootstrapMCTSConfig | None = None,
    rng: random.Random | None = None,
) -> DeterminizedPolicyResult:
    cfg = bootstrap_config or BootstrapMCTSConfig()
    py_rng = rng or random.Random()

    mask = np.asarray(state.mask, dtype=bool)
    legal_actions = np.where(mask)[0].tolist()
    if not legal_actions:
        raise ValueError("run_bootstrap_mcts_search called with no legal actions")

    total_bootstrap_requested = int(cfg.bootstrap_simulations_per_action) * len(legal_actions)
    session_target = int(mcts_config.num_simulations) + max(0, total_bootstrap_requested)
    session = create_mcts_session(
        env,
        model,
        state=state,
        turns_taken=int(turns_taken),
        device=device,
        config=MCTSConfig(
            num_simulations=int(session_target),
            c_puct=float(mcts_config.c_puct),
            temperature_moves=int(mcts_config.temperature_moves),
            temperature=float(mcts_config.temperature),
            eps=float(mcts_config.eps),
            root_dirichlet_noise=bool(mcts_config.root_dirichlet_noise),
            root_dirichlet_epsilon=float(mcts_config.root_dirichlet_epsilon),
            root_dirichlet_alpha_total=float(mcts_config.root_dirichlet_alpha_total),
            eval_batch_size=int(cfg.eval_batch_size),
            use_forced_playouts=bool(mcts_config.use_forced_playouts),
            forced_playouts_k=float(mcts_config.forced_playouts_k),
        ),
        rng=py_rng,
    )

    if total_bootstrap_requested > 0:
        bootstrap_budgets = _split_bootstrap_budget(total_bootstrap_requested, len(legal_actions))
        for action_idx, action_budget in zip(legal_actions, bootstrap_budgets):
            if action_budget <= 0:
                continue
            session.advance(int(action_budget), forced_root_action_idx=int(action_idx))

    sims_left = int(session_target) - int(session.simulations_completed)
    result = session.advance(int(sims_left)) if sims_left > 0 else session.snapshot()
    return DeterminizedPolicyResult(
        action_idx=int(result.chosen_action_idx),
        visit_probs=np.asarray(result.visit_probs, dtype=np.float32),
        root_best_value_mean=float(result.root_best_value),
        num_determinizations=1,
        q_values=np.asarray(result.q_values, dtype=np.float32),
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
    search_type: Literal["mcts", "mcts_bootstrap", "ismcts", "alphabeta", "forced_child"] = "mcts"
    gpu_batching_enabled: bool = False
    alphabeta_config: AlphaBetaConfig = field(default_factory=AlphaBetaConfig)
    forced_child_config: ForcedChildSearchConfig = field(default_factory=ForcedChildSearchConfig)
    bootstrap_config: BootstrapMCTSConfig = field(default_factory=BootstrapMCTSConfig)

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
        # Short-circuit: if there is exactly 1 legal action, return it immediately without searching.
        mask = np.asarray(state.mask, dtype=bool)
        legal_actions = np.where(mask)[0].tolist()
        if len(legal_actions) == 1:
            action_idx = legal_actions[0]
            visit_probs = np.zeros(len(mask), dtype=np.float32)
            visit_probs[action_idx] = 1.0
            q_values = np.full(len(mask), np.nan, dtype=np.float32)
            
            return DeterminizedPolicyResult(
                action_idx=action_idx,
                visit_probs=visit_probs,
                root_best_value_mean=0.0,  # 0.0 is returned as a placeholder since value doesn't matter for a forced move
                num_determinizations=1,
                q_values=q_values,
            )

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
                eval_batch_size=int(self.mcts_config.eval_batch_size),
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
                rng=rng,
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

        if self.search_type == "mcts_bootstrap":
            return run_bootstrap_mcts_search(
                env,
                self._model,
                state,
                turns_taken=int(turns_taken),
                device=self.device,
                mcts_config=self.mcts_config,
                bootstrap_config=self.bootstrap_config,
                rng=rng,
            )

        if self.search_type == "ismcts":
            eval_batch_size = int(self.mcts_config.eval_batch_size)
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
        results = []
        
        # 1. Run the first two searches
        for _ in range(2):
            payload = build_root_determinized_payload(shadow, rng=random_source)
            res = self._choose_action_from_payload(
                payload,
                turns_taken=int(payload.get("move_number", 0)),
                rng=random_source,
            )
            results.append(res)
            
        # 2. Check for early consensus
        if results[0].action_idx == results[1].action_idx:
            return results[0]

        # 3. Only run the third search if the first two differ
        payload = build_root_determinized_payload(shadow, rng=random_source)
        res_three = self._choose_action_from_payload(
            payload,
            turns_taken=int(payload.get("move_number", 0)),
            rng=random_source,
        )
        results.append(res_three)
        
        # 4. Return the winner of the tie-break (majority vote)
        from collections import Counter
        action_counts = Counter(r.action_idx for r in results)
        best_action, _ = action_counts.most_common(1)[0]
        
        for r in results:
            if r.action_idx == best_action:
                return r
                
        return results[0]

    def choose_return_actions(
        self,
        shadow: ShadowState,
        *,
        rng: random.Random | None = None,
    ) -> tuple[DeterminizedPolicyResult, list[int]]:
        from collections import Counter

        random_source = rng or random.Random()
        all_results: list[tuple[DeterminizedPolicyResult, tuple[int, ...]]] = []
        
        for i in range(3):
            # Optimization: If the first two results are identical, skip the third search
            if len(all_results) == 2 and all_results[0][1] == all_results[1][1]:
                break

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

            if first_result is not None:
                all_results.append((first_result, tuple(chosen)))

        if not all_results:
            raise RuntimeError("Return phase produced no return actions")
            
        # Count the frequency of the generated return sequences and select the most common sequence
        sequence_counts = Counter(seq for _, seq in all_results)
        best_sequence, _ = sequence_counts.most_common(1)[0]
        
        # Return the result and sequence corresponding to the most frequent combination
        for res, seq in all_results:
            if seq == best_sequence:
                return res, list(seq)
                
        return all_results[0][0], list(all_results[0][1])