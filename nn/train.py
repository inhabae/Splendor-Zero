from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .native_env import SplendorNativeEnv, StepState
from .benchmark import BenchmarkSuiteResult, matchup_by_name, run_benchmark_suite, run_matchup
from .champions import (
    append_accepted_champion,
    build_champion_entry_from_promotion,
    get_current_and_previous_champions,
    load_champion_registry,
    save_champion_registry,
)
from .checkpoints import load_checkpoint_with_metadata, save_checkpoint
from .mcts import MCTSConfig, run_mcts
from .model import MaskedPolicyValueNet
from .opponents import CheckpointMCTSOpponent, GreedyHeuristicOpponent, RandomOpponent
from .replay import ReplayBuffer, ReplaySample
from .selfplay_dataset import run_selfplay_session_parallel
from .state_schema import ACTION_DIM, STATE_DIM

try:
    from .metrics_viz import MetricsVizLogger
except Exception:
    MetricsVizLogger = None


MASK_FILL_VALUE = -1e9
SIGN_EPS = 1e-6


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} must match mask shape {mask.shape}")
    if logits.ndim != 2 or logits.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected logits/mask shape (B,{ACTION_DIM}), got {logits.shape}")
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be torch.bool")
    if not mask.any(dim=1).all():
        raise ValueError("Each sample must have at least one legal action")
    return logits.masked_fill(~mask, MASK_FILL_VALUE)


def masked_cross_entropy_loss(logits: torch.Tensor, mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    if target_idx.ndim != 1:
        raise ValueError(f"target_idx must be shape (B,), got {target_idx.shape}")
    if target_idx.shape[0] != logits.shape[0]:
        raise ValueError("target_idx batch size mismatch")
    row_idx = torch.arange(target_idx.shape[0], device=target_idx.device)
    if not mask[row_idx, target_idx].all():
        raise ValueError("All target actions must be legal under mask")
    return F.cross_entropy(masked_logits(logits, mask), target_idx)


def masked_soft_cross_entropy_loss(logits: torch.Tensor, mask: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    if target_probs.shape != logits.shape:
        raise ValueError(f"target_probs shape {target_probs.shape} must match logits shape {logits.shape}")
    if target_probs.ndim != 2 or target_probs.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected target_probs shape (B,{ACTION_DIM}), got {target_probs.shape}")
    if not torch.isfinite(target_probs).all():
        raise ValueError("target_probs contains non-finite values")
    if (target_probs < 0).any():
        raise ValueError("target_probs cannot contain negative values")
    if (target_probs[~mask] != 0).any():
        raise ValueError("target_probs must assign zero probability to illegal actions")
    row_sums = target_probs.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0.0):
        raise ValueError("Each target_probs row must sum to 1")
    masked = masked_logits(logits, mask)
    log_probs = F.log_softmax(masked, dim=-1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def select_masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.argmax(masked_logits(logits, mask), dim=-1)


def select_masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = masked_logits(logits, mask)
    return torch.distributions.Categorical(logits=masked).sample()


def _uniform_random_legal_action(mask: np.ndarray, rng: random.Random) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("Mask contains no legal actions")
    return int(rng.choice(legal.tolist()))


def _model_sample_legal_action(
    model: MaskedPolicyValueNet,
    state_np: np.ndarray,
    mask_np: np.ndarray,
    *,
    device: str,
) -> int:
    if state_np.shape != (STATE_DIM,):
        raise ValueError(f"Expected state shape ({STATE_DIM},), got {state_np.shape}")
    if mask_np.shape != (ACTION_DIM,):
        raise ValueError(f"Expected mask shape ({ACTION_DIM},), got {mask_np.shape}")
    if not bool(mask_np.any()):
        raise ValueError("Model-sample collector requires at least one legal action")

    state_t = torch.as_tensor(state_np[None, :], dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_np[None, :], dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        logits, _value = model(state_t)
        action_t = select_masked_sample(logits, mask_t)

    action = int(action_t.item())
    if not (0 <= action < ACTION_DIM):
        raise RuntimeError(f"Model-sampled action out of range: {action}")
    if not bool(mask_np[action]):
        raise RuntimeError(f"Model-sampled illegal action {action}")
    return action


@dataclass
class _EpisodeStep:
    state: np.ndarray
    mask: np.ndarray
    action_target: int
    policy_target: np.ndarray
    player_id: int


@dataclass
class EpisodeSummary:
    num_steps: int
    num_turns: int
    reached_cutoff: bool
    winner: int  # -1 draw, 0/1 winner


def _winner_to_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    if winner not in (0, 1):
        raise ValueError(f"Unexpected winner value {winner}")
    return 1.0 if winner == player_id else -1.0


@dataclass
class CollectorStats:
    random_actions: int = 0
    model_actions: int = 0
    mcts_actions: int = 0
    mcts_sum_search_seconds: float = 0.0
    mcts_sum_root_entropy: float = 0.0
    mcts_sum_root_top1_prob: float = 0.0
    mcts_sum_selected_visit_prob: float = 0.0
    mcts_sum_root_value: float = 0.0


def _policy_entropy(policy: np.ndarray, mask: np.ndarray) -> float:
    if policy.shape != (ACTION_DIM,) or mask.shape != (ACTION_DIM,):
        raise ValueError("Unexpected policy/mask shape for entropy")
    p = policy[mask].astype(np.float64, copy=False)
    if p.size == 0:
        return 0.0
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _avg_or_zero(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def _sign_bucket_torch(x: torch.Tensor, eps: float = SIGN_EPS) -> torch.Tensor:
    pos = (x > eps).to(torch.int8)
    neg = (x < -eps).to(torch.int8)
    return pos - neg


def _print_benchmark_suite(cycle_idx: int, cycles: int, suite: BenchmarkSuiteResult) -> None:
    for matchup in suite.matchups:
        print(
            f"cycle_benchmark={cycle_idx}/{cycles} "
            f"opponent={matchup.opponent_name} "
            f"games={matchup.games} "
            f"wins={matchup.candidate_wins} "
            f"losses={matchup.candidate_losses} "
            f"draws={matchup.draws} "
            f"win_rate={matchup.candidate_win_rate:.3f} "
            f"nonloss_rate={matchup.candidate_nonloss_rate:.3f} "
            f"avg_turns={matchup.avg_turns_per_game:.2f} "
            f"cutoff_rate={matchup.cutoff_rate:.3f}"
        )
    print(
        f"cycle_benchmark_summary={cycle_idx}/{cycles} "
        f"matchups={len(suite.matchups)} "
        f"wins={suite.suite_candidate_wins} "
        f"losses={suite.suite_candidate_losses} "
        f"draws={suite.suite_draws} "
        f"suite_avg_turns={suite.suite_avg_turns_per_game:.2f}"
    )
    for warning in suite.warnings:
        print(f"cycle_benchmark_warning={cycle_idx}/{cycles} {warning}")


def _build_suite_opponents_from_registry(
    *,
    champion_registry_path: str,
    eval_mcts_config: MCTSConfig,
    device: str,
    cycle_idx: int,
    cycles: int,
) -> tuple[list[object], bool]:
    suite_opponents: list[object] = [RandomOpponent(name="random")]
    has_current_champion = False
    try:
        registry = load_champion_registry(champion_registry_path)
        champs = get_current_and_previous_champions(registry)
    except Exception as exc:
        print(f"cycle_benchmark_warning={cycle_idx}/{cycles} failed_to_load_registry={exc}")
        champs = []
    if not champs:
        print(f"cycle_benchmark_note={cycle_idx}/{cycles} no_champions_available")
        return suite_opponents, has_current_champion

    suite_opponents.append(GreedyHeuristicOpponent(name="heuristic"))
    labels = ["champion_current", "champion_prev"]
    for label, champ in zip(labels, champs):
        ckpt_path = str(champ.checkpoint_path)
        if not ckpt_path:
            print(f"cycle_benchmark_warning={cycle_idx}/{cycles} {label}=missing_checkpoint_path")
            continue
        try:
            suite_opponents.append(
                CheckpointMCTSOpponent(
                    checkpoint_path=ckpt_path,
                    mcts_config=eval_mcts_config,
                    device=device,
                    name=label,
                )
            )
            if label == "champion_current":
                has_current_champion = True
        except Exception as exc:
            print(f"cycle_benchmark_warning={cycle_idx}/{cycles} {label}_load_failed={exc}")
    return suite_opponents, has_current_champion


def _promotion_decision_from_suite(
    suite_result: BenchmarkSuiteResult,
    *,
    has_current_champion: bool,
    promotion_threshold_winrate: float,
    bootstrap_min_random_winrate: float,
    bootstrap_min_heuristic_winrate: float,
) -> tuple[bool, str, dict[str, object]]:
    metrics: dict[str, object] = {
        "promotion_eval_games": 0.0,
        "promotion_eval_candidate_vs_champion_winrate": None,
        "promotion_eval_random_winrate": None,
        "promotion_eval_heuristic_winrate": None,
    }
    champion_match = matchup_by_name(suite_result, "champion_current")
    random_match = matchup_by_name(suite_result, "random")
    heuristic_match = matchup_by_name(suite_result, "heuristic")

    if champion_match is not None:
        metrics["promotion_eval_games"] = float(champion_match.games)
        metrics["promotion_eval_candidate_vs_champion_winrate"] = float(champion_match.candidate_win_rate)
    if random_match is not None:
        metrics["promotion_eval_random_winrate"] = float(random_match.candidate_win_rate)
    if heuristic_match is not None:
        metrics["promotion_eval_heuristic_winrate"] = float(heuristic_match.candidate_win_rate)

    if has_current_champion:
        if champion_match is None:
            return False, "missing_champion_current_matchup", metrics
        if float(champion_match.candidate_win_rate) >= float(promotion_threshold_winrate):
            return True, "champion_threshold_met", metrics
        return False, "champion_threshold_not_met", metrics

    # Bootstrap path
    if random_match is None:
        return False, "missing_random_matchup", metrics
    if float(random_match.candidate_win_rate) < float(bootstrap_min_random_winrate):
        return False, "bootstrap_random_floor_not_met", metrics
    return True, "bootstrap_random_floor_met", metrics


def _validate_collector_policy(collector_policy: str) -> None:
    if collector_policy not in ("random", "model-sample", "mcts"):
        raise ValueError(f"Unsupported collector_policy: {collector_policy}")


def _find_resume_replay_path(replay_out_dir: Path, resume_run_id: str) -> Path | None:
    latest = replay_out_dir / f"{resume_run_id}_replay_latest.npz"
    if latest.exists():
        return latest
    candidates = sorted(replay_out_dir.glob(f"{resume_run_id}_cycle_*_replay.npz"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _one_hot_policy_target(mask: np.ndarray, action: int) -> np.ndarray:
    if mask.shape != (ACTION_DIM,):
        raise ValueError(f"Expected mask shape ({ACTION_DIM},), got {mask.shape}")
    if not (0 <= int(action) < ACTION_DIM):
        raise ValueError(f"Action out of range for one-hot policy target: {action}")
    if not bool(mask[int(action)]):
        raise ValueError("One-hot policy target action must be legal")
    target = np.zeros((ACTION_DIM,), dtype=np.float32)
    target[int(action)] = 1.0
    return target


def collect_episode(
    env: SplendorNativeEnv,
    replay: ReplayBuffer,
    *,
    seed: int,
    max_turns: int,
    rng: random.Random,
    collector_policy: str,
    model: Optional[MaskedPolicyValueNet] = None,
    device: str = "cpu",
    collector_stats: Optional[CollectorStats] = None,
    mcts_config: Optional[MCTSConfig] = None,
) -> EpisodeSummary:
    state = env.reset(seed=seed)
    episode_steps: List[_EpisodeStep] = []

    reached_cutoff = False
    winner = -1
    turns_taken = 0

    while turns_taken < max_turns:
        if state.is_terminal:
            winner = state.winner
            break

        if state.state.shape != (STATE_DIM,):
            raise AssertionError(f"State shape mismatch: {state.state.shape}")
        if state.mask.shape != (ACTION_DIM,):
            raise AssertionError(f"Mask shape mismatch: {state.mask.shape}")
        if not state.mask.any():
            raise AssertionError("No legal actions in non-terminal state")

        player_id = env.current_player_id
        if collector_policy == "random":
            action = _uniform_random_legal_action(state.mask, rng)
            policy_target = _one_hot_policy_target(state.mask, action)
            if collector_stats is not None:
                collector_stats.random_actions += 1
        elif collector_policy == "model-sample":
            if model is None:
                raise ValueError("collector_policy='model-sample' requires a model")
            action = _model_sample_legal_action(model, state.state, state.mask, device=device)
            policy_target = _one_hot_policy_target(state.mask, action)
            if collector_stats is not None:
                collector_stats.model_actions += 1
        elif collector_policy == "mcts":
            if model is None:
                raise ValueError("collector_policy='mcts' requires a model")
            t0 = time.perf_counter()
            mcts_result = run_mcts(
                env,
                model,
                state,
                turns_taken=turns_taken,
                device=device,
                config=mcts_config,
                rng=rng,
            )
            elapsed = time.perf_counter() - t0
            action = int(mcts_result.chosen_action_idx)
            policy_target = mcts_result.visit_probs.astype(np.float32, copy=True)
            if collector_stats is not None:
                collector_stats.mcts_actions += 1
                collector_stats.mcts_sum_search_seconds += float(elapsed)
                collector_stats.mcts_sum_root_entropy += _policy_entropy(policy_target, state.mask)
                legal_probs = policy_target[state.mask]
                collector_stats.mcts_sum_root_top1_prob += float(np.max(legal_probs)) if legal_probs.size > 0 else 0.0
                collector_stats.mcts_sum_selected_visit_prob += float(policy_target[action])
                collector_stats.mcts_sum_root_value += float(mcts_result.root_value)
        else:
            raise ValueError(f"Unknown collector_policy: {collector_policy}")
        if not bool(state.mask[action]):
            raise AssertionError("Sampled action is not legal")

        prev_player_id = env.current_player_id
        episode_steps.append(
            _EpisodeStep(
                state=state.state.copy(),
                mask=state.mask.copy(),
                action_target=action,
                policy_target=policy_target.copy(),
                player_id=player_id,
            )
        )
        state = env.step(action)
        if env.current_player_id != prev_player_id:
            turns_taken += 1
        if state.is_terminal:
            winner = state.winner
            break
    else:
        reached_cutoff = True
        winner = -1

    if not reached_cutoff and not state.is_terminal:
        # Defensive fallback for unusual protocol behavior.
        winner = -1
        reached_cutoff = True

    for step in episode_steps:
        replay.add(
            ReplaySample(
                state=step.state,
                mask=step.mask,
                action_target=step.action_target,
                value_target=_winner_to_value_for_player(winner, step.player_id),
                policy_target=step.policy_target,
            )
        )

    return EpisodeSummary(
        num_steps=len(episode_steps),
        num_turns=turns_taken,
        reached_cutoff=reached_cutoff,
        winner=winner,
    )


def _collect_replay(
    env: SplendorNativeEnv,
    replay: ReplayBuffer,
    *,
    episodes: int,
    max_turns: int,
    rng: random.Random,
    collector_policy: str,
    model: MaskedPolicyValueNet,
    device: str,
    seed_start: int,
    mcts_config: Optional[MCTSConfig] = None,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    _validate_collector_policy(collector_policy)

    cutoff_count = 0
    total_steps = 0
    total_turns = 0
    terminal_episodes = 0
    collector_stats = CollectorStats()
    next_seed = seed_start

    for _ in range(episodes):
        summary = collect_episode(
            env,
            replay,
            seed=next_seed,
            max_turns=max_turns,
            rng=rng,
            collector_policy=collector_policy,
            model=model,
            device=device,
            collector_stats=collector_stats,
            mcts_config=mcts_config,
        )
        next_seed += 1
        total_steps += summary.num_steps
        total_turns += summary.num_turns
        if summary.reached_cutoff:
            cutoff_count += 1
        else:
            terminal_episodes += 1

    mcts_n = max(collector_stats.mcts_actions, 1)
    has_mcts = collector_stats.mcts_actions > 0

    return {
        "episodes": float(episodes),
        "terminal_episodes": float(terminal_episodes),
        "cutoff_episodes": float(cutoff_count),
        "replay_samples": float(len(replay)),
        "total_steps": float(total_steps),
        "total_turns": float(total_turns),
        "collector_random_actions": float(collector_stats.random_actions),
        "collector_model_actions": float(collector_stats.model_actions),
        "collector_mcts_actions": float(collector_stats.mcts_actions),
        "mcts_avg_search_ms": (1000.0 * collector_stats.mcts_sum_search_seconds / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_entropy": (collector_stats.mcts_sum_root_entropy / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_top1_visit_prob": (collector_stats.mcts_sum_root_top1_prob / mcts_n) if has_mcts else 0.0,
        "mcts_avg_selected_visit_prob": (collector_stats.mcts_sum_selected_visit_prob / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_value": (collector_stats.mcts_sum_root_value / mcts_n) if has_mcts else 0.0,
        "collector_workers_used": 1.0,
        "next_seed": int(next_seed),
    }


def _collect_replay_parallel_mcts(
    replay: ReplayBuffer,
    *,
    episodes: int,
    max_turns: int,
    model: MaskedPolicyValueNet,
    seed_start: int,
    mcts_config: MCTSConfig,
    workers: int,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    workers_used = min(int(workers), int(episodes))

    with tempfile.TemporaryDirectory(prefix="splendor_parallel_collect_") as tmp_dir:
        checkpoint = save_checkpoint(
            model,
            output_dir=tmp_dir,
            run_id="parallel_collect",
            cycle_idx=0,
            metadata={"seed": int(seed_start), "mcts_sims": int(mcts_config.num_simulations)},
        )
        session = run_selfplay_session_parallel(
            checkpoint_path=str(checkpoint.path),
            games=int(episodes),
            max_turns=int(max_turns),
            num_simulations=int(mcts_config.num_simulations),
            seed_base=int(seed_start),
            workers=int(workers_used),
        )

    steps_by_episode: dict[int, dict[str, int | bool]] = {}
    for step in session.steps:
        replay.add(
            ReplaySample(
                state=step.state,
                mask=step.mask,
                action_target=int(step.action_selected),
                value_target=float(step.value_target),
                policy_target=step.policy,
            )
        )
        ep = int(step.episode_idx)
        row = steps_by_episode.get(ep)
        if row is None:
            row = {
                "count": 0,
                "max_turn_idx": -1,
                "reached_cutoff": bool(step.reached_cutoff),
            }
            steps_by_episode[ep] = row
        row["count"] = int(row["count"]) + 1
        row["max_turn_idx"] = max(int(row["max_turn_idx"]), int(step.turn_idx))
        if bool(step.reached_cutoff):
            row["reached_cutoff"] = True

    cutoff_count = 0
    total_steps = 0
    total_turns = 0
    for ep_idx in range(int(episodes)):
        row = steps_by_episode.get(ep_idx)
        if row is None:
            continue
        total_steps += int(row["count"])
        if int(row["max_turn_idx"]) >= 0:
            total_turns += int(row["max_turn_idx"]) + 1
        if bool(row["reached_cutoff"]):
            cutoff_count += 1

    terminal_episodes = int(episodes) - int(cutoff_count)
    return {
        "episodes": float(episodes),
        "terminal_episodes": float(terminal_episodes),
        "cutoff_episodes": float(cutoff_count),
        "replay_samples": float(len(replay)),
        "total_steps": float(total_steps),
        "total_turns": float(total_turns),
        "collector_random_actions": 0.0,
        "collector_model_actions": 0.0,
        "collector_mcts_actions": float(total_steps),
        # Per-action search timing/root diagnostics are not emitted by the parallel worker API.
        "mcts_avg_search_ms": 0.0,
        "mcts_avg_root_entropy": 0.0,
        "mcts_avg_root_top1_visit_prob": 0.0,
        "mcts_avg_selected_visit_prob": 0.0,
        "mcts_avg_root_value": 0.0,
        "collector_workers_used": float(workers_used),
        "next_seed": int(seed_start + episodes),
    }


def train_one_step(
    model: MaskedPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    value_loss_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    states = batch["state"]
    masks = batch["mask"]
    action_target = batch["action_target"]
    policy_target = batch["policy_target"]
    value_target = batch["value_target"]

    logits, value_pred = model(states)
    if not torch.isfinite(logits).all() or not torch.isfinite(value_pred).all():
        raise RuntimeError("Model produced non-finite outputs")

    policy_loss = masked_soft_cross_entropy_loss(logits, masks, policy_target)
    value_loss = F.mse_loss(value_pred, value_target)
    total_loss = policy_loss + value_loss_weight * value_loss

    if not torch.isfinite(total_loss):
        raise RuntimeError("Non-finite total loss")

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    with torch.no_grad():
        row_idx = torch.arange(action_target.shape[0], device=action_target.device)
        legal_target_ok = bool(masks[row_idx, action_target].all().item())
        pred_action = select_masked_argmax(logits, masks)
        action_top1_acc = float((pred_action == action_target).float().mean().item())
        value_mae = float(torch.abs(value_pred - value_target).mean().item())
        value_sign_acc = float(
            (_sign_bucket_torch(value_pred) == _sign_bucket_torch(value_target)).float().mean().item()
        )

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(total_loss.item()),
        "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
        "legal_target_ok": 1.0 if legal_target_ok else 0.0,
        "action_top1_acc": action_top1_acc,
        "value_sign_acc": value_sign_acc,
        "value_mae": value_mae,
    }


def _train_on_replay(
    model: MaskedPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    *,
    batch_size: int,
    train_steps: int,
    log_every: int,
    device: str,
    value_loss_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
    log_prefix: str = "",
    step_metrics_callback: Optional[Callable[[int, dict[str, float]], None]] = None,
) -> dict[str, object]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if len(replay) == 0:
        raise RuntimeError("Replay buffer is empty")

    metrics: dict[str, object] = {}
    sum_policy_loss = 0.0
    sum_value_loss = 0.0
    sum_total_loss = 0.0
    sum_grad_norm = 0.0

    for step in range(1, train_steps + 1):
        batch = replay.sample_batch(min(batch_size, len(replay)), device=device)
        metrics = train_one_step(
            model,
            optimizer,
            batch,
            value_loss_weight=value_loss_weight,
            grad_clip_norm=grad_clip_norm,
        )

        sum_policy_loss += float(metrics["policy_loss"])
        sum_value_loss += float(metrics["value_loss"])
        sum_total_loss += float(metrics["total_loss"])
        sum_grad_norm += float(metrics["grad_norm"])
        if step_metrics_callback is not None:
            step_metrics_callback(step, metrics)

        if step == 1 or step % log_every == 0 or step == train_steps:
            print(
                f"{log_prefix}train_step={step}/{train_steps} "
                f"policy_loss={metrics['policy_loss']:.6f} "
                f"value_loss={metrics['value_loss']:.6f} "
                f"total_loss={metrics['total_loss']:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f}"
            )

    metrics.update(
        {
            "train_steps": float(train_steps),
            "avg_policy_loss": sum_policy_loss / train_steps,
            "avg_value_loss": sum_value_loss / train_steps,
            "avg_total_loss": sum_total_loss / train_steps,
            "avg_grad_norm": sum_grad_norm / train_steps,
        }
    )
    return metrics


def _evaluate_on_replay_full(
    model: MaskedPolicyValueNet,
    replay: ReplayBuffer,
    *,
    device: str,
    value_loss_weight: float = 1.0,
) -> dict[str, float]:
    if len(replay) == 0:
        raise RuntimeError("Replay buffer is empty")

    # ReplayBuffer.sample_batch with k=len(replay) returns the full buffer once (order shuffled).
    batch = replay.sample_batch(len(replay), device=device)
    states = batch["state"]
    masks = batch["mask"]
    action_target = batch["action_target"]
    policy_target = batch["policy_target"]
    value_target = batch["value_target"]

    model.eval()
    with torch.no_grad():
        logits, value_pred = model(states)
        policy_loss = masked_soft_cross_entropy_loss(logits, masks, policy_target)
        value_loss = F.mse_loss(value_pred, value_target)
        total_loss = policy_loss + value_loss_weight * value_loss
        pred_action = select_masked_argmax(logits, masks)
        action_top1_acc = (pred_action == action_target).float().mean()
        value_sign_acc = (_sign_bucket_torch(value_pred) == _sign_bucket_torch(value_target)).float().mean()
        value_mae = torch.abs(value_pred - value_target).mean()

    return {
        "eval_policy_loss": float(policy_loss.item()),
        "eval_value_loss": float(value_loss.item()),
        "eval_total_loss": float(total_loss.item()),
        "eval_samples": float(len(replay)),
        "eval_action_top1_acc": float(action_top1_acc.item()),
        "eval_value_sign_acc": float(value_sign_acc.item()),
        "eval_value_mae": float(value_mae.item()),
    }


def run_smoke(
    *,
    episodes: int = 5,
    max_turns: int = 80,
    batch_size: int = 32,
    collector_policy: str = "random",
    train_steps: int = 1,
    log_every: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    mcts_sims: int = 64,
    mcts_c_puct: float = 1.25,
    mcts_temperature_moves: int = 10,
    mcts_temperature: float = 1.0,
    mcts_root_dirichlet_noise: bool = False,
    mcts_root_dirichlet_epsilon: float = 0.25,
    mcts_root_dirichlet_alpha_total: float = 10.0,
    visualize: bool = False,
    viz_dir: str = "nn_artifacts/viz",
    viz_run_name: str | None = None,
    viz_save_every_cycle: int = 1,
    viz_smoke_enable: bool = False,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    _validate_collector_policy(collector_policy)

    replay = ReplayBuffer()
    rng = random.Random(seed)
    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        c_puct=mcts_c_puct,
        temperature_moves=mcts_temperature_moves,
        temperature=mcts_temperature,
        root_dirichlet_noise=mcts_root_dirichlet_noise,
        root_dirichlet_epsilon=mcts_root_dirichlet_epsilon,
        root_dirichlet_alpha_total=mcts_root_dirichlet_alpha_total,
    )

    model = MaskedPolicyValueNet().to(device)
    run_id = f"smoke_{int(time.time())}_{int(seed)}"
    do_viz = bool(viz_smoke_enable)
    viz_logger = None
    if do_viz:
        if MetricsVizLogger is None:
            raise RuntimeError(
                "Visualization dependencies unavailable. Install required packages: tensorboard and matplotlib."
            )
        run_name = str(viz_run_name) if viz_run_name else run_id
        viz_logger = MetricsVizLogger(
            mode="smoke",
            run_id=run_id,
            root_dir=viz_dir,
            run_name=run_name,
            save_every_cycle=viz_save_every_cycle,
        )

    with SplendorNativeEnv() as env:
        collection_metrics = _collect_replay(
            env,
            replay,
            episodes=episodes,
            max_turns=max_turns,
            rng=rng,
            collector_policy=collector_policy,
            model=model,
            device=device,
            seed_start=seed,
            mcts_config=mcts_config,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    smoke_step_callback = None
    if viz_logger is not None:
        def _smoke_step_callback(step: int, m: dict[str, float]) -> None:
            for metric_name, key in (
                ("train/policy_loss_step", "policy_loss"),
                ("train/value_loss_step", "value_loss"),
                ("train/total_loss_step", "total_loss"),
                ("train/grad_norm_step", "grad_norm"),
                ("train/action_top1_acc_step", "action_top1_acc"),
                ("train/value_sign_acc_step", "value_sign_acc"),
                ("train/value_mae_step", "value_mae"),
            ):
                viz_logger.log_scalar(
                    metric_name,
                    float(m[key]),
                    cycle=1,
                    global_step=int(step),
                    axis_cycle=1.0,
                    axis_step=float(step),
                )
        smoke_step_callback = _smoke_step_callback

    metrics = _train_on_replay(
        model,
        optimizer,
        replay,
        batch_size=batch_size,
        train_steps=train_steps,
        log_every=log_every,
        device=device,
        step_metrics_callback=smoke_step_callback,
    )
    if viz_logger is not None:
        viz_logger.log_scalar(
            "train/policy_loss_cycle_avg",
            float(metrics["avg_policy_loss"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "train/value_loss_cycle_avg",
            float(metrics["avg_value_loss"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "train/total_loss_cycle_avg",
            float(metrics["avg_total_loss"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        eval_metrics = _evaluate_on_replay_full(model, replay, device=device)
        metrics.update(eval_metrics)
        viz_logger.log_scalar(
            "eval/policy_loss_full_replay",
            float(eval_metrics["eval_policy_loss"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "eval/value_loss_full_replay",
            float(eval_metrics["eval_value_loss"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "eval/total_loss_full_replay",
            float(eval_metrics["eval_total_loss"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "eval/samples_full_replay",
            float(eval_metrics["eval_samples"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "eval/action_top1_acc",
            float(eval_metrics["eval_action_top1_acc"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "eval/value_sign_acc",
            float(eval_metrics["eval_value_sign_acc"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.log_scalar(
            "eval/value_mae",
            float(eval_metrics["eval_value_mae"]),
            cycle=1,
            global_step=int(metrics["train_steps"]),
            axis_cycle=1.0,
            axis_step=float(metrics["train_steps"]),
        )
        viz_logger.maybe_save(1)
        viz_logger.finalize()

    metrics.update(
        {
            "mode": "smoke",
            "collector_policy": collector_policy,
            "collector_random_actions": collection_metrics["collector_random_actions"],
            "collector_model_actions": collection_metrics["collector_model_actions"],
            "collector_mcts_actions": collection_metrics["collector_mcts_actions"],
            "mcts_avg_search_ms": collection_metrics["mcts_avg_search_ms"],
            "mcts_avg_root_entropy": collection_metrics["mcts_avg_root_entropy"],
            "mcts_avg_root_top1_visit_prob": collection_metrics["mcts_avg_root_top1_visit_prob"],
            "mcts_avg_selected_visit_prob": collection_metrics["mcts_avg_selected_visit_prob"],
            "mcts_avg_root_value": collection_metrics["mcts_avg_root_value"],
            "episodes": collection_metrics["episodes"],
            "terminal_episodes": collection_metrics["terminal_episodes"],
            "cutoff_episodes": collection_metrics["cutoff_episodes"],
            "replay_samples": collection_metrics["replay_samples"],
            "total_steps": collection_metrics["total_steps"],
            "total_turns": collection_metrics["total_turns"],
            "avg_turns_per_episode": _avg_or_zero(float(collection_metrics["total_turns"]), float(collection_metrics["episodes"])),
            "avg_steps_per_episode": _avg_or_zero(float(collection_metrics["total_steps"]), float(collection_metrics["episodes"])),
            "avg_steps_per_turn": _avg_or_zero(float(collection_metrics["total_steps"]), float(collection_metrics["total_turns"])),
        }
    )
    return metrics


def run_cycles(
    *,
    cycles: int = 3,
    episodes_per_cycle: int = 5,
    train_steps_per_cycle: int = 50,
    max_turns: int = 80,
    batch_size: int = 32,
    collector_policy: str = "random",
    log_every: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    mcts_sims: int = 64,
    mcts_c_puct: float = 1.25,
    mcts_temperature_moves: int = 10,
    mcts_temperature: float = 1.0,
    mcts_root_dirichlet_noise: bool = False,
    mcts_root_dirichlet_epsilon: float = 0.25,
    mcts_root_dirichlet_alpha_total: float = 10.0,
    save_checkpoint_every_cycles: int = 0,
    checkpoint_dir: str = "nn_artifacts/checkpoints",
    benchmark_suite: bool = False,
    benchmark_games_per_opponent: int = 40,
    benchmark_mcts_sims: int = 200,
    champion_registry_path: str = "nn_artifacts/champions.json",
    benchmark_seed: int | None = 42,
    resume_checkpoint: str | None = None,
    resume_run_id_suffix: str = "resume",
    resume_replay_path: str | None = None,
    auto_promote: bool = False,
    promotion_games: int = 50,
    promotion_threshold_winrate: float = 0.65,
    promotion_benchmark_mcts_sims: int | None = None,
    bootstrap_min_heuristic_winrate: float = 0.20,
    bootstrap_min_random_winrate: float = 0.75,
    rolling_replay: bool = False,
    replay_capacity: int = 50_000,
    replay_save_every_cycles: int = 0,
    visualize: bool = False,
    viz_dir: str = "nn_artifacts/viz",
    viz_run_name: str | None = None,
    collector_workers: int | None = None,
    benchmark_workers: int = 1,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if episodes_per_cycle <= 0:
        raise ValueError("episodes_per_cycle must be positive")
    if train_steps_per_cycle <= 0:
        raise ValueError("train_steps_per_cycle must be positive")
    if benchmark_games_per_opponent <= 0:
        raise ValueError("benchmark_games_per_opponent must be positive")
    if benchmark_mcts_sims <= 0:
        raise ValueError("benchmark_mcts_sims must be positive")
    if promotion_games <= 0:
        raise ValueError("promotion_games must be positive")
    if save_checkpoint_every_cycles < 0:
        raise ValueError("save_checkpoint_every_cycles must be >= 0")
    if auto_promote and save_checkpoint_every_cycles <= 0:
        raise ValueError("auto_promote requires save_checkpoint_every_cycles > 0")
    if not (0.0 <= promotion_threshold_winrate <= 1.0):
        raise ValueError("promotion_threshold_winrate must be in [0,1]")
    if not (0.0 <= bootstrap_min_heuristic_winrate <= 1.0):
        raise ValueError("bootstrap_min_heuristic_winrate must be in [0,1]")
    if not (0.0 <= bootstrap_min_random_winrate <= 1.0):
        raise ValueError("bootstrap_min_random_winrate must be in [0,1]")
    if replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if replay_save_every_cycles < 0:
        raise ValueError("replay_save_every_cycles must be >= 0")
    if collector_workers is not None and int(collector_workers) <= 0:
        raise ValueError("collector_workers must be positive when provided")
    if benchmark_workers <= 0:
        raise ValueError("benchmark_workers must be positive")
    _validate_collector_policy(collector_policy)

    resumed_from_metadata: dict[str, object] = {}
    resume_base_cycle_idx = 0
    resume_from_run_id: str | None = None
    if resume_checkpoint:
        loaded_ckpt = load_checkpoint_with_metadata(resume_checkpoint, device=device)
        model = loaded_ckpt.model.to(device)
        resume_base_cycle_idx = int(loaded_ckpt.cycle_idx)
        resume_from_run_id = str(loaded_ckpt.run_id)
        resumed_from_metadata = {
            "resume_from_checkpoint_path": str(loaded_ckpt.path),
            "resume_from_run_id": loaded_ckpt.run_id,
            "resume_from_cycle_idx": float(loaded_ckpt.cycle_idx),
            "resume_from_created_at": loaded_ckpt.created_at,
        }
        if resume_base_cycle_idx <= 0:
            print(f"resume_warning=missing_or_zero_cycle_idx checkpoint={loaded_ckpt.path}")
    else:
        model = MaskedPolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = random.Random(seed)
    next_episode_seed = seed
    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        c_puct=mcts_c_puct,
        temperature_moves=mcts_temperature_moves,
        temperature=mcts_temperature,
        root_dirichlet_noise=mcts_root_dirichlet_noise,
        root_dirichlet_epsilon=mcts_root_dirichlet_epsilon,
        root_dirichlet_alpha_total=mcts_root_dirichlet_alpha_total,
    )
    eval_mcts_config = MCTSConfig(
        num_simulations=int(benchmark_mcts_sims),
        c_puct=mcts_c_puct,
        temperature_moves=0,
        temperature=0.0,
        root_dirichlet_noise=False,
    )
    promotion_eval_sims = int(benchmark_mcts_sims if promotion_benchmark_mcts_sims is None else promotion_benchmark_mcts_sims)
    if promotion_eval_sims <= 0:
        raise ValueError("promotion_benchmark_mcts_sims must be positive")
    promotion_eval_mcts_config = MCTSConfig(
        num_simulations=promotion_eval_sims,
        c_puct=mcts_c_puct,
        temperature_moves=0,
        temperature=0.0,
        root_dirichlet_noise=False,
    )
    base_run_id = f"train_{int(time.time())}_{int(seed)}"
    run_id = f"{base_run_id}_{resume_run_id_suffix}" if resume_checkpoint else base_run_id
    effective_benchmark_seed = int(seed if benchmark_seed is None else benchmark_seed)
    effective_benchmark_workers = int(benchmark_workers)
    viz_logger = None
    if visualize:
        if MetricsVizLogger is None:
            raise RuntimeError(
                "Visualization dependencies unavailable. Install required packages: tensorboard and matplotlib."
            )
        run_name = str(viz_run_name) if viz_run_name else run_id
        viz_save_stride = int(save_checkpoint_every_cycles) if int(save_checkpoint_every_cycles) > 0 else 1
        viz_logger = MetricsVizLogger(
            mode="cycles",
            run_id=run_id,
            root_dir=viz_dir,
            run_name=run_name,
            save_every_cycle=viz_save_stride,
        )

    total_episodes = 0.0
    total_terminal_episodes = 0.0
    total_cutoff_episodes = 0.0
    total_replay_samples = 0.0
    total_steps = 0.0
    total_turns = 0.0
    total_random_actions = 0.0
    total_model_actions = 0.0
    total_mcts_actions = 0.0
    weighted_sum_mcts_avg_search_ms = 0.0
    weighted_sum_mcts_avg_root_entropy = 0.0
    weighted_sum_mcts_avg_root_top1_visit_prob = 0.0
    weighted_sum_mcts_avg_selected_visit_prob = 0.0
    weighted_sum_mcts_avg_root_value = 0.0

    total_train_steps = 0.0
    weighted_sum_avg_policy_loss = 0.0
    weighted_sum_avg_value_loss = 0.0
    weighted_sum_avg_total_loss = 0.0
    weighted_sum_avg_grad_norm = 0.0
    weighted_sum_eval_policy_loss = 0.0
    weighted_sum_eval_value_loss = 0.0
    weighted_sum_eval_total_loss = 0.0
    total_eval_samples = 0.0

    last_train_metrics: dict[str, object] = {}
    last_eval_metrics: dict[str, object] = {}
    last_benchmark_metrics: dict[str, object] = {}
    last_promotion_metrics: dict[str, object] = {}
    replay = ReplayBuffer(max_size=(replay_capacity if rolling_replay else None))
    auto_workers = int(os.cpu_count() or 1)
    if collector_policy == "mcts":
        requested_workers = int(collector_workers) if collector_workers is not None else auto_workers
    else:
        requested_workers = 1
    replay_out_dir = Path("nn_artifacts/replay")
    rolling_replay_state_path = replay_out_dir / f"{run_id}_replay_latest.npz"
    last_saved_replay_path: Path | None = None
    if rolling_replay:
        selected_resume_replay_path: Path | None = None
        if resume_replay_path:
            selected_resume_replay_path = Path(resume_replay_path)
        elif resume_checkpoint and resume_from_run_id:
            selected_resume_replay_path = _find_resume_replay_path(replay_out_dir, resume_from_run_id)
        if selected_resume_replay_path is not None:
            if not selected_resume_replay_path.exists():
                raise FileNotFoundError(f"resume_replay_path not found: {selected_resume_replay_path}")
            replay = ReplayBuffer.load_npz(selected_resume_replay_path)
            resumed_from_metadata["resume_from_replay_path"] = str(selected_resume_replay_path.resolve())
            resumed_from_metadata["resume_from_replay_samples"] = float(len(replay))
            print(
                f"resume_replay_loaded path={selected_resume_replay_path.resolve()} "
                f"samples={len(replay)}"
            )
    elif resume_replay_path:
        print("resume_replay_note=resume_replay_path_ignored_without_rolling_replay")

    with SplendorNativeEnv() as env:
        for cycle_idx in range(1, cycles + 1):
            global_cycle_idx = resume_base_cycle_idx + cycle_idx
            checkpoint_cycle = (
                int(save_checkpoint_every_cycles) > 0
                and (global_cycle_idx % int(save_checkpoint_every_cycles) == 0)
            )
            if not rolling_replay:
                replay = ReplayBuffer()
            replay_size_before_collect = len(replay)
            if collector_policy == "mcts" and requested_workers > 1:
                collection_metrics = _collect_replay_parallel_mcts(
                    replay,
                    episodes=episodes_per_cycle,
                    max_turns=max_turns,
                    model=model,
                    seed_start=next_episode_seed,
                    mcts_config=mcts_config,
                    workers=min(int(requested_workers), int(episodes_per_cycle)),
                )
            else:
                collection_metrics = _collect_replay(
                    env,
                    replay,
                    episodes=episodes_per_cycle,
                    max_turns=max_turns,
                    rng=rng,
                    collector_policy=collector_policy,
                    model=model,
                    device=device,
                    seed_start=next_episode_seed,
                    mcts_config=mcts_config,
                )
            next_episode_seed = int(collection_metrics["next_seed"])
            replay_added = len(replay) - replay_size_before_collect

            step_metrics_callback = None
            if viz_logger is not None:
                def _cycle_step_callback(step: int, m: dict[str, float]) -> None:
                    global_step = (global_cycle_idx - 1) * int(train_steps_per_cycle) + int(step)
                    for metric_name, key in (
                        ("train/policy_loss_step", "policy_loss"),
                        ("train/value_loss_step", "value_loss"),
                        ("train/total_loss_step", "total_loss"),
                        ("train/grad_norm_step", "grad_norm"),
                        ("train/action_top1_acc_step", "action_top1_acc"),
                        ("train/value_sign_acc_step", "value_sign_acc"),
                        ("train/value_mae_step", "value_mae"),
                    ):
                        viz_logger.log_scalar(
                            metric_name,
                            float(m[key]),
                            cycle=global_cycle_idx,
                            global_step=global_step,
                            axis_cycle=float(global_cycle_idx),
                            axis_step=float(global_step),
                        )
                step_metrics_callback = _cycle_step_callback

            train_metrics = _train_on_replay(
                model,
                optimizer,
                replay,
                batch_size=batch_size,
                train_steps=train_steps_per_cycle,
                log_every=log_every,
                device=device,
                log_prefix=f"cycle={cycle_idx}/{cycles} ",
                step_metrics_callback=step_metrics_callback,
            )
            last_train_metrics = train_metrics
            eval_metrics = _evaluate_on_replay_full(
                model,
                replay,
                device=device,
            )
            last_eval_metrics = eval_metrics
            if viz_logger is not None:
                cycle_global_step = global_cycle_idx * int(train_steps_per_cycle)
                for metric_name, key in (
                    ("train/policy_loss_cycle_avg", "avg_policy_loss"),
                    ("train/value_loss_cycle_avg", "avg_value_loss"),
                    ("train/total_loss_cycle_avg", "avg_total_loss"),
                ):
                    viz_logger.log_scalar(
                        metric_name,
                        float(train_metrics[key]),
                        cycle=global_cycle_idx,
                        global_step=cycle_global_step,
                        axis_cycle=float(global_cycle_idx),
                        axis_step=float(cycle_global_step),
                    )
                for metric_name, key in (
                    ("eval/policy_loss_full_replay", "eval_policy_loss"),
                    ("eval/value_loss_full_replay", "eval_value_loss"),
                    ("eval/total_loss_full_replay", "eval_total_loss"),
                    ("eval/samples_full_replay", "eval_samples"),
                    ("eval/action_top1_acc", "eval_action_top1_acc"),
                    ("eval/value_sign_acc", "eval_value_sign_acc"),
                    ("eval/value_mae", "eval_value_mae"),
                ):
                    viz_logger.log_scalar(
                        metric_name,
                        float(eval_metrics[key]),
                        cycle=global_cycle_idx,
                        global_step=cycle_global_step,
                        axis_cycle=float(global_cycle_idx),
                        axis_step=float(cycle_global_step),
                    )
                viz_logger.maybe_save(global_cycle_idx)

            print(
                f"cycle_summary={cycle_idx}/{cycles} "
                f"collector_policy={collector_policy} "
                f"collector_workers_used={int(collection_metrics.get('collector_workers_used', 1.0))} "
                f"rolling_replay={int(rolling_replay)} "
                f"replay_buffer_size={len(replay)} "
                f"replay_added={replay_added} "
                f"replay_samples={collection_metrics['replay_samples']} "
                f"terminal_episodes={collection_metrics['terminal_episodes']} "
                f"cutoff_episodes={collection_metrics['cutoff_episodes']} "
                f"total_steps={collection_metrics['total_steps']} "
                f"total_turns={collection_metrics['total_turns']} "
                f"avg_turns_per_episode={_avg_or_zero(float(collection_metrics['total_turns']), float(collection_metrics['episodes'])):.2f} "
                f"avg_steps_per_episode={_avg_or_zero(float(collection_metrics['total_steps']), float(collection_metrics['episodes'])):.2f} "
                f"avg_steps_per_turn={_avg_or_zero(float(collection_metrics['total_steps']), float(collection_metrics['total_turns'])):.3f} "
                f"avg_policy_loss={train_metrics['avg_policy_loss']:.6f} "
                f"avg_value_loss={train_metrics['avg_value_loss']:.6f} "
                f"avg_total_loss={train_metrics['avg_total_loss']:.6f} "
                f"avg_grad_norm={train_metrics['avg_grad_norm']:.6f} "
                f"final_total_loss={train_metrics['total_loss']:.6f} "
                f"eval_policy_loss={eval_metrics['eval_policy_loss']:.8f} "
                f"eval_value_loss={eval_metrics['eval_value_loss']:.8f} "
                f"eval_total_loss={eval_metrics['eval_total_loss']:.8f}"
            )
            if collector_policy == "mcts" and float(collection_metrics["collector_mcts_actions"]) > 0:
                print(
                    f"cycle_mcts={cycle_idx}/{cycles} "
                    f"avg_search_ms={float(collection_metrics['mcts_avg_search_ms']):.3f} "
                    f"avg_root_entropy={float(collection_metrics['mcts_avg_root_entropy']):.4f} "
                    f"avg_root_top1={float(collection_metrics['mcts_avg_root_top1_visit_prob']):.4f} "
                    f"avg_selected_visit={float(collection_metrics['mcts_avg_selected_visit_prob']):.4f} "
                    f"avg_root_value={float(collection_metrics['mcts_avg_root_value']):.4f}"
                )

            checkpoint_info = None
            should_save_checkpoint_this_cycle = checkpoint_cycle
            if should_save_checkpoint_this_cycle:
                checkpoint_info = save_checkpoint(
                    model,
                    output_dir=checkpoint_dir,
                    run_id=run_id,
                    cycle_idx=global_cycle_idx,
                    metadata={
                        "seed": seed,
                        "collector_policy": collector_policy,
                        "mcts_sims": mcts_sims,
                        "cycle_idx": global_cycle_idx,
                        **resumed_from_metadata,
                    },
                )
                print(f"cycle_checkpoint={cycle_idx}/{cycles} path={checkpoint_info.path}")

            if benchmark_suite and checkpoint_cycle:
                suite_opponents = [GreedyHeuristicOpponent(name="heuristic")]
                if checkpoint_info is None:
                    raise RuntimeError("Benchmark suite requires candidate checkpoint")
                candidate_policy = CheckpointMCTSOpponent(
                    checkpoint_path=str(checkpoint_info.path),
                    mcts_config=eval_mcts_config,
                    device=device,
                    name="candidate",
                )
                suite_result = run_benchmark_suite(
                    candidate_checkpoint=str(checkpoint_info.path),
                    candidate_policy=candidate_policy,
                    suite_opponents=suite_opponents,
                    games_per_opponent=benchmark_games_per_opponent,
                    max_turns=max_turns,
                    seed_base=effective_benchmark_seed,
                    cycle_idx=cycle_idx,
                    parallel_workers=min(int(effective_benchmark_workers), int(benchmark_games_per_opponent)),
                )
                _print_benchmark_suite(cycle_idx, cycles, suite_result)
                last_benchmark_metrics = {
                    "benchmark_matchups": float(len(suite_result.matchups)),
                    "benchmark_suite_candidate_wins": float(suite_result.suite_candidate_wins),
                    "benchmark_suite_candidate_losses": float(suite_result.suite_candidate_losses),
                    "benchmark_suite_draws": float(suite_result.suite_draws),
                    "benchmark_suite_avg_turns_per_game": float(suite_result.suite_avg_turns_per_game),
                    "benchmark_candidate_checkpoint": suite_result.candidate_checkpoint,
                }
            elif benchmark_suite:
                print(
                    f"cycle_benchmark_skip={cycle_idx}/{cycles} "
                    f"reason=interval "
                    f"every={int(save_checkpoint_every_cycles)} "
                    f"global_cycle={global_cycle_idx}"
                )

            if auto_promote and checkpoint_cycle:
                if checkpoint_info is None:
                    checkpoint_info = save_checkpoint(
                        model,
                        output_dir=checkpoint_dir,
                        run_id=run_id,
                        cycle_idx=global_cycle_idx,
                        metadata={
                            "seed": seed,
                            "collector_policy": collector_policy,
                            "mcts_sims": mcts_sims,
                            "cycle_idx": global_cycle_idx,
                            **resumed_from_metadata,
                        },
                    )
                    print(f"cycle_checkpoint={cycle_idx}/{cycles} path={checkpoint_info.path}")
                promo_suite_opponents, has_current_champion = _build_suite_opponents_from_registry(
                    champion_registry_path=champion_registry_path,
                    eval_mcts_config=promotion_eval_mcts_config,
                    device=device,
                    cycle_idx=cycle_idx,
                    cycles=cycles,
                )
                promo_candidate_policy = CheckpointMCTSOpponent(
                    checkpoint_path=str(checkpoint_info.path),
                    mcts_config=promotion_eval_mcts_config,
                    device=device,
                    name="candidate",
                )
                promo_suite = run_benchmark_suite(
                    candidate_checkpoint=str(checkpoint_info.path),
                    candidate_policy=promo_candidate_policy,
                    suite_opponents=promo_suite_opponents,
                    games_per_opponent=promotion_games,
                    max_turns=max_turns,
                    seed_base=effective_benchmark_seed + 10_000_000,
                    cycle_idx=cycle_idx,
                    parallel_workers=min(int(effective_benchmark_workers), int(promotion_games)),
                )
                should_promote, promote_reason, promote_details = _promotion_decision_from_suite(
                    promo_suite,
                    has_current_champion=has_current_champion,
                    promotion_threshold_winrate=promotion_threshold_winrate,
                    bootstrap_min_random_winrate=bootstrap_min_random_winrate,
                    bootstrap_min_heuristic_winrate=bootstrap_min_heuristic_winrate,
                )
                print(
                    f"cycle_promotion_eval={cycle_idx}/{cycles} "
                    f"games={promotion_games} "
                    f"threshold={promotion_threshold_winrate:.3f} "
                    f"has_current_champion={'true' if has_current_champion else 'false'}"
                )
                print(
                    f"cycle_promotion_result={cycle_idx}/{cycles} "
                    f"promoted={'true' if should_promote else 'false'} "
                    f"reason={promote_reason}"
                )
                for warning in promo_suite.warnings:
                    print(f"cycle_promotion_warning={cycle_idx}/{cycles} {warning}")

                if should_promote:
                    try:
                        registry = load_champion_registry(champion_registry_path)
                        promo_metrics = {
                            "promotion_games": int(promotion_games),
                            "promotion_threshold": float(promotion_threshold_winrate),
                            "promotion_reason": promote_reason,
                            "suite_candidate_wins": int(promo_suite.suite_candidate_wins),
                            "suite_candidate_losses": int(promo_suite.suite_candidate_losses),
                            "suite_draws": int(promo_suite.suite_draws),
                            "suite_avg_turns_per_game": float(promo_suite.suite_avg_turns_per_game),
                        }
                        for k, v in promote_details.items():
                            if v is not None:
                                promo_metrics[k] = v
                        append_accepted_champion(
                            registry,
                            build_champion_entry_from_promotion(
                                checkpoint_path=str(checkpoint_info.path),
                                run_id=run_id,
                                cycle_idx=global_cycle_idx,
                                metrics=promo_metrics,
                                notes="auto-promote",
                            ),
                        )
                        save_champion_registry(champion_registry_path, registry)
                        print(
                            f"cycle_promotion_registry_update={cycle_idx}/{cycles} "
                            f"checkpoint={checkpoint_info.path}"
                        )
                    except Exception as exc:
                        should_promote = False
                        promote_reason = f"registry_update_failed:{exc}"
                        print(f"cycle_promotion_warning={cycle_idx}/{cycles} {promote_reason}")

                last_promotion_metrics = {
                    "promotion_attempted": 1.0,
                    "promotion_promoted": 1.0 if should_promote else 0.0,
                    "promotion_reason": promote_reason,
                    "promotion_eval_games": float(promotion_games),
                    "promotion_threshold_winrate": float(promotion_threshold_winrate),
                    "promotion_checkpoint": str(checkpoint_info.path),
                }
                for k, v in promote_details.items():
                    if v is not None:
                        last_promotion_metrics[k] = v
            elif auto_promote:
                print(
                    f"cycle_promotion_skip={cycle_idx}/{cycles} "
                    f"reason=interval "
                    f"every={int(save_checkpoint_every_cycles)} "
                    f"global_cycle={global_cycle_idx}"
                )

            total_episodes += float(collection_metrics["episodes"])
            total_terminal_episodes += float(collection_metrics["terminal_episodes"])
            total_cutoff_episodes += float(collection_metrics["cutoff_episodes"])
            total_replay_samples += float(collection_metrics["replay_samples"])
            total_steps += float(collection_metrics["total_steps"])
            total_turns += float(collection_metrics["total_turns"])
            total_random_actions += float(collection_metrics["collector_random_actions"])
            total_model_actions += float(collection_metrics["collector_model_actions"])
            cycle_mcts_actions = float(collection_metrics["collector_mcts_actions"])
            total_mcts_actions += cycle_mcts_actions
            if cycle_mcts_actions > 0:
                weighted_sum_mcts_avg_search_ms += float(collection_metrics["mcts_avg_search_ms"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_entropy += float(collection_metrics["mcts_avg_root_entropy"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_top1_visit_prob += float(collection_metrics["mcts_avg_root_top1_visit_prob"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_selected_visit_prob += float(collection_metrics["mcts_avg_selected_visit_prob"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_value += float(collection_metrics["mcts_avg_root_value"]) * cycle_mcts_actions

            cycle_train_steps = float(train_metrics["train_steps"])
            total_train_steps += cycle_train_steps
            weighted_sum_avg_policy_loss += float(train_metrics["avg_policy_loss"]) * cycle_train_steps
            weighted_sum_avg_value_loss += float(train_metrics["avg_value_loss"]) * cycle_train_steps
            weighted_sum_avg_total_loss += float(train_metrics["avg_total_loss"]) * cycle_train_steps
            weighted_sum_avg_grad_norm += float(train_metrics["avg_grad_norm"]) * cycle_train_steps
            cycle_eval_samples = float(eval_metrics["eval_samples"])
            total_eval_samples += cycle_eval_samples
            weighted_sum_eval_policy_loss += float(eval_metrics["eval_policy_loss"]) * cycle_eval_samples
            weighted_sum_eval_value_loss += float(eval_metrics["eval_value_loss"]) * cycle_eval_samples
            weighted_sum_eval_total_loss += float(eval_metrics["eval_total_loss"]) * cycle_eval_samples

            should_save_replay = False
            if replay_save_every_cycles > 0 and (global_cycle_idx % replay_save_every_cycles == 0 or cycle_idx == cycles):
                should_save_replay = True
            if should_save_replay:
                replay_out_dir.mkdir(parents=True, exist_ok=True)
                if rolling_replay_state_path.exists():
                    rolling_replay_state_path.unlink()
                last_saved_replay_path = replay.save_npz(rolling_replay_state_path)
                print(
                    f"replay_state_saved_cycle={cycle_idx}/{cycles} "
                    f"path={last_saved_replay_path.resolve()} "
                    f"samples={len(replay)}"
                )

    if viz_logger is not None:
        viz_logger.finalize()

    if total_train_steps <= 0:
        raise RuntimeError("No training steps executed in cycle run")

    if replay_save_every_cycles > 0:
        if last_saved_replay_path is None:
            replay_out_dir.mkdir(parents=True, exist_ok=True)
            if rolling_replay_state_path.exists():
                rolling_replay_state_path.unlink()
            last_saved_replay_path = replay.save_npz(rolling_replay_state_path)
            print(f"replay_state_saved={last_saved_replay_path.resolve()} samples={len(replay)}")
        replay_state_path = last_saved_replay_path
    else:
        replay_out_dir.mkdir(parents=True, exist_ok=True)
        replay_file = replay_out_dir / f"{run_id}_cycle_{(resume_base_cycle_idx + cycles):04d}_replay.npz"
        replay_state_path = replay.save_npz(replay_file)
        print(f"replay_state_saved={replay_state_path.resolve()} samples={len(replay)}")

    result: dict[str, object] = {
        "mode": "cycles",
        "collector_policy": collector_policy,
        "collector_workers": float(requested_workers),
        "benchmark_workers": float(effective_benchmark_workers),
        "rolling_replay": float(1 if rolling_replay else 0),
        "replay_capacity": float(replay_capacity),
        "cycles": float(cycles),
        "episodes_per_cycle": float(episodes_per_cycle),
        "train_steps_per_cycle": float(train_steps_per_cycle),
        "episodes": total_episodes,
        "terminal_episodes": total_terminal_episodes,
        "cutoff_episodes": total_cutoff_episodes,
        "replay_samples_total": total_replay_samples,
        "total_steps": total_steps,
        "total_turns": total_turns,
        "avg_turns_per_episode": _avg_or_zero(total_turns, total_episodes),
        "avg_steps_per_episode": _avg_or_zero(total_steps, total_episodes),
        "avg_steps_per_turn": _avg_or_zero(total_steps, total_turns),
        "collector_random_actions": total_random_actions,
        "collector_model_actions": total_model_actions,
        "collector_mcts_actions": total_mcts_actions,
        "mcts_avg_search_ms": (weighted_sum_mcts_avg_search_ms / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_entropy": (weighted_sum_mcts_avg_root_entropy / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_top1_visit_prob": (weighted_sum_mcts_avg_root_top1_visit_prob / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_selected_visit_prob": (weighted_sum_mcts_avg_selected_visit_prob / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_value": (weighted_sum_mcts_avg_root_value / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "avg_policy_loss": weighted_sum_avg_policy_loss / total_train_steps,
        "avg_value_loss": weighted_sum_avg_value_loss / total_train_steps,
        "avg_total_loss": weighted_sum_avg_total_loss / total_train_steps,
        "avg_grad_norm": weighted_sum_avg_grad_norm / total_train_steps,
        "eval_avg_policy_loss": (weighted_sum_eval_policy_loss / total_eval_samples) if total_eval_samples > 0 else 0.0,
        "eval_avg_value_loss": (weighted_sum_eval_value_loss / total_eval_samples) if total_eval_samples > 0 else 0.0,
        "eval_avg_total_loss": (weighted_sum_eval_total_loss / total_eval_samples) if total_eval_samples > 0 else 0.0,
        "replay_state_path": str(replay_state_path.resolve()),
        "replay_state_samples": float(len(replay)),
        "replay_save_every_cycles": float(replay_save_every_cycles),
    }
    result.update(
        {
            "policy_loss": last_train_metrics.get("policy_loss"),
            "value_loss": last_train_metrics.get("value_loss"),
            "total_loss": last_train_metrics.get("total_loss"),
            "grad_norm": last_train_metrics.get("grad_norm"),
            "legal_target_ok": last_train_metrics.get("legal_target_ok"),
            "eval_policy_loss": last_eval_metrics.get("eval_policy_loss"),
            "eval_value_loss": last_eval_metrics.get("eval_value_loss"),
            "eval_total_loss": last_eval_metrics.get("eval_total_loss"),
            "eval_samples": last_eval_metrics.get("eval_samples"),
        }
    )
    if last_benchmark_metrics:
        result.update(last_benchmark_metrics)
    if resumed_from_metadata:
        result.update(resumed_from_metadata)
        result["resume_base_cycle_idx"] = float(resume_base_cycle_idx)
    if auto_promote and not last_promotion_metrics:
        last_promotion_metrics = {
            "promotion_attempted": 0.0,
            "promotion_promoted": 0.0,
            "promotion_reason": "not_attempted",
        }
    if last_promotion_metrics:
        result.update(last_promotion_metrics)
    return result


def run_checkpoint_benchmark(
    *,
    candidate_checkpoint: str,
    device: str = "cpu",
    max_turns: int = 80,
    benchmark_games_per_opponent: int = 40,
    benchmark_mcts_sims: int = 200,
    mcts_c_puct: float = 1.25,
    champion_registry_path: str = "nn_artifacts/champions.json",
    benchmark_seed: int = 42,
    benchmark_cycle_idx: int = 0,
    benchmark_workers: int = 1,
) -> dict[str, object]:
    if not str(candidate_checkpoint):
        raise ValueError("candidate_checkpoint is required")
    if benchmark_games_per_opponent <= 0:
        raise ValueError("benchmark_games_per_opponent must be positive")
    if benchmark_mcts_sims <= 0:
        raise ValueError("benchmark_mcts_sims must be positive")
    if benchmark_workers <= 0:
        raise ValueError("benchmark_workers must be positive")

    eval_mcts_config = MCTSConfig(
        num_simulations=int(benchmark_mcts_sims),
        c_puct=mcts_c_puct,
        temperature_moves=0,
        temperature=0.0,
        root_dirichlet_noise=False,
    )
    suite_opponents = [GreedyHeuristicOpponent(name="heuristic")]
    candidate_policy = CheckpointMCTSOpponent(
        checkpoint_path=str(candidate_checkpoint),
        mcts_config=eval_mcts_config,
        device=device,
        name="candidate",
    )
    suite_result = run_benchmark_suite(
        candidate_checkpoint=str(candidate_checkpoint),
        candidate_policy=candidate_policy,
        suite_opponents=suite_opponents,
        games_per_opponent=benchmark_games_per_opponent,
        max_turns=max_turns,
        seed_base=int(benchmark_seed),
        cycle_idx=int(benchmark_cycle_idx),
        parallel_workers=min(int(benchmark_workers), int(benchmark_games_per_opponent)),
    )
    _print_benchmark_suite(int(benchmark_cycle_idx), 1, suite_result)

    result: dict[str, object] = {
        "mode": "benchmark",
        "benchmark_candidate_checkpoint": suite_result.candidate_checkpoint,
        "benchmark_matchups": float(len(suite_result.matchups)),
        "benchmark_suite_candidate_wins": float(suite_result.suite_candidate_wins),
        "benchmark_suite_candidate_losses": float(suite_result.suite_candidate_losses),
        "benchmark_suite_draws": float(suite_result.suite_draws),
        "benchmark_suite_avg_turns_per_game": float(suite_result.suite_avg_turns_per_game),
        "benchmark_warnings": float(len(suite_result.warnings)),
        "benchmark_workers": float(min(int(benchmark_workers), int(benchmark_games_per_opponent))),
    }
    for opp_name in ("heuristic",):
        matchup = matchup_by_name(suite_result, opp_name)
        if matchup is None:
            continue
        prefix = f"benchmark_{opp_name}"
        result[f"{prefix}_games"] = float(matchup.games)
        result[f"{prefix}_wins"] = float(matchup.candidate_wins)
        result[f"{prefix}_losses"] = float(matchup.candidate_losses)
        result[f"{prefix}_draws"] = float(matchup.draws)
        result[f"{prefix}_win_rate"] = float(matchup.candidate_win_rate)
        result[f"{prefix}_nonloss_rate"] = float(matchup.candidate_nonloss_rate)
        result[f"{prefix}_avg_turns_per_game"] = float(matchup.avg_turns_per_game)
        result[f"{prefix}_cutoff_rate"] = float(matchup.cutoff_rate)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Splendor NN smoke pipeline (random self-play + one train step)")
    p.add_argument("--mode", type=str, choices=["smoke", "cycles", "benchmark"], default="smoke")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--collector-policy", type=str, choices=["random", "model-sample", "mcts"], default="random")
    p.add_argument("--train-steps", type=int, default=1)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--episodes-per-cycle", type=int, default=5)
    p.add_argument("--train-steps-per-cycle", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mcts-sims", type=int, default=64)
    p.add_argument("--mcts-c-puct", type=float, default=1.25)
    p.add_argument("--mcts-temperature-moves", type=int, default=10)
    p.add_argument("--mcts-temperature", type=float, default=1.0)
    p.add_argument("--mcts-root-dirichlet-noise", action="store_true")
    p.add_argument("--mcts-root-dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--mcts-root-dirichlet-alpha-total", type=float, default=10.0)
    p.add_argument("--candidate-checkpoint", type=str, default=None)
    p.add_argument("--save-checkpoint-every-cycles", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=str, default="nn_artifacts/checkpoints")
    p.add_argument("--benchmark-suite", action="store_true")
    p.add_argument("--benchmark-games-per-opponent", type=int, default=40)
    p.add_argument("--benchmark-mcts-sims", type=int, default=200)
    p.add_argument("--champion-registry-path", type=str, default="nn_artifacts/champions.json")
    p.add_argument("--benchmark-seed", type=int, default=42)
    p.add_argument("--benchmark-cycle-idx", type=int, default=0)
    p.add_argument("--resume-checkpoint", type=str, default=None)
    p.add_argument("--resume-run-id-suffix", type=str, default="resume")
    p.add_argument("--resume-replay-path", type=str, default=None)
    p.add_argument("--auto-promote", action="store_true")
    p.add_argument("--promotion-games", type=int, default=50)
    p.add_argument("--promotion-threshold-winrate", type=float, default=0.65)
    p.add_argument("--promotion-benchmark-mcts-sims", type=int, default=None)
    p.add_argument("--bootstrap-min-heuristic-winrate", type=float, default=0.40)
    p.add_argument("--bootstrap-min-random-winrate", type=float, default=0.75)
    p.add_argument("--rolling-replay", action="store_true")
    p.add_argument("--replay-capacity", type=int, default=50000)
    p.add_argument("--replay-save-every-cycles", type=int, default=0)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz-dir", type=str, default="nn_artifacts/viz")
    p.add_argument("--viz-run-name", type=str, default=None)
    p.add_argument("--viz-save-every-cycle", type=int, default=1)
    p.add_argument("--viz-smoke-enable", action="store_true")
    p.add_argument("--collector-workers", type=int, default=None)
    p.add_argument("--benchmark-workers", type=int, default=1)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "smoke":
        metrics = run_smoke(
            episodes=args.episodes,
            max_turns=args.max_turns,
            batch_size=args.batch_size,
            collector_policy=args.collector_policy,
            train_steps=args.train_steps,
            log_every=args.log_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            mcts_sims=args.mcts_sims,
            mcts_c_puct=args.mcts_c_puct,
            mcts_temperature_moves=args.mcts_temperature_moves,
            mcts_temperature=args.mcts_temperature,
            mcts_root_dirichlet_noise=args.mcts_root_dirichlet_noise,
            mcts_root_dirichlet_epsilon=args.mcts_root_dirichlet_epsilon,
            mcts_root_dirichlet_alpha_total=args.mcts_root_dirichlet_alpha_total,
            visualize=args.visualize,
            viz_dir=args.viz_dir,
            viz_run_name=args.viz_run_name,
            viz_save_every_cycle=args.viz_save_every_cycle,
            viz_smoke_enable=args.viz_smoke_enable,
        )
    elif args.mode == "cycles":
        metrics = run_cycles(
            cycles=args.cycles,
            episodes_per_cycle=args.episodes_per_cycle,
            train_steps_per_cycle=args.train_steps_per_cycle,
            max_turns=args.max_turns,
            batch_size=args.batch_size,
            collector_policy=args.collector_policy,
            log_every=args.log_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            mcts_sims=args.mcts_sims,
            mcts_c_puct=args.mcts_c_puct,
            mcts_temperature_moves=args.mcts_temperature_moves,
            mcts_temperature=args.mcts_temperature,
            mcts_root_dirichlet_noise=args.mcts_root_dirichlet_noise,
            mcts_root_dirichlet_epsilon=args.mcts_root_dirichlet_epsilon,
            mcts_root_dirichlet_alpha_total=args.mcts_root_dirichlet_alpha_total,
            save_checkpoint_every_cycles=args.save_checkpoint_every_cycles,
            checkpoint_dir=args.checkpoint_dir,
            benchmark_suite=args.benchmark_suite,
            benchmark_games_per_opponent=args.benchmark_games_per_opponent,
            benchmark_mcts_sims=args.benchmark_mcts_sims,
            champion_registry_path=args.champion_registry_path,
            benchmark_seed=args.benchmark_seed,
            resume_checkpoint=args.resume_checkpoint,
            resume_run_id_suffix=args.resume_run_id_suffix,
            resume_replay_path=args.resume_replay_path,
            auto_promote=args.auto_promote,
            promotion_games=args.promotion_games,
            promotion_threshold_winrate=args.promotion_threshold_winrate,
            promotion_benchmark_mcts_sims=args.promotion_benchmark_mcts_sims,
            bootstrap_min_heuristic_winrate=args.bootstrap_min_heuristic_winrate,
            bootstrap_min_random_winrate=args.bootstrap_min_random_winrate,
            rolling_replay=args.rolling_replay,
            replay_capacity=args.replay_capacity,
            replay_save_every_cycles=args.replay_save_every_cycles,
            visualize=args.visualize,
            viz_dir=args.viz_dir,
            viz_run_name=args.viz_run_name,
            collector_workers=args.collector_workers,
            benchmark_workers=args.benchmark_workers,
        )
    else:
        if not args.candidate_checkpoint:
            raise ValueError("--candidate-checkpoint is required for --mode benchmark")
        metrics = run_checkpoint_benchmark(
            candidate_checkpoint=args.candidate_checkpoint,
            device=args.device,
            max_turns=args.max_turns,
            benchmark_games_per_opponent=args.benchmark_games_per_opponent,
            benchmark_mcts_sims=args.benchmark_mcts_sims,
            mcts_c_puct=args.mcts_c_puct,
            champion_registry_path=args.champion_registry_path,
            benchmark_seed=int(args.seed if args.benchmark_seed is None else args.benchmark_seed),
            benchmark_cycle_idx=args.benchmark_cycle_idx,
            benchmark_workers=args.benchmark_workers,
        )
    print("Run complete")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    main()
