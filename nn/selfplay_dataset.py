from __future__ import annotations

import multiprocessing as mp
import os
import json
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoints import load_checkpoint
from .mcts import MCTSConfig, run_mcts
from .native_env import SplendorNativeEnv
from .state_schema import ACTION_DIM, STATE_DIM
from .value_targets import blend_root_and_outcome, winner_to_value_for_player

_WORKER_RUNTIME_CONFIGURED = False


@dataclass
class SelfPlayStep:
    state: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    value_target: float
    value_root: float
    value_root_best: float
    action_selected: int
    episode_idx: int
    step_idx: int
    turn_idx: int
    player_id: int
    winner: int
    reached_cutoff: bool
    current_player_id: int


@dataclass
class SelfPlaySession:
    session_id: str
    created_at: str
    metadata: dict[str, Any]
    steps: list[SelfPlayStep]

    @property
    def games(self) -> int:
        value = self.metadata.get("games", 0)
        return int(value)


def _resolve_playout_cap_settings(
    *,
    num_simulations: int,
    full_search_sims: int | None,
    fast_search_sims: int | None,
    full_search_prob: float | None,
) -> tuple[bool, int, int, float]:
    has_explicit_budgets = full_search_sims is not None or fast_search_sims is not None
    if has_explicit_budgets and (full_search_sims is None or fast_search_sims is None):
        raise ValueError("full_search_sims and fast_search_sims must be provided together")
    if full_search_prob is not None and not has_explicit_budgets:
        raise ValueError("full_search_prob requires explicit full_search_sims and fast_search_sims")
    playout_cap_randomization_enabled = (
        full_search_sims is not None or fast_search_sims is not None or full_search_prob is not None
    )
    resolved_full_search_sims = int(num_simulations if full_search_sims is None else full_search_sims)
    resolved_fast_search_sims = int(num_simulations if fast_search_sims is None else fast_search_sims)
    if resolved_full_search_sims <= 0:
        raise ValueError("full_search_sims must be positive")
    if resolved_fast_search_sims <= 0:
        raise ValueError("fast_search_sims must be positive")
    if playout_cap_randomization_enabled:
        resolved_full_search_prob = float(0.25 if full_search_prob is None else full_search_prob)
    else:
        # Legacy behavior always runs the full simulation budget every turn.
        resolved_full_search_prob = 1.0
    if not (0.0 <= resolved_full_search_prob <= 1.0):
        raise ValueError("full_search_prob must be in [0, 1]")
    return (
        bool(playout_cap_randomization_enabled),
        int(resolved_full_search_sims),
        int(resolved_fast_search_sims),
        float(resolved_full_search_prob),
    )


def run_selfplay_session(
    *,
    env: SplendorNativeEnv,
    model: Any,
    games: int,
    max_turns: int,
    num_simulations: int,
    seed_base: int,
    full_search_sims: int | None = None,
    fast_search_sims: int | None = None,
    full_search_prob: float | None = None,
    use_forced_playouts: bool = True,
    forced_playouts_k: float = 2.0,
) -> SelfPlaySession:
    if games <= 0:
        raise ValueError("games must be positive")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if forced_playouts_k <= 0.0:
        raise ValueError("forced_playouts_k must be positive")
    (
        playout_cap_randomization_enabled,
        resolved_full_search_sims,
        resolved_fast_search_sims,
        resolved_full_search_prob,
    ) = _resolve_playout_cap_settings(
        num_simulations=int(num_simulations),
        full_search_sims=full_search_sims,
        fast_search_sims=fast_search_sims,
        full_search_prob=full_search_prob,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    session_id = f"selfplay_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    full_search_config = MCTSConfig(
        num_simulations=int(resolved_full_search_sims),
        c_puct=1.25,
        temperature_moves=10,
        temperature=1.0,
        root_dirichlet_noise=bool(playout_cap_randomization_enabled),
        use_forced_playouts=bool(use_forced_playouts),
        forced_playouts_k=float(forced_playouts_k),
    )
    fast_search_config = MCTSConfig(
        num_simulations=int(resolved_fast_search_sims),
        c_puct=1.25,
        temperature_moves=10,
        temperature=1.0,
        root_dirichlet_noise=False,
        use_forced_playouts=False,
        forced_playouts_k=float(forced_playouts_k),
    )

    all_steps: list[SelfPlayStep] = []
    total_actions = 0
    full_search_actions = 0
    fast_search_actions = 0
    terminal_episodes = 0
    cutoff_episodes = 0
    total_turns = 0
    replay_games_added = 0
    mcts_sum_search_seconds = 0.0
    mcts_sum_root_entropy = 0.0
    mcts_sum_root_top1_prob = 0.0
    mcts_sum_selected_visit_prob = 0.0
    mcts_search_slots_requested = 0
    mcts_search_slots_evaluated = 0
    mcts_search_slots_drop_pending_eval = 0
    mcts_search_slots_drop_no_action = 0
    for episode_idx in range(games):
        seed = int(seed_base + episode_idx)
        rng = random.Random(seed)
        state = env.reset(seed=seed)
        episode_steps: list[SelfPlayStep] = []
        winner = -1
        reached_cutoff = False
        turns_taken = 0

        while turns_taken < max_turns:
            if state.is_terminal:
                winner = int(state.winner)
                break
            if state.state.shape != (STATE_DIM,):
                raise RuntimeError(f"Unexpected state shape {state.state.shape}")
            if state.mask.shape != (ACTION_DIM,):
                raise RuntimeError(f"Unexpected mask shape {state.mask.shape}")
            if not bool(state.mask.any()):
                raise RuntimeError("Encountered non-terminal state with no legal actions")

            player_id = int(env.current_player_id)
            is_full_search = True
            if playout_cap_randomization_enabled:
                is_full_search = bool(rng.random() < resolved_full_search_prob)
            active_config = full_search_config if is_full_search else fast_search_config
            total_actions += 1
            if is_full_search:
                full_search_actions += 1
            else:
                fast_search_actions += 1
            t0_mcts = time.perf_counter()
            mcts_result = run_mcts(
                env,
                model,
                state,
                turns_taken=int(turns_taken),
                device="cpu",
                config=active_config,
                rng=rng,
            )
            mcts_sum_search_seconds += time.perf_counter() - t0_mcts

            action = int(mcts_result.chosen_action_idx)
            policy = np.asarray(mcts_result.visit_probs, dtype=np.float32)
            if policy.shape != (ACTION_DIM,):
                raise RuntimeError(f"Unexpected visit_probs shape {policy.shape}")
            if not bool(state.mask[action]):
                raise RuntimeError(f"MCTS produced illegal action {action}")

            p_safe = np.maximum(policy, 1e-12)
            p_safe = p_safe / np.sum(p_safe)
            mcts_sum_root_entropy += float(-np.sum(p_safe * np.log(p_safe)))
            mcts_sum_root_top1_prob += float(np.max(policy))
            mcts_sum_selected_visit_prob += float(policy[action])
            mcts_search_slots_requested += int(mcts_result.search_slots_requested)
            mcts_search_slots_evaluated += int(mcts_result.search_slots_evaluated)
            mcts_search_slots_drop_pending_eval += int(getattr(mcts_result, "search_slots_drop_pending_eval", 0))
            mcts_search_slots_drop_no_action += int(getattr(mcts_result, "search_slots_drop_no_action", 0))

            if is_full_search:
                root_best_value = float(mcts_result.root_best_value)
                episode_steps.append(
                    SelfPlayStep(
                        state=state.state.copy(),
                        mask=state.mask.copy(),
                        policy=policy,
                        value_target=0.0,  # Filled after episode outcome is known.
                        value_root=root_best_value,
                        value_root_best=root_best_value,
                        action_selected=action,
                        episode_idx=int(episode_idx),
                        step_idx=len(episode_steps),
                        turn_idx=int(turns_taken),
                        player_id=player_id,
                        winner=-1,
                        reached_cutoff=False,
                        current_player_id=int(state.current_player_id),
                    )
                )

            prev_player_id = int(env.current_player_id)
            state = env.step(action)
            if int(env.current_player_id) != prev_player_id:
                turns_taken += 1
            if state.is_terminal:
                winner = int(state.winner)
                break
        else:
            reached_cutoff = True
            winner = -1

        if not reached_cutoff and not state.is_terminal:
            reached_cutoff = True
            winner = -1

        total_turns += int(turns_taken)
        if reached_cutoff:
            cutoff_episodes += 1
        else:
            terminal_episodes += 1
        if episode_steps:
            replay_games_added += 1

        for step in episode_steps:
            value_outcome = winner_to_value_for_player(winner, step.player_id)
            step.value_target = blend_root_and_outcome(step.value_root_best, value_outcome)
            step.winner = int(winner)
            step.reached_cutoff = bool(reached_cutoff)
            all_steps.append(step)

    metadata = {
        "session_id": session_id,
        "created_at": created_at,
        "games": int(games),
        "max_turns": int(max_turns),
        "num_simulations": int(num_simulations),
        "playout_cap_randomization_enabled": bool(playout_cap_randomization_enabled),
        "full_search_sims": int(resolved_full_search_sims),
        "fast_search_sims": int(resolved_fast_search_sims),
        "full_search_prob": float(resolved_full_search_prob),
        "total_actions": int(total_actions),
        "full_search_actions": int(full_search_actions),
        "fast_search_actions": int(fast_search_actions),
        "replay_steps": int(len(all_steps)),
        "replay_games_added": int(replay_games_added),
        "terminal_episodes": int(terminal_episodes),
        "cutoff_episodes": int(cutoff_episodes),
        "total_turns": int(total_turns),
        "mcts_sum_search_seconds": float(mcts_sum_search_seconds),
        "mcts_sum_root_entropy": float(mcts_sum_root_entropy),
        "mcts_sum_root_top1_prob": float(mcts_sum_root_top1_prob),
        "mcts_sum_selected_visit_prob": float(mcts_sum_selected_visit_prob),
        "mcts_search_slots_requested": int(mcts_search_slots_requested),
        "mcts_search_slots_evaluated": int(mcts_search_slots_evaluated),
        "mcts_search_slots_drop_pending_eval": int(mcts_search_slots_drop_pending_eval),
        "mcts_search_slots_drop_no_action": int(mcts_search_slots_drop_no_action),
        "use_forced_playouts": bool(use_forced_playouts),
        "k": float(forced_playouts_k),
        "forced_playouts_k": float(forced_playouts_k),
        "seed_base": int(seed_base),
    }
    return SelfPlaySession(
        session_id=session_id,
        created_at=created_at,
        metadata=metadata,
        steps=all_steps,
    )


def _pack_steps(steps: list[SelfPlayStep], *, episode_offset: int = 0) -> dict[str, Any]:
    n = len(steps)
    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    masks = np.zeros((n, ACTION_DIM), dtype=np.bool_)
    policies = np.zeros((n, ACTION_DIM), dtype=np.float32)
    value_target = np.zeros((n,), dtype=np.float32)
    value_root = np.zeros((n,), dtype=np.float32)
    value_root_best = np.zeros((n,), dtype=np.float32)
    action_selected = np.zeros((n,), dtype=np.int32)
    episode_idx = np.zeros((n,), dtype=np.int32)
    step_idx = np.zeros((n,), dtype=np.int32)
    turn_idx = np.zeros((n,), dtype=np.int32)
    player_id = np.zeros((n,), dtype=np.int32)
    winner = np.zeros((n,), dtype=np.int32)
    reached_cutoff = np.zeros((n,), dtype=np.bool_)
    current_player_id = np.zeros((n,), dtype=np.int32)
    ep_offset = int(episode_offset)

    for i, step in enumerate(steps):
        states[i] = step.state
        masks[i] = step.mask
        policies[i] = step.policy
        value_target[i] = float(step.value_target)
        value_root[i] = float(step.value_root)
        value_root_best[i] = float(step.value_root_best)
        action_selected[i] = int(step.action_selected)
        episode_idx[i] = int(step.episode_idx) + ep_offset
        step_idx[i] = int(step.step_idx)
        turn_idx[i] = int(step.turn_idx)
        player_id[i] = int(step.player_id)
        winner[i] = int(step.winner)
        reached_cutoff[i] = bool(step.reached_cutoff)
        current_player_id[i] = int(step.current_player_id)

    return {
        "state": states,
        "mask": masks,
        "policy": policies,
        "value_target": value_target,
        "value_root": value_root,
        "value_root_best": value_root_best,
        "action_selected": action_selected,
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "turn_idx": turn_idx,
        "player_id": player_id,
        "winner": winner,
        "reached_cutoff": reached_cutoff,
        "current_player_id": current_player_id,
    }


def _unpack_steps(payload: dict[str, Any]) -> list[SelfPlayStep]:
    states = np.asarray(payload["state"], dtype=np.float32)
    masks = np.asarray(payload["mask"], dtype=np.bool_)
    policies = np.asarray(payload["policy"], dtype=np.float32)
    value_target = np.asarray(payload["value_target"], dtype=np.float32)
    value_root = np.asarray(payload["value_root"], dtype=np.float32)
    value_root_best = np.asarray(payload.get("value_root_best", value_root), dtype=np.float32)
    action_selected = np.asarray(payload["action_selected"], dtype=np.int32)
    episode_idx = np.asarray(payload["episode_idx"], dtype=np.int32)
    step_idx = np.asarray(payload["step_idx"], dtype=np.int32)
    turn_idx = np.asarray(payload["turn_idx"], dtype=np.int32)
    player_id = np.asarray(payload["player_id"], dtype=np.int32)
    winner = np.asarray(payload["winner"], dtype=np.int32)
    reached_cutoff = np.asarray(payload["reached_cutoff"], dtype=np.bool_)
    current_player_id = np.asarray(payload["current_player_id"], dtype=np.int32)

    n = int(states.shape[0])
    if states.shape != (n, STATE_DIM):
        raise RuntimeError(f"Worker payload has invalid state shape: {states.shape}")
    if masks.shape != (n, ACTION_DIM):
        raise RuntimeError(f"Worker payload has invalid mask shape: {masks.shape}")
    if policies.shape != (n, ACTION_DIM):
        raise RuntimeError(f"Worker payload has invalid policy shape: {policies.shape}")
    for arr_name, arr in (
        ("value_target", value_target),
        ("value_root", value_root),
        ("value_root_best", value_root_best),
        ("action_selected", action_selected),
        ("episode_idx", episode_idx),
        ("step_idx", step_idx),
        ("turn_idx", turn_idx),
        ("player_id", player_id),
        ("winner", winner),
        ("reached_cutoff", reached_cutoff),
        ("current_player_id", current_player_id),
    ):
        if int(arr.shape[0]) != n:
            raise RuntimeError(f"Worker payload has mismatched {arr_name} length: {arr.shape}")

    steps: list[SelfPlayStep] = []
    for i in range(n):
        steps.append(
            SelfPlayStep(
                state=states[i],
                mask=masks[i],
                policy=policies[i],
                value_target=float(value_target[i]),
                value_root=float(value_root[i]),
                value_root_best=float(value_root_best[i]),
                action_selected=int(action_selected[i]),
                episode_idx=int(episode_idx[i]),
                step_idx=int(step_idx[i]),
                turn_idx=int(turn_idx[i]),
                player_id=int(player_id[i]),
                winner=int(winner[i]),
                reached_cutoff=bool(reached_cutoff[i]),
                current_player_id=int(current_player_id[i]),
            )
        )
    return steps


def _compute_games_per_worker(games: int, workers: int) -> list[int]:
    if games <= 0:
        raise ValueError("games must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    workers = min(int(workers), int(games))
    base, remainder = divmod(int(games), int(workers))
    return [base + (1 if idx < remainder else 0) for idx in range(workers)]


def _run_selfplay_worker_task(
    *,
    worker_idx: int,
    checkpoint_path: str,
    games_for_worker: int,
    episode_start_idx: int,
    max_turns: int,
    num_simulations: int,
    seed_base: int,
    full_search_sims: int | None = None,
    fast_search_sims: int | None = None,
    full_search_prob: float | None = None,
    use_forced_playouts: bool = True,
    forced_playouts_k: float = 2.0,
) -> dict[str, Any]:
    worker_t0 = time.perf_counter()
    # Avoid mutating parent-process BLAS/PyTorch thread settings when this code
    # is executed inline (workers_used == 1 fast path).
    if mp.current_process().name != "MainProcess":
        _configure_worker_runtime()
    model_t0 = time.perf_counter()
    model = load_checkpoint(checkpoint_path, device="cpu")
    model_elapsed = time.perf_counter() - model_t0
    selfplay_t0 = time.perf_counter()
    with SplendorNativeEnv() as env:
        session = run_selfplay_session(
            env=env,
            model=model,
            games=int(games_for_worker),
            max_turns=int(max_turns),
            num_simulations=int(num_simulations),
            seed_base=int(seed_base + episode_start_idx),
            full_search_sims=full_search_sims,
            fast_search_sims=fast_search_sims,
            full_search_prob=full_search_prob,
            use_forced_playouts=bool(use_forced_playouts),
            forced_playouts_k=float(forced_playouts_k),
        )
    selfplay_elapsed = time.perf_counter() - selfplay_t0
    pack_t0 = time.perf_counter()
    packed_steps = _pack_steps(session.steps, episode_offset=int(episode_start_idx))
    pack_elapsed = time.perf_counter() - pack_t0
    worker_elapsed = time.perf_counter() - worker_t0
    return {
        "worker_idx": int(worker_idx),
        "episode_start_idx": int(episode_start_idx),
        "games": int(games_for_worker),
        "steps_packed": packed_steps,
        "session_metadata": dict(session.metadata),
        "worker_timing": {
            "worker_model_load_sec": float(model_elapsed),
            "worker_selfplay_sec": float(selfplay_elapsed),
            "worker_pack_sec": float(pack_elapsed),
            "worker_total_sec": float(worker_elapsed),
        },
    }


def _configure_worker_runtime() -> None:
    global _WORKER_RUNTIME_CONFIGURED
    if _WORKER_RUNTIME_CONFIGURED:
        return
    # Keep each worker process single-threaded to avoid oversubscription.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    _WORKER_RUNTIME_CONFIGURED = True


class SelfPlayWorkerPool:
    def __init__(self, *, max_workers: int) -> None:
        if int(max_workers) <= 0:
            raise ValueError("max_workers must be positive")
        self._max_workers = int(max_workers)
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> SelfPlayWorkerPool:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.shutdown()
        return False

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=mp.get_context("spawn"),
            )
        return self._executor

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def run_session(
        self,
        *,
        checkpoint_path: str | Path,
        games: int,
        max_turns: int,
        num_simulations: int,
        seed_base: int,
        workers: int,
        full_search_sims: int | None = None,
        fast_search_sims: int | None = None,
        full_search_prob: float | None = None,
        use_forced_playouts: bool = True,
        forced_playouts_k: float = 2.0,
    ) -> SelfPlaySession:
        return _run_selfplay_session_parallel_impl(
            checkpoint_path=checkpoint_path,
            games=games,
            max_turns=max_turns,
            num_simulations=num_simulations,
            seed_base=seed_base,
            workers=workers,
            full_search_sims=full_search_sims,
            fast_search_sims=fast_search_sims,
            full_search_prob=full_search_prob,
            use_forced_playouts=use_forced_playouts,
            forced_playouts_k=forced_playouts_k,
            executor=self._ensure_executor(),
            max_workers=self._max_workers,
        )


def _run_selfplay_session_parallel_impl(
    *,
    checkpoint_path: str | Path,
    games: int,
    max_turns: int,
    num_simulations: int,
    seed_base: int,
    workers: int,
    full_search_sims: int | None = None,
    fast_search_sims: int | None = None,
    full_search_prob: float | None = None,
    use_forced_playouts: bool = True,
    forced_playouts_k: float = 2.0,
    executor: ProcessPoolExecutor | None = None,
    max_workers: int | None = None,
) -> SelfPlaySession:
    using_external_executor = executor is not None
    if games <= 0:
        raise ValueError("games must be positive")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if forced_playouts_k <= 0.0:
        raise ValueError("forced_playouts_k must be positive")
    (
        playout_cap_randomization_enabled,
        resolved_full_search_sims,
        resolved_fast_search_sims,
        resolved_full_search_prob,
    ) = _resolve_playout_cap_settings(
        num_simulations=int(num_simulations),
        full_search_sims=full_search_sims,
        fast_search_sims=fast_search_sims,
        full_search_prob=full_search_prob,
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    workers_cap = int(workers) if max_workers is None else int(max_workers)
    if workers_cap <= 0:
        raise ValueError("max_workers must be positive")
    workers_used = min(int(workers), int(games), workers_cap)
    games_per_worker = _compute_games_per_worker(int(games), workers_used)

    work_items: list[tuple[int, int, int]] = []
    episode_start = 0
    for worker_idx, games_for_worker in enumerate(games_per_worker):
        if games_for_worker <= 0:
            continue
        work_items.append((worker_idx, episode_start, games_for_worker))
        episode_start += games_for_worker

    created_at = datetime.now(timezone.utc).isoformat()
    session_id = f"selfplay_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    by_worker_idx: dict[int, dict[str, Any]] = {}
    if workers_used == 1:
        worker_idx, episode_start_idx, games_for_worker = work_items[0]
        by_worker_idx[worker_idx] = _run_selfplay_worker_task(
            worker_idx=worker_idx,
            checkpoint_path=str(ckpt_path),
            games_for_worker=games_for_worker,
            episode_start_idx=episode_start_idx,
            max_turns=int(max_turns),
            num_simulations=int(num_simulations),
            seed_base=int(seed_base),
            full_search_sims=full_search_sims,
            fast_search_sims=fast_search_sims,
            full_search_prob=full_search_prob,
            use_forced_playouts=bool(use_forced_playouts),
            forced_playouts_k=float(forced_playouts_k),
        )
    elif not using_external_executor:
        futures = {}
        with ProcessPoolExecutor(
            max_workers=workers_used,
            mp_context=mp.get_context("spawn"),
        ) as local_executor:
            for worker_idx, episode_start_idx, games_for_worker in work_items:
                fut = local_executor.submit(
                    _run_selfplay_worker_task,
                    worker_idx=worker_idx,
                    checkpoint_path=str(ckpt_path),
                    games_for_worker=games_for_worker,
                    episode_start_idx=episode_start_idx,
                    max_turns=int(max_turns),
                    num_simulations=int(num_simulations),
                    seed_base=int(seed_base),
                    full_search_sims=full_search_sims,
                    fast_search_sims=fast_search_sims,
                    full_search_prob=full_search_prob,
                    use_forced_playouts=bool(use_forced_playouts),
                    forced_playouts_k=float(forced_playouts_k),
                )
                futures[fut] = worker_idx
            for fut in as_completed(futures):
                worker_idx = futures[fut]
                try:
                    by_worker_idx[worker_idx] = fut.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"self-play worker {worker_idx} failed") from exc
    else:
        futures = {}
        for worker_idx, episode_start_idx, games_for_worker in work_items:
            fut = executor.submit(
                _run_selfplay_worker_task,
                worker_idx=worker_idx,
                checkpoint_path=str(ckpt_path),
                games_for_worker=games_for_worker,
                episode_start_idx=episode_start_idx,
                max_turns=int(max_turns),
                num_simulations=int(num_simulations),
                seed_base=int(seed_base),
                full_search_sims=full_search_sims,
                fast_search_sims=fast_search_sims,
                full_search_prob=full_search_prob,
                use_forced_playouts=bool(use_forced_playouts),
                forced_playouts_k=float(forced_playouts_k),
            )
            futures[fut] = worker_idx
        for fut in as_completed(futures):
            worker_idx = futures[fut]
            try:
                by_worker_idx[worker_idx] = fut.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"self-play worker {worker_idx} failed") from exc

    all_steps: list[SelfPlayStep] = []
    worker_timing_rows: list[dict[str, float]] = []
    for worker_idx in sorted(by_worker_idx):
        payload = by_worker_idx[worker_idx]
        timing_payload = payload.get("worker_timing")
        if isinstance(timing_payload, dict):
            worker_timing_rows.append(
                {
                    "worker_model_load_sec": float(timing_payload.get("worker_model_load_sec", 0.0)),
                    "worker_selfplay_sec": float(timing_payload.get("worker_selfplay_sec", 0.0)),
                    "worker_pack_sec": float(timing_payload.get("worker_pack_sec", 0.0)),
                    "worker_total_sec": float(timing_payload.get("worker_total_sec", 0.0)),
                }
            )
        packed_steps = payload.get("steps_packed")
        if packed_steps is not None:
            steps = _unpack_steps(packed_steps)
        else:
            # Backward compatibility for tests/mocks returning raw SelfPlayStep lists.
            steps = payload.get("steps") or []
        all_steps.extend(steps)
    all_steps.sort(key=lambda step: (int(step.episode_idx), int(step.step_idx)))

    def _series_stats(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        return (
            float(sum(values) / len(values)),
            float(min(values)),
            float(max(values)),
        )

    model_values = [float(row["worker_model_load_sec"]) for row in worker_timing_rows]
    selfplay_values = [float(row["worker_selfplay_sec"]) for row in worker_timing_rows]
    pack_values = [float(row["worker_pack_sec"]) for row in worker_timing_rows]
    total_values = [float(row["worker_total_sec"]) for row in worker_timing_rows]
    model_mean, model_min, model_max = _series_stats(model_values)
    selfplay_mean, selfplay_min, selfplay_max = _series_stats(selfplay_values)
    pack_mean, pack_min, pack_max = _series_stats(pack_values)
    total_mean, total_min, total_max = _series_stats(total_values)
    total_actions = 0
    full_search_actions = 0
    fast_search_actions = 0
    replay_steps = 0
    replay_games_added = 0
    terminal_episodes = 0
    cutoff_episodes = 0
    total_turns = 0
    mcts_sum_search_seconds = 0.0
    mcts_sum_root_entropy = 0.0
    mcts_sum_root_top1_prob = 0.0
    mcts_sum_selected_visit_prob = 0.0
    mcts_search_slots_requested = 0
    mcts_search_slots_evaluated = 0
    mcts_search_slots_drop_pending_eval = 0
    mcts_search_slots_drop_no_action = 0
    for payload in by_worker_idx.values():
        session_metadata = payload.get("session_metadata")
        if not isinstance(session_metadata, dict):
            continue
        total_actions += int(session_metadata.get("total_actions", 0))
        full_search_actions += int(session_metadata.get("full_search_actions", 0))
        fast_search_actions += int(session_metadata.get("fast_search_actions", 0))
        replay_steps += int(session_metadata.get("replay_steps", 0))
        replay_games_added += int(session_metadata.get("replay_games_added", 0))
        terminal_episodes += int(session_metadata.get("terminal_episodes", 0))
        cutoff_episodes += int(session_metadata.get("cutoff_episodes", 0))
        total_turns += int(session_metadata.get("total_turns", 0))
        mcts_sum_search_seconds += float(session_metadata.get("mcts_sum_search_seconds", 0.0))
        mcts_sum_root_entropy += float(session_metadata.get("mcts_sum_root_entropy", 0.0))
        mcts_sum_root_top1_prob += float(session_metadata.get("mcts_sum_root_top1_prob", 0.0))
        mcts_sum_selected_visit_prob += float(session_metadata.get("mcts_sum_selected_visit_prob", 0.0))
        mcts_search_slots_requested += int(session_metadata.get("mcts_search_slots_requested", 0))
        mcts_search_slots_evaluated += int(session_metadata.get("mcts_search_slots_evaluated", 0))
        mcts_search_slots_drop_pending_eval += int(session_metadata.get("mcts_search_slots_drop_pending_eval", 0))
        mcts_search_slots_drop_no_action += int(session_metadata.get("mcts_search_slots_drop_no_action", 0))

    metadata = {
        "session_id": session_id,
        "created_at": created_at,
        "games": int(games),
        "max_turns": int(max_turns),
        "num_simulations": int(num_simulations),
        "playout_cap_randomization_enabled": bool(playout_cap_randomization_enabled),
        "full_search_sims": int(resolved_full_search_sims),
        "fast_search_sims": int(resolved_fast_search_sims),
        "full_search_prob": float(resolved_full_search_prob),
        "total_actions": int(total_actions),
        "full_search_actions": int(full_search_actions),
        "fast_search_actions": int(fast_search_actions),
        "replay_steps": int(replay_steps),
        "replay_games_added": int(replay_games_added),
        "terminal_episodes": int(terminal_episodes),
        "cutoff_episodes": int(cutoff_episodes),
        "total_turns": int(total_turns),
        "mcts_sum_search_seconds": float(mcts_sum_search_seconds),
        "mcts_sum_root_entropy": float(mcts_sum_root_entropy),
        "mcts_sum_root_top1_prob": float(mcts_sum_root_top1_prob),
        "mcts_sum_selected_visit_prob": float(mcts_sum_selected_visit_prob),
        "mcts_search_slots_requested": int(mcts_search_slots_requested),
        "mcts_search_slots_evaluated": int(mcts_search_slots_evaluated),
        "mcts_search_slots_drop_pending_eval": int(mcts_search_slots_drop_pending_eval),
        "mcts_search_slots_drop_no_action": int(mcts_search_slots_drop_no_action),
        "use_forced_playouts": bool(use_forced_playouts),
        "forced_playouts_k": float(forced_playouts_k),
        "seed_base": int(seed_base),
        "workers_requested": int(workers),
        "workers_used": int(workers_used),
        "parallelism_mode": "process_pool_persistent" if using_external_executor else "process_pool",
        "games_per_worker": [int(x) for x in games_per_worker],
        "worker_timing_count": int(len(worker_timing_rows)),
        "worker_model_load_sec_mean": float(model_mean),
        "worker_model_load_sec_min": float(model_min),
        "worker_model_load_sec_max": float(model_max),
        "worker_selfplay_sec_mean": float(selfplay_mean),
        "worker_selfplay_sec_min": float(selfplay_min),
        "worker_selfplay_sec_max": float(selfplay_max),
        "worker_pack_sec_mean": float(pack_mean),
        "worker_pack_sec_min": float(pack_min),
        "worker_pack_sec_max": float(pack_max),
        "worker_total_sec_mean": float(total_mean),
        "worker_total_sec_min": float(total_min),
        "worker_total_sec_max": float(total_max),
    }
    return SelfPlaySession(
        session_id=session_id,
        created_at=created_at,
        metadata=metadata,
        steps=all_steps,
    )


def run_selfplay_session_parallel(
    *,
    checkpoint_path: str | Path,
    games: int,
    max_turns: int,
    num_simulations: int,
    seed_base: int,
    workers: int,
    full_search_sims: int | None = None,
    fast_search_sims: int | None = None,
    full_search_prob: float | None = None,
    use_forced_playouts: bool = True,
    forced_playouts_k: float = 2.0,
) -> SelfPlaySession:
    return _run_selfplay_session_parallel_impl(
        checkpoint_path=checkpoint_path,
        games=games,
        max_turns=max_turns,
        num_simulations=num_simulations,
        seed_base=seed_base,
        workers=workers,
        full_search_sims=full_search_sims,
        fast_search_sims=fast_search_sims,
        full_search_prob=full_search_prob,
        use_forced_playouts=use_forced_playouts,
        forced_playouts_k=forced_playouts_k,
    )


def save_session_npz(session: SelfPlaySession, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{session.session_id}.npz"

    n = len(session.steps)
    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    masks = np.zeros((n, ACTION_DIM), dtype=np.bool_)
    policies = np.zeros((n, ACTION_DIM), dtype=np.float32)
    value_target = np.zeros((n,), dtype=np.float32)
    value_root = np.zeros((n,), dtype=np.float32)
    value_root_best = np.zeros((n,), dtype=np.float32)
    action_selected = np.zeros((n,), dtype=np.int32)
    episode_idx = np.zeros((n,), dtype=np.int32)
    step_idx = np.zeros((n,), dtype=np.int32)
    turn_idx = np.zeros((n,), dtype=np.int32)
    player_id = np.zeros((n,), dtype=np.int32)
    winner = np.zeros((n,), dtype=np.int32)
    reached_cutoff = np.zeros((n,), dtype=np.bool_)
    current_player_id = np.zeros((n,), dtype=np.int32)

    for i, step in enumerate(session.steps):
        states[i] = step.state
        masks[i] = step.mask
        policies[i] = step.policy
        value_target[i] = float(step.value_target)
        value_root[i] = float(step.value_root)
        value_root_best[i] = float(step.value_root_best)
        action_selected[i] = int(step.action_selected)
        episode_idx[i] = int(step.episode_idx)
        step_idx[i] = int(step.step_idx)
        turn_idx[i] = int(step.turn_idx)
        player_id[i] = int(step.player_id)
        winner[i] = int(step.winner)
        reached_cutoff[i] = bool(step.reached_cutoff)
        current_player_id[i] = int(step.current_player_id)

    metadata = dict(session.metadata)
    metadata["session_id"] = session.session_id
    metadata["created_at"] = session.created_at

    np.savez_compressed(
        file_path,
        metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
        state=states,
        mask=masks,
        policy=policies,
        value_target=value_target,
        value_root=value_root,
        value_root_best=value_root_best,
        action_selected=action_selected,
        episode_idx=episode_idx,
        step_idx=step_idx,
        turn_idx=turn_idx,
        player_id=player_id,
        winner=winner,
        reached_cutoff=reached_cutoff,
        current_player_id=current_player_id,
    )
    return file_path


def load_session_npz(path: str | Path) -> SelfPlaySession:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Session file not found: {data_path}")
    with np.load(data_path, allow_pickle=False) as npz:
        metadata_raw = npz["metadata_json"]
        if metadata_raw.ndim == 0:
            metadata_json = str(metadata_raw.item())
        else:
            metadata_json = str(metadata_raw.tolist())
        metadata = json.loads(metadata_json)

        states = np.asarray(npz["state"], dtype=np.float32)
        masks = np.asarray(npz["mask"], dtype=np.bool_)
        policies = np.asarray(npz["policy"], dtype=np.float32)
        value_target = np.asarray(npz["value_target"], dtype=np.float32)
        value_root = np.asarray(npz["value_root"], dtype=np.float32)
        value_root_best = np.asarray(npz["value_root_best"], dtype=np.float32) if "value_root_best" in npz else value_root
        action_selected = np.asarray(npz["action_selected"], dtype=np.int32)
        episode_idx = np.asarray(npz["episode_idx"], dtype=np.int32)
        step_idx = np.asarray(npz["step_idx"], dtype=np.int32)
        turn_idx = np.asarray(npz["turn_idx"], dtype=np.int32)
        player_id = np.asarray(npz["player_id"], dtype=np.int32)
        winner = np.asarray(npz["winner"], dtype=np.int32)
        reached_cutoff = np.asarray(npz["reached_cutoff"], dtype=np.bool_)
        current_player_id = np.asarray(npz["current_player_id"], dtype=np.int32)

    n = int(states.shape[0])
    steps: list[SelfPlayStep] = []
    for i in range(n):
        steps.append(
            SelfPlayStep(
                state=states[i].copy(),
                mask=masks[i].copy(),
                policy=policies[i].copy(),
                value_target=float(value_target[i]),
                value_root=float(value_root[i]),
                value_root_best=float(value_root_best[i]),
                action_selected=int(action_selected[i]),
                episode_idx=int(episode_idx[i]),
                step_idx=int(step_idx[i]),
                turn_idx=int(turn_idx[i]),
                player_id=int(player_id[i]),
                winner=int(winner[i]),
                reached_cutoff=bool(reached_cutoff[i]),
                current_player_id=int(current_player_id[i]),
            )
        )

    created_at = str(metadata.get("created_at", ""))
    session_id = str(metadata.get("session_id", data_path.stem))
    return SelfPlaySession(
        session_id=session_id,
        created_at=created_at,
        metadata=metadata,
        steps=steps,
    )


def list_sessions(out_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for path in root.glob("*.npz"):
        try:
            session = load_session_npz(path)
        except Exception:
            continue
        by_episode: dict[int, int] = {}
        for step in session.steps:
            by_episode[step.episode_idx] = by_episode.get(step.episode_idx, 0) + 1
        checkpoint_path = str(session.metadata.get("checkpoint_path", ""))
        checkpoint_name = Path(checkpoint_path).name if checkpoint_path else "unknown.pt"
        seed_base = int(session.metadata.get("seed_base", 0))
        sims = int(session.metadata.get("num_simulations", 0))
        games = int(session.metadata.get("games", 0))
        display_name = f"{checkpoint_name}_seed{seed_base}_sims{sims}_games{games}"
        items.append(
            {
                "session_id": session.session_id,
                "display_name": display_name,
                "path": str(path.resolve()),
                "created_at": session.created_at,
                "games": int(session.metadata.get("games", 0)),
                "steps": int(len(session.steps)),
                "steps_per_episode": {str(k): int(v) for k, v in sorted(by_episode.items())},
                "metadata": session.metadata,
            }
        )
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return items
