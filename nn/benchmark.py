from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .checkpoints import load_model_from_spec
from .native_env import SplendorNativeEnv
from .opponents import CheckpointMCTSOpponent, GreedyHeuristicOpponent, ModelMCTSOpponent, RandomOpponent

_POLICY_CACHE_MAX_ENTRIES = 16
_POLICY_SPEC_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


@dataclass
class GameResult:
    winner: int  # absolute seat winner, -1 draw
    num_turns: int
    reached_cutoff: bool
    candidate_seat: int


@dataclass
class MatchupResult:
    opponent_name: str
    games: int
    candidate_wins: int
    candidate_losses: int
    draws: int
    candidate_win_rate: float
    candidate_nonloss_rate: float
    seat0_games: int
    seat1_games: int
    avg_turns_per_game: float
    avg_turns_candidate_wins: float | None
    avg_turns_candidate_losses: float | None
    avg_turns_draws: float | None
    cutoff_rate: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuiteResult:
    candidate_checkpoint: str
    matchups: list[MatchupResult]
    suite_candidate_wins: int
    suite_candidate_losses: int
    suite_draws: int
    suite_avg_turns_per_game: float
    warnings: list[str] = field(default_factory=list)


def _stable_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _safe_avg(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _recommend_benchmark_workers(*, requested_workers: int, games: int, max_workers: int | None = None) -> int:
    workers_cap = int(requested_workers) if max_workers is None else int(max_workers)
    workers = max(1, min(int(requested_workers), int(games), int(workers_cap)))
    if workers <= 1:
        return 1
    # Avoid over-partitioning tiny batches where IPC/spawn dominates useful work.
    if int(games) < workers * 2:
        workers = max(1, int(games) // 2)
    return max(1, min(int(workers), int(games)))


def play_game(
    env: SplendorNativeEnv,
    candidate_policy: Any,
    opponent_policy: Any,
    *,
    seed: int,
    candidate_seat: int,
    max_turns: int,
    rng: random.Random,
) -> GameResult:
    state = env.reset(seed=int(seed))
    turns_taken = 0
    reached_cutoff = False

    while turns_taken < max_turns:
        if state.is_terminal:
            return GameResult(
                winner=int(state.winner),
                num_turns=int(turns_taken),
                reached_cutoff=False,
                candidate_seat=int(candidate_seat),
            )
        actor = candidate_policy if env.current_player_id == candidate_seat else opponent_policy
        action = int(actor.select_action(env, state, turns_taken=turns_taken, rng=rng))
        if not (0 <= action < len(state.mask)) or not bool(state.mask[action]):
            raise RuntimeError(f"Benchmark actor selected illegal action {action}")
        prev_player_id = env.current_player_id
        state = env.step(action)
        if env.current_player_id != prev_player_id:
            turns_taken += 1

    reached_cutoff = True
    if state.is_terminal:
        return GameResult(
            winner=int(state.winner),
            num_turns=int(turns_taken),
            reached_cutoff=False,
            candidate_seat=int(candidate_seat),
        )
    return GameResult(winner=-1, num_turns=int(turns_taken), reached_cutoff=reached_cutoff, candidate_seat=int(candidate_seat))


def run_matchup(
    env: SplendorNativeEnv,
    candidate_policy: Any,
    opponent_policy: Any,
    *,
    games: int,
    max_turns: int,
    seed_base: int,
    cycle_idx: int,
    parallel_workers: int = 1,
    executor: ProcessPoolExecutor | None = None,
    max_workers: int | None = None,
) -> MatchupResult:
    if games <= 0:
        raise ValueError("games must be positive")
    seat0_games = games // 2
    seat1_games = games - seat0_games

    candidate_wins = 0
    candidate_losses = 0
    draws = 0
    cutoffs = 0
    turns_all: list[int] = []
    turns_wins: list[int] = []
    turns_losses: list[int] = []
    turns_draws: list[int] = []

    workers_used = _recommend_benchmark_workers(
        requested_workers=int(parallel_workers),
        games=int(games),
        max_workers=(None if max_workers is None else int(max_workers)),
    )
    if workers_used == 1:
        for game_idx in range(games):
            candidate_seat = 0 if game_idx < seat0_games else 1
            game_seed = _stable_seed(seed_base, cycle_idx, opponent_policy.name, game_idx, candidate_seat)
            game_rng = random.Random(game_seed)
            result = play_game(
                env,
                candidate_policy,
                opponent_policy,
                seed=game_seed,
                candidate_seat=candidate_seat,
                max_turns=max_turns,
                rng=game_rng,
            )
            turns_all.append(int(result.num_turns))
            if result.reached_cutoff:
                cutoffs += 1

            if result.winner == -1:
                draws += 1
                turns_draws.append(int(result.num_turns))
            elif result.winner == candidate_seat:
                candidate_wins += 1
                turns_wins.append(int(result.num_turns))
            else:
                candidate_losses += 1
                turns_losses.append(int(result.num_turns))
    else:
        candidate_spec = _policy_to_spec(candidate_policy)
        opponent_spec = _policy_to_spec(opponent_policy)
        spans: list[tuple[int, int]] = []
        base, rem = divmod(games, workers_used)
        start = 0
        for i in range(workers_used):
            n = base + (1 if i < rem else 0)
            end = start + n
            spans.append((start, end))
            start = end

        if executor is None:
            with ProcessPoolExecutor(max_workers=workers_used, mp_context=mp.get_context("spawn")) as ex:
                futures = [
                    ex.submit(
                        _run_matchup_worker_span,
                        candidate_spec=candidate_spec,
                        opponent_spec=opponent_spec,
                        game_start=s,
                        game_end=e,
                        seat0_games=seat0_games,
                        max_turns=max_turns,
                        seed_base=seed_base,
                        cycle_idx=cycle_idx,
                    )
                    for s, e in spans
                ]
                for fut in as_completed(futures):
                    part = fut.result()
                    candidate_wins += int(part["candidate_wins"])
                    candidate_losses += int(part["candidate_losses"])
                    draws += int(part["draws"])
                    cutoffs += int(part["cutoffs"])
                    turns_all.extend(int(x) for x in part["turns_all"])
                    turns_wins.extend(int(x) for x in part["turns_wins"])
                    turns_losses.extend(int(x) for x in part["turns_losses"])
                    turns_draws.extend(int(x) for x in part["turns_draws"])
        else:
            futures = [
                executor.submit(
                    _run_matchup_worker_span,
                    candidate_spec=candidate_spec,
                    opponent_spec=opponent_spec,
                    game_start=s,
                    game_end=e,
                    seat0_games=seat0_games,
                    max_turns=max_turns,
                    seed_base=seed_base,
                    cycle_idx=cycle_idx,
                )
                for s, e in spans
            ]
            for fut in as_completed(futures):
                part = fut.result()
                candidate_wins += int(part["candidate_wins"])
                candidate_losses += int(part["candidate_losses"])
                draws += int(part["draws"])
                cutoffs += int(part["cutoffs"])
                turns_all.extend(int(x) for x in part["turns_all"])
                turns_wins.extend(int(x) for x in part["turns_wins"])
                turns_losses.extend(int(x) for x in part["turns_losses"])
                turns_draws.extend(int(x) for x in part["turns_draws"])

    return MatchupResult(
        opponent_name=str(opponent_policy.name),
        games=int(games),
        candidate_wins=int(candidate_wins),
        candidate_losses=int(candidate_losses),
        draws=int(draws),
        candidate_win_rate=float(candidate_wins / games),
        candidate_nonloss_rate=float((candidate_wins + draws) / games),
        seat0_games=int(seat0_games),
        seat1_games=int(seat1_games),
        avg_turns_per_game=float(sum(turns_all) / len(turns_all)),
        avg_turns_candidate_wins=_safe_avg(turns_wins),
        avg_turns_candidate_losses=_safe_avg(turns_losses),
        avg_turns_draws=_safe_avg(turns_draws),
        cutoff_rate=float(cutoffs / games),
    )


def run_benchmark_suite(
    *,
    candidate_checkpoint: str,
    candidate_policy: Any,
    suite_opponents: list[Any],
    games_per_opponent: int = 20,
    max_turns: int = 80,
    seed_base: int = 0,
    cycle_idx: int = 0,
    parallel_workers: int = 1,
    executor: ProcessPoolExecutor | None = None,
    max_workers: int | None = None,
) -> BenchmarkSuiteResult:
    matchups: list[MatchupResult] = []
    warnings: list[str] = []

    with SplendorNativeEnv() as env:
        for opp in suite_opponents:
            try:
                matchup = run_matchup(
                    env,
                    candidate_policy,
                    opp,
                    games=games_per_opponent,
                    max_turns=max_turns,
                    seed_base=seed_base,
                    cycle_idx=cycle_idx,
                    parallel_workers=parallel_workers,
                    executor=executor,
                    max_workers=max_workers,
                )
                matchups.append(matchup)
            except Exception as exc:
                warnings.append(f"benchmark opponent={getattr(opp, 'name', 'unknown')} failed: {exc}")

    total_games = sum(m.games for m in matchups)
    total_wins = sum(m.candidate_wins for m in matchups)
    total_losses = sum(m.candidate_losses for m in matchups)
    total_draws = sum(m.draws for m in matchups)
    suite_avg_turns = 0.0
    if total_games > 0:
        suite_avg_turns = float(sum(m.avg_turns_per_game * m.games for m in matchups) / total_games)

    return BenchmarkSuiteResult(
        candidate_checkpoint=str(candidate_checkpoint),
        matchups=matchups,
        suite_candidate_wins=int(total_wins),
        suite_candidate_losses=int(total_losses),
        suite_draws=int(total_draws),
        suite_avg_turns_per_game=float(suite_avg_turns),
        warnings=warnings,
    )


def matchup_by_name(suite_result: BenchmarkSuiteResult, opponent_name: str) -> MatchupResult | None:
    for matchup in suite_result.matchups:
        if matchup.opponent_name == opponent_name:
            return matchup
    return None


def _policy_to_spec(policy: Any) -> dict[str, Any]:
    if isinstance(policy, RandomOpponent):
        return {"kind": "random", "name": str(policy.name)}
    if isinstance(policy, GreedyHeuristicOpponent):
        return {"kind": "heuristic", "name": str(policy.name)}
    if isinstance(policy, ModelMCTSOpponent):
        model = policy.model
        if not hasattr(model, "state_dict"):
            raise TypeError(f"ModelMCTSOpponent model does not expose state_dict(): {type(model)}")
        if not hasattr(model, "export_model_kwargs"):
            raise TypeError(f"ModelMCTSOpponent model does not expose export_model_kwargs(): {type(model)}")
        compat_adapter = getattr(model, "compat_adapter", None)
        base_model = getattr(model, "base_model", None)
        raw_state_dict = base_model.state_dict() if base_model is not None else model.state_dict()
        # Convert params to CPU tensors for process-safe transport.
        model_state_dict = {}
        for key, tensor in raw_state_dict.items():
            model_state_dict[str(key)] = tensor.detach().cpu()
        spec = {
            "kind": "model_mcts",
            "name": str(policy.name),
            "model_state_dict": model_state_dict,
            "model_state_digest": _state_dict_digest(model_state_dict),
            "model_kwargs": dict(model.export_model_kwargs()),
            "mcts_config": policy.mcts_config,
            "device": str(policy.device),
        }
        if compat_adapter is not None:
            spec["compat_adapter"] = str(compat_adapter)
        return spec
    if isinstance(policy, CheckpointMCTSOpponent):
        return {
            "kind": "checkpoint_mcts",
            "name": str(policy.name),
            "checkpoint_path": str(policy.checkpoint_path),
            "mcts_config": policy.mcts_config,
            "device": str(policy.device),
        }
    raise TypeError(f"Unsupported policy type for parallel benchmark: {type(policy)}")


def _policy_from_spec(spec: dict[str, Any]) -> Any:
    kind = str(spec.get("kind", ""))
    if kind == "random":
        return RandomOpponent(name=str(spec.get("name", "random")))
    if kind == "heuristic":
        return GreedyHeuristicOpponent(name=str(spec.get("name", "heuristic")))
    if kind == "model_mcts":
        model_kwargs = dict(spec.get("model_kwargs") or {})
        state_dict = spec.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("model_mcts spec missing model_state_dict")
        model = load_model_from_spec(
            model_kwargs=model_kwargs,
            state_dict=state_dict,
            device=str(spec.get("device", "cpu")),
            compat_adapter=(None if spec.get("compat_adapter") is None else str(spec.get("compat_adapter"))),
        )
        return ModelMCTSOpponent(
            model=model,
            mcts_config=spec.get("mcts_config"),
            device=str(spec.get("device", "cpu")),
            name=str(spec.get("name", "mcts_model")),
        )
    if kind == "checkpoint_mcts":
        return CheckpointMCTSOpponent(
            checkpoint_path=str(spec.get("checkpoint_path", "")),
            mcts_config=spec.get("mcts_config"),
            device=str(spec.get("device", "cpu")),
            name=str(spec.get("name", "checkpoint_mcts")),
        )
    raise ValueError(f"Unknown policy spec kind: {kind}")


def _state_dict_digest(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        tensor_cpu = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor_cpu.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
        digest.update(tensor_cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _freeze_for_cache(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return tuple((str(k), _freeze_for_cache(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_cache(v) for v in value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _policy_cache_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    kind = str(spec.get("kind", ""))
    key_parts: list[tuple[str, Any]] = [
        ("kind", kind),
        ("name", str(spec.get("name", ""))),
        ("device", str(spec.get("device", "cpu"))),
        ("mcts_config", _freeze_for_cache(spec.get("mcts_config"))),
    ]
    if kind == "checkpoint_mcts":
        key_parts.append(("checkpoint_path", str(spec.get("checkpoint_path", ""))))
    elif kind == "model_mcts":
        key_parts.append(("model_state_digest", str(spec.get("model_state_digest", ""))))
        key_parts.append(("model_kwargs", _freeze_for_cache(spec.get("model_kwargs"))))
        key_parts.append(("compat_adapter", str(spec.get("compat_adapter", ""))))
    return tuple(key_parts)


def _policy_from_spec_cached(spec: dict[str, Any]) -> Any:
    key = _policy_cache_key(spec)
    cached = _POLICY_SPEC_CACHE.get(key)
    if cached is not None:
        _POLICY_SPEC_CACHE.move_to_end(key)
        return cached
    policy = _policy_from_spec(spec)
    _POLICY_SPEC_CACHE[key] = policy
    _POLICY_SPEC_CACHE.move_to_end(key)
    while len(_POLICY_SPEC_CACHE) > int(_POLICY_CACHE_MAX_ENTRIES):
        _POLICY_SPEC_CACHE.popitem(last=False)
    return policy


def _run_matchup_worker_span(
    *,
    candidate_spec: dict[str, Any],
    opponent_spec: dict[str, Any],
    game_start: int,
    game_end: int,
    seat0_games: int,
    max_turns: int,
    seed_base: int,
    cycle_idx: int,
) -> dict[str, Any]:
    candidate_policy = _policy_from_spec_cached(candidate_spec)
    opponent_policy = _policy_from_spec_cached(opponent_spec)
    turns_all: list[int] = []
    turns_wins: list[int] = []
    turns_losses: list[int] = []
    turns_draws: list[int] = []
    candidate_wins = 0
    candidate_losses = 0
    draws = 0
    cutoffs = 0

    with SplendorNativeEnv() as env:
        for game_idx in range(int(game_start), int(game_end)):
            candidate_seat = 0 if game_idx < int(seat0_games) else 1
            game_seed = _stable_seed(seed_base, cycle_idx, opponent_policy.name, game_idx, candidate_seat)
            game_rng = random.Random(game_seed)
            result = play_game(
                env,
                candidate_policy,
                opponent_policy,
                seed=game_seed,
                candidate_seat=candidate_seat,
                max_turns=max_turns,
                rng=game_rng,
            )
            turns_all.append(int(result.num_turns))
            if result.reached_cutoff:
                cutoffs += 1
            if result.winner == -1:
                draws += 1
                turns_draws.append(int(result.num_turns))
            elif result.winner == candidate_seat:
                candidate_wins += 1
                turns_wins.append(int(result.num_turns))
            else:
                candidate_losses += 1
                turns_losses.append(int(result.num_turns))

    return {
        "candidate_wins": int(candidate_wins),
        "candidate_losses": int(candidate_losses),
        "draws": int(draws),
        "cutoffs": int(cutoffs),
        "turns_all": turns_all,
        "turns_wins": turns_wins,
        "turns_losses": turns_losses,
        "turns_draws": turns_draws,
    }
