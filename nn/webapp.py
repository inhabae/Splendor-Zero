from __future__ import annotations

import os
import random
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .checkpoints import load_checkpoint
from .mcts import MCTSConfig, run_mcts
from .native_env import SplendorNativeEnv, StepState
from .selfplay_dataset import (
    list_sessions as list_selfplay_sessions,
    load_session_npz,
    run_selfplay_session,
    run_selfplay_session_parallel,
    save_session_npz,
)
from .state_schema import (
    ACTION_DIM,
    BANK_START,
    CARD_FEATURE_LEN,
    CP_BONUSES_START,
    CP_POINTS_IDX,
    CP_RESERVED_START,
    CP_TOKENS_START,
    FACEUP_START,
    NOBLES_START,
    OP_BONUSES_START,
    OP_POINTS_IDX,
    OP_RESERVED_COUNT_IDX,
    OP_RESERVED_START,
    OP_TOKENS_START,
    STATE_DIM,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "nn_artifacts" / "checkpoints"
SELFPLAY_DIR = REPO_ROOT / "nn_artifacts" / "selfplay"
WEB_DIST_DIR = REPO_ROOT / "webui" / "dist"

_TAKE3_TRIPLETS = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 3, 4),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
)
_TAKE2_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)
_COLOR_NAMES = ("white", "blue", "green", "red", "black")
_REPLAY_MODEL_CACHE: dict[str, Any] = {}
_REPLAY_MODEL_CACHE_LOCK = threading.Lock()


class CheckpointDTO(BaseModel):
    id: str
    name: str
    path: str
    created_at: str
    size_bytes: int


class ActionInfoDTO(BaseModel):
    action_idx: int
    label: str


class MoveLogEntryDTO(BaseModel):
    turn_index: int
    actor: Literal["P0", "P1"]
    action_idx: int
    label: str


class GameConfigDTO(BaseModel):
    checkpoint_id: str
    checkpoint_path: str
    num_simulations: int
    player_seat: Literal["P0", "P1"]
    seed: int


class ColorCountsDTO(BaseModel):
    white: int
    blue: int
    green: int
    red: int
    black: int


class TokenCountsDTO(BaseModel):
    white: int
    blue: int
    green: int
    red: int
    black: int
    gold: int


class CardDTO(BaseModel):
    points: int
    bonus_color: Literal["white", "blue", "green", "red", "black"]
    cost: ColorCountsDTO
    source: Literal["faceup", "reserved_public", "reserved_private"]
    tier: int | None = None
    slot: int | None = None


class NobleDTO(BaseModel):
    points: int
    requirements: ColorCountsDTO


class TierRowDTO(BaseModel):
    tier: int
    deck_count: int
    cards: list[CardDTO]


class PlayerBoardDTO(BaseModel):
    seat: Literal["P0", "P1"]
    display_name: str
    points: int
    tokens: TokenCountsDTO
    bonuses: ColorCountsDTO
    reserved_public: list[CardDTO]
    reserved_total: int
    is_to_move: bool


class BoardMetaDTO(BaseModel):
    target_points: int
    turn_index: int
    player_to_move: Literal["P0", "P1"]


class BoardStateDTO(BaseModel):
    meta: BoardMetaDTO
    players: list[PlayerBoardDTO]
    bank: TokenCountsDTO
    nobles: list[NobleDTO]
    tiers: list[TierRowDTO]


class GameSnapshotDTO(BaseModel):
    game_id: str
    status: str
    player_to_move: Literal["P0", "P1"]
    legal_actions: list[int]
    legal_action_details: list[ActionInfoDTO]
    winner: int
    turn_index: int
    move_log: list[MoveLogEntryDTO]
    config: GameConfigDTO | None = None
    board_state: BoardStateDTO | None = None


class NewGameRequest(BaseModel):
    checkpoint_id: str
    num_simulations: int = Field(ge=1, le=5000)
    player_seat: Literal["P0", "P1"]
    seed: int | None = None


class PlayerMoveRequest(BaseModel):
    action_idx: int


class EngineApplyRequest(BaseModel):
    job_id: str


class EngineThinkResponse(BaseModel):
    job_id: str
    status: Literal["QUEUED", "RUNNING"]


class PlayerMoveResponse(BaseModel):
    snapshot: GameSnapshotDTO
    engine_should_move: bool


class EngineResultDTO(BaseModel):
    action_idx: int


class EngineJobStatusDTO(BaseModel):
    job_id: str
    status: Literal["QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"]
    error: str | None = None
    result: EngineResultDTO | None = None


class PlacementHintDTO(BaseModel):
    zone: Literal["faceup_card", "reserved_card", "bank_token", "other"]
    tier: int | None = None
    slot: int | None = None
    color: Literal["white", "blue", "green", "red", "black"] | None = None


class ActionVizDTO(BaseModel):
    action_idx: int
    label: str
    masked: bool
    policy_prob: float
    is_selected: bool
    placement_hint: PlacementHintDTO


class SelfPlayRunRequest(BaseModel):
    checkpoint_id: str
    num_simulations: int = Field(ge=1, le=5000, default=400)
    games: int = Field(ge=1, le=500, default=1)
    max_turns: int = Field(ge=1, le=400, default=100)
    seed: int | None = None
    workers: int | None = Field(default=None, ge=1, le=128)


class SelfPlayRunResponse(BaseModel):
    session_id: str
    path: str
    games: int
    steps: int
    created_at: str


class SelfPlaySessionDTO(BaseModel):
    session_id: str
    display_name: str
    path: str
    created_at: str
    games: int
    steps: int
    steps_per_episode: dict[str, int]
    metadata: dict[str, Any]


class SelfPlaySessionSummaryDTO(BaseModel):
    session_id: str
    path: str
    created_at: str
    games: int
    steps: int
    steps_per_episode: dict[str, int]
    metadata: dict[str, Any]
    winners_by_episode: dict[str, int]
    cutoff_by_episode: dict[str, bool]


class ReplayStepDTO(BaseModel):
    session_id: str
    episode_idx: int
    step_idx: int
    turn_idx: int
    player_id: int
    winner: int
    reached_cutoff: bool
    value_target: float
    value_root: float
    value_root_best: float
    model_value: float | None = None
    action_selected: int
    board_state: BoardStateDTO
    action_details: list[ActionVizDTO]
    model_action_details: list[ActionVizDTO] | None = None


@dataclass
class GameConfig:
    checkpoint_id: str
    checkpoint_path: Path
    num_simulations: int
    player_seat: str
    seed: int


@dataclass
class EngineJob:
    job_id: str
    game_id: str
    status: Literal["QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"]
    cancel_event: threading.Event
    future: Future[int] | None = None
    action_idx: int | None = None
    error: str | None = None


@dataclass
class MoveLogEntry:
    turn_index: int
    actor: str
    action_idx: int
    label: str


def _seat_str(player_id: int) -> Literal["P0", "P1"]:
    return "P0" if int(player_id) == 0 else "P1"


def _describe_action(action_idx: int) -> str:
    if 0 <= action_idx <= 11:
        return f"BUY face-up tier {action_idx // 4 + 1} slot {action_idx % 4}"
    if 12 <= action_idx <= 14:
        return f"BUY reserved slot {action_idx - 12}"
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        return f"RESERVE face-up tier {rel // 4 + 1} slot {rel % 4}"
    if 27 <= action_idx <= 29:
        return f"RESERVE from deck tier {action_idx - 27 + 1}"
    if 30 <= action_idx <= 39:
        tri = _TAKE3_TRIPLETS[action_idx - 30]
        names = ", ".join(_COLOR_NAMES[i] for i in tri)
        return f"TAKE 3 gems ({names})"
    if 40 <= action_idx <= 44:
        return f"TAKE 2 gems ({_COLOR_NAMES[action_idx - 40]})"
    if 45 <= action_idx <= 54:
        pair = _TAKE2_PAIRS[action_idx - 45]
        names = ", ".join(_COLOR_NAMES[i] for i in pair)
        return f"TAKE 2 gems ({names})"
    if 55 <= action_idx <= 59:
        return f"TAKE 1 gem ({_COLOR_NAMES[action_idx - 55]})"
    if action_idx == 60:
        return "PASS"
    if 61 <= action_idx <= 65:
        return f"RETURN gem ({_COLOR_NAMES[action_idx - 61]})"
    if 66 <= action_idx <= 68:
        return f"CHOOSE noble index {action_idx - 66}"
    return f"UNKNOWN action {action_idx}"


def _placement_hint_for_action(action_idx: int) -> PlacementHintDTO:
    if 0 <= action_idx <= 11:
        return PlacementHintDTO(
            zone="faceup_card",
            tier=(action_idx // 4) + 1,
            slot=(action_idx % 4),
        )
    if 12 <= action_idx <= 14:
        return PlacementHintDTO(
            zone="reserved_card",
            slot=(action_idx - 12),
        )
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        return PlacementHintDTO(
            zone="faceup_card",
            tier=(rel // 4) + 1,
            slot=(rel % 4),
        )
    if 27 <= action_idx <= 29:
        return PlacementHintDTO(zone="other", tier=(action_idx - 27 + 1))
    if 30 <= action_idx <= 39:
        # Use the first color in the TAKE-3 tuple as a stable placement anchor.
        color_idx = _TAKE3_TRIPLETS[action_idx - 30][0]
        return PlacementHintDTO(zone="bank_token", color=_COLOR_NAMES[color_idx])
    if 40 <= action_idx <= 44:
        return PlacementHintDTO(zone="bank_token", color=_COLOR_NAMES[action_idx - 40])
    if 45 <= action_idx <= 59:
        pair = _TAKE2_PAIRS[action_idx - 45] if action_idx <= 54 else (_COLOR_NAMES[action_idx - 55],)
        color = _COLOR_NAMES[pair[0]] if isinstance(pair[0], int) else pair[0]
        return PlacementHintDTO(zone="bank_token", color=color)
    if 61 <= action_idx <= 65:
        return PlacementHintDTO(zone="bank_token", color=_COLOR_NAMES[action_idx - 61])
    return PlacementHintDTO(zone="other")


def _action_viz_rows(mask: np.ndarray, policy: np.ndarray, selected_action: int) -> list[ActionVizDTO]:
    out: list[ActionVizDTO] = []
    for action_idx in range(ACTION_DIM):
        out.append(
            ActionVizDTO(
                action_idx=int(action_idx),
                label=_describe_action(action_idx),
                masked=not bool(mask[action_idx]),
                policy_prob=float(policy[action_idx]),
                is_selected=(int(selected_action) == int(action_idx)),
                placement_hint=_placement_hint_for_action(action_idx),
            )
        )
    return out


def _masked_softmax(policy_scores: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    probs = np.zeros((ACTION_DIM,), dtype=np.float32)
    legal = np.flatnonzero(np.asarray(legal_mask, dtype=np.bool_))
    if legal.size == 0:
        return probs
    legal_scores = np.asarray(policy_scores, dtype=np.float32)[legal]
    finite = np.isfinite(legal_scores)
    if not bool(np.any(finite)):
        probs[legal] = np.float32(1.0 / float(legal.size))
        return probs
    max_score = float(np.max(legal_scores[finite]))
    weights = np.zeros((int(legal.size),), dtype=np.float64)
    for i, score in enumerate(legal_scores):
        if np.isfinite(score):
            weights[i] = np.exp(float(score) - max_score)
    weight_sum = float(weights.sum())
    if not (weight_sum > 0.0) or not np.isfinite(weight_sum):
        probs[legal] = np.float32(1.0 / float(legal.size))
        return probs
    probs[legal] = (weights / weight_sum).astype(np.float32)
    return probs


def _load_replay_model(checkpoint_path: Path):
    key = str(checkpoint_path.resolve())
    with _REPLAY_MODEL_CACHE_LOCK:
        model = _REPLAY_MODEL_CACHE.get(key)
        if model is None:
            model = load_checkpoint(checkpoint_path, device="cpu")
            _REPLAY_MODEL_CACHE[key] = model
    return model


def _evaluate_model_replay_state(
    metadata: dict[str, Any],
    state: np.ndarray,
    mask: np.ndarray,
    selected_action: int,
) -> tuple[np.ndarray, float] | None:
    checkpoint_path_value = metadata.get("checkpoint_path")
    if checkpoint_path_value is None:
        return None
    checkpoint_path = Path(str(checkpoint_path_value))
    if not checkpoint_path.exists():
        return None

    state_np = np.asarray(state, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.bool_)
    if state_np.shape != (STATE_DIM,) or mask_np.shape != (ACTION_DIM,):
        return None
    if selected_action < 0 or selected_action >= ACTION_DIM:
        return None

    model = _load_replay_model(checkpoint_path)
    state_t = torch.as_tensor(state_np[None, :], dtype=torch.float32)
    with torch.no_grad():
        logits_t, value_t = model(state_t)

    logits = logits_t.detach().cpu().numpy().reshape(-1)
    if logits.shape != (ACTION_DIM,):
        return None
    value_arr = value_t.detach().cpu().numpy().reshape(-1)
    if value_arr.size != 1:
        return None

    policy = _masked_softmax(logits, mask_np)
    return policy, float(value_arr[0])


def _mask_to_actions(mask: np.ndarray) -> list[int]:
    return [int(v) for v in np.flatnonzero(mask)]


def _resolve_checkpoint_id(checkpoint_id: str) -> Path:
    allowed = {item.id: Path(item.path) for item in _scan_checkpoints()}
    path = allowed.get(checkpoint_id)
    if path is None:
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id")
    return path


def _selfplay_session_path(session_id: str) -> Path:
    return SELFPLAY_DIR / f"{session_id}.npz"


def _to_int(value: float, *, scale: float, max_hint: int | None = None) -> int:
    out = int(round(float(value) * scale))
    if out < 0:
        return 0
    if max_hint is not None:
        return min(out, int(max_hint))
    return out


def _decode_color_counts(block: np.ndarray, *, scale: float, max_hint: int | None = None) -> ColorCountsDTO:
    return ColorCountsDTO(
        white=_to_int(block[0], scale=scale, max_hint=max_hint),
        blue=_to_int(block[1], scale=scale, max_hint=max_hint),
        green=_to_int(block[2], scale=scale, max_hint=max_hint),
        red=_to_int(block[3], scale=scale, max_hint=max_hint),
        black=_to_int(block[4], scale=scale, max_hint=max_hint),
    )


def _decode_token_counts(block: np.ndarray) -> TokenCountsDTO:
    return TokenCountsDTO(
        white=_to_int(block[0], scale=4.0, max_hint=7),
        blue=_to_int(block[1], scale=4.0, max_hint=7),
        green=_to_int(block[2], scale=4.0, max_hint=7),
        red=_to_int(block[3], scale=4.0, max_hint=7),
        black=_to_int(block[4], scale=4.0, max_hint=7),
        gold=_to_int(block[5], scale=5.0, max_hint=7),
    )


def _decode_card(
    block: np.ndarray,
    *,
    source: Literal["faceup", "reserved_public", "reserved_private"],
    tier: int | None = None,
    slot: int | None = None,
) -> CardDTO | None:
    if block.shape != (CARD_FEATURE_LEN,):
        return None
    if not np.any(block):
        return None
    costs = _decode_color_counts(block[:5], scale=7.0, max_hint=7)
    bonus_slice = block[5:10]
    color_idx = int(np.argmax(bonus_slice))
    bonus_color = _COLOR_NAMES[color_idx]
    points = _to_int(block[10], scale=5.0, max_hint=5)
    return CardDTO(
        points=points,
        bonus_color=bonus_color,  # type: ignore[arg-type]
        cost=costs,
        source=source,
        tier=tier,
        slot=slot,
    )


def _private_reserved_placeholder(slot: int) -> CardDTO:
    return CardDTO(
        points=0,
        bonus_color="white",
        cost=ColorCountsDTO(white=0, blue=0, green=0, red=0, black=0),
        source="reserved_private",
        slot=slot,
    )


def _safe_slice(state: np.ndarray, start: int, length: int) -> np.ndarray:
    end = start + length
    if end > state.shape[0]:
        raise ValueError(f"decode slice out of range: start={start}, length={length}, shape={state.shape}")
    return state[start:end]


def _decode_board_state(step: StepState, *, turn_index: int, player_seat: str) -> BoardStateDTO:
    state = np.asarray(step.state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Unexpected state shape for board decode: {state.shape}")

    cp_tokens = _decode_token_counts(_safe_slice(state, CP_TOKENS_START, 6))
    cp_bonuses = _decode_color_counts(_safe_slice(state, CP_BONUSES_START, 5), scale=7.0, max_hint=7)
    cp_points = _to_int(state[CP_POINTS_IDX], scale=20.0, max_hint=30)

    op_tokens = _decode_token_counts(_safe_slice(state, OP_TOKENS_START, 6))
    op_bonuses = _decode_color_counts(_safe_slice(state, OP_BONUSES_START, 5), scale=7.0, max_hint=7)
    op_points = _to_int(state[OP_POINTS_IDX], scale=20.0, max_hint=30)

    cp_reserved: list[CardDTO] = []
    for i in range(3):
        block = _safe_slice(state, CP_RESERVED_START + i * CARD_FEATURE_LEN, CARD_FEATURE_LEN)
        card = _decode_card(block, source="reserved_public", slot=i)
        if card is not None:
            cp_reserved.append(card)

    op_reserved: list[CardDTO] = []
    for i in range(3):
        block = _safe_slice(state, OP_RESERVED_START + i * CARD_FEATURE_LEN, CARD_FEATURE_LEN)
        card = _decode_card(block, source="reserved_public", slot=i)
        if card is not None:
            op_reserved.append(card)

    op_reserved_total = _to_int(state[OP_RESERVED_COUNT_IDX], scale=3.0, max_hint=3)
    cp_reserved_total = len(cp_reserved)

    private_op_reserved = max(op_reserved_total - len(op_reserved), 0)
    for i in range(private_op_reserved):
        slot = len(op_reserved) + i
        op_reserved.append(_private_reserved_placeholder(slot))

    cp_id = int(step.current_player_id)
    if cp_id not in (0, 1):
        raise ValueError(f"Unexpected current_player_id: {cp_id}")
    op_id = 1 - cp_id

    players: dict[int, PlayerBoardDTO] = {
        cp_id: PlayerBoardDTO(
            seat=_seat_str(cp_id),
            display_name=f"{'You' if _seat_str(cp_id) == player_seat else 'Engine'} ({_seat_str(cp_id)})",
            points=cp_points,
            tokens=cp_tokens,
            bonuses=cp_bonuses,
            reserved_public=cp_reserved,
            reserved_total=cp_reserved_total,
            is_to_move=True,
        ),
        op_id: PlayerBoardDTO(
            seat=_seat_str(op_id),
            display_name=f"{'You' if _seat_str(op_id) == player_seat else 'Engine'} ({_seat_str(op_id)})",
            points=op_points,
            tokens=op_tokens,
            bonuses=op_bonuses,
            reserved_public=op_reserved,
            reserved_total=op_reserved_total,
            is_to_move=False,
        ),
    }

    bank = _decode_token_counts(_safe_slice(state, BANK_START, 6))

    nobles: list[NobleDTO] = []
    for i in range(3):
        block = _safe_slice(state, NOBLES_START + i * 5, 5)
        if not np.any(block):
            continue
        nobles.append(NobleDTO(points=3, requirements=_decode_color_counts(block, scale=4.0, max_hint=4)))

    tiers: list[TierRowDTO] = []
    for tier in (3, 2, 1):
        cards: list[CardDTO] = []
        tier_offset = FACEUP_START + (tier - 1) * 4 * CARD_FEATURE_LEN
        for slot in range(4):
            block = _safe_slice(state, tier_offset + slot * CARD_FEATURE_LEN, CARD_FEATURE_LEN)
            card = _decode_card(block, source="faceup", tier=tier, slot=slot)
            if card is not None:
                cards.append(card)
        tiers.append(TierRowDTO(tier=tier, deck_count=-1, cards=cards))

    return BoardStateDTO(
        meta=BoardMetaDTO(
            target_points=15,
            turn_index=int(turn_index),
            player_to_move=_seat_str(step.current_player_id),
        ),
        players=[players[0], players[1]],
        bank=bank,
        nobles=nobles,
        tiers=tiers,
    )


def _scan_checkpoints() -> list[CheckpointDTO]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[CheckpointDTO] = []
    for path in CHECKPOINT_DIR.glob("*.pt"):
        st = path.stat()
        created = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        items.append(
            CheckpointDTO(
                id=str(path.resolve()),
                name=path.name,
                path=str(path.resolve()),
                created_at=created,
                size_bytes=int(st.st_size),
            )
        )
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


def _decode_replay_step(session_id: str, session_path: Path, episode_idx: int, step_idx: int) -> ReplayStepDTO:
    session = load_session_npz(session_path)
    target = None
    for step in session.steps:
        if int(step.episode_idx) == int(episode_idx) and int(step.step_idx) == int(step_idx):
            target = step
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Replay step not found")

    step_state = StepState(
        state=target.state.copy(),
        mask=target.mask.copy(),
        is_terminal=bool(target.winner != -2),
        winner=int(target.winner),
        current_player_id=int(target.current_player_id),
    )
    board_state = _decode_board_state(
        step_state,
        turn_index=int(target.turn_idx),
        player_seat=_seat_str(int(target.current_player_id)),
    )
    model_value: float | None = None
    model_action_details: list[ActionVizDTO] | None = None
    model_eval = _evaluate_model_replay_state(
        session.metadata,
        target.state,
        target.mask,
        target.action_selected,
    )
    if model_eval is not None:
        model_policy, model_value = model_eval
        model_action_details = _action_viz_rows(target.mask, model_policy, target.action_selected)
    return ReplayStepDTO(
        session_id=session_id,
        episode_idx=int(target.episode_idx),
        step_idx=int(target.step_idx),
        turn_idx=int(target.turn_idx),
        player_id=int(target.player_id),
        winner=int(target.winner),
        reached_cutoff=bool(target.reached_cutoff),
        value_target=float(target.value_target),
        value_root=float(target.value_root),
        value_root_best=float(target.value_root_best),
        model_value=model_value,
        action_selected=int(target.action_selected),
        board_state=board_state,
        action_details=_action_viz_rows(target.mask, target.policy, target.action_selected),
        model_action_details=model_action_details,
    )


def _build_selfplay_summary(session_id: str, session_path: Path) -> SelfPlaySessionSummaryDTO:
    session = load_session_npz(session_path)
    by_episode_steps: dict[str, int] = {}
    winners: dict[str, int] = {}
    cutoffs: dict[str, bool] = {}
    for step in session.steps:
        key = str(int(step.episode_idx))
        by_episode_steps[key] = by_episode_steps.get(key, 0) + 1
        winners[key] = int(step.winner)
        cutoffs[key] = bool(step.reached_cutoff)
    return SelfPlaySessionSummaryDTO(
        session_id=session_id,
        path=str(session_path.resolve()),
        created_at=session.created_at,
        games=int(session.metadata.get("games", 0)),
        steps=len(session.steps),
        steps_per_episode=by_episode_steps,
        metadata=session.metadata,
        winners_by_episode=winners,
        cutoff_by_episode=cutoffs,
    )


class GameManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-think")
        self._env: SplendorNativeEnv | None = None
        self._game_id: str | None = None
        self._config: GameConfig | None = None
        self._move_log: list[MoveLogEntry] = []
        self._rng = random.Random(0)
        self._model_cache: dict[str, Any] = {}
        self._active_job_id: str | None = None
        self._jobs: dict[str, EngineJob] = {}
        self._forced_winner: int | None = None

    def list_checkpoints(self) -> list[CheckpointDTO]:
        return _scan_checkpoints()

    def _cancel_active_job_locked(self) -> None:
        if self._active_job_id is None:
            return
        job = self._jobs.get(self._active_job_id)
        if job is None:
            self._active_job_id = None
            return
        job.cancel_event.set()
        if job.status in ("QUEUED", "RUNNING"):
            job.status = "CANCELLED"
        self._active_job_id = None

    def _ensure_env_locked(self) -> SplendorNativeEnv:
        if self._env is None:
            self._env = SplendorNativeEnv()
        return self._env

    def _require_game_locked(self) -> tuple[SplendorNativeEnv, GameConfig, str]:
        if self._game_id is None or self._config is None:
            raise HTTPException(status_code=400, detail="No active game")
        env = self._ensure_env_locked()
        return env, self._config, self._game_id

    def _resolve_checkpoint(self, checkpoint_id: str) -> Path:
        return _resolve_checkpoint_id(checkpoint_id)

    def new_game(self, req: NewGameRequest) -> GameSnapshotDTO:
        with self._lock:
            checkpoint_path = self._resolve_checkpoint(req.checkpoint_id)
            self._cancel_active_job_locked()
            env = self._ensure_env_locked()

            seed = int(req.seed) if req.seed is not None else random.randint(0, 2**31 - 1)
            env.reset(seed=seed)

            self._game_id = str(uuid.uuid4())
            self._config = GameConfig(
                checkpoint_id=req.checkpoint_id,
                checkpoint_path=checkpoint_path,
                num_simulations=int(req.num_simulations),
                player_seat=str(req.player_seat),
                seed=seed,
            )
            self._move_log = []
            self._rng = random.Random(seed)
            self._forced_winner = None
            return self._snapshot_locked()

    def _snapshot_locked(self) -> GameSnapshotDTO:
        env, config, game_id = self._require_game_locked()
        step = env.get_state()

        winner = int(step.winner)
        status = "IN_PROGRESS"
        legal_actions = _mask_to_actions(step.mask) if not step.is_terminal else []

        if self._forced_winner is not None:
            winner = int(self._forced_winner)
            status = "RESIGNED"
            legal_actions = []
        elif step.is_terminal:
            status = "COMPLETED"

        return GameSnapshotDTO(
            game_id=game_id,
            status=status,
            player_to_move=_seat_str(step.current_player_id),
            legal_actions=legal_actions,
            legal_action_details=[ActionInfoDTO(action_idx=a, label=_describe_action(a)) for a in legal_actions],
            winner=winner,
            turn_index=len(self._move_log),
            move_log=[
                MoveLogEntryDTO(
                    turn_index=m.turn_index,
                    actor=_seat_str(0 if m.actor == "P0" else 1),
                    action_idx=m.action_idx,
                    label=m.label,
                )
                for m in self._move_log
            ],
            config=GameConfigDTO(
                checkpoint_id=config.checkpoint_id,
                checkpoint_path=str(config.checkpoint_path),
                num_simulations=config.num_simulations,
                player_seat="P0" if config.player_seat == "P0" else "P1",
                seed=config.seed,
            ),
            board_state=_decode_board_state(step, turn_index=len(self._move_log), player_seat=config.player_seat),
        )

    def get_state(self) -> GameSnapshotDTO:
        with self._lock:
            return self._snapshot_locked()

    def _is_player_turn_locked(self, step: StepState, config: GameConfig) -> bool:
        return _seat_str(step.current_player_id) == config.player_seat

    def player_move(self, req: PlayerMoveRequest) -> PlayerMoveResponse:
        with self._lock:
            env, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")

            step = env.get_state()
            if step.is_terminal:
                raise HTTPException(status_code=400, detail="Game already finished")
            if not self._is_player_turn_locked(step, config):
                raise HTTPException(status_code=400, detail="Not player's turn")
            if req.action_idx < 0 or req.action_idx >= int(step.mask.shape[0]):
                raise HTTPException(status_code=400, detail="action_idx out of bounds")
            if not bool(step.mask[int(req.action_idx)]):
                raise HTTPException(status_code=400, detail="Action is not legal")

            actor = _seat_str(step.current_player_id)
            env.step(int(req.action_idx))
            self._move_log.append(
                MoveLogEntry(
                    turn_index=len(self._move_log),
                    actor=actor,
                    action_idx=int(req.action_idx),
                    label=_describe_action(int(req.action_idx)),
                )
            )
            snapshot = self._snapshot_locked()
            engine_should_move = snapshot.status == "IN_PROGRESS" and snapshot.player_to_move != config.player_seat
            return PlayerMoveResponse(snapshot=snapshot, engine_should_move=engine_should_move)

    def _get_model_locked(self, config: GameConfig):
        key = str(config.checkpoint_path)
        model = self._model_cache.get(key)
        if model is None:
            model = load_checkpoint(config.checkpoint_path, device="cpu")
            self._model_cache[key] = model
        return model

    def start_engine_think(self) -> EngineThinkResponse:
        with self._lock:
            env, config, game_id = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")

            step = env.get_state()
            if step.is_terminal:
                raise HTTPException(status_code=400, detail="Game already finished")
            if self._is_player_turn_locked(step, config):
                raise HTTPException(status_code=400, detail="Engine cannot move on player's turn")

            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in ("QUEUED", "RUNNING"):
                    raise HTTPException(status_code=400, detail="Engine job already active")
                self._active_job_id = None

            job = EngineJob(
                job_id=str(uuid.uuid4()),
                game_id=game_id,
                status="QUEUED",
                cancel_event=threading.Event(),
            )
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id

            model = self._get_model_locked(config)
            turns_taken = len(self._move_log)

            def _run() -> int:
                with self._lock:
                    cur_job = self._jobs.get(job.job_id)
                    if cur_job is None:
                        raise RuntimeError("Engine job disappeared")
                    if cur_job.cancel_event.is_set():
                        cur_job.status = "CANCELLED"
                        raise RuntimeError("Engine job cancelled")
                    cur_job.status = "RUNNING"

                try:
                    result = run_mcts(
                        env,
                        model,
                        state=step,
                        turns_taken=turns_taken,
                        device="cpu",
                        config=MCTSConfig(
                            num_simulations=config.num_simulations,
                            c_puct=1.25,
                            temperature_moves=0,
                            temperature=0.0,
                            root_dirichlet_noise=False,
                        ),
                        rng=self._rng,
                    )
                    with self._lock:
                        cur_job = self._jobs.get(job.job_id)
                        if cur_job is None:
                            raise RuntimeError("Engine job disappeared")
                        if cur_job.cancel_event.is_set():
                            cur_job.status = "CANCELLED"
                            raise RuntimeError("Engine job cancelled")
                        cur_job.status = "DONE"
                        cur_job.action_idx = int(result.chosen_action_idx)
                        if self._active_job_id == job.job_id:
                            self._active_job_id = None
                        return int(result.chosen_action_idx)
                except Exception as exc:
                    with self._lock:
                        cur_job = self._jobs.get(job.job_id)
                        if cur_job is not None and cur_job.status != "CANCELLED":
                            cur_job.status = "FAILED"
                            cur_job.error = str(exc)
                        if self._active_job_id == job.job_id:
                            self._active_job_id = None
                    raise

            job.future = self._executor.submit(_run)
            return EngineThinkResponse(job_id=job.job_id, status="QUEUED")

    def get_engine_job(self, job_id: str) -> EngineJobStatusDTO:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            result = EngineResultDTO(action_idx=job.action_idx) if job.action_idx is not None else None
            return EngineJobStatusDTO(job_id=job.job_id, status=job.status, error=job.error, result=result)

    def apply_engine_move(self, req: EngineApplyRequest) -> GameSnapshotDTO:
        with self._lock:
            env, config, game_id = self._require_game_locked()
            job = self._jobs.get(req.job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            if job.game_id != game_id:
                raise HTTPException(status_code=400, detail="Job does not belong to current game")
            if job.status != "DONE" or job.action_idx is None:
                raise HTTPException(status_code=400, detail="Engine job is not ready")
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")

            step = env.get_state()
            if step.is_terminal:
                raise HTTPException(status_code=400, detail="Game already finished")
            if self._is_player_turn_locked(step, config):
                raise HTTPException(status_code=400, detail="It is player's turn")
            if not bool(step.mask[job.action_idx]):
                raise HTTPException(status_code=400, detail="Engine produced illegal action")

            actor = _seat_str(step.current_player_id)
            env.step(int(job.action_idx))
            self._move_log.append(
                MoveLogEntry(
                    turn_index=len(self._move_log),
                    actor=actor,
                    action_idx=int(job.action_idx),
                    label=_describe_action(int(job.action_idx)),
                )
            )
            return self._snapshot_locked()

    def resign(self) -> GameSnapshotDTO:
        with self._lock:
            _, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                return self._snapshot_locked()
            self._cancel_active_job_locked()
            self._forced_winner = 1 if config.player_seat == "P0" else 0
            return self._snapshot_locked()


manager = GameManager()
app = FastAPI(title="Splendor vs MCTS UI API")


@app.get("/api/checkpoints", response_model=list[CheckpointDTO])
def get_checkpoints() -> list[CheckpointDTO]:
    return manager.list_checkpoints()


@app.post("/api/game/new", response_model=GameSnapshotDTO)
def new_game(req: NewGameRequest) -> GameSnapshotDTO:
    return manager.new_game(req)


@app.get("/api/game/state", response_model=GameSnapshotDTO)
def game_state() -> GameSnapshotDTO:
    return manager.get_state()


@app.post("/api/game/player-move", response_model=PlayerMoveResponse)
def player_move(req: PlayerMoveRequest) -> PlayerMoveResponse:
    return manager.player_move(req)


@app.post("/api/game/engine-think", response_model=EngineThinkResponse)
def engine_think() -> EngineThinkResponse:
    return manager.start_engine_think()


@app.get("/api/game/engine-job/{job_id}", response_model=EngineJobStatusDTO)
def engine_job(job_id: str) -> EngineJobStatusDTO:
    return manager.get_engine_job(job_id)


@app.post("/api/game/engine-apply", response_model=GameSnapshotDTO)
def engine_apply(req: EngineApplyRequest) -> GameSnapshotDTO:
    return manager.apply_engine_move(req)


@app.post("/api/game/resign", response_model=GameSnapshotDTO)
def game_resign() -> GameSnapshotDTO:
    return manager.resign()


@app.get("/api/selfplay/sessions", response_model=list[SelfPlaySessionDTO])
def selfplay_sessions() -> list[SelfPlaySessionDTO]:
    rows = list_selfplay_sessions(SELFPLAY_DIR)
    return [SelfPlaySessionDTO(**row) for row in rows]


@app.post("/api/selfplay/run", response_model=SelfPlayRunResponse)
def selfplay_run(req: SelfPlayRunRequest) -> SelfPlayRunResponse:
    checkpoint_path = _resolve_checkpoint_id(req.checkpoint_id)
    seed = int(req.seed) if req.seed is not None else random.randint(1, 2**31 - 1)
    games = int(req.games)
    requested_workers = int(req.workers) if req.workers is not None else None
    auto_workers = int(os.cpu_count() or 1)
    workers_used = min(games, requested_workers if requested_workers is not None else auto_workers)
    workers_used = max(1, int(workers_used))

    try:
        if workers_used > 1:
            session = run_selfplay_session_parallel(
                checkpoint_path=checkpoint_path,
                games=games,
                max_turns=int(req.max_turns),
                num_simulations=int(req.num_simulations),
                seed_base=seed,
                workers=workers_used,
            )
        else:
            model = load_checkpoint(checkpoint_path, device="cpu")
            with SplendorNativeEnv() as env:
                session = run_selfplay_session(
                    env=env,
                    model=model,
                    games=games,
                    max_turns=int(req.max_turns),
                    num_simulations=int(req.num_simulations),
                    seed_base=seed,
                )
            session.metadata["workers_requested"] = int(requested_workers) if requested_workers is not None else None
            session.metadata["workers_used"] = int(workers_used)
            session.metadata["parallelism_mode"] = "process_pool"
            session.metadata["games_per_worker"] = [int(games)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Self-play run failed: {exc}") from exc

    session.metadata["workers_requested"] = int(requested_workers) if requested_workers is not None else None
    session.metadata["workers_used"] = int(workers_used)
    session.metadata["parallelism_mode"] = "process_pool"
    if "games_per_worker" not in session.metadata:
        session.metadata["games_per_worker"] = [int(games)]
    session.metadata["checkpoint_id"] = req.checkpoint_id
    session.metadata["checkpoint_path"] = str(checkpoint_path.resolve())
    out_path = save_session_npz(session, SELFPLAY_DIR)
    return SelfPlayRunResponse(
        session_id=session.session_id,
        path=str(out_path.resolve()),
        games=int(req.games),
        steps=len(session.steps),
        created_at=session.created_at,
    )


@app.get("/api/selfplay/session/{session_id}/summary", response_model=SelfPlaySessionSummaryDTO)
def selfplay_session_summary(session_id: str) -> SelfPlaySessionSummaryDTO:
    session_path = _selfplay_session_path(session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return _build_selfplay_summary(session_id, session_path)


@app.get("/api/selfplay/session/{session_id}/step", response_model=ReplayStepDTO)
def selfplay_session_step(session_id: str, episode_idx: int, step_idx: int) -> ReplayStepDTO:
    session_path = _selfplay_session_path(session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return _decode_replay_step(session_id, session_path, episode_idx, step_idx)


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


if WEB_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST_DIR), html=True), name="web")
