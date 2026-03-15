from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .model import MaskedPolicyValueNet
from .state_schema import (
    BANK_START,
    FACEUP_START,
    NOBLES_START,
    OPPONENT_RESERVED_SLOT_LEN,
    OP_POINTS_IDX,
    OP_RESERVED_IS_OCCUPIED_OFFSET,
    OP_RESERVED_START,
    PHASE_FLAGS_START,
    STATE_DIM,
)

LEGACY_STATE_DIM = 246
CURRENT_STATE_DIM = STATE_DIM
_LEGACY_OP_POINTS_IDX = 56
_LEGACY_OP_RESERVED_START = 57
_LEGACY_OP_RESERVED_COUNT_IDX = 90
_LEGACY_FACEUP_START = 91
_LEGACY_BANK_START = 223
_LEGACY_NOBLES_START = 229
_LEGACY_PHASE_FLAGS_START = 244


def _canonicalize_model_kwargs(raw_model_kwargs: dict[str, Any]) -> dict[str, int]:
    model_kwargs = dict(raw_model_kwargs)
    if "res_blocks" not in model_kwargs:
        # Backward compatibility: legacy checkpoints predate explicit ResBlock
        # serialization and correspond to the no-ResBlock architecture.
        model_kwargs["res_blocks"] = 0
    return {str(key): int(value) for key, value in model_kwargs.items()}


def _project_state_for_legacy_checkpoint(state: Tensor) -> Tensor:
    if state.ndim != 2:
        raise ValueError(f"Legacy checkpoint adapter expects rank-2 state tensor, got {tuple(state.shape)}")
    if state.shape[1] == LEGACY_STATE_DIM:
        return state
    if state.shape[1] != STATE_DIM:
        raise ValueError(
            f"Legacy checkpoint adapter expects state dim {STATE_DIM} or {LEGACY_STATE_DIM}, got {tuple(state.shape)}"
        )

    projected = state.new_zeros((state.shape[0], LEGACY_STATE_DIM))
    projected[:, :_LEGACY_OP_POINTS_IDX + 1] = state[:, :OP_POINTS_IDX + 1]

    for slot in range(3):
        new_start = OP_RESERVED_START + slot * OPPONENT_RESERVED_SLOT_LEN
        new_end = new_start + 11
        legacy_start = _LEGACY_OP_RESERVED_START + slot * 11
        legacy_end = legacy_start + 11
        projected[:, legacy_start:legacy_end] = state[:, new_start:new_end]
        projected[:, _LEGACY_OP_RESERVED_COUNT_IDX] += state[:, new_start + OP_RESERVED_IS_OCCUPIED_OFFSET]

    projected[:, _LEGACY_FACEUP_START:_LEGACY_BANK_START] = state[:, FACEUP_START:BANK_START]
    projected[:, _LEGACY_BANK_START:_LEGACY_NOBLES_START] = state[:, BANK_START:NOBLES_START]
    projected[:, _LEGACY_NOBLES_START:_LEGACY_PHASE_FLAGS_START] = state[:, NOBLES_START:PHASE_FLAGS_START]
    projected[:, _LEGACY_PHASE_FLAGS_START:] = state[:, PHASE_FLAGS_START:]
    return projected


class LegacyCheckpointAdapter(nn.Module):
    def __init__(self, base_model: MaskedPolicyValueNet) -> None:
        super().__init__()
        self.base_model = base_model
        self.input_dim = int(CURRENT_STATE_DIM)
        self.hidden_dim = int(getattr(base_model, "hidden_dim"))
        self.action_dim = int(getattr(base_model, "action_dim"))
        self.res_blocks = int(getattr(base_model, "res_blocks", 0))
        self.compat_adapter = "legacy_246_to_252"
        self.legacy_input_dim = int(getattr(base_model, "input_dim", LEGACY_STATE_DIM))

    def export_model_kwargs(self) -> dict[str, int]:
        return dict(self.base_model.export_model_kwargs())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.base_model(_project_state_for_legacy_checkpoint(x))


def _build_model_from_components(
    model_kwargs: dict[str, Any],
    state_dict: dict[str, Any],
    *,
    device: str,
    purpose: Literal["inference", "resume_training"],
) -> nn.Module:
    normalized_kwargs = _canonicalize_model_kwargs(model_kwargs)
    input_dim = int(normalized_kwargs.get("input_dim", CURRENT_STATE_DIM))
    model = MaskedPolicyValueNet(**normalized_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if input_dim == CURRENT_STATE_DIM:
        return model
    if input_dim != LEGACY_STATE_DIM:
        raise ValueError(
            f"Unsupported checkpoint input_dim {input_dim}; expected {CURRENT_STATE_DIM} or {LEGACY_STATE_DIM}"
        )
    if purpose == "resume_training":
        raise ValueError(
            f"Checkpoint input_dim {LEGACY_STATE_DIM} is only supported for inference/evaluation. "
            f"Resuming training requires a checkpoint with input_dim {CURRENT_STATE_DIM}."
        )
    return LegacyCheckpointAdapter(model).to(device)


@dataclass
class CheckpointInfo:
    path: Path
    run_id: str
    cycle_idx: int
    seed: int
    created_at: str
    collector_policy: str
    mcts_sims: int


@dataclass
class LoadedCheckpoint:
    model: nn.Module
    path: Path
    run_id: str
    cycle_idx: int
    created_at: str
    metadata: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None


def _load_checkpoint_payload(path: str | Path, *, device: str = "cpu") -> tuple[Path, dict[str, Any]]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility for older torch versions that do not support weights_only.
        payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload type: {type(payload)}")
    return ckpt_path, payload


def _build_model_from_payload(
    payload: dict[str, Any],
    *,
    device: str = "cpu",
    purpose: Literal["inference", "resume_training"] = "inference",
) -> nn.Module:
    raw_model_kwargs = payload.get("model_kwargs")
    if not isinstance(raw_model_kwargs, dict):
        raise ValueError("Checkpoint missing model_kwargs")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing model_state_dict")
    return _build_model_from_components(raw_model_kwargs, state_dict, device=device, purpose=purpose)


def load_checkpoint_with_metadata(
    path: str | Path,
    *,
    device: str = "cpu",
    purpose: Literal["inference", "resume_training"] = "inference",
) -> LoadedCheckpoint:
    ckpt_path, payload = _load_checkpoint_payload(path, device=device)
    model = _build_model_from_payload(payload, device=device, purpose=purpose)
    metadata = dict(payload.get("metadata") or {})
    return LoadedCheckpoint(
        model=model,
        path=ckpt_path,
        run_id=str(payload.get("run_id", "")),
        cycle_idx=int(payload.get("cycle_idx", 0) or 0),
        created_at=str(payload.get("created_at", "")),
        metadata=metadata,
        optimizer_state_dict=payload.get("optimizer_state_dict"),
    )


def save_checkpoint(
    model: MaskedPolicyValueNet,
    *,
    output_dir: str | Path,
    run_id: str,
    cycle_idx: int,
    metadata: dict[str, Any],
    optimizer: Optimizer | None = None,
) -> CheckpointInfo:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    path = out_dir / f"{run_id}_cycle_{int(cycle_idx):04d}.pt"

    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.export_model_kwargs(),
        "metadata": dict(metadata),
        "run_id": str(run_id),
        "cycle_idx": int(cycle_idx),
        "created_at": created_at,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)

    return CheckpointInfo(
        path=path,
        run_id=str(run_id),
        cycle_idx=int(cycle_idx),
        seed=int(metadata.get("seed", 0)),
        created_at=created_at,
        collector_policy=str(metadata.get("collector_policy", "")),
        mcts_sims=int(metadata.get("mcts_sims", 0)),
    )


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> nn.Module:
    return load_checkpoint_with_metadata(path, device=device).model


def load_model_from_spec(
    *,
    model_kwargs: dict[str, Any],
    state_dict: dict[str, Any],
    device: str = "cpu",
    compat_adapter: str | None = None,
) -> nn.Module:
    model = _build_model_from_components(model_kwargs, state_dict, device=device, purpose="inference")
    if compat_adapter is None:
        return model
    if compat_adapter != "legacy_246_to_252":
        raise ValueError(f"Unknown checkpoint compatibility adapter: {compat_adapter}")
    if not isinstance(model, LegacyCheckpointAdapter):
        raise ValueError("Compatibility adapter marker requested for a non-legacy model spec")
    return model
