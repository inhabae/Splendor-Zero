from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .model import MaskedPolicyValueNet


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
    model: MaskedPolicyValueNet
    path: Path
    run_id: str
    cycle_idx: int
    created_at: str
    metadata: dict[str, Any]


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


def _build_model_from_payload(payload: dict[str, Any], *, device: str = "cpu") -> MaskedPolicyValueNet:
    raw_model_kwargs = payload.get("model_kwargs")
    if not isinstance(raw_model_kwargs, dict):
        raise ValueError("Checkpoint missing model_kwargs")
    model_kwargs = dict(raw_model_kwargs)
    if "res_blocks" not in model_kwargs:
        raise ValueError(
            "Checkpoint missing model_kwargs.res_blocks. "
            "Older checkpoints are intentionally unsupported after ResBlock rollout."
        )
    model = MaskedPolicyValueNet(**model_kwargs).to(device)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_checkpoint_with_metadata(path: str | Path, *, device: str = "cpu") -> LoadedCheckpoint:
    ckpt_path, payload = _load_checkpoint_payload(path, device=device)
    model = _build_model_from_payload(payload, device=device)
    metadata = dict(payload.get("metadata") or {})
    return LoadedCheckpoint(
        model=model,
        path=ckpt_path,
        run_id=str(payload.get("run_id", "")),
        cycle_idx=int(payload.get("cycle_idx", 0) or 0),
        created_at=str(payload.get("created_at", "")),
        metadata=metadata,
    )


def save_checkpoint(
    model: MaskedPolicyValueNet,
    *,
    output_dir: str | Path,
    run_id: str,
    cycle_idx: int,
    metadata: dict[str, Any],
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


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> MaskedPolicyValueNet:
    return load_checkpoint_with_metadata(path, device=device).model
