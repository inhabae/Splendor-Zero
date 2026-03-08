from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from .state_schema import ACTION_DIM, STATE_DIM


@dataclass
class ReplaySample:
    state: np.ndarray  # (246,) float32
    mask: np.ndarray  # (69,) bool
    action_target: int  # int64-compatible
    value_target: float  # blended value target, typically in [-1, +1]
    policy_target: np.ndarray | None = None  # (69,) float32, sums to 1 over legal actions


class ReplayBuffer:
    def __init__(self, max_size: int | None = None) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive when provided")
        self._samples: List[ReplaySample] = []
        self._max_size = int(max_size) if max_size is not None else None

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: ReplaySample) -> None:
        if sample.state.shape != (STATE_DIM,):
            raise ValueError(f"Invalid state shape {sample.state.shape}")
        if sample.mask.shape != (ACTION_DIM,):
            raise ValueError(f"Invalid mask shape {sample.mask.shape}")
        action_idx = int(sample.action_target)
        if not (0 <= action_idx < ACTION_DIM):
            raise ValueError(f"action_target out of range: {sample.action_target}")
        if not bool(sample.mask[action_idx]):
            raise ValueError("action_target must be legal under sample.mask")
        if sample.policy_target is None:
            policy_target = np.zeros((ACTION_DIM,), dtype=np.float32)
            policy_target[action_idx] = 1.0
            sample.policy_target = policy_target
        if sample.policy_target.shape != (ACTION_DIM,):
            raise ValueError(f"Invalid policy_target shape {sample.policy_target.shape}")
        if not np.isfinite(sample.policy_target).all():
            raise ValueError("Non-finite values in policy_target")
        if (sample.policy_target < 0).any():
            raise ValueError("policy_target cannot contain negative values")
        if (sample.policy_target[~sample.mask] != 0).any():
            raise ValueError("policy_target must assign zero probability to illegal actions")
        prob_sum = float(sample.policy_target.sum())
        if abs(prob_sum - 1.0) > 1e-5:
            raise ValueError(f"policy_target must sum to 1 (got {prob_sum})")
        self._samples.append(sample)
        if self._max_size is not None and len(self._samples) > self._max_size:
            overflow = len(self._samples) - self._max_size
            del self._samples[:overflow]

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        for s in samples:
            self.add(s)

    def sample_batch(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if not self._samples:
            raise ValueError("ReplayBuffer is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        picks = random.sample(self._samples, k=min(batch_size, len(self._samples)))
        states = np.stack([s.state for s in picks], axis=0).astype(np.float32, copy=False)
        masks = np.stack([s.mask for s in picks], axis=0).astype(np.bool_, copy=False)
        actions = np.asarray([s.action_target for s in picks], dtype=np.int64)
        values = np.asarray([s.value_target for s in picks], dtype=np.float32)
        policies = np.stack([s.policy_target for s in picks], axis=0).astype(np.float32, copy=False)
        batch = {
            "state": torch.as_tensor(states, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(masks, dtype=torch.bool, device=device),
            "action_target": torch.as_tensor(actions, dtype=torch.long, device=device),
            "value_target": torch.as_tensor(values, dtype=torch.float32, device=device),
            "policy_target": torch.as_tensor(policies, dtype=torch.float32, device=device),
        }
        if not torch.isfinite(batch["state"]).all():
            raise ValueError("Non-finite values in replay batch state")
        return batch

    def save_npz(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self._samples)
        states = np.zeros((n, STATE_DIM), dtype=np.float32)
        masks = np.zeros((n, ACTION_DIM), dtype=np.bool_)
        actions = np.zeros((n,), dtype=np.int64)
        values = np.zeros((n,), dtype=np.float32)
        policies = np.zeros((n, ACTION_DIM), dtype=np.float32)
        for i, sample in enumerate(self._samples):
            states[i] = sample.state
            masks[i] = sample.mask
            actions[i] = int(sample.action_target)
            values[i] = float(sample.value_target)
            if sample.policy_target is None:
                policy = np.zeros((ACTION_DIM,), dtype=np.float32)
                policy[int(sample.action_target)] = 1.0
                policies[i] = policy
            else:
                policies[i] = sample.policy_target

        metadata = {
            "max_size": self._max_size,
            "count": int(n),
        }
        np.savez_compressed(
            out_path,
            metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
            state=states,
            mask=masks,
            action_target=actions,
            value_target=values,
            policy_target=policies,
        )
        return out_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "ReplayBuffer":
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Replay buffer file not found: {in_path}")
        with np.load(in_path, allow_pickle=False) as data:
            metadata_raw = data["metadata_json"]
            if metadata_raw.ndim == 0:
                metadata_json = str(metadata_raw.item())
            else:
                metadata_json = str(metadata_raw.tolist())
            metadata = json.loads(metadata_json)
            max_size_raw = metadata.get("max_size", None)
            max_size = int(max_size_raw) if max_size_raw is not None else None

            states = np.asarray(data["state"], dtype=np.float32)
            masks = np.asarray(data["mask"], dtype=np.bool_)
            actions = np.asarray(data["action_target"], dtype=np.int64)
            values = np.asarray(data["value_target"], dtype=np.float32)
            policies = np.asarray(data["policy_target"], dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"Invalid replay state shape {states.shape}")
        if masks.ndim != 2 or masks.shape[1] != ACTION_DIM:
            raise ValueError(f"Invalid replay mask shape {masks.shape}")
        if policies.ndim != 2 or policies.shape[1] != ACTION_DIM:
            raise ValueError(f"Invalid replay policy shape {policies.shape}")
        n = int(states.shape[0])
        if masks.shape[0] != n or actions.shape[0] != n or values.shape[0] != n or policies.shape[0] != n:
            raise ValueError("Replay arrays have mismatched leading dimensions")

        out = cls(max_size=max_size)
        for i in range(n):
            out.add(
                ReplaySample(
                    state=states[i].copy(),
                    mask=masks[i].copy(),
                    action_target=int(actions[i]),
                    value_target=float(values[i]),
                    policy_target=policies[i].copy(),
                )
            )
        return out
