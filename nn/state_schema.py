from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

# Canonical state/action schema shared across Python runtime code and tests.
# Prefer the native layout exported by splendor_native so Python and C++ stay in sync.


def _load_native_layout() -> dict[str, int] | None:
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"
    patterns = ("splendor_native*.so", "splendor_native*.pyd", "splendor_native*.dylib")

    if build_dir.exists():
        for pattern in patterns:
            for candidate in sorted(build_dir.rglob(pattern)):
                spec = importlib.util.spec_from_file_location("splendor_native", candidate)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["splendor_native"] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop("splendor_native", None)
                    continue
                layout = getattr(module, "STATE_LAYOUT", None)
                if isinstance(layout, dict):
                    return {str(key): int(value) for key, value in dict(layout).items()}
                sys.modules.pop("splendor_native", None)

    try:
        module = importlib.import_module("splendor_native")
    except Exception:
        return None
    layout = getattr(module, "STATE_LAYOUT", None)
    if not isinstance(layout, dict):
        return None
    return {str(key): int(value) for key, value in dict(layout).items()}


_FALLBACK_LAYOUT = {
    "CARD_FEATURE_LEN": 11,
    "CP_TOKENS_START": 0,
    "CP_BONUSES_START": 6,
    "CP_POINTS_IDX": 11,
    "CP_RESERVED_START": 12,
    "OP_TOKENS_START": 45,
    "OP_BONUSES_START": 51,
    "OP_POINTS_IDX": 56,
    "PLAYER_INDEX_IDX": 57,
    "OPPONENT_RESERVED_SLOT_LEN": 13,
    "OP_RESERVED_START": 58,
    "OP_RESERVED_IS_OCCUPIED_OFFSET": 11,
    "OP_RESERVED_TIER_OFFSET": 12,
    "FACEUP_START": 97,
    "BANK_START": 229,
    "NOBLES_START": 235,
    "PHASE_FLAGS_START": 250,
}

_LAYOUT = _load_native_layout() or _FALLBACK_LAYOUT

STATE_DIM = int(_LAYOUT["PHASE_FLAGS_START"]) + 2
ACTION_DIM = 69
CARD_FEATURE_LEN = int(_LAYOUT["CARD_FEATURE_LEN"])
OPPONENT_RESERVED_SLOT_LEN = int(_LAYOUT["OPPONENT_RESERVED_SLOT_LEN"])

CP_TOKENS_START = int(_LAYOUT["CP_TOKENS_START"])
CP_BONUSES_START = int(_LAYOUT["CP_BONUSES_START"])
CP_POINTS_IDX = int(_LAYOUT["CP_POINTS_IDX"])
CP_RESERVED_START = int(_LAYOUT["CP_RESERVED_START"])
OP_TOKENS_START = int(_LAYOUT["OP_TOKENS_START"])
OP_BONUSES_START = int(_LAYOUT["OP_BONUSES_START"])
OP_POINTS_IDX = int(_LAYOUT["OP_POINTS_IDX"])
PLAYER_INDEX_IDX = int(_LAYOUT["PLAYER_INDEX_IDX"])
OP_RESERVED_START = int(_LAYOUT["OP_RESERVED_START"])
OP_RESERVED_IS_OCCUPIED_OFFSET = int(_LAYOUT["OP_RESERVED_IS_OCCUPIED_OFFSET"])
OP_RESERVED_TIER_OFFSET = int(_LAYOUT["OP_RESERVED_TIER_OFFSET"])
FACEUP_START = int(_LAYOUT["FACEUP_START"])
BANK_START = int(_LAYOUT["BANK_START"])
NOBLES_START = int(_LAYOUT["NOBLES_START"])
PHASE_FLAGS_START = int(_LAYOUT["PHASE_FLAGS_START"])
