from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .state_schema import ACTION_DIM, STATE_DIM

_NATIVE_MODULE: Any | None = None
_NATIVE_MODULE_LOAD_ATTEMPTED = False
_NATIVE_OPT_WARNING_EMITTED = False
_REQUIRED_NATIVE_ENV_METHODS = (
    "reset",
    "get_state",
    "step",
    "run_mcts",
    "heuristic_action",
)


@dataclass
class StepState:
    state: np.ndarray  # normalized float32, shape (246,)
    mask: np.ndarray  # bool, shape (69,)
    is_terminal: bool
    winner: int
    current_player_id: int = 0


def _try_load_native_module() -> Any | None:
    global _NATIVE_MODULE, _NATIVE_MODULE_LOAD_ATTEMPTED
    if _NATIVE_MODULE_LOAD_ATTEMPTED:
        return _NATIVE_MODULE
    _NATIVE_MODULE_LOAD_ATTEMPTED = True

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"

    # Try to load the compiled native module from the build directory.
    # Use recursive search so multi-config generators (e.g. Visual Studio on Windows)
    # can load from build/Release or build/Debug without extra copying.
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
                if _is_native_module_compatible(module):
                    _NATIVE_MODULE = module
                    return _NATIVE_MODULE
                sys.modules.pop("splendor_native", None)

    try:
        module = importlib.import_module("splendor_native")
    except Exception:
        return None
    if _is_native_module_compatible(module):
        _NATIVE_MODULE = module
        return _NATIVE_MODULE

    return None


def _is_native_module_compatible(module: Any) -> bool:
    env_cls = getattr(module, "NativeEnv", None)
    if env_cls is None:
        return False
    for method_name in _REQUIRED_NATIVE_ENV_METHODS:
        if not hasattr(env_cls, method_name):
            return False
    return True


def _native_optimization_warning(native: Any) -> str | None:
    optimized = getattr(native, "BUILD_OPTIMIZED", None)
    if optimized in (None, True, 1):
        return None
    build_type = str(getattr(native, "BUILD_TYPE", "unknown"))
    return (
        "splendor_native appears to be built without optimization "
        f"(build_type={build_type!r}). Rebuild with "
        "`-DCMAKE_BUILD_TYPE=RelWithDebInfo` or `-DCMAKE_BUILD_TYPE=Release` "
        "for speed-critical runs."
    )


def _warn_if_unoptimized_native_module(native: Any) -> None:
    global _NATIVE_OPT_WARNING_EMITTED
    if _NATIVE_OPT_WARNING_EMITTED:
        return
    msg = _native_optimization_warning(native)
    if msg is None:
        return
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _NATIVE_OPT_WARNING_EMITTED = True


class SplendorNativeEnv:
    """Python adapter over the native pybind11 Splendor environment."""

    def __init__(self) -> None:
        native = _try_load_native_module()
        if native is None:
            raise ImportError(
                "splendor_native module not available. Build it (e.g. `cmake --build build --target splendor_native`)."
            )
        _warn_if_unoptimized_native_module(native)
        self._native = native
        env_cls = getattr(native, "NativeEnv", None)
        if env_cls is None:
            raise RuntimeError("splendor_native module missing NativeEnv class")
        self._env = env_cls()
        self._closed = False
        self._initialized = False
        self._current_player_id = 0

    @property
    def current_player_id(self) -> int:
        return self._current_player_id

    # Convert the raw result from c++ into a clean Python StepState dataclass.
    def _to_step_state(self, result: Any) -> StepState:
        state = np.asarray(result.state, dtype=np.float32)
        mask = np.asarray(result.mask, dtype=np.bool_)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected native state shape {state.shape}")
        if mask.shape != (ACTION_DIM,):
            raise RuntimeError(f"Unexpected native mask shape {mask.shape}")
        step = StepState(
            state=state,
            mask=mask,
            is_terminal=bool(result.is_terminal),
            winner=int(result.winner),
            current_player_id=int(result.current_player_id),
        )
        self._current_player_id = step.current_player_id
        return step

    def reset(self, seed: int = 0) -> StepState:
        result = self._env.reset(int(seed))
        self._initialized = True
        return self._to_step_state(result)

    def get_state(self) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._to_step_state(self._env.get_state())

    def step(self, action_idx: int) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._to_step_state(self._env.step(int(action_idx)))

    def heuristic_action(self) -> int:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return int(self._env.heuristic_action())

    def debug_raw_state(self) -> np.ndarray:
        """Test/debug helper exposing the native pre-normalized state vector."""
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        raw = np.asarray(self._env.debug_raw_state())
        if raw.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected native raw state shape {raw.shape}")
        return raw

    def run_mcts_native(self, evaluator, **kwargs):
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._env.run_mcts(evaluator, **kwargs)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "SplendorNativeEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def list_standard_cards() -> list[dict[str, Any]]:
    native = _try_load_native_module()
    if native is None:
        raise ImportError(
            "splendor_native module not available. Build it (e.g. `cmake --build build --target splendor_native`)."
        )
    fn = getattr(native, "list_standard_cards", None)
    if fn is None:
        raise RuntimeError(
            "splendor_native is missing list_standard_cards(). Rebuild the extension so web catalog APIs are available."
        )
    return [dict(item) for item in fn()]


def list_standard_nobles() -> list[dict[str, Any]]:
    native = _try_load_native_module()
    if native is None:
        raise ImportError(
            "splendor_native module not available. Build it (e.g. `cmake --build build --target splendor_native`)."
        )
    fn = getattr(native, "list_standard_nobles", None)
    if fn is None:
        raise RuntimeError(
            "splendor_native is missing list_standard_nobles(). Rebuild the extension so web catalog APIs are available."
        )
    return [dict(item) for item in fn()]

__all__ = [
    "StepState",
    "SplendorNativeEnv",
    "list_standard_cards",
    "list_standard_nobles",
]
