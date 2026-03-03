from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark_native_env import benchmark_native_env
from .train import run_smoke


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"All list values must be positive: {raw!r}")
        out.append(value)
    if not out:
        raise ValueError(f"Expected at least one value in list: {raw!r}")
    return out


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("nn_artifacts") / f"bench_{stamp}"


def _run_native_rows(*, sims_list: list[int], seed: int) -> list[dict[str, Any]]:
    # Warm up imports/kernel dispatch so the first measured row is not a one-time outlier.
    benchmark_native_env(
        seed=int(seed),
        reset_iterations=1,
        step_iterations=1,
        warmup_steps=8,
        include_mcts=True,
        mcts_num_simulations=int(sims_list[0]),
        mcts_turns_taken=0,
        mcts_device="cpu",
    )

    rows: list[dict[str, Any]] = []
    for sims in sims_list:
        result = benchmark_native_env(
            seed=int(seed),
            reset_iterations=500,
            step_iterations=5000,
            warmup_steps=256,
            include_mcts=True,
            mcts_num_simulations=int(sims),
            mcts_turns_taken=0,
            mcts_device="cpu",
        )
        row = asdict(result)
        row.update(
            {
                "family": "native_env",
                "case_id": f"native_mcts_{int(sims)}",
                "seed": int(seed),
            }
        )
        rows.append(row)
    return rows


def _run_smoke_rows(
    *,
    sims_list: list[int],
    seed: int,
    smoke_episodes: int,
    smoke_max_turns: int,
    smoke_batch_size: int,
    smoke_train_steps: int,
) -> list[dict[str, Any]]:
    run_smoke(
        episodes=1,
        max_turns=max(1, min(int(smoke_max_turns), 6)),
        batch_size=max(1, int(smoke_batch_size)),
        train_steps=1,
        log_every=1,
        seed=int(seed),
        device="cpu",
        mcts_sims=int(sims_list[0]),
    )

    rows: list[dict[str, Any]] = []
    for sims in sims_list:
        t0 = time.perf_counter()
        metrics = run_smoke(
            episodes=int(smoke_episodes),
            max_turns=int(smoke_max_turns),
            batch_size=int(smoke_batch_size),
            train_steps=int(smoke_train_steps),
            log_every=max(1, int(smoke_train_steps)),
            seed=int(seed),
            device="cpu",
            mcts_sims=int(sims),
        )
        wall = time.perf_counter() - t0
        rows.append(
            {
                "family": "smoke",
                "case_id": f"smoke_mcts_{int(sims)}",
                "seed": int(seed),
                "mcts_sims": int(sims),
                "wall_time_sec": float(wall),
                "mcts_avg_search_ms": float(metrics["mcts_avg_search_ms"]),
                "collector_mcts_actions": float(metrics["collector_mcts_actions"]),
                "replay_samples": float(metrics["replay_samples"]),
                "total_steps": float(metrics["total_steps"]),
                "total_turns": float(metrics["total_turns"]),
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
            }
        )
    return rows


def _write_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            vals = [str(row.get(col, "")) for col in cols]
            f.write("\t".join(vals) + "\n")


def _rows_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id:
            out[case_id] = row
    return out


def _compare_with_baseline(
    *,
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _rows_by_case(baseline_rows)
    cur = _rows_by_case(current_rows)
    compare: dict[str, Any] = {}

    for case_id, cur_row in cur.items():
        base_row = base.get(case_id)
        if base_row is None:
            continue
        if case_id.startswith("native_mcts_"):
            old = float(base_row.get("step_ops_per_sec", 0.0))
            new = float(cur_row.get("step_ops_per_sec", 0.0))
            if old > 0.0:
                compare[f"{case_id}_step_ops_speedup"] = new / old
        if case_id.startswith("smoke_mcts_"):
            old_ms = float(base_row.get("mcts_avg_search_ms", 0.0))
            new_ms = float(cur_row.get("mcts_avg_search_ms", 0.0))
            if old_ms > 0.0 and new_ms > 0.0:
                compare[f"{case_id}_mcts_latency_speedup"] = old_ms / new_ms
    return compare


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a local speed benchmark matrix and save JSON/TSV artifacts.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--native-sims", type=str, default="64,256")
    p.add_argument("--smoke-sims", type=str, default="64,128")
    p.add_argument("--smoke-episodes", type=int, default=3)
    p.add_argument("--smoke-max-turns", type=int, default=30)
    p.add_argument("--smoke-batch-size", type=int, default=32)
    p.add_argument("--smoke-train-steps", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--baseline-json", type=str, default="")
    p.add_argument("--require-thresholds", action="store_true")
    p.add_argument("--require-step-speedup", type=float, default=4.0)
    p.add_argument("--require-mcts-latency-speedup", type=float, default=3.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    native_sims = _parse_int_list(args.native_sims)
    smoke_sims = _parse_int_list(args.smoke_sims)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(_run_native_rows(sims_list=native_sims, seed=int(args.seed)))
    rows.extend(
        _run_smoke_rows(
            sims_list=smoke_sims,
            seed=int(args.seed),
            smoke_episodes=int(args.smoke_episodes),
            smoke_max_turns=int(args.smoke_max_turns),
            smoke_batch_size=int(args.smoke_batch_size),
            smoke_train_steps=int(args.smoke_train_steps),
        )
    )

    summary: dict[str, Any] = {
        "generated_at": _now_utc(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "rows": rows,
    }

    compare: dict[str, Any] = {}
    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        with baseline_path.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_rows = list(baseline.get("rows", []))
        compare = _compare_with_baseline(baseline_rows=baseline_rows, current_rows=rows)
        summary["baseline_json"] = str(baseline_path.resolve())
        summary["comparison"] = compare

        if args.require_thresholds:
            step_key = "native_mcts_256_step_ops_speedup"
            lat_key = "smoke_mcts_128_mcts_latency_speedup"
            step_speedup = float(compare.get(step_key, 0.0))
            latency_speedup = float(compare.get(lat_key, 0.0))
            if step_speedup < float(args.require_step_speedup):
                print(
                    f"threshold_failed key={step_key} "
                    f"value={step_speedup:.3f} min_required={float(args.require_step_speedup):.3f}"
                )
                return 2
            if latency_speedup < float(args.require_mcts_latency_speedup):
                print(
                    f"threshold_failed key={lat_key} "
                    f"value={latency_speedup:.3f} min_required={float(args.require_mcts_latency_speedup):.3f}"
                )
                return 2

    summary_path = out_dir / "summary.json"
    tsv_path = out_dir / "matrix.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _write_tsv(rows, tsv_path)

    print(f"perf_matrix_saved summary={summary_path.resolve()} tsv={tsv_path.resolve()}")
    if compare:
        print(json.dumps(compare, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
