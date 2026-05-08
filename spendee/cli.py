from __future__ import annotations

import argparse
import asyncio

from .runner import SpendeeBridgeConfig, SpendeeBridgeRunner

DEFAULT_SPENDEE_URL = "https://spendee.mattle.online/lobby/rooms"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Spendee bridge against the live Meteor client state.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint (.pt)")
    parser.add_argument(
        "--user-data-dir",
        default=None,
        help="Persistent Chromium profile directory for local Playwright launches; required unless a remote endpoint is used",
    )
    parser.add_argument(
        "--remote-ws-url",
        default=None,
        help="Playwright browser server websocket endpoint, for example ws://WINDOWS_IP:8888/splendor-bridge",
    )
    parser.add_argument(
        "--remote-cdp-url",
        default=None,
        help="Chrome DevTools endpoint, for example http://WINDOWS_IP:9222",
    )
    parser.add_argument(
        "--player-seat",
        choices=("P0", "P1", "auto"),
        default="auto",
        help="Seat the bot should control; default is auto-detect from Spendee",
    )
    parser.add_argument("--start-url", default=DEFAULT_SPENDEE_URL, help="Spendee page to open")
    parser.add_argument(
        "--search-type",
        choices=("mcts", "mcts_bootstrap", "ismcts", "alphabeta", "forced_child"),
        default="mcts",
        help="Search backend to use for move selection",
    )
    parser.add_argument("--num-simulations", type=int, default=5000, help="MCTS simulations per move")
    parser.add_argument(
        "--determinization-samples",
        type=int,
        default=1,
        help="Deprecated compatibility flag; native MCTS now re-determinizes hidden info per simulation",
    )
    parser.add_argument(
        "--gpu-batching-enabled",
        action="store_true",
        help="Enable GPU batching for MCTS/ISMCTS (eval_batch_size=32 instead of 1)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Explicit leaf evaluation batch size for MCTS/ISMCTS; overrides --gpu-batching-enabled when set",
    )
    parser.add_argument(
        "--bootstrap-simulations-per-action",
        type=int,
        default=0,
        help="For --search-type mcts_bootstrap: first-phase MCTS simulations allocated to each legal root action",
    )
    # --- Alpha-Beta options (only used when --search-type alphabeta) ---
    parser.add_argument(
        "--alphabeta-depth",
        type=int,
        default=3,
        help="Search depth in plies for alpha-beta (default: 3)",
    )
    parser.add_argument(
        "--alphabeta-chance-samples",
        type=int,
        default=4,
        help="Deck-draw outcomes to sample at chance nodes during alpha-beta (default: 4; set to 1 to disable)",
    )
    # --- Forced-child options (only used when --search-type forced_child) ---
    parser.add_argument(
        "--forced-child-simulations",
        type=int,
        default=2000,
        help="MCTS simulations per child node for forced_child search (default: 2000)",
    )
    parser.add_argument(
        "--forced-child-c-puct",
        type=float,
        default=1.25,
        help="c_puct for forced_child MCTS (default: 1.25)",
    )
    parser.add_argument("--poll-interval-sec", type=float, default=0.5, help="Board polling interval")
    parser.add_argument("--stable-polls", type=int, default=2, help="Matching board snapshots required before acting")
    parser.add_argument("--artifact-dir", default="nn_artifacts/spendee_bridge", help="Directory for bridge logs/artifacts")
    parser.add_argument(
        "--mode",
        choices=("play", "dry-run", "record-only"),
        default=None,
        help="Bridge mode: play submits engine moves, dry-run thinks without submitting, record-only just records the live Spendee game",
    )
    parser.add_argument("--live", action="store_true", help="Actually submit actions to Spendee")
    parser.add_argument("--observe-only", action="store_true", help="Never think or act; just observe and log")
    parser.add_argument(
        "--auto-manage-rooms",
        action="store_true",
        help="Automatically create/start/return rooms from the lobby when no active game is detected",
    )
    parser.add_argument(
        "--min-opponent-rating",
        type=int,
        default=1980,
        help="When auto-managing rooms, kick players below this rating and start only when all joined opponents meet it",
    )
    parser.add_argument(
        "--relative-rating-gap",
        type=int,
        default=150,
        help="If set, require opponent rating >= (your current rating - this gap); overrides --min-opponent-rating when your rating is available",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="Use a fixed minimum opponent rating gate; when set, overrides relative gap mode",
    )
    parser.add_argument(
        "--accept-user",
        default=None,
        help=(
            "Only accept/start against an opponent whose username ends with this exact suffix; "
            "non-matching or unknown usernames are kicked"
        ),
    )
    return parser


def _resolve_bridge_mode(args: argparse.Namespace) -> tuple[bool, bool]:
    selected_mode = str(args.mode) if args.mode is not None else None
    if selected_mode == "play":
        return False, False
    if selected_mode == "dry-run":
        return True, False
    if selected_mode == "record-only":
        return True, True
    if bool(args.observe_only):
        return True, True
    if bool(args.live):
        return False, False
    return True, False


async def _run_async(args: argparse.Namespace) -> None:
    dry_run, observe_only = _resolve_bridge_mode(args)
    eval_batch_size = None if args.eval_batch_size is None else int(args.eval_batch_size)
    if eval_batch_size is not None and eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.remote_ws_url and args.remote_cdp_url:
        raise ValueError("Choose only one of --remote-ws-url or --remote-cdp-url")
    if not args.user_data_dir and not args.remote_ws_url and not args.remote_cdp_url:
        raise ValueError("--user-data-dir is required unless --remote-ws-url or --remote-cdp-url is set")
    config = SpendeeBridgeConfig(
        start_url=str(args.start_url),
        user_data_dir=(None if args.user_data_dir is None else str(args.user_data_dir)),
        checkpoint_path=str(args.checkpoint),
        remote_ws_url=(None if args.remote_ws_url is None else str(args.remote_ws_url)),
        remote_cdp_url=(None if args.remote_cdp_url is None else str(args.remote_cdp_url)),
        player_seat=None if str(args.player_seat) == "auto" else str(args.player_seat),
        search_type=str(args.search_type),
        num_simulations=int(args.num_simulations),
        bootstrap_simulations_per_action=int(args.bootstrap_simulations_per_action),
        determinization_samples=int(args.determinization_samples),
        gpu_batching_enabled=bool(args.gpu_batching_enabled),
        eval_batch_size=eval_batch_size,
        alphabeta_depth=int(args.alphabeta_depth),
        alphabeta_chance_samples=int(args.alphabeta_chance_samples),
        forced_child_simulations=int(args.forced_child_simulations),
        forced_child_c_puct=float(args.forced_child_c_puct),
        poll_interval_sec=float(args.poll_interval_sec),
        stable_polls=int(args.stable_polls),
        dry_run=dry_run,
        observe_only=observe_only,
        auto_manage_rooms=bool(args.auto_manage_rooms),
        min_opponent_rating=int(args.min_opponent_rating),
        relative_rating_gap=(None if args.relative_rating_gap is None else int(args.relative_rating_gap)),
        min_rating=(None if args.min_rating is None else int(args.min_rating)),
        accept_user_suffix=(None if args.accept_user is None else str(args.accept_user)),
        artifact_dir=str(args.artifact_dir),
    )
    runner = SpendeeBridgeRunner(config)
    await runner.run()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
