from __future__ import annotations

import argparse
import asyncio

from .runner import SpendeeBridgeConfig, SpendeeBridgeRunner

DEFAULT_SPENDEE_URL = "https://spendee.mattle.online/lobby/rooms"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Spendee bridge against the live Meteor client state.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint (.pt)")
    parser.add_argument("--user-data-dir", required=True, help="Persistent Chromium profile directory for Playwright")
    parser.add_argument(
        "--player-seat",
        choices=("P0", "P1", "auto"),
        default="auto",
        help="Seat the bot should control; default is auto-detect from Spendee",
    )
    parser.add_argument("--start-url", default=DEFAULT_SPENDEE_URL, help="Spendee page to open")
    parser.add_argument(
        "--search-type",
        choices=("mcts", "ismcts"),
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
    parser.add_argument("--poll-interval-sec", type=float, default=0.5, help="Board polling interval")
    parser.add_argument("--stable-polls", type=int, default=2, help="Matching board snapshots required before acting")
    parser.add_argument("--artifact-dir", default="nn_artifacts/spendee_bridge", help="Directory for bridge logs/artifacts")
    parser.add_argument("--live", action="store_true", help="Actually submit actions to Spendee")
    parser.add_argument("--observe-only", action="store_true", help="Never think or act; just observe and log")
    parser.add_argument(
        "--auto-manage-rooms",
        action="store_true",
        help="Automatically create/start/return rooms from the lobby when no active game is detected",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> None:
    config = SpendeeBridgeConfig(
        start_url=str(args.start_url),
        user_data_dir=str(args.user_data_dir),
        checkpoint_path=str(args.checkpoint),
        player_seat=None if str(args.player_seat) == "auto" else str(args.player_seat),
        search_type=str(args.search_type),
        num_simulations=int(args.num_simulations),
        determinization_samples=int(args.determinization_samples),
        poll_interval_sec=float(args.poll_interval_sec),
        stable_polls=int(args.stable_polls),
        dry_run=not bool(args.live),
        observe_only=bool(args.observe_only),
        auto_manage_rooms=bool(args.auto_manage_rooms),
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
