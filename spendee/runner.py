from __future__ import annotations

import asyncio
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from nn.mcts import MCTSConfig

from .catalog import SpendeeCatalog
from .engine_policy import AlphaBetaConfig, DeterminizedMCTSPolicy, ForcedChildSearchConfig
from .executor import SpendeeExecutor
from .logging import BridgeArtifactLogger
from .observer import ObservedBoardState, SpendeeObserver
from .selectors import SpendeeSelectorConfig
from .shadow_state import ShadowState
from .webui_save import build_webui_save_payload

LIVE_GAME_PAGE_PROBE = """
(() => {
  const href = typeof location !== "undefined" ? String(location.href || "") : "";
  const hasGames = typeof Games !== "undefined" && !!Games && typeof Games.findOne === "function";
  if (!hasGames) {
    return { href, hasGames: false, gameId: null, myPlayerIndex: null };
  }
  const game = Games.findOne();
  const userId =
    typeof Meteor !== "undefined" && Meteor && typeof Meteor.userId === "function"
      ? Meteor.userId()
      : null;
  const myPlayerIndex =
    game &&
    typeof game.playerIndexForUserId === "function" &&
    userId
      ? game.playerIndexForUserId(userId)
      : null;
  return {
    href,
    hasGames: true,
    gameId: game ? String(game._id || "") : null,
    myPlayerIndex,
  };
})()
"""

PAGE_SURFACE_PROBE = """
(() => ({
  href: typeof location !== "undefined" ? String(location.href || "") : "",
  title: typeof document !== "undefined" ? String(document.title || "") : "",
  bodyText: typeof document !== "undefined" && document.body ? String(document.body.innerText || "") : "",
}))()
"""


@dataclass
class SpendeeBridgeConfig:
    start_url: str
    user_data_dir: str
    checkpoint_path: str
    player_seat: str | None = None
    search_type: Literal["mcts", "ismcts", "alphabeta", "forced_child"] = "mcts"
    num_simulations: int = 5000
    determinization_samples: int = 1
    gpu_batching_enabled: bool = False
    alphabeta_depth: int = 3
    forced_child_simulations: int = 2000
    forced_child_c_puct: float = 1.25
    poll_interval_sec: float = 0.5
    stable_polls: int = 2
    stable_board_timeout_sec: float = 8.0
    stable_board_repair_threshold: int = 2
    action_delay_min_sec: float = 0.0
    action_delay_max_sec: float = 0.0
    action_settle_timeout_sec: float = 20.0
    action_retry_count: int = 1
    unsettled_action_repair_threshold: int = 2
    dry_run: bool = True
    observe_only: bool = False
    auto_manage_rooms: bool = False
    min_opponent_rating: int = 1980
    relative_rating_gap: int | None = 150
    min_rating: int | None = None
    selectors: SpendeeSelectorConfig = field(default_factory=SpendeeSelectorConfig)
    artifact_dir: str = "nn_artifacts/spendee_bridge"

    @property
    def bridge_mode(self) -> Literal["play", "dry_run", "record_only"]:
        if self.observe_only:
            return "record_only"
        if self.dry_run:
            return "dry_run"
        return "play"


def is_actionable_turn(observed: ObservedBoardState, player_seat: str | None) -> bool:
    if player_seat is None or observed.current_turn_seat != player_seat:
        return False
    if observed.modal_state.kind in {"choose_noble", "return_gem"}:
        return True
    return observed.current_job == "SPENDEE_REGULAR"


class SpendeeBridgeRunner:
    def __init__(self, config: SpendeeBridgeConfig) -> None:
        self.config = config
        self.catalog = SpendeeCatalog.load()
        self.observer = SpendeeObserver(self.catalog, selectors=config.selectors)
        self.shadow = ShadowState(self.catalog, player_seat=config.player_seat or "")
        self.executor = SpendeeExecutor()
        self.logger = BridgeArtifactLogger(Path(config.artifact_dir))
        self.policy = DeterminizedMCTSPolicy(
            checkpoint_path=config.checkpoint_path,
            mcts_config=MCTSConfig(
                num_simulations=config.num_simulations,
                temperature_moves=0,
                temperature=0.0,
                root_dirichlet_noise=False,
            ),
            determinization_samples=config.determinization_samples,
            search_type=config.search_type,
            gpu_batching_enabled=config.gpu_batching_enabled,
            alphabeta_config=AlphaBetaConfig(depth=config.alphabeta_depth),
            forced_child_config=ForcedChildSearchConfig(
                simulations_per_child=config.forced_child_simulations,
                c_puct=config.forced_child_c_puct,
                eval_batch_size=32 if config.gpu_batching_enabled else 1,
            ),
        )
        self._last_action_idx: int | None = None
        self._seat_verified = False
        self._seat_pinned = config.player_seat is not None
        self._player_seat: str | None = config.player_seat
        self._consecutive_stable_timeouts = 0
        self._consecutive_unsettled_actions = 0
        self._webui_save_history: list[dict] = []
        self._webui_save_last_key: tuple[object, ...] | None = None
        self._webui_save_game_id: str | None = None

    def _player_index_for_seat(self, seat: str) -> int:
        return 0 if seat == "P0" else 1

    def _resolve_hidden_reserved_cards(
        self,
        *,
        previous: ObservedBoardState,
        current: ObservedBoardState,
    ) -> list[tuple[str, int, int]]:
        resolved: list[tuple[str, int, int]] = []
        for seat in ("P0", "P1"):
            prev_slots = {slot.slot: slot for slot in previous.players[seat].reserved_slots}
            cur_slots = {slot.slot: slot for slot in current.players[seat].reserved_slots}
            prev_purchased = {card.card_id for card in previous.players[seat].purchased_cards}
            cur_purchased = {card.card_id for card in current.players[seat].purchased_cards}
            purchased_delta = sorted(cur_purchased - prev_purchased)
            for slot_idx, prev_slot in prev_slots.items():
                if prev_slot.state != "hidden":
                    continue
                cur_slot = cur_slots.get(slot_idx)
                if cur_slot is None:
                    continue
                if cur_slot.state == "visible" and cur_slot.card is not None:
                    resolved.append((seat, slot_idx, int(cur_slot.card.card_id)))
                    continue
                if cur_slot.state == "empty" and len(purchased_delta) == 1:
                    resolved.append((seat, slot_idx, int(purchased_delta[0])))
        return resolved

    def _patch_webui_history_hidden_reserved_card(
        self,
        *,
        seat: str,
        slot: int,
        card_id: int,
    ) -> None:
        player_index = self._player_index_for_seat(seat)
        for snapshot in self._webui_save_history:
            exported_state = snapshot.get("exported_state")
            if not isinstance(exported_state, dict):
                continue
            players = exported_state.get("players")
            if not isinstance(players, list) or player_index >= len(players):
                continue
            player_payload = players[player_index]
            if not isinstance(player_payload, dict):
                continue
            reserved = player_payload.get("reserved")
            if not isinstance(reserved, list):
                continue
            for item in reserved:
                if not isinstance(item, dict):
                    continue
                if int(item.get("slot", -1)) != int(slot):
                    continue
                if bool(item.get("is_public", True)):
                    continue
                item["card_id"] = int(card_id)
                break

    def _page_is_closed(self, page) -> bool:
        checker = getattr(page, "is_closed", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return True
        return False

    def _write_inactive_action_artifacts(
        self,
        *,
        reason: str,
        observed: ObservedBoardState | None = None,
    ) -> None:
        payload = {
            "status": "inactive",
            "reason": reason,
            "player_seat": self._player_seat,
            "seat_verified": self._seat_verified,
            "bridge_mode": self.config.bridge_mode,
        }
        if observed is not None:
            payload.update(
                {
                    "game_id": observed.game_id,
                    "board_version": observed.board_version,
                    "current_turn_seat": observed.current_turn_seat,
                    "current_job": observed.current_job,
                    "my_player_index": observed.my_player_index,
                }
            )
        self.logger.write_json("last_action_attempt", payload)
        self.logger.write_json("last_decision", payload)

    def _write_status(self, *, stage: str, observed: ObservedBoardState | None = None, extra: dict | None = None) -> None:
        payload = {
            "stage": stage,
            "player_seat": self._player_seat,
            "seat_verified": self._seat_verified,
            "last_action_idx": self._last_action_idx,
            "bridge_mode": self.config.bridge_mode,
        }
        if observed is not None:
            payload.update(
                {
                    "game_id": observed.game_id,
                    "board_version": observed.board_version,
                    "current_turn_seat": observed.current_turn_seat,
                    "current_job": observed.current_job,
                    "my_player_index": observed.my_player_index,
                }
            )
        if extra:
            payload.update(extra)
        self.logger.write_json("last_status", payload)

    def _webui_history_key(self, observed: ObservedBoardState) -> tuple[object, ...]:
        return (
            observed.game_id,
            observed.board_version,
            observed.current_turn_seat,
            observed.current_job,
            observed.modal_state.kind,
            observed.turns_count,
            observed.no_purchase_count,
            len(observed.raw_action_items),
        )

    def _write_webui_save(self) -> None:
        observed = self.shadow.last_observation
        if observed is None:
            return
        if self._webui_save_game_id != observed.game_id:
            self._archive_webui_save(reason="game_changed")
            self._webui_save_history = []
            self._webui_save_last_key = None
            self._webui_save_game_id = observed.game_id
        payload = build_webui_save_payload(
            self.shadow,
            checkpoint_path=self.config.checkpoint_path,
            num_simulations=self.config.num_simulations,
            player_seat=self._player_seat,
            turn_index=int(observed.turns_count),
            snapshots=list(self._webui_save_history),
            analysis_mode=True,
        )
        key = self._webui_history_key(observed)
        if self._webui_save_last_key == key:
            return
        snapshot = dict(payload["snapshots"][payload["current_index"]])
        self._webui_save_history.append(snapshot)
        self._webui_save_last_key = key
        self.logger.write_json("webui_save", payload)

    def _archive_webui_save(self, *, reason: str, observed: ObservedBoardState | None = None) -> Path | None:
        if self._webui_save_game_id is None:
            return None
        stamp_source = observed.observed_at if observed is not None else None
        if stamp_source:
            try:
                dt = datetime.fromisoformat(stamp_source.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        archive_name = dt.astimezone(timezone.utc).strftime("webui_save_%Y-%m-%d_%H-%M-%S")
        archived_path = self.logger.archive_json("webui_save", archive_name)
        self._webui_save_history = []
        self._webui_save_last_key = None
        self._webui_save_game_id = None
        if archived_path is not None:
            self.logger.write_json(
                "last_status",
                {
                    "stage": "archived_webui_save",
                    "reason": reason,
                    "archived_path": str(archived_path),
                    "player_seat": self._player_seat,
                    "seat_verified": self._seat_verified,
                    "last_action_idx": self._last_action_idx,
                },
            )
        return archived_path

    def _artifact_payload(self, payload: dict, *, observed: ObservedBoardState | None = None) -> dict:
        artifact = dict(payload)
        if observed is not None:
            artifact["observed"] = asdict(observed)
        return artifact

    def _retry_allowed(self, kind: str) -> bool:
        return kind not in {"choose_noble", "buy_reserved"}

    def _reset_transient_state(self) -> None:
        self._last_action_idx = None
        self.shadow.last_observation = None
        self.shadow.hidden_reserved_tiers.clear()
        self.shadow.action_history.clear()

    async def _recreate_engine_state(
        self,
        *,
        reason: str,
        observed: ObservedBoardState,
        extra: dict | None = None,
    ) -> None:
        details = {"reason": reason}
        if extra:
            details.update(extra)
        self._write_status(stage="recreating_state", observed=observed, extra=details)
        self.logger.write_json(
            "last_action_attempt",
            {
                "status": "recreating_state",
                "reason": reason,
                "player_seat": self._player_seat,
                "board_version": observed.board_version,
            },
        )
        self._last_action_idx = None
        self.shadow.recreate_from_observation(observed)
        self._consecutive_stable_timeouts = 0
        self._consecutive_unsettled_actions = 0

    async def _recover_from_stall(
        self,
        page,
        *,
        reason: str,
        observed: ObservedBoardState | None,
        extra: dict | None = None,
    ) -> None:
        if observed is not None and not self.config.observe_only and is_actionable_turn(observed, self._player_seat):
            await self._recreate_engine_state(reason=reason, observed=observed, extra=extra)
            return
        await self._repair_page(page, reason=reason, observed=observed, extra=extra)

    async def _repair_page(
        self,
        page,
        *,
        reason: str,
        observed: ObservedBoardState | None = None,
        extra: dict | None = None,
    ) -> None:
        details = {"reason": reason}
        if extra:
            details.update(extra)
        self._write_status(stage="repairing_page", observed=observed, extra=details)
        self.logger.write_json(
            "last_action_attempt",
            {
                "status": "repairing",
                "reason": reason,
                "player_seat": self._player_seat,
                "board_version": observed.board_version if observed is not None else None,
            },
        )
        try:
            await self.logger.capture_failure(
                page,
                f"repair_{reason}",
                self._artifact_payload(details, observed=observed),
            )
        except Exception:
            pass
        self._reset_transient_state()
        self._consecutive_stable_timeouts = 0
        self._consecutive_unsettled_actions = 0
        try:
            await page.reload(wait_until="domcontentloaded")
        except Exception:
            await page.goto(self.config.start_url, wait_until="domcontentloaded")

    def _surface_indicates_finished_room(self, surface: dict[str, str]) -> bool:
        href = str(surface.get("href") or "")
        body_text = str(surface.get("bodyText") or "").lower()
        if "/room/" not in href:
            return False
        markers = (
            "finished",
            "game over",
            "winner",
            "won the game",
            "return to lobby",
            "back to lobby",
            "play again",
        )
        return any(marker in body_text for marker in markers)

    def _noble_count(self, observed: ObservedBoardState, player_seat: str | None) -> int:
        if player_seat is None:
            return 0
        player = observed.players.get(player_seat)
        if player is None:
            return 0
        return len(player.claimed_nobles)

    def _visible_noble_count(self, observed: ObservedBoardState) -> int:
        return len(observed.visible_nobles)

    def _saw_pick_noble_action(self, observed: ObservedBoardState, *, player_index: int) -> bool:
        for item in observed.raw_action_items:
            action = item.get("action")
            if not isinstance(action, dict):
                continue
            if str(action.get("type")) != "pickNoble":
                continue
            if int(action.get("playerIndex", -1)) == player_index:
                return True
        return False

    def _noble_choice_closed(
        self,
        observed: ObservedBoardState,
        *,
        previous: ObservedBoardState,
    ) -> bool:
        return previous.modal_state.kind == "choose_noble" and observed.modal_state.kind != "choose_noble"

    def _return_gem_closed(
        self,
        observed: ObservedBoardState,
        *,
        previous: ObservedBoardState,
    ) -> bool:
        return previous.modal_state.kind == "return_gem" and observed.modal_state.kind != "return_gem"

    def _token_total(self, observed: ObservedBoardState, player_seat: str) -> int:
        player = observed.players.get(player_seat)
        if player is None:
            return 0
        return sum(int(player.tokens.get(color, 0)) for color in ("white", "blue", "green", "red", "black", "gold"))

    def _return_gem_progressed(
        self,
        observed: ObservedBoardState,
        *,
        previous: ObservedBoardState,
        action_idx: int,
    ) -> bool:
        if self._player_seat is None:
            return False
        previous_player = previous.players.get(self._player_seat)
        current_player = observed.players.get(self._player_seat)
        if previous_player is None or current_player is None:
            return False
        if self._token_total(observed, self._player_seat) < self._token_total(previous, self._player_seat):
            return True
        returned_color = ("white", "blue", "green", "red", "black")[action_idx - 61]
        if int(current_player.tokens.get(returned_color, 0)) < int(previous_player.tokens.get(returned_color, 0)):
            return True
        return observed.modal_state.options != previous.modal_state.options

    def _action_is_settled(
        self,
        observed: ObservedBoardState,
        *,
        previous: ObservedBoardState,
        action_idx: int,
    ) -> bool:
        if observed.board_version != previous.board_version:
            return True
        if len(observed.raw_action_items) > len(previous.raw_action_items):
            return True
        if self._player_seat is None:
            return False
        if action_idx in {61, 62, 63, 64, 65}:
            if self._return_gem_closed(observed, previous=previous):
                return True
            if self._return_gem_progressed(observed, previous=previous, action_idx=action_idx):
                return True
        if action_idx in {66, 67, 68}:
            previous_nobles = self._noble_count(previous, self._player_seat)
            current_nobles = self._noble_count(observed, self._player_seat)
            if self._noble_choice_closed(observed, previous=previous):
                return True
            if self._visible_noble_count(observed) < self._visible_noble_count(previous):
                return True
            player_index = observed.players[self._player_seat].spendee_player_index
            if current_nobles > previous_nobles:
                return True
            if observed.current_job != "SPENDEE_PICK_NOBLE":
                return True
            if self._saw_pick_noble_action(observed, player_index=player_index):
                return True
        return False

    def _planned_action_is_stale(
        self,
        observed: ObservedBoardState,
        *,
        previous: ObservedBoardState,
    ) -> bool:
        return (
            observed.board_version != previous.board_version
            or observed.current_turn_seat != previous.current_turn_seat
            or observed.current_job != previous.current_job
            or observed.modal_state.kind != previous.modal_state.kind
        )

    async def _wait_for_action_settlement(
        self,
        page,
        *,
        previous: ObservedBoardState,
        action_idx: int,
    ) -> ObservedBoardState | None:
        deadline = asyncio.get_running_loop().time() + self.config.action_settle_timeout_sec
        while asyncio.get_running_loop().time() < deadline:
            observed = await self.observer.observe(page)
            if observed is None:
                await asyncio.sleep(self.config.poll_interval_sec)
                continue
            if self._action_is_settled(observed, previous=previous, action_idx=action_idx):
                return observed
            await asyncio.sleep(self.config.poll_interval_sec)
        self._write_status(
            stage="waiting_for_action_settlement",
            observed=previous,
            extra={"action_idx": action_idx},
        )
        return None

    async def _wait_for_stable_board(self, page) -> tuple[ObservedBoardState | None, bool]:
        deadline = asyncio.get_running_loop().time() + self.config.stable_board_timeout_sec
        stable_count = 0
        last: ObservedBoardState | None = None
        while asyncio.get_running_loop().time() < deadline:
            observed = await self.observer.observe(page)
            if observed is None:
                stable_count = 0
                last = None
                await asyncio.sleep(self.config.poll_interval_sec)
                continue
            if observed.animations_active:
                stable_count = 0
            elif last is not None and observed.board_version == last.board_version:
                stable_count += 1
            else:
                stable_count = 1
            last = observed
            if stable_count >= self.config.stable_polls:
                return observed, False
            await asyncio.sleep(self.config.poll_interval_sec)
        if last is None:
            self._write_status(
                stage="stable_board_timeout",
                extra={"stable_polls": self.config.stable_polls, "observed": False},
            )
            return None, True
        self._write_status(
            stage="stable_board_timeout",
            observed=last,
            extra={"stable_polls": self.config.stable_polls, "observed": True},
        )
        return last, True

    async def _probe_live_game_page(self, page) -> dict | None:
        if self._page_is_closed(page):
            return None
        try:
            raw = await page.evaluate(LIVE_GAME_PAGE_PROBE)
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        href = str(raw.get("href") or "")
        has_games = bool(raw.get("hasGames"))
        game_id = raw.get("gameId")
        if not has_games or not game_id:
            return None
        return {
            "href": href,
            "game_id": str(game_id),
            "my_player_index": raw.get("myPlayerIndex"),
            "room_url": "/room/" in href,
        }

    async def _probe_page_surface(self, page) -> dict[str, str]:
        if self._page_is_closed(page):
            return {"href": "", "title": "", "bodyText": ""}
        try:
            raw = await page.evaluate(PAGE_SURFACE_PROBE)
        except Exception:
            return {"href": "", "title": "", "bodyText": ""}
        if not isinstance(raw, dict):
            return {"href": "", "title": "", "bodyText": ""}
        return {
            "href": str(raw.get("href") or ""),
            "title": str(raw.get("title") or ""),
            "bodyText": str(raw.get("bodyText") or ""),
        }

    async def _find_management_page(self, context):
        pages = [page for page in list(context.pages) if not self._page_is_closed(page)]
        lobby_candidates: list[tuple[object, dict[str, str]]] = []
        room_candidates: list[tuple[object, dict[str, str]]] = []
        for page in reversed(pages):
            try:
                surface = await self._probe_page_surface(page)
            except Exception:
                continue
            href = surface["href"]
            if "/room/" in href:
                room_candidates.append((page, surface))
            elif "/lobby/rooms" in href:
                lobby_candidates.append((page, surface))
        if room_candidates:
            return room_candidates[0][0]
        if lobby_candidates:
            return lobby_candidates[0][0]
        return pages[-1] if pages else None

    async def _click_first_button(self, page, labels: tuple[str, ...]) -> bool:
        for label in labels:
            patterns = (
                re.compile(rf"^{re.escape(label)}$", re.IGNORECASE),
                re.compile(re.escape(label), re.IGNORECASE),
            )
            for pattern in patterns:
                locator = page.get_by_role("button", name=pattern)
                try:
                    count = await locator.count()
                except Exception:
                    count = 0
                if count <= 0:
                    continue
                button = locator.first
                try:
                    if await button.is_visible() and await button.is_enabled():
                        await button.click()
                        return True
                except Exception:
                    continue
        return False

    async def _click_first_text(self, page, labels: tuple[str, ...]) -> bool:
        for label in labels:
            locator = page.get_by_text(re.compile(re.escape(label), re.IGNORECASE))
            try:
                count = await locator.count()
            except Exception:
                count = 0
            if count <= 0:
                continue
            node = locator.first
            try:
                if await node.is_visible():
                    await node.click()
                    return True
            except Exception:
                continue
        return False

    async def _return_to_lobby_if_finished_room(self, page) -> bool:
        surface = await self._probe_page_surface(page)
        if not self._surface_indicates_finished_room(surface):
            return False
        if await self._click_first_button(page, ("Return to Lobby", "Back to Lobby", "Lobby")):
            return True
        if await self._click_first_text(page, ("Back", "Return to Lobby", "Back to Lobby", "Lobby")):
            return True
        return False

    def _extract_rating_from_text(self, text: str) -> int | None:
        if not text:
            return None
        # Spendee ratings are typically 3-4 digit integers shown near player names.
        matches = re.findall(r"(?<!\d)(\d{3,4})(?!\d)", text)
        if not matches:
            return None
        try:
            return int(matches[0])
        except Exception:
            return None

    def _extract_engine_rating_from_surface(self, surface: dict[str, str]) -> int | None:
        body = str(surface.get("bodyText") or "")
        if not body:
            return None
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

        # Prefer lines that explicitly mention the local player.
        for ln in lines:
            lower = ln.lower()
            if "you" in lower:
                rating = self._extract_rating_from_text(ln)
                if rating is not None:
                    return rating

        # Fallback: any line that looks like a rating line.
        for ln in lines:
            lower = ln.lower()
            if "rating" in lower or "elo" in lower:
                rating = self._extract_rating_from_text(ln)
                if rating is not None:
                    return rating

        # Room pages often only show compact entries like "name (2070)" without a "You" tag.
        # In that case, use the maximum visible player-like rating as a best-effort engine estimate.
        candidates: list[int] = []
        for ln in lines:
            if "(" in ln and ")" in ln:
                rating = self._extract_rating_from_text(ln)
                if rating is not None:
                    candidates.append(rating)
        if candidates:
            return max(candidates)
        return None

    async def _kick_low_rated_players(self, page, surface: dict[str, str]) -> tuple[int, bool]:
        effective_min_rating = (
            int(self.config.min_rating)
            if self.config.min_rating is not None
            else int(self.config.min_opponent_rating)
        )
        kick_debug: dict[str, object] = {
            "href": str(surface.get("href") or ""),
            "title": str(surface.get("title") or ""),
            "meteor": None,
            "meteor_error": None,
            "min_opponent_rating": int(self.config.min_opponent_rating),
            "min_rating": (None if self.config.min_rating is None else int(self.config.min_rating)),
            "relative_rating_gap": (None if self.config.relative_rating_gap is None else int(self.config.relative_rating_gap)),
            "dom_kicked": 0,
            "dom_button_count": 0,
            "result": "init",
        }
        meteor_result: dict | None = None
        try:
            raw = await page.evaluate(
                r"""
                                async ({ minOpponentRating, relativeRatingGap, minRating }) => {
                                    const out = {
                                        kicked: 0,
                                        roomId: null,
                                        myRating: null,
                                        effectiveMinOpponentRating: (
                                            minRating == null ? Number(minOpponentRating) : Number(minRating)
                                        ),
                                        unknownCandidates: [],
                                        knownCandidates: [],
                                        participants: [],
                                        candidateCount: 0,
                                        eligibleCount: 0,
                                        blockingLowCount: 0,
                                        noParticipantList: false,
                                        attemptErrors: [],
                                        error: null,
                                    };
                                    try {
                  if (typeof Meteor === 'undefined' || !Meteor || typeof Meteor.call !== 'function') {
                    return out;
                  }

                  const me = typeof Meteor.userId === 'function' ? Meteor.userId() : null;
                  if (!me) {
                    return out;
                  }

                  const callAsync = (name, ...args) =>
                    new Promise((resolve) => {
                      try {
                        Meteor.call(name, ...args, (err, res) => resolve({ ok: !err, err: err ? String(err) : null, res }));
                      } catch (err) {
                        resolve({ ok: false, err: String(err) });
                      }
                    });

                  const parseRating = (v) => {
                    const n = Number.parseInt(String(v), 10);
                    return Number.isFinite(n) ? n : null;
                  };

                                    const nameToRating = (() => {
                                        const map = new Map();
                                        if (typeof document === 'undefined' || !document.body) {
                                            return map;
                                        }
                                        const lines = String(document.body.innerText || '')
                                            .split(/\r?\n/)
                                            .map((s) => String(s || '').trim())
                                            .filter(Boolean);
                                        for (const line of lines) {
                                            const m = line.match(/^(.+?)\s*\((\d{3,4})\)\b/);
                                            if (!m) {
                                                continue;
                                            }
                                            const name = String(m[1] || '').trim();
                                            const rating = parseRating(m[2]);
                                            if (name && rating != null) {
                                                map.set(name.toLowerCase(), rating);
                                            }
                                        }
                                        return map;
                                    })();

                                    const users = Meteor.users && typeof Meteor.users.findOne === 'function' ? Meteor.users : null;
                  const userById = (uid) => (users ? (users.findOne(uid) || users.findOne({ _id: uid })) : null);
                                    const usernameById = (uid) => {
                                        const u = userById(uid);
                                        if (!u) {
                                            return null;
                                        }
                                        return u.username || (u.profile && (u.profile.name || u.profile.username)) || null;
                                    };
                                    const ratingFromBodyByUsername = (username) => {
                                        if (!username || typeof document === 'undefined' || !document.body) {
                                            return null;
                                        }
                                        const text = String(document.body.innerText || '');
                                        const esc = String(username).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                                        const m = text.match(new RegExp(`${esc}[^\\n\\r]*?\\((\\d{3,4})\\)`, 'i'));
                                        return m ? parseRating(m[1]) : null;
                                    };
                  const userRating = (uid) => {
                    const u = userById(uid);
                                        if (u) {
                                            const r = u.rating ?? u.elo ?? u.mmr ?? (u.profile && (u.profile.rating ?? u.profile.elo ?? u.profile.mmr));
                                            const parsed = parseRating(r);
                                            if (parsed != null) {
                                                return parsed;
                                            }
                    }
                                        const byUsername = ratingFromBodyByUsername(usernameById(uid));
                                        if (byUsername != null) {
                                            return byUsername;
                                        }
                                        const uname = usernameById(uid);
                                        if (uname) {
                                            const byMap = nameToRating.get(String(uname).toLowerCase());
                                            if (byMap != null) {
                                                return byMap;
                                            }
                                        }
                                        return null;
                  };

                                    const myRating = userRating(me);
                                    out.myRating = myRating;
                                    const gap = (relativeRatingGap == null) ? null : Number(relativeRatingGap);
                                    if (minRating != null && Number.isFinite(Number(minRating))) {
                                        out.effectiveMinOpponentRating = Number(minRating);
                                    } else if (myRating != null && gap != null && Number.isFinite(gap)) {
                                        out.effectiveMinOpponentRating = Math.max(0, Math.floor(Number(myRating) - gap));
                                    } else {
                                        out.effectiveMinOpponentRating = Number(minOpponentRating);
                                    }
                                    const threshold = Number(out.effectiveMinOpponentRating);

                  const roomCollections = [];
                  if (typeof Rooms !== 'undefined' && Rooms && typeof Rooms.findOne === 'function') {
                    roomCollections.push(Rooms);
                  }
                  if (typeof BattleRooms !== 'undefined' && BattleRooms && typeof BattleRooms.findOne === 'function') {
                    roomCollections.push(BattleRooms);
                  }

                                    const roomContainsUser = (roomDoc, uid) => {
                                        if (!roomDoc || !uid) {
                                            return false;
                                        }
                                        const uidStr = String(uid);
                                        if (Array.isArray(roomDoc.userIds) && roomDoc.userIds.some((x) => String(x) === uidStr)) {
                                            return true;
                                        }
                                        if (Array.isArray(roomDoc.playerUserIds) && roomDoc.playerUserIds.some((x) => String(x) === uidStr)) {
                                            return true;
                                        }
                                        if (Array.isArray(roomDoc.players)) {
                                            for (const p of roomDoc.players) {
                                                if (!p) {
                                                    continue;
                                                }
                                                const puid = p.userId || p._id || p.id;
                                                if (puid && String(puid) === uidStr) {
                                                    return true;
                                                }
                                            }
                                        }
                                        if (Array.isArray(roomDoc.spots)) {
                                            for (const spot of roomDoc.spots) {
                                                const p = spot && spot.player ? spot.player : null;
                                                if (!p) {
                                                    continue;
                                                }
                                                const puid = p.userId || p._id || p.id;
                                                if (puid && String(puid) === uidStr) {
                                                    return true;
                                                }
                                            }
                                        }
                                        return false;
                                    };

                                    let room = null;
                                    for (const coll of roomCollections) {
                                        room = coll.findOne({ userIds: me }) || coll.findOne({ playerUserIds: me });
                                        if (!room && typeof coll.find === 'function') {
                                            const allRooms = coll.find({}).fetch();
                                            room = Array.isArray(allRooms)
                                                ? (allRooms.find((doc) => roomContainsUser(doc, me)) || null)
                                                : null;
                                        }
                                        if (room && roomContainsUser(room, me)) {
                                            break;
                                        }
                                        room = null;
                                    }
                  if (!room) {
                    return out;
                  }

                  out.roomId = room._id ? String(room._id) : null;

                  const ownerId = room.createdByUserId || room.ownerUserId || room.hostUserId || room.userId || null;
                  if (ownerId && String(ownerId) !== String(me)) {
                    return out;
                  }

                                    const ids = new Set();
                  const pushId = (v) => {
                    if (v) {
                      ids.add(String(v));
                    }
                  };

                                    const rememberAttemptError = (label, res) => {
                                        if (!res || res.ok) {
                                            return;
                                        }
                                        if (out.attemptErrors.length >= 20) {
                                            return;
                                        }
                                        out.attemptErrors.push({ label, err: String(res.err || 'unknown') });
                                    };

                                    const spotEntries = Array.isArray(room.spots) ? room.spots : [];
                                    const hasSpots = spotEntries.some((s) => s && s.player);

                                    for (let spotIndex = 0; spotIndex < spotEntries.length; spotIndex += 1) {
                                        const spot = spotEntries[spotIndex];
                                        const player = spot && spot.player ? spot.player : null;
                                        if (!player) {
                                            continue;
                                        }
                                        const uid = String(player.userId || player._id || player.id || '');
                                        const spotName = String(player.name || player.username || '').trim();
                                        const spotRating = parseRating(player.rating ?? player.elo ?? player.mmr);
                                        const fromMap = spotName ? nameToRating.get(spotName.toLowerCase()) : null;
                                        const resolvedRating = spotRating != null ? spotRating : (fromMap != null ? fromMap : userRating(uid));
                                        out.participants.push({
                                            spotIndex,
                                            uid,
                                            name: spotName,
                                            rating: resolvedRating,
                                            rawSpotRating: spotRating,
                                            mapRating: fromMap,
                                            state: spot && spot.state,
                                            isMe: uid === String(me),
                                        });
                                    }

                                    if (!hasSpots) {
                                        if (Array.isArray(room.userIds)) {
                                            room.userIds.forEach(pushId);
                                        }
                                        if (Array.isArray(room.playerUserIds)) {
                                            room.playerUserIds.forEach(pushId);
                                        }
                                        if (Array.isArray(room.players)) {
                                            for (const p of room.players) {
                                                if (p) {
                                                    pushId(p.userId || p._id || p.id);
                                                }
                                            }
                                        }

                                        for (const uid of Array.from(ids)) {
                                            if (uid === String(me)) {
                                                continue;
                                            }
                                            const r = userRating(uid);
                                            if (r == null) {
                                                out.unknownCandidates.push(uid);
                                            } else {
                                                out.knownCandidates.push({ uid, rating: r });
                                                if (r >= threshold) {
                                                    out.eligibleCount += 1;
                                                } else {
                                                    out.blockingLowCount += 1;
                                                }
                                            }
                                            const shouldKick = r == null || r < threshold;
                                            if (!shouldKick) {
                                                continue;
                                            }

                                            const attempts = [
                                                ["uid_only", () => callAsync('kickPlayerFromRoom', uid)],
                                                ["room_uid", () => callAsync('kickPlayerFromRoom', out.roomId, uid)],
                                                ["obj_userId", () => callAsync('kickPlayerFromRoom', { roomId: out.roomId, userId: uid })],
                                                ["obj_playerUserId", () => callAsync('kickPlayerFromRoom', { roomId: out.roomId, playerUserId: uid })],
                                            ];
                                            let kicked = false;
                                            for (const [label, attempt] of attempts) {
                                                const res = await attempt();
                                                if (res && res.ok) {
                                                    kicked = true;
                                                    break;
                                                }
                                                rememberAttemptError(label, res);
                                            }
                                            if (kicked) {
                                                out.kicked += 1;
                                            }
                                        }
                                    }

                                    if (!hasSpots && ids.size === 0) {
                                        out.noParticipantList = true;
                                    }
                                    out.candidateCount = hasSpots
                                        ? spotEntries.filter((s) => s && s.player && String((s.player.userId || s.player._id || s.player.id || '')) !== String(me)).length
                                        : Array.from(ids).filter((uid) => uid !== String(me)).length;

                                    // Schema-specific fallback for Spendee rooms: participants are stored in room.spots[*].player.
                                    // Kick by spot index when available.
                                    for (let spotIndex = 0; spotIndex < spotEntries.length; spotIndex += 1) {
                                        const spot = spotEntries[spotIndex];
                                        const player = spot && spot.player ? spot.player : null;
                                        if (!player) {
                                            continue;
                                        }
                                        const uid = String(player.userId || player._id || player.id || '');
                                        const spotName = String(player.name || player.username || '').trim();
                                        if (!uid || uid === String(me)) {
                                            continue;
                                        }

                                        const spotRating = parseRating(player.rating ?? player.elo ?? player.mmr);
                                        const fromMap = spotName ? nameToRating.get(spotName.toLowerCase()) : null;
                                        const resolvedRating = spotRating != null ? spotRating : (fromMap != null ? fromMap : userRating(uid));
                                        if (resolvedRating == null) {
                                            out.unknownCandidates.push(uid);
                                        } else {
                                            out.knownCandidates.push({ uid, rating: resolvedRating, source: 'spots' });
                                            if (resolvedRating >= threshold) {
                                                out.eligibleCount += 1;
                                            } else {
                                                out.blockingLowCount += 1;
                                            }
                                        }

                                        const shouldKick = resolvedRating == null || resolvedRating < threshold;
                                        if (!shouldKick) {
                                            continue;
                                        }

                                        const attemptsByIndex = [
                                            ["room_spotIndex", () => callAsync('kickPlayerFromRoom', out.roomId, spotIndex)],
                                            ["obj_spotIndex", () => callAsync('kickPlayerFromRoom', { roomId: out.roomId, spotIndex })],
                                            ["obj_index", () => callAsync('kickPlayerFromRoom', { roomId: out.roomId, index: spotIndex })],
                                            ["uid_spotIndex", () => callAsync('kickPlayerFromRoom', uid, spotIndex)],
                                            ["room_uid", () => callAsync('kickPlayerFromRoom', out.roomId, uid)],
                                            ["uid_only", () => callAsync('kickPlayerFromRoom', uid)],
                                            ["obj_playerUserId", () => callAsync('kickPlayerFromRoom', { roomId: out.roomId, playerUserId: uid })],
                                            ["obj_room_playerUserId", () => callAsync('kickPlayerFromRoom', out.roomId, { playerUserId: uid })],
                                            ["obj_userId", () => callAsync('kickPlayerFromRoom', { roomId: out.roomId, userId: uid })],
                                        ];
                                        let kicked = false;
                                        for (const [label, attempt] of attemptsByIndex) {
                                            const res = await attempt();
                                            if (res && res.ok) {
                                                kicked = true;
                                                break;
                                            }
                                            rememberAttemptError(label, res);
                                        }

                                        if (!kicked && room && typeof room.userKickAtIndex === 'function') {
                                            try {
                                                room.userKickAtIndex(spotIndex);
                                                kicked = true;
                                            } catch (_err) {
                                                // continue
                                            }
                                        }

                                        if (kicked) {
                                            out.kicked += 1;
                                        }
                                    }

                  return out;
                                    } catch (err) {
                                        out.error = String(err && (err.stack || err.message) || err);
                                        return out;
                                    }
                }
                """,
                {
                    "minOpponentRating": int(self.config.min_opponent_rating),
                    "relativeRatingGap": (None if self.config.relative_rating_gap is None else int(self.config.relative_rating_gap)),
                    "minRating": (None if self.config.min_rating is None else int(self.config.min_rating)),
                },
            )
            if isinstance(raw, dict):
                meteor_result = raw
        except Exception as exc:
            meteor_result = None
            kick_debug["meteor_error"] = str(exc)

        kick_debug["meteor"] = meteor_result
        if isinstance(meteor_result, dict) and meteor_result.get("error"):
            kick_debug["meteor_error"] = str(meteor_result.get("error"))
        if meteor_result is None:
            # Do not bail out here: Meteor probe can fail transiently on lobby pages,
            # and DOM-based kick heuristics can still succeed.
            kick_debug["result"] = "meteor_probe_failed_falling_back_to_dom"
        unknown_candidates_count = 0
        if isinstance(meteor_result, dict):
            effective_from_probe = meteor_result.get("effectiveMinOpponentRating")
            if isinstance(effective_from_probe, (int, float)):
                effective_min_rating = max(0, int(effective_from_probe))
                kick_debug["effective_min_opponent_rating"] = effective_min_rating
            my_rating = meteor_result.get("myRating")
            if isinstance(my_rating, (int, float)):
                kick_debug["my_rating"] = int(my_rating)
            unknown_candidates = meteor_result.get("unknownCandidates")
            if isinstance(unknown_candidates, list):
                unknown_candidates_count = len(unknown_candidates)
                kick_debug["unknown_candidates"] = unknown_candidates_count
            candidate_count = meteor_result.get("candidateCount")
            if isinstance(candidate_count, int):
                kick_debug["candidate_count"] = candidate_count
            no_participant_list = bool(meteor_result.get("noParticipantList"))
            kick_debug["no_participant_list"] = no_participant_list
        if meteor_result is not None:
            kicked = int(meteor_result.get("kicked") or 0)
            if kicked > 0:
                kick_debug["result"] = "kicked_via_meteor"
                self.logger.write_json("last_room_kick_debug", kick_debug)
                return kicked, False
            # If we can see kick-eligible candidates but all kick method attempts failed, never auto-start.
            known_candidates = meteor_result.get("knownCandidates") if isinstance(meteor_result, dict) else None
            if isinstance(known_candidates, list):
                low_count = 0
                for item in known_candidates:
                    if not isinstance(item, dict):
                        continue
                    rating = item.get("rating")
                    try:
                        rv = int(rating)
                    except Exception:
                        continue
                    if rv < int(effective_min_rating):
                        low_count += 1
                if low_count > 0:
                    kick_debug["low_rating_candidates"] = low_count
                    kick_debug["result"] = "kick_attempts_failed"
                    self.logger.write_json("last_room_kick_debug", kick_debug)
                    return 0, False
            if bool(meteor_result.get("noParticipantList")):
                kick_debug["result"] = "cannot_identify_room_participants"
                self.logger.write_json("last_room_kick_debug", kick_debug)
                return 0, False
            if unknown_candidates_count > 0:
                kick_debug["result"] = "unknown_rating_candidates_not_kicked"
                self.logger.write_json("last_room_kick_debug", kick_debug)
                return 0, False

            candidate_count = int(meteor_result.get("candidateCount") or 0)
            eligible_count = int(meteor_result.get("eligibleCount") or 0)
            blocking_low_count = int(meteor_result.get("blockingLowCount") or 0)
            can_start = candidate_count > 0 and eligible_count >= candidate_count and blocking_low_count == 0
            if can_start:
                kick_debug["result"] = "all_opponents_eligible"
                self.logger.write_json("last_room_kick_debug", kick_debug)
                return 0, True

        if self.config.min_rating is None and self.config.relative_rating_gap is not None:
            fallback_my_rating = self._extract_engine_rating_from_surface(surface)
            if fallback_my_rating is not None:
                effective_min_rating = max(0, int(fallback_my_rating) - int(self.config.relative_rating_gap))
                kick_debug["my_rating"] = int(fallback_my_rating)
                kick_debug["effective_min_opponent_rating"] = int(effective_min_rating)

        kicked_by_dom = 0
        try:
            kicked_by_dom = int(
                await page.evaluate(
                    """
                    ({ minOpponentRating }) => {
                      const rowSelector = 'tr,li,[role="row"],.player,.player-row,.room-player,.member,.participant,.row,.item';
                      const controlSelector = 'button,[role="button"],a,[title*="kick" i],[aria-label*="kick" i],[data-tooltip*="kick" i],[class*="kick" i],[title*="remove" i],[aria-label*="remove" i],[class*="remove" i]';
                      const ratingFromText = (text) => {
                        const raw = String(text || '');
                        const m = raw.match(/(?:\\(|\\b)(\\d{3,4})(?:\\)|\\b)/);
                        if (!m) {
                          return null;
                        }
                        const v = Number.parseInt(m[1], 10);
                        return Number.isFinite(v) ? v : null;
                      };
                      const isVisible = (el) => {
                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);
                        return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
                      };
                      const controls = Array.from(document.querySelectorAll(controlSelector)).filter((el) => {
                        if (!isVisible(el)) {
                          return false;
                        }
                        const label = [el.innerText, el.textContent, el.getAttribute('title'), el.getAttribute('aria-label'), el.getAttribute('data-tooltip'), el.className].join(' ').toLowerCase();
                        return /kick|remove|boot/.test(label);
                      });
                      let kicked = 0;
                      for (const el of controls) {
                        const row = el.closest(rowSelector) || el.parentElement || el;
                        const rating = ratingFromText(String((row && row.innerText) || ''));
                                                if (rating == null || rating >= Number(minOpponentRating)) {
                          continue;
                        }
                        try {
                          el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                          kicked += 1;
                        } catch (_err) {}
                      }
                      return kicked;
                    }
                    """,
                    {"minOpponentRating": int(effective_min_rating)},
                )
            )
        except Exception:
            kicked_by_dom = 0

        kick_debug["dom_kicked"] = kicked_by_dom
        if kicked_by_dom > 0:
            kick_debug["result"] = "kicked_via_dom"
            self.logger.write_json("last_room_kick_debug", kick_debug)
            return kicked_by_dom, False

        kick_locator = page.get_by_role("button", name=re.compile("kick|remove|boot", re.IGNORECASE))
        try:
            count = await kick_locator.count()
        except Exception:
            kick_debug["result"] = "button_locator_count_error"
            self.logger.write_json("last_room_kick_debug", kick_debug)
            return 0, False

        kick_debug["dom_button_count"] = count
        kicked = 0
        for idx in range(count):
            button = kick_locator.nth(idx)
            try:
                if not (await button.is_visible() and await button.is_enabled()):
                    continue
                row_text = await button.evaluate(
                    """
                    (el) => {
                      const row = el.closest('tr, li, [role="row"], .player, .player-row, .room-player, .member, .participant') || el.parentElement || el;
                      return String((row && row.innerText) || el.innerText || '');
                    }
                    """
                )
            except Exception:
                continue

            player_rating = self._extract_rating_from_text(str(row_text))
            if player_rating is not None and player_rating < int(effective_min_rating):
                try:
                    await button.click()
                    kicked += 1
                    await asyncio.sleep(0.2)
                except Exception:
                    pass

        kick_debug["result"] = "kicked_via_button_fallback" if kicked > 0 else "waiting_for_eligible_opponent"
        self.logger.write_json("last_room_kick_debug", kick_debug)
        return kicked, False

    async def _set_labeled_choice(self, page, labels: tuple[str, ...], options: tuple[str, ...]) -> bool:
        for label in labels:
            control = page.get_by_label(re.compile(label, re.IGNORECASE))
            try:
                count = await control.count()
            except Exception:
                count = 0
            if count > 0:
                target = control.first
                for option in options:
                    try:
                        await target.select_option(label=option)
                        return True
                    except Exception:
                        pass
                    try:
                        await target.select_option(value=option)
                        return True
                    except Exception:
                        pass
            combobox = page.get_by_role("combobox", name=re.compile(label, re.IGNORECASE))
            try:
                count = await combobox.count()
            except Exception:
                count = 0
            if count > 0:
                box = combobox.first
                for option in options:
                    try:
                        await box.select_option(label=option)
                        return True
                    except Exception:
                        pass
                    try:
                        await box.select_option(value=option)
                        return True
                    except Exception:
                        pass
        return False

    async def _set_named_select(self, page, *, name: str, preferred_values: tuple[str, ...]) -> bool:
        locator = page.locator(f"select[name='{name}']")
        try:
            count = await locator.count()
        except Exception:
            return False
        if count <= 0:
            return False
        select = locator.first
        for value in preferred_values:
            try:
                await select.select_option(value=value)
                return True
            except Exception:
                pass
            try:
                await select.select_option(label=value)
                return True
            except Exception:
                pass
        return False

    async def _set_named_checkbox(self, page, *, name: str, checked: bool) -> bool:
        locator = page.locator(f"input[name='{name}'][type='checkbox']")
        try:
            count = await locator.count()
        except Exception:
            return False
        if count <= 0:
            return False
        box = locator.first
        try:
            current = await box.is_checked()
        except Exception:
            return False
        if current == checked:
            return True
        try:
            if checked:
                await box.check()
            else:
                await box.uncheck()
            return True
        except Exception:
            return False

    async def _ensure_room_configuration(self, page) -> None:
        configured = False
        configured = await self._set_named_select(page, name="numPlayers", preferred_values=("2",)) or configured
        configured = await self._set_named_select(page, name="numAIPlayers", preferred_values=("0",)) or configured
        configured = await self._set_named_select(page, name="speed", preferred_values=("fast", "5mins / person, +10secs / action")) or configured
        configured = await self._set_named_select(page, name="targetScore", preferred_values=("15",)) or configured
        configured = await self._set_named_checkbox(page, name="nextCardVisible", checked=False) or configured
        if configured:
            return
        await self._set_labeled_choice(page, ("players", "player count"), ("2", "2 players", "2 Players"))
        await self._set_labeled_choice(page, ("cpus", "cpu", "bots"), ("0", "0 cpus", "0 CPUs", "None"))
        await self._set_labeled_choice(page, ("speed", "game speed", "time"), ("5mins / person, +10secs / action", "5 mins", "5 min", "5 minutes"))
        await self._set_labeled_choice(page, ("target score", "score"), ("15",))
        await self._click_first_button(page, ("2 Players", "2 players"))
        await self._click_first_button(page, ("0 CPUs", "0 Cpus", "0 cpus", "No CPUs", "No bots"))
        await self._click_first_button(page, ("5mins / person, +10secs / action", "5 mins", "5 min", "5 minutes"))
        await self._click_first_button(page, ("15",))

    async def _manage_lobby_or_room(self, page) -> str:
        surface = await self._probe_page_surface(page)
        href = surface["href"]
        title = surface["title"]
        body_text = surface["bodyText"]
        body_lower = body_text.lower()
        title_lower = title.lower()
        if "/lobby/rooms" not in href and "/room/" not in href:
            await page.goto(self.config.start_url, wait_until="domcontentloaded")
            return "navigating_to_lobby"
        # Some Spendee room states remain on /lobby/rooms (e.g. title "Ready to Start") while host controls are active.
        lobby_room_like = "/lobby/rooms" in href and (
            "ready to start" in title_lower
            or "ready" in body_lower
            or "start" in body_lower
        )
        if "/room/" in href:
            if await self._return_to_lobby_if_finished_room(page):
                return "returning_to_lobby"
            kicked_count, can_start = await self._kick_low_rated_players(page, surface)
            if kicked_count > 0:
                self._write_status(
                    stage="kicking_low_rating_player",
                    extra={
                        "kicked_count": kicked_count,
                        "min_opponent_rating": int(self.config.min_opponent_rating),
                    },
                )
                return "kicking_low_rating_player"
            if not can_start:
                self._write_status(
                    stage="waiting_for_rating_probe",
                    extra={
                        "reason": "waiting_for_eligible_opponent",
                        "min_opponent_rating": int(self.config.min_opponent_rating),
                    },
                )
                return "waiting_for_opponent"
            if await self._click_first_button(page, ("Start",)):
                return "starting_room"
            return "waiting_for_opponent"
        if lobby_room_like:
            kicked_count, can_start = await self._kick_low_rated_players(page, surface)
            if kicked_count > 0:
                self._write_status(
                    stage="kicking_low_rating_player",
                    extra={
                        "kicked_count": kicked_count,
                        "min_opponent_rating": int(self.config.min_opponent_rating),
                        "surface_mode": "lobby_room_like",
                    },
                )
                return "kicking_low_rating_player"
            if not can_start:
                self._write_status(
                    stage="waiting_for_rating_probe",
                    extra={
                        "reason": "waiting_for_eligible_opponent",
                        "min_opponent_rating": int(self.config.min_opponent_rating),
                        "surface_mode": "lobby_room_like",
                    },
                )
                return "waiting_for_opponent"
            if await self._click_first_button(page, ("Start",)):
                return "starting_room"
            return "waiting_for_opponent"
        create_controls_present = "select[name='numPlayers']" in body_text or "New Room" in body_text
        create_clicked = False
        if not create_controls_present:
            create_clicked = await self._click_first_button(page, ("+", "Create Room", "Create room", "New Room", "New room"))
        if create_clicked:
            await asyncio.sleep(0.5)
            await self._ensure_room_configuration(page)
            if await self._click_first_button(page, ("Create Room", "Create room", "Create", "Confirm", "OK")):
                return "creating_room"
            return "configuring_room"
        await self._ensure_room_configuration(page)
        if await self._click_first_button(page, ("Create Room", "Create room", "Create")):
            return "creating_room"
        if await self._click_first_text(page, ("Create Room", "Create room")):
            return "opening_room_creator"
        return "waiting_for_lobby"

    async def _find_active_game_page(self, context):
        pages = [page for page in list(context.pages) if not self._page_is_closed(page)]
        room_candidates: list[tuple[object, dict]] = []
        fallback_candidates: list[tuple[object, dict]] = []
        for page in reversed(pages):
            try:
                probe = await self._probe_live_game_page(page)
            except Exception:
                continue
            if probe is None:
                continue
            if probe["room_url"]:
                room_candidates.append((page, probe))
            else:
                fallback_candidates.append((page, probe))
        for candidates in (room_candidates, fallback_candidates):
            for page, _probe in candidates:
                try:
                    observed = await self.observer.observe(page)
                except Exception:
                    continue
                if observed is not None:
                    return page
            if candidates:
                return candidates[0][0]
        return None

    async def run(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("playwright is required to run the Spendee bridge") from exc

        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=self.config.user_data_dir,
                headless=False,
            )
            try:
                page = await self._find_active_game_page(context)
                if page is None:
                    if self.config.auto_manage_rooms:
                        page = await self._find_management_page(context)
                    if page is None:
                        page = context.pages[0] if context.pages else await context.new_page()
                    if self._page_is_closed(page):
                        page = await context.new_page()
                    await page.goto(self.config.start_url, wait_until="domcontentloaded")
                self._write_status(stage="waiting_for_game")
                self._write_inactive_action_artifacts(reason="waiting_for_game")
                while True:
                    active_page = await self._find_active_game_page(context)
                    if active_page is None:
                        self._archive_webui_save(reason="waiting_for_game")
                        if self.config.auto_manage_rooms:
                            page = await self._find_management_page(context) or page
                            if self._page_is_closed(page):
                                page = context.pages[0] if context.pages else await context.new_page()
                            management_stage = await self._manage_lobby_or_room(page)
                            self._write_status(stage=management_stage)
                            self._write_inactive_action_artifacts(reason=management_stage)
                        else:
                            self._write_status(stage="waiting_for_game")
                            self._write_inactive_action_artifacts(reason="waiting_for_game")
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue
                    page = active_page
                    if await self._return_to_lobby_if_finished_room(page):
                        self._archive_webui_save(reason="finished_room")
                        self._write_status(stage="returning_to_lobby")
                        self._write_inactive_action_artifacts(reason="returning_to_lobby")
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue
                    observed, timed_out = await self._wait_for_stable_board(page)
                    if timed_out:
                        self._consecutive_stable_timeouts += 1
                        if self._consecutive_stable_timeouts >= self.config.stable_board_repair_threshold:
                            await self._recover_from_stall(
                                page,
                                reason="stable_board_timeout",
                                observed=observed,
                                extra={"count": self._consecutive_stable_timeouts},
                            )
                            await asyncio.sleep(self.config.poll_interval_sec)
                            continue
                        if observed is None:
                            self._write_inactive_action_artifacts(reason="stable_board_timeout")
                            await asyncio.sleep(self.config.poll_interval_sec)
                            continue
                    else:
                        self._consecutive_stable_timeouts = 0
                    if observed is None:
                        self._write_status(stage="waiting_for_game")
                        self._write_inactive_action_artifacts(reason="waiting_for_game")
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue
                    self._write_status(stage="observed", observed=observed)
                    if observed.my_player_index is not None:
                        detected_seat = "P0" if int(observed.my_player_index) == 0 else "P1"
                        if self._player_seat is None:
                            self._player_seat = detected_seat
                        elif self._player_seat != detected_seat and self._seat_pinned:
                            raise RuntimeError(
                                f"Configured player_seat={self._player_seat} but Spendee reports myPlayerIndex={observed.my_player_index}"
                            )
                        elif self._player_seat != detected_seat:
                            self._player_seat = detected_seat
                            self._last_action_idx = None
                        self.shadow.player_seat = self._player_seat
                        self._seat_verified = True
                        self._write_status(stage="seat_detected", observed=observed)
                    if self._player_seat is None:
                        self._write_status(stage="waiting_for_seat", observed=observed)
                        self._write_inactive_action_artifacts(reason="waiting_for_seat", observed=observed)
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue
                    if (
                        self.shadow.last_observation is not None
                        and self.shadow.last_observation.game_id != observed.game_id
                    ):
                        # A new room can arrive while the previous game's bridge
                        # shadow state is still populated. If we do not clear it
                        # here, the next webui save inherits stale action history
                        # and hidden-card knowledge from the prior game.
                        self._reset_transient_state()
                        self.shadow.player_seat = self._player_seat
                    previous_observation = self.shadow.last_observation
                    if previous_observation is not None:
                        resolved_hidden_cards = self._resolve_hidden_reserved_cards(
                            previous=previous_observation,
                            current=observed,
                        )
                        for seat, slot, card_id in resolved_hidden_cards:
                            self._patch_webui_history_hidden_reserved_card(
                                seat=seat,
                                slot=slot,
                                card_id=card_id,
                            )
                    self.shadow.apply_observation(observed, expected_action_idx=self._last_action_idx)
                    self._last_action_idx = None
                    self._write_webui_save()

                    if self.config.observe_only or not is_actionable_turn(observed, self._player_seat):
                        stage = "waiting_for_turn"
                        if self._player_seat == observed.current_turn_seat and observed.current_job != "SPENDEE_REGULAR":
                            stage = "waiting_for_job_transition"
                        self._write_status(stage=stage, observed=observed)
                        self._write_inactive_action_artifacts(reason=stage, observed=observed)
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue

                    self._write_status(stage="thinking", observed=observed)
                    rng = random.Random()
                    if observed.modal_state.kind == "return_gem":
                        decision, return_action_indices = self.policy.choose_return_actions(self.shadow, rng=rng)
                        if not return_action_indices:
                            raise RuntimeError("Return phase produced no return actions")
                        action_indices = return_action_indices
                    else:
                        decision = self.policy.choose_action(self.shadow, rng=rng)
                        action_indices = [decision.action_idx]
                    self.logger.write_json(
                        "last_decision",
                        {
                            "action_idx": decision.action_idx,
                            "action_indices": action_indices,
                            "root_best_value_mean": decision.root_best_value_mean,
                            "num_determinizations": decision.num_determinizations,
                            "current_turn_seat": observed.current_turn_seat,
                            "player_seat": self._player_seat,
                            "board_version": observed.board_version,
                        },
                    )
                    action_plan = None
                    try:
                        from .executor import plan_action, plan_return_actions
                        if observed.modal_state.kind == "return_gem":
                            action_plan = plan_return_actions(
                                action_indices,
                                player_seat=self._player_seat,
                                observation=observed,
                            )
                        else:
                            action_plan = plan_action(
                                decision.action_idx,
                                player_seat=self._player_seat,
                                observation=observed,
                            )
                    except Exception as exc:
                        self._write_status(stage="plan_error", observed=observed, extra={"error": str(exc), "action_idx": decision.action_idx})
                        self.logger.write_json(
                            "last_action_attempt",
                            {
                                "status": "plan_error",
                                "action_idx": decision.action_idx,
                                "action_indices": action_indices,
                                "error": str(exc),
                                "board_version": observed.board_version,
                            },
                        )
                        raise
                    self._write_status(
                        stage="action_planned",
                        observed=observed,
                        extra={"action_idx": decision.action_idx, "kind": action_plan.kind},
                    )
                    self.logger.write_json(
                        "last_action_attempt",
                        {
                            "status": "planned",
                            "action_idx": decision.action_idx,
                            "action_indices": action_indices,
                            "kind": action_plan.kind,
                            "payload": action_plan.payload,
                            "board_version": observed.board_version,
                            "player_seat": self._player_seat,
                        },
                    )
                    execution_observed = observed
                    if not self.config.dry_run:
                        delay_sec = 0.0 if action_plan.kind == "return_gem" else random.uniform(
                            self.config.action_delay_min_sec,
                            self.config.action_delay_max_sec,
                        )
                        self._write_status(
                            stage="delaying_action",
                            observed=observed,
                            extra={
                                "action_idx": decision.action_idx,
                                "kind": action_plan.kind,
                                "delay_sec": round(delay_sec, 3),
                            },
                        )
                        await asyncio.sleep(delay_sec)
                        execution_observed, timed_out = await self._wait_for_stable_board(page)
                        if timed_out:
                            self._consecutive_stable_timeouts += 1
                            if self._consecutive_stable_timeouts >= self.config.stable_board_repair_threshold:
                                await self._recover_from_stall(
                                    page,
                                    reason="stable_board_timeout",
                                    observed=execution_observed,
                                    extra={"count": self._consecutive_stable_timeouts, "action_idx": decision.action_idx},
                                )
                                await asyncio.sleep(self.config.poll_interval_sec)
                                continue
                            if execution_observed is None:
                                self._write_status(
                                    stage="action_plan_stale",
                                    observed=observed,
                                    extra={"action_idx": decision.action_idx, "kind": action_plan.kind, "reason": "board_unobservable"},
                                )
                                self.logger.write_json(
                                    "last_action_attempt",
                                {
                                    "status": "stale_before_submit",
                                    "action_idx": decision.action_idx,
                                    "action_indices": action_indices,
                                    "kind": action_plan.kind,
                                    "payload": action_plan.payload,
                                    "planned_board_version": observed.board_version,
                                    "current_board_version": None,
                                    "player_seat": self._player_seat,
                                        "reason": "board_unobservable",
                                    },
                                )
                                await asyncio.sleep(self.config.poll_interval_sec)
                                continue
                        else:
                            self._consecutive_stable_timeouts = 0
                        if execution_observed is None:
                            await asyncio.sleep(self.config.poll_interval_sec)
                            continue
                        if self._planned_action_is_stale(execution_observed, previous=observed):
                            self._write_status(
                                stage="action_plan_stale",
                                observed=execution_observed,
                                extra={"action_idx": decision.action_idx, "kind": action_plan.kind},
                            )
                            self.logger.write_json(
                                "last_action_attempt",
                                {
                                    "status": "stale_before_submit",
                                    "action_idx": decision.action_idx,
                                    "action_indices": action_indices,
                                    "kind": action_plan.kind,
                                    "payload": action_plan.payload,
                                    "planned_board_version": observed.board_version,
                                    "current_board_version": execution_observed.board_version,
                                    "player_seat": self._player_seat,
                                },
                            )
                            await asyncio.sleep(self.config.poll_interval_sec)
                            continue
                    try:
                        self._write_status(
                            stage="submitting_action",
                            observed=execution_observed,
                            extra={"action_idx": decision.action_idx, "kind": action_plan.kind},
                        )
                        await self.executor.execute_plan(page, action_plan, dry_run=self.config.dry_run)
                    except Exception as exc:
                        self._write_status(
                            stage="action_submit_error",
                            observed=execution_observed,
                            extra={"action_idx": decision.action_idx, "kind": action_plan.kind, "error": str(exc)},
                        )
                        await self.logger.capture_failure(
                            page,
                            "action_submit_error",
                            self._artifact_payload(
                                {
                                    "action_idx": decision.action_idx,
                                    "action_indices": action_indices,
                                    "kind": action_plan.kind,
                                    "payload": action_plan.payload,
                                    "planned_board_version": observed.board_version,
                                    "current_board_version": execution_observed.board_version,
                                    "player_seat": self._player_seat,
                                    "error": str(exc),
                                },
                                observed=execution_observed,
                            ),
                        )
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue
                    self._write_status(
                        stage="action_submitted" if not self.config.dry_run else "dry_run_action",
                        observed=execution_observed,
                        extra={"action_idx": decision.action_idx, "kind": action_plan.kind},
                    )
                    self.logger.write_json(
                        "last_action_attempt",
                        {
                            "status": "submitted" if not self.config.dry_run else "dry_run",
                            "action_idx": decision.action_idx,
                            "action_indices": action_indices,
                            "kind": action_plan.kind,
                            "payload": action_plan.payload,
                            "board_version": execution_observed.board_version,
                            "player_seat": self._player_seat,
                        },
                    )
                    if not self.config.dry_run:
                        settled = await self._wait_for_action_settlement(
                            page,
                            previous=execution_observed,
                            action_idx=decision.action_idx,
                        )
                        if settled is None:
                            for _ in range(self.config.action_retry_count if self._retry_allowed(action_plan.kind) else 0):
                                await self.executor.execute_plan(page, action_plan, dry_run=False)
                                self._write_status(
                                    stage="action_retried",
                                    observed=execution_observed,
                                    extra={"action_idx": decision.action_idx, "kind": action_plan.kind},
                                )
                                settled = await self._wait_for_action_settlement(
                                    page,
                                    previous=execution_observed,
                                    action_idx=decision.action_idx,
                                )
                                if settled is not None:
                                    break
                            if settled is None:
                                self._consecutive_unsettled_actions += 1
                                self.logger.write_json(
                                    "last_action_attempt",
                                    {
                                        "status": "submitted_unsettled",
                                        "action_idx": decision.action_idx,
                                        "action_indices": action_indices,
                                        "kind": action_plan.kind,
                                        "payload": action_plan.payload,
                                        "board_version": execution_observed.board_version,
                                        "player_seat": self._player_seat,
                                    },
                                )
                                if self._consecutive_unsettled_actions >= self.config.unsettled_action_repair_threshold:
                                    await self._recover_from_stall(
                                        page,
                                        reason="submitted_unsettled",
                                        observed=execution_observed,
                                        extra={"count": self._consecutive_unsettled_actions, "action_idx": decision.action_idx},
                                    )
                                    await asyncio.sleep(self.config.poll_interval_sec)
                                    continue
                            else:
                                self._consecutive_unsettled_actions = 0
                        if settled is not None:
                            previous_observation = self.shadow.last_observation
                            if previous_observation is not None:
                                resolved_hidden_cards = self._resolve_hidden_reserved_cards(
                                    previous=previous_observation,
                                    current=settled,
                                )
                                for seat, slot, card_id in resolved_hidden_cards:
                                    self._patch_webui_history_hidden_reserved_card(
                                        seat=seat,
                                        slot=slot,
                                        card_id=card_id,
                                    )
                            self.shadow.apply_observation(settled, expected_action_idx=decision.action_idx)
                            self._last_action_idx = None
                            self._write_webui_save()
                        else:
                            self._last_action_idx = decision.action_idx
                    await asyncio.sleep(self.config.poll_interval_sec)
            finally:
                try:
                    await context.close()
                except Exception as exc:
                    # The driver can already be gone if Chromium exited unexpectedly.
                    # Treat close as best-effort to avoid masking the real runtime outcome.
                    try:
                        self._write_status(stage="context_close_failed", extra={"error": str(exc)})
                    except Exception:
                        pass