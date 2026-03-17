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
from .engine_policy import DeterminizedMCTSPolicy
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
    search_type: Literal["mcts", "ismcts"] = "mcts"
    num_simulations: int = 5000
    determinization_samples: int = 1
    poll_interval_sec: float = 0.5
    stable_polls: int = 2
    stable_board_timeout_sec: float = 8.0
    stable_board_repair_threshold: int = 2
    action_delay_min_sec: float = 3.0
    action_delay_max_sec: float = 5.0
    action_settle_timeout_sec: float = 20.0
    action_retry_count: int = 1
    unsettled_action_repair_threshold: int = 2
    dry_run: bool = True
    observe_only: bool = False
    auto_manage_rooms: bool = False
    selectors: SpendeeSelectorConfig = field(default_factory=SpendeeSelectorConfig)
    artifact_dir: str = "nn_artifacts/spendee_bridge"


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
            analysis_mode=True,
        )
        key = self._webui_history_key(observed)
        if self._webui_save_last_key == key:
            return
        self._webui_save_history.append(payload)
        self._webui_save_last_key = key
        wrapped_payload = dict(payload)
        wrapped_payload["history"] = list(self._webui_save_history)
        wrapped_payload["history_length"] = len(self._webui_save_history)
        self.logger.write_json("webui_save", wrapped_payload)

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
        body_text = surface["bodyText"]
        if "/lobby/rooms" not in href and "/room/" not in href:
            await page.goto(self.config.start_url, wait_until="domcontentloaded")
            return "navigating_to_lobby"
        if "/room/" in href:
            if await self._return_to_lobby_if_finished_room(page):
                return "returning_to_lobby"
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
                        self._last_action_idx = decision.action_idx
                    await asyncio.sleep(self.config.poll_interval_sec)
            finally:
                await context.close()
