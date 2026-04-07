from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .catalog import COLORS, SpendeeCatalog
from .selectors import SpendeeSelectorConfig, build_probe_script


@dataclass(frozen=True)
class ObservedCard:
    card_id: int
    spendee_card_index: int
    tier: int
    points: int
    bonus_color: str
    cost: dict[str, int]
    is_private: bool = False


@dataclass(frozen=True)
class ObservedNoble:
    noble_id: int
    spendee_noble_index: int
    points: int
    requirements: dict[str, int]


@dataclass(frozen=True)
class ObservedReservedSlot:
    slot: int
    state: str
    card: ObservedCard | None = None
    tier_hint: int | None = None


@dataclass(frozen=True)
class ObservedPlayerState:
    seat: str
    spendee_player_index: int
    points: int
    tokens: dict[str, int]
    bonuses: dict[str, int]
    purchased_cards: tuple[ObservedCard, ...]
    reserved_slots: tuple[ObservedReservedSlot, ...]
    claimed_nobles: tuple[ObservedNoble, ...]


@dataclass(frozen=True)
class ObservedModalState:
    kind: str
    options: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ObservedBoardState:
    game_id: str
    players: dict[str, ObservedPlayerState]
    bank: dict[str, int]
    faceup: tuple[tuple[ObservedCard | None, ...], ...]
    deck_counts: tuple[int, int, int]
    deck_card_ids_by_tier: tuple[tuple[int, ...], ...]
    visible_nobles: tuple[ObservedNoble, ...]
    current_turn_seat: str
    current_job: str
    turns_count: int
    no_purchase_count: int
    my_player_index: int | None
    modal_state: ObservedModalState
    animations_active: bool
    board_version: str
    observed_at: str
    raw_action_items: tuple[dict[str, Any], ...] = ()
    raw_active_statuses: tuple[dict[str, Any], ...] = ()


def _normalize_color_counts(payload: dict[str, Any], *, allow_gold: bool) -> dict[str, int]:
    out = {color: int(payload.get(color, 0)) for color in COLORS}
    if allow_gold:
        out["gold"] = int(payload.get("gold", payload.get("joker", 0)))
    return out


def _color_counts_from_sequence(values: list[int] | tuple[int, ...], *, gold: int | None = None) -> dict[str, int]:
    if len(values) != 5:
        raise ValueError(f"Expected 5 color counts, got {len(values)}")
    out = {color: int(values[idx]) for idx, color in enumerate(COLORS)}
    if gold is not None:
        out["gold"] = int(gold)
    return out


def _observed_card_from_engine_id(
    card_id: int,
    catalog: SpendeeCatalog,
    *,
    is_private: bool = False,
    spendee_card_index: int | None = None,
) -> ObservedCard:
    card = dict(catalog.cards_by_id[card_id])
    return ObservedCard(
        card_id=card_id,
        spendee_card_index=(
            catalog.engine_card_id_to_spendee(card_id) if spendee_card_index is None else int(spendee_card_index)
        ),
        tier=int(card["tier"]),
        points=int(card["points"]),
        bonus_color=str(card["bonus_color"]),
        cost=_normalize_color_counts(dict(card["cost"]), allow_gold=False),
        is_private=is_private,
    )


def _observed_noble_from_engine_id(
    noble_id: int,
    catalog: SpendeeCatalog,
    *,
    spendee_noble_index: int | None = None,
) -> ObservedNoble:
    noble = dict(catalog.nobles_by_id[noble_id])
    return ObservedNoble(
        noble_id=noble_id,
        spendee_noble_index=(
            catalog.engine_noble_id_to_spendee(noble_id) if spendee_noble_index is None else int(spendee_noble_index)
        ),
        points=int(noble["points"]),
        requirements=_normalize_color_counts(dict(noble["requirements"]), allow_gold=False),
    )


def _points_from_cards_and_nobles(cards: tuple[ObservedCard, ...], nobles: tuple[ObservedNoble, ...]) -> int:
    return sum(card.points for card in cards) + sum(noble.points for noble in nobles)


def _bonuses_from_cards(cards: tuple[ObservedCard, ...]) -> dict[str, int]:
    bonuses = {color: 0 for color in COLORS}
    for card in cards:
        bonuses[card.bonus_color] += 1
    return bonuses


def _normalize_card(payload: dict[str, Any], catalog: SpendeeCatalog) -> ObservedCard:
    cost = _normalize_color_counts(dict(payload.get("cost", {})), allow_gold=False)
    card_id = catalog.resolve_card_id(
        tier=int(payload["tier"]),
        points=int(payload["points"]),
        bonus_color=str(payload["bonus_color"]),
        cost=cost,
    )
    return ObservedCard(
        card_id=card_id,
        spendee_card_index=catalog.engine_card_id_to_spendee(card_id),
        tier=int(payload["tier"]),
        points=int(payload["points"]),
        bonus_color=str(payload["bonus_color"]),
        cost=cost,
        is_private=bool(payload.get("is_private", False)),
    )


def _normalize_noble(payload: dict[str, Any], catalog: SpendeeCatalog) -> ObservedNoble:
    requirements = _normalize_color_counts(dict(payload.get("requirements", {})), allow_gold=False)
    noble_id = catalog.resolve_noble_id(points=int(payload.get("points", 3)), requirements=requirements)
    return ObservedNoble(
        noble_id=noble_id,
        spendee_noble_index=catalog.engine_noble_id_to_spendee(noble_id),
        points=int(payload.get("points", 3)),
        requirements=requirements,
    )


def normalize_probe_payload(raw: dict[str, Any], catalog: SpendeeCatalog, *, observed_at: str | None = None) -> ObservedBoardState:
    players_raw = raw.get("players")
    if not isinstance(players_raw, dict):
        raise TypeError("probe payload must include a players dict")

    players: dict[str, ObservedPlayerState] = {}
    for seat in ("P0", "P1"):
        seat_raw = players_raw.get(seat)
        if not isinstance(seat_raw, dict):
            raise TypeError(f"Missing probe payload for seat {seat}")
        purchased_cards = tuple(_normalize_card(dict(item), catalog) for item in seat_raw.get("purchased_cards", []))

        reserved_slots: list[ObservedReservedSlot] = []
        for raw_slot in seat_raw.get("reserved_slots", []):
            slot_payload = dict(raw_slot)
            state = str(slot_payload.get("state", "empty"))
            card = _normalize_card(dict(slot_payload["card"]), catalog) if state == "visible" and slot_payload.get("card") else None
            reserved_slots.append(
                ObservedReservedSlot(
                    slot=int(slot_payload.get("slot", len(reserved_slots))),
                    state=state,
                    card=card,
                    tier_hint=(int(slot_payload["tier_hint"]) if slot_payload.get("tier_hint") is not None else None),
                )
            )

        claimed_nobles = tuple(_normalize_noble(dict(item), catalog) for item in seat_raw.get("claimed_nobles", []))
        players[seat] = ObservedPlayerState(
            seat=seat,
            spendee_player_index=0 if seat == "P0" else 1,
            points=int(seat_raw.get("points", 0)),
            tokens=_normalize_color_counts(dict(seat_raw.get("tokens", {})), allow_gold=True),
            bonuses=_normalize_color_counts(dict(seat_raw.get("bonuses", {})), allow_gold=False),
            purchased_cards=purchased_cards,
            reserved_slots=tuple(reserved_slots),
            claimed_nobles=claimed_nobles,
        )

    faceup_rows_raw = raw.get("faceup", [])
    if len(faceup_rows_raw) != 3:
        raise ValueError("probe payload must contain exactly three faceup rows")

    faceup_rows: list[tuple[ObservedCard | None, ...]] = []
    deck_counts: list[int] = []
    for row_payload in faceup_rows_raw:
        row = dict(row_payload)
        deck_counts.append(int(row.get("deck_count", 0)))
        cards = []
        for card_payload in row.get("cards", []):
            cards.append(_normalize_card(dict(card_payload), catalog) if card_payload else None)
        while len(cards) < 4:
            cards.append(None)
        faceup_rows.append(tuple(cards[:4]))

    visible_nobles = tuple(_normalize_noble(dict(item), catalog) for item in raw.get("nobles", []))
    modal_raw = dict(raw.get("modal", {"kind": "none", "options": []}))
    modal_state = ObservedModalState(
        kind=str(modal_raw.get("kind", "none")),
        options=tuple(dict(option) for option in modal_raw.get("options", [])),
    )

    stamp = observed_at or datetime.now(timezone.utc).isoformat()
    canonical = {
        "players": {
            seat: {
                "points": player.points,
                "tokens": player.tokens,
                "bonuses": player.bonuses,
                "purchased_cards": [asdict(card) for card in player.purchased_cards],
                "reserved_slots": [asdict(slot) for slot in player.reserved_slots],
                "claimed_nobles": [asdict(noble) for noble in player.claimed_nobles],
            }
            for seat, player in players.items()
        },
        "bank": _normalize_color_counts(dict(raw.get("bank", {})), allow_gold=True),
        "faceup": [[asdict(card) if card else None for card in row] for row in faceup_rows],
        "deck_counts": deck_counts,
        "visible_nobles": [asdict(noble) for noble in visible_nobles],
        "current_turn_seat": str(raw.get("current_turn_seat", "P0")),
        "modal_state": asdict(modal_state),
        "animations_active": bool(raw.get("animations_active", False)),
    }
    board_version = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    return ObservedBoardState(
        game_id=str(raw.get("game_id", "")),
        players=players,
        bank=_normalize_color_counts(dict(raw.get("bank", {})), allow_gold=True),
        faceup=tuple(faceup_rows),
        deck_counts=(deck_counts[0], deck_counts[1], deck_counts[2]),
        deck_card_ids_by_tier=(tuple(), tuple(), tuple()),
        visible_nobles=visible_nobles,
        current_turn_seat=str(raw.get("current_turn_seat", "P0")),
        current_job=str(raw.get("current_job", "SPENDEE_REGULAR")),
        turns_count=int(raw.get("turns_count", 0)),
        no_purchase_count=int(raw.get("no_purchase_count", 0)),
        my_player_index=(int(raw["my_player_index"]) if raw.get("my_player_index") is not None else None),
        modal_state=modal_state,
        animations_active=bool(raw.get("animations_active", False)),
        board_version=board_version,
        observed_at=stamp,
    )


def _infer_modal_kind(
    *,
    current_turn_seat: str,
    players: dict[str, ObservedPlayerState],
    visible_nobles: tuple[ObservedNoble, ...],
    current_job: str,
    noble_picked: bool,
) -> str:
    current_player = players[current_turn_seat]
    total_tokens = sum(current_player.tokens[color] for color in COLORS) + int(current_player.tokens.get("gold", 0))
    if total_tokens > 10:
        return "return_gem"

    if current_job != "SPENDEE_REGULAR":
        claimable = _claimable_nobles(current_player, visible_nobles)
        # Spendee can pause in PICK_NOBLE even when only one noble is claimable.
        # Expose that as actionable so the runner can submit pickNoble instead of stalling.
        if claimable and not noble_picked:
            return "choose_noble"
    return "none"


def _claimable_nobles(
    player: ObservedPlayerState,
    visible_nobles: tuple[ObservedNoble, ...],
) -> list[ObservedNoble]:
    bonuses = player.bonuses
    return [
        noble
        for noble in visible_nobles
        if all(int(bonuses[color]) >= int(noble.requirements[color]) for color in COLORS)
    ]


def _reconstruct_reserved_history(action_items: list[dict[str, Any]], *, num_players: int) -> dict[int, list[dict[str, int | str]]]:
    history: dict[int, list[dict[str, int | str]]] = {player_index: [] for player_index in range(num_players)}
    for item in action_items:
        action = item.get("action", {})
        if not isinstance(action, dict):
            continue
        player_index = action.get("playerIndex")
        if not isinstance(player_index, int) or player_index not in history:
            continue
        action_type = str(action.get("type", ""))
        records = history[player_index]
        if action_type == "reserveShowedCard":
            card_index = action.get("cardIndex")
            if isinstance(card_index, int):
                records.append({"state": "visible", "spendee_card_index": card_index})
        elif action_type == "reserveHiddenCard":
            level = action.get("level")
            if isinstance(level, int):
                records.append({"state": "hidden", "tier_hint": int(level) + 1})
        elif action_type == "buyReservedCard":
            card_index = action.get("cardIndex")
            remove_idx = next(
                (
                    idx
                    for idx, record in enumerate(records)
                    if record.get("state") == "visible" and record.get("spendee_card_index") == card_index
                ),
                None,
            )
            if remove_idx is None:
                remove_idx = next((idx for idx, record in enumerate(records) if record.get("state") == "hidden"), None)
            if remove_idx is None and records:
                remove_idx = 0
            if remove_idx is not None:
                del records[remove_idx]
    return history


def _reserved_slots_from_meteor_payload(
    *,
    player_index: int,
    my_player_index: int | None,
    reserved_card_indices: list[int],
    reserved_history: list[dict[str, int | str]],
    catalog: SpendeeCatalog,
) -> tuple[ObservedReservedSlot, ...]:
    records = [dict(record) for record in reserved_history]
    slots: list[ObservedReservedSlot] = []
    for slot, card_index in enumerate(reserved_card_indices):
        card_index = int(card_index)
        if slot < len(records):
            record = records[slot]
            if record.get("state") == "visible":
                if int(record.get("spendee_card_index", -1)) != card_index:
                    match_idx = next(
                        (
                            idx
                            for idx in range(slot + 1, len(records))
                            if records[idx].get("state") == "visible"
                            and int(records[idx].get("spendee_card_index", -1)) == card_index
                        ),
                        None,
                    )
                    if match_idx is not None:
                        records[slot], records[match_idx] = records[match_idx], records[slot]
                        record = records[slot]
                if int(record.get("spendee_card_index", -1)) == card_index:
                    slots.append(
                        ObservedReservedSlot(
                            slot=slot,
                            state="visible",
                            card=_observed_card_from_engine_id(
                                catalog.spendee_card_index_to_engine(card_index),
                                catalog,
                                is_private=False,
                                spendee_card_index=card_index,
                            ),
                        )
                    )
                    continue
            if record.get("state") == "hidden":
                # Meteor payload already includes the concrete reserved card
                # index, even when the UI treats the slot as hidden. Preserve
                # the actual card identity and keep it marked private instead
                # of downgrading it to a tier-hint-only placeholder.
                slots.append(
                    ObservedReservedSlot(
                        slot=slot,
                        state="visible",
                        card=_observed_card_from_engine_id(
                            catalog.spendee_card_index_to_engine(card_index),
                            catalog,
                            is_private=True,
                            spendee_card_index=card_index,
                        ),
                    )
                )
                continue

        slots.append(
            ObservedReservedSlot(
                slot=slot,
                state="visible",
                card=_observed_card_from_engine_id(
                    catalog.spendee_card_index_to_engine(card_index),
                    catalog,
                    is_private=True,
                    spendee_card_index=card_index,
                ),
            )
        )
    return tuple(slots)


def normalize_meteor_game_payload(raw: dict[str, Any], catalog: SpendeeCatalog, *, observed_at: str | None = None) -> ObservedBoardState:
    if not isinstance(raw.get("data"), dict):
        raise TypeError("Meteor game payload is missing data")
    data = dict(raw["data"])
    bank = dict(data.get("bank", {}))
    players_raw = list(data.get("players", []))
    if len(players_raw) != 2:
        raise ValueError("Expected exactly two players in game data")

    hidden_rows = list(bank.get("hiddenCards", []))
    showed_rows = list(bank.get("showedCards", []))
    noble_indices = list(bank.get("nobles", []))
    if len(hidden_rows) != 3 or len(showed_rows) != 3 or len(noble_indices) > 3:
        raise ValueError("Unexpected bank layout in Meteor game data")
    action_items = [dict(item) for item in raw.get("actionItems", [])]
    my_player_index = int(raw["myPlayerIndex"]) if raw.get("myPlayerIndex") is not None else None
    reserved_history = _reconstruct_reserved_history(action_items, num_players=len(players_raw))

    players: dict[str, ObservedPlayerState] = {}
    for player_index, player_payload in enumerate(players_raw):
        seat = f"P{player_index}"
        reserved_slots = _reserved_slots_from_meteor_payload(
            player_index=player_index,
            my_player_index=my_player_index,
            reserved_card_indices=[int(card_index) for card_index in player_payload.get("reservedCards", [])],
            reserved_history=reserved_history[player_index],
            catalog=catalog,
        )
        purchased_cards = tuple(
            _observed_card_from_engine_id(
                catalog.spendee_card_index_to_engine(int(card_index)),
                catalog,
                spendee_card_index=int(card_index),
            )
            for card_index in player_payload.get("purchasedCards", [])
        )
        claimed_nobles = tuple(
            _observed_noble_from_engine_id(
                catalog.spendee_noble_index_to_engine(int(noble_index)),
                catalog,
                spendee_noble_index=int(noble_index),
            )
            for noble_index in player_payload.get("nobles", [])
        )
        bonuses = _bonuses_from_cards(purchased_cards)
        players[seat] = ObservedPlayerState(
            seat=seat,
            spendee_player_index=player_index,
            points=_points_from_cards_and_nobles(purchased_cards, claimed_nobles),
            tokens=_color_counts_from_sequence(list(player_payload.get("chips", [])), gold=int(player_payload.get("goldChips", 0))),
            bonuses=bonuses,
            purchased_cards=purchased_cards,
            reserved_slots=reserved_slots,
            claimed_nobles=claimed_nobles,
        )

    faceup_rows: list[tuple[ObservedCard | None, ...]] = []
    deck_counts: list[int] = []
    for hidden_row, showed_row in zip(hidden_rows, showed_rows):
        deck_counts.append(len(hidden_row))
        cards = tuple(
            None
            if card_index is None
            else _observed_card_from_engine_id(
                catalog.spendee_card_index_to_engine(int(card_index)),
                catalog,
                spendee_card_index=int(card_index),
            )
            for card_index in showed_row
        )
        faceup_rows.append(cards)

    visible_nobles = tuple(
        _observed_noble_from_engine_id(
            catalog.spendee_noble_index_to_engine(int(noble_index)),
            catalog,
            spendee_noble_index=int(noble_index),
        )
        for noble_index in noble_indices
    )

    state = dict(data.get("state", {}))
    current_player_raw = state.get("currentPlayerIndex", 0)
    current_player_index = int(current_player_raw) if current_player_raw is not None else None
    current_turn_seat = f"P{current_player_index}" if current_player_index is not None else "NONE"
    current_job = str(state.get("currentJob", "SPENDEE_REGULAR"))
    noble_picked = bool(state.get("noblePicked", False))
    turns_count = int(data.get("turnsCount", 0))
    no_purchase_count = int(data.get("noPurchaseCount", 0))
    claimable_nobles: list[ObservedNoble] = []
    modal_kind = "none"
    if current_player_index is not None:
        current_player = players[current_turn_seat]
        claimable_nobles = _claimable_nobles(current_player, visible_nobles)
        modal_kind = _infer_modal_kind(
            current_turn_seat=current_turn_seat,
            players=players,
            visible_nobles=visible_nobles,
            current_job=current_job,
            noble_picked=noble_picked,
        )
    modal_options: list[dict[str, Any]] = []
    if modal_kind == "choose_noble":
        modal_options = [
            {"noble_id": noble.noble_id, "spendee_noble_index": noble.spendee_noble_index}
            for noble in claimable_nobles
        ]
    modal_state = ObservedModalState(kind=modal_kind, options=tuple(modal_options))

    stamp = observed_at or datetime.now(timezone.utc).isoformat()
    canonical = {
        "game_id": str(raw.get("gameId", raw.get("_id", ""))),
        "players": {
            seat: {
                "points": player.points,
                "tokens": player.tokens,
                "bonuses": player.bonuses,
                "purchased_cards": [asdict(card) for card in player.purchased_cards],
                "reserved_slots": [asdict(slot) for slot in player.reserved_slots],
                "claimed_nobles": [asdict(noble) for noble in player.claimed_nobles],
            }
            for seat, player in players.items()
        },
        "bank": _color_counts_from_sequence(list(bank.get("chips", [])), gold=int(bank.get("goldChips", 0))),
        "faceup": [[asdict(card) if card else None for card in row] for row in faceup_rows],
        "deck_counts": deck_counts,
        "visible_nobles": [asdict(noble) for noble in visible_nobles],
        "current_turn_seat": current_turn_seat,
        "current_job": current_job,
        "turns_count": turns_count,
        "no_purchase_count": no_purchase_count,
        "modal_state": asdict(modal_state),
    }
    board_version = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    action_items = tuple(action_items)
    active_statuses = tuple(dict(item) for item in raw.get("activeStatuses", []))
    return ObservedBoardState(
        game_id=str(raw.get("gameId", raw.get("_id", ""))),
        players=players,
        bank=_color_counts_from_sequence(list(bank.get("chips", [])), gold=int(bank.get("goldChips", 0))),
        faceup=tuple(faceup_rows),
        deck_counts=(deck_counts[0], deck_counts[1], deck_counts[2]),
        deck_card_ids_by_tier=(tuple(), tuple(), tuple()),
        visible_nobles=visible_nobles,
        current_turn_seat=current_turn_seat,
        current_job=current_job,
        turns_count=turns_count,
        no_purchase_count=no_purchase_count,
        my_player_index=my_player_index,
        modal_state=modal_state,
        animations_active=False,
        board_version=board_version,
        observed_at=stamp,
        raw_action_items=action_items,
        raw_active_statuses=active_statuses,
    )


METEOR_GAME_SNAPSHOT_SCRIPT = """
(() => {
  if (typeof Games === "undefined" || !Games || typeof Games.find !== "function") {
    return null;
  }
  const game = Games.findOne();
  if (!game) {
    return null;
  }
  const userId = (typeof Meteor !== "undefined" && Meteor && typeof Meteor.userId === "function") ? Meteor.userId() : null;
  const myPlayerIndex =
    typeof game.playerIndexForUserId === "function" && userId
      ? game.playerIndexForUserId(userId)
      : null;
  return {
    _id: game._id,
    gameId: game._id,
    status: game.status,
    playersMeta: game.players,
    actionItems: game.actionItems || [],
    activeStatuses: game.activeStatuses || [],
    myPlayerIndex,
    data: game.data,
  };
})()
"""


class SpendeeObserver:
    def __init__(
        self,
        catalog: SpendeeCatalog,
        selectors: SpendeeSelectorConfig | None = None,
    ) -> None:
        self._catalog = catalog
        self._selectors = selectors or SpendeeSelectorConfig()
        self._probe_script = build_probe_script(self._selectors)

    async def observe(self, page: Any) -> ObservedBoardState | None:
        raw = await page.evaluate(METEOR_GAME_SNAPSHOT_SCRIPT)
        if raw is not None:
            if not isinstance(raw, dict):
                raise RuntimeError("Meteor snapshot returned a non-dict payload")
            try:
                return normalize_meteor_game_payload(raw, self._catalog)
            except ValueError as exc:
                msg = str(exc)
                if (
                    "Expected exactly two players in game data" in msg
                    or "Unexpected bank layout in Meteor game data" in msg
                ):
                    return None
                raise

        raw = await page.evaluate(self._probe_script)
        if not isinstance(raw, dict):
            return None
        players = raw.get("players")
        if (
            not isinstance(players, dict)
            or "P0" not in players
            or "P1" not in players
            or not isinstance(players.get("P0"), dict)
            or not isinstance(players.get("P1"), dict)
        ):
            return None
        faceup = raw.get("faceup")
        if not isinstance(faceup, list) or len(faceup) != 3:
            return None
        nobles = raw.get("nobles")
        if not isinstance(nobles, list):
            return None
        return normalize_probe_payload(raw, self._catalog)
