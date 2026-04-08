from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import SpendeeCatalog
from .observer import ObservedBoardState


def _player_payload(player) -> dict[str, object]:
    purchased = [card.card_id for card in player.purchased_cards]
    reserved = []
    for slot in player.reserved_slots:
        if slot.state == "visible" and slot.card is not None:
            reserved.append({"slot": slot.slot, "card_id": slot.card.card_id, "is_public": not slot.card.is_private})
    return {
        "tokens": {
            "white": player.tokens["white"],
            "blue": player.tokens["blue"],
            "green": player.tokens["green"],
            "red": player.tokens["red"],
            "black": player.tokens["black"],
            "gold": player.tokens["gold"],
        },
        "bonuses": dict(player.bonuses),
        "points": player.points,
        "purchased_card_ids": purchased,
        "reserved": reserved,
        "claimed_noble_ids": [noble.noble_id for noble in player.claimed_nobles],
    }


@dataclass
class HiddenReservedKnowledge:
    seat: str
    slot: int
    tier: int


@dataclass
class ShadowState:
    catalog: SpendeeCatalog
    player_seat: str
    last_observation: ObservedBoardState | None = None
    hidden_reserved_tiers: dict[tuple[str, int], int] = field(default_factory=dict)
    hidden_reserved_card_ids: dict[tuple[str, int], int] = field(default_factory=dict)
    action_history: list[int] = field(default_factory=list)
    visible_noble_slots: dict[int, int] = field(default_factory=dict)

    def bootstrap(self, observation: ObservedBoardState) -> None:
        self.last_observation = observation
        self.hidden_reserved_tiers.clear()
        self.hidden_reserved_card_ids.clear()
        self.action_history.clear()
        self.visible_noble_slots.clear()
        self._sync_visible_noble_slots(observation)

    def recreate_from_observation(self, observation: ObservedBoardState) -> None:
        self.last_observation = observation
        self.hidden_reserved_tiers.clear()
        self.hidden_reserved_card_ids.clear()
        for seat in ("P0", "P1"):
            for slot in observation.players[seat].reserved_slots:
                if slot.state == "hidden" and slot.tier_hint is not None:
                    self.hidden_reserved_tiers[(seat, slot.slot)] = int(slot.tier_hint)
        self.action_history = [0 for _ in range(max(int(observation.turns_count), 0))]
        self.visible_noble_slots.clear()
        self._sync_visible_noble_slots(observation)

    def _record_hidden_slot(self, seat: str, slot: int, tier: int | None) -> None:
        if tier is None:
            self.hidden_reserved_tiers.pop((seat, slot), None)
            self.hidden_reserved_card_ids.pop((seat, slot), None)
            return
        self.hidden_reserved_tiers[(seat, slot)] = int(tier)

    def assign_hidden_card_id(self, seat: str, slot: int, card_id: int) -> None:
        self.hidden_reserved_card_ids[(seat, slot)] = int(card_id)

    def _sync_visible_noble_slots(self, observation: ObservedBoardState) -> None:
        current_noble_ids = [int(noble.noble_id) for noble in observation.visible_nobles]
        if not current_noble_ids:
            self.visible_noble_slots.clear()
            return

        next_slots: dict[int, int] = {}
        used_slots: set[int] = set()
        for noble_id in current_noble_ids:
            prior_slot = self.visible_noble_slots.get(noble_id)
            if prior_slot in (0, 1, 2) and prior_slot not in used_slots:
                next_slots[noble_id] = int(prior_slot)
                used_slots.add(int(prior_slot))

        free_slots = [slot for slot in range(3) if slot not in used_slots]
        for noble_id in current_noble_ids:
            if noble_id in next_slots:
                continue
            if not free_slots:
                break
            next_slots[noble_id] = int(free_slots.pop(0))
        self.visible_noble_slots = next_slots

    def apply_observation(self, observation: ObservedBoardState, *, expected_action_idx: int | None = None) -> None:
        if self.last_observation is None:
            self.bootstrap(observation)
            return

        previous = self.last_observation
        if expected_action_idx is not None:
            self.action_history.append(int(expected_action_idx))

        for seat in ("P0", "P1"):
            prev_slots = {slot.slot: slot for slot in previous.players[seat].reserved_slots}
            cur_slots = {slot.slot: slot for slot in observation.players[seat].reserved_slots}
            for slot_idx, cur_slot in cur_slots.items():
                prev_slot = prev_slots.get(slot_idx)
                if prev_slot is None:
                    continue
                if prev_slot.state == "empty" and cur_slot.state == "hidden":
                    inferred_tier = cur_slot.tier_hint or self._infer_hidden_tier(previous, observation, seat, expected_action_idx)
                    self._record_hidden_slot(seat, slot_idx, inferred_tier)
                elif prev_slot.state == "hidden" and cur_slot.state == "empty":
                    self.hidden_reserved_tiers.pop((seat, slot_idx), None)
                    self.hidden_reserved_card_ids.pop((seat, slot_idx), None)
                elif prev_slot.state == "hidden" and cur_slot.state == "visible" and cur_slot.card is not None:
                    self.hidden_reserved_tiers.pop((seat, slot_idx), None)
                    self.hidden_reserved_card_ids.pop((seat, slot_idx), None)
                elif cur_slot.state == "visible" and cur_slot.card is not None and (seat, slot_idx) in self.hidden_reserved_tiers:
                    self.hidden_reserved_tiers.pop((seat, slot_idx), None)
                    self.hidden_reserved_card_ids.pop((seat, slot_idx), None)

        self._sync_visible_noble_slots(observation)
        self.last_observation = observation

    def _infer_hidden_tier(
        self,
        previous: ObservedBoardState,
        current: ObservedBoardState,
        seat: str,
        expected_action_idx: int | None,
    ) -> int | None:
        if seat == self.player_seat and expected_action_idx is not None and 27 <= expected_action_idx <= 29:
            return int(expected_action_idx - 27 + 1)

        deck_diffs = [
            tier + 1
            for tier, (prev_count, cur_count) in enumerate(zip(previous.deck_counts, current.deck_counts))
            if cur_count == prev_count - 1
        ]
        if len(deck_diffs) == 1:
            return deck_diffs[0]

        changed_faceup_tiers = []
        for tier, (prev_row, cur_row) in enumerate(zip(previous.faceup, current.faceup), start=1):
            prev_ids = [card.card_id if card else 0 for card in prev_row]
            cur_ids = [card.card_id if card else 0 for card in cur_row]
            if prev_ids != cur_ids:
                changed_faceup_tiers.append(tier)
        if len(changed_faceup_tiers) == 1:
            return changed_faceup_tiers[0]
        return None

    def used_card_ids(self, observation: ObservedBoardState | None = None) -> set[int]:
        observed = observation or self.last_observation
        if observed is None:
            return set()
        used: set[int] = set()
        for deck in observed.deck_card_ids_by_tier:
            used.update(int(card_id) for card_id in deck)
        for player in observed.players.values():
            for card in player.purchased_cards:
                used.add(card.card_id)
            for slot in player.reserved_slots:
                if slot.card is not None:
                    used.add(slot.card.card_id)
        for row in observed.faceup:
            for card in row:
                if card is not None:
                    used.add(card.card_id)
        return used

    def build_base_payload(self, observation: ObservedBoardState | None = None) -> dict[str, object]:
        observed = observation or self.last_observation
        if observed is None:
            raise RuntimeError("ShadowState has not been bootstrapped")

        players_payload = []
        for seat in ("P0", "P1"):
            player = observed.players[seat]
            payload = _player_payload(player)
            reserved = list(payload["reserved"])  # type: ignore[arg-type]
            for slot in player.reserved_slots:
                if slot.state != "hidden":
                    continue
                tier_hint = self.hidden_reserved_tiers.get((seat, slot.slot))
                reserved.append({"slot": slot.slot, "tier_hint": tier_hint, "is_public": False})
            reserved.sort(key=lambda item: int(item.get("slot", 0)))
            payload["reserved"] = reserved
            players_payload.append(payload)

        return {
            "current_player": 0 if observed.current_turn_seat == "P0" else 1,
            "move_number": len(self.action_history),
            "players": players_payload,
            "faceup_card_ids": [
                [card.card_id if card is not None else 0 for card in row]
                for row in observed.faceup
            ],
            "deck_card_ids_by_tier": [list(deck) for deck in observed.deck_card_ids_by_tier],
            "available_noble_ids": [noble.noble_id for noble in observed.visible_nobles],
            "bank": dict(observed.bank),
            "phase_flags": {
                "is_return_phase": observed.modal_state.kind == "return_gem",
                "is_noble_choice_phase": observed.modal_state.kind == "choose_noble",
            },
            "metadata": {
                "game_id": observed.game_id,
                "current_job": observed.current_job,
                "turns_count": observed.turns_count,
                "no_purchase_count": observed.no_purchase_count,
                "my_player_index": observed.my_player_index,
                "hidden_reserved_tiers": {
                    f"{seat}:{slot}": tier for (seat, slot), tier in sorted(self.hidden_reserved_tiers.items())
                },
                "spendee_visible_noble_slots": {
                    str(slot): noble_id
                    for noble_id, slot in sorted(self.visible_noble_slots.items(), key=lambda item: item[1])
                },
                "board_version": observed.board_version,
            },
        }

    def unresolved_hidden_slots(self, observation: ObservedBoardState | None = None) -> list[HiddenReservedKnowledge]:
        observed = observation or self.last_observation
        if observed is None:
            return []
        unresolved: list[HiddenReservedKnowledge] = []
        for seat in ("P0", "P1"):
            for slot in observed.players[seat].reserved_slots:
                if slot.state != "hidden":
                    continue
                tier = self.hidden_reserved_tiers.get((seat, slot.slot))
                if tier is None:
                    unresolved.append(HiddenReservedKnowledge(seat=seat, slot=slot.slot, tier=-1))
                else:
                    unresolved.append(HiddenReservedKnowledge(seat=seat, slot=slot.slot, tier=tier))
        return unresolved
