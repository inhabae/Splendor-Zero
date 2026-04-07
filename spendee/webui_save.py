from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .shadow_state import ShadowState

def _game_name_from_timestamp(raw_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("Spendee %Y-%m-%d %H:%M:%S UTC")


def _deterministic_hidden_card_id(
    available_by_tier: dict[int, list[int]],
    *,
    tier_hint: int | None,
) -> int:
    if tier_hint in (1, 2, 3):
        pool = available_by_tier[int(tier_hint)]
        if not pool:
            raise RuntimeError(f"No remaining cards available for hidden tier {tier_hint}")
        return pool.pop(0)

    for tier in (1, 2, 3):
        pool = available_by_tier[tier]
        if pool:
            return pool.pop(0)
    raise RuntimeError("No remaining cards available for hidden reserved slot")


def _build_analysis_exported_state(shadow: ShadowState) -> dict[str, Any]:
    observed = shadow.last_observation
    if observed is None:
        raise RuntimeError("ShadowState has not been bootstrapped")

    exported_state = dict(shadow.build_base_payload(observed))
    exported_state.pop("deck_card_ids_by_tier", None)

    used = shadow.used_card_ids(observed)
    available_by_tier = {
        int(tier): sorted(int(card_id) for card_id in card_ids)
        for tier, card_ids in shadow.catalog.remaining_card_ids_by_tier(used).items()
    }

    assigned_keys: set[tuple[str, int]] = set()
    players = list(exported_state.get("players", []))
    for player_index, player_payload in enumerate(players):
        seat = f"P{player_index}"
        reserved_entries = []
        for entry in list(player_payload.get("reserved", [])):
            reserved_entry = dict(entry)
            slot = int(reserved_entry.get("slot", len(reserved_entries)))
            key = (seat, slot)

            if "card_id" in reserved_entry:
                assigned_keys.add(key)
                reserved_entries.append(reserved_entry)
                continue

            tier_hint_raw = reserved_entry.pop("tier_hint", None)
            tier_hint = int(tier_hint_raw) if tier_hint_raw in (1, 2, 3) else None
            assigned = shadow.hidden_reserved_card_ids.get(key)
            if assigned is not None:
                assigned_tier = int(shadow.catalog.cards_by_id[int(assigned)]["tier"])
                pool = available_by_tier.get(assigned_tier, [])
                if int(assigned) in pool and (tier_hint is None or assigned_tier == tier_hint):
                    pool.remove(int(assigned))
                else:
                    assigned = None
            if assigned is None:
                assigned = _deterministic_hidden_card_id(available_by_tier, tier_hint=tier_hint)
                shadow.assign_hidden_card_id(seat, slot, assigned)

            reserved_entry["card_id"] = int(assigned)
            reserved_entries.append(reserved_entry)
            assigned_keys.add(key)
        player_payload["reserved"] = reserved_entries

    stale_keys = [key for key in shadow.hidden_reserved_card_ids if key not in assigned_keys]
    for key in stale_keys:
        shadow.hidden_reserved_card_ids.pop(key, None)

    return exported_state


def build_webui_save_payload(
    shadow: ShadowState,
    *,
    checkpoint_path: str,
    num_simulations: int,
    player_seat: str | None,
    turn_index: int,
    snapshots: list[dict[str, Any]],
    analysis_mode: bool = True,
) -> dict[str, Any]:
    observed = shadow.last_observation
    if observed is None:
        raise RuntimeError("ShadowState has not been bootstrapped")

    exported_state = _build_analysis_exported_state(shadow)
    metadata = dict(exported_state.get("metadata", {}))
    metadata.update(
        {
            "source": "spendee_bridge",
            "spendee_game_id": observed.game_id,
            "spendee_board_version": observed.board_version,
            "spendee_observed_at": observed.observed_at,
            "spendee_player_seat": player_seat,
            "spendee_current_turn_seat": observed.current_turn_seat,
            "spendee_current_job": observed.current_job,
            "spendee_action_items_count": len(observed.raw_action_items),
            # This is the bridge-local sequence of actions we explicitly
            # submitted/expected, not a full authoritative move list for both
            # players.
            "spendee_action_history_scope": "bridge_local_expected_actions",
            "spendee_action_history": [int(action_idx) for action_idx in shadow.action_history],
        }
    )
    exported_state["metadata"] = metadata

    checkpoint_ref = Path(checkpoint_path).name
    saved_at = datetime.now(timezone.utc).isoformat()
    game_name = _game_name_from_timestamp(observed.observed_at)
    return {
        "version": 1,
        "game_name": game_name,
        "saved_at": saved_at,
        "game_id": observed.game_id,
        "config": {
            "checkpoint_id": checkpoint_ref,
            "checkpoint_path": checkpoint_ref,
            "num_simulations": int(num_simulations),
            "player_seat": "P0" if player_seat not in ("P0", "P1") else player_seat,
            "seed": 0,
            "manual_reveal_mode": False,
            "analysis_mode": bool(analysis_mode),
        },
        "snapshots": [*snapshots, {"turn_index": int(turn_index), "exported_state": exported_state}],
        "current_index": len(snapshots),
    }
