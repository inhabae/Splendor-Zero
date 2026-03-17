from __future__ import annotations

from typing import Any


def acting_player_has_hidden_uncertainty(payload: dict[str, Any]) -> bool:
    current_player = int(payload["current_player"])
    opponent = 1 - current_player

    for row in payload.get("deck_card_ids_by_tier") or ():
        if len(row) > 0:
            return True

    players = list(payload["players"])
    opponent_reserved = list(dict(players[opponent])["reserved"])
    return any(not bool(dict(entry).get("is_public", True)) for entry in opponent_reserved)
