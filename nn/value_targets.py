from __future__ import annotations


def winner_to_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    if winner not in (0, 1):
        raise ValueError(f"Unexpected winner value {winner}")
    return 1.0 if winner == player_id else -1.0


def blend_root_and_outcome(value_root: float, value_outcome: float) -> float:
    return 0.5 * (float(value_root) + float(value_outcome))
