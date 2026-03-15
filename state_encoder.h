#pragma once

#include <array>
#include <cstdint>

#include "game_logic.h"

namespace state_encoder {

inline constexpr int ACTION_DIM = 69;
inline constexpr int STATE_DIM = 246;

struct TerminalMetadata {
    bool is_terminal = false;
    int winner = -2;
    int current_player_id = 0;
};

std::array<int, STATE_DIM> build_raw_state(const GameState& state);
std::array<float, STATE_DIM> encode_state(const GameState& state);
std::array<std::uint8_t, ACTION_DIM> build_legal_mask(const GameState& state);
TerminalMetadata build_terminal_metadata(const GameState& state);

}  // namespace state_encoder
