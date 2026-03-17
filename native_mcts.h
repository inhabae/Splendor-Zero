#pragma once

#include <array>
#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "game_logic.h"

struct NativeMCTSResult {
    int chosen_action_idx = 0;
    std::array<float, 69> visit_probs{};
    std::array<float, 69> q_values{};
    float root_best_value = 0.0f;
    int search_slots_requested = 0;
    int search_slots_evaluated = 0;
    int search_slots_drop_pending_eval = 0;
    int search_slots_drop_no_action = 0;

    pybind11::array_t<float> visit_probs_array() const;
    pybind11::array_t<float> q_values_array() const;
};

// Note: search performs per-simulation root determinization by shuffling hidden
// tier deck orders and re-sampling opponent hidden reserved cards before each
// simulation.
NativeMCTSResult run_native_mcts(
    const GameState& root_state,
    pybind11::function evaluator,
    int turns_taken,
    int num_simulations = 64,
    float c_puct = 1.25f,
    int temperature_moves = 10,
    float temperature = 1.0f,
    float eps = 1e-8f,
    bool root_dirichlet_noise = false,
    float root_dirichlet_epsilon = 0.25f,
    float root_dirichlet_alpha_total = 10.0f,
    int eval_batch_size = 32,
    std::uint64_t rng_seed = 0,
    bool use_forced_playouts = false,
    float forced_playouts_k = 2.0f
);

NativeMCTSResult run_native_ismcts(
    const GameState& root_state,
    pybind11::function evaluator,
    int num_simulations = 64,
    float c_puct = 1.25f,
    float eps = 1e-8f,
    int eval_batch_size = 32,
    std::uint64_t rng_seed = 0,
    int root_parallel_workers = 1
);
