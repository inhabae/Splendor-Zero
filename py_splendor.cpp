#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "game_logic.h"
#include "native_mcts.h"
#include "state_encoder.h"

namespace py = pybind11;

#ifndef SPLENDOR_BUILD_TYPE
#define SPLENDOR_BUILD_TYPE "unknown"
#endif

#ifndef SPLENDOR_BUILD_OPTIMIZED
#if defined(NDEBUG)
#define SPLENDOR_BUILD_OPTIMIZED 1
#else
#define SPLENDOR_BUILD_OPTIMIZED 0
#endif
#endif

namespace {

constexpr int kActionDim = state_encoder::ACTION_DIM;
constexpr int kStateDim = state_encoder::STATE_DIM;
constexpr int kCardFeatureLen = 11;
constexpr int kCpTokensStart = 0;
constexpr int kCpBonusesStart = 6;
constexpr int kCpReservedStart = 12;
constexpr int kFaceupStart = 91;
constexpr int kBankStart = 223;

constexpr std::array<std::array<int, 3>, 10> kTake3Triplets{{
    {{0, 1, 2}}, {{0, 1, 3}}, {{0, 1, 4}}, {{0, 2, 3}}, {{0, 2, 4}},
    {{0, 3, 4}}, {{1, 2, 3}}, {{1, 2, 4}}, {{1, 3, 4}}, {{2, 3, 4}},
}};
constexpr std::array<std::array<int, 2>, 10> kTake2Pairs{{
    {{0, 1}}, {{0, 2}}, {{0, 3}}, {{0, 4}}, {{1, 2}},
    {{1, 3}}, {{1, 4}}, {{2, 3}}, {{2, 4}}, {{3, 4}},
}};

std::array<float, 6> raw_cp_tokens(const std::array<float, kStateDim>& norm_state) {
    std::array<float, 6> out{};
    for (int c = 0; c < 5; ++c) {
        out[static_cast<std::size_t>(c)] = norm_state[static_cast<std::size_t>(kCpTokensStart + c)] * 4.0f;
    }
    out[5] = norm_state[static_cast<std::size_t>(kCpTokensStart + 5)] * 5.0f;
    return out;
}

std::array<float, 5> raw_cp_bonuses(const std::array<float, kStateDim>& norm_state) {
    std::array<float, 5> out{};
    for (int c = 0; c < 5; ++c) {
        out[static_cast<std::size_t>(c)] = norm_state[static_cast<std::size_t>(kCpBonusesStart + c)] * 7.0f;
    }
    return out;
}

std::array<float, 6> raw_bank_tokens(const std::array<float, kStateDim>& norm_state) {
    std::array<float, 6> out{};
    for (int c = 0; c < 5; ++c) {
        out[static_cast<std::size_t>(c)] = norm_state[static_cast<std::size_t>(kBankStart + c)] * 4.0f;
    }
    out[5] = norm_state[static_cast<std::size_t>(kBankStart + 5)] * 5.0f;
    return out;
}

std::array<float, kCardFeatureLen> card_block_at(const std::array<float, kStateDim>& norm_state, int start) {
    std::array<float, kCardFeatureLen> block{};
    for (int i = 0; i < kCardFeatureLen; ++i) {
        block[static_cast<std::size_t>(i)] = norm_state[static_cast<std::size_t>(start + i)];
    }
    return block;
}

bool is_empty_card_block(const std::array<float, kCardFeatureLen>& block) {
    for (float v : block) {
        if (v != 0.0f) {
            return false;
        }
    }
    return true;
}

struct DecodedCard {
    std::array<int, 5> costs{};
    int points = 0;
    int bonus_idx = -1;
};

DecodedCard decode_card(const std::array<float, kCardFeatureLen>& block) {
    DecodedCard out;
    for (int c = 0; c < 5; ++c) {
        out.costs[static_cast<std::size_t>(c)] = static_cast<int>(std::lround(static_cast<double>(block[static_cast<std::size_t>(c)] * 7.0f)));
    }
    out.points = static_cast<int>(std::lround(static_cast<double>(block[10] * 5.0f)));
    float best = -std::numeric_limits<float>::infinity();
    float sum = 0.0f;
    int best_idx = -1;
    for (int c = 0; c < 5; ++c) {
        float v = block[static_cast<std::size_t>(5 + c)];
        sum += v;
        if (v > best) {
            best = v;
            best_idx = c;
        }
    }
    out.bonus_idx = (sum > 0.0f ? best_idx : -1);
    return out;
}

float card_progress_score(
    const std::array<float, kStateDim>& norm_state,
    const std::array<float, kCardFeatureLen>& block
) {
    const DecodedCard decoded = decode_card(block);
    const auto tokens = raw_cp_tokens(norm_state);
    const auto bonuses = raw_cp_bonuses(norm_state);
    float total_need = 0.0f;
    float total_shortage = 0.0f;
    for (int c = 0; c < 5; ++c) {
        const float need = std::max(0.0f, static_cast<float>(decoded.costs[static_cast<std::size_t>(c)]) - bonuses[static_cast<std::size_t>(c)]);
        const float shortage = std::max(0.0f, need - tokens[static_cast<std::size_t>(c)]);
        total_need += need;
        total_shortage += shortage;
    }
    float efficiency = 0.0f;
    if (total_need > 0.0f) {
        efficiency = (static_cast<float>(decoded.points) + 1.0f) / total_need;
    }
    float noble_proxy = 0.0f;
    if (decoded.bonus_idx >= 0) {
        noble_proxy = 0.5f + 0.1f * bonuses[static_cast<std::size_t>(decoded.bonus_idx)];
    }
    return 12.0f * static_cast<float>(decoded.points) + 3.0f * efficiency + 1.5f * noble_proxy - 0.4f * total_shortage;
}

std::array<float, 5> color_usefulness(const std::array<float, kStateDim>& norm_state) {
    const auto tokens = raw_cp_tokens(norm_state);
    const auto bonuses = raw_cp_bonuses(norm_state);
    std::array<float, 5> usefulness{};

    auto accumulate_card = [&](const std::array<float, kCardFeatureLen>& block) {
        const DecodedCard decoded = decode_card(block);
        bool all_zero = true;
        for (int c = 0; c < 5; ++c) {
            if (decoded.costs[static_cast<std::size_t>(c)] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            return;
        }
        std::array<float, 5> shortage{};
        float total_short = 0.0f;
        for (int c = 0; c < 5; ++c) {
            const float need = std::max(0.0f, static_cast<float>(decoded.costs[static_cast<std::size_t>(c)]) - bonuses[static_cast<std::size_t>(c)]);
            shortage[static_cast<std::size_t>(c)] = std::max(0.0f, need - tokens[static_cast<std::size_t>(c)]);
            total_short += shortage[static_cast<std::size_t>(c)];
        }
        if (total_short <= 0.0f) {
            return;
        }
        const float weight = 1.0f + static_cast<float>(decoded.points);
        for (int c = 0; c < 5; ++c) {
            usefulness[static_cast<std::size_t>(c)] += weight * (shortage[static_cast<std::size_t>(c)] / total_short);
        }
    };

    for (int i = 0; i < 12; ++i) {
        const auto block = card_block_at(norm_state, kFaceupStart + i * kCardFeatureLen);
        if (!is_empty_card_block(block)) {
            accumulate_card(block);
        }
    }
    for (int i = 0; i < 3; ++i) {
        const auto block = card_block_at(norm_state, kCpReservedStart + i * kCardFeatureLen);
        if (!is_empty_card_block(block)) {
            accumulate_card(block);
        }
    }

    const auto bank = raw_bank_tokens(norm_state);
    for (int c = 0; c < 5; ++c) {
        usefulness[static_cast<std::size_t>(c)] += 0.05f * bank[static_cast<std::size_t>(c)];
    }
    return usefulness;
}

std::array<float, 5> gems_taken_from_action(int action_idx) {
    std::array<float, 5> out{};
    if (action_idx >= 30 && action_idx <= 39) {
        const auto& triplet = kTake3Triplets[static_cast<std::size_t>(action_idx - 30)];
        out[static_cast<std::size_t>(triplet[0])] += 1.0f;
        out[static_cast<std::size_t>(triplet[1])] += 1.0f;
        out[static_cast<std::size_t>(triplet[2])] += 1.0f;
    } else if (action_idx >= 40 && action_idx <= 44) {
        out[static_cast<std::size_t>(action_idx - 40)] = 2.0f;
    } else if (action_idx >= 45 && action_idx <= 54) {
        const auto& pair = kTake2Pairs[static_cast<std::size_t>(action_idx - 45)];
        out[static_cast<std::size_t>(pair[0])] += 1.0f;
        out[static_cast<std::size_t>(pair[1])] += 1.0f;
    } else if (action_idx >= 55 && action_idx <= 59) {
        out[static_cast<std::size_t>(action_idx - 55)] = 1.0f;
    }
    return out;
}

float score_heuristic_action(
    int action,
    const std::array<float, kStateDim>& norm_state,
    const std::array<float, 5>& usefulness,
    const std::array<float, 6>& cp_tokens,
    const std::array<float, 6>& bank
) {
    if (action >= 0 && action <= 11) {
        const auto block = card_block_at(norm_state, kFaceupStart + action * kCardFeatureLen);
        return 100.0f + card_progress_score(norm_state, block);
    }
    if (action >= 12 && action <= 14) {
        const auto block = card_block_at(norm_state, kCpReservedStart + (action - 12) * kCardFeatureLen);
        return 95.0f + card_progress_score(norm_state, block);
    }
    if (action >= 15 && action <= 26) {
        const auto block = card_block_at(norm_state, kFaceupStart + (action - 15) * kCardFeatureLen);
        const float joker_bonus = bank[5] > 0.0f ? 3.0f : 0.0f;
        return 25.0f + 0.55f * card_progress_score(norm_state, block) + joker_bonus;
    }
    if (action >= 27 && action <= 29) {
        const float joker_bonus = bank[5] > 0.0f ? 4.0f : 0.0f;
        return 10.0f + joker_bonus - 0.1f * static_cast<float>(action - 27);
    }
    if (action >= 30 && action <= 59) {
        const auto taken = gems_taken_from_action(action);
        float total_taken = 0.0f;
        float weighted_use = 0.0f;
        int diversity = 0;
        for (int c = 0; c < 5; ++c) {
            const float v = taken[static_cast<std::size_t>(c)];
            total_taken += v;
            weighted_use += usefulness[static_cast<std::size_t>(c)] * v;
            if (v > 0.0f) {
                diversity += 1;
            }
        }
        const float token_total_after = cp_tokens[0] + cp_tokens[1] + cp_tokens[2] + cp_tokens[3] + cp_tokens[4] + cp_tokens[5] + total_taken;
        const float overcap_penalty = std::max(token_total_after - 10.0f, 0.0f) * 2.0f;
        const float single_color_penalty = diversity == 1 ? 0.4f : 0.0f;
        return 8.0f + 1.2f * weighted_use + 0.7f * static_cast<float>(diversity) + 0.2f * total_taken - overcap_penalty - single_color_penalty;
    }
    if (action == 60) {
        return -100.0f;
    }
    if (action >= 61 && action <= 65) {
        const int c = action - 61;
        const float abundance = cp_tokens[static_cast<std::size_t>(c)];
        return -2.0f * usefulness[static_cast<std::size_t>(c)] + 0.25f * abundance;
    }
    if (action >= 66 && action <= 68) {
        return -0.01f * static_cast<float>(action);
    }
    return -1.0e6f;
}

int choose_heuristic_action(
    const std::array<float, kStateDim>& norm_state,
    const std::array<std::uint8_t, kActionDim>& mask
) {
    int best_action = -1;
    for (int i = 0; i < kActionDim; ++i) {
        if (mask[static_cast<std::size_t>(i)] != 0) {
            best_action = i;
            break;
        }
    }
    if (best_action < 0) {
        throw std::runtime_error("GreedyHeuristicOpponent: no legal actions");
    }

    const auto usefulness = color_usefulness(norm_state);
    const auto cp_tokens = raw_cp_tokens(norm_state);
    const auto bank = raw_bank_tokens(norm_state);
    float best_score = -std::numeric_limits<float>::infinity();

    for (int action = 0; action < kActionDim; ++action) {
        if (mask[static_cast<std::size_t>(action)] == 0) {
            continue;
        }
        const float score = score_heuristic_action(action, norm_state, usefulness, cp_tokens, bank);
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

struct StepResult {
    std::array<float, kStateDim> state{};
    std::array<std::uint8_t, kActionDim> mask{};
    bool is_terminal = false;
    int winner = -2;
    int current_player_id = 0;

    py::array_t<float> state_array() const {
        py::array_t<float> arr(kStateDim);
        std::memcpy(arr.mutable_data(), state.data(), sizeof(float) * static_cast<std::size_t>(kStateDim));
        return arr;
    }

    py::array_t<bool> mask_array() const {
        py::array_t<bool> arr(kActionDim);
        auto view = arr.mutable_unchecked<1>();
        for (int i = 0; i < kActionDim; ++i) {
            view(i) = (mask[static_cast<std::size_t>(i)] != 0);
        }
        return arr;
    }
};

class NativeEnv {
public:
    NativeEnv() = default;

    StepResult reset(unsigned int seed = 0) {
        initializeGame(state_, seed);
        initialized_ = true;
        return make_step_result();
    }

    StepResult get_state() const {
        ensure_initialized();
        return make_step_result();
    }

    StepResult step(int action_idx) {
        ensure_initialized();
        validate_action_idx(action_idx);
        const auto mask = getValidMoveMask(state_);
        if (!mask[static_cast<std::size_t>(action_idx)]) {
            throw std::invalid_argument("action is not valid in current state");
        }
        applyMove(state_, actionIndexToMove(action_idx));
        return make_step_result();
    }

    py::array_t<int> debug_raw_state() const {
        ensure_initialized();
        const auto raw = state_encoder::build_raw_state(state_);
        py::array_t<int> arr(kStateDim);
        std::memcpy(arr.mutable_data(), raw.data(), sizeof(int) * static_cast<std::size_t>(kStateDim));
        return arr;
    }

    NativeMCTSResult run_mcts(
        py::function evaluator,
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
    ) const {
        ensure_initialized();
        return run_native_mcts(
            state_,
            std::move(evaluator),
            turns_taken,
            num_simulations,
            c_puct,
            temperature_moves,
            temperature,
            eps,
            root_dirichlet_noise,
            root_dirichlet_epsilon,
            root_dirichlet_alpha_total,
            eval_batch_size,
            rng_seed,
            use_forced_playouts,
            forced_playouts_k
        );
    }

    int heuristic_action() const {
        ensure_initialized();
        const auto encoded = state_encoder::encode_state(state_);
        const auto mask = state_encoder::build_legal_mask(state_);
        return choose_heuristic_action(encoded, mask);
    }

private:
    void ensure_initialized() const {
        if (!initialized_) {
            throw std::runtime_error("Game not initialized; call reset() first");
        }
    }

    static void validate_action_idx(int action_idx) {
        if (action_idx < 0 || action_idx >= kActionDim) {
            throw std::out_of_range("action_idx must be in [0, 68]");
        }
    }

    StepResult make_step_result() const {
        StepResult out;
        out.state = state_encoder::encode_state(state_);
        out.mask = state_encoder::build_legal_mask(state_);
        const auto terminal = state_encoder::build_terminal_metadata(state_);
        out.is_terminal = terminal.is_terminal;
        out.winner = terminal.winner;
        out.current_player_id = terminal.current_player_id;
        return out;
    }

    GameState state_{};
    bool initialized_ = false;
};

}  // namespace

PYBIND11_MODULE(splendor_native, m) {
    m.doc() = "High-throughput pybind11 bindings for Splendor game logic";

    m.attr("ACTION_DIM") = py::int_(kActionDim);
    m.attr("STATE_DIM") = py::int_(kStateDim);
    m.attr("BUILD_TYPE") = py::str(SPLENDOR_BUILD_TYPE);
    m.attr("BUILD_OPTIMIZED") = py::bool_(SPLENDOR_BUILD_OPTIMIZED != 0);

    py::class_<StepResult>(m, "StepResult")
        .def_property_readonly("state", &StepResult::state_array)
        .def_property_readonly("mask", &StepResult::mask_array)
        .def_readonly("is_terminal", &StepResult::is_terminal)
        .def_readonly("winner", &StepResult::winner)
        .def_readonly("current_player_id", &StepResult::current_player_id);

    py::class_<NativeMCTSResult>(m, "NativeMCTSResult")
        .def_property_readonly("visit_probs", &NativeMCTSResult::visit_probs_array)
        .def_readonly("chosen_action_idx", &NativeMCTSResult::chosen_action_idx)
        .def_readonly("root_value", &NativeMCTSResult::root_value)
        .def_readonly("root_best_value", &NativeMCTSResult::root_best_value);

    py::class_<NativeEnv>(m, "NativeEnv")
        .def(py::init<>())
        .def("reset", &NativeEnv::reset, py::arg("seed") = 0)
        .def("get_state", &NativeEnv::get_state)
        .def("step", &NativeEnv::step, py::arg("action_idx"))
        .def("heuristic_action", &NativeEnv::heuristic_action)
        .def("debug_raw_state", &NativeEnv::debug_raw_state)
        .def(
            "run_mcts",
            &NativeEnv::run_mcts,
            py::arg("evaluator"),
            py::arg("turns_taken"),
            py::arg("num_simulations") = 64,
            py::arg("c_puct") = 1.25f,
            py::arg("temperature_moves") = 10,
            py::arg("temperature") = 1.0f,
            py::arg("eps") = 1e-8f,
            py::arg("root_dirichlet_noise") = false,
            py::arg("root_dirichlet_epsilon") = 0.25f,
            py::arg("root_dirichlet_alpha_total") = 10.0f,
            py::arg("eval_batch_size") = 32,
            py::arg("rng_seed") = static_cast<std::uint64_t>(0),
            py::arg("use_forced_playouts") = false,
            py::arg("forced_playouts_k") = 2.0f
        );
}
