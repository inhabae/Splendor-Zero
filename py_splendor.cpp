#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <unordered_set>

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

const char* color_name(Color color) {
    switch (color) {
        case Color::White: return "white";
        case Color::Blue: return "blue";
        case Color::Green: return "green";
        case Color::Red: return "red";
        case Color::Black: return "black";
        case Color::Joker: return "joker";
    }
    throw std::out_of_range("Invalid card color");
}

py::dict tokens_dict(const Tokens& tokens) {
    py::dict out;
    out["white"] = tokens.white;
    out["blue"] = tokens.blue;
    out["green"] = tokens.green;
    out["red"] = tokens.red;
    out["black"] = tokens.black;
    return out;
}

py::dict full_tokens_dict(const Tokens& tokens) {
    py::dict out = tokens_dict(tokens);
    out["joker"] = tokens.joker;
    return out;
}

py::list list_standard_cards_py() {
    py::list out;
    for (const Card& card : standardCards()) {
        py::dict item;
        item["id"] = card.id;
        item["tier"] = card.level;
        item["points"] = card.points;
        item["bonus_color"] = color_name(card.color);
        item["cost"] = tokens_dict(card.cost);
        out.append(std::move(item));
    }
    return out;
}

py::list list_standard_nobles_py() {
    py::list out;
    for (const Noble& noble : standardNobles()) {
        py::dict item;
        item["id"] = noble.id;
        item["points"] = noble.points;
        item["requirements"] = tokens_dict(noble.requirements);
        out.append(std::move(item));
    }
    return out;
}

template <typename Container>
auto find_card_by_id(Container& cards, int card_id) {
    return std::find_if(cards.begin(), cards.end(), [card_id](const Card& card) {
        return card.id == card_id;
    });
}

template <typename Container>
auto find_noble_by_id(Container& nobles, int noble_id) {
    return std::find_if(nobles.begin(), nobles.end(), [noble_id](const Noble& noble) {
        return noble.id == noble_id;
    });
}

const Card& require_standard_card(int card_id) {
    const auto& cards = standardCards();
    auto it = find_card_by_id(cards, card_id);
    if (it == cards.end()) {
        throw std::runtime_error("Unknown card id: " + std::to_string(card_id));
    }
    return *it;
}

const Noble& require_standard_noble(int noble_id) {
    const auto& nobles = standardNobles();
    auto it = find_noble_by_id(nobles, noble_id);
    if (it == nobles.end()) {
        throw std::runtime_error("Unknown noble id: " + std::to_string(noble_id));
    }
    return *it;
}

py::dict require_dict(py::handle value, const char* field_name) {
    if (!py::isinstance<py::dict>(value)) {
        throw std::runtime_error(std::string(field_name) + " must be a dict");
    }
    return py::reinterpret_borrow<py::dict>(value);
}

py::list require_list(py::handle value, const char* field_name) {
    if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) {
        throw std::runtime_error(std::string(field_name) + " must be a sequence");
    }
    return py::list(py::reinterpret_borrow<py::object>(value));
}

py::handle require_field(const py::dict& obj, const char* field_name) {
    if (!obj.contains(field_name)) {
        throw std::runtime_error(std::string("Missing required field: ") + field_name);
    }
    return obj[field_name];
}

int optional_int_field(const py::dict& obj, const char* field_name, int default_value) {
    if (!obj.contains(field_name)) {
        return default_value;
    }
    return py::cast<int>(obj[field_name]);
}

bool optional_bool_field(const py::dict& obj, const char* field_name, bool default_value) {
    if (!obj.contains(field_name)) {
        return default_value;
    }
    return py::cast<bool>(obj[field_name]);
}

Tokens parse_tokens_payload(py::handle value, const char* field_name, bool allow_bonus_only = false) {
    const py::dict obj = require_dict(value, field_name);
    Tokens out;
    out.white = optional_int_field(obj, "white", 0);
    out.blue = optional_int_field(obj, "blue", 0);
    out.green = optional_int_field(obj, "green", 0);
    out.red = optional_int_field(obj, "red", 0);
    out.black = optional_int_field(obj, "black", 0);
    out.joker = optional_int_field(obj, "joker", optional_int_field(obj, "gold", 0));
    if (allow_bonus_only && out.joker != 0) {
        throw std::runtime_error(std::string(field_name) + " cannot include joker/gold");
    }
    if (out.white < 0 || out.blue < 0 || out.green < 0 || out.red < 0 || out.black < 0 || out.joker < 0) {
        throw std::runtime_error(std::string(field_name) + " cannot contain negative counts");
    }
    return out;
}

std::vector<int> parse_int_list(py::handle value, const char* field_name) {
    const py::list seq = require_list(value, field_name);
    std::vector<int> out;
    out.reserve(seq.size());
    for (py::handle item : seq) {
        out.push_back(py::cast<int>(item));
    }
    return out;
}

void record_unique_card_id(int card_id, std::unordered_set<int>& seen, const char* context) {
    if (card_id <= 0) {
        throw std::runtime_error(std::string(context) + " must use positive card ids");
    }
    if (!seen.insert(card_id).second) {
        throw std::runtime_error(std::string("Duplicate card id detected in ") + context + ": " + std::to_string(card_id));
    }
}

void record_unique_noble_id(int noble_id, std::unordered_set<int>& seen, const char* context) {
    if (noble_id <= 0) {
        throw std::runtime_error(std::string(context) + " must use positive noble ids");
    }
    if (!seen.insert(noble_id).second) {
        throw std::runtime_error(std::string("Duplicate noble id detected in ") + context + ": " + std::to_string(noble_id));
    }
}

Tokens bonuses_from_cards(const std::vector<Card>& cards) {
    Tokens out;
    for (const Card& card : cards) {
        out[card.color] += 1;
    }
    return out;
}

int points_from_cards_and_nobles(const std::vector<Card>& cards, const std::vector<Noble>& nobles) {
    int total = 0;
    for (const Card& card : cards) {
        total += card.points;
    }
    for (const Noble& noble : nobles) {
        total += noble.points;
    }
    return total;
}

py::dict reserved_card_dict(const ReservedCard& reserved) {
    py::dict out;
    out["card_id"] = reserved.card.id;
    out["is_public"] = reserved.is_public;
    out["tier"] = reserved.card.level;
    return out;
}

py::dict player_state_dict(const Player& player) {
    py::dict out;
    out["tokens"] = full_tokens_dict(player.tokens);
    out["bonuses"] = tokens_dict(player.bonuses);
    out["points"] = player.points;

    py::list purchased;
    for (const Card& card : player.cards) {
        purchased.append(py::int_(card.id));
    }
    out["purchased_card_ids"] = std::move(purchased);

    py::list reserved;
    for (const ReservedCard& card : player.reserved) {
        reserved.append(reserved_card_dict(card));
    }
    out["reserved"] = std::move(reserved);

    py::list claimed_nobles;
    for (const Noble& noble : player.nobles) {
        claimed_nobles.append(py::int_(noble.id));
    }
    out["claimed_noble_ids"] = std::move(claimed_nobles);
    return out;
}

void validate_public_state_consistency(const GameState& state) {
    Tokens totals = state.bank;
    for (const Player& player : state.players) {
        totals += player.tokens;
    }
    if (totals.white != 4 || totals.blue != 4 || totals.green != 4 || totals.red != 4 || totals.black != 4 || totals.joker != 5) {
        throw std::runtime_error("Token totals must sum to the standard 2-player bank counts");
    }
    for (int player_idx = 0; player_idx < 2; ++player_idx) {
        const Player& player = state.players[static_cast<std::size_t>(player_idx)];
        const Tokens expected_bonuses = bonuses_from_cards(player.cards);
        if (player.bonuses.white != expected_bonuses.white ||
            player.bonuses.blue != expected_bonuses.blue ||
            player.bonuses.green != expected_bonuses.green ||
            player.bonuses.red != expected_bonuses.red ||
            player.bonuses.black != expected_bonuses.black ||
            player.bonuses.joker != 0) {
            throw std::runtime_error("Player bonuses do not match purchased cards for player " + std::to_string(player_idx));
        }
        const int expected_points = points_from_cards_and_nobles(player.cards, player.nobles);
        if (player.points != expected_points) {
            throw std::runtime_error("Player points do not match purchased cards and nobles for player " + std::to_string(player_idx));
        }
        if (player.reserved.size() > 3) {
            throw std::runtime_error("Players cannot have more than 3 reserved cards");
        }
        for (const ReservedCard& reserved : player.reserved) {
            if (reserved.card.id <= 0) {
                throw std::runtime_error("Reserved cards must use concrete positive card ids");
            }
        }
    }
    if (state.current_player < 0 || state.current_player >= 2) {
        throw std::runtime_error("current_player must be 0 or 1");
    }
    if (state.noble_count < 0 || state.noble_count > 3) {
        throw std::runtime_error("noble_count must be in [0, 3]");
    }
}

constexpr int kActionDim = state_encoder::ACTION_DIM;
constexpr int kStateDim = state_encoder::STATE_DIM;
constexpr int kCardFeatureLen = state_encoder::CARD_FEATURE_LEN;
constexpr int kCpTokensStart = state_encoder::CP_TOKENS_START;
constexpr int kCpBonusesStart = state_encoder::CP_BONUSES_START;
constexpr int kCpReservedStart = state_encoder::CP_RESERVED_START;
constexpr int kFaceupStart = state_encoder::FACEUP_START;
constexpr int kBankStart = state_encoder::BANK_START;

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
        hydration_metadata_ = py::dict();
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

    StepResult load_state(py::dict payload) {
        GameState next{};
        std::unordered_set<int> seen_card_ids;
        std::unordered_set<int> seen_noble_ids;

        const int current_player = py::cast<int>(require_field(payload, "current_player"));
        validate_player_id(current_player);
        next.current_player = current_player;
        next.move_number = std::max(0, optional_int_field(payload, "move_number", 0));

        const py::list players = require_list(require_field(payload, "players"), "players");
        if (players.size() != 2) {
            throw std::runtime_error("players must contain exactly two entries");
        }

        for (int player_idx = 0; player_idx < 2; ++player_idx) {
            const py::dict player_payload = require_dict(players[static_cast<py::ssize_t>(player_idx)], "players[*]");
            Player& player = next.players[static_cast<std::size_t>(player_idx)];
            player.tokens = parse_tokens_payload(require_field(player_payload, "tokens"), "players[*].tokens");

            const Tokens provided_bonuses = parse_tokens_payload(require_field(player_payload, "bonuses"), "players[*].bonuses", true);
            const int provided_points = py::cast<int>(require_field(player_payload, "points"));

            const std::vector<int> purchased_ids = parse_int_list(require_field(player_payload, "purchased_card_ids"), "players[*].purchased_card_ids");
            for (int card_id : purchased_ids) {
                record_unique_card_id(card_id, seen_card_ids, "players[*].purchased_card_ids");
                player.cards.push_back(require_standard_card(card_id));
            }

            const py::list reserved_payload = require_list(require_field(player_payload, "reserved"), "players[*].reserved");
            if (reserved_payload.size() > 3) {
                throw std::runtime_error("players[*].reserved cannot contain more than 3 cards");
            }
            for (py::handle item : reserved_payload) {
                const py::dict reserved_obj = require_dict(item, "players[*].reserved[*]");
                const int card_id = py::cast<int>(require_field(reserved_obj, "card_id"));
                const bool is_public = optional_bool_field(reserved_obj, "is_public", true);
                record_unique_card_id(card_id, seen_card_ids, "players[*].reserved[*]");
                player.reserved.push_back(ReservedCard{require_standard_card(card_id), is_public});
            }

            const std::vector<int> claimed_noble_ids = parse_int_list(require_field(player_payload, "claimed_noble_ids"), "players[*].claimed_noble_ids");
            for (int noble_id : claimed_noble_ids) {
                record_unique_noble_id(noble_id, seen_noble_ids, "players[*].claimed_noble_ids");
                player.nobles.push_back(require_standard_noble(noble_id));
            }

            const Tokens derived_bonuses = bonuses_from_cards(player.cards);
            if (provided_bonuses.white != derived_bonuses.white ||
                provided_bonuses.blue != derived_bonuses.blue ||
                provided_bonuses.green != derived_bonuses.green ||
                provided_bonuses.red != derived_bonuses.red ||
                provided_bonuses.black != derived_bonuses.black) {
                throw std::runtime_error("players[*].bonuses does not match purchased_card_ids");
            }
            player.bonuses = derived_bonuses;

            const int derived_points = points_from_cards_and_nobles(player.cards, player.nobles);
            if (provided_points != derived_points) {
                throw std::runtime_error("players[*].points does not match purchased cards and claimed nobles");
            }
            player.points = derived_points;
        }

        const py::list faceup_rows = require_list(require_field(payload, "faceup_card_ids"), "faceup_card_ids");
        if (faceup_rows.size() != 3) {
            throw std::runtime_error("faceup_card_ids must contain exactly three tiers");
        }
        for (int tier = 0; tier < 3; ++tier) {
            const py::list row = require_list(faceup_rows[static_cast<py::ssize_t>(tier)], "faceup_card_ids[*]");
            if (row.size() != 4) {
                throw std::runtime_error("Each faceup_card_ids tier must contain exactly four slots");
            }
            for (int slot = 0; slot < 4; ++slot) {
                const int card_id = py::cast<int>(row[static_cast<py::ssize_t>(slot)]);
                if (card_id == 0) {
                    next.faceup[static_cast<std::size_t>(tier)][static_cast<std::size_t>(slot)] = Card{};
                    continue;
                }
                record_unique_card_id(card_id, seen_card_ids, "faceup_card_ids");
                const Card& card = require_standard_card(card_id);
                if (card.level != tier + 1) {
                    throw std::runtime_error("faceup_card_ids contains a card in the wrong tier");
                }
                next.faceup[static_cast<std::size_t>(tier)][static_cast<std::size_t>(slot)] = card;
            }
        }

        const std::vector<int> available_noble_ids = parse_int_list(require_field(payload, "available_noble_ids"), "available_noble_ids");
        if (available_noble_ids.size() > 3) {
            throw std::runtime_error("available_noble_ids cannot contain more than 3 nobles");
        }
        next.noble_count = static_cast<int>(available_noble_ids.size());
        for (int idx = 0; idx < next.noble_count; ++idx) {
            const int noble_id = available_noble_ids[static_cast<std::size_t>(idx)];
            record_unique_noble_id(noble_id, seen_noble_ids, "available_noble_ids");
            next.available_nobles[static_cast<std::size_t>(idx)] = require_standard_noble(noble_id);
        }

        next.bank = parse_tokens_payload(require_field(payload, "bank"), "bank");

        const py::dict phase_flags = require_dict(require_field(payload, "phase_flags"), "phase_flags");
        next.is_return_phase = optional_bool_field(phase_flags, "is_return_phase", false);
        next.is_noble_choice_phase = optional_bool_field(phase_flags, "is_noble_choice_phase", false);
        if (next.is_return_phase && next.is_noble_choice_phase) {
            throw std::runtime_error("State cannot be in return phase and noble choice phase simultaneously");
        }

        const bool payload_has_explicit_decks = payload.contains("deck_card_ids_by_tier");
        if (payload_has_explicit_decks) {
            const py::list deck_rows = require_list(require_field(payload, "deck_card_ids_by_tier"), "deck_card_ids_by_tier");
            if (deck_rows.size() != 3) {
                throw std::runtime_error("deck_card_ids_by_tier must contain exactly three tiers");
            }
            for (int tier = 0; tier < 3; ++tier) {
                const py::list deck_row = require_list(deck_rows[static_cast<py::ssize_t>(tier)], "deck_card_ids_by_tier[*]");
                for (py::handle item : deck_row) {
                    const int card_id = py::cast<int>(item);
                    record_unique_card_id(card_id, seen_card_ids, "deck_card_ids_by_tier");
                    const Card& card = require_standard_card(card_id);
                    if (card.level != tier + 1) {
                        throw std::runtime_error("deck_card_ids_by_tier contains a card in the wrong tier");
                    }
                    next.deck[static_cast<std::size_t>(tier)].push_back(card);
                }
            }
        } else {
            for (const Card& card : standardCards()) {
                if (seen_card_ids.find(card.id) != seen_card_ids.end()) {
                    continue;
                }
                next.deck[static_cast<std::size_t>(card.level - 1)].push_back(card);
                seen_card_ids.insert(card.id);
            }
        }

        for (const Card& card : standardCards()) {
            if (seen_card_ids.find(card.id) == seen_card_ids.end()) {
                throw std::runtime_error("Hydrated state is missing standard card id " + std::to_string(card.id));
            }
        }

        validate_public_state_consistency(next);
        state_ = std::move(next);
        hydration_metadata_ = payload.contains("metadata") ? py::dict(payload["metadata"]) : py::dict();
        initialized_ = true;
        return make_step_result();
    }

    py::dict export_state() const {
        ensure_initialized();
        py::dict out;
        out["current_player"] = state_.current_player;
        out["move_number"] = state_.move_number;

        py::list players;
        for (const Player& player : state_.players) {
            players.append(player_state_dict(player));
        }
        out["players"] = std::move(players);

        py::list faceup_rows;
        for (int tier = 0; tier < 3; ++tier) {
            py::list row;
            for (int slot = 0; slot < 4; ++slot) {
                row.append(py::int_(state_.faceup[static_cast<std::size_t>(tier)][static_cast<std::size_t>(slot)].id));
            }
            faceup_rows.append(std::move(row));
        }
        out["faceup_card_ids"] = std::move(faceup_rows);

        py::list available_nobles;
        for (int idx = 0; idx < state_.noble_count; ++idx) {
            available_nobles.append(py::int_(state_.available_nobles[static_cast<std::size_t>(idx)].id));
        }
        out["available_noble_ids"] = std::move(available_nobles);
        out["bank"] = full_tokens_dict(state_.bank);

        py::dict phase_flags;
        phase_flags["is_return_phase"] = state_.is_return_phase;
        phase_flags["is_noble_choice_phase"] = state_.is_noble_choice_phase;
        out["phase_flags"] = std::move(phase_flags);

        py::list deck_rows;
        for (int tier = 0; tier < 3; ++tier) {
            py::list deck_row;
            for (const Card& card : state_.deck[static_cast<std::size_t>(tier)]) {
                deck_row.append(py::int_(card.id));
            }
            deck_rows.append(std::move(deck_row));
        }
        out["deck_card_ids_by_tier"] = std::move(deck_rows);
        out["metadata"] = py::dict(hydration_metadata_);
        return out;
    }

    NativeEnv clone() const {
        ensure_initialized();
        NativeEnv cloned;
        cloned.state_ = state_;
        cloned.initialized_ = initialized_;
        cloned.hydration_metadata_ = py::dict(hydration_metadata_);
        return cloned;
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

    py::dict hidden_deck_card_ids_by_tier() const {
        ensure_initialized();
        py::dict out;
        for (int tier = 0; tier < 3; ++tier) {
            py::list ids;
            for (const Card& card : state_.deck[static_cast<std::size_t>(tier)]) {
                ids.append(py::int_(card.id));
            }
            out[py::int_(tier + 1)] = std::move(ids);
        }
        return out;
    }

    py::dict hidden_faceup_reveal_candidates() const {
        ensure_initialized();
        py::dict out;
        const auto hidden_opponent_cards = hidden_opponent_reserved_cards_by_tier();
        for (int tier = 0; tier < 3; ++tier) {
            const auto& row = state_.faceup[static_cast<std::size_t>(tier)];
            const auto& deck = state_.deck[static_cast<std::size_t>(tier)];
            const auto& hidden_reserved = hidden_opponent_cards[static_cast<std::size_t>(tier)];
            for (int slot = 0; slot < 4; ++slot) {
                const Card& current = row[static_cast<std::size_t>(slot)];
                if (current.id == 0) {
                    continue;
                }
                py::list ids;
                ids.append(py::int_(current.id));
                for (const Card& card : deck) {
                    ids.append(py::int_(card.id));
                }
                for (const Card& card : hidden_reserved) {
                    ids.append(py::int_(card.id));
                }
                out[py::str(std::to_string(tier + 1) + ":" + std::to_string(slot))] = std::move(ids);
            }
        }
        return out;
    }

    py::dict hidden_reserved_reveal_candidates() const {
        ensure_initialized();
        py::dict out;
        for (int player_id = 0; player_id < 2; ++player_id) {
            const auto& reserved = state_.players[static_cast<std::size_t>(player_id)].reserved;
            for (int slot = 0; slot < static_cast<int>(reserved.size()); ++slot) {
                const ReservedCard& current = reserved[static_cast<std::size_t>(slot)];
                if (current.is_public || current.card.id <= 0) {
                    continue;
                }
                const int tier = current.card.level - 1;
                if (tier < 0 || tier >= 3) {
                    throw std::runtime_error("Hidden reserved card has invalid tier");
                }
                py::list ids;
                for (const Card& card : state_.deck[static_cast<std::size_t>(tier)]) {
                    ids.append(py::int_(card.id));
                }
                for (int other_player = 0; other_player < 2; ++other_player) {
                    const auto& other_reserved = state_.players[static_cast<std::size_t>(other_player)].reserved;
                    for (const ReservedCard& other : other_reserved) {
                        if (other.is_public || other.card.id <= 0 || other.card.level != tier + 1) {
                            continue;
                        }
                        ids.append(py::int_(other.card.id));
                    }
                }
                out[py::str(std::string(player_id == 0 ? "P0:" : "P1:") + std::to_string(slot))] = std::move(ids);
            }
        }
        return out;
    }

    StepResult set_faceup_card(int tier, int slot, int card_id) {
        ensure_initialized();
        validate_tier(tier);
        validate_faceup_slot(slot);
        validate_positive_card_id(card_id);

        Card& current = state_.faceup[static_cast<std::size_t>(tier)][static_cast<std::size_t>(slot)];
        if (current.id <= 0) {
            throw std::runtime_error("Face-up slot is empty");
        }
        if (current.id == card_id) {
            return make_step_result();
        }

        auto& deck = state_.deck[static_cast<std::size_t>(tier)];
        auto deck_it = find_card_by_id(deck, card_id);
        if (deck_it != deck.end()) {
            const Card replacement = *deck_it;
            *deck_it = current;
            current = replacement;
            return make_step_result();
        }

        const int current_player = state_.current_player;
        if (current_player < 0 || current_player >= 2) {
            throw std::runtime_error("Current player out of range");
        }
        Player& opponent = state_.players[static_cast<std::size_t>(1 - current_player)];
        for (ReservedCard& reserved : opponent.reserved) {
            if (reserved.is_public || reserved.card.level != tier + 1 || reserved.card.id != card_id) {
                continue;
            }
            std::swap(current, reserved.card);
            return make_step_result();
        }

        throw std::runtime_error("Requested card is not available in the hidden candidate pool for that face-up slot");
    }

    StepResult set_faceup_card_any(int tier, int slot, int card_id) {
        ensure_initialized();
        validate_tier(tier);
        validate_faceup_slot(slot);
        validate_positive_card_id(card_id);

        Card& current = state_.faceup[static_cast<std::size_t>(tier)][static_cast<std::size_t>(slot)];
        if (current.id == card_id) {
            return make_step_result();
        }

        auto& row = state_.faceup[static_cast<std::size_t>(tier)];
        for (int other_slot = 0; other_slot < 4; ++other_slot) {
            if (other_slot == slot) {
                continue;
            }
            Card& other = row[static_cast<std::size_t>(other_slot)];
            if (other.id == card_id) {
                std::swap(current, other);
                return make_step_result();
            }
        }

        auto& deck = state_.deck[static_cast<std::size_t>(tier)];
        auto deck_it = find_card_by_id(deck, card_id);
        if (deck_it != deck.end()) {
            const Card replacement = *deck_it;
            if (current.id > 0) {
                *deck_it = current;
            } else {
                deck.erase(deck_it);
            }
            current = replacement;
            return make_step_result();
        }

        const auto& all_cards = standardCards();
        auto standard_it = find_card_by_id(all_cards, card_id);
        if (standard_it == all_cards.end()) {
            throw std::runtime_error("Unknown card id");
        }
        if (standard_it->level != tier + 1) {
            throw std::runtime_error("Requested card belongs to a different tier");
        }
        current = *standard_it;
        return make_step_result();
    }

    StepResult set_noble(int slot, int noble_id) {
        ensure_initialized();
        validate_noble_slot(slot);
        validate_positive_noble_id(noble_id);
        if (slot >= state_.noble_count) {
            throw std::runtime_error("Noble slot does not exist");
        }

        Noble& current = state_.available_nobles[static_cast<std::size_t>(slot)];
        if (current.id == noble_id) {
            return make_step_result();
        }
        for (int other_slot = 0; other_slot < state_.noble_count; ++other_slot) {
            if (other_slot == slot) {
                continue;
            }
            Noble& other = state_.available_nobles[static_cast<std::size_t>(other_slot)];
            if (other.id == noble_id) {
                std::swap(current, other);
                return make_step_result();
            }
        }
        throw std::runtime_error("Requested noble is not available among visible nobles");
    }

    StepResult set_noble_any(int slot, int noble_id) {
        ensure_initialized();
        validate_noble_slot(slot);
        validate_positive_noble_id(noble_id);
        if (slot >= state_.noble_count) {
            state_.noble_count = slot + 1;
        }

        Noble& current = state_.available_nobles[static_cast<std::size_t>(slot)];
        if (current.id == noble_id) {
            return make_step_result();
        }
        for (int other_slot = 0; other_slot < state_.noble_count; ++other_slot) {
            if (other_slot == slot) {
                continue;
            }
            Noble& other = state_.available_nobles[static_cast<std::size_t>(other_slot)];
            if (other.id == noble_id) {
                std::swap(current, other);
                return make_step_result();
            }
        }

        const auto& nobles = standardNobles();
        auto it = find_noble_by_id(nobles, noble_id);
        if (it == nobles.end()) {
            throw std::runtime_error("Unknown noble id");
        }
        current = *it;
        return make_step_result();
    }

    StepResult set_reserved_card(int player_id, int slot, int card_id) {
        ensure_initialized();
        validate_player_id(player_id);
        validate_reserved_slot(slot);
        validate_positive_card_id(card_id);

        auto& reserved = state_.players[static_cast<std::size_t>(player_id)].reserved;
        if (slot >= static_cast<int>(reserved.size())) {
            throw std::runtime_error("Reserved slot does not exist");
        }

        ReservedCard& current = reserved[static_cast<std::size_t>(slot)];
        if (current.card.id <= 0) {
            throw std::runtime_error("Reserved slot is empty");
        }

        const int tier = current.card.level - 1;
        validate_tier(tier);

        if (current.card.id == card_id) {
            current.is_public = true;
            return make_step_result();
        }

        auto& deck = state_.deck[static_cast<std::size_t>(tier)];
        auto deck_it = find_card_by_id(deck, card_id);
        if (deck_it != deck.end()) {
            const Card replacement = *deck_it;
            *deck_it = current.card;
            current.card = replacement;
            current.is_public = true;
            return make_step_result();
        }

        for (int other_player_id = 0; other_player_id < 2; ++other_player_id) {
            auto& other_reserved = state_.players[static_cast<std::size_t>(other_player_id)].reserved;
            for (int other_slot = 0; other_slot < static_cast<int>(other_reserved.size()); ++other_slot) {
                if (other_player_id == player_id && other_slot == slot) {
                    continue;
                }
                ReservedCard& other = other_reserved[static_cast<std::size_t>(other_slot)];
                if (other.is_public || other.card.level != tier + 1 || other.card.id != card_id) {
                    continue;
                }
                std::swap(current.card, other.card);
                current.is_public = true;
                return make_step_result();
            }
        }

        throw std::runtime_error("Requested card is not available in the hidden candidate pool for that reserved slot");
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

    static void validate_tier(int tier) {
        if (tier < 0 || tier >= 3) {
            throw std::out_of_range("tier must be in [0, 2]");
        }
    }

    static void validate_faceup_slot(int slot) {
        if (slot < 0 || slot >= 4) {
            throw std::out_of_range("face-up slot must be in [0, 3]");
        }
    }

    static void validate_noble_slot(int slot) {
        if (slot < 0 || slot >= 3) {
            throw std::out_of_range("noble slot must be in [0, 2]");
        }
    }

    static void validate_reserved_slot(int slot) {
        if (slot < 0 || slot >= 3) {
            throw std::out_of_range("reserved slot must be in [0, 2]");
        }
    }

    static void validate_positive_card_id(int card_id) {
        if (card_id <= 0) {
            throw std::out_of_range("card_id must be positive");
        }
    }

    static void validate_positive_noble_id(int noble_id) {
        if (noble_id <= 0) {
            throw std::out_of_range("noble_id must be positive");
        }
    }

    static void validate_player_id(int player_id) {
        if (player_id < 0 || player_id >= 2) {
            throw std::out_of_range("player_id must be in [0, 1]");
        }
    }

    std::array<std::vector<Card>, 3> hidden_opponent_reserved_cards_by_tier() const {
        const int current_player = state_.current_player;
        if (current_player < 0 || current_player >= 2) {
            throw std::runtime_error("Current player out of range");
        }
        const int opponent = 1 - current_player;
        std::array<std::vector<Card>, 3> out;
        const auto& reserved = state_.players[static_cast<std::size_t>(opponent)].reserved;
        for (const ReservedCard& card : reserved) {
            if (card.is_public || card.card.id <= 0) {
                continue;
            }
            const int tier = card.card.level - 1;
            if (tier < 0 || tier >= 3) {
                throw std::runtime_error("Hidden reserved card has invalid tier");
            }
            out[static_cast<std::size_t>(tier)].push_back(card.card);
        }
        return out;
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
    py::dict hydration_metadata_;
};

}  // namespace

PYBIND11_MODULE(splendor_native, m) {
    m.doc() = "High-throughput pybind11 bindings for Splendor game logic";

    m.attr("ACTION_DIM") = py::int_(kActionDim);
    m.attr("STATE_DIM") = py::int_(kStateDim);
    py::dict state_layout;
    state_layout["CARD_FEATURE_LEN"] = py::int_(state_encoder::CARD_FEATURE_LEN);
    state_layout["CP_TOKENS_START"] = py::int_(state_encoder::CP_TOKENS_START);
    state_layout["CP_BONUSES_START"] = py::int_(state_encoder::CP_BONUSES_START);
    state_layout["CP_POINTS_IDX"] = py::int_(state_encoder::CP_POINTS_IDX);
    state_layout["CP_RESERVED_START"] = py::int_(state_encoder::CP_RESERVED_START);
    state_layout["OP_TOKENS_START"] = py::int_(state_encoder::OP_TOKENS_START);
    state_layout["OP_BONUSES_START"] = py::int_(state_encoder::OP_BONUSES_START);
    state_layout["OP_POINTS_IDX"] = py::int_(state_encoder::OP_POINTS_IDX);
    state_layout["PLAYER_INDEX_IDX"] = py::int_(state_encoder::PLAYER_INDEX_IDX);
    state_layout["OPPONENT_RESERVED_SLOT_LEN"] = py::int_(state_encoder::OPPONENT_RESERVED_SLOT_LEN);
    state_layout["OP_RESERVED_START"] = py::int_(state_encoder::OP_RESERVED_START);
    state_layout["OP_RESERVED_IS_OCCUPIED_OFFSET"] = py::int_(state_encoder::OPPONENT_RESERVED_OCCUPIED_OFFSET);
    state_layout["OP_RESERVED_TIER_OFFSET"] = py::int_(state_encoder::OPPONENT_RESERVED_TIER_OFFSET);
    state_layout["FACEUP_START"] = py::int_(state_encoder::FACEUP_START);
    state_layout["BANK_START"] = py::int_(state_encoder::BANK_START);
    state_layout["NOBLES_START"] = py::int_(state_encoder::NOBLES_START);
    state_layout["PHASE_FLAGS_START"] = py::int_(state_encoder::PHASE_FLAGS_START);
    m.attr("STATE_LAYOUT") = std::move(state_layout);
    m.attr("BUILD_TYPE") = py::str(SPLENDOR_BUILD_TYPE);
    m.attr("BUILD_OPTIMIZED") = py::bool_(SPLENDOR_BUILD_OPTIMIZED != 0);
    m.def("list_standard_cards", &list_standard_cards_py);
    m.def("list_standard_nobles", &list_standard_nobles_py);

    py::class_<StepResult>(m, "StepResult")
        .def_property_readonly("state", &StepResult::state_array)
        .def_property_readonly("mask", &StepResult::mask_array)
        .def_readonly("is_terminal", &StepResult::is_terminal)
        .def_readonly("winner", &StepResult::winner)
        .def_readonly("current_player_id", &StepResult::current_player_id);

    py::class_<NativeMCTSResult>(m, "NativeMCTSResult")
        .def_property_readonly("visit_probs", &NativeMCTSResult::visit_probs_array)
        .def_readonly("chosen_action_idx", &NativeMCTSResult::chosen_action_idx)
        .def_readonly("root_best_value", &NativeMCTSResult::root_best_value);

    py::class_<NativeEnv>(m, "NativeEnv")
        .def(py::init<>())
        .def("reset", &NativeEnv::reset, py::arg("seed") = 0)
        .def("get_state", &NativeEnv::get_state)
        .def("load_state", &NativeEnv::load_state, py::arg("payload"))
        .def("export_state", &NativeEnv::export_state)
        .def("clone", &NativeEnv::clone)
        .def("step", &NativeEnv::step, py::arg("action_idx"))
        .def("heuristic_action", &NativeEnv::heuristic_action)
        .def("debug_raw_state", &NativeEnv::debug_raw_state)
        .def("hidden_deck_card_ids_by_tier", &NativeEnv::hidden_deck_card_ids_by_tier)
        .def("hidden_faceup_reveal_candidates", &NativeEnv::hidden_faceup_reveal_candidates)
        .def("hidden_reserved_reveal_candidates", &NativeEnv::hidden_reserved_reveal_candidates)
        .def("set_faceup_card", &NativeEnv::set_faceup_card, py::arg("tier"), py::arg("slot"), py::arg("card_id"))
        .def("set_faceup_card_any", &NativeEnv::set_faceup_card_any, py::arg("tier"), py::arg("slot"), py::arg("card_id"))
        .def("set_noble", &NativeEnv::set_noble, py::arg("slot"), py::arg("noble_id"))
        .def("set_noble_any", &NativeEnv::set_noble_any, py::arg("slot"), py::arg("noble_id"))
        .def("set_reserved_card", &NativeEnv::set_reserved_card, py::arg("player_id"), py::arg("slot"), py::arg("card_id"))
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
