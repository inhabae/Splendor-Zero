#include "native_mcts.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "state_encoder.h"

namespace {

constexpr int kActionDim = state_encoder::ACTION_DIM;
constexpr int kStateDim = state_encoder::STATE_DIM;
constexpr float kVirtualLoss = 1.0f;
constexpr float kBlockingPriorBoost = 10.0f;
constexpr float kBlockingPriorImmediateThreshold = 0.5f;
constexpr int kMaxAutoTreeWorkers = 32;
constexpr int kISMCTSRootMinVisitFloor = 0;
constexpr float kISMCTSDeckReserveKeepPenaltyTier1Max = 0.02f;
constexpr float kISMCTSDeckReserveKeepPenaltyTier2Max = 0.06f;
constexpr float kISMCTSDeckReserveKeepPenaltyTier3Max = 0.12f;
constexpr const char* kTreeWorkersEnvVar = "SPLENDOR_MCTS_TREE_WORKERS";
constexpr const char* kISMCTSTraceRootActionEnvVar = "SPLENDOR_ISMCTS_TRACE_ROOT_ACTION";

struct MCTSNode {
    std::array<float, kActionDim> priors{};
    std::array<int, kActionDim> visit_count{};
    std::array<float, kActionDim> value_sum{};
    std::array<int, kActionDim> child_index{};
    bool expanded = false;
    bool pending_eval = false;

    MCTSNode() {
        child_index.fill(-1);
    }
};

struct PathStep {
    int node_index = -1;
    int action = -1;
    bool same_player = false;
    bool virtual_loss_applied = false;
};

struct PendingLeafEval {
    int node_index = -1;
    GameState state_snapshot{};
    std::array<float, kStateDim> state{};
    std::array<std::uint8_t, kActionDim> mask{};
    std::vector<PathStep> path;
};

struct ReadyBackup {
    float value = 0.0f;
    int tree_depth = -1;
    int resolved_depth = -1;
    std::vector<PathStep> path;
};

struct NodeMetadata {
    std::array<std::uint8_t, kActionDim> mask{};
    state_encoder::TerminalMetadata terminal{};
};

struct ISMCTSNode {
    std::array<float, kActionDim> priors{};
    std::array<int, kActionDim> visit_count{};
    std::array<float, kActionDim> value_sum{};
    int total_visit_count = 0;
};

struct InfoSetKey {
    std::array<int, kStateDim> raw_state{};

    bool operator==(const InfoSetKey& other) const {
        return raw_state == other.raw_state;
    }
};

struct InfoSetKeyHasher {
    std::size_t operator()(const InfoSetKey& key) const noexcept {
        std::size_t out = 1469598103934665603ULL;
        for (int value : key.raw_state) {
            out ^= static_cast<std::size_t>(value + 1315423911U);
            out *= 1099511628211ULL;
        }
        return out;
    }
};

struct ISPendingLeafEval {
    InfoSetKey key{};
    std::array<float, kStateDim> state{};
    std::array<std::uint8_t, kActionDim> mask{};
    std::vector<PathStep> path;
};

struct ISMCTSWorkerResult {
    ISMCTSNode root{};
    int search_slots_requested = 0;
    int search_slots_evaluated = 0;
    int search_slots_drop_pending_eval = 0;
    int search_slots_drop_no_action = 0;
    int search_slots_drop_cycle = 0;
};

NodeMetadata build_node_metadata(const GameState& state);

bool is_modal_subturn_state(const GameState& state) {
    return state.is_return_phase || state.is_noble_choice_phase;
}

int select_max_policy_legal_action(
    const std::array<float, kActionDim>& policy_scores,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    int best_action = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const float score = policy_scores[static_cast<std::size_t>(a)];
        if (!std::isfinite(static_cast<double>(score))) {
            continue;
        }
        if (best_action < 0 || score > best_score) {
            best_action = a;
            best_score = score;
        }
    }
    if (best_action >= 0) {
        return best_action;
    }
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] != 0) {
            return a;
        }
    }
    return -1;
}

void record_depth_histogram(std::vector<int>& hist, int depth) {
    if (depth < 0) {
        return;
    }
    if (static_cast<std::size_t>(depth) >= hist.size()) {
        hist.resize(static_cast<std::size_t>(depth) + 1U, 0);
    }
    hist[static_cast<std::size_t>(depth)] += 1;
}

std::vector<int> sorted_legal_action_indices(const GameState& state) {
    const auto legal_moves = findAllValidMoves(state);
    std::vector<int> actions;
    actions.reserve(legal_moves.size());
    for (const Move& move : legal_moves) {
        actions.push_back(moveToActionIndex(move));
    }
    std::sort(actions.begin(), actions.end());
    actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
    return actions;
}

int reserve_action_index_for_faceup_card(int tier, int slot) {
    if (tier < 0 || tier >= 3 || slot < 0 || slot >= 4) {
        return -1;
    }
    return 15 + tier * 4 + slot;
}

bool noble_claimable_with_extra_bonus(const Tokens& bonuses, const Noble& noble, Color gained_color) {
    Tokens prospective_bonuses = bonuses;
    prospective_bonuses[gained_color] += 1;
    return prospective_bonuses.white >= noble.requirements.white &&
           prospective_bonuses.blue  >= noble.requirements.blue &&
           prospective_bonuses.green >= noble.requirements.green &&
           prospective_bonuses.red   >= noble.requirements.red &&
           prospective_bonuses.black >= noble.requirements.black;
}

std::vector<int> collect_winning_threat_reserve_actions(
    const GameState& root_state,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    const int current_player = root_state.current_player;
    if (current_player != 0 && current_player != 1) {
        throw std::runtime_error("Native MCTS encountered invalid current_player while scanning threats");
    }
    const int opponent = 1 - current_player;
    const Player& opponent_player = root_state.players[static_cast<std::size_t>(opponent)];
    GameState threat_state = root_state;
    threat_state.current_player = opponent;
    threat_state.is_return_phase = false;
    threat_state.is_noble_choice_phase = false;
    const auto opponent_mask = getValidMoveMask(threat_state);
    std::vector<int> reserve_actions;
    reserve_actions.reserve(4);

    for (int tier = 0; tier < 3; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            const Card& card = root_state.faceup[tier][static_cast<std::size_t>(slot)];
            if (card.id <= 0) {
                continue;
            }
            const int buy_action = tier * 4 + slot;
            if (opponent_mask[static_cast<std::size_t>(buy_action)] == 0) {
                continue;
            }

            int resulting_points = opponent_player.points + card.points;
            for (int noble_idx = 0; noble_idx < root_state.noble_count; ++noble_idx) {
                const Noble& noble = root_state.available_nobles[static_cast<std::size_t>(noble_idx)];
                if (noble_claimable_with_extra_bonus(opponent_player.bonuses, noble, card.color)) {
                    resulting_points += noble.points;
                    break;
                }
            }
            if (resulting_points < 15) {
                continue;
            }

            const int reserve_action = reserve_action_index_for_faceup_card(tier, slot);
            if (reserve_action >= 0 && legal_mask[static_cast<std::size_t>(reserve_action)] != 0) {
                reserve_actions.push_back(reserve_action);
            }
        }
    }

    std::sort(reserve_actions.begin(), reserve_actions.end());
    reserve_actions.erase(std::unique(reserve_actions.begin(), reserve_actions.end()), reserve_actions.end());
    return reserve_actions;
}

void renormalize_legal_priors(
    std::array<float, kActionDim>& priors,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    double sum = 0.0;
    int legal_count = 0;
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
            priors[static_cast<std::size_t>(a)] = 0.0f;
            continue;
        }
        ++legal_count;
        sum += static_cast<double>(priors[static_cast<std::size_t>(a)]);
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        const float uniform = legal_count > 0 ? (1.0f / static_cast<float>(legal_count)) : 0.0f;
        for (int a = 0; a < kActionDim; ++a) {
            priors[static_cast<std::size_t>(a)] = (legal_mask[static_cast<std::size_t>(a)] != 0) ? uniform : 0.0f;
        }
        return;
    }
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] != 0) {
            priors[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(priors[static_cast<std::size_t>(a)]) / sum
            );
        } else {
            priors[static_cast<std::size_t>(a)] = 0.0f;
        }
    }
}

int apply_root_blocking_prior_adjustment(
    std::array<float, kActionDim>& priors,
    const GameState& root_state,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    const std::vector<int> threat_reserve_actions =
        collect_winning_threat_reserve_actions(root_state, legal_mask);
    if (threat_reserve_actions.empty()) {
        return -1;
    }

    for (int action : threat_reserve_actions) {
        priors[static_cast<std::size_t>(action)] *= kBlockingPriorBoost;
    }
    renormalize_legal_priors(priors, legal_mask);

    int immediate_action = -1;
    float best_prior = kBlockingPriorImmediateThreshold;
    for (int action : threat_reserve_actions) {
        const float prior = priors[static_cast<std::size_t>(action)];
        if (prior > best_prior) {
            best_prior = prior;
            immediate_action = action;
        }
    }
    return immediate_action;
}

// Compute terminal value from the perspective of a player evaluating the outcome.
// winner: 0/1 for the winning player, -1 for draw
// evaluating_player_id: player whose perspective we're taking (0 or 1)
// Returns: +1 if evaluating_player_id is the winner, -1 if loser, 0 if draw
float get_terminal_value_from_player_perspective(int winner, int evaluating_player_id) {
    if (winner == -1) {
        // Draw: neutral value from all perspectives
        return 0.0f;
    }
    if (winner != 0 && winner != 1) {
        throw std::runtime_error("Unexpected winner value in native MCTS");
    }
    // Winner sees +1 from their perspective, loser sees -1 from theirs
    return winner == evaluating_player_id ? 1.0f : -1.0f;
}


template <typename Rng>
void apply_dirichlet_root_noise(
    std::array<float, kActionDim>& priors,
    const std::array<std::uint8_t, kActionDim>& mask,
    float epsilon,
    float alpha_total,
    Rng& rng
) {
    if (epsilon <= 0.0f) {
        return;
    }
    std::vector<int> legal;
    legal.reserve(kActionDim);
    for (int i = 0; i < kActionDim; ++i) {
        if (mask[static_cast<std::size_t>(i)] != 0) {
            legal.push_back(i);
        } else {
            priors[static_cast<std::size_t>(i)] = 0.0f;
        }
    }
    if (legal.size() < 2U) {
        return;
    }

    const double alpha = static_cast<double>(alpha_total) / static_cast<double>(legal.size());
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(legal.size(), 0.0);
    double noise_sum = 0.0;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    if (!(noise_sum > 0.0) || !std::isfinite(noise_sum)) {
        for (double& x : noise) {
            x = 1.0 / static_cast<double>(legal.size());
        }
    } else {
        for (double& x : noise) {
            x /= noise_sum;
        }
    }

    double mixed_sum = 0.0;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        const int a = legal[i];
        const double mixed =
            (1.0 - static_cast<double>(epsilon)) * static_cast<double>(priors[static_cast<std::size_t>(a)]) +
            static_cast<double>(epsilon) * noise[i];
        priors[static_cast<std::size_t>(a)] = static_cast<float>(mixed);
        mixed_sum += mixed;
    }
    if (!(mixed_sum > 0.0) || !std::isfinite(mixed_sum)) {
        const float u = 1.0f / static_cast<float>(legal.size());
        for (int a : legal) {
            priors[static_cast<std::size_t>(a)] = u;
        }
    } else {
        for (int a : legal) {
            priors[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(priors[static_cast<std::size_t>(a)]) / mixed_sum
            );
        }
    }
}

int forced_playout_target_visits_for_child(float total_parent_visits, float k, float prior) {
    if (!(k > 0.0f) || !(total_parent_visits > 0.0f) || !(prior > 0.0f) || !std::isfinite(static_cast<double>(prior))) {
        return 0;
    }
    const double target =
        std::sqrt(static_cast<double>(k) * static_cast<double>(prior) * static_cast<double>(total_parent_visits));
    if (!(target > 0.0) || !std::isfinite(target)) {
        return 0;
    }
    return std::max(0, static_cast<int>(std::floor(target)));
}

float puct_score_for_action(
    const MCTSNode& node,
    int action,
    float total_parent_visits,
    float c_puct,
    float eps,
    int override_n = -1
) {
    const int raw_n = node.visit_count[static_cast<std::size_t>(action)];
    const float n = static_cast<float>((override_n >= 0) ? override_n : raw_n);
    const float q = (raw_n <= 0) ? 0.0f : (node.value_sum[static_cast<std::size_t>(action)] / static_cast<float>(raw_n));
    const float sqrt_parent = std::sqrt(total_parent_visits + eps);
    const float u = c_puct * node.priors[static_cast<std::size_t>(action)] * sqrt_parent / (1.0f + n);
    return q + u;
}

int select_puct_action(
    const std::vector<MCTSNode>& nodes,
    int node_index,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps,
    bool use_forced_playouts,
    float forced_playouts_k
) {
    const MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
    float parent_n = 0.0f;
    for (int i = 0; i < kActionDim; ++i) {
        parent_n += static_cast<float>(node.visit_count[static_cast<std::size_t>(i)]);
    }
    int best_action = -1;
    bool best_forced = false;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < kActionDim; ++action) {
        if (legal_mask[static_cast<std::size_t>(action)] == 0) {
            continue;
        }
        const int child_idx = node.child_index[static_cast<std::size_t>(action)];
        if (child_idx >= 0 && nodes[static_cast<std::size_t>(child_idx)].pending_eval) {
            continue;
        }
        const int n = node.visit_count[static_cast<std::size_t>(action)];
        const int n_forced = (use_forced_playouts && node_index == 0)
                                 ? forced_playout_target_visits_for_child(
                                       parent_n,
                                       forced_playouts_k,
                                       node.priors[static_cast<std::size_t>(action)]
                                   )
                                 : 0;
        const bool forced = n_forced > 0 && n < n_forced;
        const float score = puct_score_for_action(node, action, parent_n, c_puct, eps);
        if (best_action < 0 || (forced && !best_forced) || (forced == best_forced && score > best_score)) {
            best_score = score;
            best_action = action;
            best_forced = forced;
        }
    }
    return best_action;
}

int resolve_tree_worker_limit() {
    const char* raw = std::getenv(kTreeWorkersEnvVar);
    if (raw != nullptr && raw[0] != '\0') {
        char* end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && end != nullptr && *end == '\0' && parsed > 0L) {
            return static_cast<int>(parsed);
        }
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    const int auto_workers = hw > 0U ? static_cast<int>(hw) : 1;
    return std::max(1, std::min(auto_workers, kMaxAutoTreeWorkers));
}

int resolve_ismcts_trace_root_action() {
    const char* raw = std::getenv(kISMCTSTraceRootActionEnvVar);
    if (raw == nullptr || raw[0] == '\0') {
        return -1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
        return -1;
    }
    if (parsed < 0 || parsed >= kActionDim) {
        return -1;
    }
    return static_cast<int>(parsed);
}

void maybe_trace_ismcts_root_backup(
    int trace_root_action,
    const std::vector<PathStep>& path,
    float leaf_value,
    bool terminal_leaf
) {
    if (trace_root_action < 0 || path.empty()) {
        return;
    }
    const PathStep& root_step = path.front();
    if (root_step.node_index != 0 || root_step.action != trace_root_action) {
        return;
    }

    float value = leaf_value;
    bool found_root = false;
    float root_backed = 0.0f;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        const float backed = it->same_player ? value : -value;
        if (it->node_index == 0 && it->action == trace_root_action) {
            root_backed = backed;
            found_root = true;
        }
        value = backed;
    }
    if (!found_root) {
        return;
    }

    std::ostringstream oss;
    oss << "[ISMCTS_TRACE] root_action=" << trace_root_action
        << " terminal_leaf=" << (terminal_leaf ? 1 : 0)
        << " leaf_value=" << leaf_value
        << " root_backed=" << root_backed
        << " path_actions=";
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << path[i].action;
    }
    oss << " path_same=";
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << (path[i].same_player ? 1 : 0);
    }
    std::cout << oss.str() << std::endl;
}

float random_keep_penalty_for_deck_reserve_action(int action) {
    float max_penalty = 0.0f;
    if (action == 27) {
        max_penalty = kISMCTSDeckReserveKeepPenaltyTier1Max;
    } else if (action == 28) {
        max_penalty = kISMCTSDeckReserveKeepPenaltyTier2Max;
    } else if (action == 29) {
        max_penalty = kISMCTSDeckReserveKeepPenaltyTier3Max;
    } else {
        return 0.0f;
    }
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, max_penalty);
    return dist(rng);
}

int select_puct_action_with_pending_flags(
    const MCTSNode& node,
    int node_index,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps,
    const std::atomic_uint8_t* pending_flags,
    int root_min_visit_floor,
    bool use_forced_playouts,
    float forced_playouts_k
) {
    float parent_n = 0.0f;
    for (int i = 0; i < kActionDim; ++i) {
        parent_n += static_cast<float>(node.visit_count[static_cast<std::size_t>(i)]);
    }

    // Root-only exploration floor: ensure every legal root action gets at least
    // N visits before pure PUCT competition dominates.
    if (node_index == 0 && root_min_visit_floor > 0) {
        int floor_best_action = -1;
        int floor_best_visits = std::numeric_limits<int>::max();
        float floor_best_score = -std::numeric_limits<float>::infinity();
        for (int action = 0; action < kActionDim; ++action) {
            if (legal_mask[static_cast<std::size_t>(action)] == 0) {
                continue;
            }
            const int child_idx = node.child_index[static_cast<std::size_t>(action)];
            if (child_idx >= 0 &&
                pending_flags[static_cast<std::size_t>(child_idx)].load(std::memory_order_acquire) != 0U) {
                continue;
            }
            const int n = node.visit_count[static_cast<std::size_t>(action)];
            if (n >= root_min_visit_floor) {
                continue;
            }
            const float score = puct_score_for_action(node, action, parent_n, c_puct, eps);
            if (floor_best_action < 0 ||
                n < floor_best_visits ||
                (n == floor_best_visits && score > floor_best_score)) {
                floor_best_action = action;
                floor_best_visits = n;
                floor_best_score = score;
            }
        }
        if (floor_best_action >= 0) {
            return floor_best_action;
        }
    }

    int best_action = -1;
    bool best_forced = false;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < kActionDim; ++action) {
        if (legal_mask[static_cast<std::size_t>(action)] == 0) {
            continue;
        }
        const int child_idx = node.child_index[static_cast<std::size_t>(action)];
        if (child_idx >= 0 &&
            pending_flags[static_cast<std::size_t>(child_idx)].load(std::memory_order_acquire) != 0U) {
            continue;
        }
        const int n = node.visit_count[static_cast<std::size_t>(action)];
        const int n_forced = (use_forced_playouts && node_index == 0)
                                 ? forced_playout_target_visits_for_child(
                                       parent_n,
                                       forced_playouts_k,
                                       node.priors[static_cast<std::size_t>(action)]
                                   )
                                 : 0;
        const bool forced = n_forced > 0 && n < n_forced;
        const float score = puct_score_for_action(node, action, parent_n, c_puct, eps);
        if (best_action < 0 || (forced && !best_forced) || (forced == best_forced && score > best_score)) {
            best_score = score;
            best_action = action;
            best_forced = forced;
        }
    }
    return best_action;
}

int select_puct_action_for_infoset(
    const ISMCTSNode& node,
    int node_index,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps,
    int root_min_visit_floor
) {
    if (node_index == 0 && root_min_visit_floor > 0) {
        int floor_best_action = -1;
        int floor_best_visits = std::numeric_limits<int>::max();
        float floor_best_score = -std::numeric_limits<float>::infinity();
        const float parent_n_floor = static_cast<float>(std::max(node.total_visit_count, 1));
        for (int action = 0; action < kActionDim; ++action) {
            if (legal_mask[static_cast<std::size_t>(action)] == 0) {
                continue;
            }
            const int n = node.visit_count[static_cast<std::size_t>(action)];
            if (n >= root_min_visit_floor) {
                continue;
            }
            const float q = (n <= 0) ? 0.0f : (node.value_sum[static_cast<std::size_t>(action)] / static_cast<float>(n));
            const float u =
                c_puct * node.priors[static_cast<std::size_t>(action)] * std::sqrt(parent_n_floor + eps) /
                (1.0f + static_cast<float>(n));
            const float random_keep_penalty = (node_index == 0) ? random_keep_penalty_for_deck_reserve_action(action) : 0.0f;
            const float score = q + u - random_keep_penalty;
            if (floor_best_action < 0 ||
                n < floor_best_visits ||
                (n == floor_best_visits && score > floor_best_score)) {
                floor_best_action = action;
                floor_best_visits = n;
                floor_best_score = score;
            }
        }
        if (floor_best_action >= 0) {
            return floor_best_action;
        }
    }

    const float parent_n = static_cast<float>(std::max(node.total_visit_count, 1));
    int best_action = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < kActionDim; ++action) {
        if (legal_mask[static_cast<std::size_t>(action)] == 0) {
            continue;
        }
        const int n = node.visit_count[static_cast<std::size_t>(action)];
        const float q = (n <= 0) ? 0.0f : (node.value_sum[static_cast<std::size_t>(action)] / static_cast<float>(n));
        const float u =
            c_puct * node.priors[static_cast<std::size_t>(action)] * std::sqrt(parent_n + eps) / (1.0f + static_cast<float>(n));
        const float random_keep_penalty = (node_index == 0) ? random_keep_penalty_for_deck_reserve_action(action) : 0.0f;
        const float score = q + u - random_keep_penalty;
        if (best_action < 0 || score > best_score) {
            best_action = action;
            best_score = score;
        }
    }
    return best_action;
}

template <typename Rng>
void sample_root_hidden_information(GameState& state, Rng& rng) {
    // Determinize deck order first.
    for (int tier = 0; tier < 3; ++tier) {
        std::shuffle(state.deck[tier].begin(), state.deck[tier].end(), rng);
    }

    const int current_player = state.current_player;
    if (current_player != 0 && current_player != 1) {
        throw std::runtime_error("Native MCTS root determinization encountered invalid current_player");
    }
    const int opponent = 1 - current_player;
    Player& opponent_player = state.players[static_cast<std::size_t>(opponent)];

    std::array<std::vector<int>, 3> hidden_slot_indices_by_tier;
    std::array<std::vector<Card>, 3> hidden_cards_by_tier;

    // Pass 1: collect hidden reserved slot indices/tier/card before any overwrite.
    for (int slot = 0; slot < static_cast<int>(opponent_player.reserved.size()); ++slot) {
        const ReservedCard& reserved = opponent_player.reserved[static_cast<std::size_t>(slot)];
        if (reserved.is_public) {
            continue;
        }
        const int tier = reserved.card.level - 1;
        if (tier < 0 || tier >= 3) {
            throw std::runtime_error("Native MCTS root determinization encountered hidden reserved card with invalid tier");
        }
        hidden_slot_indices_by_tier[static_cast<std::size_t>(tier)].push_back(slot);
        hidden_cards_by_tier[static_cast<std::size_t>(tier)].push_back(reserved.card);
    }

    for (int tier = 0; tier < 3; ++tier) {
        const auto& slot_indices = hidden_slot_indices_by_tier[static_cast<std::size_t>(tier)];
        const auto& hidden_cards = hidden_cards_by_tier[static_cast<std::size_t>(tier)];
        if (slot_indices.empty()) {
            continue;
        }

        std::vector<Card> pool;
        pool.reserve(state.deck[tier].size() + hidden_cards.size());
        pool.insert(pool.end(), state.deck[tier].begin(), state.deck[tier].end());
        pool.insert(pool.end(), hidden_cards.begin(), hidden_cards.end());
        std::shuffle(pool.begin(), pool.end(), rng);

        const std::size_t hidden_count = slot_indices.size();
        if (pool.size() < hidden_count) {
            throw std::runtime_error("Native MCTS root determinization built an invalid hidden-card pool");
        }
        for (std::size_t i = 0; i < hidden_count; ++i) {
            const int slot = slot_indices[i];
            if (slot < 0 || slot >= static_cast<int>(opponent_player.reserved.size())) {
                throw std::runtime_error("Native MCTS root determinization encountered invalid reserved slot index");
            }
            // Keep slot identity and visibility; only randomize the hidden card identity.
            opponent_player.reserved[static_cast<std::size_t>(slot)].card = pool[i];
        }
        state.deck[tier].assign(pool.begin() + static_cast<std::ptrdiff_t>(hidden_count), pool.end());
    }
}

template <typename Rng>
int sample_action_from_visits(
    const std::array<float, kActionDim>& visit_probs,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    int turns_taken,
    int temperature_moves,
    float temperature,
    Rng& rng
) {
    std::vector<int> legal;
    legal.reserve(kActionDim);
    for (int i = 0; i < kActionDim; ++i) {
        if (legal_mask[static_cast<std::size_t>(i)] != 0) {
            legal.push_back(i);
        }
    }
    if (legal.empty()) {
        throw std::runtime_error("No legal actions for final native MCTS action selection");
    }
    if (turns_taken >= temperature_moves) {
        int best_action = legal.front();
        float best_prob = visit_probs[static_cast<std::size_t>(best_action)];
        for (int a : legal) {
            const float p = visit_probs[static_cast<std::size_t>(a)];
            if (p > best_prob) {
                best_prob = p;
                best_action = a;
            }
        }
        return best_action;
    }

    std::vector<double> weights;
    weights.reserve(legal.size());
    if (temperature <= 0.0f) {
        for (int a : legal) {
            weights.push_back(static_cast<double>(visit_probs[static_cast<std::size_t>(a)]));
        }
    } else {
        for (int a : legal) {
            const double base = static_cast<double>(visit_probs[static_cast<std::size_t>(a)]);
            weights.push_back(temperature == 1.0f ? base : std::pow(base, 1.0 / static_cast<double>(temperature)));
        }
    }
    double weight_sum = 0.0;
    for (double w : weights) {
        weight_sum += w;
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        weights.assign(legal.size(), 1.0);
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return legal[static_cast<std::size_t>(dist(rng))];
}

std::array<float, kActionDim> visit_probs_from_counts(
    const MCTSNode& root,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    std::array<float, kActionDim> probs{};
    double total_visits = 0.0;
    int legal_count = 0;
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] != 0) {
            total_visits += static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]);
            ++legal_count;
        }
    }
    if (total_visits > 0.0) {
        for (int a = 0; a < kActionDim; ++a) {
            probs[static_cast<std::size_t>(a)] =
                (legal_mask[static_cast<std::size_t>(a)] != 0)
                    ? static_cast<float>(static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]) / total_visits)
                    : 0.0f;
        }
    } else {
        const float u = legal_count > 0 ? (1.0f / static_cast<float>(legal_count)) : 0.0f;
        for (int a = 0; a < kActionDim; ++a) {
            probs[static_cast<std::size_t>(a)] = (legal_mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
        }
    }

    double prob_sum = 0.0;
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
            probs[static_cast<std::size_t>(a)] = 0.0f;
        }
        prob_sum += static_cast<double>(probs[static_cast<std::size_t>(a)]);
    }
    if (prob_sum > 0.0 && std::isfinite(prob_sum)) {
        for (int a = 0; a < kActionDim; ++a) {
            probs[static_cast<std::size_t>(a)] =
                static_cast<float>(static_cast<double>(probs[static_cast<std::size_t>(a)]) / prob_sum);
        }
    }
    return probs;
}

std::array<float, kActionDim> visit_probs_from_counts(
    const ISMCTSNode& root,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    std::array<float, kActionDim> probs{};
    double total_visits = 0.0;
    int legal_count = 0;
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] != 0) {
            total_visits += static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]);
            ++legal_count;
        }
    }
    if (total_visits > 0.0) {
        for (int a = 0; a < kActionDim; ++a) {
            probs[static_cast<std::size_t>(a)] =
                (legal_mask[static_cast<std::size_t>(a)] != 0)
                    ? static_cast<float>(static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]) / total_visits)
                    : 0.0f;
        }
    } else {
        const float u = legal_count > 0 ? (1.0f / static_cast<float>(legal_count)) : 0.0f;
        for (int a = 0; a < kActionDim; ++a) {
            probs[static_cast<std::size_t>(a)] = (legal_mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
        }
    }
    return probs;
}

void set_ismcts_priors_from_logits(
    ISMCTSNode& node,
    const std::array<std::uint8_t, kActionDim>& mask,
    const float* logits
) {
    int legal_count = 0;
    bool has_finite_legal_score = false;
    float max_score = -std::numeric_limits<float>::infinity();
    for (int a = 0; a < kActionDim; ++a) {
        if (mask[static_cast<std::size_t>(a)] == 0) {
            node.priors[static_cast<std::size_t>(a)] = 0.0f;
            continue;
        }
        ++legal_count;
        const float s = logits[static_cast<std::size_t>(a)];
        if (std::isfinite(static_cast<double>(s))) {
            if (!has_finite_legal_score || s > max_score) {
                max_score = s;
            }
            has_finite_legal_score = true;
        }
    }

    double sum = 0.0;
    if (has_finite_legal_score) {
        for (int a = 0; a < kActionDim; ++a) {
            if (mask[static_cast<std::size_t>(a)] == 0) {
                node.priors[static_cast<std::size_t>(a)] = 0.0f;
                continue;
            }
            const float s = logits[static_cast<std::size_t>(a)];
            if (!std::isfinite(static_cast<double>(s))) {
                node.priors[static_cast<std::size_t>(a)] = 0.0f;
                continue;
            }
            const float w = std::exp(s - max_score);
            node.priors[static_cast<std::size_t>(a)] = w;
            sum += static_cast<double>(w);
        }
    }

    if (!has_finite_legal_score || !(sum > 0.0) || !std::isfinite(sum)) {
        const float u = 1.0f / static_cast<float>(legal_count);
        for (int a = 0; a < kActionDim; ++a) {
            node.priors[static_cast<std::size_t>(a)] =
                (mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
        }
        return;
    }

    for (int a = 0; a < kActionDim; ++a) {
        if (mask[static_cast<std::size_t>(a)] != 0) {
            node.priors[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(node.priors[static_cast<std::size_t>(a)]) / sum
            );
        } else {
            node.priors[static_cast<std::size_t>(a)] = 0.0f;
        }
    }
}

int apply_root_blocking_prior_adjustment(
    ISMCTSNode& node,
    const GameState& root_state,
    const std::array<std::uint8_t, kActionDim>& legal_mask
) {
    const std::vector<int> threat_reserve_actions =
        collect_winning_threat_reserve_actions(root_state, legal_mask);
    if (threat_reserve_actions.empty()) {
        return -1;
    }

    for (int action : threat_reserve_actions) {
        node.priors[static_cast<std::size_t>(action)] *= kBlockingPriorBoost;
    }
    renormalize_legal_priors(node.priors, legal_mask);

    int immediate_action = -1;
    float best_prior = kBlockingPriorImmediateThreshold;
    for (int action : threat_reserve_actions) {
        const float prior = node.priors[static_cast<std::size_t>(action)];
        if (prior > best_prior) {
            best_prior = prior;
            immediate_action = action;
        }
    }
    return immediate_action;
}

ISMCTSNode evaluate_ismcts_root_node(
    const GameState& root_state,
    const NodeMetadata& root_data,
    pybind11::function& evaluator
) {
    ISMCTSNode root_node;
    pybind11::gil_scoped_acquire acquire;
    const pybind11::ssize_t batch = 1;
    pybind11::array_t<float> states({batch, static_cast<pybind11::ssize_t>(kStateDim)});
    pybind11::array_t<bool> masks({batch, static_cast<pybind11::ssize_t>(kActionDim)});
    const auto root_encoded = state_encoder::encode_state(root_state);
    std::memcpy(states.mutable_data(0, 0), root_encoded.data(), sizeof(float) * static_cast<std::size_t>(kStateDim));
    auto masks_view = masks.mutable_unchecked<2>();
    for (int j = 0; j < kActionDim; ++j) {
        masks_view(0, j) = (root_data.mask[static_cast<std::size_t>(j)] != 0);
    }
    pybind11::object out_obj = evaluator(states, masks);
    pybind11::tuple out = out_obj.cast<pybind11::tuple>();
    if (out.size() != 2) {
        throw std::runtime_error("ISMCTS evaluator must return (priors, values)");
    }
    auto priors_arr =
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[0]);
    if (!priors_arr) {
        throw std::runtime_error("ISMCTS evaluator outputs must be float arrays");
    }
    if (priors_arr.ndim() != 2 ||
        priors_arr.shape(0) != batch ||
        priors_arr.shape(1) != static_cast<pybind11::ssize_t>(kActionDim)) {
        throw std::runtime_error("ISMCTS evaluator priors must have shape (B, ACTION_DIM)");
    }
    set_ismcts_priors_from_logits(root_node, root_data.mask, priors_arr.data(0, 0));
    return root_node;
}

ISMCTSWorkerResult run_native_ismcts_worker(
    const GameState& root_state,
    const NodeMetadata& root_data,
    const ISMCTSNode& root_node,
    int key_observer_player,
    pybind11::function& evaluator,
    int num_simulations,
    float c_puct,
    float eps,
    int eval_batch_size,
    std::uint64_t rng_seed
) {
    std::vector<ISMCTSNode> nodes;
    nodes.reserve(static_cast<std::size_t>(num_simulations + 1));
    std::unordered_map<InfoSetKey, int, InfoSetKeyHasher> node_lookup;
    node_lookup.reserve(static_cast<std::size_t>(num_simulations + 1));

    const InfoSetKey root_key{state_encoder::build_raw_state_for_observer(root_state, key_observer_player)};
    nodes.push_back(root_node);
    node_lookup.emplace(root_key, 0);

    std::mt19937_64 rng(rng_seed);
    std::vector<ISPendingLeafEval> pending;
    std::unordered_set<InfoSetKey, InfoSetKeyHasher> pending_keys;
    pending.reserve(static_cast<std::size_t>(eval_batch_size));
    const int trace_root_action = resolve_ismcts_trace_root_action();

    int completed = 0;
    int total_slots_requested = 0;
    int total_slots_drop_pending_eval = 0;
    int total_slots_drop_no_action = 0;
    int total_slots_drop_cycle = 0;

    while (completed < num_simulations) {
        const int target_batch = std::min(eval_batch_size, num_simulations - completed);
        const int completed_before_batch = completed;
        total_slots_requested += target_batch;
        pending.clear();
        pending_keys.clear();

        for (int slot = 0; slot < target_batch; ++slot) {
            GameState sim_state = root_state;
            sample_root_hidden_information(sim_state, rng);
            int node_index = 0;
            std::vector<PathStep> path;
            path.reserve(64);
            std::unordered_set<InfoSetKey, InfoSetKeyHasher> visited_in_path;
            const InfoSetKey root_key{state_encoder::build_raw_state_for_observer(root_state, key_observer_player)};
            visited_in_path.insert(root_key);

            while (true) {
                const NodeMetadata node_data = build_node_metadata(sim_state);
                if (node_data.terminal.is_terminal) {
                    // Terminal value from the perspective of the current player at terminal state
                    float value = get_terminal_value_from_player_perspective(
                        node_data.terminal.winner,
                        node_data.terminal.current_player_id
                    );
                    maybe_trace_ismcts_root_backup(trace_root_action, path, value, true);
                    // Backup through path: value represents the player-to-move at that state
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        const float backed = it->same_player ? value : -value;
                        ISMCTSNode& node = nodes[static_cast<std::size_t>(it->node_index)];
                        node.total_visit_count += 1;
                        node.visit_count[static_cast<std::size_t>(it->action)] += 1;
                        node.value_sum[static_cast<std::size_t>(it->action)] += backed;
                        value = backed;
                    }
                    completed += 1;
                    break;
                }

                ISMCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
                const int action =
                    select_puct_action_for_infoset(node, node_index, node_data.mask, c_puct, eps, kISMCTSRootMinVisitFloor);
                if (action < 0) {
                    total_slots_drop_no_action += 1;
                    break;
                }

                const int parent_to_play = sim_state.current_player;
                applyMove(sim_state, actionIndexToMove(action));
                const bool same_player = sim_state.current_player == parent_to_play;
                path.push_back(PathStep{node_index, action, same_player, false});

                const InfoSetKey next_key{state_encoder::build_raw_state_for_observer(sim_state, key_observer_player)};
                if (visited_in_path.find(next_key) != visited_in_path.end()) {
                    total_slots_drop_cycle += 1;
                    break;
                }
                visited_in_path.insert(next_key);
                const auto found = node_lookup.find(next_key);
                if (found != node_lookup.end()) {
                    node_index = found->second;
                    continue;
                }
                if (pending_keys.find(next_key) != pending_keys.end()) {
                    total_slots_drop_pending_eval += 1;
                    break;
                }

                ISPendingLeafEval req;
                req.key = next_key;
                req.state = state_encoder::encode_state(sim_state);
                req.mask = state_encoder::build_legal_mask(sim_state);
                req.path = std::move(path);
                pending_keys.insert(req.key);
                pending.push_back(std::move(req));
                break;
            }
        }

        if (pending.empty()) {
            if (completed > completed_before_batch) {
                continue;
            }
            throw std::runtime_error("Native ISMCTS made no progress while gathering leaves");
        }

        {
            pybind11::gil_scoped_acquire acquire;
            const pybind11::ssize_t batch = static_cast<pybind11::ssize_t>(pending.size());
            pybind11::array_t<float> states({batch, static_cast<pybind11::ssize_t>(kStateDim)});
            pybind11::array_t<bool> masks({batch, static_cast<pybind11::ssize_t>(kActionDim)});
            auto masks_view = masks.mutable_unchecked<2>();
            for (pybind11::ssize_t i = 0; i < batch; ++i) {
                const ISPendingLeafEval& req = pending[static_cast<std::size_t>(i)];
                std::memcpy(states.mutable_data(i, 0), req.state.data(), sizeof(float) * static_cast<std::size_t>(kStateDim));
                for (int j = 0; j < kActionDim; ++j) {
                    masks_view(i, j) = (req.mask[static_cast<std::size_t>(j)] != 0);
                }
            }

            pybind11::object out_obj = evaluator(states, masks);
            pybind11::tuple out = out_obj.cast<pybind11::tuple>();
            if (out.size() != 2) {
                throw std::runtime_error("ISMCTS evaluator must return (priors, values)");
            }
            auto priors_arr =
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[0]);
            auto values_arr =
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[1]);
            if (!priors_arr || !values_arr) {
                throw std::runtime_error("ISMCTS evaluator outputs must be float arrays");
            }
            if (priors_arr.ndim() != 2 ||
                priors_arr.shape(0) != batch ||
                priors_arr.shape(1) != static_cast<pybind11::ssize_t>(kActionDim)) {
                throw std::runtime_error("ISMCTS evaluator priors must have shape (B, ACTION_DIM)");
            }
            if (values_arr.ndim() != 1 || values_arr.shape(0) != batch) {
                throw std::runtime_error("ISMCTS evaluator values must have shape (B,)");
            }
            auto values_view = values_arr.unchecked<1>();

            for (pybind11::ssize_t i = 0; i < batch; ++i) {
                const ISPendingLeafEval& req = pending[static_cast<std::size_t>(i)];
                ISMCTSNode node;
                set_ismcts_priors_from_logits(node, req.mask, priors_arr.data(i, 0));

                const int node_id = static_cast<int>(nodes.size());
                nodes.push_back(node);
                node_lookup.emplace(req.key, node_id);

                float value = values_view(i);
                if (!std::isfinite(static_cast<double>(value))) {
                    throw std::runtime_error("ISMCTS evaluator values contain non-finite entries");
                }
                maybe_trace_ismcts_root_backup(trace_root_action, req.path, value, false);
                // Backup through path: value from the perspective of the player-to-move at this leaf.
                // same_player tracks whether control of the turn stayed with the same player (e.g., gem return, noble choice)
                // or switched to the opponent. This correctly handles multi-step sequences.
                for (auto it = req.path.rbegin(); it != req.path.rend(); ++it) {
                    const float backed = it->same_player ? value : -value;
                    ISMCTSNode& parent = nodes[static_cast<std::size_t>(it->node_index)];
                    parent.total_visit_count += 1;
                    parent.visit_count[static_cast<std::size_t>(it->action)] += 1;
                    parent.value_sum[static_cast<std::size_t>(it->action)] += backed;
                    value = backed;
                }
            }
        }

        completed += static_cast<int>(pending.size());
    }

    ISMCTSWorkerResult result;
    result.root = nodes[0];
    result.search_slots_requested = total_slots_requested;
    result.search_slots_evaluated = completed;
    result.search_slots_drop_pending_eval = total_slots_drop_pending_eval;
    result.search_slots_drop_no_action = total_slots_drop_no_action;
    result.search_slots_drop_cycle = total_slots_drop_cycle;
    return result;
}

std::array<float, kActionDim> prune_policy_target_visit_probs(
    const MCTSNode& root,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps,
    float forced_playouts_k
) {
    std::array<int, kActionDim> pruned_counts = root.visit_count;
    int total_root_visits = 0;
    int best_action = -1;
    int best_visits = -1;
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        total_root_visits += n;
        if (n > best_visits) {
            best_visits = n;
            best_action = a;
        }
    }
    if (best_action < 0 || total_root_visits <= 0) {
        return visit_probs_from_counts(root, legal_mask);
    }

    const float total_visits_f = static_cast<float>(total_root_visits);
    const float best_score = puct_score_for_action(root, best_action, total_visits_f, c_puct, eps);

    for (int a = 0; a < kActionDim; ++a) {
        if (a == best_action || legal_mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        if (n <= 0) {
            continue;
        }
        const int n_forced = forced_playout_target_visits_for_child(
            total_visits_f,
            forced_playouts_k,
            root.priors[static_cast<std::size_t>(a)]
        );
        if (n_forced <= 0) {
            continue;
        }
        const int n_reduced = std::max(1, n - n_forced);
        const float reduced_score = puct_score_for_action(root, a, total_visits_f, c_puct, eps, n_reduced);
        if (reduced_score < best_score) {
            const int after_subtract = n - n_forced;
            pruned_counts[static_cast<std::size_t>(a)] = (after_subtract <= 1) ? 0 : after_subtract;
        }
    }

    std::array<float, kActionDim> probs{};
    double total_pruned = 0.0;
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        total_pruned += static_cast<double>(std::max(0, pruned_counts[static_cast<std::size_t>(a)]));
    }
    if (!(total_pruned > 0.0) || !std::isfinite(total_pruned)) {
        probs[static_cast<std::size_t>(best_action)] = 1.0f;
        return probs;
    }
    for (int a = 0; a < kActionDim; ++a) {
        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
            probs[static_cast<std::size_t>(a)] = 0.0f;
            continue;
        }
        const int n = std::max(0, pruned_counts[static_cast<std::size_t>(a)]);
        probs[static_cast<std::size_t>(a)] = static_cast<float>(static_cast<double>(n) / total_pruned);
    }
    return probs;
}

NodeMetadata build_node_metadata(const GameState& state) {
    NodeMetadata out;
    out.mask = state_encoder::build_legal_mask(state);
    out.terminal = state_encoder::build_terminal_metadata(state);
    return out;
}

std::array<std::uint8_t, kActionDim> restrict_root_mask(
    const std::array<std::uint8_t, kActionDim>& mask,
    int forced_root_action_idx
) {
    if (forced_root_action_idx < 0) {
        return mask;
    }
    if (forced_root_action_idx >= kActionDim) {
        throw std::invalid_argument("forced_root_action_idx out of bounds");
    }
    if (mask[static_cast<std::size_t>(forced_root_action_idx)] == 0) {
        throw std::invalid_argument("forced_root_action_idx must be legal at the root");
    }
    std::array<std::uint8_t, kActionDim> restricted{};
    restricted[static_cast<std::size_t>(forced_root_action_idx)] = 1;
    return restricted;
}

}  // namespace

pybind11::array_t<float> NativeMCTSResult::visit_probs_array() const {
    pybind11::array_t<float> arr(kActionDim);
    std::memcpy(arr.mutable_data(), visit_probs.data(), sizeof(float) * static_cast<std::size_t>(kActionDim));
    return arr;
}

pybind11::array_t<float> NativeMCTSResult::q_values_array() const {
    pybind11::array_t<float> arr(kActionDim);
    std::memcpy(arr.mutable_data(), q_values.data(), sizeof(float) * static_cast<std::size_t>(kActionDim));
    return arr;
}

pybind11::array_t<int> NativeMCTSResult::leaf_depth_histogram_array() const {
    pybind11::array_t<int> arr(static_cast<pybind11::ssize_t>(leaf_depth_histogram.size()));
    if (!leaf_depth_histogram.empty()) {
        std::memcpy(
            arr.mutable_data(),
            leaf_depth_histogram.data(),
            sizeof(int) * leaf_depth_histogram.size()
        );
    }
    return arr;
}

pybind11::array_t<int> NativeMCTSResult::resolved_depth_histogram_array() const {
    pybind11::array_t<int> arr(static_cast<pybind11::ssize_t>(resolved_depth_histogram.size()));
    if (!resolved_depth_histogram.empty()) {
        std::memcpy(
            arr.mutable_data(),
            resolved_depth_histogram.data(),
            sizeof(int) * resolved_depth_histogram.size()
        );
    }
    return arr;
}

NativeMCTSResult run_native_mcts(
    const GameState& root_state,
    pybind11::function evaluator,
    int turns_taken,
    int num_simulations,
    float c_puct,
    int temperature_moves,
    float temperature,
    float eps,
    bool root_dirichlet_noise,
    float root_dirichlet_epsilon,
    float root_dirichlet_alpha_total,
    int eval_batch_size,
    std::uint64_t rng_seed,
    bool use_forced_playouts,
    float forced_playouts_k,
    int forced_root_action_idx
) {
    if (num_simulations <= 0) {
        throw std::invalid_argument("num_simulations must be positive");
    }
    if (!(root_dirichlet_epsilon >= 0.0f && root_dirichlet_epsilon <= 1.0f)) {
        throw std::invalid_argument("root_dirichlet_epsilon must be in [0,1]");
    }
    if (!(root_dirichlet_alpha_total > 0.0f)) {
        throw std::invalid_argument("root_dirichlet_alpha_total must be positive");
    }
    if (eval_batch_size <= 0) {
        throw std::invalid_argument("eval_batch_size must be positive");
    }
    if (!(forced_playouts_k > 0.0f)) {
        throw std::invalid_argument("forced_playouts_k must be positive");
    }

    const NodeMetadata root_data = build_node_metadata(root_state);
    if (root_data.terminal.is_terminal) {
        throw std::invalid_argument("run_mcts called on terminal state");
    }
    const std::array<std::uint8_t, kActionDim> root_mask =
        restrict_root_mask(root_data.mask, forced_root_action_idx);

    bool has_legal = false;
    for (int i = 0; i < kActionDim; ++i) {
        if (root_mask[static_cast<std::size_t>(i)] != 0) {
            has_legal = true;
            break;
        }
    }
    if (!has_legal) {
        throw std::invalid_argument("MCTS root has no legal actions");
    }

    const int max_nodes = num_simulations + 1;
    std::vector<MCTSNode> nodes(static_cast<std::size_t>(max_nodes));
    std::vector<std::mutex> node_mutexes(static_cast<std::size_t>(max_nodes));
    auto pending_flags = std::make_unique<std::atomic_uint8_t[]>(static_cast<std::size_t>(max_nodes));
    for (int i = 0; i < max_nodes; ++i) {
        pending_flags[static_cast<std::size_t>(i)].store(0U, std::memory_order_relaxed);
    }
    std::atomic<int> next_node_index(1);

    std::mt19937_64 rng(rng_seed);
    bool root_noise_applied = false;
    int completed = 0;
    const int tree_worker_limit = resolve_tree_worker_limit();
    std::vector<int> leaf_depth_histogram;
    std::vector<int> resolved_depth_histogram;
    int max_leaf_depth = 0;
    int max_resolved_depth = 0;

    auto evaluate_pending = [&](std::vector<PendingLeafEval>& pending, std::vector<ReadyBackup>& backups) {
        if (pending.empty()) {
            return;
        }

        auto evaluate_batch = [&](const std::vector<std::array<float, kStateDim>>& states_in,
                                  const std::vector<std::array<std::uint8_t, kActionDim>>& masks_in) {
            const pybind11::ssize_t batch = static_cast<pybind11::ssize_t>(states_in.size());
            pybind11::array_t<float> states({batch, static_cast<pybind11::ssize_t>(kStateDim)});
            pybind11::array_t<bool> masks({batch, static_cast<pybind11::ssize_t>(kActionDim)});
            auto masks_view = masks.mutable_unchecked<2>();
            for (pybind11::ssize_t i = 0; i < batch; ++i) {
                float* state_row = states.mutable_data(i, 0);
                std::memcpy(
                    state_row,
                    states_in[static_cast<std::size_t>(i)].data(),
                    sizeof(float) * static_cast<std::size_t>(kStateDim)
                );
                for (int j = 0; j < kActionDim; ++j) {
                    masks_view(i, j) = (masks_in[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] != 0);
                }
            }

            pybind11::object out_obj = evaluator(states, masks);
            pybind11::tuple out = out_obj.cast<pybind11::tuple>();
            if (out.size() != 2) {
                throw std::runtime_error("MCTS evaluator must return (priors, values)");
            }

            auto priors_arr =
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[0]);
            auto values_arr =
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[1]);
            if (!priors_arr || !values_arr) {
                throw std::runtime_error("MCTS evaluator outputs must be float arrays");
            }
            if (priors_arr.ndim() != 2 ||
                priors_arr.shape(0) != batch ||
                priors_arr.shape(1) != static_cast<pybind11::ssize_t>(kActionDim)) {
                throw std::runtime_error("MCTS evaluator priors must have shape (B, ACTION_DIM)");
            }
            if (values_arr.ndim() != 1 || values_arr.shape(0) != batch) {
                throw std::runtime_error("MCTS evaluator values must have shape (B,)");
            }
            return std::make_pair(priors_arr, values_arr);
        };

        auto assign_priors_from_scores =
            [&](MCTSNode& node,
                const std::array<std::uint8_t, kActionDim>& legal_mask,
                const pybind11::detail::unchecked_reference<float, 2>& priors_view,
                pybind11::ssize_t row_idx) {
                int legal_count = 0;
                bool has_finite_legal_score = false;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int a = 0; a < kActionDim; ++a) {
                    if (legal_mask[static_cast<std::size_t>(a)] == 0) {
                        node.priors[static_cast<std::size_t>(a)] = 0.0f;
                        continue;
                    }
                    ++legal_count;
                    const float s = priors_view(row_idx, a);
                    if (std::isfinite(static_cast<double>(s))) {
                        if (!has_finite_legal_score || s > max_score) {
                            max_score = s;
                        }
                        has_finite_legal_score = true;
                    }
                }

                double sum = 0.0;
                if (has_finite_legal_score) {
                    for (int a = 0; a < kActionDim; ++a) {
                        if (legal_mask[static_cast<std::size_t>(a)] == 0) {
                            node.priors[static_cast<std::size_t>(a)] = 0.0f;
                            continue;
                        }
                        const float s = priors_view(row_idx, a);
                        if (!std::isfinite(static_cast<double>(s))) {
                            node.priors[static_cast<std::size_t>(a)] = 0.0f;
                            continue;
                        }
                        const float w = std::exp(s - max_score);
                        node.priors[static_cast<std::size_t>(a)] = w;
                        sum += static_cast<double>(w);
                    }
                }

                if (!has_finite_legal_score || !(sum > 0.0) || !std::isfinite(sum)) {
                    const float u = 1.0f / static_cast<float>(legal_count);
                    for (int a = 0; a < kActionDim; ++a) {
                        node.priors[static_cast<std::size_t>(a)] =
                            (legal_mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
                    }
                } else {
                    renormalize_legal_priors(node.priors, legal_mask);
                }
            };

        std::vector<std::array<float, kStateDim>> initial_states;
        std::vector<std::array<std::uint8_t, kActionDim>> initial_masks;
        initial_states.reserve(pending.size());
        initial_masks.reserve(pending.size());
        for (const PendingLeafEval& req : pending) {
            initial_states.push_back(req.state);
            initial_masks.push_back(req.mask);
        }
        auto [initial_priors_arr, initial_values_arr] = evaluate_batch(initial_states, initial_masks);
        const auto initial_priors_view = initial_priors_arr.unchecked<2>();
        const auto initial_values_view = initial_values_arr.unchecked<1>();

        for (pybind11::ssize_t i = 0; i < static_cast<pybind11::ssize_t>(pending.size()); ++i) {
            PendingLeafEval& req = pending[static_cast<std::size_t>(i)];
            MCTSNode& node = nodes[static_cast<std::size_t>(req.node_index)];
            assign_priors_from_scores(node, req.mask, initial_priors_view, i);

            if (req.node_index == 0) {
                apply_root_blocking_prior_adjustment(node.priors, root_state, req.mask);
            }

            float value = 0.0f;
            GameState rollout_state = req.state_snapshot;
            const int leaf_player = rollout_state.current_player;
            const int tree_depth = static_cast<int>(req.path.size());
            int resolved_depth = tree_depth;
            if (is_modal_subturn_state(rollout_state)) {
                std::array<float, kActionDim> current_policy_scores{};
                for (int a = 0; a < kActionDim; ++a) {
                    current_policy_scores[static_cast<std::size_t>(a)] = initial_priors_view(i, a);
                }

                while (true) {
                    NodeMetadata rollout_meta = build_node_metadata(rollout_state);
                    if (rollout_meta.terminal.is_terminal) {
                        value = get_terminal_value_from_player_perspective(
                            rollout_meta.terminal.winner,
                            rollout_meta.terminal.current_player_id
                        );
                        if (rollout_meta.terminal.current_player_id != leaf_player) {
                            value = -value;
                        }
                        break;
                    }

                    if (!is_modal_subturn_state(rollout_state)) {
                        std::vector<std::array<float, kStateDim>> final_states{state_encoder::encode_state(rollout_state)};
                        std::vector<std::array<std::uint8_t, kActionDim>> final_masks{rollout_meta.mask};
                        auto [final_priors_arr, final_values_arr] = evaluate_batch(final_states, final_masks);
                        (void)final_priors_arr;
                        const auto final_values_view = final_values_arr.unchecked<1>();
                        value = final_values_view(0);
                        if (!std::isfinite(static_cast<double>(value))) {
                            throw std::runtime_error("MCTS evaluator values contain non-finite entries");
                        }
                        if (rollout_state.current_player != leaf_player) {
                            value = -value;
                        }
                        break;
                    }

                    const int action = select_max_policy_legal_action(current_policy_scores, rollout_meta.mask);
                    if (action < 0) {
                        throw std::runtime_error("Modal subturn state has no legal action to resolve");
                    }
                    applyMove(rollout_state, actionIndexToMove(action));
                    resolved_depth += 1;

                    if (rollout_state.current_player != leaf_player) {
                        NodeMetadata resolved_meta = build_node_metadata(rollout_state);
                        if (resolved_meta.terminal.is_terminal) {
                            value = get_terminal_value_from_player_perspective(
                                resolved_meta.terminal.winner,
                                resolved_meta.terminal.current_player_id
                            );
                            value = -value;
                        } else {
                            std::vector<std::array<float, kStateDim>> final_states{state_encoder::encode_state(rollout_state)};
                            std::vector<std::array<std::uint8_t, kActionDim>> final_masks{resolved_meta.mask};
                            auto [final_priors_arr, final_values_arr] = evaluate_batch(final_states, final_masks);
                            (void)final_priors_arr;
                            const auto final_values_view = final_values_arr.unchecked<1>();
                            value = final_values_view(0);
                            if (!std::isfinite(static_cast<double>(value))) {
                                throw std::runtime_error("MCTS evaluator values contain non-finite entries");
                            }
                            value = -value;
                        }
                        break;
                    }

                    NodeMetadata next_meta = build_node_metadata(rollout_state);
                    if (next_meta.terminal.is_terminal) {
                        value = get_terminal_value_from_player_perspective(
                            next_meta.terminal.winner,
                            next_meta.terminal.current_player_id
                        );
                        break;
                    }

                    std::vector<std::array<float, kStateDim>> next_states{state_encoder::encode_state(rollout_state)};
                    std::vector<std::array<std::uint8_t, kActionDim>> next_masks{next_meta.mask};
                    auto [next_priors_arr, next_values_arr] = evaluate_batch(next_states, next_masks);
                    (void)next_values_arr;
                    const auto next_priors_view = next_priors_arr.unchecked<2>();
                    for (int a = 0; a < kActionDim; ++a) {
                        current_policy_scores[static_cast<std::size_t>(a)] = next_priors_view(0, a);
                    }
                }
            } else {
                value = initial_values_view(i);
                if (!std::isfinite(static_cast<double>(value))) {
                    throw std::runtime_error("MCTS evaluator values contain non-finite entries");
                }
            }

            node.expanded = true;
            node.pending_eval = false;
            pending_flags[static_cast<std::size_t>(req.node_index)].store(0U, std::memory_order_release);

            if (req.node_index == 0 && !root_noise_applied && root_dirichlet_noise) {
                apply_dirichlet_root_noise(
                    node.priors,
                    req.mask,
                    root_dirichlet_epsilon,
                    root_dirichlet_alpha_total,
                    rng
                );
                root_noise_applied = true;
            }

            ReadyBackup ready;
            ready.value = value;
            ready.tree_depth = tree_depth > 0 ? tree_depth : -1;
            ready.resolved_depth = resolved_depth > 0 ? resolved_depth : -1;
            ready.path = std::move(req.path);
            backups.push_back(std::move(ready));
        }
    };

    auto rollback_virtual_path = [&](const std::vector<PathStep>& path) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if (!it->virtual_loss_applied) {
                continue;
            }
            std::lock_guard<std::mutex> guard(node_mutexes[static_cast<std::size_t>(it->node_index)]);
            MCTSNode& parent = nodes[static_cast<std::size_t>(it->node_index)];
            int& n = parent.visit_count[static_cast<std::size_t>(it->action)];
            if (n <= 0) {
                throw std::runtime_error("Native MCTS virtual-loss rollback underflow");
            }
            n -= 1;
            parent.value_sum[static_cast<std::size_t>(it->action)] += kVirtualLoss;
        }
    };

    // auto apply_ready_backup_serial = [&](ReadyBackup& ready) {
    //     float value = ready.value;
    //     for (auto it = ready.path.rbegin(); it != ready.path.rend(); ++it) {
    //         const float backed = it->same_player ? value : -value;
    //         MCTSNode& parent = nodes[static_cast<std::size_t>(it->node_index)];
    //         parent.visit_count[static_cast<std::size_t>(it->action)] += 1;
    //         parent.value_sum[static_cast<std::size_t>(it->action)] += backed;
    //         value = backed;
    //     }
    // };

    auto apply_ready_backup_virtual = [&](ReadyBackup& ready) {
        // Backup value through the path from leaf to root.
        // Semantics: at each step, `value` represents the outcome from the current player's perspective.
        // - If same_player=true (e.g., gem return or noble choice during same turn), perspective unchanged
        // - If same_player=false (opponent just moved), flip sign because opponent's +1 is our -1
        // This handles multi-step sequences correctly: TAKE_GEMS (P0) -> RETURN_GEM (P0) -> NOBLE_CHOICE (P0) -> END_TURN (P0->P1)
        float value = ready.value;
        for (auto it = ready.path.rbegin(); it != ready.path.rend(); ++it) {
            const float backed = it->same_player ? value : -value;
            std::lock_guard<std::mutex> guard(node_mutexes[static_cast<std::size_t>(it->node_index)]);
            MCTSNode& parent = nodes[static_cast<std::size_t>(it->node_index)];
            parent.value_sum[static_cast<std::size_t>(it->action)] += (kVirtualLoss + backed);
            value = backed;
        }
    };

    // Pre-expand the root so each simulation traverses at least one root edge.
    {
        {
            std::lock_guard<std::mutex> guard(node_mutexes[0]);
            MCTSNode& root = nodes[0];
            root.pending_eval = true;
            pending_flags[0].store(1U, std::memory_order_release);
        }
        std::vector<PendingLeafEval> pending_root;
        std::vector<ReadyBackup> root_backups;
        pending_root.reserve(1);
        root_backups.reserve(1);
        PendingLeafEval req;
        req.node_index = 0;
        req.state_snapshot = root_state;
        req.state = state_encoder::encode_state(root_state);
        req.mask = root_mask;
        pending_root.push_back(std::move(req));
        evaluate_pending(pending_root, root_backups);
    }

    // Keep boosted root priors, but do not short-circuit search. We want
    // rollout-backed visit/Q statistics even when a blocking reserve exists.

    std::vector<PendingLeafEval> pending;
    pending.reserve(static_cast<std::size_t>(eval_batch_size));
    std::vector<ReadyBackup> backups;
    backups.reserve(static_cast<std::size_t>(eval_batch_size));

    int total_slots_requested = 0;
    int total_slots_drop_pending_eval = 0;
    int total_slots_drop_no_action = 0;

    while (completed < num_simulations) {
        const int target_batch = std::min(eval_batch_size, num_simulations - completed);
        total_slots_requested += target_batch;
        pending.clear();
        backups.clear();
        if (pending.capacity() < static_cast<std::size_t>(target_batch)) {
            pending.reserve(static_cast<std::size_t>(target_batch));
        }
        if (backups.capacity() < static_cast<std::size_t>(target_batch)) {
            backups.reserve(static_cast<std::size_t>(target_batch));
        }

        const int tree_workers = std::max(1, std::min(tree_worker_limit, target_batch));
        if (tree_workers <= 1) {
            pybind11::gil_scoped_release release;
            for (int slot = 0; slot < target_batch; ++slot) {
                GameState sim_state = root_state;
                sample_root_hidden_information(sim_state, rng);
                int node_index = 0;
                std::vector<PathStep> path;
                path.reserve(64);

                while (true) {
                    NodeMetadata node_data = build_node_metadata(sim_state);
                    if (node_index == 0) {
                        node_data.mask = root_mask;
                    }
                    MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
                    if (node_data.terminal.is_terminal) {
                        ReadyBackup ready;
                        // Terminal value from the perspective of the current player at terminal state
                        ready.value = get_terminal_value_from_player_perspective(
                            node_data.terminal.winner,
                            node_data.terminal.current_player_id
                        );
                        ready.tree_depth = static_cast<int>(path.size());
                        ready.resolved_depth = static_cast<int>(path.size());
                        ready.path = std::move(path);
                        backups.push_back(std::move(ready));
                        break;
                    }
                    if (!node.expanded) {
                        if (node.pending_eval) {
                            rollback_virtual_path(path);
                            total_slots_drop_pending_eval += 1;
                            break;
                        }
                        node.pending_eval = true;
                        pending_flags[static_cast<std::size_t>(node_index)].store(1U, std::memory_order_release);
                        PendingLeafEval req;
                        req.node_index = node_index;
                        req.state_snapshot = sim_state;
                        req.state = state_encoder::encode_state(sim_state);
                        req.mask = node_data.mask;
                        req.path = std::move(path);
                        pending.push_back(std::move(req));
                        break;
                    }

                    const int action = select_puct_action_with_pending_flags(
                        node,
                        node_index,
                        node_data.mask,
                        c_puct,
                        eps,
                        pending_flags.get(),
                        0,
                        use_forced_playouts,
                        forced_playouts_k
                    );
                    if (action < 0) {
                        rollback_virtual_path(path);
                        total_slots_drop_no_action += 1;
                        break;
                    }

                    int child_idx = node.child_index[static_cast<std::size_t>(action)];
                    if (child_idx < 0) {
                        const int alloc = next_node_index.fetch_add(1, std::memory_order_relaxed);
                        if (alloc >= max_nodes) {
                            throw std::runtime_error("Native MCTS node capacity exceeded");
                        }
                        child_idx = alloc;
                        node.child_index[static_cast<std::size_t>(action)] = child_idx;
                    }
                    node.visit_count[static_cast<std::size_t>(action)] += 1;
                    node.value_sum[static_cast<std::size_t>(action)] -= kVirtualLoss;

                    const int parent_to_play = node_data.terminal.current_player_id;
                    applyMove(sim_state, actionIndexToMove(action));
                    const bool same_player = sim_state.current_player == parent_to_play;
                    path.push_back(PathStep{node_index, action, same_player, true});
                    node_index = child_idx;
                }
            }
        } else {
            std::vector<std::uint64_t> slot_seeds(static_cast<std::size_t>(target_batch), 0U);
            for (int slot = 0; slot < target_batch; ++slot) {
                slot_seeds[static_cast<std::size_t>(slot)] = rng();
            }

            std::atomic<int> next_slot(0);
            std::atomic<bool> failed(false);
            std::exception_ptr first_exc = nullptr;
            std::mutex exc_mutex;
            std::mutex merge_mutex;
            std::atomic<int> batch_drop_pending_eval(0);
            std::atomic<int> batch_drop_no_action(0);

            auto worker = [&]() {
                std::vector<PendingLeafEval> local_pending;
                std::vector<ReadyBackup> local_backups;
                local_pending.reserve(8);
                local_backups.reserve(8);

                while (!failed.load(std::memory_order_acquire)) {
                    const int slot = next_slot.fetch_add(1, std::memory_order_relaxed);
                    if (slot >= target_batch) {
                        break;
                    }

                    try {
                        GameState sim_state = root_state;
                        std::mt19937_64 slot_rng(slot_seeds[static_cast<std::size_t>(slot)]);
                        sample_root_hidden_information(sim_state, slot_rng);
                        int node_index = 0;
                        std::vector<PathStep> path;
                        path.reserve(64);

                        while (true) {
                            NodeMetadata node_data = build_node_metadata(sim_state);
                            if (node_index == 0) {
                                node_data.mask = root_mask;
                            }
                            if (node_data.terminal.is_terminal) {
                                ReadyBackup ready;
                                // Terminal value from the perspective of the current player at terminal state
                                ready.value = get_terminal_value_from_player_perspective(
                                    node_data.terminal.winner,
                                    node_data.terminal.current_player_id
                                );
                                ready.tree_depth = static_cast<int>(path.size());
                                ready.resolved_depth = static_cast<int>(path.size());
                                ready.path = std::move(path);
                                local_backups.push_back(std::move(ready));
                                break;
                            }

                            int action = -1;
                            int child_idx = -1;
                            {
                                std::lock_guard<std::mutex> guard(node_mutexes[static_cast<std::size_t>(node_index)]);
                                MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
                                if (!node.expanded) {
                                    if (node.pending_eval) {
                                        // Another traversal already owns this leaf eval; cancel temporary virtual loss.
                                        rollback_virtual_path(path);
                                        batch_drop_pending_eval.fetch_add(1, std::memory_order_relaxed);
                                        break;
                                    }
                                    node.pending_eval = true;
                                    pending_flags[static_cast<std::size_t>(node_index)].store(
                                        1U,
                                        std::memory_order_release
                                    );
                                    PendingLeafEval req;
                                    req.node_index = node_index;
                                    req.state_snapshot = sim_state;
                                    req.state = state_encoder::encode_state(sim_state);
                                    req.mask = node_data.mask;
                                    req.path = std::move(path);
                                    local_pending.push_back(std::move(req));
                                    break;
                                }

                                action = select_puct_action_with_pending_flags(
                                    node,
                                    node_index,
                                    node_data.mask,
                                    c_puct,
                                    eps,
                                    pending_flags.get(),
                                    0,
                                    use_forced_playouts,
                                    forced_playouts_k
                                );
                                if (action < 0) {
                                    rollback_virtual_path(path);
                                    batch_drop_no_action.fetch_add(1, std::memory_order_relaxed);
                                    break;
                                }

                                child_idx = node.child_index[static_cast<std::size_t>(action)];
                                if (child_idx < 0) {
                                    const int alloc = next_node_index.fetch_add(1, std::memory_order_relaxed);
                                    if (alloc >= max_nodes) {
                                        throw std::runtime_error("Native MCTS node capacity exceeded");
                                    }
                                    child_idx = alloc;
                                    node.child_index[static_cast<std::size_t>(action)] = child_idx;
                                }
                                node.visit_count[static_cast<std::size_t>(action)] += 1;
                                node.value_sum[static_cast<std::size_t>(action)] -= kVirtualLoss;
                            }

                            const int parent_to_play = node_data.terminal.current_player_id;
                            applyMove(sim_state, actionIndexToMove(action));
                            const bool same_player = sim_state.current_player == parent_to_play;
                            path.push_back(PathStep{node_index, action, same_player, true});
                            node_index = child_idx;
                        }
                    } catch (...) {
                        failed.store(true, std::memory_order_release);
                        std::lock_guard<std::mutex> guard(exc_mutex);
                        if (!first_exc) {
                            first_exc = std::current_exception();
                        }
                        break;
                    }
                }

                if (!local_pending.empty() || !local_backups.empty()) {
                    std::lock_guard<std::mutex> guard(merge_mutex);
                    pending.insert(
                        pending.end(),
                        std::make_move_iterator(local_pending.begin()),
                        std::make_move_iterator(local_pending.end())
                    );
                    backups.insert(
                        backups.end(),
                        std::make_move_iterator(local_backups.begin()),
                        std::make_move_iterator(local_backups.end())
                    );
                }
            };

            {
                pybind11::gil_scoped_release release;
                std::vector<std::thread> workers;
                workers.reserve(static_cast<std::size_t>(tree_workers));
                for (int worker_idx = 0; worker_idx < tree_workers; ++worker_idx) {
                    workers.emplace_back(worker);
                }
                for (std::thread& t : workers) {
                    t.join();
                }
            }

            if (first_exc) {
                std::rethrow_exception(first_exc);
            }

            total_slots_drop_pending_eval += batch_drop_pending_eval.load(std::memory_order_relaxed);
            total_slots_drop_no_action += batch_drop_no_action.load(std::memory_order_relaxed);
        }

        evaluate_pending(pending, backups);

        if (backups.empty()) {
            throw std::runtime_error("Native MCTS made no progress while gathering/evaluating leaves");
        }

        pybind11::gil_scoped_release release;
        for (ReadyBackup& ready : backups) {
            if (ready.tree_depth >= 0) {
                record_depth_histogram(leaf_depth_histogram, ready.tree_depth);
                max_leaf_depth = std::max(max_leaf_depth, ready.tree_depth);
            }
            if (ready.resolved_depth >= 0) {
                record_depth_histogram(resolved_depth_histogram, ready.resolved_depth);
                max_resolved_depth = std::max(max_resolved_depth, ready.resolved_depth);
            }
            apply_ready_backup_virtual(ready);
        }

        completed += static_cast<int>(backups.size());
    }

    const MCTSNode& root = nodes[0];
    NativeMCTSResult result;
    result.search_slots_requested = total_slots_requested;
    result.search_slots_evaluated = completed - 1; // subtract 1 for the pre-expanded root
    result.search_slots_drop_pending_eval = total_slots_drop_pending_eval;
    result.search_slots_drop_no_action = total_slots_drop_no_action;
    result.max_leaf_depth = max_leaf_depth;
    result.max_resolved_depth = max_resolved_depth;
    result.leaf_depth_histogram = std::move(leaf_depth_histogram);
    result.resolved_depth_histogram = std::move(resolved_depth_histogram);
    const auto raw_visit_probs = visit_probs_from_counts(root, root_mask);
    result.chosen_action_idx = sample_action_from_visits(
        raw_visit_probs, root_mask, turns_taken, temperature_moves, temperature, rng
    );
    if (use_forced_playouts) {
        result.visit_probs = prune_policy_target_visit_probs(
            root,
            root_mask,
            c_puct,
            eps,
            forced_playouts_k
        );
    } else {
        result.visit_probs = raw_visit_probs;
    }

    for (int a = 0; a < kActionDim; ++a) {
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        result.q_values[static_cast<std::size_t>(a)] =
            (n <= 0)
                ? 0.0f
                : (root.value_sum[static_cast<std::size_t>(a)] / static_cast<float>(n));
    }

    int best_visit_action = -1;
    int best_visit_count = -1;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        if (n > best_visit_count) {
            best_visit_count = n;
            best_visit_action = a;
        }
    }
    if (best_visit_action >= 0) {
        const int n = root.visit_count[static_cast<std::size_t>(best_visit_action)];
        result.root_best_value =
            (n <= 0)
                ? 0.0f
                : (root.value_sum[static_cast<std::size_t>(best_visit_action)] / static_cast<float>(n));
    }

    return result;
}

NativeMCTSResult run_native_ismcts(
    const GameState& root_state,
    pybind11::function evaluator,
    int num_simulations,
    float c_puct,
    float eps,
    int eval_batch_size,
    std::uint64_t rng_seed,
    int root_parallel_workers,
    int forced_root_action_idx
) {
    if (num_simulations <= 0) {
        throw std::invalid_argument("num_simulations must be positive");
    }
    if (eval_batch_size <= 0) {
        throw std::invalid_argument("eval_batch_size must be positive");
    }
    if (root_parallel_workers <= 0) {
        throw std::invalid_argument("root_parallel_workers must be positive");
    }

    const NodeMetadata root_data = build_node_metadata(root_state);
    if (root_data.terminal.is_terminal) {
        throw std::invalid_argument("run_ismcts called on terminal state");
    }
    const std::array<std::uint8_t, kActionDim> root_mask =
        restrict_root_mask(root_data.mask, forced_root_action_idx);
    bool has_legal = false;
    for (int i = 0; i < kActionDim; ++i) {
        if (root_mask[static_cast<std::size_t>(i)] != 0) {
            has_legal = true;
            break;
        }
    }
    if (!has_legal) {
        throw std::invalid_argument("ISMCTS root has no legal actions");
    }

    NodeMetadata effective_root_data = root_data;
    effective_root_data.mask = root_mask;
    ISMCTSNode root_node = evaluate_ismcts_root_node(root_state, effective_root_data, evaluator);
    // Keep boosted root priors, but do not short-circuit search. We want
    // rollout-backed visit/Q statistics even when a blocking reserve exists.
    apply_root_blocking_prior_adjustment(root_node, root_state, root_mask);

    const int worker_count = std::min(root_parallel_workers, num_simulations);
    const int key_observer_player = root_state.current_player;
    std::vector<int> worker_budgets(static_cast<std::size_t>(worker_count), num_simulations / worker_count);
    for (int worker_idx = 0; worker_idx < (num_simulations % worker_count); ++worker_idx) {
        worker_budgets[static_cast<std::size_t>(worker_idx)] += 1;
    }

    std::mt19937_64 seeder(rng_seed);
    std::vector<std::uint64_t> worker_seeds(static_cast<std::size_t>(worker_count), 0U);
    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
        worker_seeds[static_cast<std::size_t>(worker_idx)] = seeder();
    }

    std::vector<ISMCTSWorkerResult> worker_results(static_cast<std::size_t>(worker_count));
    if (worker_count == 1) {
        pybind11::gil_scoped_release release;
        worker_results[0] = run_native_ismcts_worker(
            root_state,
            effective_root_data,
            root_node,
            key_observer_player,
            evaluator,
            worker_budgets[0],
            c_puct,
            eps,
            eval_batch_size,
            worker_seeds[0]
        );
    } else {
        std::atomic<bool> failed(false);
        std::exception_ptr first_exc = nullptr;
        std::mutex exc_mutex;
        auto worker = [&](int worker_idx) {
            try {
                worker_results[static_cast<std::size_t>(worker_idx)] = run_native_ismcts_worker(
                    root_state,
                    effective_root_data,
                    root_node,
                    key_observer_player,
                    evaluator,
                    worker_budgets[static_cast<std::size_t>(worker_idx)],
                    c_puct,
                    eps,
                    eval_batch_size,
                    worker_seeds[static_cast<std::size_t>(worker_idx)]
                );
            } catch (...) {
                failed.store(true, std::memory_order_release);
                std::lock_guard<std::mutex> guard(exc_mutex);
                if (!first_exc) {
                    first_exc = std::current_exception();
                }
            }
        };

        {
            pybind11::gil_scoped_release release;
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(worker_count));
            for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
                workers.emplace_back(worker, worker_idx);
            }
            for (std::thread& t : workers) {
                t.join();
            }
        }
        if (failed.load(std::memory_order_acquire) && first_exc) {
            std::rethrow_exception(first_exc);
        }
    }

    ISMCTSNode root;
    int total_slots_requested = 0;
    int total_slots_evaluated = 0;
    int total_slots_drop_pending_eval = 0;
    int total_slots_drop_no_action = 0;
    int total_slots_drop_cycle = 0;
    for (const ISMCTSWorkerResult& worker_result : worker_results) {
        root.total_visit_count += worker_result.root.total_visit_count;
        total_slots_requested += worker_result.search_slots_requested;
        total_slots_evaluated += worker_result.search_slots_evaluated;
        total_slots_drop_pending_eval += worker_result.search_slots_drop_pending_eval;
        total_slots_drop_no_action += worker_result.search_slots_drop_no_action;
        total_slots_drop_cycle += worker_result.search_slots_drop_cycle;
        for (int a = 0; a < kActionDim; ++a) {
            root.visit_count[static_cast<std::size_t>(a)] += worker_result.root.visit_count[static_cast<std::size_t>(a)];
            root.value_sum[static_cast<std::size_t>(a)] += worker_result.root.value_sum[static_cast<std::size_t>(a)];
        }
    }

    NativeMCTSResult result;
    result.search_slots_requested = total_slots_requested;
    result.search_slots_evaluated = total_slots_evaluated;
    result.search_slots_drop_pending_eval = total_slots_drop_pending_eval;
    result.search_slots_drop_no_action = total_slots_drop_no_action;
    result.visit_probs = visit_probs_from_counts(root, root_mask);

    int best_action = -1;
    int best_visits = -1;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        result.q_values[static_cast<std::size_t>(a)] =
            (n <= 0) ? 0.0f : (root.value_sum[static_cast<std::size_t>(a)] / static_cast<float>(n));
        if (n > best_visits) {
            best_visits = n;
            best_action = a;
        }
    }
    result.chosen_action_idx = best_action >= 0 ? best_action : 0;
    result.root_best_value =
        (best_action >= 0 && root.visit_count[static_cast<std::size_t>(best_action)] > 0)
            ? (root.value_sum[static_cast<std::size_t>(best_action)] /
               static_cast<float>(root.visit_count[static_cast<std::size_t>(best_action)]))
            : 0.0f;
    return result;
}
