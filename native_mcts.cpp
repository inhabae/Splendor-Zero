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
#include <stdexcept>
#include <thread>
#include <vector>

#include "state_encoder.h"

namespace {

constexpr int kActionDim = state_encoder::ACTION_DIM;
constexpr int kStateDim = state_encoder::STATE_DIM;
constexpr float kVirtualLoss = 1.0f;
constexpr int kMaxAutoTreeWorkers = 32;
constexpr const char* kTreeWorkersEnvVar = "SPLENDOR_MCTS_TREE_WORKERS";

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
    std::array<float, kStateDim> state{};
    std::array<std::uint8_t, kActionDim> mask{};
    std::vector<PathStep> path;
};

struct ReadyBackup {
    float value = 0.0f;
    std::vector<PathStep> path;
};

struct NodeMetadata {
    std::array<std::uint8_t, kActionDim> mask{};
    state_encoder::TerminalMetadata terminal{};
};

float winner_to_value_for_player(int winner, int player_id) {
    if (winner == -1) {
        return 0.0f;
    }
    if (winner != 0 && winner != 1) {
        throw std::runtime_error("Unexpected winner value in native MCTS");
    }
    return winner == player_id ? 1.0f : -1.0f;
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

int select_puct_action(
    const std::vector<MCTSNode>& nodes,
    int node_index,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps
) {
    const MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
    float parent_n = 0.0f;
    for (int i = 0; i < kActionDim; ++i) {
        parent_n += static_cast<float>(node.visit_count[static_cast<std::size_t>(i)]);
    }
    const float sqrt_parent = std::sqrt(parent_n + eps);

    int best_action = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < kActionDim; ++action) {
        if (legal_mask[static_cast<std::size_t>(action)] == 0) {
            continue;
        }
        const int child_idx = node.child_index[static_cast<std::size_t>(action)];
        if (child_idx >= 0 && nodes[static_cast<std::size_t>(child_idx)].pending_eval) {
            continue;
        }
        const float n = static_cast<float>(node.visit_count[static_cast<std::size_t>(action)]);
        const float q = (n <= 0.0f) ? 0.0f : (node.value_sum[static_cast<std::size_t>(action)] / n);
        const float u = c_puct * node.priors[static_cast<std::size_t>(action)] * sqrt_parent / (1.0f + n);
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = action;
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

int select_puct_action_with_pending_flags(
    const MCTSNode& node,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps,
    const std::atomic_uint8_t* pending_flags
) {
    float parent_n = 0.0f;
    for (int i = 0; i < kActionDim; ++i) {
        parent_n += static_cast<float>(node.visit_count[static_cast<std::size_t>(i)]);
    }
    const float sqrt_parent = std::sqrt(parent_n + eps);

    int best_action = -1;
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
        const float n = static_cast<float>(node.visit_count[static_cast<std::size_t>(action)]);
        const float q = (n <= 0.0f) ? 0.0f : (node.value_sum[static_cast<std::size_t>(action)] / n);
        const float u = c_puct * node.priors[static_cast<std::size_t>(action)] * sqrt_parent / (1.0f + n);
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

template <typename Rng>
void sample_root_decks(GameState& state, Rng& rng) {
    for (int tier = 0; tier < 3; ++tier) {
        std::shuffle(state.deck[tier].begin(), state.deck[tier].end(), rng);
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

NodeMetadata build_node_metadata(const GameState& state) {
    NodeMetadata out;
    out.mask = state_encoder::build_legal_mask(state);
    out.terminal = state_encoder::build_terminal_metadata(state);
    return out;
}

}  // namespace

pybind11::array_t<float> NativeMCTSResult::visit_probs_array() const {
    pybind11::array_t<float> arr(kActionDim);
    std::memcpy(arr.mutable_data(), visit_probs.data(), sizeof(float) * static_cast<std::size_t>(kActionDim));
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
    std::uint64_t rng_seed
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

    const NodeMetadata root_data = build_node_metadata(root_state);
    if (root_data.terminal.is_terminal) {
        throw std::invalid_argument("run_mcts called on terminal state");
    }

    bool has_legal = false;
    for (int i = 0; i < kActionDim; ++i) {
        if (root_data.mask[static_cast<std::size_t>(i)] != 0) {
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

    auto evaluate_pending = [&](std::vector<PendingLeafEval>& pending, std::vector<ReadyBackup>& backups) {
        if (pending.empty()) {
            return;
        }

        const pybind11::ssize_t batch = static_cast<pybind11::ssize_t>(pending.size());
        pybind11::array_t<float> states({batch, static_cast<pybind11::ssize_t>(kStateDim)});
        pybind11::array_t<bool> masks({batch, static_cast<pybind11::ssize_t>(kActionDim)});
        auto masks_view = masks.mutable_unchecked<2>();
        for (pybind11::ssize_t i = 0; i < batch; ++i) {
            const PendingLeafEval& req = pending[static_cast<std::size_t>(i)];
            float* state_row = states.mutable_data(i, 0);
            std::memcpy(state_row, req.state.data(), sizeof(float) * static_cast<std::size_t>(kStateDim));
            for (int j = 0; j < kActionDim; ++j) {
                masks_view(i, j) = (req.mask[static_cast<std::size_t>(j)] != 0);
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

        auto priors_view = priors_arr.unchecked<2>();
        auto values_view = values_arr.unchecked<1>();

        for (pybind11::ssize_t i = 0; i < batch; ++i) {
            PendingLeafEval& req = pending[static_cast<std::size_t>(i)];
            MCTSNode& node = nodes[static_cast<std::size_t>(req.node_index)];
            int legal_count = 0;
            bool has_finite_legal_score = false;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int a = 0; a < kActionDim; ++a) {
                if (req.mask[static_cast<std::size_t>(a)] == 0) {
                    node.priors[static_cast<std::size_t>(a)] = 0.0f;
                    continue;
                }
                ++legal_count;
                const float s = priors_view(i, a);  // Python returns policy scores/logits.
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
                    if (req.mask[static_cast<std::size_t>(a)] == 0) {
                        node.priors[static_cast<std::size_t>(a)] = 0.0f;
                        continue;
                    }
                    const float s = priors_view(i, a);
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
                        (req.mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
                }
            } else {
                for (int a = 0; a < kActionDim; ++a) {
                    if (req.mask[static_cast<std::size_t>(a)] != 0) {
                        node.priors[static_cast<std::size_t>(a)] = static_cast<float>(
                            static_cast<double>(node.priors[static_cast<std::size_t>(a)]) / sum
                        );
                    } else {
                        node.priors[static_cast<std::size_t>(a)] = 0.0f;
                    }
                }
            }

            const float value = values_view(i);
            if (!std::isfinite(static_cast<double>(value))) {
                throw std::runtime_error("MCTS evaluator values contain non-finite entries");
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

    auto apply_ready_backup_serial = [&](ReadyBackup& ready) {
        float value = ready.value;
        for (auto it = ready.path.rbegin(); it != ready.path.rend(); ++it) {
            const float backed = it->same_player ? value : -value;
            MCTSNode& parent = nodes[static_cast<std::size_t>(it->node_index)];
            parent.visit_count[static_cast<std::size_t>(it->action)] += 1;
            parent.value_sum[static_cast<std::size_t>(it->action)] += backed;
            value = backed;
        }
    };

    auto apply_ready_backup_virtual = [&](ReadyBackup& ready) {
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
        req.state = state_encoder::encode_state(root_state);
        req.mask = root_data.mask;
        pending_root.push_back(std::move(req));
        evaluate_pending(pending_root, root_backups);
    }

    std::vector<PendingLeafEval> pending;
    pending.reserve(static_cast<std::size_t>(eval_batch_size));
    std::vector<ReadyBackup> backups;
    backups.reserve(static_cast<std::size_t>(eval_batch_size));

    while (completed < num_simulations) {
        const int target_batch = std::min(eval_batch_size, num_simulations - completed);
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
                sample_root_decks(sim_state, rng);
                int node_index = 0;
                std::vector<PathStep> path;
                path.reserve(64);

                while (true) {
                    const NodeMetadata node_data = build_node_metadata(sim_state);
                    MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
                    if (node_data.terminal.is_terminal) {
                        ReadyBackup ready;
                        ready.value = winner_to_value_for_player(
                            node_data.terminal.winner,
                            node_data.terminal.current_player_id
                        );
                        ready.path = std::move(path);
                        backups.push_back(std::move(ready));
                        break;
                    }
                    if (!node.expanded) {
                        if (node.pending_eval) {
                            break;
                        }
                        node.pending_eval = true;
                        pending_flags[static_cast<std::size_t>(node_index)].store(1U, std::memory_order_release);
                        PendingLeafEval req;
                        req.node_index = node_index;
                        req.state = state_encoder::encode_state(sim_state);
                        req.mask = node_data.mask;
                        req.path = std::move(path);
                        pending.push_back(std::move(req));
                        break;
                    }

                    const int action = select_puct_action(
                        nodes,
                        node_index,
                        node_data.mask,
                        c_puct,
                        eps
                    );
                    if (action < 0) {
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

                    const int parent_to_play = node_data.terminal.current_player_id;
                    applyMove(sim_state, actionIndexToMove(action));
                    const state_encoder::TerminalMetadata child_meta = state_encoder::build_terminal_metadata(sim_state);
                    const bool same_player = child_meta.current_player_id == parent_to_play;
                    path.push_back(PathStep{node_index, action, same_player, false});
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
                        sample_root_decks(sim_state, slot_rng);
                        int node_index = 0;
                        std::vector<PathStep> path;
                        path.reserve(64);

                        while (true) {
                            const NodeMetadata node_data = build_node_metadata(sim_state);
                            if (node_data.terminal.is_terminal) {
                                ReadyBackup ready;
                                ready.value = winner_to_value_for_player(
                                    node_data.terminal.winner,
                                    node_data.terminal.current_player_id
                                );
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
                                        break;
                                    }
                                    node.pending_eval = true;
                                    pending_flags[static_cast<std::size_t>(node_index)].store(
                                        1U,
                                        std::memory_order_release
                                    );
                                    PendingLeafEval req;
                                    req.node_index = node_index;
                                    req.state = state_encoder::encode_state(sim_state);
                                    req.mask = node_data.mask;
                                    req.path = std::move(path);
                                    local_pending.push_back(std::move(req));
                                    break;
                                }

                                action = select_puct_action_with_pending_flags(
                                    node,
                                    node_data.mask,
                                    c_puct,
                                    eps,
                                    pending_flags.get()
                                );
                                if (action < 0) {
                                    rollback_virtual_path(path);
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
                            const state_encoder::TerminalMetadata child_meta =
                                state_encoder::build_terminal_metadata(sim_state);
                            const bool same_player = child_meta.current_player_id == parent_to_play;
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
        }

        evaluate_pending(pending, backups);

        if (backups.empty()) {
            throw std::runtime_error("Native MCTS made no progress while gathering/evaluating leaves");
        }

        pybind11::gil_scoped_release release;
        if (tree_workers <= 1) {
            for (ReadyBackup& ready : backups) {
                apply_ready_backup_serial(ready);
            }
        } else {
            for (ReadyBackup& ready : backups) {
                apply_ready_backup_virtual(ready);
            }
        }

        completed += static_cast<int>(backups.size());
    }

    const MCTSNode& root = nodes[0];
    NativeMCTSResult result;

    double total_visits = 0.0;
    for (int a = 0; a < kActionDim; ++a) {
        total_visits += static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]);
    }
    if (total_visits > 0.0) {
        for (int a = 0; a < kActionDim; ++a) {
            result.visit_probs[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]) / total_visits
            );
        }
    } else {
        int legal_count = 0;
        for (int a = 0; a < kActionDim; ++a) {
            if (root_data.mask[static_cast<std::size_t>(a)] != 0) {
                ++legal_count;
            }
        }
        const float u = 1.0f / static_cast<float>(legal_count);
        for (int a = 0; a < kActionDim; ++a) {
            result.visit_probs[static_cast<std::size_t>(a)] =
                (root_data.mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
        }
    }
    double prob_sum = 0.0;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_data.mask[static_cast<std::size_t>(a)] == 0) {
            result.visit_probs[static_cast<std::size_t>(a)] = 0.0f;
        }
        prob_sum += static_cast<double>(result.visit_probs[static_cast<std::size_t>(a)]);
    }
    if (prob_sum > 0.0 && std::isfinite(prob_sum)) {
        for (int a = 0; a < kActionDim; ++a) {
            result.visit_probs[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(result.visit_probs[static_cast<std::size_t>(a)]) / prob_sum
            );
        }
    }

    result.chosen_action_idx = sample_action_from_visits(
        result.visit_probs, root_data.mask, turns_taken, temperature_moves, temperature, rng
    );

    int best_visit_action = -1;
    int best_visit_count = -1;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_data.mask[static_cast<std::size_t>(a)] == 0) {
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

    double q_sum = 0.0;
    int q_count = 0;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_data.mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        const float q = (n <= 0) ? 0.0f : (root.value_sum[static_cast<std::size_t>(a)] / static_cast<float>(n));
        q_sum += static_cast<double>(q);
        ++q_count;
    }
    result.root_value = q_count > 0 ? static_cast<float>(q_sum / static_cast<double>(q_count)) : 0.0f;
    return result;
}
