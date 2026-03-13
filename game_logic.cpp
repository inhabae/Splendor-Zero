// game_logic.cpp
#include "game_logic.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>
#include <ctime>

// ─── Built-in dataset ───────────
namespace {

static const std::vector<Card> standard_cards = {
    Card{1, 1, 0, Color::Black, Tokens{1, 1, 1, 1, 0, 0}},
    Card{2, 1, 0, Color::Black, Tokens{1, 2, 1, 1, 0, 0}},
    Card{3, 1, 0, Color::Black, Tokens{2, 2, 0, 1, 0, 0}},
    Card{4, 1, 0, Color::Black, Tokens{0, 0, 1, 3, 1, 0}},
    Card{5, 1, 0, Color::Black, Tokens{0, 0, 2, 1, 0, 0}},
    Card{6, 1, 0, Color::Black, Tokens{2, 0, 2, 0, 0, 0}},
    Card{7, 1, 0, Color::Black, Tokens{0, 0, 3, 0, 0, 0}},
    Card{8, 1, 1, Color::Black, Tokens{0, 4, 0, 0, 0, 0}},
    Card{9, 1, 0, Color::Blue, Tokens{1, 0, 1, 1, 1, 0}},
    Card{10, 1, 0, Color::Blue, Tokens{1, 0, 1, 2, 1, 0}},
    Card{11, 1, 0, Color::Blue, Tokens{1, 0, 2, 2, 0, 0}},
    Card{12, 1, 0, Color::Blue, Tokens{0, 1, 3, 1, 0, 0}},
    Card{13, 1, 0, Color::Blue, Tokens{1, 0, 0, 0, 2, 0}},
    Card{14, 1, 0, Color::Blue, Tokens{0, 0, 2, 0, 2, 0}},
    Card{15, 1, 0, Color::Blue, Tokens{0, 0, 0, 0, 3, 0}},
    Card{16, 1, 1, Color::Blue, Tokens{0, 0, 0, 4, 0, 0}},
    Card{17, 1, 0, Color::White, Tokens{0, 1, 1, 1, 1, 0}},
    Card{18, 1, 0, Color::White, Tokens{0, 1, 2, 1, 1, 0}},
    Card{19, 1, 0, Color::White, Tokens{0, 2, 2, 0, 1, 0}},
    Card{20, 1, 0, Color::White, Tokens{3, 1, 0, 0, 1, 0}},
    Card{21, 1, 0, Color::White, Tokens{0, 0, 0, 2, 1, 0}},
    Card{22, 1, 0, Color::White, Tokens{0, 2, 0, 0, 2, 0}},
    Card{23, 1, 0, Color::White, Tokens{0, 3, 0, 0, 0, 0}},
    Card{24, 1, 1, Color::White, Tokens{0, 0, 4, 0, 0, 0}},
    Card{25, 1, 0, Color::Green, Tokens{1, 1, 0, 1, 1, 0}},
    Card{26, 1, 0, Color::Green, Tokens{1, 1, 0, 1, 2, 0}},
    Card{27, 1, 0, Color::Green, Tokens{0, 1, 0, 2, 2, 0}},
    Card{28, 1, 0, Color::Green, Tokens{1, 3, 1, 0, 0, 0}},
    Card{29, 1, 0, Color::Green, Tokens{2, 1, 0, 0, 0, 0}},
    Card{30, 1, 0, Color::Green, Tokens{0, 2, 0, 2, 0, 0}},
    Card{31, 1, 0, Color::Green, Tokens{0, 0, 0, 3, 0, 0}},
    Card{32, 1, 1, Color::Green, Tokens{0, 0, 0, 0, 4, 0}},
    Card{33, 1, 0, Color::Red, Tokens{1, 1, 1, 0, 1, 0}},
    Card{34, 1, 0, Color::Red, Tokens{2, 1, 1, 0, 1, 0}},
    Card{35, 1, 0, Color::Red, Tokens{2, 0, 1, 0, 2, 0}},
    Card{36, 1, 0, Color::Red, Tokens{1, 0, 0, 1, 3, 0}},
    Card{37, 1, 0, Color::Red, Tokens{0, 2, 1, 0, 0, 0}},
    Card{38, 1, 0, Color::Red, Tokens{2, 0, 0, 2, 0, 0}},
    Card{39, 1, 0, Color::Red, Tokens{3, 0, 0, 0, 0, 0}},
    Card{40, 1, 1, Color::Red, Tokens{4, 0, 0, 0, 0, 0}},
    Card{41, 2, 1, Color::Black, Tokens{3, 2, 2, 0, 0, 0}},
    Card{42, 2, 1, Color::Black, Tokens{3, 0, 3, 0, 2, 0}},
    Card{43, 2, 2, Color::Black, Tokens{0, 1, 4, 2, 0, 0}},
    Card{44, 2, 2, Color::Black, Tokens{0, 0, 5, 3, 0, 0}},
    Card{45, 2, 2, Color::Black, Tokens{5, 0, 0, 0, 0, 0}},
    Card{46, 2, 3, Color::Black, Tokens{0, 0, 0, 0, 6, 0}},
    Card{47, 2, 1, Color::Blue, Tokens{0, 2, 2, 3, 0, 0}},
    Card{48, 2, 1, Color::Blue, Tokens{0, 2, 3, 0, 3, 0}},
    Card{49, 2, 2, Color::Blue, Tokens{5, 3, 0, 0, 0, 0}},
    Card{50, 2, 2, Color::Blue, Tokens{2, 0, 0, 1, 4, 0}},
    Card{51, 2, 2, Color::Blue, Tokens{0, 5, 0, 0, 0, 0}},
    Card{52, 2, 3, Color::Blue, Tokens{0, 6, 0, 0, 0, 0}},
    Card{53, 2, 1, Color::White, Tokens{0, 0, 3, 2, 2, 0}},
    Card{54, 2, 1, Color::White, Tokens{2, 3, 0, 3, 0, 0}},
    Card{55, 2, 2, Color::White, Tokens{0, 0, 1, 4, 2, 0}},
    Card{56, 2, 2, Color::White, Tokens{0, 0, 0, 5, 3, 0}},
    Card{57, 2, 2, Color::White, Tokens{0, 0, 0, 5, 0, 0}},
    Card{58, 2, 3, Color::White, Tokens{6, 0, 0, 0, 0, 0}},
    Card{59, 2, 1, Color::Green, Tokens{3, 0, 2, 3, 0, 0}},
    Card{60, 2, 1, Color::Green, Tokens{2, 3, 0, 0, 2, 0}},
    Card{61, 2, 2, Color::Green, Tokens{4, 2, 0, 0, 1, 0}},
    Card{62, 2, 2, Color::Green, Tokens{0, 5, 3, 0, 0, 0}},
    Card{63, 2, 2, Color::Green, Tokens{0, 0, 5, 0, 0, 0}},
    Card{64, 2, 3, Color::Green, Tokens{0, 0, 6, 0, 0, 0}},
    Card{65, 2, 1, Color::Red, Tokens{2, 0, 0, 2, 3, 0}},
    Card{66, 2, 1, Color::Red, Tokens{0, 3, 0, 2, 3, 0}},
    Card{67, 2, 2, Color::Red, Tokens{1, 4, 2, 0, 0, 0}},
    Card{68, 2, 2, Color::Red, Tokens{3, 0, 0, 0, 5, 0}},
    Card{69, 2, 2, Color::Red, Tokens{0, 0, 0, 0, 5, 0}},
    Card{70, 2, 3, Color::Red, Tokens{0, 0, 0, 6, 0, 0}},
    Card{71, 3, 3, Color::Black, Tokens{3, 3, 5, 3, 0, 0}},
    Card{72, 3, 4, Color::Black, Tokens{0, 0, 0, 7, 0, 0}},
    Card{73, 3, 4, Color::Black, Tokens{0, 0, 3, 6, 3, 0}},
    Card{74, 3, 5, Color::Black, Tokens{0, 0, 0, 7, 3, 0}},
    Card{75, 3, 3, Color::Blue, Tokens{3, 0, 3, 3, 5, 0}},
    Card{76, 3, 4, Color::Blue, Tokens{7, 0, 0, 0, 0, 0}},
    Card{77, 3, 4, Color::Blue, Tokens{6, 3, 0, 0, 3, 0}},
    Card{78, 3, 5, Color::Blue, Tokens{7, 3, 0, 0, 0, 0}},
    Card{79, 3, 3, Color::White, Tokens{0, 3, 3, 5, 3, 0}},
    Card{80, 3, 4, Color::White, Tokens{0, 0, 0, 0, 7, 0}},
    Card{81, 3, 4, Color::White, Tokens{3, 0, 0, 3, 6, 0}},
    Card{82, 3, 5, Color::White, Tokens{3, 0, 0, 0, 7, 0}},
    Card{83, 3, 3, Color::Green, Tokens{5, 3, 0, 3, 3, 0}},
    Card{84, 3, 4, Color::Green, Tokens{0, 7, 0, 0, 0, 0}},
    Card{85, 3, 4, Color::Green, Tokens{3, 6, 3, 0, 0, 0}},
    Card{86, 3, 5, Color::Green, Tokens{0, 7, 3, 0, 0, 0}},
    Card{87, 3, 3, Color::Red, Tokens{3, 5, 3, 0, 3, 0}},
    Card{88, 3, 4, Color::Red, Tokens{0, 0, 7, 0, 0, 0}},
    Card{89, 3, 4, Color::Red, Tokens{0, 3, 6, 3, 0, 0}},
    Card{90, 3, 5, Color::Red, Tokens{0, 0, 7, 3, 0, 0}},
};

static const std::vector<Noble> standard_nobles = {
    Noble{1, 3, Tokens{3, 3, 0, 0, 3, 0}},
    Noble{2, 3, Tokens{0, 0, 3, 3, 3, 0}},
    Noble{3, 3, Tokens{3, 0, 0, 3, 3, 0}},
    Noble{4, 3, Tokens{0, 3, 3, 3, 0, 0}},
    Noble{5, 3, Tokens{3, 3, 3, 0, 0, 0}},
    Noble{6, 3, Tokens{0, 0, 0, 4, 4, 0}},
    Noble{7, 3, Tokens{4, 0, 0, 0, 4, 0}},
    Noble{8, 3, Tokens{0, 4, 4, 0, 0, 0}},
    Noble{9, 3, Tokens{4, 4, 0, 0, 0, 0}},
    Noble{10, 3, Tokens{0, 0, 4, 4, 0, 0}},
};

}  // namespace

// ─── Built-in standard dataset accessors ────────────────────────────────
const std::vector<Card>& standardCards() {
    return standard_cards;
}

const std::vector<Noble>& standardNobles() {
    return standard_nobles;
}

void initializeGame(GameState& state, unsigned int seed) {
    initializeGame(state, standardCards(), standardNobles(), seed);
}

// ─── initializeGame ───────────────────────────────────
void initializeGame(GameState&                state,
                    const std::vector<Card>&  all_cards,
                    const std::vector<Noble>& all_nobles,
                    unsigned int              seed) {
    // Reset state
    state = GameState();

    // Seed RNG
    if (seed == 0) seed = static_cast<unsigned int>(std::time(nullptr));
    std::mt19937 rng(seed);

    // ── Separate cards by tier ──
    std::vector<Card> tier[3];
    for (const Card& c : all_cards) {
        if (c.level >= 1 && c.level <= 3)
            tier[c.level - 1].push_back(c);
    }

    // ── Shuffle each tier ──
    for (int t = 0; t < 3; t++)
        std::shuffle(tier[t].begin(), tier[t].end(), rng);

    // ── Deal 4 face-up cards per tier, rest go to deck ──
    for (int t = 0; t < 3; t++) {
        for (int slot = 0; slot < 4; slot++) {
            if (slot < static_cast<int>(tier[t].size()))
                state.faceup[t][static_cast<std::size_t>(slot)] =
                    tier[t][static_cast<std::size_t>(slot)];
            // else stays as default Card{id=0} — empty slot
        }
        // Remaining cards go to deck (index 4 onwards)
        for (int i = 4; i < static_cast<int>(tier[t].size()); i++)
            state.deck[t].push_back(tier[t][static_cast<std::size_t>(i)]);
    }

    // ── Shuffle and pick 3 nobles ──
    std::vector<Noble> shuffled_nobles = all_nobles;
    std::shuffle(shuffled_nobles.begin(), shuffled_nobles.end(), rng);

    state.noble_count = 0;
    for (int i = 0; i < 3 && i < static_cast<int>(shuffled_nobles.size()); i++) {
        state.available_nobles[static_cast<std::size_t>(i)] =
            shuffled_nobles[static_cast<std::size_t>(i)];
        state.noble_count++;
    }

    // ── Initialize bank ──
    // 2-player game: 4 of each color, 5 jokers
    state.bank.white = 4;
    state.bank.blue  = 4;
    state.bank.green = 4;
    state.bank.red   = 4;
    state.bank.black = 4;
    state.bank.joker = 5;

    // ── Players start with nothing ──
    state.current_player  = 0;
    state.move_number     = 0;
    state.is_return_phase = false;
    state.is_noble_choice_phase = false;
}

// ─── Helper: replace faceup slot from deck ────────────
static void refillSlot(GameState& state, int tier, int slot) {
    if (state.faceup[tier][static_cast<std::size_t>(slot)].id == 0)
        throw std::runtime_error("Cannot refill an empty face-up slot");
    if (!state.deck[tier].empty()) {
        state.faceup[tier][static_cast<std::size_t>(slot)] = state.deck[tier].back();
        state.deck[tier].pop_back();
    } else {
        state.faceup[tier][static_cast<std::size_t>(slot)] = Card{}; // empty slot id=0
    }
}

#ifdef SPLENDOR_TEST_HOOKS
void testHook_refillSlot(GameState& state, int tier, int slot) {
    refillSlot(state, tier, slot);
}
#endif

static bool canClaimNoble(const Player& player, const Noble& noble) {
    return player.bonuses.white >= noble.requirements.white &&
           player.bonuses.blue  >= noble.requirements.blue  &&
           player.bonuses.green >= noble.requirements.green &&
           player.bonuses.red   >= noble.requirements.red   &&
           player.bonuses.black >= noble.requirements.black;
}

#ifdef SPLENDOR_TEST_HOOKS
bool testHook_canClaimNoble(const Player& player, const Noble& noble) {
    return canClaimNoble(player, noble);
}
#endif

static void validatePlayerIndex(int player_idx) {
    if (player_idx < 0 || player_idx >= 2)
        throw std::invalid_argument("Invalid player index");
}

static std::vector<int> getClaimableNobleIndices(const GameState& state, int player_idx) {
    validatePlayerIndex(player_idx);
    std::vector<int> claimable;
    const Player& player = state.players[player_idx];
    for (int i = 0; i < state.noble_count; i++) {
        if (canClaimNoble(player, state.available_nobles[static_cast<std::size_t>(i)]))
            claimable.push_back(i);
    }
    return claimable;
}

#ifdef SPLENDOR_TEST_HOOKS
std::vector<int> testHook_getClaimableNobleIndices(const GameState& state, int player_idx) {
    return getClaimableNobleIndices(state, player_idx);
}
#endif

static void claimNobleByIndex(GameState& state, int player_idx, int noble_idx) {
    validatePlayerIndex(player_idx);
    if (noble_idx < 0 || noble_idx >= state.noble_count)
        throw std::runtime_error("Invalid noble index");

    Player& player = state.players[player_idx];
    Noble noble = state.available_nobles[static_cast<std::size_t>(noble_idx)];
    if (!canClaimNoble(player, noble))
        throw std::runtime_error("Selected noble is not claimable");

    player.nobles.push_back(noble);
    player.points += noble.points;
    for (int j = noble_idx; j < state.noble_count - 1; j++)
        state.available_nobles[static_cast<std::size_t>(j)] =
            state.available_nobles[static_cast<std::size_t>(j + 1)];
    state.available_nobles[static_cast<std::size_t>(state.noble_count - 1)] = Noble{};
    state.noble_count--;
}

#ifdef SPLENDOR_TEST_HOOKS
void testHook_claimNobleByIndex(GameState& state, int player_idx, int noble_idx) {
    claimNobleByIndex(state, player_idx, noble_idx);
}
#endif

// ─── Helper: effective cost of card after bonuses ─────
static Tokens effectiveCost(const Card& card, const Player& player) {
    Tokens cost;
    cost.white = std::max(0, card.cost.white - player.bonuses.white);
    cost.blue  = std::max(0, card.cost.blue  - player.bonuses.blue);
    cost.green = std::max(0, card.cost.green - player.bonuses.green);
    cost.red   = std::max(0, card.cost.red   - player.bonuses.red);
    cost.black = std::max(0, card.cost.black - player.bonuses.black);
    return cost;
}

// ─── Helper: can player afford card ───────────────────
static bool canAfford(const Card& card, const Player& player) {
    Tokens cost = effectiveCost(card, player);
    int jokers_available = player.tokens.joker;
    // check each color
    int shortfall = 0;
    shortfall += std::max(0, cost.white - player.tokens.white);
    shortfall += std::max(0, cost.blue  - player.tokens.blue);
    shortfall += std::max(0, cost.green - player.tokens.green);
    shortfall += std::max(0, cost.red   - player.tokens.red);
    shortfall += std::max(0, cost.black - player.tokens.black);
    return shortfall <= jokers_available;
}

static bool isNonNegativeTokens(const Tokens& t) {
    return t.white >= 0 && t.blue >= 0 && t.green >= 0 &&
           t.red >= 0 && t.black >= 0 && t.joker >= 0;
}

static int nonJokerTotal(const Tokens& t) {
    return t.white + t.blue + t.green + t.red + t.black;
}

static int nonJokerNonZeroCount(const Tokens& t) {
    int count = 0;
    count += (t.white > 0);
    count += (t.blue > 0);
    count += (t.green > 0);
    count += (t.red > 0);
    count += (t.black > 0);
    return count;
}

static int bankAvailableColorCount(const Tokens& bank) {
    int count = 0;
    count += (bank.white > 0);
    count += (bank.blue > 0);
    count += (bank.green > 0);
    count += (bank.red > 0);
    count += (bank.black > 0);
    return count;
}

static void validateTier(int tier) {
    if (tier < 0 || tier >= 3) throw std::invalid_argument("Invalid card tier");
}

static void validateFaceupSlot(int slot) {
    if (slot < 0 || slot >= 4) throw std::invalid_argument("Invalid face-up card slot");
}

static void validateReservedSlot(const Player& player, int slot) {
    if (slot < 0 || slot >= static_cast<int>(player.reserved.size()))
        throw std::runtime_error("Invalid reserved card slot");
}

static void validateSingleReturnGem(const Tokens& t) {
    if (!isNonNegativeTokens(t)) throw std::invalid_argument("Return gem payload has negative counts");
    if (t.joker != 0) throw std::invalid_argument("Returning joker is not supported in current action space");
    if (nonJokerTotal(t) != 1) throw std::invalid_argument("Return move must return exactly one colored gem");
}

static void validateBuyMove(const GameState& state, int p, const Move& move) {
    if (state.is_return_phase) throw std::runtime_error("Cannot buy during return phase");
    if (state.is_noble_choice_phase) throw std::runtime_error("Must choose a noble before taking another action");
    const Player& player = state.players[p];

    if (move.from_deck) throw std::invalid_argument("BUY_CARD cannot be from deck");

    const Card* card = nullptr;
    if (move.from_reserved) {
        validateReservedSlot(player, move.card_slot);
        card = &player.reserved[static_cast<std::size_t>(move.card_slot)].card;
    } else {
        validateTier(move.card_tier);
        validateFaceupSlot(move.card_slot);
        card = &state.faceup[move.card_tier][static_cast<std::size_t>(move.card_slot)];
    }

    if (card->id <= 0) throw std::runtime_error("Selected card slot is empty");
    if (!canAfford(*card, player)) throw std::runtime_error("Player cannot afford selected card");
}

static void validateReserveMove(const GameState& state, int p, const Move& move) {
    if (state.is_return_phase) throw std::runtime_error("Cannot reserve during return phase");
    if (state.is_noble_choice_phase) throw std::runtime_error("Must choose a noble before taking another action");
    const Player& player = state.players[p];
    if (static_cast<int>(player.reserved.size()) >= 3) throw std::runtime_error("Player cannot reserve more than 3 cards");

    validateTier(move.card_tier);
    if (move.from_deck) {
        if (state.deck[move.card_tier].empty()) throw std::runtime_error("Cannot reserve from empty deck");
    } else {
        validateFaceupSlot(move.card_slot);
        if (state.faceup[move.card_tier][static_cast<std::size_t>(move.card_slot)].id <= 0)
            throw std::runtime_error("Selected face-up card slot is empty");
    }
}

static void validateTakeGemsMove(const GameState& state, int p, const Move& move) {
    (void)p;
    if (state.is_return_phase) throw std::runtime_error("Cannot take gems during return phase");
    if (state.is_noble_choice_phase) throw std::runtime_error("Must choose a noble before taking another action");
    const Tokens& t = move.gems_taken;
    if (!isNonNegativeTokens(t)) throw std::invalid_argument("Take gems payload has negative counts");
    if (t.joker != 0) throw std::invalid_argument("Cannot take joker gems with TAKE_GEMS");

    if (t.white > state.bank.white || t.blue > state.bank.blue || t.green > state.bank.green ||
        t.red > state.bank.red || t.black > state.bank.black) {
        throw std::runtime_error("Cannot take more gems than available in bank");
    }

    int total = nonJokerTotal(t);
    int nonzero = nonJokerNonZeroCount(t);
    int colors_available = bankAvailableColorCount(state.bank);
    bool any_two_same = (t.white == 2 || t.blue == 2 || t.green == 2 || t.red == 2 || t.black == 2);
    bool any_over_two = (t.white > 2 || t.blue > 2 || t.green > 2 || t.red > 2 || t.black > 2);
    if (any_over_two) throw std::invalid_argument("Cannot take more than 2 gems of one color");

    if (total == 3) {
        if (nonzero != 3) throw std::invalid_argument("TAKE_GEMS total=3 must be three different colors");
        if (colors_available < 3) throw std::runtime_error("Three-color gem take not available");
        return;
    }

    if (total == 2) {
        if (any_two_same) {
            if (nonzero != 1) throw std::invalid_argument("Two-same gem take must be exactly one color x2");
            if ((t.white == 2 && state.bank.white < 4) ||
                (t.blue == 2 && state.bank.blue < 4) ||
                (t.green == 2 && state.bank.green < 4) ||
                (t.red == 2 && state.bank.red < 4) ||
                (t.black == 2 && state.bank.black < 4)) {
                throw std::runtime_error("Two-same gem take requires 4 gems in bank");
            }
            return;
        }
        if (nonzero != 2) throw std::invalid_argument("TAKE_GEMS total=2 must be one color x2 or two colors x1");
        if (colors_available != 2) throw std::runtime_error("Two-different gem take only allowed when exactly two colors are available");
        return;
    }

    if (total == 1) {
        if (nonzero != 1) throw std::invalid_argument("TAKE_GEMS total=1 must be exactly one color");
        if (colors_available != 1) throw std::runtime_error("Single gem take only allowed when exactly one color is available");
        return;
    }

    throw std::invalid_argument("Invalid TAKE_GEMS total");
}

static void validateReturnGemMove(const GameState& state, int p, const Move& move) {
    if (state.is_noble_choice_phase) throw std::runtime_error("Cannot return gems during noble choice phase");
    if (!state.is_return_phase) throw std::runtime_error("Not in return phase");
    const Player& player = state.players[p];
    validateSingleReturnGem(move.gem_returned);
    if (move.gem_returned.white > player.tokens.white || move.gem_returned.blue > player.tokens.blue ||
        move.gem_returned.green > player.tokens.green || move.gem_returned.red > player.tokens.red ||
        move.gem_returned.black > player.tokens.black) {
        throw std::runtime_error("Player does not have gem to return");
    }
}

static void validateChooseNobleMove(const GameState& state, int p, const Move& move) {
    if (!state.is_noble_choice_phase) throw std::runtime_error("Not in noble choice phase");
    if (move.noble_idx < 0 || move.noble_idx >= state.noble_count)
        throw std::runtime_error("Invalid noble index");

    std::vector<int> claimable = getClaimableNobleIndices(state, p);
    for (int idx : claimable) {
        if (idx == move.noble_idx) return;
    }
    throw std::runtime_error("Selected noble is not claimable");
}

static void validateMoveForApply(const GameState& state, const Move& move) {
    if (state.current_player < 0 || state.current_player > 1)
        throw std::runtime_error("Invalid current_player index");
    int p = state.current_player;

    switch (move.type) {
        case BUY_CARD:
            validateBuyMove(state, p, move);
            return;
        case RESERVE_CARD:
            validateReserveMove(state, p, move);
            return;
        case TAKE_GEMS:
            validateTakeGemsMove(state, p, move);
            return;
        case RETURN_GEM:
            validateReturnGemMove(state, p, move);
            return;
        case CHOOSE_NOBLE:
            validateChooseNobleMove(state, p, move);
            return;
        case PASS_TURN:
            if (state.is_return_phase)
                throw std::runtime_error("Must return a gem before ending turn");
            if (state.is_noble_choice_phase)
                throw std::runtime_error("Must choose a noble before ending turn");
            return;
        default:
            throw std::invalid_argument("Invalid move type");
    }
}

#ifdef SPLENDOR_TEST_HOOKS
void testHook_validateMoveForApply(const GameState& state, const Move& move) {
    validateMoveForApply(state, move);
}
#endif

static void resolveNobleAndEndTurn(GameState& state, int p) {
    std::vector<int> claimable = getClaimableNobleIndices(state, p);
    if (claimable.size() == 1) {
        claimNobleByIndex(state, p, claimable[0]);
    } else if (claimable.size() > 1) {
        state.is_return_phase = false;
        state.is_noble_choice_phase = true;
        return;
    }
    state.is_return_phase = false;
    state.is_noble_choice_phase = false;
    state.current_player = 1 - p;
    state.move_number++;
}

// ─── applyMove ────────────────────────────────────────
void applyMove(GameState& state, const Move& move) {
    validateMoveForApply(state, move);
    int p = state.current_player;
    Player& player = state.players[p];

    switch (move.type) {

        case BUY_CARD: {
            Card* card = nullptr;
            int tier = move.card_tier;
            int slot = move.card_slot;

            if (move.from_reserved) {
                card = &player.reserved[static_cast<std::size_t>(slot)].card;
            } else {
                card = &state.faceup[tier][static_cast<std::size_t>(slot)];
            }

            // Calculate payment
            Tokens cost = effectiveCost(*card, player);
            Tokens payment;
            payment.white = std::min(cost.white, player.tokens.white);
            payment.blue  = std::min(cost.blue,  player.tokens.blue);
            payment.green = std::min(cost.green,  player.tokens.green);
            payment.red   = std::min(cost.red,   player.tokens.red);
            payment.black = std::min(cost.black,  player.tokens.black);
            int shortfall = (cost.white - payment.white) +
                            (cost.blue  - payment.blue)  +
                            (cost.green - payment.green) +
                            (cost.red   - payment.red)   +
                            (cost.black - payment.black);
            payment.joker = shortfall;

            // Apply payment
            player.tokens -= payment;
            state.bank    += payment;

            // Add card to player
            player.bonuses[card->color]++;
            player.points += card->points;
            player.cards.push_back(*card);

            // Remove card from source
            if (move.from_reserved) {
                player.reserved.erase(player.reserved.begin() + slot);
            } else {
                refillSlot(state, tier, slot);
            }

            resolveNobleAndEndTurn(state, p);
            break;
        }

        case RESERVE_CARD: {
            Card card;
            if (move.from_deck) {
                int tier = move.card_tier;
                card = state.deck[tier].back();
                state.deck[tier].pop_back();
            } else {
                int tier = move.card_tier;
                int slot = move.card_slot;
                card = state.faceup[tier][static_cast<std::size_t>(slot)];
                refillSlot(state, tier, slot);
            }

            player.reserved.push_back(ReservedCard{card, !move.from_deck});

            // Give joker if available
            if (state.bank.joker > 0) {
                player.tokens.joker++;
                state.bank.joker--;
            }

            // Check if return phase needed
            if (player.tokens.total() > 10) {
                state.is_return_phase = true;
                state.is_noble_choice_phase = false;
            } else {
                resolveNobleAndEndTurn(state, p);
            }
            break;
        }

        case TAKE_GEMS: {
            player.tokens += move.gems_taken;
            state.bank    -= move.gems_taken;

            // Check if return phase needed
            if (player.tokens.total() > 10) {
                state.is_return_phase = true;
                state.is_noble_choice_phase = false;
            } else {
                resolveNobleAndEndTurn(state, p);
            }
            break;
        }

        case RETURN_GEM: {
            // Return exactly 1 token of specified color
            player.tokens -= move.gem_returned;
            state.bank    += move.gem_returned;

            // Check if still over 10
            if (player.tokens.total() > 10) {
                state.is_return_phase = true;
                state.is_noble_choice_phase = false;
            } else {
                resolveNobleAndEndTurn(state, p);
            }
            break;
        }

        case PASS_TURN: {
            resolveNobleAndEndTurn(state, p);
            break;
        }
        case CHOOSE_NOBLE: {
            claimNobleByIndex(state, p, move.noble_idx);
            state.is_return_phase = false;
            state.is_noble_choice_phase = false;
            state.current_player = 1 - p;
            state.move_number++;
            break;
        }
        default:
            throw std::invalid_argument("Invalid move type");
    }
}

// ─── findAllValidMoves ────────────────────────────────
std::vector<Move> findAllValidMoves(const GameState& state) {
    std::vector<Move> moves;
    int p = state.current_player;
    const Player& player = state.players[p];

    // ── Return phase: only return moves ──
    if (state.is_return_phase) {
        const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};
        for (int i = 0; i < 5; i++) {
            if (player.tokens[colors[i]] > 0) {
                Move m;
                m.type = RETURN_GEM;
                m.gem_returned[colors[i]] = 1;
                moves.push_back(m);
            }
        }
        return moves;
    }

    // ── Noble choice phase: only choose among currently claimable nobles ──
    if (state.is_noble_choice_phase) {
        std::vector<int> claimable = getClaimableNobleIndices(state, p);
        for (int noble_idx : claimable) {
            Move m;
            m.type = CHOOSE_NOBLE;
            m.noble_idx = noble_idx;
            moves.push_back(m);
        }
        return moves;
    }

    // ── BUY face-up ──
    for (int t = 0; t < 3; t++) {
        for (int s = 0; s < 4; s++) {
            const Card& card = state.faceup[t][static_cast<std::size_t>(s)];
            if (card.id > 0 && canAfford(card, player)) {
                Move m;
                m.type      = BUY_CARD;
                m.card_tier = t;
                m.card_slot = s;
                moves.push_back(m);
            }
        }
    }

    // ── BUY reserved ──
    for (int s = 0; s < static_cast<int>(player.reserved.size()); s++) {
        if (canAfford(player.reserved[static_cast<std::size_t>(s)].card, player)) {
            Move m;
            m.type         = BUY_CARD;
            m.card_slot    = s;
            m.from_reserved = true;
            moves.push_back(m);
        }
    }

    // ── RESERVE face-up ──
    if (static_cast<int>(player.reserved.size()) < 3) {
        for (int t = 0; t < 3; t++) {
            for (int s = 0; s < 4; s++) {
                if (state.faceup[t][static_cast<std::size_t>(s)].id > 0) {
                    Move m;
                    m.type      = RESERVE_CARD;
                    m.card_tier = t;
                    m.card_slot = s;
                    moves.push_back(m);
                }
            }
        }
        // ── RESERVE from deck ──
        for (int t = 0; t < 3; t++) {
            if (!state.deck[t].empty()) {
                Move m;
                m.type      = RESERVE_CARD;
                m.card_tier = t;
                m.from_deck = true;
                moves.push_back(m);
            }
        }
    }

    // ── TAKE GEMS ──
    int colors_available = 0;
    const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};
    bool avail[5];
    for (int i = 0; i < 5; i++) {
        avail[i] = state.bank[colors[i]] > 0;
        if (avail[i]) colors_available++;
    }

    // Take 3 different
    if (colors_available >= 3) {
        for (int i = 0; i < 5; i++) if (avail[i])
        for (int j = i+1; j < 5; j++) if (avail[j])
        for (int k = j+1; k < 5; k++) if (avail[k]) {
            Move m; m.type = TAKE_GEMS;
            m.gems_taken[colors[i]] = 1;
            m.gems_taken[colors[j]] = 1;
            m.gems_taken[colors[k]] = 1;
            moves.push_back(m);
        }
    }
    // Take 2 same
    for (int i = 0; i < 5; i++) {
        if (state.bank[colors[i]] >= 4) {
            Move m; m.type = TAKE_GEMS;
            m.gems_taken[colors[i]] = 2;
            moves.push_back(m);
        }
    }
    // Take 2 different (only when < 3 colors available)
    if (colors_available == 2) {
        for (int i = 0; i < 5; i++) if (avail[i])
        for (int j = i+1; j < 5; j++) if (avail[j]) {
            Move m; m.type = TAKE_GEMS;
            m.gems_taken[colors[i]] = 1;
            m.gems_taken[colors[j]] = 1;
            moves.push_back(m);
        }
    }
    // Take 1 (only when exactly 1 color available)
    if (colors_available == 1) {
        for (int i = 0; i < 5; i++) {
            if (avail[i]) {
                Move m; m.type = TAKE_GEMS;
                m.gems_taken[colors[i]] = 1;
                moves.push_back(m);
            }
        }
    }

    // ── PASS (only if no other moves) ──
    if (moves.empty()) {
        Move m; m.type = PASS_TURN;
        moves.push_back(m);
    }

    return moves;
}

// ─── isGameOver ───────────────────────────────────────
bool isGameOver(const GameState& state) {
    // Don't end mid return or noble choice phase
    if (state.is_return_phase || state.is_noble_choice_phase) return false;

    bool p0_has_15 = state.players[0].points >= 15;
    bool p1_has_15 = state.players[1].points >= 15;

    if (!p0_has_15 && !p1_has_15) return false;

    // Player 1 (second) reached 15 — game ends immediately
    if (p1_has_15) return true;

    // Player 0 (first) reached 15 — player 1 gets last turn
    // current_player==0 means player 1 just finished their turn
    if (p0_has_15 && state.current_player == 0) return true;

    return false;
}

// ─── determineWinner ──────────────────────────────────
int determineWinner(const GameState& state) {
    const Player& p0 = state.players[0];
    const Player& p1 = state.players[1];

    // Higher points wins
    if (p0.points > p1.points) return 0;
    if (p1.points > p0.points) return 1;

    // Tiebreaker: fewer purchased cards
    if (static_cast<int>(p0.cards.size()) < static_cast<int>(p1.cards.size())) return 0;
    if (static_cast<int>(p1.cards.size()) < static_cast<int>(p0.cards.size())) return 1;

    return -1; // draw
}

// ─── moveToActionIndex ────────────────────────────────
int moveToActionIndex(const Move& move) {
    switch (move.type) {

        case BUY_CARD:
            if (move.from_reserved)
                return 12 + move.card_slot;          // 12-14
            return move.card_tier * 4 + move.card_slot; // 0-11

        case RESERVE_CARD:
            if (move.from_deck)
                return 27 + move.card_tier;           // 27-29
            return 15 + move.card_tier * 4 + move.card_slot; // 15-26

        case TAKE_GEMS: {
            const Tokens& t = move.gems_taken;
            int total = t.white + t.blue + t.green + t.red + t.black;

            if (total == 2 && (t.white==2||t.blue==2||t.green==2||
                               t.red==2||t.black==2)) {
                // Take 2 same (40-44)
                if (t.white==2) return 40;
                if (t.blue ==2) return 41;
                if (t.green==2) return 42;
                if (t.red  ==2) return 43;
                if (t.black==2) return 44;
            }

            // Encode which colors taken as bitmask w=1,b=2,g=4,r=8,k=16
            int mask = (t.white?1:0)|(t.blue?2:0)|(t.green?4:0)|
                       (t.red?8:0)|(t.black?16:0);

            if (total == 3) {
                // Take 3 different (30-39)
                // Order: wbg=7,wbr=11,wbk=19,wgr=13,wgk=21,wrk=25,
                //        bgr=14,bgk=22,brk=26,grk=28
                switch(mask) {
                    case 7:  return 30; // w+b+g
                    case 11: return 31; // w+b+r
                    case 19: return 32; // w+b+k
                    case 13: return 33; // w+g+r
                    case 21: return 34; // w+g+k
                    case 25: return 35; // w+r+k
                    case 14: return 36; // b+g+r
                    case 22: return 37; // b+g+k
                    case 26: return 38; // b+r+k
                    case 28: return 39; // g+r+k
                }
            }

            if (total == 2) {
                // Take 2 different (45-54)
                switch(mask) {
                    case 3:  return 45; // w+b
                    case 5:  return 46; // w+g
                    case 9:  return 47; // w+r
                    case 17: return 48; // w+k
                    case 6:  return 49; // b+g
                    case 10: return 50; // b+r
                    case 18: return 51; // b+k
                    case 12: return 52; // g+r
                    case 20: return 53; // g+k
                    case 24: return 54; // r+k
                }
            }

            if (total == 1) {
                // Take 1 (55-59)
                if (t.white) return 55;
                if (t.blue)  return 56;
                if (t.green) return 57;
                if (t.red)   return 58;
                if (t.black) return 59;
            }
            return -1;
        }

        case PASS_TURN:
            return 60;

        case RETURN_GEM: {
            const Tokens& r = move.gem_returned;
            if (r.white) return 61;
            if (r.blue)  return 62;
            if (r.green) return 63;
            if (r.red)   return 64;
            if (r.black) return 65;
            return -1;
        }
        case CHOOSE_NOBLE:
            if (move.noble_idx >= 0 && move.noble_idx < 3) return 66 + move.noble_idx;
            return -1;
    }
    return -1;
}

// ─── actionIndexToMove ────────────────────────────────
Move actionIndexToMove(int idx) {
    Move m;

    // Buy face-up (0-11)
    if (idx >= 0 && idx <= 11) {
        m.type      = BUY_CARD;
        m.card_tier = idx / 4;
        m.card_slot = idx % 4;
        return m;
    }
    // Buy reserved (12-14)
    if (idx >= 12 && idx <= 14) {
        m.type         = BUY_CARD;
        m.card_slot    = idx - 12;
        m.from_reserved = true;
        return m;
    }
    // Reserve face-up (15-26)
    if (idx >= 15 && idx <= 26) {
        m.type      = RESERVE_CARD;
        int rel     = idx - 15;
        m.card_tier = rel / 4;
        m.card_slot = rel % 4;
        return m;
    }
    // Reserve from deck (27-29)
    if (idx >= 27 && idx <= 29) {
        m.type      = RESERVE_CARD;
        m.card_tier = idx - 27;
        m.from_deck = true;
        return m;
    }

    const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};

    // Take 3 different (30-39)
    if (idx >= 30 && idx <= 39) {
        m.type = TAKE_GEMS;
        // Map index to color triplets
        const int triplets[10][3] = {
            {0,1,2},{0,1,3},{0,1,4},{0,2,3},{0,2,4},
            {0,3,4},{1,2,3},{1,2,4},{1,3,4},{2,3,4}
        };
        int rel = idx - 30;
        m.gems_taken[colors[triplets[rel][0]]] = 1;
        m.gems_taken[colors[triplets[rel][1]]] = 1;
        m.gems_taken[colors[triplets[rel][2]]] = 1;
        return m;
    }
    // Take 2 same (40-44)
    if (idx >= 40 && idx <= 44) {
        m.type = TAKE_GEMS;
        m.gems_taken[colors[idx-40]] = 2;
        return m;
    }
    // Take 2 different (45-54)
    if (idx >= 45 && idx <= 54) {
        m.type = TAKE_GEMS;
        const int pairs[10][2] = {
            {0,1},{0,2},{0,3},{0,4},{1,2},
            {1,3},{1,4},{2,3},{2,4},{3,4}
        };
        int rel = idx - 45;
        m.gems_taken[colors[pairs[rel][0]]] = 1;
        m.gems_taken[colors[pairs[rel][1]]] = 1;
        return m;
    }
    // Take 1 (55-59)
    if (idx >= 55 && idx <= 59) {
        m.type = TAKE_GEMS;
        m.gems_taken[colors[idx-55]] = 1;
        return m;
    }
    // Pass (60)
    if (idx == 60) {
        m.type = PASS_TURN;
        return m;
    }
    // Return gem (61-65)
    if (idx >= 61 && idx <= 65) {
        m.type = RETURN_GEM;
        m.gem_returned[colors[idx-61]] = 1;
        return m;
    }
    // Choose noble (66-68)
    if (idx >= 66 && idx <= 68) {
        m.type = CHOOSE_NOBLE;
        m.noble_idx = idx - 66;
        return m;
    }

    return m; // fallback PASS
}

// ─── getValidMoveMask ─────────────────────────────────
std::array<int, 69> getValidMoveMask(const GameState& state) {
    std::array<int, 69> mask = {};
    std::vector<Move> valid = findAllValidMoves(state);
    for (const Move& move : valid) {
        int idx = moveToActionIndex(move);
        if (idx >= 0 && idx < 69)
            mask[static_cast<std::size_t>(idx)] = 1;
    }
    return mask;
}
