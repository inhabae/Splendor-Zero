from __future__ import annotations

# Canonical state/action schema shared across Python runtime code and tests.
# The production encoder lives in py_splendor.cpp and must match these constants.

STATE_DIM = 246
ACTION_DIM = 69
CARD_FEATURE_LEN = 11

# State layout offsets (side-to-move canonical perspective)
CP_TOKENS_START = 0
CP_BONUSES_START = 6
CP_POINTS_IDX = 11
CP_RESERVED_START = 12
OP_TOKENS_START = 45
OP_BONUSES_START = 51
OP_POINTS_IDX = 56
OP_RESERVED_START = 57
OP_RESERVED_COUNT_IDX = 90
FACEUP_START = 91
BANK_START = 223
NOBLES_START = 229
PHASE_FLAGS_START = 244

# Compatibility constants retained for checkpoint adapters that still know how
# to project newer 252-dim states into the legacy 246-dim schema.
PLAYER_INDEX_IDX = 57
OPPONENT_RESERVED_SLOT_LEN = 13
OP_RESERVED_IS_OCCUPIED_OFFSET = 11
OP_RESERVED_TIER_OFFSET = 12
