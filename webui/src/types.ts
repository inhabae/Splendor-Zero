export type Seat = 'P0' | 'P1';
export type JobStatus = 'QUEUED' | 'RUNNING' | 'DONE' | 'FAILED' | 'CANCELLED';

export interface CheckpointDTO {
  id: string;
  name: string;
  path: string;
  created_at: string;
  size_bytes: number;
}

export interface ActionInfoDTO {
  action_idx: number;
  label: string;
}

export interface MoveLogEntryDTO {
  turn_index: number;
  actor: Seat;
  action_idx: number;
  label: string;
}

export interface GameConfigDTO {
  checkpoint_id: string;
  checkpoint_path: string;
  num_simulations: number;
  player_seat: Seat;
  seed: number;
}

export interface ColorCountsDTO {
  white: number;
  blue: number;
  green: number;
  red: number;
  black: number;
}

export interface TokenCountsDTO extends ColorCountsDTO {
  gold: number;
}

export interface CardDTO {
  points: number;
  bonus_color: 'white' | 'blue' | 'green' | 'red' | 'black';
  cost: ColorCountsDTO;
  source: 'faceup' | 'reserved_public' | 'reserved_private';
  tier?: number;
  slot?: number;
}

export interface NobleDTO {
  points: number;
  requirements: ColorCountsDTO;
}

export interface TierRowDTO {
  tier: number;
  deck_count: number;
  cards: CardDTO[];
}

export interface PlayerBoardDTO {
  seat: Seat;
  display_name: string;
  points: number;
  tokens: TokenCountsDTO;
  bonuses: ColorCountsDTO;
  reserved_public: CardDTO[];
  reserved_total: number;
  is_to_move: boolean;
}

export interface BoardStateDTO {
  meta: {
    target_points: number;
    turn_index: number;
    player_to_move: Seat;
  };
  players: [PlayerBoardDTO, PlayerBoardDTO];
  bank: TokenCountsDTO;
  nobles: NobleDTO[];
  tiers: [TierRowDTO, TierRowDTO, TierRowDTO];
}

export interface GameSnapshotDTO {
  game_id: string;
  status: string;
  player_to_move: Seat;
  legal_actions: number[];
  legal_action_details: ActionInfoDTO[];
  winner: number;
  turn_index: number;
  move_log: MoveLogEntryDTO[];
  config?: GameConfigDTO;
  board_state?: BoardStateDTO | null;
}

export interface EngineThinkResponse {
  job_id: string;
  status: 'QUEUED' | 'RUNNING';
}

export interface EngineJobStatusDTO {
  job_id: string;
  status: JobStatus;
  error?: string | null;
  result?: {
    action_idx: number;
  } | null;
}

export interface PlayerMoveResponse {
  snapshot: GameSnapshotDTO;
  engine_should_move: boolean;
}

export interface PlacementHintDTO {
  zone: 'faceup_card' | 'reserved_card' | 'bank_token' | 'other';
  tier?: number;
  slot?: number;
  color?: 'white' | 'blue' | 'green' | 'red' | 'black';
}

export interface ActionVizDTO {
  action_idx: number;
  label: string;
  masked: boolean;
  policy_prob: number;
  is_selected: boolean;
  placement_hint: PlacementHintDTO;
}

export interface SelfPlayRunRequest {
  checkpoint_id: string;
  num_simulations: number;
  games: number;
  max_turns: number;
  seed?: number;
}

export interface SelfPlayRunResponse {
  session_id: string;
  path: string;
  games: number;
  steps: number;
  created_at: string;
}

export interface SelfPlaySessionDTO {
  session_id: string;
  display_name: string;
  path: string;
  created_at: string;
  games: number;
  steps: number;
  steps_per_episode: Record<string, number>;
  metadata: Record<string, unknown>;
}

export interface SelfPlaySessionSummaryDTO {
  session_id: string;
  path: string;
  created_at: string;
  games: number;
  steps: number;
  steps_per_episode: Record<string, number>;
  metadata: Record<string, unknown>;
  winners_by_episode: Record<string, number>;
  cutoff_by_episode: Record<string, boolean>;
}

export interface ReplayStepDTO {
  session_id: string;
  episode_idx: number;
  step_idx: number;
  turn_idx: number;
  player_id: number;
  winner: number;
  reached_cutoff: boolean;
  value_target: number;
  value_root: number;
  value_root_best: number;
  model_value?: number | null;
  action_selected: number;
  board_state: BoardStateDTO;
  action_details: ActionVizDTO[];
  model_action_details?: ActionVizDTO[] | null;
}
