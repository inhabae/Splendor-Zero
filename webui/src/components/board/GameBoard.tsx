import { ActionVizDTO, BoardStateDTO, TokenCountsDTO } from '../../types';
import { NobleView } from './NobleView';
import { PlayerStrip } from './PlayerStrip';
import { TierRow } from './TierRow';
import { TokenPill } from './TokenPill';

const TOKEN_ORDER: Array<keyof TokenCountsDTO> = ['white', 'blue', 'green', 'red', 'black', 'gold'];
const COLOR_ORDER = ['white', 'blue', 'green', 'red', 'black'] as const;
const TAKE3_TRIPLETS = [
  [0, 1, 2],
  [0, 1, 3],
  [0, 1, 4],
  [0, 2, 3],
  [0, 2, 4],
  [0, 3, 4],
  [1, 2, 3],
  [1, 2, 4],
  [1, 3, 4],
  [2, 3, 4],
] as const;
const TAKE2_PAIRS = [
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [1, 2],
  [1, 3],
  [1, 4],
  [2, 3],
  [2, 4],
  [3, 4],
] as const;

function seatLabel(seat: 'P0' | 'P1'): 'P1' | 'P2' {
  return seat === 'P0' ? 'P1' : 'P2';
}

function actionBankColors(action: ActionVizDTO | null | undefined): Set<string> {
  const out = new Set<string>();
  if (!action) {
    return out;
  }
  const idx = action.action_idx;
  if (30 <= idx && idx <= 39) {
    for (const colorIdx of TAKE3_TRIPLETS[idx - 30]) {
      out.add(COLOR_ORDER[colorIdx]);
    }
    return out;
  }
  if (40 <= idx && idx <= 44) {
    out.add(COLOR_ORDER[idx - 40]);
    return out;
  }
  if (45 <= idx && idx <= 54) {
    for (const colorIdx of TAKE2_PAIRS[idx - 45]) {
      out.add(COLOR_ORDER[colorIdx]);
    }
    return out;
  }
  if (55 <= idx && idx <= 59) {
    out.add(COLOR_ORDER[idx - 55]);
    return out;
  }
  if (61 <= idx && idx <= 65) {
    out.add(COLOR_ORDER[idx - 61]);
    return out;
  }
  if (action.placement_hint.zone === 'bank_token' && action.placement_hint.color) {
    out.add(action.placement_hint.color);
  }
  return out;
}

export function GameBoard({
  board,
  mctsTopAction = null,
  modelTopAction = null,
  onCardClick,
  onNobleClick,
  onReservedCardClick,
}: {
  board: BoardStateDTO;
  mctsTopAction?: ActionVizDTO | null;
  modelTopAction?: ActionVizDTO | null;
  onCardClick?: (tier: number, slot: number) => void;
  onNobleClick?: (slot: number) => void;
  onReservedCardClick?: (seat: 'P0' | 'P1', slot: number) => void;
}) {
  const mctsBankColors = actionBankColors(mctsTopAction);
  const modelBankColors = actionBankColors(modelTopAction);
  return (
    <section className="board-surface">
      <header className="board-meta">
        <div>Target: {board.meta.target_points}</div>
        <div>Turn: {board.meta.turn_index}</div>
        <div>To Move: {seatLabel(board.meta.player_to_move)}</div>
      </header>
      <section className="board-main">
        <aside className="board-left">
          <PlayerStrip player={board.players[0]} seat="P0" mctsTopAction={mctsTopAction} modelTopAction={modelTopAction} onReservedCardClick={onReservedCardClick} />
          <PlayerStrip player={board.players[1]} seat="P1" mctsTopAction={mctsTopAction} modelTopAction={modelTopAction} onReservedCardClick={onReservedCardClick} />
        </aside>

        <section className="board-right">
          <div className="nobles-row">
            <div className="nobles-grid">
              {board.nobles.length === 0 && <div className="empty-note">No nobles available</div>}
              {board.nobles.map((noble, idx) => (
                <NobleView key={`noble-${idx}`} noble={noble} onClick={noble.slot != null ? () => onNobleClick?.(noble.slot as number) : undefined} />
              ))}
            </div>
          </div>
          <div className="bank-row bank-row-inline">
            {TOKEN_ORDER.filter((c) => c !== 'gold').map((color) => (
              <TokenPill
                key={`bank-${color}`}
                color={color}
                count={board.bank[color]}
                showMcts={mctsBankColors.has(color)}
                showModel={modelBankColors.has(color)}
              />
            ))}
            <TokenPill key="bank-gold" color="gold" count={board.bank.gold} />
          </div>
          <div className="tiers-wrap">
            {board.tiers.map((tier) => (
              <TierRow key={`tier-row-${tier.tier}`} tier={tier} mctsTopAction={mctsTopAction} modelTopAction={modelTopAction} onCardClick={onCardClick} />
            ))}
          </div>
        </section>
      </section>
    </section>
  );
}
