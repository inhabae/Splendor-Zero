import { ActionVizDTO, BoardStateDTO, TokenCountsDTO } from '../../types';
import { NobleView } from './NobleView';
import { PlayerStrip } from './PlayerStrip';
import { TierDeckBadge, TierRow } from './TierRow';
import { TokenPill } from './TokenPill';

const TOKEN_ORDER: Array<keyof TokenCountsDTO> = ['gold', 'white', 'blue', 'green', 'red', 'black'];
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
      <section className="board-main">
        <aside className="board-left">
          <PlayerStrip player={board.players[0]} seat="P0" mctsTopAction={mctsTopAction} modelTopAction={modelTopAction} onReservedCardClick={onReservedCardClick} />
          <PlayerStrip player={board.players[1]} seat="P1" mctsTopAction={mctsTopAction} modelTopAction={modelTopAction} onReservedCardClick={onReservedCardClick} />
        </aside>

        <section className="board-right">
          <div className="board-play-shell">
            <div className="nobles-row">
              <div className="nobles-grid">
                {board.nobles.length === 0 && <div className="empty-note">No nobles available</div>}
                {board.nobles.map((noble, idx) => (
                  <NobleView key={`noble-${idx}`} noble={noble} onClick={noble.slot != null ? () => onNobleClick?.(noble.slot as number) : undefined} />
                ))}
              </div>
            </div>
            <div className="bank-row bank-row-inline">
              {TOKEN_ORDER.map((color) => (
                <TokenPill
                  key={`bank-${color}`}
                  color={color}
                  count={board.bank[color]}
                  showMcts={mctsBankColors.has(color)}
                  showModel={modelBankColors.has(color)}
                />
              ))}
            </div>
            <div className="board-cards-shell">
              {board.tiers.map((tier) => (
                <div key={`board-cards-row-${tier.tier}`} className="board-cards-row">
                  <div className="tier-decks-slot" aria-label={`Tier ${tier.tier} deck`}>
                    <TierDeckBadge tier={tier} />
                  </div>
                  <TierRow
                    key={`tier-row-${tier.tier}`}
                    tier={tier}
                    mctsTopAction={mctsTopAction}
                    modelTopAction={modelTopAction}
                    onCardClick={onCardClick}
                    showDeck={false}
                  />
                </div>
              ))}
            </div>
          </div>
        </section>
      </section>
    </section>
  );
}
