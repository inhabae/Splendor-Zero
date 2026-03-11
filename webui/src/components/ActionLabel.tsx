import { BoardStateDTO, CardDTO } from '../types';

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

const COLOR_ORDER = ['white', 'blue', 'green', 'red', 'black'] as const;
const COLOR_EMOJI: Record<(typeof COLOR_ORDER)[number], string> = {
  white: '⚪',
  blue: '🔵',
  green: '🟢',
  red: '🔴',
  black: '⚫',
};

function faceupCard(board: BoardStateDTO | null | undefined, tier: number, slot: number): CardDTO | null {
  const row = board?.tiers.find((item) => item.tier === tier);
  if (!row) {
    return null;
  }
  return row.cards.find((card) => card.slot === slot) ?? null;
}

function costEntries(card: CardDTO): Array<{ color: (typeof COLOR_ORDER)[number]; count: number }> {
  return COLOR_ORDER.filter((color) => card.cost[color] > 0).map((color) => ({
    color,
    count: card.cost[color],
  }));
}

function CardActionLabel({ verb, card }: { verb: 'BUY' | 'RESERVE'; card: CardDTO | null }) {
  if (!card) {
    return <span className="action-verb">{verb}</span>;
  }
  const toneClass = `action-tone token-${card.bonus_color}`;
  return (
    <>
      <span className={`action-verb action-verb-card ${toneClass}`}>{verb}</span>
      <span className={`action-points ${toneClass}`}>
        +{card.points}
      </span>
      {costEntries(card).map(({ color, count }) => (
        <span key={`${verb}-${color}-${count}`} className={`action-cost ${toneClass}`}>
          <span className="action-cost-emoji">{COLOR_EMOJI[color]}</span>
          <span>{count}</span>
        </span>
      ))}
    </>
  );
}

function TakeLabel({ colors, duplicate = 1 }: { colors: readonly number[]; duplicate?: number }) {
  return (
    <>
      <span className="action-verb">TAKE</span>
      {colors.map((colorIdx, idx) => {
        const color = COLOR_ORDER[colorIdx];
        return (
          <span key={`${color}-${idx}`} className="action-gem">
            {COLOR_EMOJI[color]}
            {duplicate > 1 && idx === 0 ? <span className="action-mult">x{duplicate}</span> : null}
          </span>
        );
      })}
    </>
  );
}

function DeckReserveLabel({ tier }: { tier: 1 | 2 | 3 }) {
  return (
    <>
      <span className={`action-verb action-verb-tier action-tier-${tier}`}>RESERVE</span>
      <span className={`action-meta action-tier-${tier}`}>DECK T{tier}</span>
    </>
  );
}

export function ActionLabel({
  actionIdx,
  board,
  showPlayed = false,
}: {
  actionIdx: number;
  board?: BoardStateDTO | null;
  showPlayed?: boolean;
}) {
  let content: JSX.Element | JSX.Element[] = <span className="action-verb">UNKNOWN</span>;

  if (0 <= actionIdx && actionIdx <= 11) {
    const tier = Math.floor(actionIdx / 4) + 1;
    const slot = actionIdx % 4;
    content = <CardActionLabel verb="BUY" card={faceupCard(board, tier, slot)} />;
  } else if (12 <= actionIdx && actionIdx <= 14) {
    content = (
      <>
        <span className="action-verb">BUY</span>
        <span className="action-meta">RESERVED {actionIdx - 12}</span>
      </>
    );
  } else if (15 <= actionIdx && actionIdx <= 26) {
    const rel = actionIdx - 15;
    const tier = Math.floor(rel / 4) + 1;
    const slot = rel % 4;
    content = <CardActionLabel verb="RESERVE" card={faceupCard(board, tier, slot)} />;
  } else if (27 <= actionIdx && actionIdx <= 29) {
    content = <DeckReserveLabel tier={(actionIdx - 26) as 1 | 2 | 3} />;
  } else if (30 <= actionIdx && actionIdx <= 39) {
    content = <TakeLabel colors={TAKE3_TRIPLETS[actionIdx - 30]} />;
  } else if (40 <= actionIdx && actionIdx <= 44) {
    content = <TakeLabel colors={[actionIdx - 40]} duplicate={2} />;
  } else if (45 <= actionIdx && actionIdx <= 54) {
    content = <TakeLabel colors={TAKE2_PAIRS[actionIdx - 45]} />;
  } else if (55 <= actionIdx && actionIdx <= 59) {
    content = <TakeLabel colors={[actionIdx - 55]} />;
  } else if (actionIdx === 60) {
    content = <span className="action-verb">PASS</span>;
  } else if (61 <= actionIdx && actionIdx <= 65) {
    const color = COLOR_ORDER[actionIdx - 61];
    content = (
      <>
        <span className="action-verb">RETURN</span>
        <span className="action-gem">{COLOR_EMOJI[color]}</span>
      </>
    );
  } else if (66 <= actionIdx && actionIdx <= 68) {
    content = (
      <>
        <span className="action-verb">NOBLE</span>
        <span className="action-meta">#{actionIdx - 66}</span>
      </>
    );
  }

  return (
    <span className="action-label">
      {content}
      {showPlayed ? <span className="action-state-pill">Played</span> : null}
    </span>
  );
}
