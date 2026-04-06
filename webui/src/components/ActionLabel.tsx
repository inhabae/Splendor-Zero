import { ActionDisplayDTO, BoardStateDTO, CardDTO } from '../types';

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

function faceupCard(board: BoardStateDTO | null | undefined, tier: number, slot: number): CardDTO | null {
  const row = board?.tiers.find((item) => item.tier === tier);
  if (!row) {
    return null;
  }
  return row.cards.find((card) => card.slot === slot) ?? null;
}

function reservedCard(board: BoardStateDTO | null | undefined, slot: number): CardDTO | null {
  const seatToMove = board?.meta.player_to_move;
  const player = board?.players.find((item) => item.seat === seatToMove) ?? board?.players[0];
  if (!player) {
    return null;
  }
  return player.reserved_public.find((card) => card.slot === slot) ?? null;
}

function costEntries(card: CardDTO): Array<{ color: (typeof COLOR_ORDER)[number]; count: number }> {
  return COLOR_ORDER.filter((color) => card.cost[color] > 0).map((color) => ({
    color,
    count: card.cost[color],
  }));
}

function GemChip({ color }: { color: (typeof COLOR_ORDER)[number] }) {
  return <span className={`action-gem-chip token-${color}`} aria-hidden="true" />;
}

function CardActionLabel({ verb, card }: { verb: 'BUY' | 'RESERVE'; card: CardDTO | null }) {
  if (!card) {
    return <span className="action-verb">{verb}</span>;
  }
  const toneClass = `action-tone token-${card.bonus_color}`;
  return (
    <>
      <span className="action-verb">{verb}</span>
      <span className={`action-group action-group-card-reqs ${toneClass}`}>
        {costEntries(card).map(({ color, count }) => (
          <span key={`${verb}-${color}-${count}`} className="action-cost action-cost-inline">
            <GemChip color={color} />
            <span>{count}</span>
          </span>
        ))}
      </span>
    </>
  );
}

function TakeLabel({ verb, colors, duplicate = 1 }: { verb: 'TAKE' | 'RETURN'; colors: readonly number[]; duplicate?: number }) {
  return (
    <span className="action-group action-group-take">
      <span className="action-verb">{verb}</span>
      {colors.map((colorIdx, idx) => {
        const color = COLOR_ORDER[colorIdx];
        return (
          <span key={`${verb}-${color}-${idx}`} className="action-gem">
            <GemChip color={color} />
            {duplicate > 1 && idx === 0 ? <span className="action-mult">x{duplicate}</span> : null}
          </span>
        );
      })}
    </span>
  );
}

function DeckReserveLabel({ tier }: { tier: 1 | 2 | 3 }) {
  return (
    <>
      <span className="action-verb">RESERVE</span>
      <span className="action-meta">from</span>
      <span className={`action-meta action-tier-${tier}`}>T{tier} deck</span>
    </>
  );
}

export function actionTextLabel(actionIdx: number): string {
  if (0 <= actionIdx && actionIdx <= 11) return 'BUY';
  if (12 <= actionIdx && actionIdx <= 14) return 'BUY';
  if (15 <= actionIdx && actionIdx <= 26) return 'RESERVE';
  if (27 <= actionIdx && actionIdx <= 29) return `RESERVE from T${actionIdx - 26} deck`;
  if (30 <= actionIdx && actionIdx <= 39) return 'TAKE';
  if (40 <= actionIdx && actionIdx <= 44) return 'TAKE x2';
  if (45 <= actionIdx && actionIdx <= 54) return 'TAKE';
  if (55 <= actionIdx && actionIdx <= 59) return 'TAKE';
  if (actionIdx === 60) return 'PASS';
  if (61 <= actionIdx && actionIdx <= 65) return 'RETURN';
  if (66 <= actionIdx && actionIdx <= 68) return `NOBLE #${actionIdx - 66}`;
  return 'UNKNOWN';
}

export function ActionLabel({
  actionIdx,
  board,
  display,
  showPlayed = false,
}: {
  actionIdx: number;
  board?: BoardStateDTO | null;
  display?: ActionDisplayDTO | null;
  showPlayed?: boolean;
}) {
  let content: JSX.Element | JSX.Element[] = <span className="action-verb">UNKNOWN</span>;

  if (display) {
    if (display.kind === 'card') {
      content = <CardActionLabel verb={display.verb as 'BUY' | 'RESERVE'} card={display.card ?? null} />;
    } else if (display.kind === 'deck' && display.tier != null) {
      content = <DeckReserveLabel tier={display.tier as 1 | 2 | 3} />;
    } else if (display.kind === 'tokens') {
      const colors = (display.token_colors ?? [])
        .map((color) => COLOR_ORDER.indexOf(color))
        .filter((idx) => idx >= 0);
      content = <TakeLabel verb={display.verb as 'TAKE' | 'RETURN'} colors={colors} duplicate={display.token_duplicate ?? 1} />;
    } else if (display.kind === 'pass') {
      content = <span className="action-verb">PASS</span>;
    } else if (display.kind === 'noble') {
      content = (
        <>
          <span className="action-verb">NOBLE</span>
          <span className="action-meta">#{display.noble_slot ?? 0}</span>
        </>
      );
    }
  } else if (0 <= actionIdx && actionIdx <= 11) {
    const tier = Math.floor(actionIdx / 4) + 1;
    const slot = actionIdx % 4;
    content = <CardActionLabel verb="BUY" card={faceupCard(board, tier, slot)} />;
  } else if (12 <= actionIdx && actionIdx <= 14) {
    content = <CardActionLabel verb="BUY" card={reservedCard(board, actionIdx - 12)} />;
  } else if (15 <= actionIdx && actionIdx <= 26) {
    const rel = actionIdx - 15;
    const tier = Math.floor(rel / 4) + 1;
    const slot = rel % 4;
    content = <CardActionLabel verb="RESERVE" card={faceupCard(board, tier, slot)} />;
  } else if (27 <= actionIdx && actionIdx <= 29) {
    content = <DeckReserveLabel tier={(actionIdx - 26) as 1 | 2 | 3} />;
  } else if (30 <= actionIdx && actionIdx <= 39) {
    content = <TakeLabel verb="TAKE" colors={TAKE3_TRIPLETS[actionIdx - 30]} />;
  } else if (40 <= actionIdx && actionIdx <= 44) {
    content = <TakeLabel verb="TAKE" colors={[actionIdx - 40]} duplicate={2} />;
  } else if (45 <= actionIdx && actionIdx <= 54) {
    content = <TakeLabel verb="TAKE" colors={TAKE2_PAIRS[actionIdx - 45]} />;
  } else if (55 <= actionIdx && actionIdx <= 59) {
    content = <TakeLabel verb="TAKE" colors={[actionIdx - 55]} />;
  } else if (actionIdx === 60) {
    content = <span className="action-verb">PASS</span>;
  } else if (61 <= actionIdx && actionIdx <= 65) {
    content = <TakeLabel verb="RETURN" colors={[actionIdx - 61]} />;
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
