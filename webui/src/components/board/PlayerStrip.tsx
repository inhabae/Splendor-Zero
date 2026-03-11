import { ActionVizDTO, PlayerBoardDTO, Seat, TokenCountsDTO } from '../../types';
import { CardView } from './CardView';
import { ColorBadge, TokenPill } from './TokenPill';

const TOKEN_ORDER: Array<keyof TokenCountsDTO> = ['white', 'blue', 'green', 'red', 'black', 'gold'];

export function PlayerStrip({
  player,
  seat,
  mctsTopAction,
  modelTopAction,
  onReservedCardClick,
}: {
  player: PlayerBoardDTO;
  seat: Seat;
  mctsTopAction?: ActionVizDTO | null;
  modelTopAction?: ActionVizDTO | null;
  onReservedCardClick?: (seat: Seat, slot: number) => void;
}) {
  const visibleReserved = player.reserved_public.filter((c) => c.source !== 'reserved_private').length;
  const mctsReservedSlot = player.is_to_move && mctsTopAction?.placement_hint.zone === 'reserved_card' ? mctsTopAction.placement_hint.slot : undefined;
  const modelReservedSlot = player.is_to_move && modelTopAction?.placement_hint.zone === 'reserved_card' ? modelTopAction.placement_hint.slot : undefined;
  return (
    <section className="player-strip" aria-label={`Player ${seat} state`}>
      <div className="player-strip-header">
        <h3>{player.display_name}</h3>
        <div className="point-badge">{player.points}★</div>
        {player.is_to_move && <div className="turn-badge">To Move</div>}
      </div>

      <div className="player-row compact">
        <div className="token-grid token-grid-tokens">
          {TOKEN_ORDER.map((color) => (
            <TokenPill key={`${seat}-tk-${color}`} color={color} count={player.tokens[color]} />
          ))}
        </div>

        <div className="token-grid token-grid-bonuses">
          <ColorBadge color="white" count={player.bonuses.white} />
          <ColorBadge color="blue" count={player.bonuses.blue} />
          <ColorBadge color="green" count={player.bonuses.green} />
          <ColorBadge color="red" count={player.bonuses.red} />
          <ColorBadge color="black" count={player.bonuses.black} />
          <div className="color-badge color-badge-empty" aria-hidden="true" />
        </div>
      </div>

      <div>
        <h4>Reserved ({visibleReserved}/3)</h4>
        <div className="reserved-row">
          {player.reserved_public.map((card, idx) => (
            <CardView
              key={`${seat}-reserved-${idx}-${card.points}-${card.bonus_color}`}
              card={card}
              showMcts={mctsReservedSlot != null && card.slot === mctsReservedSlot}
              showModel={modelReservedSlot != null && card.slot === modelReservedSlot}
              onClick={card.slot != null ? () => onReservedCardClick?.(seat, card.slot as number) : undefined}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
