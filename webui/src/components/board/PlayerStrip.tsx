import { ActionVizDTO, PlayerBoardDTO, Seat, TokenCountsDTO } from '../../types';
import { CardView } from './CardView';
import { ColorBadge, TokenPill } from './TokenPill';

const TOKEN_ORDER: Array<keyof TokenCountsDTO> = ['gold', 'white', 'blue', 'green', 'red', 'black'];

export function PlayerStrip({
  player,
  seat,
  isTerminal = false,
  mctsTopAction,
  modelTopAction,
  onReservedCardClick,
}: {
  player: PlayerBoardDTO;
  seat: Seat;
  isTerminal?: boolean;
  mctsTopAction?: ActionVizDTO | null;
  modelTopAction?: ActionVizDTO | null;
  onReservedCardClick?: (seat: Seat, slot: number) => void;
}) {
  const mctsReservedSlot = player.is_to_move && mctsTopAction?.placement_hint.zone === 'reserved_card' ? mctsTopAction.placement_hint.slot : undefined;
  const modelReservedSlot = player.is_to_move && modelTopAction?.placement_hint.zone === 'reserved_card' ? modelTopAction.placement_hint.slot : undefined;
  return (
    <section className="player-strip" aria-label={`Player ${seat} state`}>
      <div className="player-strip-header">
        <h3>{player.display_name}</h3>
        {!isTerminal && player.is_to_move && <div className="turn-badge">To Move</div>}
        <div className="point-badge">{player.points}★</div>
      </div>

      <div className="player-row compact">
        <div className="token-grid token-grid-tokens">
          {TOKEN_ORDER.map((color) => (
            <TokenPill key={`${seat}-tk-${color}`} color={color} count={player.tokens[color]} />
          ))}
        </div>

        <div className="token-grid token-grid-bonuses">
          <div className="color-badge color-badge-empty" aria-hidden="true" />
          <ColorBadge color="white" count={player.bonuses.white} />
          <ColorBadge color="blue" count={player.bonuses.blue} />
          <ColorBadge color="green" count={player.bonuses.green} />
          <ColorBadge color="red" count={player.bonuses.red} />
          <ColorBadge color="black" count={player.bonuses.black} />
        </div>
      </div>

      <div className="player-strip-reserved">
        <h4>Reserved ({player.reserved_total}/3)</h4>
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
