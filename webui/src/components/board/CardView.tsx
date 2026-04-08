import { CardDTO } from '../../types';

const COST_ORDER: Array<keyof CardDTO['cost']> = ['white', 'blue', 'green', 'red', 'black'];

export function CardView(props: {
  card: CardDTO;
  showMcts?: boolean;
  showModel?: boolean;
  onClick?: () => void;
}) {
  const { card, onClick } = props;
  const isPrivate = card.source === 'reserved_private';
  const isPlaceholder = card.is_placeholder === true;
  const isHiddenReservedCard = isPrivate && isPlaceholder;
  const hiddenReservedTier = isHiddenReservedCard && (card.tier === 1 || card.tier === 2 || card.tier === 3) ? card.tier : null;
  const summary = hiddenReservedTier != null
    ? `Hidden reserved tier ${hiddenReservedTier} card`
    : `Card ${card.bonus_color} bonus, ${card.points} points`;
  const reqs = COST_ORDER.filter((color) => card.cost[color] > 0);
  return (
    <article
      className={`card-view card-${card.bonus_color} ${isPrivate ? 'card-private' : ''} ${isPlaceholder ? 'card-placeholder' : ''} ${isHiddenReservedCard ? 'card-hidden-reserved' : ''} ${hiddenReservedTier != null ? `card-hidden-reserved-tier-${hiddenReservedTier}` : ''} ${onClick ? 'card-clickable' : ''}`}
      aria-label={summary}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onClick();
        }
      } : undefined}
    >
      <header className="card-head">
        <span className="card-points">
          {isHiddenReservedCard ? '' : isPlaceholder ? '?' : card.points > 0 ? card.points : ''}
        </span>
      </header>
      <div className={`card-costs card-costs-count-${reqs.length}`}>
        {hiddenReservedTier != null && (
          <div className="card-hidden-reserved-markers" aria-hidden="true">
            {Array.from({ length: hiddenReservedTier }, (_, idx) => (
              <span key={`hidden-reserved-pip-${hiddenReservedTier}-${idx}`} />
            ))}
          </div>
        )}
        {isPlaceholder && hiddenReservedTier == null && <div className="card-placeholder-mark">?</div>}
        {!isPlaceholder && reqs.map((color) => (
          <span key={color} className={`cost-chip cost-circle token-${color}`}>
            <b>{card.cost[color]}</b>
          </span>
        ))}
      </div>
    </article>
  );
}
