import { CardDTO } from '../../types';

const COST_ORDER: Array<keyof CardDTO['cost']> = ['white', 'blue', 'green', 'red', 'black'];

export function CardView({ card, showMcts = false, showModel = false }: { card: CardDTO; showMcts?: boolean; showModel?: boolean }) {
  const isPrivate = card.source === 'reserved_private';
  const summary = `Card ${card.bonus_color} bonus, ${card.points} points`;
  const reqs = COST_ORDER.filter((color) => card.cost[color] > 0);
  const bonusLabel = card.bonus_color === 'black' ? 'K' : card.bonus_color === 'white' ? 'W' : card.bonus_color[0].toUpperCase();
  return (
    <article className={`card-view card-${card.bonus_color} ${isPrivate ? 'card-private' : ''}`} aria-label={summary}>
      <header className="card-head">
        <span className="card-points">{card.points}</span>
        <span className="card-bonus">{bonusLabel}</span>
      </header>
      {(showMcts || showModel) && (
        <div className="top-marker-row" aria-label="Top move markers">
          {showMcts && <span className="top-marker mcts" title="MCTS top move" />}
          {showModel && <span className="top-marker model" title="Model top move" />}
        </div>
      )}
      <div className="card-costs">
        {reqs.map((color) => (
          <span key={color} className={`cost-chip cost-circle token-${color}`}>
            <b>{card.cost[color]}</b>
            <small>{color === 'black' ? 'K' : color === 'white' ? 'W' : color[0].toUpperCase()}</small>
          </span>
        ))}
      </div>
    </article>
  );
}
