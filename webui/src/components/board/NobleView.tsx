import { NobleDTO } from '../../types';

const REQ_ORDER: Array<keyof NobleDTO['requirements']> = ['white', 'blue', 'green', 'red', 'black'];

export function NobleView({ noble, onClick }: { noble: NobleDTO; onClick?: () => void }) {
  const reqs = REQ_ORDER.filter((color) => noble.requirements[color] > 0);
  const isPlaceholder = noble.is_placeholder === true;
  return (
    <article
      className={`noble-view ${isPlaceholder ? 'noble-placeholder' : ''} ${onClick ? 'card-clickable' : ''}`}
      aria-label={`Noble worth ${noble.points} points`}
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
      <header className="noble-head">{isPlaceholder ? '?' : noble.points}</header>
      <div className="noble-reqs">
        {isPlaceholder && <div className="card-placeholder-mark">?</div>}
        {!isPlaceholder && reqs.map((color) => (
          <span key={color} className={`req-chip cost-circle token-${color}`}>
            <b>{noble.requirements[color]}</b>
            <small>{color === 'black' ? 'K' : color === 'white' ? 'W' : color[0].toUpperCase()}</small>
          </span>
        ))}
      </div>
    </article>
  );
}
