import { ColorCountsDTO, TokenCountsDTO } from '../../types';

type TokenKey = keyof TokenCountsDTO;

const TOKEN_LABELS: Record<TokenKey, string> = {
  white: 'W',
  blue: 'B',
  green: 'G',
  red: 'R',
  black: 'K',
  gold: 'Gd',
};

export function TokenPill({ color, count, showMcts = false, showModel = false }: { color: TokenKey; count: number; showMcts?: boolean; showModel?: boolean }) {
  return (
    <div className={`token-pill token-${color}`} aria-label={`${color} token count ${count}`}>
      <span className="token-pill-label">{TOKEN_LABELS[color]}</span>
      <span className="token-pill-count">{count}</span>
      {(showMcts || showModel) && (
        <span className="token-marker-row" aria-label="Top move markers">
          {showMcts && <span className="top-marker mcts" title="MCTS top move">MC</span>}
          {showModel && <span className="top-marker model" title="Model top move">NN</span>}
        </span>
      )}
    </div>
  );
}

export function ColorBadge({ color, count }: { color: keyof ColorCountsDTO; count: number }) {
  const label = color === 'black' ? 'K' : color[0].toUpperCase();
  return (
    <div className={`color-badge token-${color}`} aria-label={`${color} bonus count ${count}`}>
      <span>{label}</span>
      <strong>{count}</strong>
    </div>
  );
}
