import { ColorCountsDTO, TokenCountsDTO } from '../../types';

type TokenKey = keyof TokenCountsDTO;

export function TokenPill(props: { color: TokenKey; count: number; showMcts?: boolean; showModel?: boolean }) {
  const { color, count } = props;
  return (
    <div className={`token-pill token-${color} ${count === 0 ? 'is-zero-count' : ''}`} aria-label={`${color} token count ${count}`}>
      <span className="token-pill-count">{count}</span>
    </div>
  );
}

export function ColorBadge({ color, count }: { color: keyof ColorCountsDTO; count: number }) {
  return (
    <div className={`color-badge token-${color} ${count === 0 ? 'is-zero-count' : ''}`} aria-label={`${color} bonus count ${count}`}>
      <strong>{count}</strong>
    </div>
  );
}
