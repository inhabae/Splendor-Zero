import { ActionVizDTO, TierRowDTO } from '../../types';
import { CardView } from './CardView';

export function TierRow({
  tier,
  mctsTopAction,
  modelTopAction,
  onCardClick,
}: {
  tier: TierRowDTO;
  mctsTopAction?: ActionVizDTO | null;
  modelTopAction?: ActionVizDTO | null;
  onCardClick?: (tier: number, slot: number) => void;
}) {
  const mctsTier = mctsTopAction?.placement_hint.zone === 'faceup_card' ? mctsTopAction.placement_hint.tier : undefined;
  const mctsSlot = mctsTopAction?.placement_hint.zone === 'faceup_card' ? mctsTopAction.placement_hint.slot : undefined;
  const modelTier = modelTopAction?.placement_hint.zone === 'faceup_card' ? modelTopAction.placement_hint.tier : undefined;
  const modelSlot = modelTopAction?.placement_hint.zone === 'faceup_card' ? modelTopAction.placement_hint.slot : undefined;
  return (
    <section className="tier-row" aria-label={`Tier ${tier.tier} cards`}>
      <div className="tier-row-inline">
        <div className={`tier-deck-badge tier-deck-badge-${tier.tier}`} aria-label={`Tier ${tier.tier} deck`}>
          <div className="tier-deck-stack">
            <div className="tier-deck-face">
              <div className="tier-deck-pips" aria-hidden="true">
                {Array.from({ length: tier.tier }, (_, idx) => (
                  <span key={`deck-pip-${tier.tier}-${idx}`} />
                ))}
              </div>
            </div>
          </div>
          <div className="tier-deck-count">{tier.deck_count < 0 ? '?' : tier.deck_count}</div>
        </div>
        <div className="tier-row-body">
          <div className="tier-cards">
            {tier.cards.length === 0 && <div className="empty-note">No visible cards</div>}
            {tier.cards.map((card, idx) => (
              <CardView
                key={`tier-${tier.tier}-${idx}-${card.points}-${card.bonus_color}`}
                card={card}
                showMcts={mctsTier === tier.tier && mctsSlot != null && card.slot === mctsSlot}
                showModel={modelTier === tier.tier && modelSlot != null && card.slot === modelSlot}
                onClick={card.slot != null ? () => onCardClick?.(tier.tier, card.slot as number) : undefined}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
