import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  CatalogCardDTO,
  CatalogNobleDTO,
  BoardStateDTO,
  CheckpointDTO,
  EngineJobStatusDTO,
  EngineThinkResponse,
  GameSnapshotDTO,
  LiveSaveStatusDTO,
  PlayerMoveResponse,
  RevealCardResponse,
  SavedGameDTO,
  SearchType,
  Seat,
} from './types';
import { GameBoard } from './components/board/GameBoard';
import { ActionLabel } from './components/ActionLabel';
import { CardView } from './components/board/CardView';
import { NobleView } from './components/board/NobleView';

type UiStatus = 'IDLE' | 'WAITING_ENGINE' | 'WAITING_PLAYER' | 'WAITING_REVEAL' | 'GAME_OVER';
type HomeView = 'HOME' | 'QUICK' | 'SETUP' | 'ANALYSIS' | 'LIVE';
const COLOR_ORDER: CatalogCardDTO['bonus_color'][] = ['white', 'blue', 'green', 'red', 'black'];

const POLL_MS = 400;
const LIVE_POLL_MS = 1000;
const ACTIONS_PAGE_SIZE = 10;
const LIVE_SEARCH_MAX_SIMULATIONS = 500_000;

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // Ignore parse errors and keep status text.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function isBlockingPendingReveal(reveal: GameSnapshotDTO['pending_reveals'][number]): boolean {
  return reveal.zone !== 'reserved_card';
}

function parseRevealKey(key: string): { zone: 'faceup_card' | 'reserved_card' | 'noble'; tier: number; slot: number; seat?: Seat } | null {
  const [zone, tier, slot, seat] = key.split('-');
  if ((zone !== 'faceup_card' && zone !== 'reserved_card' && zone !== 'noble') || tier == null || slot == null) {
    return null;
  }
  if (zone === 'reserved_card' && seat !== 'P0' && seat !== 'P1') {
    return null;
  }
  return {
    zone,
    tier: Number(tier),
    slot: Number(slot),
    seat: zone === 'reserved_card' ? (seat as Seat) : undefined,
  };
}

export function App() {
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [catalogCards, setCatalogCards] = useState<CatalogCardDTO[]>([]);
  const [catalogNobles, setCatalogNobles] = useState<CatalogNobleDTO[]>([]);
  const [checkpointId, setCheckpointId] = useState('');
  const [numSimulations] = useState(400);
  const [searchSimulations, setSearchSimulations] = useState(400);
  const [searchType, setSearchType] = useState<SearchType>('mcts');
  const [playerSeat] = useState<Seat>('P0');
  const [seed] = useState('');
  const [homeView, setHomeView] = useState<HomeView>('HOME');
  const [revealSelections, setRevealSelections] = useState<Record<string, string>>({});
  const [activeRevealKey, setActiveRevealKey] = useState<string | null>(null);
  const [liveActionsPage, setLiveActionsPage] = useState(1);

  const [snapshot, setSnapshot] = useState<GameSnapshotDTO | null>(null);
  const [jobStatus, setJobStatus] = useState<EngineJobStatusDTO | null>(null);
  const [uiStatus, setUiStatus] = useState<UiStatus>('IDLE');
  const [liveSaveStatus, setLiveSaveStatus] = useState<LiveSaveStatusDTO | null>(null);

  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);
  const livePollRef = useRef<number | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const loadInputRef = useRef<HTMLInputElement | null>(null);
  const isSetupLikeView = homeView === 'SETUP' || homeView === 'ANALYSIS';
  const lastLiveSaveUpdatedAtRef = useRef<string | null>(null);
  const lastAutoAnalyzeKeyRef = useRef<string | null>(null);

  const selectedCheckpoint = useMemo(
    () => checkpoints.find((item) => item.id === checkpointId) ?? null,
    [checkpoints, checkpointId],
  );
  const cardsByTier = useMemo(() => {
    return catalogCards.reduce<Record<number, CatalogCardDTO[]>>((acc, card) => {
      if (!acc[card.tier]) {
        acc[card.tier] = [];
      }
      acc[card.tier].push(card);
      return acc;
    }, {});
  }, [catalogCards]);
  const cardsByTierAndColor = useMemo(() => {
    const grouped: Record<number, Record<CatalogCardDTO['bonus_color'], CatalogCardDTO[]>> = {
      1: { white: [], blue: [], green: [], red: [], black: [] },
      2: { white: [], blue: [], green: [], red: [], black: [] },
      3: { white: [], blue: [], green: [], red: [], black: [] },
    };
    for (const card of catalogCards) {
      grouped[card.tier][card.bonus_color].push(card);
    }
    for (const tier of [1, 2, 3] as const) {
      for (const color of COLOR_ORDER) {
        grouped[tier][color].sort((a, b) => {
          const aTotal = a.cost.white + a.cost.blue + a.cost.green + a.cost.red + a.cost.black;
          const bTotal = b.cost.white + b.cost.blue + b.cost.green + b.cost.red + b.cost.black;
          if (aTotal !== bTotal) return aTotal - bTotal;
          if (a.points !== b.points) return a.points - b.points;
          return a.id - b.id;
        });
      }
    }
    return grouped;
  }, [catalogCards]);

  useEffect(() => {
    void (async () => {
      try {
        const list = await fetchJSON<CheckpointDTO[]>('/api/checkpoints');
        const cards = await fetchJSON<CatalogCardDTO[]>('/api/cards');
        const nobles = await fetchJSON<CatalogNobleDTO[]>('/api/nobles');
        setCheckpoints(list);
        setCatalogCards(cards);
        setCatalogNobles(nobles);
        if (!checkpointId && list.length > 0) {
          setCheckpointId(list[0].id);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
      }
      if (livePollRef.current !== null) {
        window.clearInterval(livePollRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!snapshot?.pending_reveals.length) {
      return;
    }
    setRevealSelections((prev) => {
      const next = { ...prev };
      for (const reveal of snapshot.pending_reveals) {
        const key = revealKey(reveal.zone, reveal.tier, reveal.slot, reveal.actor ?? undefined);
        if (!(key in next)) {
          if (reveal.zone === 'noble') {
            next[key] = catalogNobles[0] ? String(catalogNobles[0].id) : '';
          } else {
            next[key] = cardsByTier[reveal.tier]?.[0] ? String(cardsByTier[reveal.tier][0].id) : '';
          }
        }
      }
      return next;
    });
  }, [snapshot, cardsByTier, catalogNobles]);

  useEffect(() => {
    if (!snapshot?.pending_reveals.length) {
      setActiveRevealKey(null);
      return;
    }
    setActiveRevealKey((prev) => {
      if (prev && snapshot.pending_reveals.some((reveal) => revealKey(reveal.zone, reveal.tier, reveal.slot, reveal.actor ?? undefined) === prev)) {
        return prev;
      }
      const firstBlocking = snapshot.pending_reveals.find((reveal) => isBlockingPendingReveal(reveal));
      if (!firstBlocking) {
        return null;
      }
      return revealKey(firstBlocking.zone, firstBlocking.tier, firstBlocking.slot, firstBlocking.actor ?? undefined);
    });
  }, [snapshot]);

  function clearPolling(): void {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    activeJobIdRef.current = null;
  }

  function clearLivePolling(): void {
    if (livePollRef.current !== null) {
      window.clearInterval(livePollRef.current);
      livePollRef.current = null;
    }
  }

  function deriveUiStatus(nextSnapshot: GameSnapshotDTO): UiStatus {
    if (nextSnapshot.status !== 'IN_PROGRESS') {
      return 'GAME_OVER';
    }
    if (nextSnapshot.pending_reveals.some((reveal) => isBlockingPendingReveal(reveal))) {
      return 'WAITING_REVEAL';
    }
    return 'WAITING_PLAYER';
  }

  function revealKey(zone: 'faceup_card' | 'reserved_card' | 'noble', tier: number, slot: number, seat?: Seat): string {
    return zone === 'reserved_card' ? `${zone}-${tier}-${slot}-${seat ?? 'P0'}` : `${zone}-${tier}-${slot}`;
  }

  function shouldAutoAnalyze(nextSnapshot: GameSnapshotDTO | null): boolean {
    if (!nextSnapshot || !nextSnapshot.config?.analysis_mode) {
      return false;
    }
    if (nextSnapshot.status !== 'IN_PROGRESS') {
      return false;
    }
    return !(nextSnapshot.pending_reveals?.some((reveal) => isBlockingPendingReveal(reveal)) ?? false);
  }

  function autoAnalyzeKey(nextSnapshot: GameSnapshotDTO): string {
    return [
      nextSnapshot.game_id,
      nextSnapshot.status,
      nextSnapshot.turn_index,
      nextSnapshot.player_to_move,
      nextSnapshot.winner,
      nextSnapshot.pending_reveals.length,
      nextSnapshot.move_log.length,
    ].join(':');
  }

  async function handleSnapshotUpdate(nextSnapshot: GameSnapshotDTO, engineShouldMove = false): Promise<void> {
    clearPolling();
    setJobStatus(null);
    setSnapshot(nextSnapshot);
    setUiStatus(deriveUiStatus(nextSnapshot));
    const nextAutoAnalyzeKey = autoAnalyzeKey(nextSnapshot);
    const shouldStartSearch =
      engineShouldMove ||
      (shouldAutoAnalyze(nextSnapshot) && lastAutoAnalyzeKeyRef.current !== nextAutoAnalyzeKey);
    if (shouldStartSearch) {
      lastAutoAnalyzeKeyRef.current = nextAutoAnalyzeKey;
      await startEngineThink(searchSimulations);
    }
  }

  function describePendingReveal(reason: GameSnapshotDTO['pending_reveals'][number]['reason']): string {
    if (reason === 'initial_setup') return 'Initial board setup';
    if (reason === 'initial_noble_setup') return 'Initial noble setup';
    if (reason === 'replacement_after_buy') return 'Replacement after buy';
    if (reason === 'replacement_after_reserve') return 'Replacement after reserve';
    return 'Reveal reserved deck card';
  }

  function cardOptionLabel(card: CatalogCardDTO): string {
    const cost = Object.entries(card.cost)
      .filter(([, count]) => count > 0)
      .map(([color, count]) => `${count}${color[0].toUpperCase()}`)
      .join(' ');
    return `#${card.id} ${card.bonus_color} ${card.points}pt${cost ? ` | ${cost}` : ''}`;
  }

  function nobleOptionLabel(noble: CatalogNobleDTO): string {
    const reqs = Object.entries(noble.requirements)
      .filter(([, count]) => count > 0)
      .map(([color, count]) => `${count}${color[0].toUpperCase()}`)
      .join(' ');
    return `#${noble.id} ${noble.points}pt${reqs ? ` | ${reqs}` : ''}`;
  }

  function findCatalogCardId(card: BoardStateDTO['tiers'][number]['cards'][number]): number | null {
    const match = catalogCards.find((candidate) =>
      candidate.tier === card.tier &&
      candidate.points === card.points &&
      candidate.bonus_color === card.bonus_color &&
      candidate.cost.white === card.cost.white &&
      candidate.cost.blue === card.cost.blue &&
      candidate.cost.green === card.cost.green &&
      candidate.cost.red === card.cost.red &&
      candidate.cost.black === card.cost.black
    );
    return match?.id ?? null;
  }

  function findCatalogNobleId(noble: BoardStateDTO['nobles'][number]): number | null {
    const match = catalogNobles.find((candidate) =>
      candidate.points === noble.points &&
      candidate.requirements.white === noble.requirements.white &&
      candidate.requirements.blue === noble.requirements.blue &&
      candidate.requirements.green === noble.requirements.green &&
      candidate.requirements.red === noble.requirements.red &&
      candidate.requirements.black === noble.requirements.black
    );
    return match?.id ?? null;
  }

  async function startEngineThink(customNumSimulations?: number): Promise<void> {
    setError(null);
    const requested = searchSimulations;
    const fallback = snapshot?.config?.num_simulations ?? numSimulations;
    const nextNumSimulations =
      Number.isInteger(requested) && requested >= 1
        ? requested
        : fallback;
    const think = await fetchJSON<EngineThinkResponse>('/api/game/engine-think', {
      method: 'POST',
      body: JSON.stringify({
        num_simulations: nextNumSimulations,
        search_type: searchType,
        continuous_until_cancel: homeView === 'LIVE',
        max_total_simulations: homeView === 'LIVE' ? LIVE_SEARCH_MAX_SIMULATIONS : nextNumSimulations,
      }),
    });
    setUiStatus('WAITING_ENGINE');
    clearPolling();
    activeJobIdRef.current = think.job_id;

    pollRef.current = window.setInterval(() => {
      void pollEngineJob(think.job_id);
    }, POLL_MS);
  }

  async function pollEngineJob(nextJobId: string): Promise<void> {
    try {
      const status = await fetchJSON<EngineJobStatusDTO>(`/api/game/engine-job/${nextJobId}`);
      if (activeJobIdRef.current !== nextJobId) {
        return;
      }
      setJobStatus(status);
      if (status.status === 'DONE') {
        clearPolling();
        setUiStatus(snapshot?.status === 'IN_PROGRESS' ? 'WAITING_PLAYER' : 'GAME_OVER');
      } else if (status.status === 'FAILED' || status.status === 'CANCELLED') {
        clearPolling();
        setUiStatus('WAITING_PLAYER');
      }
    } catch (err) {
      clearPolling();
      setError((err as Error).message);
      setUiStatus('WAITING_PLAYER');
    }
  }

  async function startGame(manualRevealMode: boolean, playerSeatOverride?: Seat, analysisModeOverride?: boolean): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    setRevealSelections({});
    setActiveRevealKey(null);
    lastAutoAnalyzeKeyRef.current = null;

    if (!checkpointId) {
      throw new Error('Please choose a checkpoint');
    }
      const payload = {
        checkpoint_id: checkpointId,
        num_simulations: Number(numSimulations),
        player_seat: playerSeatOverride ?? playerSeat,
        manual_reveal_mode: manualRevealMode,
        analysis_mode: analysisModeOverride ?? true,
        ...(seed.trim().length > 0 ? { seed: Number(seed) } : {}),
      };

    const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/new', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    await handleSnapshotUpdate(nextSnapshot);
  }

  async function onStartGame(event: FormEvent): Promise<void> {
    event.preventDefault();
    try {
      await startGame(false, playerSeat, false);
      setHomeView('QUICK');
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function onOpenManualView(view: 'SETUP' | 'ANALYSIS'): void {
    setError(null);
    clearPolling();
    clearLivePolling();
    setJobStatus(null);
    setSnapshot(null);
    lastAutoAnalyzeKeyRef.current = null;
    setHomeView(view);
  }

  function onOpenLiveView(): void {
    setError(null);
    clearPolling();
    clearLivePolling();
    setJobStatus(null);
    setSnapshot(null);
    setRevealSelections({});
    setActiveRevealKey(null);
    setLiveSaveStatus(null);
    lastLiveSaveUpdatedAtRef.current = null;
    lastAutoAnalyzeKeyRef.current = null;
    setHomeView('LIVE');
  }

  async function onStartManualGame(event: FormEvent, view: 'SETUP' | 'ANALYSIS'): Promise<void> {
    event.preventDefault();
    try {
      await startGame(true, playerSeat, view === 'ANALYSIS');
      setHomeView(view);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onPlayerMove(actionIdx: number): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    try {
      const result = await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
        method: 'POST',
        body: JSON.stringify({ action_idx: actionIdx }),
      });
      await handleSnapshotUpdate(result.snapshot, result.engine_should_move);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onUndo(): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    try {
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/undo', {
        method: 'POST',
        body: '{}',
      });
      await handleSnapshotUpdate(nextSnapshot);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onRedo(): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    try {
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/redo', {
        method: 'POST',
        body: '{}',
      });
      await handleSnapshotUpdate(nextSnapshot);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onRevealCardWithId(tier: number, slot: number, cardId?: number): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    const key = revealKey('faceup_card', tier, slot);
    const selected = cardId != null ? String(cardId) : revealSelections[key];
    if (!selected) {
      setError(`Choose a card for tier ${tier} slot ${slot}`);
      return;
    }

    try {
      const result = await fetchJSON<RevealCardResponse>('/api/game/reveal-card', {
        method: 'POST',
        body: JSON.stringify({ tier, slot, card_id: Number(selected) }),
      });
      setRevealSelections((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      await handleSnapshotUpdate(result.snapshot, result.engine_should_move);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onRevealReservedCardWithId(seat: Seat, tier: number, slot: number, cardId?: number): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    const key = revealKey('reserved_card', tier, slot, seat);
    const selected = cardId != null ? String(cardId) : revealSelections[key];
    if (!selected) {
      setError(`Choose a card for ${seat} reserved slot ${slot}`);
      return;
    }

    try {
      const result = await fetchJSON<RevealCardResponse>('/api/game/reveal-reserved-card', {
        method: 'POST',
        body: JSON.stringify({ seat, slot, card_id: Number(selected) }),
      });
      setRevealSelections((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      await handleSnapshotUpdate(result.snapshot, result.engine_should_move);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onRevealNobleWithId(slot: number, nobleId?: number): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    const key = revealKey('noble', 0, slot);
    const selected = nobleId != null ? String(nobleId) : revealSelections[key];
    if (!selected) {
      setError(`Choose a noble for slot ${slot}`);
      return;
    }

    try {
      const result = await fetchJSON<RevealCardResponse>('/api/game/reveal-noble', {
        method: 'POST',
        body: JSON.stringify({ slot, noble_id: Number(selected) }),
      });
      setRevealSelections((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      await handleSnapshotUpdate(result.snapshot, result.engine_should_move);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function openReveal(zone: 'faceup_card' | 'reserved_card' | 'noble', tier: number, slot: number, seat?: Seat): void {
    const key = revealKey(zone, tier, slot, seat);
    const hasPending = snapshot?.pending_reveals.some((reveal) => revealKey(reveal.zone, reveal.tier, reveal.slot, reveal.actor ?? undefined) === key) ?? false;
    const setupEditable =
      isSetupLikeView &&
      snapshot?.pending_reveals.some((reveal) => reveal.reason === 'initial_setup' || reveal.reason === 'initial_noble_setup') &&
      (zone === 'faceup_card' || zone === 'noble');
    const manualRevealEditable = Boolean(snapshot?.config?.manual_reveal_mode) && (zone === 'faceup_card' || zone === 'reserved_card' || zone === 'noble');
    if (!hasPending && !setupEditable && !manualRevealEditable) {
      return;
    }
    if ((setupEditable || manualRevealEditable) && snapshot?.board_state) {
      if (zone === 'faceup_card') {
        const row = snapshot.board_state.tiers.find((item) => item.tier === tier);
        const current = row?.cards.find((card) => card.slot === slot);
        if (current && !current.is_placeholder) {
          const cardId = findCatalogCardId(current);
          if (cardId != null) {
            setRevealSelections((prev) => ({ ...prev, [key]: String(cardId) }));
          }
        }
      } else if (zone === 'reserved_card' && seat) {
        const player = snapshot.board_state.players.find((item) => item.seat === seat);
        const current = player?.reserved_public.find((card) => card.slot === slot);
        if (current && !current.is_placeholder) {
          const cardId = findCatalogCardId(current);
          if (cardId != null) {
            setRevealSelections((prev) => ({ ...prev, [key]: String(cardId) }));
          }
        }
      } else if (zone === 'noble') {
        const current = snapshot.board_state.nobles.find((noble) => noble.slot === slot);
        if (current && !current.is_placeholder) {
          const nobleId = findCatalogNobleId(current);
          if (nobleId != null) {
            setRevealSelections((prev) => ({ ...prev, [key]: String(nobleId) }));
          }
        }
      }
    }
    setActiveRevealKey(key);
  }

  function deriveHomeViewFromSnapshot(nextSnapshot: GameSnapshotDTO): HomeView {
    if (nextSnapshot.config?.manual_reveal_mode) {
      return nextSnapshot.config.analysis_mode ? 'ANALYSIS' : 'SETUP';
    }
    return 'QUICK';
  }

  async function onSaveGame(): Promise<void> {
    setError(null);
    try {
      const saved = await fetchJSON<SavedGameDTO>('/api/game/save');
      const blob = new Blob([JSON.stringify(saved, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      const safeTime = saved.saved_at.replace(/:/g, '-');
      anchor.href = url;
      anchor.download = `splendor-game-${safeTime}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function onLoadGameClick(): void {
    loadInputRef.current?.click();
  }

  async function onLoadGameFile(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) {
      return;
    }

    setError(null);
    clearPolling();
    setJobStatus(null);
    setRevealSelections({});
    setActiveRevealKey(null);
    lastAutoAnalyzeKeyRef.current = null;

    try {
      const raw = await file.text();
      const saved = JSON.parse(raw) as SavedGameDTO;
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/load', {
        method: 'POST',
        body: JSON.stringify(saved),
      });
      if (nextSnapshot.config?.checkpoint_id) {
        setCheckpointId(nextSnapshot.config.checkpoint_id);
      }
      setHomeView(deriveHomeViewFromSnapshot(nextSnapshot));
      await handleSnapshotUpdate(nextSnapshot);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const canStart = Boolean(selectedCheckpoint) && numSimulations > 0 && numSimulations <= 10000;
  const isAnalysisSession = Boolean(snapshot?.config?.analysis_mode);
  const canMove =
    homeView !== 'LIVE' &&
    snapshot?.status === 'IN_PROGRESS' &&
    !(snapshot.pending_reveals?.some((reveal) => isBlockingPendingReveal(reveal)) ?? false) &&
    (isAnalysisSession || snapshot.player_to_move === snapshot.config?.player_seat);
  const activeReveal = useMemo(() => {
    if (!snapshot || !activeRevealKey) {
      return null;
    }
    const pending = snapshot.pending_reveals.find(
      (reveal) => revealKey(reveal.zone, reveal.tier, reveal.slot, reveal.actor ?? undefined) === activeRevealKey,
    );
    if (pending) {
      return pending;
    }
    const parsed = parseRevealKey(activeRevealKey);
    if (
      parsed &&
      isSetupLikeView &&
      snapshot.pending_reveals.some((reveal) => reveal.reason === 'initial_setup' || reveal.reason === 'initial_noble_setup')
    ) {
      return {
        zone: parsed.zone,
        tier: parsed.tier,
        slot: parsed.slot,
        actor: parsed.seat ?? null,
        reason: (parsed.zone === 'noble' ? 'initial_noble_setup' : 'initial_setup') as 'initial_noble_setup' | 'initial_setup',
        action_idx: null,
      };
    }
    if (parsed && parsed.zone === 'faceup_card' && snapshot.config?.manual_reveal_mode) {
      return {
        zone: parsed.zone,
        tier: parsed.tier,
        slot: parsed.slot,
        actor: null,
        reason: 'replacement_after_buy' as const,
        action_idx: null,
      };
    }
    if (parsed && parsed.zone === 'reserved_card' && parsed.seat && snapshot.config?.manual_reveal_mode) {
      return {
        zone: parsed.zone,
        tier: parsed.tier,
        slot: parsed.slot,
        actor: parsed.seat,
        reason: 'reserved_from_deck' as const,
        action_idx: null,
      };
    }
    if (parsed && parsed.zone === 'noble' && snapshot.config?.manual_reveal_mode) {
      return {
        zone: parsed.zone,
        tier: parsed.tier,
        slot: parsed.slot,
        actor: null,
        reason: 'initial_noble_setup' as const,
        action_idx: null,
      };
    }
    return null;
  }, [snapshot, activeRevealKey, isSetupLikeView]);
  const liveMctsTopAction = useMemo(() => {
    const details = jobStatus?.result?.action_details;
    if (!details?.length) return null;
    let best: typeof details[number] | null = null;
    for (const action of details) {
      if (action.masked) continue;
      if (best == null || action.policy_prob > best.policy_prob) best = action;
    }
    return best;
  }, [jobStatus]);
  const liveModelTopAction = useMemo(() => {
    const details = jobStatus?.result?.model_action_details;
    if (!details?.length) return null;
    let best: typeof details[number] | null = null;
    for (const action of details) {
      if (action.masked) continue;
      if (best == null || action.policy_prob > best.policy_prob) best = action;
    }
    return best;
  }, [jobStatus]);
  const liveRows = useMemo(() => {
    const mcts = jobStatus?.result?.action_details;
    if (!mcts?.length) {
      return (snapshot?.legal_action_details ?? []).map((action) => ({
        action: {
          ...action,
          masked: false,
          policy_prob: 0,
          q_value: null,
          is_selected: false,
          placement_hint: { zone: 'other' as const },
        },
        modelProb: 0,
      }));
    }
    const model = jobStatus?.result?.model_action_details;
    return mcts
      .map((action, idx) => ({ action, modelProb: model?.[idx]?.policy_prob ?? 0 }))
      .filter((row) => !row.action.masked)
      .sort((a, b) => {
        if (b.action.policy_prob !== a.action.policy_prob) return b.action.policy_prob - a.action.policy_prob;
        return b.modelProb - a.modelProb;
      });
  }, [jobStatus, snapshot]);
  const displayBoard = useMemo(() => {
    if (!snapshot?.board_state) {
      return null;
    }
    const board: BoardStateDTO = structuredClone(snapshot.board_state);
    const pendingByKey = new Set(snapshot.pending_reveals.map((reveal) => revealKey(reveal.zone, reveal.tier, reveal.slot)));

    board.tiers = board.tiers.map((tier) => {
      const bySlot = new Map(tier.cards.map((card) => [card.slot ?? -1, card]));
      const cards = Array.from({ length: 4 }, (_, slot) => {
        const key = revealKey('faceup_card', tier.tier, slot);
        if (pendingByKey.has(key)) {
          return {
            points: 0,
            bonus_color: 'white',
            cost: { white: 0, blue: 0, green: 0, red: 0, black: 0 },
            source: 'faceup' as const,
            tier: tier.tier,
            slot,
            is_placeholder: true,
          };
        }
        return bySlot.get(slot) ?? {
          points: 0,
          bonus_color: 'white',
          cost: { white: 0, blue: 0, green: 0, red: 0, black: 0 },
          source: 'faceup' as const,
          tier: tier.tier,
          slot,
          is_placeholder: true,
        };
      });
      return { ...tier, cards };
    }) as BoardStateDTO['tiers'];

    const nobleBySlot = new Map((board.nobles ?? []).map((noble) => [noble.slot ?? -1, noble]));
    board.nobles = Array.from({ length: 3 }, (_, slot) => {
      const key = revealKey('noble', 0, slot);
      if (pendingByKey.has(key)) {
        return {
          points: 0,
          requirements: { white: 0, blue: 0, green: 0, red: 0, black: 0 },
          slot,
          is_placeholder: true,
        };
      }
      return nobleBySlot.get(slot) ?? {
        points: 0,
        requirements: { white: 0, blue: 0, green: 0, red: 0, black: 0 },
        slot,
        is_placeholder: true,
      };
    }) as BoardStateDTO['nobles'];

    return board;
  }, [snapshot, activeRevealKey]);
  const activeTierBoardCards = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'faceup_card' || !displayBoard) {
      return [] as BoardStateDTO['tiers'][number]['cards'];
    }
    return displayBoard.tiers.find((tier) => tier.tier === activeReveal.tier)?.cards ?? [];
  }, [activeReveal, displayBoard]);
  const activeBoardNobles = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'noble' || !displayBoard) {
      return [] as BoardStateDTO['nobles'];
    }
    return displayBoard.nobles ?? [];
  }, [activeReveal, displayBoard]);
  const setupUnavailableCardIds = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'faceup_card' || !isSetupLikeView || activeReveal.reason !== 'initial_setup' || !displayBoard) {
      return new Set<number>();
    }
    const ids = new Set<number>();
    const cards = displayBoard.tiers.find((tier) => tier.tier === activeReveal.tier)?.cards ?? [];
    for (const card of cards) {
      if (card.slot === activeReveal.slot || card.is_placeholder) {
        continue;
      }
      const id = findCatalogCardId(card);
      if (id != null) {
        ids.add(id);
      }
    }
    return ids;
  }, [activeReveal, isSetupLikeView, displayBoard, catalogCards]);
  const occupiedBoardCardIds = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'faceup_card' || !displayBoard) {
      return new Set<number>();
    }
    const ids = new Set<number>();
    const cards = displayBoard.tiers.find((tier) => tier.tier === activeReveal.tier)?.cards ?? [];
    for (const card of cards) {
      if (card.slot === activeReveal.slot || card.is_placeholder) {
        continue;
      }
      const id = findCatalogCardId(card);
      if (id != null) {
        ids.add(id);
      }
    }
    return ids;
  }, [activeReveal, displayBoard, catalogCards]);
  const setupUnavailableNobleIds = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'noble' || !isSetupLikeView || activeReveal.reason !== 'initial_noble_setup' || !displayBoard) {
      return new Set<number>();
    }
    const ids = new Set<number>();
    for (const noble of displayBoard.nobles) {
      if (noble.slot === activeReveal.slot || noble.is_placeholder) {
        continue;
      }
      const id = findCatalogNobleId(noble);
      if (id != null) {
        ids.add(id);
      }
    }
    return ids;
  }, [activeReveal, isSetupLikeView, displayBoard, catalogNobles]);
  const liveAvailableCardIds = useMemo(() => {
    if (!activeReveal || !snapshot) {
      return new Set<number>();
    }
    if (activeReveal.zone === 'faceup_card') {
      const pendingKey = `${activeReveal.tier}:${activeReveal.slot}`;
      const hasPendingFaceupReveal = snapshot.pending_reveals.some(
        (reveal) =>
          reveal.zone === 'faceup_card' &&
          reveal.tier === activeReveal.tier &&
          reveal.slot === activeReveal.slot,
      );
      const ids = new Set<number>(
        hasPendingFaceupReveal
          ? (snapshot.hidden_faceup_reveal_candidates[pendingKey] ?? [])
          : (snapshot.hidden_deck_card_ids_by_tier[activeReveal.tier] ?? []),
      );
      if (!hasPendingFaceupReveal) {
        const tierCards = displayBoard?.tiers.find((tier) => tier.tier === activeReveal.tier)?.cards ?? [];
        for (const card of tierCards) {
          if (card.is_placeholder) {
            continue;
          }
          const id = findCatalogCardId(card);
          if (id != null) {
            ids.add(id);
          }
        }
      }
      return ids;
    }
    if (activeReveal.zone === 'reserved_card' && activeReveal.actor) {
      const ids = new Set(snapshot.hidden_reserved_reveal_candidates[`${activeReveal.actor}:${activeReveal.slot}`] ?? []);
      for (const cardId of snapshot.hidden_deck_card_ids_by_tier[activeReveal.tier] ?? []) {
        ids.add(cardId);
      }
      const player = displayBoard?.players.find((item) => item.seat === activeReveal.actor);
      const current = player?.reserved_public.find((item) => item.slot === activeReveal.slot);
      if (current && !current.is_placeholder) {
        const id = findCatalogCardId(current);
        if (id != null) {
          ids.add(id);
        }
      }
      return ids;
    }
    return new Set<number>();
  }, [activeReveal, snapshot, displayBoard, catalogCards]);
  const hasPendingFaceupReveal = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'faceup_card' || !snapshot) {
      return false;
    }
    return snapshot.pending_reveals.some(
      (reveal) => reveal.zone === 'faceup_card' && reveal.tier === activeReveal.tier && reveal.slot === activeReveal.slot,
    );
  }, [activeReveal, snapshot]);
  const liveActionsPageCount = useMemo(
    () => Math.max(1, Math.ceil(liveRows.length / ACTIONS_PAGE_SIZE)),
    [liveRows.length],
  );
  const pagedLiveRows = useMemo(() => {
    const start = (liveActionsPage - 1) * ACTIONS_PAGE_SIZE;
    return liveRows.slice(start, start + ACTIONS_PAGE_SIZE);
  }, [liveRows, liveActionsPage]);

  useEffect(() => {
    setLiveActionsPage(1);
  }, [jobStatus]);

  useEffect(() => {
    setLiveActionsPage((prev) => Math.min(prev, liveActionsPageCount));
  }, [liveActionsPageCount]);

  useEffect(() => {
    if (homeView !== 'LIVE') {
      clearLivePolling();
      return;
    }

    async function pollLiveSave(): Promise<void> {
      try {
        const status = await fetchJSON<LiveSaveStatusDTO>('/api/game/live-save/status');
        setLiveSaveStatus(status);
        if (!status.exists || !status.updated_at) {
          return;
        }
        if (status.updated_at === lastLiveSaveUpdatedAtRef.current) {
          return;
        }
        clearPolling();
        setJobStatus(null);
        const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/live-save/load', {
          method: 'POST',
          body: '{}',
        });
        lastLiveSaveUpdatedAtRef.current = status.updated_at;
        if (nextSnapshot.config?.checkpoint_id) {
          setCheckpointId(nextSnapshot.config.checkpoint_id);
        }
        await handleSnapshotUpdate(nextSnapshot);
      } catch (err) {
        setError((err as Error).message);
      }
    }

    void pollLiveSave();
    clearLivePolling();
    livePollRef.current = window.setInterval(() => {
      void pollLiveSave();
    }, LIVE_POLL_MS);
    return () => {
      clearLivePolling();
    };
  }, [homeView]);

  const isBoardView = (homeView === 'QUICK' || isSetupLikeView || homeView === 'LIVE') && snapshot;

  return (
    <main className="app-shell">
      <header>
        <h1>Splendor vs MCTS</h1>
        <div className="header-actions">
          <input
            ref={loadInputRef}
            type="file"
            accept="application/json"
            className="visually-hidden"
            onChange={(event) => void onLoadGameFile(event)}
          />
          <button type="button" onClick={() => void onSaveGame()} disabled={!snapshot}>
            Save Game
          </button>
          <button type="button" onClick={onLoadGameClick}>
            Load Game
          </button>
          {homeView !== 'HOME' && (
            <button type="button" onClick={() => setHomeView('HOME')}>
              Back
            </button>
          )}
        </div>
      </header>

      {homeView === 'HOME' && (
        <section className="panel">
          <h2>Start</h2>
          <div className="home-mode-grid">
            <button type="button" className="home-mode-card" onClick={() => setHomeView('QUICK')}>
              <strong>Quick Game</strong>
              <span>Standard engine vs human game with random setup.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={() => onOpenManualView('SETUP')}>
              <strong>Set Up</strong>
              <span>Start from placeholders and fill the opening board manually.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={() => onOpenManualView('ANALYSIS')}>
              <strong>Analysis</strong>
              <span>Manual board setup with continuous engine analysis on every turn.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={onOpenLiveView}>
              <strong>Live</strong>
              <span>Watch the latest Spendee bridge save and refresh analysis whenever it changes.</span>
            </button>
          </div>
        </section>
      )}

      {homeView === 'QUICK' && (
        <section className="panel">
        <h2>Setup</h2>
        <form onSubmit={(event) => void onStartGame(event)} className="grid-form">
          <label>
            Checkpoint
            <select
              value={checkpointId}
              onChange={(event) => setCheckpointId(event.target.value)}
              disabled={checkpoints.length === 0}
            >
              {checkpoints.length === 0 && <option value="">No checkpoints found</option>}
              {checkpoints.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>

          <button type="submit" disabled={!canStart}>
            Start Game
          </button>
        </form>
        </section>
      )}

      {isSetupLikeView && !snapshot && (
        <section className="panel">
          <h2>{homeView === 'ANALYSIS' ? 'Analysis' : 'Set Up'}</h2>
          <p>Choose a checkpoint, then fill the opening cards and nobles manually.</p>
          <form onSubmit={(event) => void onStartManualGame(event, homeView === 'ANALYSIS' ? 'ANALYSIS' : 'SETUP')} className="grid-form">
            <label>
              Checkpoint
              <select value={checkpointId} onChange={(event) => setCheckpointId(event.target.value)} disabled={checkpoints.length === 0}>
                {checkpoints.map((item) => (
                  <option key={`setup-${item.id}`} value={item.id}>
                  {item.name}
                </option>
              ))}
              </select>
            </label>
            <button type="submit" disabled={!canStart}>
              {homeView === 'ANALYSIS' ? 'Start Analysis' : 'Start Setup'}
            </button>
          </form>
        </section>
      )}

      {homeView === 'LIVE' && !snapshot && (
        <section className="panel">
          <h2>Live</h2>
          <p>Watching the latest bridge save and loading it automatically when it changes.</p>
          <p>{liveSaveStatus?.path ?? 'Waiting for live save path...'}</p>
          <p>{liveSaveStatus?.exists ? `Last update: ${liveSaveStatus.updated_at ?? 'unknown'}` : 'No live save file found yet.'}</p>
        </section>
      )}

      {isBoardView && (
        <section className="panel game-layout">
          <div className="board-column">
            <h2>Game Board</h2>
            {displayBoard ? (
              <GameBoard
                board={displayBoard}
                mctsTopAction={liveMctsTopAction}
                modelTopAction={liveModelTopAction}
                onCardClick={(tier, slot) => openReveal('faceup_card', tier, slot)}
                onNobleClick={(slot) => openReveal('noble', 0, slot)}
                onReservedCardClick={(seat, slot) => {
                  const player = displayBoard.players.find((item) => item.seat === seat);
                  const card = player?.reserved_public.find((item) => item.slot === slot);
                  const tier = card?.tier ?? snapshot.pending_reveals.find((item) => item.zone === 'reserved_card' && item.actor === seat && item.slot === slot)?.tier;
                  if (tier != null) {
                    openReveal('reserved_card', tier, slot, seat);
                  }
                }}
              />
            ) : (
              <div className="empty-note">Board data unavailable</div>
            )}
          </div>

          <aside className="side-column">
            <div className="engine-box">
              <h2>Analysis</h2>
              {homeView === 'LIVE' && <p>Watching {liveSaveStatus?.path ?? 'live save file'}.</p>}
              {uiStatus === 'WAITING_ENGINE' && <p className="spinner">Engine analyzing...</p>}
              {uiStatus === 'WAITING_REVEAL' && <p>Waiting for board update before the next move.</p>}
              {jobStatus?.error && <p className="error">Engine error: {jobStatus.error}</p>}
              {uiStatus !== 'WAITING_ENGINE' && snapshot.status === 'IN_PROGRESS' && (snapshot.config?.analysis_mode || snapshot.player_to_move !== snapshot.config?.player_seat) && (
                <div className="analysis-search-row">
                  <select
                    value={searchType}
                    onChange={(event) => setSearchType(event.target.value as SearchType)}
                    aria-label="Search type"
                  >
                    <option value="mcts">MCTS</option>
                    <option value="ismcts">ISMCTS</option>
                  </select>
                  <input
                    type="number"
                    min={1}
                    max={LIVE_SEARCH_MAX_SIMULATIONS}
                    value={searchSimulations}
                    onChange={(event) => setSearchSimulations(Number(event.target.value))}
                    aria-label="Search simulations"
                  />
                  <button onClick={() => void startEngineThink(searchSimulations)} disabled={searchSimulations < 1}>
                    {homeView === 'LIVE' ? 'Analyze Turn' : 'Run Search'}
                  </button>
                </div>
              )}
              {homeView === 'LIVE' && (
                <p>
                  Live mode keeps refining the current turn in every 400 sim chunks until the board updates or
                  {' '}500,000 total sims.
                  {jobStatus?.result?.total_simulations != null && ` Current total: ${jobStatus.result.total_simulations}.`}
                </p>
              )}
              <p className="analysis-root-value">
                Root value: <strong>{jobStatus?.result?.root_value != null ? jobStatus.result.root_value.toFixed(3) : '-'}</strong>
              </p>
              <div className="analysis-nav-row">
                <button type="button" onClick={() => void onUndo()} disabled={!snapshot.can_undo}>
                  Prev
                </button>
                <button type="button" onClick={() => void onRedo()} disabled={!snapshot.can_redo}>
                  Next
                </button>
              </div>
            </div>

            {pagedLiveRows.length > 0 && (
              <div className="actions-table-wrap">
                <h3>Actions</h3>
                <table className="actions-table">
                  <thead>
                    <tr>
                      <th>Action</th>
                      <th>MCTS</th>
                      <th>Q</th>
                      <th>Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedLiveRows.map(({ action, modelProb }) => {
                      const rowClickable = canMove;
                      return (
                      <tr
                        key={`live-${action.action_idx}`}
                        className={`action-row ${liveMctsTopAction?.action_idx === action.action_idx ? 'mcts-best' : ''} ${liveModelTopAction?.action_idx === action.action_idx ? 'model-best' : ''} ${rowClickable ? 'clickable' : ''}`}
                        onClick={() => {
                          if (canMove) {
                            void onPlayerMove(action.action_idx);
                          }
                        }}
                        onKeyDown={(event) => {
                          if (!rowClickable) return;
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            if (canMove) {
                              void onPlayerMove(action.action_idx);
                            }
                          }
                        }}
                        role={rowClickable ? 'button' : undefined}
                        tabIndex={rowClickable ? 0 : undefined}
                      >
                        <td>
                          <ActionLabel actionIdx={action.action_idx} board={snapshot.board_state} />
                        </td>
                        <td>
                          <span className="policy-value">{(action.policy_prob * 100).toFixed(2)}%</span>
                        </td>
                        <td>
                          <span className="policy-value">{action.q_value != null ? action.q_value.toFixed(3) : '-'}</span>
                        </td>
                        <td>
                          <span className="policy-value">{(modelProb * 100).toFixed(2)}%</span>
                        </td>
                      </tr>
                      );
                    })}
                  </tbody>
                </table>
                {liveRows.length > ACTIONS_PAGE_SIZE && (
                  <div className="actions-pagination">
                    <button type="button" onClick={() => setLiveActionsPage((prev) => Math.max(1, prev - 1))} disabled={liveActionsPage <= 1}>
                      Prev
                    </button>
                    <span>
                      Page {liveActionsPage} of {liveActionsPageCount}
                    </span>
                    <button
                      type="button"
                      onClick={() => setLiveActionsPage((prev) => Math.min(liveActionsPageCount, prev + 1))}
                      disabled={liveActionsPage >= liveActionsPageCount}
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            )}
          </aside>
        </section>
      )}

      {activeReveal && (
        <section className="reveal-modal-backdrop" onClick={() => setActiveRevealKey(null)}>
          <div className="reveal-modal" onClick={(event) => event.stopPropagation()}>
            <h3>
              {activeReveal.zone === 'noble'
                ? `Fill Noble Slot ${activeReveal.slot}`
                : activeReveal.zone === 'reserved_card'
                  ? `Reveal ${activeReveal.actor} Reserved Slot ${activeReveal.slot}`
                  : `Fill Tier ${activeReveal.tier} Slot ${activeReveal.slot}`}
            </h3>
            {activeReveal.reason !== 'initial_setup' && activeReveal.reason !== 'initial_noble_setup' && (
              <p>
                {describePendingReveal(activeReveal.reason)}
                {activeReveal.actor ? ` • ${activeReveal.actor}` : ''}
              </p>
            )}
            {activeReveal.zone === 'noble' ? (
              <>
                {isSetupLikeView && activeReveal.reason === 'initial_noble_setup' && (
                  <div className="setup-tier-slot-row noble-slot-row">
                    {activeBoardNobles.map((noble) => (
                      <div
                        key={`setup-noble-slot-${noble.slot}`}
                        className="setup-tier-slot"
                        onClick={() => noble.slot != null && openReveal('noble', 0, noble.slot)}
                      >
                        <NobleView noble={noble} />
                      </div>
                    ))}
                  </div>
                )}
                <div className="noble-catalog-grid">
                  {catalogNobles.map((noble) => {
                    const isAvailable = !setupUnavailableNobleIds.has(noble.id);
                    return (
                      <div
                        key={`noble-catalog-${noble.id}`}
                        className={`noble-catalog-option ${isAvailable ? 'available' : 'unavailable'}`}
                        onClick={() => {
                          if (!isAvailable) return;
                          void onRevealNobleWithId(activeReveal.slot, noble.id);
                        }}
                        title={nobleOptionLabel(noble)}
                      >
                        <NobleView noble={{ points: noble.points, requirements: noble.requirements, slot: activeReveal.slot }} />
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <>
                {isSetupLikeView && activeReveal.zone === 'faceup_card' && activeReveal.reason === 'initial_setup' && (
                  <div className="setup-tier-slot-row">
                    {activeTierBoardCards.map((card) => (
                      <div
                        key={`setup-tier-slot-${activeReveal.tier}-${card.slot}`}
                        className={`setup-tier-slot ${card.slot === activeReveal.slot ? 'selected' : ''}`}
                        onClick={() => card.slot != null && openReveal('faceup_card', activeReveal.tier, card.slot)}
                      >
                        <CardView card={card} />
                      </div>
                    ))}
                  </div>
                )}
                <div className="tier-catalog-grid">
                  {COLOR_ORDER.map((color) => (
                    <div key={`tier-catalog-row-${activeReveal.tier}-${color}`} className="tier-catalog-row">
                      <div className="tier-catalog-cards">
                        {cardsByTierAndColor[activeReveal.tier][color].map((card) => {
                          const isSetup = isSetupLikeView && activeReveal.zone === 'faceup_card' && activeReveal.reason === 'initial_setup';
                          const isRefillFaceup = activeReveal.zone === 'faceup_card' && (isSetup || hasPendingFaceupReveal);
                          const allowOccupiedSwap =
                            activeReveal.zone === 'faceup_card' &&
                            !isRefillFaceup &&
                            activeReveal.reason !== 'replacement_after_reserve';
                          const isOccupiedSwap = allowOccupiedSwap && occupiedBoardCardIds.has(card.id);
                          const isAvailable = isSetup ? !setupUnavailableCardIds.has(card.id) : (liveAvailableCardIds.has(card.id) || isOccupiedSwap);
                          const optionClass = isAvailable ? (isOccupiedSwap ? 'swap' : 'available') : 'unavailable';
                          return (
                            <div
                              key={`tier-catalog-card-${card.id}`}
                              className={`tier-catalog-option ${optionClass}`}
                              onClick={() => {
                                if (!isAvailable) return;
                                if (activeReveal.zone === 'reserved_card') {
                                  if (!activeReveal.actor) return;
                                  void onRevealReservedCardWithId(activeReveal.actor, activeReveal.tier, activeReveal.slot, card.id);
                                  return;
                                }
                                void onRevealCardWithId(activeReveal.tier, activeReveal.slot, card.id);
                              }}
                              title={cardOptionLabel(card)}
                            >
                              <CardView
                                card={{
                                  points: card.points,
                                  bonus_color: card.bonus_color,
                                  cost: card.cost,
                                  source: activeReveal.zone === 'reserved_card' ? 'reserved_public' : 'faceup',
                                  tier: card.tier,
                                  slot: activeReveal.slot,
                                }}
                              />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
            <button type="button" className="secondary-button" onClick={() => setActiveRevealKey(null)}>
              Close
            </button>
          </div>
        </section>
      )}

      {error && <section className="panel error">Error: {error}</section>}
    </main>
  );
}
