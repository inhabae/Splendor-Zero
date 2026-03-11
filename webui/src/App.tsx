import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  CatalogCardDTO,
  CatalogNobleDTO,
  BoardStateDTO,
  CheckpointDTO,
  EngineJobStatusDTO,
  EngineThinkResponse,
  GameSnapshotDTO,
  PlayerMoveResponse,
  RevealCardResponse,
  ReplayStepDTO,
  Seat,
  SelfPlayRunResponse,
  SelfPlaySessionDTO,
  SelfPlaySessionSummaryDTO,
} from './types';
import { GameBoard } from './components/board/GameBoard';
import { ActionLabel } from './components/ActionLabel';
import { CardView } from './components/board/CardView';
import { NobleView } from './components/board/NobleView';

type UiStatus = 'IDLE' | 'WAITING_ENGINE' | 'WAITING_PLAYER' | 'WAITING_REVEAL' | 'GAME_OVER';
type HomeView = 'HOME' | 'QUICK' | 'SETUP' | 'DEV';
const COLOR_ORDER: CatalogCardDTO['bonus_color'][] = ['white', 'blue', 'green', 'red', 'black'];

const POLL_MS = 400;
const ACTIONS_PAGE_SIZE = 10;

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

function winnerLabel(winner: number): string {
  if (winner === -2) return 'In progress';
  if (winner === -1) return 'Draw';
  return winner === 0 ? 'Winner: P1' : 'Winner: P2';
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
  const [playerSeat] = useState<Seat>('P0');
  const [seed] = useState('');
  const [homeView, setHomeView] = useState<HomeView>('HOME');
  const [revealSelections, setRevealSelections] = useState<Record<string, string>>({});
  const [activeRevealKey, setActiveRevealKey] = useState<string | null>(null);
  const [liveActionsPage, setLiveActionsPage] = useState(1);

  const [snapshot, setSnapshot] = useState<GameSnapshotDTO | null>(null);
  const [jobStatus, setJobStatus] = useState<EngineJobStatusDTO | null>(null);
  const [uiStatus, setUiStatus] = useState<UiStatus>('IDLE');

  const [selfplaySims, setSelfplaySims] = useState(400);
  const [selfplayGames, setSelfplayGames] = useState(1);
  const [selfplayMaxTurns, setSelfplayMaxTurns] = useState(100);
  const [selfplaySeed, setSelfplaySeed] = useState('');
  const [selfplaySessions, setSelfplaySessions] = useState<SelfPlaySessionDTO[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [selectedEpisodeIdx, setSelectedEpisodeIdx] = useState(0);
  const [selectedStepIdx, setSelectedStepIdx] = useState(0);
  const [sessionSummary, setSessionSummary] = useState<SelfPlaySessionSummaryDTO | null>(null);
  const [replayStep, setReplayStep] = useState<ReplayStepDTO | null>(null);
  const [selfplayRunInfo, setSelfplayRunInfo] = useState<SelfPlayRunResponse | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [selfplayLoading, setSelfplayLoading] = useState(false);

  const pollRef = useRef<number | null>(null);

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

  const availableEpisodes = useMemo(() => {
    if (!sessionSummary) {
      return [] as number[];
    }
    return Object.keys(sessionSummary.steps_per_episode)
      .map((v) => Number(v))
      .sort((a, b) => a - b);
  }, [sessionSummary]);

  const maxStepForEpisode = useMemo(() => {
    if (!sessionSummary) {
      return 0;
    }
    const count = Number(sessionSummary.steps_per_episode[String(selectedEpisodeIdx)] ?? 0);
    return Math.max(0, count - 1);
  }, [sessionSummary, selectedEpisodeIdx]);

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
        const sessions = await fetchJSON<SelfPlaySessionDTO[]>('/api/selfplay/sessions');
        setSelfplaySessions(sessions);
        if (sessions.length > 0) {
          setSelectedSessionId((prev) => (prev ? prev : sessions[0].session_id));
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
  }

  function deriveUiStatus(nextSnapshot: GameSnapshotDTO): UiStatus {
    if (nextSnapshot.status !== 'IN_PROGRESS') {
      return 'GAME_OVER';
    }
    if (nextSnapshot.pending_reveals.some((reveal) => isBlockingPendingReveal(reveal))) {
      return 'WAITING_REVEAL';
    }
    if (nextSnapshot.config?.analysis_mode) {
      return 'WAITING_PLAYER';
    }
    return nextSnapshot.player_to_move === nextSnapshot.config?.player_seat
      ? 'WAITING_PLAYER'
      : 'WAITING_ENGINE';
  }

  function shouldAutoSearch(nextSnapshot: GameSnapshotDTO): boolean {
    return (
      nextSnapshot.status === 'IN_PROGRESS' &&
      !nextSnapshot.pending_reveals.some((reveal) => isBlockingPendingReveal(reveal))
    );
  }

  function revealKey(zone: 'faceup_card' | 'reserved_card' | 'noble', tier: number, slot: number, seat?: Seat): string {
    return zone === 'reserved_card' ? `${zone}-${tier}-${slot}-${seat ?? 'P0'}` : `${zone}-${tier}-${slot}`;
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

  async function refreshSelfplaySessions(): Promise<void> {
    const sessions = await fetchJSON<SelfPlaySessionDTO[]>('/api/selfplay/sessions');
    setSelfplaySessions(sessions);
    if (sessions.length > 0) {
      setSelectedSessionId((prev) => (prev && sessions.some((s) => s.session_id === prev) ? prev : sessions[0].session_id));
    }
  }

  async function loadSessionSummary(sessionId: string): Promise<void> {
    const summary = await fetchJSON<SelfPlaySessionSummaryDTO>(`/api/selfplay/session/${sessionId}/summary`);
    setSessionSummary(summary);
  }

  async function loadReplayStep(sessionId: string, episodeIdx: number, stepIdx: number): Promise<void> {
    const step = await fetchJSON<ReplayStepDTO>(
      `/api/selfplay/session/${sessionId}/step?episode_idx=${episodeIdx}&step_idx=${stepIdx}`,
    );
    setReplayStep(step);
    setSelectedEpisodeIdx(episodeIdx);
    setSelectedStepIdx(stepIdx);
  }

  async function onRunSelfplay(event: FormEvent): Promise<void> {
    event.preventDefault();
    setError(null);
    setSelfplayLoading(true);
    setSelfplayRunInfo(null);
    try {
      if (!checkpointId) {
        throw new Error('Please choose a checkpoint for self-play');
      }
      const result = await fetchJSON<SelfPlayRunResponse>('/api/selfplay/run', {
        method: 'POST',
        body: JSON.stringify({
          checkpoint_id: checkpointId,
          num_simulations: Number(selfplaySims),
          games: Number(selfplayGames),
          max_turns: Number(selfplayMaxTurns),
          ...(selfplaySeed.trim().length > 0 ? { seed: Number(selfplaySeed) } : {}),
        }),
      });
      setSelfplayRunInfo(result);
      await refreshSelfplaySessions();
      setSelectedSessionId(result.session_id);
      await loadSessionSummary(result.session_id);
      await loadReplayStep(result.session_id, 0, 0);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSelfplayLoading(false);
    }
  }

  async function onLoadSession(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!selectedSessionId) {
      return;
    }
    setError(null);
    setSelfplayLoading(true);
    try {
      await loadSessionSummary(selectedSessionId);
      await loadReplayStep(selectedSessionId, selectedEpisodeIdx, selectedStepIdx);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSelfplayLoading(false);
    }
  }

  async function onJumpStep(episodeIdx: number, stepIdx: number): Promise<void> {
    if (!selectedSessionId) {
      return;
    }
    setError(null);
    try {
      await loadReplayStep(selectedSessionId, episodeIdx, stepIdx);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function startEngineThink(customNumSimulations?: number): Promise<void> {
    setError(null);
    const think = await fetchJSON<EngineThinkResponse>('/api/game/engine-think', {
      method: 'POST',
      body: JSON.stringify(
        customNumSimulations != null ? { num_simulations: Number(customNumSimulations) } : { num_simulations: Number(searchSimulations) }
      ),
    });
    setUiStatus('WAITING_ENGINE');
    clearPolling();

    pollRef.current = window.setInterval(() => {
      void pollEngineJob(think.job_id);
    }, POLL_MS);
  }

  async function pollEngineJob(nextJobId: string): Promise<void> {
    try {
      const status = await fetchJSON<EngineJobStatusDTO>(`/api/game/engine-job/${nextJobId}`);
      setJobStatus(status);
      if (status.status === 'DONE') {
        clearPolling();
        if (snapshot?.config?.analysis_mode) {
          setUiStatus(snapshot.status === 'IN_PROGRESS' ? 'WAITING_PLAYER' : 'GAME_OVER');
        } else {
          const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/engine-apply', {
            method: 'POST',
            body: JSON.stringify({ job_id: nextJobId }),
          });
          setSnapshot(nextSnapshot);
          const nextUiStatus = deriveUiStatus(nextSnapshot);
          setUiStatus(nextUiStatus);
          if (nextUiStatus === 'WAITING_ENGINE') {
            await startEngineThink();
          }
        }
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

  async function startGame(manualRevealMode: boolean): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);

    if (!checkpointId) {
      throw new Error('Please choose a checkpoint');
    }
      const payload = {
        checkpoint_id: checkpointId,
        num_simulations: Number(numSimulations),
        player_seat: playerSeat,
        manual_reveal_mode: manualRevealMode,
        analysis_mode: true,
        ...(seed.trim().length > 0 ? { seed: Number(seed) } : {}),
      };

    const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/new', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
      setSnapshot(nextSnapshot);
      const status = deriveUiStatus(nextSnapshot);
      setUiStatus(status);
  }

  async function onStartGame(event: FormEvent): Promise<void> {
    event.preventDefault();
    try {
      await startGame(false);
      setHomeView('QUICK');
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function onOpenSetup(): void {
    setError(null);
    clearPolling();
    setJobStatus(null);
    setSnapshot(null);
    setHomeView('SETUP');
  }

  async function onStartSetup(event: FormEvent): Promise<void> {
    event.preventDefault();
    try {
      await startGame(true);
      setHomeView('SETUP');
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
      setSnapshot(result.snapshot);
      const nextStatus = deriveUiStatus(result.snapshot);
      setUiStatus(nextStatus);
      if (result.snapshot.status !== 'IN_PROGRESS') {
        return;
      }
      if (shouldAutoSearch(result.snapshot)) {
        await startEngineThink(searchSimulations);
      }
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
      setSnapshot(nextSnapshot);
      setUiStatus(deriveUiStatus(nextSnapshot));
      if (shouldAutoSearch(nextSnapshot)) {
        await startEngineThink(searchSimulations);
      }
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
      setSnapshot(nextSnapshot);
      setUiStatus(deriveUiStatus(nextSnapshot));
      if (shouldAutoSearch(nextSnapshot)) {
        await startEngineThink(searchSimulations);
      }
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
      setSnapshot(result.snapshot);
      setRevealSelections((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      const status = deriveUiStatus(result.snapshot);
      setUiStatus(status);
      if (shouldAutoSearch(result.snapshot)) {
        await startEngineThink(searchSimulations);
      }
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
      setSnapshot(result.snapshot);
      setRevealSelections((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      const status = deriveUiStatus(result.snapshot);
      setUiStatus(status);
      if (shouldAutoSearch(result.snapshot)) {
        await startEngineThink(searchSimulations);
      }
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
      setSnapshot(result.snapshot);
      setRevealSelections((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      const status = deriveUiStatus(result.snapshot);
      setUiStatus(status);
      if (shouldAutoSearch(result.snapshot)) {
        await startEngineThink(searchSimulations);
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function openReveal(zone: 'faceup_card' | 'reserved_card' | 'noble', tier: number, slot: number, seat?: Seat): void {
    const key = revealKey(zone, tier, slot, seat);
    const hasPending = snapshot?.pending_reveals.some((reveal) => revealKey(reveal.zone, reveal.tier, reveal.slot, reveal.actor ?? undefined) === key) ?? false;
    const setupEditable =
      homeView === 'SETUP' &&
      snapshot?.pending_reveals.some((reveal) => reveal.reason === 'initial_setup' || reveal.reason === 'initial_noble_setup') &&
      (zone === 'faceup_card' || zone === 'noble');
    const boardCardEditable = zone === 'faceup_card' && snapshot?.config?.manual_reveal_mode;
    if (!hasPending && !setupEditable && !boardCardEditable) {
      return;
    }
    if ((setupEditable || boardCardEditable) && snapshot?.board_state) {
      if (zone === 'faceup_card') {
        const row = snapshot.board_state.tiers.find((item) => item.tier === tier);
        const current = row?.cards.find((card) => card.slot === slot);
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

  const canStart = Boolean(selectedCheckpoint) && numSimulations > 0 && numSimulations <= 10000;
  const canMove =
    uiStatus === 'WAITING_PLAYER' &&
    snapshot?.status === 'IN_PROGRESS' &&
    !(snapshot.pending_reveals?.some((reveal) => isBlockingPendingReveal(reveal)) ?? false);
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
      homeView === 'SETUP' &&
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
    return null;
  }, [snapshot, activeRevealKey, homeView]);
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
    if (!mcts?.length) return [];
    const model = jobStatus?.result?.model_action_details;
    return mcts
      .map((action, idx) => ({ action, modelProb: model?.[idx]?.policy_prob ?? 0 }))
      .filter((row) => !row.action.masked)
      .sort((a, b) => {
        if (b.action.policy_prob !== a.action.policy_prob) return b.action.policy_prob - a.action.policy_prob;
        return b.modelProb - a.modelProb;
      });
  }, [jobStatus]);
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
    if (!activeReveal || activeReveal.zone !== 'faceup_card' || homeView !== 'SETUP' || activeReveal.reason !== 'initial_setup' || !displayBoard) {
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
  }, [activeReveal, homeView, displayBoard, catalogCards]);
  const setupUnavailableNobleIds = useMemo(() => {
    if (!activeReveal || activeReveal.zone !== 'noble' || homeView !== 'SETUP' || activeReveal.reason !== 'initial_noble_setup' || !displayBoard) {
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
  }, [activeReveal, homeView, displayBoard, catalogNobles]);
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
      return ids;
    }
    if (activeReveal.zone === 'reserved_card' && activeReveal.actor) {
      return new Set(snapshot.hidden_reserved_reveal_candidates[`${activeReveal.actor}:${activeReveal.slot}`] ?? []);
    }
    return new Set<number>();
  }, [activeReveal, snapshot, displayBoard, catalogCards]);
  const replayModelBestActionIdx = useMemo(() => {
    if (!replayStep?.model_action_details) {
      return null;
    }
    let bestIdx: number | null = null;
    let bestProb = -1;
    for (const action of replayStep.model_action_details) {
      if (action.masked) {
        continue;
      }
      if (action.policy_prob > bestProb) {
        bestProb = action.policy_prob;
        bestIdx = action.action_idx;
      }
    }
    return bestIdx;
  }, [replayStep]);
  const replayMctsBestActionIdx = useMemo(() => {
    if (!replayStep?.action_details) {
      return null;
    }
    let bestIdx: number | null = null;
    let bestProb = -1;
    for (const action of replayStep.action_details) {
      if (action.masked) {
        continue;
      }
      if (action.policy_prob > bestProb) {
        bestProb = action.policy_prob;
        bestIdx = action.action_idx;
      }
    }
    return bestIdx;
  }, [replayStep]);
  const replayMctsTopAction = useMemo(() => {
    if (!replayStep || replayMctsBestActionIdx == null) {
      return null;
    }
    return replayStep.action_details.find((a) => a.action_idx === replayMctsBestActionIdx) ?? null;
  }, [replayStep, replayMctsBestActionIdx]);
  const replayModelTopAction = useMemo(() => {
    if (!replayStep || replayModelBestActionIdx == null) {
      return null;
    }
    return replayStep.action_details.find((a) => a.action_idx === replayModelBestActionIdx) ?? null;
  }, [replayStep, replayModelBestActionIdx]);
  const replayRows = useMemo(() => {
    if (!replayStep) {
      return [];
    }
    return replayStep.action_details
      .map((action, idx) => ({
        action,
        modelProb: replayStep.model_action_details?.[idx]?.policy_prob ?? 0,
      }))
      .filter((row) => !row.action.masked)
      .sort((a, b) => {
        if (b.modelProb !== a.modelProb) {
          return b.modelProb - a.modelProb;
        }
        return b.action.policy_prob - a.action.policy_prob;
      });
  }, [replayStep]);
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

  return (
    <main className="app-shell">
      <header>
        <h1>Splendor vs MCTS</h1>
        {homeView !== 'HOME' && (
          <button type="button" onClick={() => setHomeView('HOME')}>
            Back
          </button>
        )}
      </header>

      {homeView === 'HOME' && (
        <section className="panel">
          <h2>Start</h2>
          <div className="home-mode-grid">
            <button type="button" className="home-mode-card" onClick={() => setHomeView('QUICK')}>
              <strong>Quick Game</strong>
              <span>Standard engine vs human game with random setup.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={() => onOpenSetup()}>
              <strong>Set Up</strong>
              <span>Start from placeholders and fill the opening board manually.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={() => setHomeView('DEV')}>
              <strong>Dev</strong>
              <span>Replay and self-play tools.</span>
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

      {homeView === 'SETUP' && !snapshot && (
        <section className="panel">
          <h2>Set Up</h2>
          <p>Choose a checkpoint, then fill the opening cards and nobles manually.</p>
          <form onSubmit={(event) => void onStartSetup(event)} className="grid-form">
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
              Start Setup
            </button>
          </form>
        </section>
      )}

      {(homeView === 'QUICK' || homeView === 'SETUP') && snapshot && (
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
              {uiStatus === 'WAITING_ENGINE' && <p className="spinner">Engine analyzing...</p>}
              {uiStatus === 'WAITING_REVEAL' && <p>Waiting for board update before the next move.</p>}
              {jobStatus?.error && <p className="error">Engine error: {jobStatus.error}</p>}
              {uiStatus !== 'WAITING_ENGINE' && snapshot.status === 'IN_PROGRESS' && (
                <div className="analysis-search-row">
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={searchSimulations}
                    onChange={(event) => setSearchSimulations(Number(event.target.value))}
                    aria-label="Search simulations"
                  />
                  <button onClick={() => void startEngineThink(searchSimulations)} disabled={searchSimulations < 1 || searchSimulations > 10000}>
                    Run Search
                  </button>
                </div>
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
                      <th>Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedLiveRows.map(({ action, modelProb }) => (
                      <tr
                        key={`live-${action.action_idx}`}
                        className={`action-row ${liveMctsTopAction?.action_idx === action.action_idx ? 'mcts-best' : ''} ${liveModelTopAction?.action_idx === action.action_idx ? 'model-best' : ''} ${canMove ? 'clickable' : ''}`}
                        onClick={() => {
                          if (!canMove) return;
                          void onPlayerMove(action.action_idx);
                        }}
                        onKeyDown={(event) => {
                          if (!canMove) return;
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            void onPlayerMove(action.action_idx);
                          }
                        }}
                        role={canMove ? 'button' : undefined}
                        tabIndex={canMove ? 0 : undefined}
                      >
                        <td>
                          <ActionLabel actionIdx={action.action_idx} board={snapshot.board_state} />
                        </td>
                        <td>
                          <span className="policy-value">{(action.policy_prob * 100).toFixed(2)}%</span>
                        </td>
                        <td>
                          <span className="policy-value">{(modelProb * 100).toFixed(2)}%</span>
                        </td>
                      </tr>
                    ))}
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

      {homeView === 'DEV' && (
      <section className="panel">
        <h2>Self-Play Replay</h2>
        <form onSubmit={(event) => void onRunSelfplay(event)} className="grid-form">
          <label>
            Checkpoint
            <select value={checkpointId} onChange={(event) => setCheckpointId(event.target.value)}>
              {checkpoints.map((item) => (
                <option key={`sp-${item.id}`} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Sims
            <input type="number" min={1} max={10000} value={selfplaySims} onChange={(e) => setSelfplaySims(Number(e.target.value))} />
          </label>
          <label>
            Games
            <input type="number" min={1} max={500} value={selfplayGames} onChange={(e) => setSelfplayGames(Number(e.target.value))} />
          </label>
          <label>
            Max turns
            <input type="number" min={1} max={400} value={selfplayMaxTurns} onChange={(e) => setSelfplayMaxTurns(Number(e.target.value))} />
          </label>
          <label>
            Seed (optional)
            <input value={selfplaySeed} onChange={(event) => setSelfplaySeed(event.target.value)} placeholder="Random if blank" />
          </label>
          <button type="submit" disabled={selfplayLoading || !checkpointId}>Run Self-Play</button>
        </form>

        {selfplayRunInfo && (
          <p>
            Latest run: session <strong>{selfplayRunInfo.session_id}</strong> | games={selfplayRunInfo.games} | steps={selfplayRunInfo.steps}
          </p>
        )}

        <form onSubmit={(event) => void onLoadSession(event)} className="grid-form">
          <label>
            Session
            <select value={selectedSessionId} onChange={(event) => setSelectedSessionId(event.target.value)}>
              {selfplaySessions.length === 0 && <option value="">No sessions</option>}
              {selfplaySessions.map((session) => (
                <option key={session.session_id} value={session.session_id}>
                  {session.display_name} ({session.steps} steps)
                </option>
              ))}
            </select>
          </label>
          <label>
            Episode
            <input
              type="number"
              min={0}
              max={availableEpisodes.length > 0 ? Math.max(...availableEpisodes) : 0}
              value={selectedEpisodeIdx}
              onChange={(event) => setSelectedEpisodeIdx(Number(event.target.value))}
            />
          </label>
          <label>
            Step
            <input
              type="number"
              min={0}
              max={maxStepForEpisode}
              value={selectedStepIdx}
              onChange={(event) => setSelectedStepIdx(Number(event.target.value))}
            />
          </label>
          <button type="submit" disabled={!selectedSessionId || selfplayLoading}>Load Step</button>
          <button
            type="button"
            disabled={!selectedSessionId || selfplayLoading || selectedStepIdx <= 0}
            onClick={() => void onJumpStep(selectedEpisodeIdx, Math.max(0, selectedStepIdx - 1))}
          >
            Prev Step
          </button>
          <button
            type="button"
            disabled={!selectedSessionId || selfplayLoading || selectedStepIdx >= maxStepForEpisode}
            onClick={() => void onJumpStep(selectedEpisodeIdx, Math.min(maxStepForEpisode, selectedStepIdx + 1))}
          >
            Next Step
          </button>
          <button type="button" disabled={selfplayLoading} onClick={() => void refreshSelfplaySessions()}>
            Refresh Sessions
          </button>
        </form>

        {sessionSummary && (
          <p>
            Session summary: games={sessionSummary.games}, total steps={sessionSummary.steps}, created={sessionSummary.created_at}
          </p>
        )}
      </section>
      )}

      {homeView === 'DEV' && replayStep && (
        <section className="panel game-layout">
          <div className="board-column">
            <h2>Replay Board</h2>
            <GameBoard board={replayStep.board_state} mctsTopAction={replayMctsTopAction} modelTopAction={replayModelTopAction} />
          </div>
          <aside className="side-column">
            <h2>Replay Inspector</h2>
            <p>
              Session: <strong>{replayStep.session_id}</strong> | Episode: <strong>{replayStep.episode_idx}</strong> | Step: <strong>{replayStep.step_idx}</strong>
            </p>
            <p>
              Value target: <strong>{replayStep.value_target.toFixed(3)}</strong> | Model value:{' '}
              <strong>{replayStep.model_value == null ? 'N/A' : replayStep.model_value.toFixed(3)}</strong>
            </p>
            <p>
              Winner: <strong>{winnerLabel(replayStep.winner)}</strong> | Cutoff: <strong>{String(replayStep.reached_cutoff)}</strong>
            </p>
            <div className="actions-table-wrap">
              <h3>Legal Actions</h3>
              <table className="actions-table">
                <thead>
                  <tr>
                    <th>Idx</th>
                    <th>Action</th>
                    <th>MCTS</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {replayRows.map(({ action, modelProb }) => {
                    const isModelBest = replayModelBestActionIdx != null && replayModelBestActionIdx === action.action_idx;
                    const isMctsBest = replayMctsBestActionIdx != null && replayMctsBestActionIdx === action.action_idx;
                    return (
                    <tr
                      key={`viz-${action.action_idx}`}
                      className={`action-row ${action.is_selected ? 'selected' : ''} ${isMctsBest ? 'mcts-best' : ''} ${isModelBest ? 'model-best' : ''}`}
                    >
                      <td className="action-idx-cell">{action.action_idx}</td>
                      <td>
                        <ActionLabel
                          actionIdx={action.action_idx}
                          board={replayStep.board_state}
                          showPlayed={action.is_selected}
                        />
                      </td>
                      <td>
                        <div className="policy-cell">
                          <span className="policy-value">{(action.policy_prob * 100).toFixed(2)}%</span>
                          <div className="policy-bar">
                            <span style={{ width: `${Math.max(0, Math.min(100, action.policy_prob * 100))}%` }} />
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="policy-cell">
                          <span className="policy-value">{(modelProb * 100).toFixed(2)}%</span>
                          <div className="policy-bar">
                            <span style={{ width: `${Math.max(0, Math.min(100, modelProb * 100))}%` }} />
                          </div>
                        </div>
                      </td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
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
                {homeView === 'SETUP' && activeReveal.reason === 'initial_noble_setup' && (
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
                {homeView === 'SETUP' && activeReveal.zone === 'faceup_card' && activeReveal.reason === 'initial_setup' && (
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
                          const isSetup = homeView === 'SETUP' && activeReveal.zone === 'faceup_card' && activeReveal.reason === 'initial_setup';
                          const isAvailable = isSetup ? !setupUnavailableCardIds.has(card.id) : liveAvailableCardIds.has(card.id);
                          return (
                            <div
                              key={`tier-catalog-card-${card.id}`}
                              className={`tier-catalog-option ${isAvailable ? 'available' : 'unavailable'}`}
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
