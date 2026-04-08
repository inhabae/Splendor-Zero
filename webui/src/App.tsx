import { ChangeEvent, FormEvent, Fragment, useEffect, useMemo, useRef, useState } from 'react';
import {
  CatalogCardDTO,
  CatalogNobleDTO,
  BoardStateDTO,
  CheckpointDTO,
  EngineJobStatusDTO,
  EngineThinkResponse,
  GameSnapshotDTO,
  LiveSaveStatusDTO,
  MoveLogEntryDTO,
  PlayerMoveResponse,
  RevealCardResponse,
  SavedGameDTO,
  SavedGameWithDeepAnalysisDTO,
  SearchType,
  Seat,
} from './types';
import { GameBoard } from './components/board/GameBoard';
import { ActionLabel, actionTextLabel } from './components/ActionLabel';
import { CardView } from './components/board/CardView';
import { NobleView } from './components/board/NobleView';

type UiStatus = 'IDLE' | 'WAITING_ENGINE' | 'WAITING_PLAYER' | 'WAITING_REVEAL' | 'GAME_OVER';
type HomeView = 'HOME' | 'QUICK' | 'ANALYSIS' | 'LIVE';
type AnalysisPanelTab = 'ANALYSIS' | 'MOVES';
const COLOR_ORDER: CatalogCardDTO['bonus_color'][] = ['white', 'blue', 'green', 'red', 'black'];

const POLL_MS = 400;
const LIVE_POLL_MS = 1000;
const LIVE_SEARCH_MAX_SIMULATIONS = 500_000;
const DEFAULT_DEEP_ANALYSIS_SIMULATIONS = 50_000;

function isContinuationAction(actionIdx: number): boolean {
  return actionIdx >= 61 && actionIdx <= 68;
}

interface MoveLogRow {
  moveNumber: number;
  moveNumberLabel: string;
  p0?: MoveLogDisplayEntry;
  p1?: MoveLogDisplayEntry;
}

type MoveLogDisplayEntry = MoveLogEntryDTO & {
  notation: string;
  turnLabel: string;
  fullMoveNumber: number;
  continuationIndex: number;
};

interface HighlightedMove {
  actor: Seat;
  resultTurnIndex: number;
  resultSnapshotIndex: number;
}

interface HighlightedVariation {
  branchId: number;
  moveIndex: number;
}

interface VariationMove {
  kind: 'move' | 'edit_faceup' | 'edit_reserved' | 'edit_noble';
  actor: Seat;
  actionIdx: number;
  label: string;
  fullMoveNumber: number;
  targetSnapshotIndex: number;
  targetTurnIndex: number;
  jumpBySnapshot: boolean;
  tier?: number;
  slot?: number;
  seat?: Seat;
  cardId?: number;
  nobleId?: number;
}

interface VariationBranch {
  id: number;
  anchorSnapshotIndex: number;
  moves: VariationMove[];
}

type DeepAnalysisCategory = 'Best' | 'Good' | 'Mistake' | 'Blunder' | 'Unknown';

interface DeepAnalysisEntry {
  category: DeepAnalysisCategory;
  playedActionIdx: number;
  bestActionIdx: number | null;
  playedQ: number | null;
  bestQ: number | null;
  qLoss: number | null;
}

type DeepAnalysisSearchResult = NonNullable<EngineJobStatusDTO['result']>;

function moveAnalysisKey(move: Pick<MoveLogEntryDTO, 'result_snapshot_index' | 'turn_index' | 'actor' | 'action_idx'>): string {
  return `${move.result_snapshot_index}:${move.turn_index}:${move.actor}:${move.action_idx}`;
}

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

function continuationSuffix(index: number): string {
  if (index <= 0) {
    return '';
  }
  let out = '';
  let value = index;
  while (value > 0) {
    const rem = (value - 1) % 26;
    out = String.fromCharCode(97 + rem) + out;
    value = Math.floor((value - 1) / 26);
  }
  return out;
}

function formatEvalBar(value: number | null | undefined): string {
  return value != null && Number.isFinite(value) ? Math.abs(value).toFixed(2) : '--';
}

function topMoveEvalClass(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) {
    return 'neutral';
  }
  return value > 0 ? 'white-side' : 'black-side';
}

function formatTopMoveEval(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return '--';
  }
  const magnitude = Math.abs(value).toFixed(2);
  return value > 0 ? `+${magnitude}` : `-${magnitude}`;
}

function p1WinningEval(value: number | null | undefined, playerToMove: Seat | null | undefined): number | null {
  if (value == null || !Number.isFinite(value) || playerToMove == null) {
    return null;
  }
  return playerToMove === 'P1' ? value : -value;
}

function p0WinningEval(value: number | null | undefined, playerToMove: Seat | null | undefined): number | null {
  const p1Value = p1WinningEval(value, playerToMove);
  return p1Value == null ? null : -p1Value;
}

function parsePlayerNamesFromFilename(filename: string): Record<Seat, string> | null {
  const trimmed = filename
    .replace(/\.[^.]+$/, '')
    .replace(/^Game\s+\S+\s*/i, '')
    .trim();
  const parts = trimmed.split(/\s+vs\s+/i);
  if (parts.length !== 2) {
    return null;
  }
  const stripElo = (value: string): string => value.replace(/\s*\(\d+\)\s*$/i, '').trim();
  const p0Name = stripElo(parts[0] ?? '');
  const p1Name = stripElo(parts[1] ?? '');
  if (!p0Name || !p1Name) {
    return null;
  }
  return {
    P0: p0Name,
    P1: p1Name,
  };
}

export function App() {
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [catalogCards, setCatalogCards] = useState<CatalogCardDTO[]>([]);
  const [catalogNobles, setCatalogNobles] = useState<CatalogNobleDTO[]>([]);
  const [checkpointId, setCheckpointId] = useState('');
  const [numSimulations] = useState(400);
  const [searchSimulations, setSearchSimulations] = useState(400);
  const [deepAnalysisSimulations, setDeepAnalysisSimulations] = useState(DEFAULT_DEEP_ANALYSIS_SIMULATIONS);
  const [searchType, setSearchType] = useState<SearchType>('mcts');
  const [playerSeat] = useState<Seat>('P0');
  const [seed] = useState('');
  const [homeView, setHomeView] = useState<HomeView>('HOME');
  const [revealSelections, setRevealSelections] = useState<Record<string, string>>({});
  const [activeRevealKey, setActiveRevealKey] = useState<string | null>(null);
  const [showBoardAnalysis, setShowBoardAnalysis] = useState(true);
  const [showAnalysisSettings, setShowAnalysisSettings] = useState(false);
  const [hideAllExceptBoard, setHideAllExceptBoard] = useState(false);

  const [snapshot, setSnapshot] = useState<GameSnapshotDTO | null>(null);
  const [loadedMoveLog, setLoadedMoveLog] = useState<MoveLogEntryDTO[] | null>(null);
  const [loadedPlayerNames, setLoadedPlayerNames] = useState<Record<Seat, string> | null>(null);
  const [variationBranches, setVariationBranches] = useState<VariationBranch[]>([]);
  const [jobStatus, setJobStatus] = useState<EngineJobStatusDTO | null>(null);
  const [uiStatus, setUiStatus] = useState<UiStatus>('IDLE');
  const [liveSaveStatus, setLiveSaveStatus] = useState<LiveSaveStatusDTO | null>(null);
const [displayedP0EvalValue, setDisplayedP0EvalValue] = useState<number | null>(null);
  const [analysisPanelTab, setAnalysisPanelTab] = useState<AnalysisPanelTab>('ANALYSIS');
  const [deepAnalysisBySnapshot, setDeepAnalysisBySnapshot] = useState<Record<string, DeepAnalysisEntry>>({});
  const [deepAnalysisSearchBySnapshot, setDeepAnalysisSearchBySnapshot] = useState<Record<string, DeepAnalysisSearchResult>>({});
  const [isLoadedPostAnalysisGame, setIsLoadedPostAnalysisGame] = useState(false);
  const [isDeepAnalysisRunning, setIsDeepAnalysisRunning] = useState(false);
  const [deepAnalysisProgress, setDeepAnalysisProgress] = useState<{ done: number; total: number } | null>(null);

  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);
  const livePollRef = useRef<number | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const activeVariationBranchIdRef = useRef<number | null>(null);
  const variationBranchIdCounterRef = useRef<number>(1);
  const loadedHistoricalMainlineLengthRef = useRef<number>(0);
  const loadedHistoricalMainlineTailSnapshotRef = useRef<number>(0);
  const loadedSavedGameRef = useRef<SavedGameWithDeepAnalysisDTO | null>(null);
  const loadInputRef = useRef<HTMLInputElement | null>(null);
  const analysisSettingsRef = useRef<HTMLDivElement | null>(null);
  const moveLogGridRef = useRef<HTMLDivElement | null>(null);
  const evalAnimationFrameRef = useRef<number | null>(null);
const displayedP0EvalRef = useRef<number | null>(null);
  const isSetupLikeView = homeView === 'ANALYSIS';
  const lastLiveSaveUpdatedAtRef = useRef<string | null>(null);
  const lastAutoAnalyzeKeyRef = useRef<string | null>(null);
  const lastSnapshotSearchKeyRef = useRef<string | null>(null);
  const autoAnalyzeOnNavigation = showBoardAnalysis;

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
  const moveLogEntries = useMemo<MoveLogEntryDTO[]>(() => {
    if (loadedMoveLog && loadedMoveLog.length > 0) {
      return loadedMoveLog;
    }
    return snapshot?.move_log ?? [];
  }, [loadedMoveLog, snapshot?.move_log]);
  const moveLogDisplayEntries = useMemo<MoveLogDisplayEntry[]>(() => {
    let fullMoveNumber = 0;
    let continuationIndex = 0;

    return moveLogEntries.map((move) => {
      const isContinuation = isContinuationAction(move.action_idx);
      const displayActor = move.actor;

      if (isContinuation) {
        continuationIndex += 1;
      } else {
        continuationIndex = 0;
        if (displayActor === 'P0') {
          fullMoveNumber += 1;
        } else if (fullMoveNumber <= 0) {
          fullMoveNumber = 1;
        }
      }

      const suffix = continuationSuffix(continuationIndex);
      const base = `${fullMoveNumber}${suffix}`;
      const notation = displayActor === 'P0' ? `${base}.` : `${base}...`;

      return {
        ...move,
        actor: displayActor,
        notation,
        turnLabel: base,
        fullMoveNumber,
        continuationIndex,
      };
    });
  }, [moveLogEntries]);
  const moveLogRows = useMemo<MoveLogRow[]>(() => {
    const rows: MoveLogRow[] = [];
    const rowByLabel = new Map<string, number>();
    for (const move of moveLogDisplayEntries) {
      const moveNumberLabel = move.turnLabel;
      const existingIdx = rowByLabel.get(move.turnLabel);
      if (existingIdx != null) {
        const existing = rows[existingIdx];
        if (move.actor === 'P0') {
          if (existing.p0 == null) {
            existing.p0 = move;
          } else {
            rows.push({ moveNumber: move.fullMoveNumber, moveNumberLabel, p0: move });
          }
        } else if (existing.p1 == null) {
          existing.p1 = move;
        } else {
          rows.push({ moveNumber: move.fullMoveNumber, moveNumberLabel, p1: move });
        }
      } else {
        rows.push(
          move.actor === 'P0'
            ? { moveNumber: move.fullMoveNumber, moveNumberLabel, p0: move }
            : { moveNumber: move.fullMoveNumber, moveNumberLabel, p1: move }
        );
        rowByLabel.set(move.turnLabel, rows.length - 1);
      }
    }
    return rows;
  }, [moveLogDisplayEntries]);
  const mainlineMoveNumberBySnapshot = useMemo<Map<number, number>>(() => {
    const out = new Map<number, number>();
    for (const row of moveLogRows) {
      if (row.p0?.result_snapshot_index != null) {
        out.set(row.p0.result_snapshot_index, row.moveNumber);
      }
      if (row.p1?.result_snapshot_index != null) {
        out.set(row.p1.result_snapshot_index, row.moveNumber);
      }
    }
    return out;
  }, [moveLogRows]);
  const variationBranchByAnchor = useMemo<Map<number, VariationBranch[]>>(() => {
    const out = new Map<number, VariationBranch[]>();
    for (const branch of variationBranches) {
      const existing = out.get(branch.anchorSnapshotIndex);
      if (existing) {
        existing.push(branch);
      } else {
        out.set(branch.anchorSnapshotIndex, [branch]);
      }
    }
    return out;
  }, [variationBranches]);
  const currentSnapshotIndex = useMemo<number>(() => {
    if (!snapshot) {
      return 0;
    }
    if (snapshot.current_snapshot_index != null) {
      return Number(snapshot.current_snapshot_index);
    }
    if (moveLogEntries.length === 0) {
      return 0;
    }
    let bestSnapshotIndex = 0;
    for (const move of moveLogEntries) {
      if (move.result_turn_index > snapshot.turn_index) {
        continue;
      }
      if (move.result_snapshot_index > bestSnapshotIndex) {
        bestSnapshotIndex = move.result_snapshot_index;
      }
    }
    return bestSnapshotIndex;
  }, [snapshot, moveLogEntries]);
  const mainlineMoveSnapshotIndices = useMemo<number[]>(() => {
    const indices = moveLogEntries
      .map((move) => move.result_snapshot_index)
      .filter((value) => Number.isFinite(value) && value > 0);
    const uniqueInOrder = Array.from(new Set(indices));
    return [0, ...uniqueInOrder];
  }, [moveLogEntries]);
  const mainlineMoveTurnIndices = useMemo<number[]>(() => {
    const indices = moveLogEntries
      .map((move) => move.result_turn_index)
      .filter((value) => Number.isFinite(value) && value > 0);
    const uniqueInOrder = Array.from(new Set(indices));
    return [0, ...uniqueInOrder];
  }, [moveLogEntries]);
  const isLoadedMainlineExtensionState = useMemo<boolean>(() => {
    return Boolean(
      snapshot &&
      loadedHistoricalMainlineLengthRef.current > 0 &&
      snapshot.current_snapshot_index == null &&
      currentSnapshotIndex > loadedHistoricalMainlineTailSnapshotRef.current
    );
  }, [snapshot, currentSnapshotIndex]);
  const useTurnNavigationForVisibleMainline = Boolean(snapshot?.current_snapshot_index == null && !isLoadedMainlineExtensionState);
  const visibleMainlineTargets = useMemo<number[]>(() => {
    return useTurnNavigationForVisibleMainline ? mainlineMoveTurnIndices : mainlineMoveSnapshotIndices;
  }, [useTurnNavigationForVisibleMainline, mainlineMoveTurnIndices, mainlineMoveSnapshotIndices]);
  const visibleMainlinePosition = useMemo<number>(() => {
    if (!snapshot) {
      return 0;
    }
    return useTurnNavigationForVisibleMainline ? snapshot.turn_index : currentSnapshotIndex;
  }, [snapshot, useTurnNavigationForVisibleMainline, currentSnapshotIndex]);
  const canStepVisibleMainlineBackward = useMemo<boolean>(() => {
    return visibleMainlineTargets.some((target) => target < visibleMainlinePosition);
  }, [visibleMainlineTargets, visibleMainlinePosition]);
  const canStepVisibleMainlineForward = useMemo<boolean>(() => {
    return visibleMainlineTargets.some((target) => target > visibleMainlinePosition);
  }, [visibleMainlineTargets, visibleMainlinePosition]);
  const highlightedMove = useMemo<HighlightedMove | null>(() => {
    if (moveLogDisplayEntries.length === 0 || currentSnapshotIndex <= 0) {
      return null;
    }
    let best: MoveLogDisplayEntry | null = null;
    for (const move of moveLogDisplayEntries) {
      if (move.result_snapshot_index > currentSnapshotIndex) {
        continue;
      }
      if (
        !best
        || move.result_snapshot_index > best.result_snapshot_index
      ) {
        best = move;
      }
    }
    if (best) {
      return {
        actor: best.actor,
        resultTurnIndex: best.result_turn_index,
        resultSnapshotIndex: best.result_snapshot_index,
      };
    }
    return null;
  }, [moveLogDisplayEntries, currentSnapshotIndex]);
  const highlightedVariation = useMemo<HighlightedVariation | null>(() => {
    if (!snapshot) {
      return null;
    }
    for (const branch of variationBranches) {
      for (let idx = 0; idx < branch.moves.length; idx += 1) {
        const move = branch.moves[idx];
        if (move.jumpBySnapshot && move.targetSnapshotIndex === currentSnapshotIndex) {
          return { branchId: branch.id, moveIndex: idx };
        }
      }
    }
    // Branched positions do not always map to a mainline snapshot index.
    if (snapshot.current_snapshot_index == null) {
      const activeBranchId = activeVariationBranchIdRef.current;
      if (activeBranchId != null) {
        const activeBranch = variationBranches.find((branch) => branch.id === activeBranchId) ?? null;
        if (activeBranch) {
          for (let idx = activeBranch.moves.length - 1; idx >= 0; idx -= 1) {
            if (activeBranch.moves[idx].targetTurnIndex === snapshot.turn_index) {
              return { branchId: activeBranch.id, moveIndex: idx };
            }
          }
        }
      }
      for (const branch of variationBranches) {
        for (let idx = branch.moves.length - 1; idx >= 0; idx -= 1) {
          const move = branch.moves[idx];
          if (move.targetTurnIndex === snapshot.turn_index) {
            return { branchId: branch.id, moveIndex: idx };
          }
        }
      }
    }
    return null;
  }, [variationBranches, currentSnapshotIndex, snapshot]);
  function isHighlightedMainlineMove(move: MoveLogDisplayEntry | null | undefined): boolean {
    if (!move || highlightedVariation != null || highlightedMove == null) {
      return false;
    }
    if (snapshot?.current_snapshot_index != null) {
      return (
        move.actor === highlightedMove.actor
        && move.result_snapshot_index === highlightedMove.resultSnapshotIndex
      );
    }
    return (
      move.actor === highlightedMove.actor
      && move.result_turn_index === highlightedMove.resultTurnIndex
    );
  }

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
      if (evalAnimationFrameRef.current !== null) {
        window.cancelAnimationFrame(evalAnimationFrameRef.current);
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

  function snapshotSearchKey(nextSnapshot: GameSnapshotDTO): string {
    return JSON.stringify({
      checkpointId: nextSnapshot.config?.checkpoint_id ?? null,
      status: nextSnapshot.status,
      winner: nextSnapshot.winner,
      turnIndex: nextSnapshot.turn_index,
      playerToMove: nextSnapshot.player_to_move,
      boardState: nextSnapshot.board_state ?? null,
      legalActions: nextSnapshot.legal_actions,
      pendingReveals: nextSnapshot.pending_reveals.map((reveal) => ({
        zone: reveal.zone,
        tier: reveal.tier,
        slot: reveal.slot,
        actor: reveal.actor ?? null,
        reason: reveal.reason,
        actionIdx: reveal.action_idx ?? null,
      })),
    });
  }

  async function handleSnapshotUpdate(
    nextSnapshot: GameSnapshotDTO,
    engineShouldMove = false,
    deepSearchOverride: Record<string, DeepAnalysisSearchResult> | null = null,
    suppressAutoAnalyze = false,
    preserveActiveSearch = false,
  ): Promise<void> {
    if (!preserveActiveSearch) {
      clearPolling();
    }
    const snapshotIndex = nextSnapshot.current_snapshot_index != null
      ? Number(nextSnapshot.current_snapshot_index)
      : null;
    const searchSource = deepSearchOverride ?? deepAnalysisSearchBySnapshot;
    const deepResult = snapshotIndex != null ? (searchSource[snapshotIndex] ?? null) : null;
    if (deepResult || !preserveActiveSearch) {
      setJobStatus(
        deepResult
          ? {
              job_id: `deep-${snapshotIndex}`,
              status: 'DONE',
              result: deepResult,
              error: null,
            }
          : null,
      );
    }
    setSnapshot(nextSnapshot);
    if (nextSnapshot.config?.analysis_mode) {
      setLoadedMoveLog((prev) => {
        const incoming = nextSnapshot.move_log ?? [];
        const preserveLoadedMainline = activeVariationBranchIdRef.current != null && nextSnapshot.current_snapshot_index == null;
        if (preserveLoadedMainline && prev && prev.length > 0) {
          return prev;
        }
        const loadedHistoricalLength = loadedHistoricalMainlineLengthRef.current;
        const loadedHistoricalTailSnapshot = loadedHistoricalMainlineTailSnapshotRef.current;
        const extendingLoadedMainline =
          activeVariationBranchIdRef.current == null
          && nextSnapshot.current_snapshot_index == null
          && loadedHistoricalLength > 0
          && incoming.length >= loadedHistoricalLength;
        if (extendingLoadedMainline) {
          const prefix = incoming.slice(0, loadedHistoricalLength);
          const suffix = incoming.slice(loadedHistoricalLength).map((move, idx) => ({
            ...move,
            result_snapshot_index: loadedHistoricalTailSnapshot + idx + 1,
          }));
          return [...prefix, ...suffix];
        }
        if (!prev || prev.length === 0) {
          return incoming;
        }
        if (incoming.length >= prev.length) {
          return incoming;
        }
        const isIncomingPrefix = incoming.every((move, idx) => {
          const prior = prev[idx];
          return prior
            && prior.result_turn_index === move.result_turn_index
            && prior.result_snapshot_index === move.result_snapshot_index
            && prior.action_idx === move.action_idx
            && prior.actor === move.actor;
        });
        return isIncomingPrefix ? prev : incoming;
      });
    }
    lastSnapshotSearchKeyRef.current = snapshotSearchKey(nextSnapshot);
    setUiStatus(deriveUiStatus(nextSnapshot));
    const nextAutoAnalyzeKey = autoAnalyzeKey(nextSnapshot);
    const shouldStartSearch =
      engineShouldMove ||
      (!suppressAutoAnalyze && shouldAutoAnalyze(nextSnapshot) && lastAutoAnalyzeKeyRef.current !== nextAutoAnalyzeKey);
    if (shouldStartSearch) {
      lastAutoAnalyzeKeyRef.current = nextAutoAnalyzeKey;
      await startEngineThink();
    }
  }

  function describePendingReveal(reason: GameSnapshotDTO['pending_reveals'][number]['reason']): string {
    if (reason === 'initial_setup') return 'Setup';
    if (reason === 'initial_noble_setup') return 'Noble setup';
    if (reason === 'replacement_after_buy') return 'After buy';
    if (reason === 'replacement_after_reserve') return 'After reserve';
    return 'Reserved reveal';
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

  function findCatalogCard(card: BoardStateDTO['tiers'][number]['cards'][number]): CatalogCardDTO | null {
    const matches = catalogCards.filter((candidate) =>
      (card.tier == null || candidate.tier === card.tier) &&
      candidate.points === card.points &&
      candidate.bonus_color === card.bonus_color &&
      candidate.cost.white === card.cost.white &&
      candidate.cost.blue === card.cost.blue &&
      candidate.cost.green === card.cost.green &&
      candidate.cost.red === card.cost.red &&
      candidate.cost.black === card.cost.black
    );
    if (matches.length === 0) {
      return null;
    }
    return matches[0];
  }

  function findCatalogCardId(card: BoardStateDTO['tiers'][number]['cards'][number]): number | null {
    return findCatalogCard(card)?.id ?? null;
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

  async function startEngineThink(options?: {
    searchTypeOverride?: SearchType;
    snapshotOverride?: GameSnapshotDTO | null;
  }): Promise<void> {
    setError(null);
    const requested = searchSimulations;
    const baseSnapshot = options?.snapshotOverride ?? snapshot;
    const fallback = baseSnapshot?.config?.num_simulations ?? numSimulations;
    const nextNumSimulations =
      Number.isInteger(requested) && requested >= 1
        ? requested
        : fallback;
    const thinkRequest: Record<string, unknown> = {
      num_simulations: nextNumSimulations,
      search_type: options?.searchTypeOverride ?? searchType,
      continuous_until_cancel: homeView === 'LIVE',
      max_total_simulations: homeView === 'LIVE' ? LIVE_SEARCH_MAX_SIMULATIONS : nextNumSimulations,
    };

    const think = await fetchJSON<EngineThinkResponse>('/api/game/engine-think', {
      method: 'POST',
      body: JSON.stringify(thinkRequest),
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
    setLoadedMoveLog(null);
    loadedSavedGameRef.current = null;
    loadedHistoricalMainlineLengthRef.current = 0;
    loadedHistoricalMainlineTailSnapshotRef.current = 0;
    setLoadedPlayerNames(null);
    setVariationBranches([]);
    setDeepAnalysisBySnapshot({});
    setDeepAnalysisSearchBySnapshot({});
    setIsLoadedPostAnalysisGame(false);
    setDeepAnalysisProgress(null);
    setIsDeepAnalysisRunning(false);
    activeVariationBranchIdRef.current = null;
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

  function onOpenManualView(): void {
    setError(null);
    clearPolling();
    clearLivePolling();
    setJobStatus(null);
    setSnapshot(null);
    setLoadedMoveLog(null);
    loadedSavedGameRef.current = null;
    loadedHistoricalMainlineLengthRef.current = 0;
    loadedHistoricalMainlineTailSnapshotRef.current = 0;
    setLoadedPlayerNames(null);
    setVariationBranches([]);
    setDeepAnalysisBySnapshot({});
    setDeepAnalysisSearchBySnapshot({});
    setIsLoadedPostAnalysisGame(false);
    setDeepAnalysisProgress(null);
    setIsDeepAnalysisRunning(false);
    activeVariationBranchIdRef.current = null;
    lastAutoAnalyzeKeyRef.current = null;
    setHomeView('ANALYSIS');
  }

  function onOpenLiveView(): void {
    setError(null);
    clearPolling();
    clearLivePolling();
    setJobStatus(null);
    setSnapshot(null);
    setLoadedMoveLog(null);
    loadedSavedGameRef.current = null;
    loadedHistoricalMainlineLengthRef.current = 0;
    loadedHistoricalMainlineTailSnapshotRef.current = 0;
    setLoadedPlayerNames(null);
    setVariationBranches([]);
    setDeepAnalysisBySnapshot({});
    setDeepAnalysisSearchBySnapshot({});
    setIsLoadedPostAnalysisGame(false);
    setDeepAnalysisProgress(null);
    setIsDeepAnalysisRunning(false);
    activeVariationBranchIdRef.current = null;
    setRevealSelections({});
    setActiveRevealKey(null);
    setLiveSaveStatus(null);
    lastLiveSaveUpdatedAtRef.current = null;
    lastAutoAnalyzeKeyRef.current = null;
    setHomeView('LIVE');
  }

  async function onStartManualGame(event: FormEvent): Promise<void> {
    event.preventDefault();
    try {
      await startGame(true, playerSeat, true);
      setHomeView('ANALYSIS');
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function waitMs(durationMs: number): Promise<void> {
    await new Promise<void>((resolve) => {
      window.setTimeout(resolve, durationMs);
    });
  }

  function classifyDeepAnalysisFromSearch(
    playedActionIdx: number,
    bestActionIdx: number | null,
    bestQ: number | null,
    playedQ: number | null,
  ): DeepAnalysisEntry {
    if (
      bestActionIdx == null
      || bestQ == null
      || !Number.isFinite(bestQ)
      || playedQ == null
      || !Number.isFinite(playedQ)
    ) {
      return {
        category: 'Unknown',
        playedActionIdx,
        bestActionIdx,
        playedQ,
        bestQ,
        qLoss: null,
      };
    }

    const qLoss = Math.max(0, bestQ - playedQ);
    let category: DeepAnalysisCategory;
    if (playedActionIdx === bestActionIdx) {
      category = 'Best';
    } else if (qLoss < 0.1) {
      category = 'Good';
    } else if (qLoss < 0.3) {
      category = 'Mistake';
    } else {
      category = 'Blunder';
    }

    return {
      category,
      playedActionIdx,
      bestActionIdx,
      playedQ,
      bestQ,
      qLoss,
    };
  }

  function deepAnalysisBadgeSymbol(entry: DeepAnalysisEntry): string {
    return entry.category === 'Blunder' ? '??' : '?';
  }

  function shouldShowDeepAnalysisBadge(entry: DeepAnalysisEntry): boolean {
    return entry.category === 'Mistake' || entry.category === 'Blunder';
  }

  async function runSingleDeepAnalysis(
    simulations: number,
    forcedRootActionIdx?: number,
    searchTypeOverride?: SearchType,
  ): Promise<EngineJobStatusDTO> {
    const think = await fetchJSON<EngineThinkResponse>('/api/game/engine-think', {
      method: 'POST',
      body: JSON.stringify({
        num_simulations: simulations,
        search_type: searchTypeOverride ?? searchType,
        continuous_until_cancel: false,
        max_total_simulations: simulations,
        ...(forcedRootActionIdx != null ? { forced_root_action_idx: forcedRootActionIdx } : {}),
      }),
    });
    for (;;) {
      await waitMs(200);
      const status = await fetchJSON<EngineJobStatusDTO>(`/api/game/engine-job/${think.job_id}`);
      if (status.status === 'DONE') {
        return status;
      }
      if (status.status === 'FAILED' || status.status === 'CANCELLED') {
        throw new Error(status.error ?? `Deep analysis job ${status.status.toLowerCase()}`);
      }
    }
  }

  async function restoreSnapshotForDeepAnalysis(targetSnapshotIndex: number): Promise<GameSnapshotDTO> {
    if (loadedSavedGameRef.current != null) {
      await fetchJSON<GameSnapshotDTO>('/api/game/load', {
        method: 'POST',
        body: JSON.stringify(loadedSavedGameRef.current),
      });
    }
    return fetchJSON<GameSnapshotDTO>('/api/game/jump-to-snapshot', {
      method: 'POST',
      body: JSON.stringify({ snapshot_index: targetSnapshotIndex }),
    });
  }

  async function onRunDeepAnalysis(): Promise<void> {
    if (!snapshot || moveLogEntries.length === 0 || isDeepAnalysisRunning || deepAnalysisSimulations < 1) {
      return;
    }

    const startSnapshotIndex = currentSnapshotIndex;
    const targets = moveLogEntries.filter((move) => move.result_snapshot_index > 0);
    if (targets.length === 0) {
      return;
    }

    setError(null);
    clearPolling();
    setJobStatus(null);
    setIsDeepAnalysisRunning(true);
    setDeepAnalysisProgress({ done: 0, total: targets.length });
    setDeepAnalysisBySnapshot({});
    setDeepAnalysisSearchBySnapshot({});

    try {
      for (let idx = 0; idx < targets.length; idx += 1) {
        const move = targets[idx];
        const moveKey = moveAnalysisKey(move);
        const beforeSnapshotIndex = Math.max(0, move.result_snapshot_index - 1);
        await restoreSnapshotForDeepAnalysis(beforeSnapshotIndex);
        const prerequisiteMoves = targets.slice(0, idx).filter((candidate) =>
          candidate.result_snapshot_index === move.result_snapshot_index,
        );
        for (const prerequisite of prerequisiteMoves) {
          await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
            method: 'POST',
            body: JSON.stringify({ action_idx: prerequisite.action_idx }),
          });
        }
        const status = await runSingleDeepAnalysis(deepAnalysisSimulations, undefined, searchType);
        const regularResult = status.result;
        const bestActionIdx = regularResult?.action_idx ?? null;
        const bestQ = regularResult?.selected_action_q ?? null;
        const regularPlayedQ = regularResult?.action_details.find((detail) => detail.action_idx === move.action_idx)?.q_value ?? null;
        let playedQ = regularPlayedQ;
        if (bestActionIdx == null || bestQ == null) {
          playedQ = null;
        } else if (move.action_idx === bestActionIdx) {
          playedQ = regularPlayedQ ?? bestQ;
        } else {
          const forcedStatus = await runSingleDeepAnalysis(deepAnalysisSimulations, move.action_idx, searchType);
          playedQ = forcedStatus.result?.selected_action_q ?? null;
        }
        const classified = classifyDeepAnalysisFromSearch(move.action_idx, bestActionIdx, bestQ, playedQ);
        setDeepAnalysisBySnapshot((prev) => ({ ...prev, [moveKey]: classified }));
        if (regularResult != null) {
          setDeepAnalysisSearchBySnapshot((prev) => {
            const result = regularResult as DeepAnalysisSearchResult;
            const next = {
              ...prev,
              [moveKey]: result,
            };
            if (!Object.prototype.hasOwnProperty.call(next, String(beforeSnapshotIndex))) {
              next[String(beforeSnapshotIndex)] = result;
            }
            return next;
          });
        }
        setDeepAnalysisProgress({ done: idx + 1, total: targets.length });
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      try {
        const restoredSnapshot = await restoreSnapshotForDeepAnalysis(startSnapshotIndex);
        await handleSnapshotUpdate(restoredSnapshot);
      } catch {
        // Keep current state if restore fails.
      }
      setIsDeepAnalysisRunning(false);
      setDeepAnalysisProgress(null);
    }
  }

  function deriveVariationContext(beforeSnapshot: GameSnapshotDTO, beforeSnapshotIndex: number, actor: Seat): {
    expectedMainlineMove: MoveLogEntryDTO | null;
    isFromHistoricalMainline: boolean;
    isOnMainlineSnapshot: boolean;
    baseFullMoveNumber: number;
  } | null {
    if (!loadedMoveLog || loadedMoveLog.length === 0) {
      return null;
    }
    const expectedMainlineMove = loadedMoveLog
      .filter((move) => move.result_snapshot_index > beforeSnapshotIndex)
      .sort((a, b) => a.result_snapshot_index - b.result_snapshot_index)[0] ?? null;
    const mainlineTailSnapshotIndex = loadedMoveLog[loadedMoveLog.length - 1].result_snapshot_index;
    const isFromHistoricalMainline = beforeSnapshot.current_snapshot_index != null
      && beforeSnapshot.current_snapshot_index < mainlineTailSnapshotIndex;
    const isOnMainlineSnapshot = beforeSnapshot.current_snapshot_index != null;
    const anchorMainlineMove = loadedMoveLog.find((move) => move.result_snapshot_index === beforeSnapshotIndex) ?? null;
    const anchorMainlineMoveNumber = anchorMainlineMove == null
      ? null
      : (moveLogRows.find((row) =>
          row.p0?.result_snapshot_index === anchorMainlineMove.result_snapshot_index
          || row.p1?.result_snapshot_index === anchorMainlineMove.result_snapshot_index
        )?.moveNumber ?? null);
    const lastRow = moveLogRows.length > 0 ? moveLogRows[moveLogRows.length - 1] : null;
    const fallbackBaseMoveNumber = (() => {
      if (!lastRow) return 1;
      if (actor === 'P0') {
        return lastRow.p1 != null ? lastRow.moveNumber + 1 : lastRow.moveNumber;
      }
      return lastRow.moveNumber;
    })();
    const baseFullMoveNumber = (() => {
      if (anchorMainlineMove != null && anchorMainlineMoveNumber != null) {
        return anchorMainlineMove.actor === 'P0'
          ? anchorMainlineMoveNumber
          : anchorMainlineMoveNumber + 1;
      }
      if (expectedMainlineMove) {
        return mainlineMoveNumberBySnapshot.get(expectedMainlineMove.result_snapshot_index) ?? fallbackBaseMoveNumber;
      }
      return fallbackBaseMoveNumber;
    })();
    return {
      expectedMainlineMove,
      isFromHistoricalMainline,
      isOnMainlineSnapshot,
      baseFullMoveNumber,
    };
  }

  async function onPlayerMove(
    actionIdx: number,
    options?: {
      suppressAutoAnalyze?: boolean;
      analyzeWithSearchType?: SearchType | null;
    },
  ): Promise<void> {
    const beforeSnapshot = snapshot;
    const beforeSnapshotIndex = currentSnapshotIndex;
    setError(null);
    clearPolling();
    setJobStatus(null);
    setUiStatus('WAITING_PLAYER');
    try {
      const result = await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
        method: 'POST',
        body: JSON.stringify({ action_idx: actionIdx }),
      });

      if (beforeSnapshot) {
        const actor = beforeSnapshot.player_to_move;
        const label = beforeSnapshot.legal_action_details.find((item) => item.action_idx === actionIdx)?.label ?? `Action ${actionIdx}`;
        const variationCtx = deriveVariationContext(beforeSnapshot, beforeSnapshotIndex, actor);
        const expectedMainlineMove = variationCtx?.expectedMainlineMove ?? null;
        const isOnMainlineSnapshot = variationCtx?.isOnMainlineSnapshot ?? false;
        const baseFullMoveNumber = variationCtx?.baseFullMoveNumber ?? 1;

        if (activeVariationBranchIdRef.current == null) {
          const isDeviation = isOnMainlineSnapshot && expectedMainlineMove != null && expectedMainlineMove.action_idx !== actionIdx;
          if (isDeviation) {
            const branchId = variationBranchIdCounterRef.current++;
            activeVariationBranchIdRef.current = branchId;
            setVariationBranches((prev) => [
              ...prev,
              {
                id: branchId,
                anchorSnapshotIndex: beforeSnapshotIndex,
                moves: [{
                  kind: 'move',
                  actor,
                  actionIdx,
                  label,
                  fullMoveNumber: baseFullMoveNumber,
                  targetSnapshotIndex: result.snapshot.current_snapshot_index ?? -1,
                  targetTurnIndex: result.snapshot.turn_index,
                  jumpBySnapshot: result.snapshot.current_snapshot_index != null,
                }],
              },
            ]);
          }
        } else {
          const activeId = activeVariationBranchIdRef.current;
          setVariationBranches((prev) => prev.map((branch) => {
            if (branch.id !== activeId) {
              return branch;
            }
            const shouldMergeContinuation =
              isContinuationAction(actionIdx)
              && branch.moves.length > 0
              && branch.moves[branch.moves.length - 1].kind === 'move'
              && branch.moves[branch.moves.length - 1].actor === actor;
            if (shouldMergeContinuation) {
              const updatedMoves = [...branch.moves];
              const last = updatedMoves[updatedMoves.length - 1];
              updatedMoves[updatedMoves.length - 1] = {
                ...last,
                label: `${last.label} + ${label}`,
                targetSnapshotIndex: result.snapshot.current_snapshot_index ?? -1,
                targetTurnIndex: result.snapshot.turn_index,
                jumpBySnapshot: result.snapshot.current_snapshot_index != null,
              };
              return {
                ...branch,
                moves: updatedMoves,
              };
            }
            return {
              ...branch,
              moves: [...branch.moves, {
                kind: 'move',
                actor,
                actionIdx,
                label,
                fullMoveNumber: (() => {
                  const last = branch.moves[branch.moves.length - 1];
                  if (!last) return baseFullMoveNumber;
                  return last.actor === 'P1' && actor === 'P0'
                    ? last.fullMoveNumber + 1
                    : last.fullMoveNumber;
                })(),
                targetSnapshotIndex: result.snapshot.current_snapshot_index ?? -1,
                targetTurnIndex: result.snapshot.turn_index,
                jumpBySnapshot: result.snapshot.current_snapshot_index != null,
              }],
            };
          }));
        }
      }

      const forcedSearchType = options?.analyzeWithSearchType ?? null;
      const shouldSuppressAutoAnalyze = options?.suppressAutoAnalyze ?? false;
      const shouldSuppressInHandle = shouldSuppressAutoAnalyze || forcedSearchType != null || isLoadedPostAnalysisGame;
      if (shouldSuppressInHandle) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(result.snapshot);
      }
      await handleSnapshotUpdate(
        result.snapshot,
        !shouldSuppressInHandle && result.engine_should_move,
        null,
        shouldSuppressInHandle,
      );
      if (forcedSearchType && shouldAutoAnalyze(result.snapshot)) {
        await startEngineThink({
          searchTypeOverride: forcedSearchType,
          snapshotOverride: result.snapshot,
        });
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }
  void onPlayerMove;

  async function onSelectAnalysisAction(actionIdx: number): Promise<void> {
    if (!snapshot) {
      return;
    }

    const actor = snapshot.player_to_move;
    const variationCtx = deriveVariationContext(snapshot, currentSnapshotIndex, actor);
    const mainlineMove = variationCtx?.expectedMainlineMove ?? null;
    if (mainlineMove && mainlineMove.action_idx === actionIdx) {
      await onJumpToSnapshot(mainlineMove.result_snapshot_index, false, !autoAnalyzeOnNavigation, false);
      return;
    }

    if (highlightedVariation) {
      const activeBranch = variationBranches.find((branch) => branch.id === highlightedVariation.branchId) ?? null;
      const nextMove = activeBranch?.moves[highlightedVariation.moveIndex + 1] ?? null;
      if (activeBranch && nextMove?.kind === 'move' && nextMove.actionIdx === actionIdx) {
        await onJumpToVariationMove(activeBranch, highlightedVariation.moveIndex + 1, !autoAnalyzeOnNavigation);
        return;
      }
    }

    const anchoredBranch = (variationBranchByAnchor.get(currentSnapshotIndex) ?? []).find((branch) => {
      const firstMove = branch.moves[0];
      return firstMove?.kind === 'move' && firstMove.actionIdx === actionIdx;
    }) ?? null;
    if (anchoredBranch) {
      await onJumpToVariationMove(anchoredBranch, 0, !autoAnalyzeOnNavigation);
      return;
    }

    await onPlayerMove(actionIdx, {
      suppressAutoAnalyze: true,
      analyzeWithSearchType: 'mcts',
    });
  }

  async function onUndoToStart(suppressAutoAnalyze = false): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    try {
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/undo-to-start', {
        method: 'POST',
        body: '{}',
      });
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(nextSnapshot);
      }
      await handleSnapshotUpdate(
        nextSnapshot,
        !shouldSuppressAutoAnalyze && shouldAutoAnalyze(nextSnapshot),
        null,
        shouldSuppressAutoAnalyze,
      );
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onRedoToEnd(suppressAutoAnalyze = false): Promise<void> {
    setError(null);
    clearPolling();
    setJobStatus(null);
    try {
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/redo-to-end', {
        method: 'POST',
        body: '{}',
      });
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(nextSnapshot);
      }
      await handleSnapshotUpdate(
        nextSnapshot,
        !shouldSuppressAutoAnalyze && shouldAutoAnalyze(nextSnapshot),
        null,
        shouldSuppressAutoAnalyze,
      );
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onJumpToTurn(
    turnIndex: number,
    keepActiveVariationBranch = false,
    suppressAutoAnalyze = false,
  ): Promise<void> {
    if (!snapshot || turnIndex === snapshot.turn_index) {
      return;
    }
    setError(null);
    clearPolling();
    setJobStatus(null);
    if (!keepActiveVariationBranch) {
      activeVariationBranchIdRef.current = null;
    }
    try {
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/jump-to-turn', {
        method: 'POST',
        body: JSON.stringify({ turn_index: turnIndex }),
      });
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(nextSnapshot);
      }
      await handleSnapshotUpdate(
        nextSnapshot,
        !shouldSuppressAutoAnalyze && shouldAutoAnalyze(nextSnapshot),
        null,
        shouldSuppressAutoAnalyze,
      );
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onJumpToSnapshot(
    snapshotIndex: number,
    keepActiveVariationBranch = false,
    suppressAutoAnalyze = false,
    fallbackToTurn = true,
  ): Promise<void> {
    if (!snapshot) {
      return;
    }
    setError(null);
    clearPolling();
    setJobStatus(null);
    if (!keepActiveVariationBranch) {
      activeVariationBranchIdRef.current = null;
    }
    try {
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/jump-to-snapshot', {
        method: 'POST',
        body: JSON.stringify({ snapshot_index: snapshotIndex }),
      });
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(nextSnapshot);
      }
      await handleSnapshotUpdate(
        nextSnapshot,
        !shouldSuppressAutoAnalyze && shouldAutoAnalyze(nextSnapshot),
        null,
        shouldSuppressAutoAnalyze,
      );
    } catch {
      if (!fallbackToTurn) {
        return;
      }
      const fallbackTurnIndex = (() => {
        if (snapshotIndex <= 0) {
          return 0;
        }
        let bestSnapshotIndex = -1;
        let bestTurnIndex: number | null = null;
        for (const move of moveLogEntries) {
          if (move.result_snapshot_index > snapshotIndex) {
            continue;
          }
          if (bestTurnIndex == null || move.result_snapshot_index > bestSnapshotIndex) {
            bestSnapshotIndex = move.result_snapshot_index;
            bestTurnIndex = move.result_turn_index;
          }
        }
        return bestTurnIndex ?? 0;
      })();
      // Fallback for non-snapshot sessions.
      await onJumpToTurn(fallbackTurnIndex, false, suppressAutoAnalyze || isLoadedPostAnalysisGame);
    }
  }

  async function onJumpToVisibleMainlineStart(suppressAutoAnalyze = false): Promise<void> {
    if (loadedHistoricalMainlineLengthRef.current > 0) {
      await onJumpToSnapshot(0, false, suppressAutoAnalyze, false);
      return;
    }
    await onUndoToStart(suppressAutoAnalyze);
  }

  async function onJumpToVisibleMainlineEnd(suppressAutoAnalyze = false): Promise<void> {
    const finalSnapshotIndex = mainlineMoveSnapshotIndices.length > 0
      ? mainlineMoveSnapshotIndices[mainlineMoveSnapshotIndices.length - 1]
      : 0;
    if (loadedHistoricalMainlineLengthRef.current > 0) {
      if (finalSnapshotIndex > loadedHistoricalMainlineTailSnapshotRef.current) {
        await onJumpToLoadedMainlineExtension(finalSnapshotIndex, suppressAutoAnalyze);
        return;
      }
      await onJumpToSnapshot(finalSnapshotIndex, false, suppressAutoAnalyze, false);
      return;
    }
    await onRedoToEnd(suppressAutoAnalyze);
  }

  async function onJumpToLoadedMainlineExtension(
    snapshotIndex: number,
    suppressAutoAnalyze = false,
  ): Promise<void> {
    if (!snapshot) {
      return;
    }
    const historicalLength = loadedHistoricalMainlineLengthRef.current;
    const historicalTailSnapshot = loadedHistoricalMainlineTailSnapshotRef.current;
    if (historicalLength <= 0 || snapshotIndex <= historicalTailSnapshot) {
      await onJumpToSnapshot(snapshotIndex, false, suppressAutoAnalyze, true);
      return;
    }
    const extensionMoves = moveLogEntries.slice(historicalLength);
    const extensionCount = snapshotIndex - historicalTailSnapshot;
    if (extensionCount <= 0 || extensionCount > extensionMoves.length) {
      setError(`Snapshot ${snapshotIndex} is out of bounds for the loaded mainline extension`);
      return;
    }

    setError(null);
    clearPolling();
    setJobStatus(null);
    activeVariationBranchIdRef.current = null;

    try {
      // Appended post-load mainline moves now live in backend snapshot history,
      // so prefer a direct snapshot jump instead of replaying from the tail.
      const directSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/jump-to-snapshot', {
        method: 'POST',
        body: JSON.stringify({ snapshot_index: snapshotIndex }),
      });
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(directSnapshot);
      }
      await handleSnapshotUpdate(directSnapshot, false, null, shouldSuppressAutoAnalyze);
      return;
    } catch {
      // Older sessions can still require replaying the extension from the
      // loaded historical tail. Fall back to that slower path only if a
      // direct snapshot jump is unavailable.
    }

    try {
      let nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/jump-to-snapshot', {
        method: 'POST',
        body: JSON.stringify({ snapshot_index: historicalTailSnapshot }),
      });
      for (let idx = 0; idx < extensionCount; idx += 1) {
        const item = extensionMoves[idx];
        const result = await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
          method: 'POST',
          body: JSON.stringify({ action_idx: item.action_idx }),
        });
        nextSnapshot = result.snapshot;
      }
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(nextSnapshot);
      }
      await handleSnapshotUpdate(nextSnapshot, false, null, shouldSuppressAutoAnalyze);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onStepMainline(delta: -1 | 1, suppressAutoAnalyze = true): Promise<void> {
    if (!snapshot || mainlineMoveSnapshotIndices.length === 0) {
      return;
    }

    const useTurnNavigation = snapshot.current_snapshot_index == null && !isLoadedMainlineExtensionState;
    const navigationTargets = useTurnNavigation ? mainlineMoveTurnIndices : mainlineMoveSnapshotIndices;
    const activeSnapshotIndex = useTurnNavigation ? snapshot.turn_index : currentSnapshotIndex;
    let baseIdx = 0;
    for (let i = 0; i < navigationTargets.length; i += 1) {
      if (navigationTargets[i] <= activeSnapshotIndex) {
        baseIdx = i;
      }
    }

    const nextPos = baseIdx + delta;
    if (nextPos < 0 || nextPos >= navigationTargets.length) {
      return;
    }

    const nextSnapshotIndex = navigationTargets[nextPos];
    if (nextSnapshotIndex === activeSnapshotIndex) {
      return;
    }

    if (useTurnNavigation) {
      await onJumpToTurn(nextSnapshotIndex, false, suppressAutoAnalyze);
      return;
    }
    if (
      loadedHistoricalMainlineLengthRef.current > 0 &&
      nextSnapshotIndex > loadedHistoricalMainlineTailSnapshotRef.current
    ) {
      await onJumpToLoadedMainlineExtension(nextSnapshotIndex, suppressAutoAnalyze);
      return;
    }
    await onJumpToSnapshot(nextSnapshotIndex, false, suppressAutoAnalyze, false);
  }

  async function onJumpToVariationMove(
    branch: VariationBranch,
    moveIndex: number,
    suppressAutoAnalyze = false,
  ): Promise<void> {
    if (!snapshot || moveIndex < 0 || moveIndex >= branch.moves.length) {
      return;
    }
    setError(null);
    clearPolling();
    setJobStatus(null);
    activeVariationBranchIdRef.current = branch.id;

    try {
      // Always rebuild branch state from its anchor snapshot to avoid
      // accidentally resolving turn jumps on the loaded mainline.
      let nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/jump-to-snapshot', {
        method: 'POST',
        body: JSON.stringify({ snapshot_index: branch.anchorSnapshotIndex }),
      });
      for (let idx = 0; idx <= moveIndex; idx += 1) {
        const item = branch.moves[idx];
        if (item.kind === 'move') {
          const result = await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
            method: 'POST',
            body: JSON.stringify({ action_idx: item.actionIdx }),
          });
          nextSnapshot = result.snapshot;
          continue;
        }
        if (item.kind === 'edit_faceup') {
          const result = await fetchJSON<RevealCardResponse>('/api/game/reveal-card', {
            method: 'POST',
            body: JSON.stringify({ tier: item.tier, slot: item.slot, card_id: item.cardId }),
          });
          nextSnapshot = result.snapshot;
          continue;
        }
        if (item.kind === 'edit_reserved') {
          const result = await fetchJSON<RevealCardResponse>('/api/game/reveal-reserved-card', {
            method: 'POST',
            body: JSON.stringify({ seat: item.seat, slot: item.slot, card_id: item.cardId }),
          });
          nextSnapshot = result.snapshot;
          continue;
        }
        if (item.kind === 'edit_noble') {
          const result = await fetchJSON<RevealCardResponse>('/api/game/reveal-noble', {
            method: 'POST',
            body: JSON.stringify({ slot: item.slot, noble_id: item.nobleId }),
          });
          nextSnapshot = result.snapshot;
        }
      }
      const shouldSuppressAutoAnalyze = suppressAutoAnalyze || isLoadedPostAnalysisGame;
      if (shouldSuppressAutoAnalyze) {
        lastAutoAnalyzeKeyRef.current = autoAnalyzeKey(nextSnapshot);
      }
      await handleSnapshotUpdate(nextSnapshot, false, null, shouldSuppressAutoAnalyze);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function variationMoveLabel(move: VariationMove): string {
    const moveNumber = Math.max(1, move.fullMoveNumber);
    const prefix = move.actor === 'P0' ? `${moveNumber}.` : `${moveNumber}...`;
    return `${prefix} ${actionTextLabel(move.actionIdx)}`;
  }

  function appendVariationEditNode(
    beforeSnapshot: GameSnapshotDTO,
    beforeSnapshotIndex: number,
    actor: Seat,
    label: string,
    resultSnapshot: GameSnapshotDTO,
    kind: 'edit_faceup' | 'edit_reserved' | 'edit_noble',
    payload: { tier?: number; slot?: number; seat?: Seat; cardId?: number; nobleId?: number },
  ): void {
    const variationCtx = deriveVariationContext(beforeSnapshot, beforeSnapshotIndex, actor);
    const isOnMainlineSnapshot = variationCtx?.isOnMainlineSnapshot ?? false;
    const baseFullMoveNumber = variationCtx?.baseFullMoveNumber ?? 1;

    if (activeVariationBranchIdRef.current == null) {
      if (!isOnMainlineSnapshot) {
        return;
      }
      const branchId = variationBranchIdCounterRef.current++;
      activeVariationBranchIdRef.current = branchId;
      setVariationBranches((prev) => [
        ...prev,
        {
          id: branchId,
          anchorSnapshotIndex: beforeSnapshotIndex,
          moves: [{
            kind,
            actor,
            actionIdx: -1,
            label,
            fullMoveNumber: baseFullMoveNumber,
            targetSnapshotIndex: resultSnapshot.current_snapshot_index ?? -1,
            targetTurnIndex: resultSnapshot.turn_index,
            jumpBySnapshot: resultSnapshot.current_snapshot_index != null,
            ...payload,
          }],
        },
      ]);
      return;
    }

    const activeId = activeVariationBranchIdRef.current;
    setVariationBranches((prev) => prev.map((branch) => {
      if (branch.id !== activeId) {
        return branch;
      }
      const last = branch.moves[branch.moves.length - 1];
      const fullMoveNumber = !last
        ? baseFullMoveNumber
        : (last.actor === 'P1' && actor === 'P0' ? last.fullMoveNumber + 1 : last.fullMoveNumber);
      return {
        ...branch,
        moves: [
          ...branch.moves,
          {
            kind,
            actor,
            actionIdx: -1,
            label,
            fullMoveNumber,
            targetSnapshotIndex: resultSnapshot.current_snapshot_index ?? -1,
            targetTurnIndex: resultSnapshot.turn_index,
            jumpBySnapshot: resultSnapshot.current_snapshot_index != null,
            ...payload,
          },
        ],
      };
    }));
  }

  async function onRevealCardWithId(tier: number, slot: number, cardId?: number): Promise<void> {
    const beforeSnapshot = snapshot;
    const beforeSnapshotIndex = currentSnapshotIndex;
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
      if (beforeSnapshot) {
        const tierRow = beforeSnapshot.board_state?.tiers.find((item) => item.tier === tier);
        const prior = tierRow?.cards.find((item) => item.slot === slot);
        const priorId = prior ? findCatalogCardId(prior) : null;
        const label = `[Edit] T${tier}S${slot}: #${priorId ?? '?'} -> #${Number(selected)}`;
        appendVariationEditNode(
          beforeSnapshot,
          beforeSnapshotIndex,
          beforeSnapshot.player_to_move,
          label,
          result.snapshot,
          'edit_faceup',
          { tier, slot, cardId: Number(selected) },
        );
      }
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
    const beforeSnapshot = snapshot;
    const beforeSnapshotIndex = currentSnapshotIndex;
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
      if (beforeSnapshot) {
        const player = beforeSnapshot.board_state?.players.find((item) => item.seat === seat);
        const prior = player?.reserved_public.find((item) => item.slot === slot);
        const priorId = prior ? findCatalogCardId(prior) : null;
        const label = `[Edit] ${seat}R${slot}: #${priorId ?? '?'} -> #${Number(selected)}`;
        appendVariationEditNode(
          beforeSnapshot,
          beforeSnapshotIndex,
          beforeSnapshot.player_to_move,
          label,
          result.snapshot,
          'edit_reserved',
          { seat, tier, slot, cardId: Number(selected) },
        );
      }
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
    const beforeSnapshot = snapshot;
    const beforeSnapshotIndex = currentSnapshotIndex;
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
      if (beforeSnapshot) {
        const prior = beforeSnapshot.board_state?.nobles.find((item) => item.slot === slot);
        const priorId = prior ? findCatalogNobleId(prior) : null;
        const label = `[Edit] N${slot}: #${priorId ?? '?'} -> #${Number(selected)}`;
        appendVariationEditNode(
          beforeSnapshot,
          beforeSnapshotIndex,
          beforeSnapshot.player_to_move,
          label,
          result.snapshot,
          'edit_noble',
          { slot, nobleId: Number(selected) },
        );
      }
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
    const freeEditEnabled = Boolean(snapshot) && (zone === 'faceup_card' || zone === 'reserved_card' || zone === 'noble');
    if (!hasPending && !setupEditable && !manualRevealEditable && !freeEditEnabled) {
      return;
    }
    if ((setupEditable || manualRevealEditable || freeEditEnabled) && snapshot?.board_state) {
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
    if (nextSnapshot.config?.analysis_mode) {
      return 'ANALYSIS';
    }
    return 'QUICK';
  }

  async function onSaveGame(): Promise<void> {
    setError(null);
    try {
      const saved = await fetchJSON<SavedGameDTO>('/api/game/save');
      const savedWithAnalysis: SavedGameWithDeepAnalysisDTO = {
        ...saved,
        deep_analysis: {
          move_categories_by_snapshot: { ...deepAnalysisBySnapshot },
          search_by_snapshot: { ...deepAnalysisSearchBySnapshot },
        },
      };
      const blob = new Blob([JSON.stringify(savedWithAnalysis, null, 2)], { type: 'application/json' });
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
    setLoadedMoveLog(null);
    loadedSavedGameRef.current = null;
    loadedHistoricalMainlineLengthRef.current = 0;
    loadedHistoricalMainlineTailSnapshotRef.current = 0;
    setLoadedPlayerNames(parsePlayerNamesFromFilename(file.name));
    setVariationBranches([]);
    setDeepAnalysisBySnapshot({});
    setDeepAnalysisSearchBySnapshot({});
    setIsLoadedPostAnalysisGame(false);
    setDeepAnalysisProgress(null);
    setIsDeepAnalysisRunning(false);
    activeVariationBranchIdRef.current = null;
    setRevealSelections({});
    setActiveRevealKey(null);
    lastAutoAnalyzeKeyRef.current = null;

    try {
      const raw = await file.text();
      const saved = JSON.parse(raw) as SavedGameWithDeepAnalysisDTO;
      if (!selectedCheckpoint) {
        throw new Error('Please choose a checkpoint before loading a saved game');
      }
      const loadPayload: SavedGameWithDeepAnalysisDTO = {
        ...saved,
        config: {
          ...saved.config,
          checkpoint_id: selectedCheckpoint.id,
          checkpoint_path: selectedCheckpoint.path,
        },
      };
      loadedSavedGameRef.current = loadPayload;
      const restoredCategories: Record<string, DeepAnalysisEntry> = {};
      const restoredSearch: Record<string, DeepAnalysisSearchResult> = {};
      if (saved.deep_analysis) {
        for (const [key, value] of Object.entries(saved.deep_analysis.move_categories_by_snapshot ?? {})) {
          if (value != null) {
            restoredCategories[key] = value as DeepAnalysisEntry;
          }
        }
        for (const [key, value] of Object.entries(saved.deep_analysis.search_by_snapshot ?? {})) {
          if (value != null) {
            restoredSearch[key] = value as DeepAnalysisSearchResult;
          }
        }
      }
      const hasRestoredDeepAnalysis = Object.keys(restoredCategories).length > 0 || Object.keys(restoredSearch).length > 0;
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/load', {
        method: 'POST',
        body: JSON.stringify(loadPayload),
      });
      loadedHistoricalMainlineLengthRef.current = nextSnapshot.move_log.length;
      loadedHistoricalMainlineTailSnapshotRef.current = nextSnapshot.move_log.length > 0
        ? nextSnapshot.move_log[nextSnapshot.move_log.length - 1].result_snapshot_index
        : 0;
      setDeepAnalysisBySnapshot(restoredCategories);
      setDeepAnalysisSearchBySnapshot(restoredSearch);
      setIsLoadedPostAnalysisGame(hasRestoredDeepAnalysis);
      setLoadedMoveLog(nextSnapshot.move_log);
      setVariationBranches([]);
      activeVariationBranchIdRef.current = null;
      setHomeView(deriveHomeViewFromSnapshot(nextSnapshot));
      await handleSnapshotUpdate(nextSnapshot, false, restoredSearch, hasRestoredDeepAnalysis);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const canStart = Boolean(selectedCheckpoint) && numSimulations > 0 && numSimulations <= 10000;
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
    if (parsed && parsed.zone === 'faceup_card') {
      return {
        zone: parsed.zone,
        tier: parsed.tier,
        slot: parsed.slot,
        actor: null,
        reason: 'replacement_after_buy' as const,
        action_idx: null,
      };
    }
    if (parsed && parsed.zone === 'reserved_card' && parsed.seat) {
      return {
        zone: parsed.zone,
        tier: parsed.tier,
        slot: parsed.slot,
        actor: parsed.seat,
        reason: 'reserved_from_deck' as const,
        action_idx: null,
      };
    }
    if (parsed && parsed.zone === 'noble') {
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
  const currentMainlineMove = useMemo(() => {
    let nextMove: MoveLogEntryDTO | null = null;
    for (const move of moveLogEntries) {
      if (move.result_snapshot_index <= currentSnapshotIndex) {
        continue;
      }
      if (nextMove == null || move.result_snapshot_index < nextMove.result_snapshot_index) {
        nextMove = move;
      }
    }
    return nextMove;
  }, [moveLogEntries, currentSnapshotIndex]);
  const currentDeepAnalysisEntry = useMemo(() => {
    if (!currentMainlineMove) {
      return null;
    }
    return deepAnalysisBySnapshot[moveAnalysisKey(currentMainlineMove)]
      ?? deepAnalysisBySnapshot[String(currentMainlineMove.result_snapshot_index)]
      ?? null;
  }, [currentMainlineMove, deepAnalysisBySnapshot]);
  const currentDeepAnalysisSearch = useMemo(() => {
    return deepAnalysisSearchBySnapshot[String(currentSnapshotIndex)] ?? null;
  }, [currentSnapshotIndex, deepAnalysisSearchBySnapshot]);
  const preferredAnalysisResult = useMemo<DeepAnalysisSearchResult | EngineJobStatusDTO['result'] | null>(() => {
    return jobStatus?.result ?? currentDeepAnalysisSearch ?? null;
  }, [jobStatus, currentDeepAnalysisSearch]);
  const analysisEvalValue = useMemo<number | null>(() => {
    if (homeView === 'ANALYSIS') {
      return preferredAnalysisResult?.selected_action_q
        ?? preferredAnalysisResult?.root_value
        ?? currentDeepAnalysisEntry?.bestQ
        ?? null;
    }
    return jobStatus?.result?.root_value ?? null;
  }, [homeView, currentDeepAnalysisEntry, preferredAnalysisResult, jobStatus]);
  const p0EvalValue = useMemo<number | null>(() => {
    return p0WinningEval(analysisEvalValue, snapshot?.player_to_move ?? null);
  }, [analysisEvalValue, snapshot]);
  useEffect(() => {
    displayedP0EvalRef.current = displayedP0EvalValue;
  }, [displayedP0EvalValue]);

  useEffect(() => {
    if (p0EvalValue == null || !Number.isFinite(p0EvalValue)) {
      return;
    }
    if (evalAnimationFrameRef.current !== null) {
      window.cancelAnimationFrame(evalAnimationFrameRef.current);
      evalAnimationFrameRef.current = null;
    }
    setDisplayedP0EvalValue((current) => {
      if (current == null || !Number.isFinite(current)) {
        return p0EvalValue;
      }
      return current;
    });
    const startValue = displayedP0EvalRef.current != null && Number.isFinite(displayedP0EvalRef.current)
      ? displayedP0EvalRef.current
      : p0EvalValue;
    if (Math.abs(startValue - p0EvalValue) < 0.0001) {
      setDisplayedP0EvalValue(p0EvalValue);
      return;
    }
    const startedAt = performance.now();
    const durationMs = 525;
    const step = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      const nextValue = startValue + (p0EvalValue - startValue) * eased;
      displayedP0EvalRef.current = nextValue;
      setDisplayedP0EvalValue(nextValue);
      if (progress < 1) {
        evalAnimationFrameRef.current = window.requestAnimationFrame(step);
      } else {
        evalAnimationFrameRef.current = null;
      }
    };
    evalAnimationFrameRef.current = window.requestAnimationFrame(step);
  }, [p0EvalValue]);
  const evalBarTopHeight = useMemo<number>(() => {
    if (displayedP0EvalValue == null || !Number.isFinite(displayedP0EvalValue)) {
      return 50;
    }
    return Math.max(0, Math.min(100, ((displayedP0EvalValue + 1) / 2) * 100));
  }, [displayedP0EvalValue]);
  const evalBarBottomHeight = 100 - evalBarTopHeight;
  const evalBarValueClass = useMemo<string>(() => {
    return topMoveEvalClass(displayedP0EvalValue);
  }, [displayedP0EvalValue]);
  const topAnalysisMoves = useMemo(() => {
    if (snapshot?.status !== 'IN_PROGRESS') {
      return [];
    }
    const details = preferredAnalysisResult?.action_details ?? [];
    return details
      .filter((detail) => !detail.masked)
      .slice()
      .sort((a, b) => {
        if (b.policy_prob !== a.policy_prob) return b.policy_prob - a.policy_prob;
        return a.action_idx - b.action_idx;
      })
      .slice(0, 3);
  }, [preferredAnalysisResult, snapshot?.status]);
  const playedAnalysisMove = useMemo(() => {
    if (!currentDeepAnalysisEntry) {
      return null;
    }
    const details = preferredAnalysisResult?.action_details ?? [];
    return details.find((detail) => detail.action_idx === currentDeepAnalysisEntry.playedActionIdx) ?? null;
  }, [currentDeepAnalysisEntry, preferredAnalysisResult]);
  const allAnalysisMoves = useMemo(() => {
    const details = snapshot?.legal_action_details ?? [];
    return details
      .slice()
      .sort((a, b) => a.action_idx - b.action_idx);
  }, [snapshot]);
  const displayBoard = useMemo(() => {
    if (!snapshot?.board_state) {
      return null;
    }
    const board: BoardStateDTO = structuredClone(snapshot.board_state);
    const pendingByKey = new Set(snapshot.pending_reveals.map((reveal) => revealKey(reveal.zone, reveal.tier, reveal.slot)));

    board.players = board.players.map((player) => {
      const overrideName = loadedPlayerNames?.[player.seat];
      if (!overrideName) {
        return player;
      }
      return {
        ...player,
        display_name: `${player.display_name} ${overrideName}`,
      };
    }) as BoardStateDTO['players'];

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
      return nobleBySlot.get(slot) ?? null;
    }).filter((noble): noble is NonNullable<typeof noble> => noble != null) as BoardStateDTO['nobles'];

    return board;
  }, [snapshot, activeRevealKey, loadedPlayerNames]);
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
  useEffect(() => {
    if (!showAnalysisSettings) {
      return undefined;
    }
    const onPointerDown = (event: MouseEvent) => {
      if (analysisSettingsRef.current && !analysisSettingsRef.current.contains(event.target as Node)) {
        setShowAnalysisSettings(false);
      }
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [showAnalysisSettings]);

  useEffect(() => {
    const container = moveLogGridRef.current;
    if (!container || moveLogEntries.length === 0) {
      return;
    }
    let frameId = 0;
    frameId = window.requestAnimationFrame(() => {
      if (currentSnapshotIndex <= 0 && highlightedVariation == null) {
        container.scrollTop = 0;
        return;
      }
      const activeElements = Array.from(
        container.querySelectorAll<HTMLElement>('.move-log-btn.active, .move-log-variation-btn.active'),
      );
      const target = activeElements.length > 0 ? activeElements[activeElements.length - 1] : null;
      if (!target) {
        return;
      }
      const containerRect = container.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const pad = 6;
      const visibleTop = containerRect.top + pad;
      const visibleBottom = containerRect.bottom - pad;

      if (targetRect.top < visibleTop) {
        container.scrollTop -= (visibleTop - targetRect.top);
        return;
      }
      if (targetRect.bottom > visibleBottom) {
        container.scrollTop += (targetRect.bottom - visibleBottom);
      }
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [
    currentSnapshotIndex,
    highlightedVariation,
    moveLogEntries.length,
    moveLogRows.length,
    showBoardAnalysis,
    topAnalysisMoves.length,
  ]);

  useEffect(() => {
    const keyboardNavigationEnabled = homeView === 'QUICK' || homeView === 'ANALYSIS' || isSetupLikeView || homeView === 'LIVE';
    const activeSnapshot = snapshot;
    if (!activeSnapshot || !keyboardNavigationEnabled || moveLogEntries.length === 0 || isDeepAnalysisRunning) {
      return;
    }
    const snapshotForKeys: GameSnapshotDTO = activeSnapshot;
    const isLoadedMainlineExtensionState =
      loadedHistoricalMainlineLengthRef.current > 0 &&
      snapshotForKeys.current_snapshot_index == null &&
      currentSnapshotIndex > loadedHistoricalMainlineTailSnapshotRef.current;

    function onKeyDown(event: KeyboardEvent): void {
      if (event.defaultPrevented) {
        return;
      }
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight' && event.key !== 'ArrowUp' && event.key !== 'ArrowDown') {
        return;
      }

      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT'
          || target.tagName === 'TEXTAREA'
          || target.tagName === 'SELECT'
          || target.isContentEditable)
      ) {
        return;
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault();
        if (snapshotForKeys.current_snapshot_index == null && !isLoadedMainlineExtensionState) {
          void onJumpToTurn(0, false, !autoAnalyzeOnNavigation);
          return;
        }
        void onJumpToSnapshot(0, false, !autoAnalyzeOnNavigation, false);
        return;
      }

      if (event.key === 'ArrowDown') {
        const useTurnNavigation = snapshotForKeys.current_snapshot_index == null && !isLoadedMainlineExtensionState;
        const finalSnapshotIndex = useTurnNavigation
          ? (mainlineMoveTurnIndices.length > 0 ? mainlineMoveTurnIndices[mainlineMoveTurnIndices.length - 1] : 0)
          : (mainlineMoveSnapshotIndices.length > 0 ? mainlineMoveSnapshotIndices[mainlineMoveSnapshotIndices.length - 1] : 0);
        event.preventDefault();
        if (useTurnNavigation) {
          void onJumpToTurn(finalSnapshotIndex, false, !autoAnalyzeOnNavigation);
          return;
        }
        if (
          loadedHistoricalMainlineLengthRef.current > 0 &&
          finalSnapshotIndex > loadedHistoricalMainlineTailSnapshotRef.current
        ) {
          void onJumpToLoadedMainlineExtension(finalSnapshotIndex, !autoAnalyzeOnNavigation);
          return;
        }
        void onJumpToSnapshot(finalSnapshotIndex, false, !autoAnalyzeOnNavigation, false);
        return;
      }

      const delta: -1 | 1 = event.key === 'ArrowLeft' ? -1 : 1;

      event.preventDefault();
      void onStepMainline(delta, !autoAnalyzeOnNavigation);
    }

    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [
    snapshot,
    homeView,
    isSetupLikeView,
    moveLogEntries.length,
    isDeepAnalysisRunning,
    autoAnalyzeOnNavigation,
    mainlineMoveSnapshotIndices,
    mainlineMoveTurnIndices,
    currentSnapshotIndex,
  ]);

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
        const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/live-save/load', {
          method: 'POST',
          body: '{}',
        });
        const nextSearchKey = snapshotSearchKey(nextSnapshot);
        const preserveActiveSearch =
          activeJobIdRef.current !== null &&
          lastSnapshotSearchKeyRef.current === nextSearchKey;
        if (!preserveActiveSearch) {
          clearPolling();
          setJobStatus(null);
        }
        lastLiveSaveUpdatedAtRef.current = status.updated_at;
        if (nextSnapshot.config?.checkpoint_id) {
          setCheckpointId(nextSnapshot.config.checkpoint_id);
        }
        await handleSnapshotUpdate(nextSnapshot, false, null, false, preserveActiveSearch);
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

  useEffect(() => {
    if (!hideAllExceptBoard) {
      return;
    }

    function onKeyDown(event: KeyboardEvent): void {
      if (event.key !== 'Enter') {
        return;
      }
      const target = event.target;
      if (
        target instanceof HTMLElement
        && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)
      ) {
        return;
      }
      event.preventDefault();
      setHideAllExceptBoard(false);
    }

    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [hideAllExceptBoard]);

  const isBoardView = (homeView === 'QUICK' || isSetupLikeView || homeView === 'LIVE') && snapshot;

  function renderMoveContent(move: MoveLogDisplayEntry): JSX.Element {
    const parts = move.label.split(' + ').filter((part) => part.length > 0);
    const extras = parts.slice(1);
    return (
      <span className="action-label">
        <ActionLabel actionIdx={move.action_idx} display={move.display ?? null} board={displayBoard ?? snapshot?.board_state ?? null} />
        {extras.map((part, idx) => (
          <span key={`${move.result_snapshot_index}-extra-${idx}`} className="action-meta">{` + ${part}`}</span>
        ))}
      </span>
    );
  }

  function renderMoveLabel(move: MoveLogDisplayEntry | undefined): JSX.Element | string {
    if (!move) {
      return '-';
    }
    const entry = deepAnalysisBySnapshot[moveAnalysisKey(move)]
      ?? deepAnalysisBySnapshot[String(move.result_snapshot_index)];
    if (!entry || !shouldShowDeepAnalysisBadge(entry)) {
      return renderMoveContent(move);
    }
    const categoryClass = entry.category.toLowerCase();
    return (
      <span className="move-log-label-wrap">
        <span className="move-log-label-main">
          {renderMoveContent(move)}
        </span>
        <span
          className={`deep-analysis-badge ${categoryClass}`}
          aria-label={entry.category}
          title={entry.category}
        >
          <span aria-hidden="true">{deepAnalysisBadgeSymbol(entry)}</span>
        </span>
      </span>
    );
  }
  return (
    <main className={`app-shell ${isBoardView ? 'app-shell-board' : ''} ${hideAllExceptBoard ? 'board-only-mode' : ''}`}>
      {!hideAllExceptBoard && (
      <header>
        <h1>AhinLab</h1>
        <div className="header-actions">
          {homeView !== 'HOME' && (
            <>
              <input
                ref={loadInputRef}
                type="file"
                accept="application/json"
                className="visually-hidden"
                onChange={(event) => void onLoadGameFile(event)}
              />
              {homeView === 'ANALYSIS' && snapshot && (
                <>
                  <button
                    type="button"
                    onClick={() => void onRunDeepAnalysis()}
                    disabled={isDeepAnalysisRunning || moveLogEntries.length === 0 || deepAnalysisSimulations < 1}
                    title={`Run deep analysis across all logged moves (${deepAnalysisSimulations.toLocaleString()} sims per move)`}
                  >
                    {isDeepAnalysisRunning ? 'Running Deep Analysis...' : 'Run Deep Analysis'}
                  </button>
                  {deepAnalysisProgress && (
                    <span className="header-inline-status">
                      {deepAnalysisProgress.done} / {deepAnalysisProgress.total}
                    </span>
                  )}
                </>
              )}
              {(homeView !== 'ANALYSIS' || snapshot) && (
                <>
                  <button type="button" onClick={() => void onSaveGame()} disabled={!snapshot}>
                    Save Game
                  </button>
                  <button type="button" onClick={onLoadGameClick}>
                    Load Game
                  </button>
                </>
              )}
            </>
          )}
          {homeView !== 'HOME' && (
            <button type="button" onClick={() => setHomeView('HOME')}>
              Back
            </button>
          )}
        </div>
      </header>
      )}

      {homeView === 'HOME' && (
        <section className="home-landing">
          <div className="home-hero">
            <h2>Dashboard</h2>
          </div>
          <div className="home-mode-grid">
            <button type="button" className="home-mode-card" onClick={() => setHomeView('QUICK')}>
              <span className="home-mode-kicker">01</span>
              <strong>Quick Game</strong>
              <span>Engine vs human from a random opening.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={onOpenManualView}>
              <span className="home-mode-kicker">02</span>
              <strong>Analysis</strong>
              <span>Manual setup with continuous analysis.</span>
            </button>
            <button type="button" className="home-mode-card" onClick={onOpenLiveView}>
              <span className="home-mode-kicker">03</span>
              <strong>Live</strong>
              <span>Track the latest live bridge save.</span>
            </button>
          </div>
        </section>
      )}

      {homeView === 'QUICK' && (
        <section className="panel quick-game-panel">
        <h2>Quick Game</h2>
        <form onSubmit={(event) => void onStartGame(event)} className="grid-form quick-game-form">
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
          <h2>Analysis</h2>
          <form onSubmit={(event) => void onStartManualGame(event)} className="grid-form analysis-setup-form">
            <label className="analysis-setup-field">
              Checkpoint
              <select value={checkpointId} onChange={(event) => setCheckpointId(event.target.value)} disabled={checkpoints.length === 0}>
                <option value="">Select checkpoint</option>
                {checkpoints.map((item) => (
                  <option key={`setup-${item.id}`} value={item.id}>
                  {item.name}
                </option>
              ))}
              </select>
            </label>
            <button type="button" onClick={onLoadGameClick}>
              Load Game
            </button>
            <button type="submit" disabled={!canStart}>
              Setup
            </button>
          </form>
        </section>
      )}

      {homeView === 'LIVE' && !snapshot && (
        <section className="panel">
          <h2>Live</h2>
          <p>{liveSaveStatus?.path ?? 'Waiting for live save path...'}</p>
          <p>{liveSaveStatus?.exists ? `Last update: ${liveSaveStatus.updated_at ?? 'unknown'}` : 'No live save file found yet.'}</p>
        </section>
      )}

      {isBoardView && (
        <section className={`panel game-layout ${hideAllExceptBoard ? 'board-only-mode' : ''}`}>
          <div className="board-column">
            <div className="board-analysis-shell">
              <div
                className={`eval-bar-wrap ${showBoardAnalysis && !hideAllExceptBoard ? '' : 'hidden'}`}
                aria-label="Evaluation bar"
                aria-hidden={!showBoardAnalysis || hideAllExceptBoard}
              >
                {showBoardAnalysis && !hideAllExceptBoard && (
                  <>
                    <div className="eval-bar">
                      <div className="eval-bar-top" style={{ height: `${evalBarTopHeight}%` }} />
                      <div className="eval-bar-bottom" style={{ height: `${evalBarBottomHeight}%` }} />
                      <div className={`eval-bar-value ${evalBarValueClass}`}>{formatEvalBar(displayedP0EvalValue)}</div>
                    </div>
                  </>
                )}
              </div>
              <div className="board-stage">
                {displayBoard ? (
                  <GameBoard
                    board={displayBoard}
                    isTerminal={snapshot.status !== 'IN_PROGRESS'}
                    mctsTopAction={liveMctsTopAction}
                    modelTopAction={liveModelTopAction}
                    onCardClick={(tier, slot) => openReveal('faceup_card', tier, slot)}
                    onNobleClick={(slot) => openReveal('noble', 0, slot)}
                    onReservedCardClick={(seat, slot) => {
                      const player = displayBoard.players.find((item) => item.seat === seat);
                      const card = player?.reserved_public.find((item) => item.slot === slot);
                      const inferredTier = card ? (findCatalogCard(card)?.tier ?? null) : null;
                      const tier = card?.tier
                        ?? inferredTier
                        ?? snapshot.pending_reveals.find((item) => item.zone === 'reserved_card' && item.actor === seat && item.slot === slot)?.tier;
                      if (tier != null) {
                        openReveal('reserved_card', tier, slot, seat);
                      }
                    }}
                  />
                ) : (
                  <div className="empty-note">Board data unavailable</div>
                )}
              </div>
            </div>
          </div>

          {!hideAllExceptBoard && (
          <aside className="engine-column">
            <div className="engine-box">
              <div className="analysis-panel-tabs" role="tablist" aria-label="Analysis panel sections">
                <button
                  type="button"
                  role="tab"
                  aria-selected={analysisPanelTab === 'ANALYSIS'}
                  className={`analysis-panel-tab ${analysisPanelTab === 'ANALYSIS' ? 'active' : ''}`}
                  onClick={() => setAnalysisPanelTab('ANALYSIS')}
                >
                  <span>Analysis</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={analysisPanelTab === 'MOVES'}
                  className={`analysis-panel-tab ${analysisPanelTab === 'MOVES' ? 'active' : ''}`}
                  onClick={() => setAnalysisPanelTab('MOVES')}
                >
                  <span>Moves</span>
                </button>
              </div>
              {analysisPanelTab === 'ANALYSIS' && (
                <div className="analysis-controls-row">
                <label className="analysis-toggle">
                  <input
                    type="checkbox"
                    checked={showBoardAnalysis}
                    onChange={(event) => setShowBoardAnalysis(event.target.checked)}
                  />
                  <span>Analysis</span>
                </label>
                <div className="analysis-settings-wrap" ref={analysisSettingsRef}>
                  <button
                    type="button"
                    className={`analysis-settings-btn ${showAnalysisSettings ? 'active' : ''}`}
                    title={homeView === 'LIVE'
                      ? `${searchType.toUpperCase()} • publish every ${searchSimulations.toLocaleString()} sims`
                      : `${searchType.toUpperCase()} • ${searchSimulations.toLocaleString()} sims`}
                    aria-expanded={showAnalysisSettings}
                    aria-haspopup="dialog"
                    onClick={() => setShowAnalysisSettings((value) => !value)}
                  >
                    <svg className="analysis-settings-icon" viewBox="0 0 20 20" aria-hidden="true">
                      <path d="M11.8 1.5a1 1 0 0 0-1.96 0l-.2 1.2a7.4 7.4 0 0 0-1.45.6l-1.02-.67a1 1 0 0 0-1.32.2L4.7 4.12a1 1 0 0 0 .2 1.32l.94.74c-.1.28-.18.57-.24.86l-1.16.2a1 1 0 0 0 0 1.96l1.16.2c.06.3.14.58.24.86l-.94.74a1 1 0 0 0-.2 1.32l1.15 1.29a1 1 0 0 0 1.32.2l1.02-.67c.46.25.94.45 1.45.6l.2 1.2a1 1 0 0 0 1.96 0l.2-1.2c.5-.15.99-.35 1.45-.6l1.02.67a1 1 0 0 0 1.32-.2l1.15-1.29a1 1 0 0 0-.2-1.32l-.94-.74c.1-.28.18-.57.24-.86l1.16-.2a1 1 0 0 0 0-1.96l-1.16-.2a6.4 6.4 0 0 0-.24-.86l.94-.74a1 1 0 0 0 .2-1.32l-1.15-1.29a1 1 0 0 0-1.32-.2l-1.02.67a7.4 7.4 0 0 0-1.45-.6zm-1 5.2a2.3 2.3 0 1 1-1.6 4.3 2.3 2.3 0 0 1 1.6-4.3Z" />
                    </svg>
                  </button>
                  {showAnalysisSettings && (
                    <div className="analysis-settings-popover" role="dialog" aria-label="Analysis settings">
                      {snapshot.status === 'IN_PROGRESS' && (snapshot.config?.analysis_mode || snapshot.player_to_move !== snapshot.config?.player_seat) && (
                        <div className="analysis-settings-section analysis-search-row">
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
                            aria-label={homeView === 'LIVE' ? 'Intermediate publish simulations' : 'Search simulations'}
                            title={homeView === 'LIVE' ? 'Publish updated live analysis every N simulations during the same search job' : 'Search simulations'}
                          />
                          <button
                            onClick={() => {
                              void startEngineThink();
                              setShowAnalysisSettings(false);
                            }}
                            disabled={searchSimulations < 1 || uiStatus === 'WAITING_ENGINE'}
                          >
                            {homeView === 'LIVE' ? 'Analyze Turn' : 'Run Search'}
                          </button>
                        </div>
                      )}
                      {homeView === 'LIVE' && (
                        <div className="analysis-settings-section analysis-search-row">
                          <span>Limit</span>
                          <span>{LIVE_SEARCH_MAX_SIMULATIONS.toLocaleString()} sims</span>
                        </div>
                      )}
                      {homeView !== 'LIVE' && (
                        <div className="analysis-settings-section analysis-search-row">
                          <span>Deep</span>
                          <input
                            type="number"
                            min={1}
                            max={LIVE_SEARCH_MAX_SIMULATIONS}
                            value={deepAnalysisSimulations}
                            onChange={(event) => setDeepAnalysisSimulations(Number(event.target.value))}
                            aria-label="Deep analysis simulations"
                            title="Deep analysis simulations per move"
                          />
                        </div>
                      )}
                      <div className="analysis-settings-section">
                        <label className="analysis-toggle">
                          <input
                            type="checkbox"
                            checked={hideAllExceptBoard}
                            onChange={(event) => {
                              const nextValue = event.target.checked;
                              setHideAllExceptBoard(nextValue);
                              if (nextValue) {
                                setShowAnalysisSettings(false);
                              }
                            }}
                          />
                          <span>Hide all except board</span>
                        </label>
                      </div>
                    </div>
                  )}
                </div>
                </div>
              )}
              {uiStatus === 'WAITING_REVEAL' && <p>Waiting for board update before the next move.</p>}
              {jobStatus?.error && <p className="error">Engine error: {jobStatus.error}</p>}
              {homeView === 'LIVE' && (
                <p>
                  Live mode runs one search job up to {LIVE_SEARCH_MAX_SIMULATIONS.toLocaleString()} sims and publishes updated analysis every {searchSimulations.toLocaleString()} sims.
                  {jobStatus?.status === 'RUNNING' && ' Search in progress.'}
                  {jobStatus?.result?.total_simulations != null && ` Current total: ${jobStatus.result.total_simulations}.`}
                </p>
              )}
              <div className="analysis-panel-body">
                {analysisPanelTab === 'ANALYSIS' && showBoardAnalysis && (
                  <div className="analysis-lines" role="list">
                      {currentDeepAnalysisEntry && (
                        <div className="analysis-played-block">
                          <div className="analysis-section-header">Move played</div>
                          <div className="analysis-line" role="listitem">
                            <div className="analysis-line-stats">
                              <span
                                className={`analysis-line-q ${topMoveEvalClass(
                                  p0WinningEval(currentDeepAnalysisEntry.playedQ, snapshot?.player_to_move ?? null),
                                )}`}
                              >
                                {formatTopMoveEval(
                                  p0WinningEval(currentDeepAnalysisEntry.playedQ, snapshot?.player_to_move ?? null),
                                )}
                              </span>
                              <span className="analysis-line-visit analysis-line-visit-placeholder" aria-hidden="true">
                                --
                              </span>
                            </div>
                            <div className="analysis-line-name">
                              {playedAnalysisMove
                                ? (
                                    <ActionLabel
                                      actionIdx={playedAnalysisMove.action_idx}
                                      display={playedAnalysisMove.display ?? null}
                                      board={displayBoard ?? snapshot?.board_state ?? null}
                                    />
                                  )
                                : currentDeepAnalysisEntry.playedActionIdx}
                            </div>
                          </div>
                        </div>
                      )}
                      {snapshot?.status === 'IN_PROGRESS' && (
                        <>
                          <div className="analysis-section-header">Top moves</div>
                          {Array.from({ length: 3 }, (_, index) => {
                            const detail = topAnalysisMoves[index] ?? null;
                            const absoluteEval = detail
                              ? p0WinningEval(detail.q_value, snapshot?.player_to_move ?? null)
                              : null;
                            const evalClass = topMoveEvalClass(absoluteEval);
                            return (
                              <button
                                key={detail ? `analysis-line-${detail.action_idx}` : `analysis-line-placeholder-${index}`}
                                type="button"
                                className={`analysis-line analysis-line-button ${detail ? '' : 'placeholder'}`}
                                role="listitem"
                                disabled={!detail}
                                onClick={() => {
                                  if (detail) {
                                    void onSelectAnalysisAction(detail.action_idx);
                                  }
                                }}
                              >
                                <div className="analysis-line-stats">
                                  <span className={`analysis-line-q ${evalClass}`}>
                                    {detail ? formatTopMoveEval(absoluteEval) : '--'}
                                  </span>
                                  <span className="analysis-line-visit">
                                    {detail ? `${(detail.policy_prob * 100).toFixed(1)}%` : '--'}
                                  </span>
                                </div>
                                <div className="analysis-line-name">
                                  {detail
                                    ? <ActionLabel
                                        actionIdx={detail.action_idx}
                                        display={detail.display ?? null}
                                        board={displayBoard ?? snapshot?.board_state ?? null}
                                      />
                                    : 'Waiting for search...'}
                                </div>
                              </button>
                            );
                          })}
                        </>
                      )}
                  </div>
                )}
                {analysisPanelTab === 'ANALYSIS' && !showBoardAnalysis && (
                  <div className="analysis-line placeholder analysis-panel-empty" role="status">Analysis hidden</div>
                )}
                {analysisPanelTab === 'MOVES' && (
                  <div className="analysis-moves-list" role="list">
                    {allAnalysisMoves.length === 0 ? (
                      <div className="analysis-line placeholder analysis-panel-empty" role="listitem">Waiting for search...</div>
                    ) : (
                      allAnalysisMoves.map((detail) => (
                        <button
                          key={`analysis-move-${detail.action_idx}`}
                          type="button"
                          className="analysis-line analysis-line-move-only analysis-line-button"
                          role="listitem"
                          onClick={() => {
                            void onSelectAnalysisAction(detail.action_idx);
                          }}
                        >
                          <div className="analysis-line-name">
                            <ActionLabel
                              actionIdx={detail.action_idx}
                              display={detail.display ?? null}
                              board={displayBoard ?? snapshot?.board_state ?? null}
                            />
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="move-log-wrap">
              <div className="move-log-title-grid" aria-hidden="true">
                <span>#</span>
                <span>P1</span>
                <span>P2</span>
              </div>
              {moveLogEntries.length === 0 ? (
                <p className="empty-note">No moves yet.</p>
              ) : (
                <div className="move-log-grid" role="list" ref={moveLogGridRef}>
                  {variationBranchByAnchor.has(0) && (
                    <div className="move-log-variation-row" key="variation-start">
                      <div className="move-log-number" />
                      <div className="move-log-variation" style={{ gridColumn: '2 / span 2' }}>
                        {variationBranchByAnchor.get(0)?.map((branch) => (
                          <div key={`variation-start-branch-${branch.id}`} className="move-log-variation-line">
                            {branch.moves.map((move, idx) => (
                              <button
                                key={`variation-start-${branch.id}-${idx}`}
                                type="button"
                                className={`move-log-variation-btn ${
                                  highlightedVariation != null
                                  && highlightedVariation.branchId === branch.id
                                  && highlightedVariation.moveIndex === idx
                                    ? 'active'
                                    : ''
                                }`}
                                onClick={() => void onJumpToVariationMove(branch, idx, !autoAnalyzeOnNavigation)}
                              >
                                {variationMoveLabel(move)}
                              </button>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {moveLogRows.map((row, rowIdx) => {
                    const p0TargetSnapshot = row.p0?.result_snapshot_index ?? null;
                    const p1TargetSnapshot = row.p1?.result_snapshot_index ?? null;
                    const rowBranches = [
                      ...(p0TargetSnapshot != null ? (variationBranchByAnchor.get(p0TargetSnapshot) ?? []) : []),
                      ...(p1TargetSnapshot != null ? (variationBranchByAnchor.get(p1TargetSnapshot) ?? []) : []),
                    ];
                    return (
                      <Fragment key={`move-row-${row.moveNumberLabel}-${rowIdx}`}>
                        <div className="move-log-row">
                          <div className="move-log-number">{row.moveNumberLabel}.</div>
                          <button
                            type="button"
                            className={`move-log-btn ${
                              isHighlightedMainlineMove(row.p0)
                                ? 'active'
                                : ''
                            }`}
                            disabled={
                              p0TargetSnapshot == null
                              || isHighlightedMainlineMove(row.p0)
                            }
                            onClick={() => {
                              if (p0TargetSnapshot != null) {
                                const isLoadedMainlineExtension =
                                  loadedHistoricalMainlineLengthRef.current > 0
                                  && p0TargetSnapshot > loadedHistoricalMainlineTailSnapshotRef.current;
                                if (isLoadedMainlineExtension) {
                                  void onJumpToLoadedMainlineExtension(p0TargetSnapshot, !autoAnalyzeOnNavigation);
                                } else {
                                  void onJumpToSnapshot(p0TargetSnapshot, false, !autoAnalyzeOnNavigation);
                                }
                              }
                            }}
                          >
                            {renderMoveLabel(row.p0)}
                          </button>
                          <button
                            type="button"
                            className={`move-log-btn ${
                              isHighlightedMainlineMove(row.p1)
                                ? 'active'
                                : ''
                            }`}
                            disabled={
                              p1TargetSnapshot == null
                              || isHighlightedMainlineMove(row.p1)
                            }
                            onClick={() => {
                              if (p1TargetSnapshot != null) {
                                const isLoadedMainlineExtension =
                                  loadedHistoricalMainlineLengthRef.current > 0
                                  && p1TargetSnapshot > loadedHistoricalMainlineTailSnapshotRef.current;
                                if (isLoadedMainlineExtension) {
                                  void onJumpToLoadedMainlineExtension(p1TargetSnapshot, !autoAnalyzeOnNavigation);
                                } else {
                                  void onJumpToSnapshot(p1TargetSnapshot, false, !autoAnalyzeOnNavigation);
                                }
                              }
                            }}
                          >
                            {renderMoveLabel(row.p1)}
                          </button>
                        </div>
                        {rowBranches.length > 0 && (
                          <div className="move-log-variation-row">
                            <div className="move-log-number" />
                            <div className="move-log-variation" style={{ gridColumn: '2 / span 2' }}>
                              {rowBranches.map((branch) => (
                                <div key={`variation-row-${row.moveNumber}-branch-${branch.id}`} className="move-log-variation-line">
                                  {branch.moves.map((move, idx) => (
                                    <button
                                      key={`variation-${row.moveNumber}-${branch.id}-${idx}`}
                                      type="button"
                                      className={`move-log-variation-btn ${
                                        highlightedVariation != null
                                        && highlightedVariation.branchId === branch.id
                                        && highlightedVariation.moveIndex === idx
                                          ? 'active'
                                          : ''
                                      }`}
                                      disabled={
                                        highlightedVariation != null
                                        && highlightedVariation.branchId === branch.id
                                        && highlightedVariation.moveIndex === idx
                                      }
                                      onClick={() => void onJumpToVariationMove(branch, idx, !autoAnalyzeOnNavigation)}
                                    >
                                      {variationMoveLabel(move)}
                                    </button>
                                  ))}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </Fragment>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="analysis-nav-panel">
              <div className="analysis-nav-row">
                <button type="button" onClick={() => void onJumpToVisibleMainlineStart(!autoAnalyzeOnNavigation)} disabled={!canStepVisibleMainlineBackward} aria-label="First move" title="First move">
                  {'<<'}
                </button>
                <button type="button" onClick={() => void onStepMainline(-1, !autoAnalyzeOnNavigation)} disabled={!canStepVisibleMainlineBackward}>
                  {'<'}
                </button>
                <button type="button" onClick={() => void onStepMainline(1, !autoAnalyzeOnNavigation)} disabled={!canStepVisibleMainlineForward}>
                  {'>'}
                </button>
                <button type="button" onClick={() => void onJumpToVisibleMainlineEnd(!autoAnalyzeOnNavigation)} disabled={!canStepVisibleMainlineForward} aria-label="Last position" title="Last position">
                  {'>>'}
                </button>
              </div>
            </div>
          </aside>
          )}
        </section>
      )}

      {activeReveal && (
        <section className="reveal-modal-backdrop" onClick={() => setActiveRevealKey(null)}>
          <div className="reveal-modal" onClick={(event) => event.stopPropagation()}>
            <h3 className="reveal-modal-title">
              {activeReveal.zone === 'noble'
                ? `Noble ${activeReveal.slot}`
                : activeReveal.zone === 'reserved_card'
                  ? `${activeReveal.actor} Reserved ${activeReveal.slot}`
                  : `Tier ${activeReveal.tier} Slot ${activeReveal.slot}`}
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
                  <div className="current-noble-slot-row noble-slot-row">
                    {activeBoardNobles.map((noble) => (
                      <div
                        key={`setup-noble-slot-${noble.slot}`}
                        className="current-noble-slot"
                        onClick={() => noble.slot != null && openReveal('noble', 0, noble.slot)}
                      >
                        <NobleView noble={noble} />
                      </div>
                    ))}
                  </div>
                )}
                <div className="noble-catalog-grid">
                  {catalogNobles.map((noble) => {
                    const isAvailable =
                      !isSetupLikeView || activeReveal.reason !== 'initial_noble_setup'
                        ? true
                        : !setupUnavailableNobleIds.has(noble.id);
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
                          const isFreeEdit = !isSetup;
                          const isReservedReplace = activeReveal.zone === 'reserved_card';
                          const isAvailable = isReservedReplace
                            ? liveAvailableCardIds.has(card.id)
                            : (isFreeEdit
                              ? true
                              : (isSetup ? !setupUnavailableCardIds.has(card.id) : (liveAvailableCardIds.has(card.id) || isOccupiedSwap)));
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
