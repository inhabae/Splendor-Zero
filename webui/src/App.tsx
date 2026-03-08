import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  CheckpointDTO,
  EngineJobStatusDTO,
  EngineThinkResponse,
  GameSnapshotDTO,
  PlayerMoveResponse,
  ReplayStepDTO,
  Seat,
  SelfPlayRunResponse,
  SelfPlaySessionDTO,
  SelfPlaySessionSummaryDTO,
} from './types';
import { GameBoard } from './components/board/GameBoard';
import { ActionLabel } from './components/ActionLabel';

type UiStatus = 'IDLE' | 'WAITING_ENGINE' | 'WAITING_PLAYER' | 'GAME_OVER';

const POLL_MS = 400;

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
  return winner === 0 ? 'Winner: P0' : 'Winner: P1';
}

export function App() {
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [checkpointId, setCheckpointId] = useState('');
  const [numSimulations, setNumSimulations] = useState(400);
  const [playerSeat, setPlayerSeat] = useState<Seat>('P0');
  const [seed, setSeed] = useState('');

  const [snapshot, setSnapshot] = useState<GameSnapshotDTO | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
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
        setCheckpoints(list);
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
    return nextSnapshot.player_to_move === nextSnapshot.config?.player_seat
      ? 'WAITING_PLAYER'
      : 'WAITING_ENGINE';
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

  async function startEngineThink(): Promise<void> {
    setError(null);
    const think = await fetchJSON<EngineThinkResponse>('/api/game/engine-think', {
      method: 'POST',
      body: '{}',
    });
    setJobId(think.job_id);
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

  async function onStartGame(event: FormEvent): Promise<void> {
    event.preventDefault();
    setError(null);
    clearPolling();
    setJobId(null);
    setJobStatus(null);

    try {
      if (!checkpointId) {
        throw new Error('Please choose a checkpoint');
      }
      const payload = {
        checkpoint_id: checkpointId,
        num_simulations: Number(numSimulations),
        player_seat: playerSeat,
        ...(seed.trim().length > 0 ? { seed: Number(seed) } : {}),
      };

      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/new', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      setSnapshot(nextSnapshot);
      const status = deriveUiStatus(nextSnapshot);
      setUiStatus(status);
      if (status === 'WAITING_ENGINE') {
        await startEngineThink();
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onPlayerMove(actionIdx: number): Promise<void> {
    setError(null);
    try {
      const result = await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
        method: 'POST',
        body: JSON.stringify({ action_idx: actionIdx }),
      });
      setSnapshot(result.snapshot);
      if (result.snapshot.status !== 'IN_PROGRESS') {
        setUiStatus('GAME_OVER');
        return;
      }
      if (result.engine_should_move) {
        await startEngineThink();
      } else {
        setUiStatus('WAITING_PLAYER');
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onResign(): Promise<void> {
    setError(null);
    clearPolling();
    try {
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/resign', {
        method: 'POST',
        body: '{}',
      });
      setSnapshot(nextSnapshot);
      setUiStatus('GAME_OVER');
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const canStart = Boolean(selectedCheckpoint) && numSimulations > 0 && numSimulations <= 5000;
  const canMove = uiStatus === 'WAITING_PLAYER' && snapshot?.status === 'IN_PROGRESS';
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

  return (
    <main className="app-shell">
      <header>
        <h1>Splendor vs MCTS</h1>
      </header>

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

          <label>
            MCTS sims per move
            <input
              type="number"
              min={1}
              max={5000}
              value={numSimulations}
              onChange={(event) => setNumSimulations(Number(event.target.value))}
            />
          </label>

          <label>
            Play as
            <select value={playerSeat} onChange={(event) => setPlayerSeat(event.target.value as Seat)}>
              <option value="P0">P0 (first)</option>
              <option value="P1">P1 (second)</option>
            </select>
          </label>

          <label>
            Seed (optional)
            <input value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="Random if blank" />
          </label>

          <button type="submit" disabled={!canStart}>
            Start Game
          </button>
        </form>
      </section>

      {snapshot && (
        <section className="panel game-layout">
          <div className="board-column">
            <h2>Game Board</h2>
            {snapshot.board_state ? (
              <GameBoard board={snapshot.board_state} />
            ) : (
              <div className="empty-note">Board data unavailable</div>
            )}
            <div className="actions actions-bottom">
              <h3>Legal actions</h3>
              <ul>
                {snapshot.legal_action_details.map((action) => (
                  <li key={action.action_idx}>
                    <button disabled={!canMove} onClick={() => void onPlayerMove(action.action_idx)}>
                      <span className="action-idx-pill">{action.action_idx}</span>
                      <ActionLabel actionIdx={action.action_idx} board={snapshot.board_state} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <aside className="side-column">
            <h2>Game Controls</h2>
            <p>
              Status: <strong>{snapshot.status}</strong> | Winner: <strong>{winnerLabel(snapshot.winner)}</strong>
            </p>

            <div className="engine-box">
              <h3>Engine</h3>
              <p>UI status: {uiStatus}</p>
              {uiStatus === 'WAITING_ENGINE' && <p className="spinner">Engine thinking...</p>}
              {jobId && <p>Job ID: {jobId}</p>}
              {jobStatus && <p>Job status: {jobStatus.status}</p>}
              {jobStatus?.error && <p className="error">Engine error: {jobStatus.error}</p>}
              {uiStatus !== 'WAITING_ENGINE' &&
                snapshot.status === 'IN_PROGRESS' &&
                snapshot.player_to_move !== snapshot.config?.player_seat && (
                  <button onClick={() => void startEngineThink()}>Retry Engine Move</button>
                )}
            </div>

            <div className="controls">
              <button onClick={() => void onResign()} disabled={snapshot.status !== 'IN_PROGRESS'}>
                Resign
              </button>
            </div>

            <div className="history">
              <h3>Move History</h3>
              <ol>
                {snapshot.move_log.map((entry) => (
                  <li key={`${entry.turn_index}-${entry.action_idx}`}>
                    {entry.turn_index}. {entry.actor} {'->'} [{entry.action_idx}] {entry.label}
                  </li>
                ))}
              </ol>
            </div>
          </aside>
        </section>
      )}

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
            <input type="number" min={1} max={5000} value={selfplaySims} onChange={(e) => setSelfplaySims(Number(e.target.value))} />
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

      {replayStep && (
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
              Value target: <strong>{replayStep.value_target.toFixed(3)}</strong> | Best-root Q: <strong>{replayStep.value_root_best.toFixed(3)}</strong> | MCTS root value:{' '}
              <strong>{replayStep.value_root.toFixed(3)}</strong> | Model value:{' '}
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

      {error && <section className="panel error">Error: {error}</section>}
    </main>
  );
}
