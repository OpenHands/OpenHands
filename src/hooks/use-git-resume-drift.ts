import { useCallback, useEffect, useRef, useState } from "react";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { AgentState } from "#/types/agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

/**
 * Persisted workspace baseline, compared against the live state when
 * resuming an idle conversation. Tied to `conversationId` so each
 * conversation carries its own snapshot across resume cycles.
 */
interface GitSnapshot {
  /** HEAD commit SHA at the time of the last completed agent turn. */
  headSha: string | null;
  /** Number of working-tree changes (staged + unstaged + untracked). */
  changeCount: number;
  /** The git worktree path the snapshot was taken from — guards against
   *  accidentally matching a different checkout at the same HEAD SHA. */
  worktreeKey: string;
  /** Wall-clock time the snapshot was written. */
  capturedAt: number;
}

const SNAPSHOT_KEY = (conversationId: string) =>
  `openhands:git-snapshot:${conversationId}`;

function loadSnapshot(conversationId: string): GitSnapshot | null {
  try {
    const raw = localStorage.getItem(SNAPSHOT_KEY(conversationId));
    if (!raw) return null;
    return JSON.parse(raw) as GitSnapshot;
  } catch {
    return null;
  }
}

function saveSnapshot(conversationId: string, snap: GitSnapshot): void {
  try {
    localStorage.setItem(SNAPSHOT_KEY(conversationId), JSON.stringify(snap));
  } catch {
    // localStorage can throw on quota — silently drop, drift is advisory.
  }
}

function clearSnapshot(conversationId: string): void {
  try {
    localStorage.removeItem(SNAPSHOT_KEY(conversationId));
  } catch {
    /* no-op */
  }
}

interface GitDriftResult {
  detected: boolean;
  /** Previous HEAD SHA (the baseline the agent last worked with). */
  previousHead: string | null;
  /** Current HEAD SHA from the live runtime. */
  currentHead: string | null;
  /** Human-readable summary of what changed, or null when nothing did. */
  summary: string | null;
}

/**
 * Detect git workspace drift when resuming an idle conversation.
 *
 * On every conversation change and whenever the agent transitions back
 * into RUNNING (a resume/pause-resume or a new run), fetch the live HEAD
 * and working-tree change count via the agent-server git endpoints and
 * compare with the baseline snapshot captured after the previous turn.
 *
 * When the agent is idle again (STOPPED / PAUSED / AWAITING_USER_INPUT /
 * FINISHED / ERROR), refresh the baseline so the agent's own edits don't
 * flag as drift on the *next* resume. Conversions to MISSING (archived
 * sandbox) clear the snapshot — there is no runtime to compare against.
 *
 * Non-git workspaces, missing snapshots (first turn of a conversation),
 * and runtime errors all resolve cleanly with `detected = false`.
 */
export function useGitResumeDrift(agentState: AgentState | null): {
  drift: GitDriftResult | null;
  isChecking: boolean;
  /**
   * Force a baseline refresh from the live runtime — useful after the user
   * has acknowledged a drift warning and wants to stop seeing it.
   */
  refreshBaseline: () => void;
} {
  // useActiveConversation() returns the underlying react-query UseQueryResult,
  // so the conversation object lives on `.data`.
  const { data: conversation } = useActiveConversation();
  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;

  const [drift, setDrift] = useState<GitDriftResult | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  // Guards against stale runtime responses racing a conversation switch.
  const requestSeqRef = useRef(0);
  const conversationIdRef = useRef<string | null>(conversationId ?? null);

  const workspacePath =
    conversation?.workspace?.working_dir ??
    conversation?.selected_workspace ??
    "";

  /**
   * Pull live git state from the runtime and return a fresh snapshot, or
   * null when the runtime is missing / not a git repo / errors out.
   */
  const fetchLiveSnapshot =
    useCallback(async (): Promise<GitSnapshot | null> => {
      if (!conversationId || !conversationUrl || !sessionApiKey) return null;

      try {
        const [commits, changes] = await Promise.all([
          AgentServerGitService.getGitCommits(
            conversationUrl,
            sessionApiKey,
            workspacePath || ".",
            1,
          ),
          AgentServerGitService.getGitChanges(
            conversationId,
            conversationUrl,
            sessionApiKey,
            workspacePath || ".",
          ),
        ]);

        const headSha = commits?.commits?.[0]?.sha ?? null;
        const changeCount = Array.isArray(changes) ? changes.length : 0;
        if (!headSha && changeCount === 0) {
          // Not a git repo — baseline is meaningless here.
          return null;
        }
        return {
          headSha,
          changeCount,
          worktreeKey: workspacePath || ".",
          capturedAt: Date.now(),
        };
      } catch {
        // Runtime unreachable, CORS, or non-git workspace. Treat as "no drift".
        return null;
      }
    }, [conversationId, conversationUrl, sessionApiKey, workspacePath]);

  /**
   * Compare a fresh live snapshot against the stored baseline.
   */
  const computeDrift = useCallback(
    (
      live: GitSnapshot | null,
      baseline: GitSnapshot | null,
    ): GitDriftResult => {
      if (!live) {
        // No runtime / not a git repo — nothing to warn about.
        return {
          detected: false,
          previousHead: null,
          currentHead: null,
          summary: null,
        };
      }
      if (!baseline) {
        // First turn of this conversation — establish baseline, no warning.
        return {
          detected: false,
          previousHead: null,
          currentHead: live.headSha,
          summary: null,
        };
      }

      const headChanged =
        baseline.headSha != null &&
        live.headSha != null &&
        baseline.headSha !== live.headSha;
      const workingTreeChanged = baseline.changeCount !== live.changeCount;

      if (!headChanged && !workingTreeChanged) {
        return {
          detected: false,
          previousHead: baseline.headSha,
          currentHead: live.headSha,
          summary: null,
        };
      }

      const parts: string[] = [];
      if (headChanged) {
        parts.push(
          `HEAD ${baseline.headSha?.slice(0, 8) ?? "(unknown)"} → ${live.headSha?.slice(0, 8) ?? "(unknown)"}`,
        );
      }
      if (workingTreeChanged) {
        parts.push(
          `working tree changes: ${baseline.changeCount} → ${live.changeCount}`,
        );
      }
      return {
        detected: true,
        previousHead: baseline.headSha,
        currentHead: live.headSha,
        summary: parts.join(" · "),
      };
    },
    [],
  );

  // --- Lifecycle effects ---

  // Conversation switched: clear any stale drift state and kick off an
  // initial baseline refresh so we don't flag a new conversation as drifted.
  useEffect(() => {
    if (conversationIdRef.current === conversationId) return;
    conversationIdRef.current = conversationId ?? null;
    setDrift(null);

    if (!conversationId) return;

    // If there's no snapshot yet, establish one silently — no warning on
    // first resume. If there *is* a snapshot, we're about to re-run the
    // comparison below from the RUNNING transition.
    void (async () => {
      const live = await fetchLiveSnapshot();
      if (live && !loadSnapshot(conversationId)) {
        saveSnapshot(conversationId, live);
      }
    })();
  }, [conversationId, fetchLiveSnapshot]);

  // Agent transitioned to RUNNING — resume just happened. Compare live
  // against stored baseline to surface any drift that accumulated while
  // the agent was idle.
  useEffect(() => {
    if (!conversationId || agentState !== AgentState.RUNNING) return;

    const seq = ++requestSeqRef.current;
    setIsChecking(true);
    void (async () => {
      const live = await fetchLiveSnapshot();
      if (seq !== requestSeqRef.current) return; // conversation switched mid-flight
      const baseline = loadSnapshot(conversationId);
      const result = computeDrift(live, baseline);
      setDrift(result);
      setIsChecking(false);
    })();
  }, [conversationId, agentState, fetchLiveSnapshot, computeDrift]);

  // Agent went idle again — refresh the baseline so the agent's own edits
  // don't flag as drift the next time it resumes.
  useEffect(() => {
    if (!conversationId) return;
    const idleStates: AgentState[] = [
      AgentState.STOPPED,
      AgentState.PAUSED,
      AgentState.AWAITING_USER_INPUT,
      AgentState.FINISHED,
      AgentState.ERROR,
    ];
    if (!agentState || !idleStates.includes(agentState)) return;

    void (async () => {
      const live = await fetchLiveSnapshot();
      if (live) {
        saveSnapshot(conversationId, live);
      }
    })();
  }, [conversationId, agentState, fetchLiveSnapshot]);

  // Archived / missing runtime — snapshot is meaningless, clear it.
  useEffect(() => {
    if (!conversationId) return;
    if (agentState === null) {
      clearSnapshot(conversationId);
      setDrift(null);
    }
  }, [conversationId, agentState]);

  const refreshBaseline = useCallback(() => {
    if (!conversationId) return;
    void (async () => {
      const live = await fetchLiveSnapshot();
      if (live) {
        saveSnapshot(conversationId, live);
        setDrift({
          detected: false,
          previousHead: live.headSha,
          currentHead: live.headSha,
          summary: null,
        });
      }
    })();
  }, [conversationId, fetchLiveSnapshot]);

  return { drift, isChecking, refreshBaseline };
}
