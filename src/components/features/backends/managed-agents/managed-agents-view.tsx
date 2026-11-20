import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Loader2, RefreshCw, Search } from "lucide-react";

import DigitalOceanLogo from "#/assets/branding/digitalocean-logo.svg?react";
import {
  getMarsErrorMessage,
  getSessionPhase,
  requireMarsBridge,
  type MarsAgentConfig,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import type { Backend } from "#/api/backend-registry/types";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { ErrorAlert } from "#/components/shared/error-alert";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  useMarsAgents,
  useMarsAuthState,
  type MarsAgentsSnapshot,
} from "#/hooks/query/use-mars-agents";
import {
  MarsLaunchConnectError,
  useLaunchMarsSession,
} from "#/hooks/use-launch-mars-session";
import { useMarsSignIn } from "#/hooks/use-mars-sign-in";
import { useMarsTunnelBackend } from "#/hooks/use-mars-tunnel-backend";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { AccountBar } from "./account-bar";
import {
  AgentGroupSection,
  type VisibleAgentGroup,
} from "./agent-group-section";
import { ManagedAgentsSignIn } from "./managed-agents-sign-in";
import { ManagedAgentsSessionRow } from "./managed-agents-session-row";
import { agentLabel, getSessionDisplayName } from "./managed-agents-labels";
import { getConnectStage } from "./use-session-meta";

/** A destructive action awaiting the user's confirmation. */
type PendingAction =
  | { kind: "stop"; session: MarsSession }
  | { kind: "delete-agent"; config: MarsAgentConfig };

/** A session that is still around to be stopped (or must be, before its agent goes). */
function isLiveSession(session: MarsSession): boolean {
  return getSessionPhase(session.status) !== "ended";
}

type StatusFilter = "all" | "ready" | "paused";

const STATUS_FILTERS: { value: StatusFilter; label: I18nKey }[] = [
  { value: "all", label: I18nKey.DO_AGENTS$FILTER_ALL },
  { value: "ready", label: I18nKey.COMMON$RUNNING },
  { value: "paused", label: I18nKey.DO_AGENTS$STATUS_PAUSED },
];

interface ManagedAgentsViewProps {
  onBack: () => void;
  /** Leave the backends modal entirely — used once a session is opened. */
  onDone: () => void;
}

function matchesQuery(text: string | null | undefined, query: string) {
  return (text ?? "").toLowerCase().includes(query);
}

/**
 * Groups and other sessions left after the search and status filter. Each
 * group keeps the counts of its unfiltered sessions for the header and for
 * gating Delete agent.
 */
function filterAgents(
  data: MarsAgentsSnapshot | undefined,
  query: string,
  statusFilter: StatusFilter,
): { groups: VisibleAgentGroup[]; otherSessions: MarsSession[] } {
  const normalizedQuery = query.trim().toLowerCase();
  const filterSessions = (sessions: MarsSession[], agentMatches: boolean) =>
    sessions.filter((session) => {
      const phase = getSessionPhase(session.status);
      if (statusFilter !== "all" && phase !== statusFilter) return false;
      if (!normalizedQuery || agentMatches) return true;
      return (
        matchesQuery(session.name, normalizedQuery) ||
        matchesQuery(session.session_id, normalizedQuery)
      );
    });

  const groups = (data?.groups ?? [])
    .map((group): VisibleAgentGroup & { agentMatches: boolean } => {
      const agentMatches =
        !normalizedQuery ||
        matchesQuery(agentLabel(group.config), normalizedQuery);
      return {
        config: group.config,
        sessions: filterSessions(
          group.sessions,
          !!normalizedQuery && agentMatches,
        ),
        totalSessions: group.sessions.length,
        liveSessions: group.sessions.filter(isLiveSession).length,
        agentMatches,
      };
    })
    .filter(
      (group) =>
        group.sessions.length > 0 ||
        (statusFilter === "all" && group.agentMatches),
    )
    .map(({ agentMatches, ...group }) => group);

  return {
    groups,
    otherSessions: filterSessions(data?.otherSessions ?? [], false),
  };
}

/**
 * DigitalOcean Managed Agents: sign in once, then see every agent and session
 * on the team with live status, and connect any of them as a backend.
 *
 * Connecting opens a port-forward tunnel in the Electron main process and
 * registers `http://127.0.0.1:<port>` as an ordinary local backend — after
 * the tunnel the far end genuinely is an agent-server, so conversations,
 * files, and the terminal all work unmodified.
 */
export function ManagedAgentsView({ onBack, onDone }: ManagedAgentsViewProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backends } = useActiveBackendContext();
  const { detach } = useMarsTunnelBackend();
  const launcher = useLaunchMarsSession();
  const [isAddingConnection, setIsAddingConnection] = React.useState(false);
  const auth = useMarsSignIn({
    onAuthChanged: () => setIsAddingConnection(false),
  });

  const [query, setQuery] = React.useState("");
  const [statusFilter, setStatusFilter] = React.useState<StatusFilter>("all");
  const [busySessionId, setBusySessionId] = React.useState<string | null>(null);
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const [pendingAction, setPendingAction] =
    React.useState<PendingAction | null>(null);
  const [isConfirming, setIsConfirming] = React.useState(false);
  // One launch at a time: overlapping connects reset each other's progress
  // and navigate twice, and a create can take up to a minute.
  const isLaunching = launcher.isBusy;

  const setError = (key: string, message: string | null) =>
    setErrors((previous) => {
      const next = { ...previous };
      if (message) next[key] = message;
      else delete next[key];
      return next;
    });

  const authQuery = useMarsAuthState();
  const authState = authQuery.data;
  const activeConnection = authState?.active ?? null;
  const isSignedIn =
    activeConnection !== null &&
    !activeConnection.isExpired &&
    !isAddingConnection;

  const agentsQuery = useMarsAgents(isSignedIn ? activeConnection.id : null);

  const connectedBySession = React.useMemo(() => {
    const map = new Map<string, Backend>();
    for (const backend of backends) {
      if (backend.marsSessionId) map.set(backend.marsSessionId, backend);
    }
    return map;
  }, [backends]);

  const refreshAgents = () =>
    queryClient.invalidateQueries({
      queryKey: MARS_QUERY_KEYS.agents(activeConnection?.id ?? null),
    });

  /** Run a connect or launch, then leave the modal for the session. */
  const runLaunch = async (
    errorKey: string,
    action: () => Promise<unknown>,
  ) => {
    setError(errorKey, null);
    try {
      await action();
      void refreshAgents();
      onDone();
    } catch (error) {
      // A launch that created its session but failed to connect reports on
      // that session's row, where a later connect clears it.
      setError(
        error instanceof MarsLaunchConnectError ? error.sessionId : errorKey,
        launcher.describeError(error),
      );
    }
  };

  const openSession = (session: MarsSession, config?: MarsAgentConfig) =>
    runLaunch(session.session_id, () => launcher.open(session, config));

  const launchSession = (config: MarsAgentConfig) =>
    runLaunch(config.id, () => launcher.launch(config));

  const runSessionAction = async (
    sessionId: string,
    action: () => Promise<unknown>,
  ) => {
    setError(sessionId, null);
    setBusySessionId(sessionId);
    try {
      await action();
      await refreshAgents();
    } catch (error) {
      setError(
        sessionId,
        getMarsErrorMessage(error) ??
          t(I18nKey.DO_AGENTS$SESSION_ACTION_FAILED),
      );
    } finally {
      setBusySessionId(null);
    }
  };

  // Stop = destroy: harness-api tears the sandbox down and the session is
  // gone for good, so it sits behind a confirmation. The destroy goes first:
  // it can be refused (409 while a checkpoint/fork/rollback holds the
  // session, 423 locked), and then the session is still running and the
  // user must keep their connection to it. Only once it succeeded is a
  // connected backend detached, so nothing keeps probing a torn-down sandbox.
  const stopSession = async (session: MarsSession) => {
    const backend = connectedBySession.get(session.session_id);
    await runSessionAction(session.session_id, async () => {
      await requireMarsBridge().destroySession(session.session_id);
      if (backend) await detach(backend);
    });
  };

  // harness-api soft-deletes a config without looking at its sessions, so
  // the UI only offers it once none is live; otherwise the sessions would
  // silently drop out of their group while still running (and billing).
  const deleteAgent = async (config: MarsAgentConfig) => {
    setError(config.id, null);
    try {
      await requireMarsBridge().deleteAgentConfig(config.id);
      await refreshAgents();
    } catch (error) {
      setError(
        config.id,
        getMarsErrorMessage(error) ??
          t(I18nKey.DO_AGENTS$SESSION_ACTION_FAILED),
      );
    }
  };

  const confirmPendingAction = async () => {
    if (!pendingAction) return;
    setIsConfirming(true);
    try {
      if (pendingAction.kind === "stop")
        await stopSession(pendingAction.session);
      else await deleteAgent(pendingAction.config);
    } finally {
      setIsConfirming(false);
      setPendingAction(null);
    }
  };

  const data = agentsQuery.data;
  const { groups: visibleGroups, otherSessions: visibleOther } = React.useMemo(
    () => filterAgents(data, query, statusFilter),
    [data, query, statusFilter],
  );
  const hasAnyAgentsOrSessions =
    (data?.groups.length ?? 0) > 0 || (data?.otherSessions.length ?? 0) > 0;
  const connectedCount = connectedBySession.size;

  const renderSession = (session: MarsSession, config?: MarsAgentConfig) => {
    const backend = connectedBySession.get(session.session_id);
    return (
      <ManagedAgentsSessionRow
        key={session.session_id}
        session={session}
        isConnected={backend !== undefined}
        connectStage={getConnectStage(launcher.connecting, session)}
        isBusy={isLaunching || busySessionId === session.session_id}
        error={errors[session.session_id] ?? null}
        onConnect={() => void openSession(session, config)}
        // The registry reuses a live tunnel, so this is a fast re-probe that
        // also lands in the session's latest chat — or revives a dead tunnel.
        onOpen={() => void openSession(session, config)}
        onDisconnect={() => {
          if (backend) {
            void runSessionAction(session.session_id, () => detach(backend));
          }
        }}
        onPause={() =>
          void runSessionAction(session.session_id, () =>
            requireMarsBridge().pauseSession(session.session_id),
          )
        }
        onStop={() => setPendingAction({ kind: "stop", session })}
      />
    );
  };

  let body: React.ReactNode;
  if (authQuery.isPending) {
    body = (
      <div className="flex justify-center py-10">
        <Loader2 className="size-5 animate-spin text-muted" aria-hidden />
      </div>
    );
  } else if (!isSignedIn) {
    body = (
      <ManagedAgentsSignIn
        canUseOAuth={authState?.canUseOAuth ?? false}
        isExpired={!isAddingConnection && activeConnection?.isExpired === true}
        isSigningIn={auth.isSigningIn}
        isSavingToken={auth.isSavingToken}
        error={auth.error}
        onSignInWithOAuth={auth.signInWithOAuth}
        onSaveToken={auth.saveToken}
        onCancel={
          isAddingConnection
            ? () => {
                setIsAddingConnection(false);
                auth.clearError();
              }
            : undefined
        }
      />
    );
  } else {
    body = (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <label className="relative flex min-w-48 flex-1 items-center">
            <Search
              className="pointer-events-none absolute left-2.5 size-4 text-muted"
              aria-hidden
            />
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t(I18nKey.DO_AGENTS$SEARCH_PLACEHOLDER)}
              aria-label={t(I18nKey.DO_AGENTS$SEARCH_PLACEHOLDER)}
              data-testid="managed-agents-search"
              className="h-9 w-full rounded-md border border-border bg-transparent pl-8 pr-3 text-sm text-white placeholder:text-muted focus:outline-none"
            />
          </label>
          <SegmentedToggle<StatusFilter>
            value={statusFilter}
            options={STATUS_FILTERS.map((filter) => ({
              value: filter.value,
              label: t(filter.label),
            }))}
            onChange={setStatusFilter}
            ariaLabel={t(I18nKey.COMMON$STATUS)}
            testId="managed-agents-filter"
          />
          <button
            type="button"
            onClick={() => void refreshAgents()}
            aria-label={t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_REFRESH)}
            title={t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_REFRESH)}
            data-testid="managed-agents-refresh"
            className="inline-flex size-9 cursor-pointer items-center justify-center rounded-md text-muted transition-colors hover:bg-interactive-hover hover:text-white"
          >
            <RefreshCw
              className={cn("size-4", agentsQuery.isFetching && "animate-spin")}
              aria-hidden
            />
          </button>
        </div>

        {agentsQuery.error ? (
          <ErrorAlert testId="managed-agents-load-error">
            {getMarsErrorMessage(agentsQuery.error) ??
              t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_FAILED)}
          </ErrorAlert>
        ) : null}

        <div
          className="min-h-0 flex-1 overflow-y-auto custom-scrollbar-always"
          data-testid="managed-agents-list"
        >
          {agentsQuery.isPending ? (
            <p className="py-6 text-center text-sm text-muted">
              {t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_LOADING)}
            </p>
          ) : null}

          {data && !hasAnyAgentsOrSessions ? (
            <p
              data-testid="managed-agents-empty"
              className="py-6 text-center text-sm text-muted"
            >
              {t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_EMPTY)}
            </p>
          ) : null}

          {data &&
          hasAnyAgentsOrSessions &&
          visibleGroups.length === 0 &&
          visibleOther.length === 0 ? (
            <p
              data-testid="managed-agents-no-matches"
              className="py-6 text-center text-sm text-muted"
            >
              {t(I18nKey.DO_AGENTS$NO_MATCHES)}
            </p>
          ) : null}

          <div className="flex flex-col gap-3">
            {visibleGroups.map((group) => (
              <AgentGroupSection
                key={group.config.id}
                group={group}
                isCreating={launcher.launchingConfigId === group.config.id}
                isLaunching={isLaunching}
                error={errors[group.config.id] ?? null}
                onNewSession={() => void launchSession(group.config)}
                onDeleteAgent={() =>
                  setPendingAction({
                    kind: "delete-agent",
                    config: group.config,
                  })
                }
              >
                {group.sessions.map((session) =>
                  renderSession(session, group.config),
                )}
              </AgentGroupSection>
            ))}

            {visibleOther.length > 0 ? (
              <section
                data-testid="managed-agents-other-sessions"
                className="overflow-hidden rounded-lg border border-border bg-surface-raised"
              >
                <header className="border-b border-border px-3 py-2">
                  <h4 className="text-sm font-medium text-white">
                    {t(I18nKey.DO_AGENTS$OTHER_SESSIONS)}
                  </h4>
                </header>
                <ul className="divide-y divide-border">
                  {visibleOther.map((session) => renderSession(session))}
                </ul>
              </section>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid="managed-agents-view"
      className="flex min-h-0 flex-1 flex-col"
    >
      <div className="flex flex-col gap-3 p-5 pb-4 pr-12">
        <BackNavButton onClick={onBack} testId="managed-agents-back">
          {t(I18nKey.DO_AGENTS$BACK)}
        </BackNavButton>
        <div className="flex items-center gap-3">
          <DigitalOceanLogo width={28} height={28} aria-hidden />
          <div className="flex min-w-0 flex-col">
            <h2 className="text-lg font-medium text-white">
              {t(I18nKey.DO_AGENTS$TITLE)}
            </h2>
            <p className="text-xs text-muted">
              {t(I18nKey.DO_AGENTS$ENTRY_DESCRIPTION)}
            </p>
          </div>
          {isSignedIn && connectedCount > 0 ? (
            <span className="ml-auto shrink-0 rounded-full border border-primary/40 px-2 py-0.5 text-xs text-primary">
              {t(I18nKey.DO_AGENTS$CONNECTED_COUNT, { count: connectedCount })}
            </span>
          ) : null}
        </div>

        {isSignedIn ? (
          <AccountBar
            authState={authState}
            activeConnection={activeConnection}
            onSwitchConnection={auth.switchConnection}
            onAddConnection={() => setIsAddingConnection(true)}
            onSignOut={auth.signOut}
          />
        ) : null}
      </div>

      <div className="flex min-h-0 flex-1 flex-col px-5 pb-5">{body}</div>

      {pendingAction ? (
        <ConfirmationModal
          text={
            pendingAction.kind === "stop"
              ? t(I18nKey.DO_AGENTS$STOP_CONFIRM, {
                  name: getSessionDisplayName(pendingAction.session, t),
                })
              : t(I18nKey.DO_AGENTS$DELETE_AGENT_CONFIRM, {
                  name: agentLabel(pendingAction.config),
                })
          }
          confirmText={
            pendingAction.kind === "stop"
              ? t(I18nKey.DO_AGENTS$STOP)
              : t(I18nKey.DO_AGENTS$DELETE_AGENT)
          }
          isConfirming={isConfirming}
          onConfirm={() => void confirmPendingAction()}
          onCancel={() => setPendingAction(null)}
        />
      ) : null}
    </div>
  );
}
