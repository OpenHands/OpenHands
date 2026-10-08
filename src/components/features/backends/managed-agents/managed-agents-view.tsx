import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ChevronLeft, Loader2, Plus, RefreshCw, Search } from "lucide-react";

import DigitalOceanLogo from "#/assets/branding/digitalocean-logo.svg?react";
import {
  buildNewSessionName,
  getMarsBridge,
  getMarsErrorMessage,
  getSessionPhase,
  type MarsAgentConfig,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import type { Backend } from "#/api/backend-registry/types";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  useMarsAgents,
  useMarsAuthState,
  type MarsAgentGroup,
} from "#/hooks/query/use-mars-agents";
import { useConnectMarsSession } from "#/hooks/use-connect-mars-session";
import { useMarsTunnelBackend } from "#/hooks/use-mars-tunnel-backend";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { ManagedAgentsSignIn } from "./managed-agents-sign-in";
import { ManagedAgentsSessionRow } from "./managed-agents-session-row";
import { agentLabel, buildMarsBackendName } from "./managed-agents-labels";

type StatusFilter = "all" | "ready" | "paused";

const STATUS_FILTERS: { value: StatusFilter; label: I18nKey }[] = [
  { value: "all", label: I18nKey.DO_AGENTS$FILTER_ALL },
  { value: "ready", label: I18nKey.DO_AGENTS$STATUS_READY },
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
  const bridge = getMarsBridge();
  const queryClient = useQueryClient();
  const { backends } = useActiveBackendContext();
  const { detach } = useMarsTunnelBackend();
  const {
    connect: connectSession,
    connecting,
    describeError,
  } = useConnectMarsSession();

  const [query, setQuery] = React.useState("");
  const [statusFilter, setStatusFilter] = React.useState<StatusFilter>("all");
  const [authError, setAuthError] = React.useState<string | null>(null);
  const [isAddingConnection, setIsAddingConnection] = React.useState(false);
  const [busySessionId, setBusySessionId] = React.useState<string | null>(null);
  const [creatingConfigId, setCreatingConfigId] = React.useState<string | null>(
    null,
  );
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  // One launch at a time: overlapping connects reset each other's progress
  // and navigate twice, and a create can take up to a minute.
  const isLaunching = connecting !== null || creatingConfigId !== null;

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

  const onAuthChanged = () => {
    setIsAddingConnection(false);
    setAuthError(null);
    void queryClient.invalidateQueries({ queryKey: MARS_QUERY_KEYS.all });
  };
  const onAuthFailed = (error: unknown) =>
    setAuthError(
      getMarsErrorMessage(error) ?? t(I18nKey.BACKEND$DIGITALOCEAN_AUTH_FAILED),
    );

  const signIn = useMutation({
    mutationFn: () => bridge!.signInWithOAuth(),
    onSuccess: onAuthChanged,
    onError: onAuthFailed,
  });
  const saveToken = useMutation({
    mutationFn: (token: string) => bridge!.savePat({ token }),
    onSuccess: onAuthChanged,
    onError: onAuthFailed,
  });
  const switchConnection = useMutation({
    mutationFn: (id: string) => bridge!.setActiveConnection(id),
    onSuccess: onAuthChanged,
    onError: onAuthFailed,
  });
  const signOut = useMutation({
    mutationFn: () => bridge!.signOut(),
    onSuccess: onAuthChanged,
    onError: onAuthFailed,
  });

  const refreshAgents = () =>
    queryClient.invalidateQueries({
      queryKey: MARS_QUERY_KEYS.agents(activeConnection?.id ?? null),
    });

  const connect = async (session: MarsSession, config?: MarsAgentConfig) => {
    setError(session.session_id, null);
    try {
      await connectSession({
        session,
        name: buildMarsBackendName(session, config, t),
        configId: config?.id,
      });
      void refreshAgents();
      onDone();
    } catch (error) {
      setError(session.session_id, describeError(error));
    }
  };

  const launchSession = async (config: MarsAgentConfig) => {
    setError(config.id, null);
    setCreatingConfigId(config.id);
    let session: MarsSession;
    try {
      session = await bridge!.createSession(
        config.id,
        buildNewSessionName(config.name),
      );
    } catch (error) {
      setError(
        config.id,
        getMarsErrorMessage(error) ??
          t(I18nKey.BACKEND$DIGITALOCEAN_SESSION_CREATE_FAILED),
      );
      return;
    } finally {
      setCreatingConfigId(null);
    }
    void refreshAgents();
    await connect(session, config);
  };

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

  const data = agentsQuery.data;
  const visibleGroups = (data?.groups ?? [])
    .map((group): MarsAgentGroup => {
      const agentMatches =
        !!normalizedQuery &&
        matchesQuery(agentLabel(group.config), normalizedQuery);
      return {
        config: group.config,
        sessions: filterSessions(group.sessions, agentMatches),
      };
    })
    .filter(
      (group) =>
        group.sessions.length > 0 ||
        (statusFilter === "all" &&
          (!normalizedQuery ||
            matchesQuery(agentLabel(group.config), normalizedQuery))),
    );
  const visibleOther = filterSessions(data?.otherSessions ?? [], false);
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
        connectStage={
          connecting?.sessionId === session.session_id ? connecting.stage : null
        }
        isBusy={isLaunching || busySessionId === session.session_id}
        error={errors[session.session_id] ?? null}
        onConnect={() => void connect(session, config)}
        // The registry reuses a live tunnel, so this is a fast re-probe that
        // also lands in the session's latest chat — or revives a dead tunnel.
        onOpen={() => void connect(session, config)}
        onDisconnect={() => {
          if (backend) {
            void runSessionAction(session.session_id, () => detach(backend));
          }
        }}
        onPause={() =>
          void runSessionAction(session.session_id, () =>
            bridge!.pauseSession(session.session_id),
          )
        }
      />
    );
  };

  let body: React.ReactNode;
  if (authQuery.isPending) {
    body = (
      <div className="flex justify-center py-10">
        <Loader2
          className="size-5 animate-spin text-[var(--oh-muted)]"
          aria-hidden
        />
      </div>
    );
  } else if (!isSignedIn) {
    body = (
      <ManagedAgentsSignIn
        canUseOAuth={authState?.canUseOAuth ?? false}
        isExpired={!isAddingConnection && activeConnection?.isExpired === true}
        isSigningIn={signIn.isPending}
        isSavingToken={saveToken.isPending}
        error={authError}
        onSignInWithOAuth={() => signIn.mutate()}
        onSaveToken={(token) => saveToken.mutate(token)}
        onCancel={
          isAddingConnection
            ? () => {
                setIsAddingConnection(false);
                setAuthError(null);
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
              className="pointer-events-none absolute left-2.5 size-4 text-[var(--oh-muted)]"
              aria-hidden
            />
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t(I18nKey.DO_AGENTS$SEARCH_PLACEHOLDER)}
              aria-label={t(I18nKey.DO_AGENTS$SEARCH_PLACEHOLDER)}
              data-testid="managed-agents-search"
              className="h-9 w-full rounded-md border border-[var(--oh-border)] bg-transparent pl-8 pr-3 text-sm text-white placeholder:text-[var(--oh-muted)] focus:border-[var(--oh-border-strong,var(--oh-border))] focus:outline-none"
            />
          </label>
          <div
            role="radiogroup"
            className="flex rounded-md border border-[var(--oh-border)] p-0.5"
          >
            {STATUS_FILTERS.map((filter) => (
              <button
                key={filter.value}
                type="button"
                role="radio"
                aria-checked={statusFilter === filter.value}
                onClick={() => setStatusFilter(filter.value)}
                data-testid={`managed-agents-filter-${filter.value}`}
                className={cn(
                  "cursor-pointer rounded px-2.5 py-1 text-xs transition-colors",
                  statusFilter === filter.value
                    ? "bg-[var(--oh-interactive-hover)] text-white"
                    : "text-[var(--oh-muted)] hover:text-white",
                )}
              >
                {t(filter.label)}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => void refreshAgents()}
            aria-label={t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_REFRESH)}
            title={t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_REFRESH)}
            data-testid="managed-agents-refresh"
            className="inline-flex size-9 cursor-pointer items-center justify-center rounded-md text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-white"
          >
            <RefreshCw
              className={cn("size-4", agentsQuery.isFetching && "animate-spin")}
              aria-hidden
            />
          </button>
        </div>

        {agentsQuery.error ? (
          <div
            role="alert"
            data-testid="managed-agents-load-error"
            className="rounded-md border border-[var(--oh-status-error)]/40 bg-[var(--oh-status-error)]/10 p-3 text-sm text-[var(--oh-status-error)] whitespace-pre-wrap break-words"
          >
            {getMarsErrorMessage(agentsQuery.error) ??
              t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_FAILED)}
          </div>
        ) : null}

        <div
          className="min-h-0 flex-1 overflow-y-auto custom-scrollbar-always"
          data-testid="managed-agents-list"
        >
          {agentsQuery.isPending ? (
            <p className="py-6 text-center text-sm text-[var(--oh-muted)]">
              {t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_LOADING)}
            </p>
          ) : null}

          {data && !hasAnyAgentsOrSessions ? (
            <p
              data-testid="managed-agents-empty"
              className="py-6 text-center text-sm text-[var(--oh-muted)]"
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
              className="py-6 text-center text-sm text-[var(--oh-muted)]"
            >
              {t(I18nKey.DO_AGENTS$NO_MATCHES)}
            </p>
          ) : null}

          <div className="flex flex-col gap-3">
            {visibleGroups.map(({ config, sessions }) => {
              const totalSessions =
                data?.groups.find((g) => g.config.id === config.id)?.sessions
                  .length ?? 0;
              const isCreating = creatingConfigId === config.id;
              return (
                <section
                  key={config.id}
                  data-testid={`managed-agents-agent-${config.id}`}
                  className="overflow-hidden rounded-lg border border-[var(--oh-border)] bg-surface-raised"
                >
                  <header className="flex items-center gap-3 border-b border-[var(--oh-border)] px-3 py-2">
                    <div className="flex min-w-0 flex-1 items-baseline gap-2">
                      <h4 className="truncate text-sm font-medium text-white">
                        {agentLabel(config)}
                      </h4>
                      <span className="shrink-0 text-xs text-[var(--oh-muted)]">
                        {t(I18nKey.DO_AGENTS$SESSION_COUNT, {
                          count: totalSessions,
                        })}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => void launchSession(config)}
                      disabled={isLaunching}
                      data-testid={`managed-agents-new-session-${config.id}`}
                      className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-xs text-primary transition-colors hover:bg-[var(--oh-interactive-hover)] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {isCreating ? (
                        <Loader2
                          className="size-3.5 animate-spin"
                          aria-hidden
                        />
                      ) : (
                        <Plus className="size-3.5" aria-hidden />
                      )}
                      {isCreating
                        ? t(I18nKey.BACKEND$DIGITALOCEAN_CREATING_SESSION)
                        : t(I18nKey.DO_AGENTS$NEW_SESSION)}
                    </button>
                  </header>
                  {errors[config.id] ? (
                    <p
                      role="alert"
                      className="border-b border-[var(--oh-border)] px-3 py-2 text-xs text-[var(--oh-status-error)]"
                    >
                      {errors[config.id]}
                    </p>
                  ) : null}
                  {sessions.length > 0 ? (
                    <ul className="divide-y divide-[var(--oh-border)]">
                      {sessions.map((session) =>
                        renderSession(session, config),
                      )}
                    </ul>
                  ) : (
                    <p className="px-3 py-3 text-xs text-[var(--oh-muted)]">
                      {t(I18nKey.DO_AGENTS$NO_SESSIONS)}
                    </p>
                  )}
                </section>
              );
            })}

            {visibleOther.length > 0 ? (
              <section
                data-testid="managed-agents-other-sessions"
                className="overflow-hidden rounded-lg border border-[var(--oh-border)] bg-surface-raised"
              >
                <header className="border-b border-[var(--oh-border)] px-3 py-2">
                  <h4 className="text-sm font-medium text-white">
                    {t(I18nKey.DO_AGENTS$OTHER_SESSIONS)}
                  </h4>
                </header>
                <ul className="divide-y divide-[var(--oh-border)]">
                  {visibleOther.map((session) => renderSession(session))}
                </ul>
              </section>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  const connections = authState?.connections ?? [];

  return (
    <div
      data-testid="managed-agents-view"
      className="flex min-h-0 flex-1 flex-col"
    >
      <div className="flex flex-col gap-3 p-5 pb-4 pr-12">
        <button
          type="button"
          onClick={onBack}
          data-testid="managed-agents-back"
          className="inline-flex w-fit cursor-pointer items-center gap-1 text-xs text-[var(--oh-muted)] transition-colors hover:text-white"
        >
          <ChevronLeft className="size-4" aria-hidden />
          {t(I18nKey.DO_AGENTS$BACK)}
        </button>
        <div className="flex items-center gap-3">
          <DigitalOceanLogo width={28} height={28} aria-hidden />
          <div className="flex min-w-0 flex-col">
            <h2 className="text-lg font-medium text-white">
              {t(I18nKey.DO_AGENTS$TITLE)}
            </h2>
            <p className="text-xs text-[var(--oh-muted)]">
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
          <div
            data-testid="managed-agents-account"
            className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-[var(--oh-border)] px-3 py-2 text-xs"
          >
            {connections.length > 1 ? (
              <label className="flex min-w-0 flex-1 items-center gap-2 text-[var(--oh-muted)]">
                <span className="shrink-0">
                  {t(I18nKey.BACKEND$DIGITALOCEAN_SWITCH_CONNECTION)}
                </span>
                <select
                  value={activeConnection.id}
                  onChange={(event) =>
                    switchConnection.mutate(event.target.value)
                  }
                  data-testid="managed-agents-connection-switcher"
                  className="min-w-0 flex-1 cursor-pointer rounded border border-[var(--oh-border)] bg-transparent px-1.5 py-0.5 text-white"
                >
                  {connections.map((connection) => (
                    <option key={connection.id} value={connection.id}>
                      {connection.teamName ?? connection.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <span className="flex min-w-0 flex-1 items-center gap-2 truncate text-white">
                <span
                  aria-hidden
                  className="size-1.5 rounded-full bg-[var(--oh-status-success)]"
                />
                {activeConnection.teamName ?? activeConnection.label}
              </span>
            )}
            <button
              type="button"
              onClick={() => setIsAddingConnection(true)}
              data-testid="managed-agents-add-connection"
              className="cursor-pointer text-primary hover:underline"
            >
              {t(I18nKey.BACKEND$DIGITALOCEAN_ADD_CONNECTION)}
            </button>
            <button
              type="button"
              onClick={() => signOut.mutate()}
              data-testid="managed-agents-sign-out"
              className="cursor-pointer text-[var(--oh-muted)] hover:text-white"
            >
              {t(I18nKey.BACKEND$DIGITALOCEAN_SIGN_OUT)}
            </button>
            {authState?.isPersistent === false ? (
              <p className="w-full text-[var(--oh-warning)]">
                {t(I18nKey.BACKEND$DIGITALOCEAN_NOT_PERSISTENT)}
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="flex min-h-0 flex-1 flex-col px-5 pb-5">{body}</div>
    </div>
  );
}
