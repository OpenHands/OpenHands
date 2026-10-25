import React from "react";
import { useTranslation } from "react-i18next";
import { ChevronRight, Loader2, Plus, RefreshCw } from "lucide-react";

import {
  getMarsErrorMessage,
  getSessionPhase,
  isConnectableSession,
  type MarsAgentConfig,
  type MarsSession,
  type NewMarsAgentInput,
} from "#/api/mars/mars-tunnel-backend";
import {
  getUsableConnectionId,
  useMarsAgents,
  useMarsAuthState,
  type MarsAgentGroup,
} from "#/hooks/query/use-mars-agents";
import { useLaunchMarsSession } from "#/hooks/use-launch-mars-session";
import { useMarsSignIn } from "#/hooks/use-mars-sign-in";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { DigitalOceanNewAgentForm } from "./digitalocean-new-agent-form";
import {
  agentLabel,
  connectionLabel,
  SESSION_COUNT_I18N_KEY,
  STAGE_LABEL_KEY,
} from "./managed-agents-labels";
import { ManagedAgentsSignIn } from "./managed-agents-sign-in";
import { SessionStatusDot } from "./session-status-dot";
import { SignInStorageNote } from "./sign-in-storage-note";
import {
  getConnectStage,
  useSessionActivityLabel,
  useSessionMeta,
} from "./use-session-meta";

const PANEL_CLASS =
  "flex min-h-[13.5rem] w-full flex-col rounded-xl border border-border";
const ICON_BUTTON_CLASS =
  "inline-flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted hover:bg-interactive-hover hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300 disabled:cursor-not-allowed disabled:opacity-40";

type Launcher = ReturnType<typeof useLaunchMarsSession>;

interface SessionChildRowProps {
  session: MarsSession;
  launcher: Launcher;
  onOpen: () => void;
}

function SessionChildRow({ session, launcher, onOpen }: SessionChildRowProps) {
  const { t } = useTranslation("openhands");
  const stage = getConnectStage(launcher.connecting, session);
  const { name, phase, meta, stageLabel } = useSessionMeta(session, stage);

  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        disabled={launcher.isBusy || !isConnectableSession(session)}
        aria-label={t(I18nKey.DO_AGENTS$OPEN_AGENT, { name })}
        data-testid={`digitalocean-session-${session.session_id}`}
        className="flex w-full min-w-0 cursor-pointer items-center gap-2.5 rounded-md px-2 py-1.5 text-left hover:bg-surface-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-blue-300 disabled:cursor-not-allowed disabled:opacity-60"
      >
        <SessionStatusDot phase={phase} className="size-1.5" />
        <span className="min-w-0 flex-1 truncate text-xs text-white">
          {name}
        </span>
        <span className="shrink-0 truncate text-xs tabular-nums text-muted">
          {stageLabel ?? meta}
        </span>
        {stage ? (
          <Loader2
            className="size-3.5 shrink-0 animate-spin text-muted"
            aria-hidden
          />
        ) : null}
      </button>
    </li>
  );
}

interface AgentRowProps {
  group: MarsAgentGroup;
  launcher: Launcher;
  error: string | null;
  onOpen: () => void;
  onOpenSession: (session: MarsSession) => void;
  onNewSession: () => void;
}

/** An agent with its sessions nested beneath it, collapsed by default. */
function AgentRow({
  group,
  launcher,
  error,
  onOpen,
  onOpenSession,
  onNewSession,
}: AgentRowProps) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = React.useState(false);
  const { config, sessions } = group;
  const name = agentLabel(config);
  const latest = sessions[0];
  const activity = useSessionActivityLabel(latest);
  const sessionIds = new Set(sessions.map((s) => s.session_id));
  const stage =
    launcher.connecting && sessionIds.has(launcher.connecting.sessionId)
      ? launcher.connecting.stage
      : null;
  const isLaunching = launcher.launchingConfigId === config.id;
  const isWorking = isLaunching || stage !== null;
  const sessionsId = `digitalocean-agent-sessions-${config.id}`;

  let meta: string;
  if (isLaunching) meta = t(I18nKey.BACKEND$DIGITALOCEAN_CREATING_SESSION);
  else if (stage) meta = t(STAGE_LABEL_KEY[stage]);
  else if (!latest) meta = t(I18nKey.DO_AGENTS$NO_SESSIONS_SHORT);
  else {
    meta = [t(SESSION_COUNT_I18N_KEY, { count: sessions.length }), activity]
      .filter(Boolean)
      .join(" · ");
  }

  return (
    <li data-testid={`digitalocean-agent-${config.id}`}>
      <div className="flex items-center gap-1 pl-1 pr-2">
        <button
          type="button"
          onClick={() => setIsExpanded((value) => !value)}
          aria-expanded={isExpanded}
          aria-controls={sessionsId}
          aria-label={t(
            isExpanded
              ? I18nKey.DO_AGENTS$HIDE_SESSIONS
              : I18nKey.DO_AGENTS$SHOW_SESSIONS,
            { name },
          )}
          data-testid={`digitalocean-agent-toggle-${config.id}`}
          className={cn(ICON_BUTTON_CLASS, "size-6")}
        >
          <ChevronRight
            className={cn(
              "size-4 transition-transform duration-150",
              isExpanded && "rotate-90",
            )}
            aria-hidden
          />
        </button>
        <button
          type="button"
          onClick={onOpen}
          disabled={launcher.isBusy}
          aria-label={t(I18nKey.DO_AGENTS$OPEN_AGENT, { name })}
          data-testid={`digitalocean-agent-open-${config.id}`}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 rounded-lg px-2 py-2.5 text-left hover:bg-surface-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-blue-300 disabled:cursor-not-allowed"
        >
          <span
            aria-hidden
            className="relative flex size-8 shrink-0 items-center justify-center rounded-md bg-surface-raised text-sm font-medium text-white"
          >
            {name.charAt(0).toUpperCase()}
            {latest ? (
              <SessionStatusDot
                phase={getSessionPhase(latest.status)}
                className="absolute -bottom-0.5 -right-0.5 ring-2 ring-surface"
              />
            ) : null}
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-sm text-white">{name}</span>
            <span className="truncate text-xs tabular-nums text-muted">
              {meta}
            </span>
          </span>
          {isWorking ? (
            <Loader2
              className="size-4 shrink-0 animate-spin text-muted"
              aria-hidden
            />
          ) : null}
        </button>
        <button
          type="button"
          onClick={onNewSession}
          disabled={launcher.isBusy}
          aria-label={t(I18nKey.DO_AGENTS$NEW_SESSION_FOR, { name })}
          title={t(I18nKey.DO_AGENTS$NEW_SESSION)}
          data-testid={`digitalocean-agent-new-${config.id}`}
          className={ICON_BUTTON_CLASS}
        >
          <Plus className="size-4" aria-hidden />
        </button>
      </div>
      {error ? (
        <p
          role="alert"
          data-testid={`digitalocean-agent-error-${config.id}`}
          className="mx-3 mb-2 text-pretty text-xs text-status-error"
        >
          {error}
        </p>
      ) : null}
      {isExpanded ? (
        <ul
          id={sessionsId}
          className="mb-1 ml-14 mr-2 flex flex-col border-l border-border pl-2"
        >
          {sessions.length === 0 ? (
            <li className="px-2 py-1.5 text-xs text-muted">
              {t(I18nKey.DO_AGENTS$NO_SESSIONS_SHORT)}
            </li>
          ) : (
            sessions.map((session) => (
              <SessionChildRow
                key={session.session_id}
                session={session}
                launcher={launcher}
                onOpen={() => onOpenSession(session)}
              />
            ))
          )}
        </ul>
      ) : null}
    </li>
  );
}

interface DigitalOceanConnectPanelProps {
  /** A session is open and the user is now in its conversation. */
  onConnected: () => void;
}

/**
 * The "DigitalOcean" option in the connect chooser: sign in once, then pick
 * one of the team's OpenHands agents (or create one) to open. Each agent
 * lists its sessions beneath it; opening the agent itself resumes its latest
 * session (or starts its first).
 */
export function DigitalOceanConnectPanel({
  onConnected,
}: DigitalOceanConnectPanelProps) {
  const { t } = useTranslation("openhands");
  const authQuery = useMarsAuthState();
  const authState = authQuery.data;
  const connectionId = getUsableConnectionId(authState);
  const agentsQuery = useMarsAgents(connectionId);
  const signIn = useMarsSignIn();
  const launcher = useLaunchMarsSession();
  const [failure, setFailure] = React.useState<{
    configId: string;
    message: string;
  } | null>(null);
  const [isNewAgentOpen, setIsNewAgentOpen] = React.useState(false);
  const [newAgentError, setNewAgentError] = React.useState<string | null>(null);

  const run = async (configId: string, action: () => Promise<unknown>) => {
    setFailure(null);
    try {
      await action();
      onConnected();
    } catch (error) {
      setFailure({ configId, message: launcher.describeError(error) });
    }
  };

  const createAgent = async (input: NewMarsAgentInput) => {
    setNewAgentError(null);
    let config: MarsAgentConfig;
    try {
      config = await launcher.createAgent(input);
    } catch (error) {
      setNewAgentError(launcher.describeError(error));
      return;
    }
    setIsNewAgentOpen(false);
    await run(config.id, () => launcher.launch(config));
  };

  const closeNewAgent = () => {
    setIsNewAgentOpen(false);
    setNewAgentError(null);
  };

  if (authQuery.isPending) {
    return (
      <div className={cn(PANEL_CLASS, "items-center justify-center")}>
        <Loader2 className="size-5 animate-spin text-muted" aria-hidden />
      </div>
    );
  }

  if (!connectionId) {
    return (
      <div
        data-testid="digitalocean-connect-panel"
        className={cn(PANEL_CLASS, "items-center justify-center px-5")}
      >
        <ManagedAgentsSignIn
          showBranding={false}
          canUseOAuth={authState?.canUseOAuth ?? false}
          isExpired={authState?.active?.isExpired === true}
          isSigningIn={signIn.isSigningIn}
          isSavingToken={signIn.isSavingToken}
          error={signIn.error}
          onSignInWithOAuth={signIn.signInWithOAuth}
          onSaveToken={signIn.saveToken}
        />
      </div>
    );
  }

  const groups = agentsQuery.data?.groups ?? [];
  const active = authState?.active;

  return (
    <div data-testid="digitalocean-connect-panel" className={PANEL_CLASS}>
      <div className="flex items-center gap-2 px-4 pb-2 pt-3">
        <h3 className="min-w-0 flex-1 truncate text-sm font-medium text-white">
          {t(I18nKey.DO_AGENTS$AGENTS_HEADING)}
        </h3>
        <button
          type="button"
          onClick={() =>
            isNewAgentOpen ? closeNewAgent() : setIsNewAgentOpen(true)
          }
          disabled={launcher.isBusy}
          aria-expanded={isNewAgentOpen}
          data-testid="digitalocean-new-agent"
          className={cn(
            "inline-flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-xs font-medium text-muted hover:bg-interactive-hover hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300 disabled:cursor-not-allowed disabled:opacity-40",
            isNewAgentOpen && "bg-interactive-hover text-white",
          )}
        >
          <Plus className="size-4" aria-hidden />
          {t(I18nKey.DO_AGENTS$NEW_AGENT)}
        </button>
        <button
          type="button"
          onClick={() => void agentsQuery.refetch()}
          aria-label={t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_REFRESH)}
          title={t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_REFRESH)}
          data-testid="digitalocean-agents-refresh"
          className={ICON_BUTTON_CLASS}
        >
          <RefreshCw
            className={cn("size-4", agentsQuery.isFetching && "animate-spin")}
            aria-hidden
          />
        </button>
      </div>

      {isNewAgentOpen ? (
        <DigitalOceanNewAgentForm
          isCreating={launcher.isCreatingAgent}
          error={newAgentError}
          onSubmit={(input) => void createAgent(input)}
          onCancel={closeNewAgent}
        />
      ) : null}

      <div className="max-h-72 min-h-0 flex-1 overflow-y-auto px-1 custom-scrollbar-always">
        {agentsQuery.isPending ? (
          <p className="px-3 py-6 text-center text-sm text-muted">
            {t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_LOADING)}
          </p>
        ) : null}
        {agentsQuery.error ? (
          <p
            role="alert"
            className="mx-3 my-2 text-pretty text-sm text-status-error"
          >
            {getMarsErrorMessage(agentsQuery.error) ??
              t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_FAILED)}
          </p>
        ) : null}
        {agentsQuery.data && groups.length === 0 && !isNewAgentOpen ? (
          <p
            data-testid="digitalocean-agents-empty"
            className="mx-auto max-w-sm px-3 py-6 text-center text-pretty text-sm text-muted"
          >
            {t(I18nKey.DO_AGENTS$NO_OPENHANDS_AGENTS)}
          </p>
        ) : null}
        {groups.length > 0 ? (
          <ul className="flex flex-col">
            {groups.map((group) => (
              <AgentRow
                key={group.config.id}
                group={group}
                launcher={launcher}
                error={
                  failure?.configId === group.config.id ? failure.message : null
                }
                onOpen={() =>
                  void run(group.config.id, () => launcher.openAgent(group))
                }
                onOpenSession={(session) =>
                  void run(group.config.id, () =>
                    launcher.open(session, group.config),
                  )
                }
                onNewSession={() =>
                  void run(group.config.id, () => launcher.launch(group.config))
                }
              />
            ))}
          </ul>
        ) : null}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-0.5 border-t border-border px-4 py-2.5 text-xs">
        <span
          aria-hidden
          className="size-1.5 shrink-0 rounded-full bg-status-success"
        />
        <span
          data-testid="digitalocean-account"
          className="min-w-0 flex-1 truncate text-muted"
        >
          {active ? connectionLabel(active) : null}
        </span>
        <button
          type="button"
          onClick={signIn.signOut}
          disabled={signIn.isSigningOut}
          data-testid="digitalocean-sign-out"
          className="shrink-0 cursor-pointer text-muted hover:text-white disabled:cursor-not-allowed"
        >
          {t(I18nKey.BACKEND$DIGITALOCEAN_SIGN_OUT)}
        </button>
        <SignInStorageNote authState={authState} className="w-full pl-4.5" />
      </div>
    </div>
  );
}
