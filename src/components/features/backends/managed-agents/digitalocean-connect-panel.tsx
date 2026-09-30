import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ChevronRight, Loader2, Plus, RefreshCw } from "lucide-react";

import {
  getMarsBridge,
  getMarsErrorMessage,
  getSessionPhase,
} from "#/api/mars/mars-tunnel-backend";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  getUsableConnectionId,
  useMarsAgents,
  useMarsAuthState,
  type MarsAgentGroup,
} from "#/hooks/query/use-mars-agents";
import { useLaunchMarsSession } from "#/hooks/use-launch-mars-session";
import { useMarsSignIn } from "#/hooks/use-mars-sign-in";
import { I18nKey } from "#/i18n/declaration";
import {
  formatRelativeTime,
  isInvalidTimestamp,
} from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
import { agentLabel } from "./managed-agents-labels";
import { ManagedAgentsSignIn } from "./managed-agents-sign-in";
import { PHASE_DOT_CLASS, STAGE_LABEL_KEY } from "./managed-agents-session-row";

const PANEL_CLASS =
  "flex min-h-[13.5rem] w-full flex-col rounded-xl border border-[var(--oh-border)]";
const ICON_BUTTON_CLASS =
  "inline-flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300 disabled:cursor-not-allowed disabled:opacity-40";

interface AgentRowProps {
  group: MarsAgentGroup;
  launcher: ReturnType<typeof useLaunchMarsSession>;
  error: string | null;
  onOpen: () => void;
  onNewSession: () => void;
}

function AgentRow({
  group,
  launcher,
  error,
  onOpen,
  onNewSession,
}: AgentRowProps) {
  const { t, i18n } = useTranslation("openhands");
  const { config, sessions } = group;
  const name = agentLabel(config);
  const latest = sessions[0];
  const lastActivity = latest?.last_event_at ?? latest?.created_at;
  const sessionIds = new Set(sessions.map((s) => s.session_id));
  const stage =
    launcher.connecting && sessionIds.has(launcher.connecting.sessionId)
      ? launcher.connecting.stage
      : null;
  const isLaunching = launcher.launchingConfigId === config.id;
  const isWorking = isLaunching || stage !== null;

  let meta: string;
  if (isLaunching) meta = t(I18nKey.BACKEND$DIGITALOCEAN_CREATING_SESSION);
  else if (stage) meta = t(STAGE_LABEL_KEY[stage]);
  else if (!latest) meta = t(I18nKey.DO_AGENTS$NO_SESSIONS_SHORT);
  else {
    meta = [
      t(I18nKey.DO_AGENTS$SESSION_COUNT, { count: sessions.length }),
      isInvalidTimestamp(lastActivity)
        ? null
        : formatRelativeTime(lastActivity!, i18n.language, t),
    ]
      .filter(Boolean)
      .join(" · ");
  }

  return (
    <li data-testid={`digitalocean-agent-${config.id}`}>
      <div className="flex items-center gap-1 pr-2">
        <button
          type="button"
          onClick={onOpen}
          disabled={launcher.isBusy}
          aria-label={t(I18nKey.DO_AGENTS$OPEN_AGENT, { name })}
          data-testid={`digitalocean-agent-open-${config.id}`}
          className="group flex min-w-0 flex-1 cursor-pointer items-center gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-[var(--oh-surface-raised)] focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-blue-300 disabled:cursor-not-allowed"
        >
          <span
            aria-hidden
            className="relative flex size-8 shrink-0 items-center justify-center rounded-md bg-[var(--oh-surface-raised)] text-sm font-medium text-white"
          >
            {name.charAt(0).toUpperCase()}
            {latest ? (
              <span
                className={cn(
                  "absolute -bottom-0.5 -right-0.5 size-2 rounded-full ring-2 ring-[var(--oh-surface)]",
                  PHASE_DOT_CLASS[getSessionPhase(latest.status)],
                )}
              />
            ) : null}
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-sm text-white">{name}</span>
            <span className="truncate text-xs tabular-nums text-[var(--oh-muted)]">
              {meta}
            </span>
          </span>
          {isWorking ? (
            <Loader2
              className="size-4 shrink-0 animate-spin text-[var(--oh-muted)]"
              aria-hidden
            />
          ) : (
            <ChevronRight
              className="size-4 shrink-0 text-[var(--oh-muted)] opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100"
              aria-hidden
            />
          )}
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
          className="mx-3 mb-2 text-pretty text-xs text-[var(--oh-status-error)]"
        >
          {error}
        </p>
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
 * one of the team's OpenHands agents to open. Opening resumes the agent's
 * latest session (or starts its first), and from then on the sidebar lists
 * the agent's sessions the way it lists conversations.
 */
export function DigitalOceanConnectPanel({
  onConnected,
}: DigitalOceanConnectPanelProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
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

  const signOut = useMutation({
    mutationFn: () => getMarsBridge()!.signOut(),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: MARS_QUERY_KEYS.all }),
  });

  const run = async (configId: string, action: () => Promise<unknown>) => {
    setFailure(null);
    try {
      await action();
      onConnected();
    } catch (error) {
      setFailure({ configId, message: launcher.describeError(error) });
    }
  };

  if (authQuery.isPending) {
    return (
      <div className={cn(PANEL_CLASS, "items-center justify-center")}>
        <Loader2
          className="size-5 animate-spin text-[var(--oh-muted)]"
          aria-hidden
        />
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

      <div className="max-h-72 min-h-0 flex-1 overflow-y-auto px-1 custom-scrollbar-always">
        {agentsQuery.isPending ? (
          <p className="px-3 py-6 text-center text-sm text-[var(--oh-muted)]">
            {t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_LOADING)}
          </p>
        ) : null}
        {agentsQuery.error ? (
          <p
            role="alert"
            className="mx-3 my-2 text-pretty text-sm text-[var(--oh-status-error)]"
          >
            {getMarsErrorMessage(agentsQuery.error) ??
              t(I18nKey.BACKEND$DIGITALOCEAN_AGENTS_FAILED)}
          </p>
        ) : null}
        {agentsQuery.data && groups.length === 0 ? (
          <p
            data-testid="digitalocean-agents-empty"
            className="mx-auto max-w-sm px-3 py-6 text-center text-pretty text-sm text-[var(--oh-muted)]"
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
                onNewSession={() =>
                  void run(group.config.id, () => launcher.launch(group.config))
                }
              />
            ))}
          </ul>
        ) : null}
      </div>

      <div className="mt-2 flex items-center gap-3 border-t border-[var(--oh-border)] px-4 py-2.5 text-xs">
        <span
          aria-hidden
          className="size-1.5 shrink-0 rounded-full bg-[var(--oh-status-success)]"
        />
        <span
          data-testid="digitalocean-account"
          className="min-w-0 flex-1 truncate text-[var(--oh-muted)]"
        >
          {active?.teamName ?? active?.label}
        </span>
        <button
          type="button"
          onClick={() => signOut.mutate()}
          disabled={signOut.isPending}
          data-testid="digitalocean-sign-out"
          className="shrink-0 cursor-pointer text-[var(--oh-muted)] hover:text-white disabled:cursor-not-allowed"
        >
          {t(I18nKey.BACKEND$DIGITALOCEAN_SIGN_OUT)}
        </button>
      </div>
    </div>
  );
}
