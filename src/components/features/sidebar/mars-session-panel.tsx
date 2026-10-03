import React from "react";
import { useTranslation } from "react-i18next";
import { Check, ChevronsUpDown, Loader2, Plus, Settings2 } from "lucide-react";

import DigitalOceanLogo from "#/assets/branding/digitalocean-logo.svg?react";
import type { Backend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  getSessionPhase,
  isConnectableSession,
  SESSION_STATUS_READY,
  type MarsAgentConfig,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import {
  agentLabel,
  getSessionDisplayName,
} from "#/components/features/backends/managed-agents/managed-agents-labels";
import {
  PHASE_DOT_CLASS,
  PHASE_LABEL_KEY,
  STAGE_LABEL_KEY,
} from "#/components/features/backends/managed-agents/managed-agents-session-row";
import { ConversationStatusDot } from "#/components/features/conversation-panel/conversation-status-dot";
import { NEW_CONVERSATION_DROPDOWN_SURFACE } from "#/components/features/conversation-panel/new-conversation-dropdown-styles";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useNavigation } from "#/context/navigation-context";
import {
  getUsableConnectionId,
  useMarsAgents,
  useMarsAuthState,
  type MarsAgentsSnapshot,
} from "#/hooks/query/use-mars-agents";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import { useLaunchMarsSession } from "#/hooks/use-launch-mars-session";
import { I18nKey } from "#/i18n/declaration";
import { Divider } from "#/ui/divider";
import {
  dropdownMenuListClassName,
  dropdownMenuRowClassName,
} from "#/utils/dropdown-classes";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";

const ManageBackendsModal = React.lazy(() =>
  import("#/components/features/backends/manage-backends-modal").then((m) => ({
    default: m.ManageBackendsModal,
  })),
);

/** Listing sessions is a harness-api read; it never wakes a sandbox. */
const SESSIONS_REFETCH_MS = 30_000;
const NEW_CHAT_PATH = "/conversations";

const ICON_BUTTON_CLASS =
  "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-border)] disabled:cursor-not-allowed disabled:opacity-50";

interface SessionContext {
  config: MarsAgentConfig | undefined;
  sessions: MarsSession[];
}

/**
 * The agent the active session belongs to and that agent's sessions; for a
 * session launched outside any listed agent, the team's other sessions.
 */
function resolveSessionContext(
  data: MarsAgentsSnapshot | undefined,
  backend: Backend,
): SessionContext {
  const groups = data?.groups ?? [];
  const group =
    groups.find((g) => g.config.id === backend.marsConfigId) ??
    groups.find((g) =>
      g.sessions.some((s) => s.session_id === backend.marsSessionId),
    );
  return {
    config: group?.config,
    sessions: group?.sessions ?? data?.otherSessions ?? [],
  };
}

/**
 * The list can lag a just-created or just-connected session; the one the
 * backend points at must still show, or its conversations would vanish.
 */
function withCurrentSession(
  sessions: MarsSession[],
  backend: Backend,
): MarsSession[] {
  const sessionId = backend.marsSessionId;
  if (!sessionId || sessions.some((s) => s.session_id === sessionId)) {
    return sessions;
  }
  // Connected implies its tunnel answered, so it reads as running.
  return [{ session_id: sessionId, status: SESSION_STATUS_READY }, ...sessions];
}

/** The active session's conversations, nested under its row. */
function SessionConversations({
  conversations,
}: {
  conversations: AppConversation[];
}) {
  const { t } = useTranslation("openhands");
  const { conversationId } = useNavigation();
  const backendScopedPath = useBackendScopedPath();

  if (conversations.length === 0) return null;
  return (
    <ul
      data-testid="mars-session-conversations"
      className="mt-0.5 flex flex-col gap-0.5"
    >
      {conversations.map((conversation) => (
        <li key={conversation.id}>
          <NavigationLink
            to={backendScopedPath(`/conversations/${conversation.id}`)}
            data-testid={`mars-session-conversation-${conversation.id}`}
            className={cn(
              "flex h-8 min-w-0 items-center gap-2 rounded-md pl-6 pr-2 text-sm text-white hover:bg-[var(--oh-surface)]",
              conversation.id === conversationId && "bg-[var(--oh-surface)]",
            )}
          >
            <ConversationStatusDot
              executionStatus={conversation.execution_status}
              sandboxStatus={conversation.sandbox_status}
              showTooltip={false}
            />
            <span className="min-w-0 flex-1 truncate">
              {conversation.title || t(I18nKey.CONVERSATION$UNTITLED)}
            </span>
            <time className="shrink-0 text-xs tabular-nums text-[var(--oh-muted)]">
              {formatTimeDelta(
                conversation.updated_at ?? conversation.created_at,
              )}
            </time>
          </NavigationLink>
        </li>
      ))}
    </ul>
  );
}

interface MarsSessionPanelProps {
  /** The active backend; must carry a `marsSessionId`. */
  backend: Backend;
}

/**
 * Sidebar list for a DigitalOcean Managed Agents backend, in place of the
 * conversation list. Sessions of the active agent read like thread folders:
 * the one you are on expands to its conversations, any other opens with a
 * click, and the header launches a new session or switches agent.
 */
export function MarsSessionPanel({ backend }: MarsSessionPanelProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const backendScopedPath = useBackendScopedPath();
  const launcher = useLaunchMarsSession();

  const [menuOpen, setMenuOpen] = React.useState(false);
  const [manageOpen, setManageOpen] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const menuRef = React.useRef<HTMLDivElement>(null);

  const authQuery = useMarsAuthState();
  const agentsQuery = useMarsAgents(getUsableConnectionId(authQuery.data), {
    refetchIntervalMs: SESSIONS_REFETCH_MS,
  });
  const context = resolveSessionContext(agentsQuery.data, backend);
  const { config } = context;
  const sessions = withCurrentSession(context.sessions, backend);
  const groups = agentsQuery.data?.groups ?? [];
  const { data: conversationPages } = usePaginatedConversations();
  const conversations =
    conversationPages?.pages.flatMap((page) => page.items) ?? [];
  const latestConversationId = conversations[0]?.id;

  React.useEffect(() => {
    if (!menuOpen) return undefined;
    const onPointerDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  const run = async (action: () => Promise<unknown>) => {
    setError(null);
    try {
      await action();
    } catch (failure) {
      setError(launcher.describeError(failure));
    }
  };

  const isLaunching = config ? launcher.launchingConfigId === config.id : false;
  const title = config ? agentLabel(config) : backend.name;

  return (
    <div
      data-testid="mars-session-panel"
      className="flex h-full min-h-0 w-full flex-col"
    >
      <div className="-ml-2.5 box-border w-[calc(100%+0.625rem)] max-w-none">
        <div className="flex min-w-0 flex-nowrap items-center gap-x-1 py-2 pl-2.5 pr-2.5">
          <div ref={menuRef} className="relative min-w-0 flex-1">
            <button
              type="button"
              onClick={() => setMenuOpen((open) => !open)}
              aria-expanded={menuOpen}
              aria-haspopup="menu"
              aria-label={t(I18nKey.DO_AGENTS$SWITCH_AGENT)}
              data-testid="mars-session-panel-agent"
              className="flex h-7 max-w-full cursor-pointer items-center gap-2 rounded-md px-1.5 text-sm font-medium text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-border)]"
            >
              <DigitalOceanLogo
                width={14}
                height={14}
                className="shrink-0"
                aria-hidden
              />
              <span className="truncate">{title}</span>
              <ChevronsUpDown className="size-3.5 shrink-0" aria-hidden />
            </button>

            {menuOpen ? (
              <div
                role="menu"
                data-testid="mars-session-panel-agent-menu"
                className={cn(
                  NEW_CONVERSATION_DROPDOWN_SURFACE,
                  "absolute left-0 top-full z-20 mt-1 w-64",
                )}
              >
                <p className="px-2 pb-1 pt-1.5 text-xs text-[var(--oh-text-tertiary)]">
                  {t(I18nKey.DO_AGENTS$AGENTS_HEADING)}
                </p>
                <div
                  className={cn(
                    "max-h-[40vh] overflow-y-auto",
                    dropdownMenuListClassName,
                  )}
                >
                  {groups.map((group) => {
                    const isCurrent = group.config.id === config?.id;
                    return (
                      <button
                        key={group.config.id}
                        type="button"
                        role="menuitem"
                        disabled={launcher.isBusy}
                        onClick={() => {
                          setMenuOpen(false);
                          if (!isCurrent) {
                            void run(() => launcher.openAgent(group));
                          }
                        }}
                        data-testid={`mars-session-panel-agent-${group.config.id}`}
                        className={dropdownMenuRowClassName}
                      >
                        <span className="min-w-0 flex-1 truncate">
                          {agentLabel(group.config)}
                        </span>
                        {isCurrent ? (
                          <Check
                            className="size-3.5 shrink-0 text-primary"
                            aria-hidden
                          />
                        ) : null}
                      </button>
                    );
                  })}
                  <Divider inset="menu" />
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setMenuOpen(false);
                      setManageOpen(true);
                    }}
                    data-testid="mars-session-panel-manage"
                    className={dropdownMenuRowClassName}
                  >
                    <Settings2 className="size-4" aria-hidden />
                    {t(I18nKey.DO_AGENTS$MANAGE)}
                  </button>
                </div>
              </div>
            ) : null}
          </div>

          {config ? (
            <button
              type="button"
              onClick={() => void run(() => launcher.launch(config))}
              disabled={launcher.isBusy}
              aria-label={t(I18nKey.DO_AGENTS$NEW_SESSION)}
              title={t(I18nKey.DO_AGENTS$NEW_SESSION)}
              data-testid="mars-session-panel-new"
              className={ICON_BUTTON_CLASS}
            >
              {isLaunching ? (
                <Loader2 className="size-3.5 animate-spin" aria-hidden />
              ) : (
                <Plus className="size-4" aria-hidden />
              )}
            </button>
          ) : null}
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden overscroll-contain custom-scrollbar-always">
        {isLaunching ? (
          <p
            data-testid="mars-session-panel-launching"
            className="flex h-8 items-center gap-2 pl-2 text-sm text-[var(--oh-muted)]"
          >
            <Loader2 className="size-3.5 animate-spin" aria-hidden />
            {t(I18nKey.BACKEND$DIGITALOCEAN_CREATING_SESSION)}
          </p>
        ) : null}

        {error ? (
          <p
            role="alert"
            data-testid="mars-session-panel-error"
            className="mx-1 my-1 text-pretty rounded-md bg-[var(--oh-status-error)]/10 p-2 text-xs text-[var(--oh-status-error)]"
          >
            {error}
          </p>
        ) : null}

        {agentsQuery.data && sessions.length === 0 ? (
          <p className="px-2 py-2 text-xs text-[var(--oh-muted)]">
            {t(I18nKey.DO_AGENTS$NO_SESSIONS)}
          </p>
        ) : null}

        <ul className="flex flex-col gap-0.5">
          {sessions.map((session) => {
            const isCurrent = session.session_id === backend.marsSessionId;
            const phase = getSessionPhase(session.status);
            const stage =
              launcher.connecting?.sessionId === session.session_id
                ? launcher.connecting.stage
                : null;
            const name = getSessionDisplayName(session, t);
            const lastActivity = session.last_event_at ?? session.created_at;
            let trailing: React.ReactNode = null;
            if (stage) {
              trailing = (
                <Loader2
                  className="size-3.5 shrink-0 animate-spin"
                  aria-hidden
                />
              );
            } else if (!isCurrent && lastActivity) {
              trailing = (
                <time className="shrink-0 text-xs tabular-nums">
                  {formatTimeDelta(lastActivity)}
                </time>
              );
            }
            return (
              <li
                key={session.session_id}
                data-testid={`mars-session-${session.session_id}`}
              >
                <div
                  className={cn(
                    "group flex h-8 w-full min-w-0 items-center gap-0.5 rounded-md pl-2 pr-1 text-sm hover:bg-[var(--oh-surface-raised)]",
                    isCurrent
                      ? "text-white"
                      : "text-[var(--oh-muted)] hover:text-white",
                  )}
                >
                  <button
                    type="button"
                    disabled={
                      launcher.isBusy ||
                      (!isCurrent && !isConnectableSession(session))
                    }
                    aria-current={isCurrent ? "true" : undefined}
                    title={
                      stage
                        ? t(STAGE_LABEL_KEY[stage])
                        : t(PHASE_LABEL_KEY[phase])
                    }
                    onClick={() =>
                      isCurrent
                        ? navigate(
                            backendScopedPath(
                              latestConversationId
                                ? `/conversations/${latestConversationId}`
                                : NEW_CHAT_PATH,
                            ),
                          )
                        : void run(() => launcher.open(session, config))
                    }
                    data-testid={`mars-session-open-${session.session_id}`}
                    className="flex min-h-8 min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-md py-1 text-left text-inherit outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-border)] disabled:cursor-not-allowed"
                  >
                    <span
                      aria-hidden
                      className={cn(
                        "size-2 shrink-0 rounded-full",
                        PHASE_DOT_CLASS[phase],
                      )}
                    />
                    <span className="min-w-0 flex-1 truncate">{name}</span>
                    {stage ? (
                      <span className="sr-only">
                        {t(STAGE_LABEL_KEY[stage])}
                      </span>
                    ) : null}
                    {trailing}
                  </button>
                  {isCurrent ? (
                    <button
                      type="button"
                      onClick={() => navigate(backendScopedPath(NEW_CHAT_PATH))}
                      aria-label={t(I18nKey.DO_AGENTS$NEW_CHAT_IN_SESSION, {
                        name,
                      })}
                      title={t(I18nKey.DO_AGENTS$NEW_CHAT_IN_SESSION, { name })}
                      data-testid="mars-session-new-chat"
                      className={ICON_BUTTON_CLASS}
                    >
                      <Plus className="size-3.5" aria-hidden strokeWidth={2} />
                    </button>
                  ) : null}
                </div>
                {isCurrent ? (
                  <SessionConversations conversations={conversations} />
                ) : null}
              </li>
            );
          })}
        </ul>
      </div>

      {manageOpen ? (
        <React.Suspense fallback={null}>
          <ManageBackendsModal
            initialView="managed-agents"
            onClose={() => setManageOpen(false)}
          />
        </React.Suspense>
      ) : null}
    </div>
  );
}
