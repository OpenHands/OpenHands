import type React from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Plus, Trash2 } from "lucide-react";

import type { MarsAgentGroup } from "#/hooks/query/use-mars-agents";
import { I18nKey } from "#/i18n/declaration";
import { agentLabel, SESSION_COUNT_I18N_KEY } from "./managed-agents-labels";

/** A filtered agent group plus counts taken from its unfiltered sessions. */
export interface VisibleAgentGroup extends MarsAgentGroup {
  totalSessions: number;
  liveSessions: number;
}

interface AgentGroupSectionProps {
  group: VisibleAgentGroup;
  /** A session is being created for this agent. */
  isCreating: boolean;
  /** Any launch or connect is in flight. */
  isLaunching: boolean;
  error: string | null;
  onNewSession: () => void;
  onDeleteAgent: () => void;
  children: React.ReactNode;
}

/** One agent's header, actions, and (as children) its session rows. */
export function AgentGroupSection({
  group,
  isCreating,
  isLaunching,
  error,
  onNewSession,
  onDeleteAgent,
  children,
}: AgentGroupSectionProps) {
  const { t } = useTranslation("openhands");
  const { config, sessions, totalSessions, liveSessions } = group;
  const deleteTitle =
    liveSessions > 0
      ? t(I18nKey.DO_AGENTS$DELETE_AGENT_BLOCKED)
      : t(I18nKey.DO_AGENTS$DELETE_AGENT);

  return (
    <section
      data-testid={`managed-agents-agent-${config.id}`}
      className="overflow-hidden rounded-lg border border-border bg-surface-raised"
    >
      <header className="flex items-center gap-3 border-b border-border px-3 py-2">
        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <h4 className="truncate text-sm font-medium text-white">
            {agentLabel(config)}
          </h4>
          <span className="shrink-0 text-xs text-muted">
            {t(SESSION_COUNT_I18N_KEY, { count: totalSessions })}
          </span>
        </div>
        <button
          type="button"
          onClick={onNewSession}
          disabled={isLaunching}
          data-testid={`managed-agents-new-session-${config.id}`}
          className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-xs text-primary transition-colors hover:bg-interactive-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isCreating ? (
            <Loader2 className="size-3.5 animate-spin" aria-hidden />
          ) : (
            <Plus className="size-3.5" aria-hidden />
          )}
          {isCreating
            ? t(I18nKey.BACKEND$DIGITALOCEAN_CREATING_SESSION)
            : t(I18nKey.DO_AGENTS$NEW_SESSION)}
        </button>
        <button
          type="button"
          onClick={onDeleteAgent}
          disabled={isLaunching || liveSessions > 0}
          aria-label={t(I18nKey.DO_AGENTS$DELETE_AGENT)}
          title={deleteTitle}
          data-testid={`managed-agents-delete-agent-${config.id}`}
          className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted transition-colors hover:bg-interactive-hover hover:text-status-error disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Trash2 className="size-3.5" aria-hidden />
        </button>
      </header>
      {error ? (
        <p
          role="alert"
          className="border-b border-border px-3 py-2 text-xs text-status-error"
        >
          {error}
        </p>
      ) : null}
      {sessions.length > 0 ? (
        <ul className="divide-y divide-border">{children}</ul>
      ) : (
        <p className="px-3 py-3 text-xs text-muted">
          {t(I18nKey.DO_AGENTS$NO_SESSIONS)}
        </p>
      )}
    </section>
  );
}
