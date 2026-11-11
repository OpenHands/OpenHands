import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import ProjectsService, {
  type ProjectAttention,
  type ProjectConversation,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { projectsQueryRoot } from "#/hooks/query/use-projects";
import {
  autoResolveMessage,
  parseAutoResolveNote,
} from "#/utils/conversation-tag-flags";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";

/** The board's columns, most urgent first, and the attention each one holds. */
export const BOARD_GROUPS: Array<{
  id: "needs_you" | "working" | "done" | "resolved";
  label: I18nKey;
  attention: ProjectAttention[];
}> = [
  {
    id: "needs_you",
    label: I18nKey.PROJECTS$GROUP_NEEDS_YOU,
    attention: ["needs_you", "needs_input"],
  },
  {
    id: "working",
    label: I18nKey.PROJECTS$GROUP_WORKING,
    attention: ["working"],
  },
  {
    id: "done",
    label: I18nKey.PROJECTS$GROUP_DONE,
    attention: ["finished", "idle"],
  },
  {
    id: "resolved",
    label: I18nKey.PROJECTS$GROUP_RESOLVED,
    attention: ["resolved"],
  },
];

const conversationPath = (id: string) =>
  `/conversations/${encodeURIComponent(id.replace(/-/g, ""))}`;

/** Resolve and Reopen go through the project's routes; the server owns the tags. */
export const setResolved = (
  projectId: string,
  conversationId: string,
  resolved: boolean,
) =>
  ProjectsService.setConversationResolved(projectId, conversationId, resolved);

function ProjectCard({
  projectId,
  conversation,
}: {
  projectId: string;
  conversation: ProjectConversation;
}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const refresh = () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) }),
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] }),
    ]);

  const isResolved = conversation.attention === "resolved";
  const autoNote =
    isResolved && conversation.resolved === "auto"
      ? parseAutoResolveNote(conversation.resolve_note)
      : null;
  const title = conversation.title || conversation.conversation_id;

  const resolve = useMutation({
    mutationFn: () =>
      setResolved(projectId, conversation.conversation_id, !isResolved),
    meta: { disableToast: true },
    onSettled: refresh,
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPDATE_FAILED)),
  });
  const remove = useMutation({
    mutationFn: () =>
      ProjectsService.removeConversation(
        projectId,
        conversation.conversation_id,
      ),
    meta: { disableToast: true },
    onSettled: refresh,
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPDATE_FAILED)),
  });
  const busy = resolve.isPending || remove.isPending;

  return (
    <li
      data-testid="project-card"
      data-attention={conversation.attention}
      className="flex flex-col gap-2 rounded-lg border border-control bg-base-secondary px-3 py-2.5"
    >
      <NavigationLink
        to={conversationPath(conversation.conversation_id)}
        className="text-sm font-medium text-foreground hover:underline"
        title={conversation.conversation_id}
      >
        {title}
      </NavigationLink>
      {autoNote ? (
        <p
          data-testid="project-card-auto-resolved"
          className="text-xs text-muted"
        >
          {t(...autoResolveMessage(autoNote))}
        </p>
      ) : null}
      {conversation.turn === "asked" && conversation.ask ? (
        <p className="line-clamp-3 text-sm text-foreground">
          <span className="text-muted">{t(I18nKey.PROJECTS$ASKS)}</span>{" "}
          {conversation.ask}
        </p>
      ) : null}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span>{conversation.execution_status}</span>
        <span>{formatTimeDelta(conversation.updated_at)}</span>
        <span className="ml-auto flex gap-3">
          <button
            type="button"
            data-testid="project-card-resolve"
            className="text-link hover:underline disabled:opacity-50"
            disabled={busy}
            aria-label={`${t(isResolved ? I18nKey.PROJECTS$REOPEN : I18nKey.PROJECTS$RESOLVE)} ${title}`}
            onClick={() => resolve.mutate()}
          >
            {t(isResolved ? I18nKey.PROJECTS$REOPEN : I18nKey.PROJECTS$RESOLVE)}
          </button>
          <button
            type="button"
            data-testid="project-card-remove"
            className="text-muted hover:text-foreground hover:underline disabled:opacity-50"
            disabled={busy}
            aria-label={`${t(I18nKey.PROJECTS$REMOVE)} ${title}`}
            onClick={() => remove.mutate()}
          >
            {t(I18nKey.PROJECTS$REMOVE)}
          </button>
        </span>
      </div>
    </li>
  );
}

export function ProjectBoard({
  projectId,
  conversations,
}: {
  projectId: string;
  conversations: ProjectConversation[];
}) {
  const { t } = useTranslation("openhands");
  if (!conversations.length) {
    return (
      <p data-testid="project-board-empty" className="text-sm text-muted">
        {t(I18nKey.PROJECTS$BOARD_EMPTY)}
      </p>
    );
  }
  return (
    <div
      data-testid="project-board"
      className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"
    >
      {BOARD_GROUPS.map((group) => {
        const items = conversations.filter((c) =>
          group.attention.includes(c.attention),
        );
        return (
          <section
            key={group.id}
            data-testid={`project-group-${group.id}`}
            aria-label={t(group.label)}
            className="flex min-w-0 flex-col gap-2"
          >
            <h3
              className={cn(
                "text-xs font-medium uppercase tracking-wide",
                group.id === "needs_you" && items.length
                  ? "text-foreground"
                  : "text-muted",
              )}
            >
              {t(group.label)} · {items.length}
            </h3>
            <ul className="flex flex-col gap-2">
              {items.map((conversation) => (
                <ProjectCard
                  key={conversation.conversation_id}
                  projectId={projectId}
                  conversation={conversation}
                />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
