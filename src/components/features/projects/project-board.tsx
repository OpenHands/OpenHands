import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import ProjectsService, {
  type ProjectConversation,
  type ProjectPullRequest,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { projectsQueryRoot } from "#/hooks/query/use-projects";
import { usePullRequestStatuses } from "#/hooks/query/use-pull-request-status";
import {
  autoResolveMessage,
  parseAutoResolveNote,
} from "#/utils/conversation-tag-flags";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formatTimeDelta } from "#/utils/format-time-delta";
import {
  conversationStateLabel,
  needsUser,
} from "#/utils/project-conversation-state";
import { cn } from "#/utils/utils";
import {
  BOARD_GROUPS,
  boardGroupOf,
  readyPullRequestOf,
} from "#/utils/project-board-group";

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
  readyPullRequest,
}: {
  projectId: string;
  conversation: ProjectConversation;
  /** The pull request that moved it to Needs you, if one did. */
  readyPullRequest?: ProjectPullRequest;
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
  const [isConfirmingRemove, setIsConfirmingRemove] = React.useState(false);

  return (
    <li
      data-testid="project-card"
      data-attention={conversation.attention}
      className="flex flex-col gap-2 rounded-lg border border-control bg-base-secondary px-3 py-2.5"
    >
      <NavigationLink
        to={conversationPath(conversation.conversation_id)}
        className="break-words text-sm font-medium text-foreground hover:underline"
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
      {conversation.turn === "asking" && conversation.ask ? (
        <p className="line-clamp-3 text-sm text-foreground">
          <span className="text-muted">{t(I18nKey.PROJECTS$ASKS)}</span>{" "}
          {conversation.ask}
        </p>
      ) : null}
      {readyPullRequest ? (
        <p
          data-testid="project-card-ready-pr"
          className="text-sm text-foreground"
        >
          <a
            href={`https://github.com/${readyPullRequest.owner}/${readyPullRequest.repo}/pull/${readyPullRequest.number}`}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium underline-offset-2 hover:underline"
          >
            #{readyPullRequest.number}
          </a>{" "}
          <span className="text-muted">
            · {t(I18nKey.PROJECTS$PR_READY_HINT)}
          </span>
        </p>
      ) : null}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span
          data-testid="project-card-state"
          className={cn(needsUser(conversation) && "text-foreground")}
        >
          {t(conversationStateLabel(conversation))}
        </span>
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
            onClick={() => setIsConfirmingRemove(true)}
          >
            {t(I18nKey.PROJECTS$REMOVE)}
          </button>
        </span>
      </div>
      {isConfirmingRemove ? (
        <ConfirmationModal
          text={t(I18nKey.PROJECTS$REMOVE_CONFIRM, { title })}
          confirmText={t(I18nKey.PROJECTS$REMOVE)}
          isConfirming={remove.isPending}
          onCancel={() => setIsConfirmingRemove(false)}
          onConfirm={() =>
            remove.mutate(undefined, {
              onSettled: () => setIsConfirmingRemove(false),
            })
          }
        />
      ) : null}
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
  const statuses = usePullRequestStatuses(
    React.useMemo(
      () => conversations.flatMap((c) => c.pull_requests),
      [conversations],
    ),
  );
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
        const items = conversations.filter(
          (c) => boardGroupOf(c, statuses) === group.id,
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
                  readyPullRequest={readyPullRequestOf(conversation, statuses)}
                />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
