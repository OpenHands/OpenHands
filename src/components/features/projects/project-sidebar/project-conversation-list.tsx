import React from "react";
import { useTranslation } from "react-i18next";
import { FileText } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import {
  projectIdKey,
  type ProjectConversation,
  type ProjectPullRequest,
} from "#/api/projects-service/projects-service.api";
import {
  useProjectArtifacts,
  useProjectConversations,
} from "#/hooks/query/use-projects";
import type { PullRequestStatus } from "#/api/git-service/pull-request-status.api";
import {
  pullRequestRef,
  usePullRequestStatuses,
} from "#/hooks/query/use-pull-request-status";
import { BOARD_GROUPS } from "#/components/features/projects/project-board";
import {
  autoResolveMessage,
  parseAutoResolveNote,
} from "#/utils/conversation-tag-flags";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { conversationStateLabel } from "#/utils/project-conversation-state";
import { cn } from "#/utils/utils";
import { ProjectPanelGroup } from "#/components/features/projects/project-panel/project-panel-group";
import {
  ChecksIcon,
  PULL_REQUEST_STATE_COLOR,
  PULL_REQUEST_STATE_ICON,
} from "#/components/features/projects/project-panel/project-panel-prs";

type StateMark = (typeof BOARD_GROUPS)[number]["id"];

const markOf = (conversation: ProjectConversation): StateMark =>
  BOARD_GROUPS.find((group) => group.attention.includes(conversation.attention))
    ?.id ?? "idle";

/** How much of the circle is filled: none, half, three quarters, or all. */
const MARK_FILL: Record<StateMark, string | null> = {
  idle: null,
  needs_you: "M8 2a6 6 0 0 0 0 12z",
  working: "M8 2a6 6 0 1 1-6 6h6z",
  resolved: "M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2z",
};

const MARK_COLOR: Record<StateMark, string> = {
  needs_you: "text-warning",
  working: "text-success",
  idle: "text-foreground",
  resolved: "text-muted",
};

/** A small circle whose fill says where the conversation stands. */
function StateIcon({ mark }: { mark: StateMark }) {
  const fill = MARK_FILL[mark];
  return (
    <svg
      aria-hidden
      data-mark={mark}
      viewBox="0 0 16 16"
      className={cn("mt-0.5 size-4 shrink-0", MARK_COLOR[mark])}
    >
      <circle
        cx="8"
        cy="8"
        r="6.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      {fill ? <path d={fill} fill="currentColor" /> : null}
    </svg>
  );
}

/**
 * The one muted line under a title: the question it waits on, else why it
 * resolved, else its state. The age sits beside the title.
 */
function StatusLine({ conversation }: { conversation: ProjectConversation }) {
  const { t } = useTranslation("openhands");
  if (conversation.turn === "asking" && conversation.ask) {
    return (
      <>
        {t(I18nKey.PROJECTS$ASKS)} {conversation.ask}
      </>
    );
  }
  const autoNote =
    conversation.attention === "resolved" && conversation.resolved === "auto"
      ? parseAutoResolveNote(conversation.resolve_note)
      : null;
  if (autoNote) return <>{t(...autoResolveMessage(autoNote))}</>;
  return <>{t(conversationStateLabel(conversation))}</>;
}

/**
 * A small stamp under a row: a hairline frame in its own ink, no fill, so a
 * row's marks read like notes printed beside it rather than buttons.
 */
const STAMP =
  "inline-flex h-4.5 items-center gap-1 rounded-facet-control border border-current/35 px-1.5 text-2xs leading-none tabular-nums";

/**
 * A pull request the conversation opened: its state in the icon and ink
 * (open, merged; a closed one muted and struck through, an unread one muted),
 * and its checks after a hairline.
 */
function PullRequestStamp({
  pr,
  status,
}: {
  pr: ProjectPullRequest;
  status: PullRequestStatus | undefined;
}) {
  const ref = pullRequestRef(pr);
  const Icon = PULL_REQUEST_STATE_ICON[status?.state ?? "open"];
  const checks = status && status.checks !== "none" ? status.checks : null;
  return (
    <span
      data-testid="project-conversation-pr"
      data-state={status?.state ?? "unknown"}
      title={status ? `${ref} · ${status.title}` : ref}
      className={cn(
        STAMP,
        // A closed one recedes: red is kept for what needs the user, such as
        // failing checks.
        status && status.state !== "closed"
          ? PULL_REQUEST_STATE_COLOR[status.state]
          : "text-muted",
      )}
    >
      <Icon width={11} height={11} aria-hidden className="shrink-0" />
      <span className={cn(status?.state === "closed" && "line-through")}>
        #{pr.number}
      </span>
      {checks ? (
        <span className="-mr-0.5 flex h-full items-center border-l border-current/25 pl-1">
          <ChecksIcon checks={checks} size={10} />
        </span>
      ) : null}
    </span>
  );
}

function ConversationRow({
  conversation,
  mark,
  title,
  isCurrent,
  fileCount,
  statuses,
}: {
  conversation: ProjectConversation;
  mark: StateMark;
  title: string;
  isCurrent: boolean;
  fileCount: number;
  statuses: Map<string, PullRequestStatus>;
}) {
  const { t } = useTranslation("openhands");
  const id = projectIdKey(conversation.conversation_id);
  return (
    <li>
      <NavigationLink
        to={`/conversations/${id}`}
        data-testid="project-conversation"
        data-attention={conversation.attention}
        aria-current={isCurrent ? "page" : undefined}
        title={id}
        className={cn(
          "flex gap-2.5 rounded-lg px-2 py-1.5 hover:bg-contrast/5",
          isCurrent && "bg-surface",
        )}
      >
        <StateIcon mark={mark} />
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="flex items-baseline gap-2">
            <span className="min-w-0 flex-1 truncate text-sm text-foreground">
              {title}
            </span>
            <span
              data-testid="project-conversation-age"
              className="shrink-0 text-2xs text-muted"
            >
              {formatTimeDelta(conversation.updated_at)}
            </span>
          </span>
          <span
            data-testid="project-conversation-status"
            className="line-clamp-2 text-xs text-muted"
          >
            {isCurrent ? `${t(I18nKey.PROJECTS$YOU_ARE_HERE)} · ` : null}
            <StatusLine conversation={conversation} />
          </span>
          {fileCount || conversation.pull_requests.length ? (
            <span className="mt-1 flex flex-wrap gap-1">
              {conversation.pull_requests.map((pr) => (
                <PullRequestStamp
                  key={pullRequestRef(pr)}
                  pr={pr}
                  status={statuses.get(pullRequestRef(pr))}
                />
              ))}
              {fileCount ? (
                <span
                  data-testid="project-conversation-files"
                  className={cn(STAMP, "border-dashed text-muted")}
                >
                  <FileText width={11} height={11} aria-hidden />
                  {t(I18nKey.PROJECTS$FILE_COUNT, { count: fileCount })}
                </span>
              ) : null}
            </span>
          ) : null}
        </span>
      </NavigationLink>
    </li>
  );
}

/**
 * The project's conversations grouped as on the board, with the coordinator
 * pinned on top and the open conversation marked. In a project view it takes
 * the place of the sidebar's conversation list.
 */
export function ProjectConversationList({
  projectId,
  conversationId,
}: {
  projectId: string;
  /** The open conversation, if it belongs to the project. */
  conversationId?: string;
}) {
  const { t } = useTranslation("openhands");
  const { data: board, isError } = useProjectConversations(projectId);
  const { data: artifacts } = useProjectArtifacts(projectId);
  const current = conversationId ? projectIdKey(conversationId) : null;
  const pullRequests = React.useMemo(
    () =>
      [
        ...(board?.coordinator ? [board.coordinator] : []),
        ...(board?.conversations ?? []),
      ].flatMap((c) => c.pull_requests),
    [board],
  );
  const statuses = usePullRequestStatuses(pullRequests);

  const fileCounts = React.useMemo(() => {
    const counts = new Map<string, number>();
    for (const artifact of artifacts ?? []) {
      if (artifact.conversation_id) {
        const key = projectIdKey(artifact.conversation_id);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }
    return counts;
  }, [artifacts]);

  if (isError) {
    return (
      <p className="text-sm text-danger">{t(I18nKey.PROJECTS$LOAD_FAILED)}</p>
    );
  }
  if (!board) {
    return <p className="text-sm text-muted">{t(I18nKey.PROJECTS$LOADING)}</p>;
  }

  const row = (conversation: ProjectConversation, title: string) => {
    const key = projectIdKey(conversation.conversation_id);
    return (
      <ConversationRow
        key={key}
        conversation={conversation}
        mark={markOf(conversation)}
        title={title}
        isCurrent={key === current}
        fileCount={fileCounts.get(key) ?? 0}
        statuses={statuses}
      />
    );
  };

  return (
    <div data-testid="project-conversations" className="flex flex-col gap-3">
      {board.coordinator ? (
        <ul>{row(board.coordinator, t(I18nKey.PROJECTS$COORDINATOR))}</ul>
      ) : null}
      {BOARD_GROUPS.map((group) => {
        const items = board.conversations.filter((c) =>
          group.attention.includes(c.attention),
        );
        return (
          <ProjectPanelGroup
            key={group.id}
            id={group.id}
            label={t(group.label)}
            count={items.length}
            // Resolved starts folded, unless the open conversation is in it.
            defaultOpen={
              group.id !== "resolved" ||
              items.some((c) => projectIdKey(c.conversation_id) === current)
            }
            emphasized={group.id === "needs_you"}
          >
            {items.map((conversation) =>
              row(
                conversation,
                conversation.title ||
                  projectIdKey(conversation.conversation_id),
              ),
            )}
          </ProjectPanelGroup>
        );
      })}
      {!board.conversations.length ? (
        <p className="text-sm text-muted">{t(I18nKey.PROJECTS$BOARD_EMPTY)}</p>
      ) : null}
    </div>
  );
}
