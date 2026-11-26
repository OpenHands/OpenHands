import React from "react";
import { useTranslation } from "react-i18next";
import {
  CircleDashed,
  CircleX,
  FileText,
  GitMerge,
  GitPullRequest,
  GitPullRequestClosed,
  GitPullRequestDraft,
} from "lucide-react";
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
import {
  autoResolveMessage,
  parseAutoResolveNote,
} from "#/utils/conversation-tag-flags";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { conversationStateLabel } from "#/utils/project-conversation-state";
import { cn } from "#/utils/utils";
import { ProjectPanelGroup } from "#/components/features/projects/project-panel/project-panel-group";
import {
  ATTENTION_LABEL,
  ATTENTION_ORDER,
  pullRequestAttention,
  type PullRequestAttention,
} from "#/utils/pull-request-attention";
import {
  BOARD_GROUPS,
  boardGroupOf,
  readyPullRequestOf,
  type BoardGroupId,
} from "#/utils/project-board-group";

type StateMark = BoardGroupId;

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
 * The one muted line under a title: the question it waits on, else the pull
 * request that moved it to Needs you, else why it resolved, else its state.
 * The age sits beside the title.
 */
function StatusLine({
  conversation,
  statuses,
}: {
  conversation: ProjectConversation;
  statuses: Map<string, PullRequestStatus>;
}) {
  const { t } = useTranslation("openhands");
  if (conversation.turn === "asking" && conversation.ask) {
    return (
      <>
        {t(I18nKey.PROJECTS$ASKS)} {conversation.ask}
      </>
    );
  }
  const ready = readyPullRequestOf(conversation, statuses);
  if (ready) {
    return (
      <>
        #{ready.number} · {t(I18nKey.PROJECTS$PR_READY_HINT)}
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

interface RowPullRequest {
  pr: ProjectPullRequest;
  ref: string;
  status: PullRequestStatus | undefined;
  attention: PullRequestAttention;
}

const pullRequestUrl = ({ pr, status }: RowPullRequest) =>
  status?.url ?? `https://github.com/${pr.owner}/${pr.repo}/pull/${pr.number}`;

/** A row's pull requests, most pressing first. */
function rowPullRequests(
  conversation: ProjectConversation,
  statuses: Map<string, PullRequestStatus>,
): RowPullRequest[] {
  return conversation.pull_requests
    .map((pr) => {
      const ref = pullRequestRef(pr);
      const status = statuses.get(ref);
      return { pr, ref, status, attention: pullRequestAttention(status) };
    })
    .sort(
      (a, b) =>
        ATTENTION_ORDER.indexOf(a.attention) -
        ATTENTION_ORDER.indexOf(b.attention),
    );
}

const ATTENTION_ICON: Record<PullRequestAttention, typeof GitPullRequest> = {
  ready: GitPullRequest,
  failing: CircleX,
  checking: CircleDashed,
  draft: GitPullRequestDraft,
  merged: GitMerge,
  closed: GitPullRequestClosed,
  unknown: GitPullRequest,
};

/**
 * How loud a chip is follows what it asks of the user: a solid one is
 * theirs to review, a red frame is failing, a dashed one is on its way, and
 * a finished one is plain text that steps back.
 */
const CHIP_LOOK: Record<PullRequestAttention, string> = {
  ready: "bg-primary text-on-primary font-medium",
  failing: "border border-danger/60 text-danger",
  checking: "border border-dashed border-current/40 text-muted",
  draft: "border border-dashed border-current/40 text-muted",
  unknown: "text-muted",
  merged: "text-muted",
  closed: "text-muted/70",
};

/** The word a chip says, only where the user may act on it. */
const CHIP_WORD: Partial<Record<PullRequestAttention, I18nKey>> = {
  ready: I18nKey.PROJECTS$PR_READY,
  failing: I18nKey.PROJECTS$PR_FAILING,
};

function PullRequestChip({ item }: { item: RowPullRequest }) {
  const { t } = useTranslation("openhands");
  const Icon = ATTENTION_ICON[item.attention];
  const word = CHIP_WORD[item.attention];
  const label = t(ATTENTION_LABEL[item.attention]);
  return (
    <a
      href={pullRequestUrl(item)}
      target="_blank"
      rel="noopener noreferrer"
      data-testid="project-conversation-pr"
      data-attention={item.attention}
      aria-label={`${t(I18nKey.PROJECTS$PR_OPEN_ON_GITHUB, { ref: item.ref })}: ${label}`}
      title={`${label} · ${item.ref}${item.status ? ` · ${item.status.title}` : ""}`}
      className={cn(
        "inline-flex h-5 items-center gap-1 rounded-full px-2 text-2xs leading-none tabular-nums",
        "transition-opacity hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-1",
        CHIP_LOOK[item.attention],
        // Plain-text chips line up with the framed ones' text.
        !CHIP_LOOK[item.attention].includes("border") &&
          !CHIP_LOOK[item.attention].includes("bg-") &&
          "px-1",
      )}
    >
      <Icon width={12} height={12} aria-hidden className="shrink-0" />
      <span className={cn(item.attention === "closed" && "line-through")}>
        #{item.pr.number}
      </span>
      {word ? <span>{t(word)}</span> : null}
    </a>
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
  const pullRequests = rowPullRequests(conversation, statuses);
  const hasMarks = fileCount > 0 || pullRequests.length > 0;
  return (
    // The row links to the conversation; its pull requests link to GitHub,
    // so they sit beside the link rather than inside it.
    <li
      className={cn(
        "rounded-lg hover:bg-contrast/5",
        isCurrent && "bg-surface hover:bg-surface",
      )}
    >
      <NavigationLink
        to={`/conversations/${id}`}
        data-testid="project-conversation"
        data-attention={conversation.attention}
        aria-current={isCurrent ? "page" : undefined}
        title={id}
        className={cn("flex gap-2.5 px-2 pt-1.5", hasMarks ? "pb-1" : "pb-1.5")}
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
            <StatusLine conversation={conversation} statuses={statuses} />
          </span>
        </span>
      </NavigationLink>
      {hasMarks ? (
        // Lined up under the title: the mark is 16px wide, with a 10px gap.
        <div className="pr-2 pb-1.5 pl-8.5">
          <span className="flex flex-wrap items-center gap-1">
            {pullRequests.map((item) => (
              <PullRequestChip key={item.ref} item={item} />
            ))}
            {fileCount ? (
              <span
                data-testid="project-conversation-files"
                className="inline-flex h-5 items-center gap-1 px-1 text-2xs text-muted"
              >
                <FileText width={12} height={12} aria-hidden />
                {t(I18nKey.PROJECTS$FILE_COUNT, { count: fileCount })}
              </span>
            ) : null}
          </span>
        </div>
      ) : null}
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
        mark={boardGroupOf(conversation, statuses)}
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
        const items = board.conversations.filter(
          (c) => boardGroupOf(c, statuses) === group.id,
        );
        const hasCurrent = items.some(
          (c) => projectIdKey(c.conversation_id) === current,
        );
        return (
          <ProjectPanelGroup
            key={group.id}
            id={group.id}
            label={t(group.label)}
            count={items.length}
            // Resolved starts folded, unless the open conversation is in it.
            defaultOpen={group.id !== "resolved" || hasCurrent}
            activeKey={hasCurrent ? current : null}
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
