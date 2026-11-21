import React from "react";
import { useTranslation } from "react-i18next";
import {
  CircleCheck,
  CircleDashed,
  CircleX,
  GitMerge,
  GitPullRequest,
  GitPullRequestClosed,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import {
  projectIdKey,
  type ProjectConversation,
  type ProjectPullRequest,
} from "#/api/projects-service/projects-service.api";
import type { PullRequestStatus } from "#/api/git-service/pull-request-status.api";
import { useProjectConversations } from "#/hooks/query/use-projects";
import {
  pullRequestRef,
  usePullRequestStatuses,
} from "#/hooks/query/use-pull-request-status";
import { cn } from "#/utils/utils";
import { ProjectPanelGroup } from "./project-panel-group";

const CHECKS_LABEL: Record<PullRequestStatus["checks"], I18nKey | null> = {
  success: I18nKey.PROJECTS$CHECKS_PASSING,
  failure: I18nKey.PROJECTS$CHECKS_FAILING,
  pending: I18nKey.PROJECTS$CHECKS_PENDING,
  none: null,
};

/** A mark for a pull request's CI: passing, failing or still running. */
export function ChecksIcon({
  checks,
  size = 14,
}: {
  checks: PullRequestStatus["checks"];
  size?: number;
}) {
  const { t } = useTranslation("openhands");
  const label = CHECKS_LABEL[checks];
  if (!label) return null;
  const Icon = {
    success: CircleCheck,
    failure: CircleX,
    pending: CircleDashed,
  }[checks as "success" | "failure" | "pending"];
  return (
    <Icon
      width={size}
      height={size}
      role="img"
      aria-label={t(label)}
      data-checks={checks}
      className={cn(
        "shrink-0",
        checks === "success" && "text-success",
        checks === "failure" && "text-danger",
        checks === "pending" && "text-muted",
      )}
    />
  );
}

/** A pull request's icon by state; an unread state shows as open. */
export const PULL_REQUEST_STATE_ICON = {
  open: GitPullRequest,
  merged: GitMerge,
  closed: GitPullRequestClosed,
} as const;

/** A pull request's color by state: open green, merged primary, closed red. */
export const PULL_REQUEST_STATE_COLOR: Record<
  PullRequestStatus["state"],
  string
> = {
  open: "text-success",
  merged: "text-primary",
  closed: "text-danger",
};

/** A pull request a project conversation opened, and who opened it. */
interface OpenedPullRequest {
  pr: ProjectPullRequest;
  opener: string;
}

type Group = "open" | "merged" | "closed" | "unknown";

const GROUPS: Array<{ id: Group; label: I18nKey }> = [
  { id: "open", label: I18nKey.PROJECTS$PRS_OPEN },
  { id: "merged", label: I18nKey.PROJECTS$PRS_MERGED },
  { id: "closed", label: I18nKey.PROJECTS$PRS_CLOSED },
  { id: "unknown", label: I18nKey.PROJECTS$PRS_UNKNOWN },
];

function PullRequestRow({
  opened,
  status,
}: {
  opened: OpenedPullRequest;
  status: PullRequestStatus | undefined;
}) {
  const ref = pullRequestRef(opened.pr);
  const Icon = PULL_REQUEST_STATE_ICON[status?.state ?? "open"];
  return (
    <li>
      <a
        href={
          status?.url ??
          `https://github.com/${opened.pr.owner}/${opened.pr.repo}/pull/${opened.pr.number}`
        }
        target="_blank"
        rel="noopener noreferrer"
        data-testid="project-panel-pr"
        data-state={status?.state ?? "unknown"}
        className="-mx-2 flex gap-2.5 rounded-lg px-2 py-1.5 hover:bg-contrast/5"
      >
        <Icon
          width={16}
          height={16}
          aria-hidden
          className={cn(
            "mt-0.5 shrink-0",
            status ? PULL_REQUEST_STATE_COLOR[status.state] : "text-muted",
          )}
        />
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="truncate text-sm text-foreground">
            {status?.title ?? ref}
          </span>
          <span className="truncate text-xs text-muted">
            {ref} · {opened.opener}
          </span>
        </span>
        {status ? <ChecksIcon checks={status.checks} /> : null}
      </a>
    </li>
  );
}

/**
 * The pull requests the project's conversations opened, by state, each with
 * its CI and the conversation that opened it. A row opens the pull request.
 */
export function ProjectPanelPullRequests({ projectId }: { projectId: string }) {
  const { t } = useTranslation("openhands");
  const { data: board } = useProjectConversations(projectId);

  const opened = React.useMemo(() => {
    const members: Array<{
      conversation: ProjectConversation;
      opener: string;
    }> = [
      ...(board?.coordinator
        ? [
            {
              conversation: board.coordinator,
              opener: t(I18nKey.PROJECTS$COORDINATOR),
            },
          ]
        : []),
      ...(board?.conversations ?? []).map((conversation) => ({
        conversation,
        opener:
          conversation.title ||
          projectIdKey(conversation.conversation_id).slice(0, 8),
      })),
    ];
    const byRef = new Map<string, OpenedPullRequest>();
    for (const { conversation, opener } of members) {
      for (const pr of conversation.pull_requests) {
        if (!byRef.has(pullRequestRef(pr))) {
          byRef.set(pullRequestRef(pr), { pr, opener });
        }
      }
    }
    return [...byRef.values()];
  }, [board, t]);

  const statuses = usePullRequestStatuses(opened.map((o) => o.pr));

  if (!board) {
    return <p className="text-sm text-muted">{t(I18nKey.PROJECTS$LOADING)}</p>;
  }
  if (!opened.length) {
    return (
      <p data-testid="project-panel-prs-empty" className="text-sm text-muted">
        {t(I18nKey.PROJECTS$PRS_EMPTY)}
      </p>
    );
  }

  const groupOf = (o: OpenedPullRequest): Group =>
    statuses.get(pullRequestRef(o.pr))?.state ?? "unknown";

  return (
    <div data-testid="project-panel-prs" className="flex flex-col gap-3">
      {GROUPS.map((group) => {
        // Newest first: a conversation lists its pull requests oldest first.
        const items = opened.filter((o) => groupOf(o) === group.id).reverse();
        return (
          <ProjectPanelGroup
            key={group.id}
            id={`prs-${group.id}`}
            label={t(group.label)}
            count={items.length}
            defaultOpen={group.id !== "closed"}
          >
            {items.map((o) => (
              <PullRequestRow
                key={pullRequestRef(o.pr)}
                opened={o}
                status={statuses.get(pullRequestRef(o.pr))}
              />
            ))}
          </ProjectPanelGroup>
        );
      })}
    </div>
  );
}
