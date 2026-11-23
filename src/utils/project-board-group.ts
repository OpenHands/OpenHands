import type { PullRequestStatus } from "#/api/git-service/pull-request-status.api";
import type {
  ProjectAttention,
  ProjectConversation,
} from "#/api/projects-service/projects-service.api";
import { I18nKey } from "#/i18n/declaration";
import { pullRequestRef } from "#/hooks/query/use-pull-request-status";
import { pullRequestAttention } from "#/utils/pull-request-attention";

/** The board's columns, most urgent first, and the attention each one holds. */
export const BOARD_GROUPS: Array<{
  id: "needs_you" | "working" | "idle" | "resolved";
  label: I18nKey;
  attention: ProjectAttention[];
}> = [
  {
    id: "needs_you",
    label: I18nKey.PROJECTS$GROUP_NEEDS_YOU,
    attention: ["needs_you", "asking"],
  },
  {
    id: "working",
    label: I18nKey.PROJECTS$GROUP_WORKING,
    attention: ["working"],
  },
  {
    id: "idle",
    label: I18nKey.PROJECTS$GROUP_IDLE,
    attention: ["idle"],
  },
  {
    id: "resolved",
    label: I18nKey.PROJECTS$GROUP_RESOLVED,
    attention: ["resolved"],
  },
];

export type BoardGroupId = (typeof BOARD_GROUPS)[number]["id"];

/**
 * The pull request that moved an idle conversation to Needs you: the first
 * one ready for the user's review. `undefined` for a conversation that is
 * not idle, so a working one and a resolved one keep their place.
 */
export const readyPullRequestOf = (
  conversation: ProjectConversation,
  statuses: Map<string, PullRequestStatus>,
) =>
  conversation.attention === "idle"
    ? conversation.pull_requests.find(
        (pr) =>
          pullRequestAttention(statuses.get(pullRequestRef(pr))) === "ready",
      )
    : undefined;

/**
 * The board group a conversation sits in, in the board and the project
 * sidebar alike: its attention, except that an idle conversation with a pull
 * request ready for the user's review needs them.
 */
export function boardGroupOf(
  conversation: ProjectConversation,
  statuses: Map<string, PullRequestStatus>,
): BoardGroupId {
  if (readyPullRequestOf(conversation, statuses)) return "needs_you";
  return (
    BOARD_GROUPS.find((group) =>
      group.attention.includes(conversation.attention),
    )?.id ?? "idle"
  );
}
