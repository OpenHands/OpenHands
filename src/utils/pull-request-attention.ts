import type { PullRequestStatus } from "#/api/git-service/pull-request-status.api";
import { I18nKey } from "#/i18n/declaration";

/**
 * What a pull request asks of the user, from the user's side rather than
 * GitHub's: ready for their review, failing, still on its way (checks
 * running, a draft), or finished (merged, closed). `unknown` when its state
 * could not be read.
 */
export type PullRequestAttention =
  "ready" | "failing" | "checking" | "draft" | "merged" | "closed" | "unknown";

export function pullRequestAttention(
  status: PullRequestStatus | undefined,
): PullRequestAttention {
  if (!status) return "unknown";
  if (status.state !== "open") return status.state;
  if (status.checks === "failure") return "failing";
  if (status.draft) return "draft";
  if (status.checks === "pending") return "checking";
  return "ready";
}

/** Most pressing first: what needs the user, then what is on its way, then
 * what is done. */
export const ATTENTION_ORDER: readonly PullRequestAttention[] = [
  "ready",
  "failing",
  "checking",
  "draft",
  "unknown",
  "merged",
  "closed",
];

export const ATTENTION_LABEL: Record<PullRequestAttention, I18nKey> = {
  ready: I18nKey.PROJECTS$PR_READY_HINT,
  failing: I18nKey.PROJECTS$CHECKS_FAILING,
  checking: I18nKey.PROJECTS$CHECKS_PENDING,
  draft: I18nKey.PROJECTS$PR_DRAFT,
  merged: I18nKey.PROJECTS$PRS_MERGED,
  closed: I18nKey.PROJECTS$PRS_CLOSED,
  unknown: I18nKey.PROJECTS$PRS_UNKNOWN,
};
