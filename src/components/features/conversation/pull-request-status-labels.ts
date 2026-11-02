import type {
  GitProviderAccessProblem,
  PullRequestChecksState,
  PullRequestLifecycle,
  PullRequestMergeability,
  PullRequestReviewState,
} from "#/api/git-provider-pull-request-status-service";
import { I18nKey } from "#/i18n/declaration";

/** Shared wording for PR state, so every surface describes a PR the same way. */
export const PULL_REQUEST_LIFECYCLE_LABEL: Record<
  PullRequestLifecycle,
  I18nKey
> = {
  draft: I18nKey.CONVERSATION$PR_LIFECYCLE_DRAFT,
  open: I18nKey.CONVERSATION$PR_LIFECYCLE_OPEN,
  merged: I18nKey.CONVERSATION$PR_LIFECYCLE_MERGED,
  closed: I18nKey.CONVERSATION$PR_LIFECYCLE_CLOSED,
  unknown: I18nKey.CONVERSATION$PR_LIFECYCLE_UNKNOWN,
};

export const PULL_REQUEST_CHECKS_LABEL: Record<
  PullRequestChecksState,
  I18nKey
> = {
  passing: I18nKey.CONVERSATION$PR_CHECKS_PASSING,
  pending: I18nKey.CONVERSATION$PR_CHECKS_PENDING,
  failing: I18nKey.CONVERSATION$PR_CHECKS_FAILING,
  none: I18nKey.CONVERSATION$PR_CHECKS_NONE,
  unknown: I18nKey.CONVERSATION$PR_CHECKS_UNKNOWN,
};

export const PULL_REQUEST_REVIEW_LABEL: Record<
  PullRequestReviewState,
  I18nKey
> = {
  approved: I18nKey.CONVERSATION$PR_REVIEW_APPROVED,
  changes_requested: I18nKey.CONVERSATION$PR_REVIEW_CHANGES_REQUESTED,
  none: I18nKey.CONVERSATION$PR_REVIEW_NONE,
  unknown: I18nKey.CONVERSATION$PR_REVIEW_UNKNOWN,
};

// `not_applicable` (merged or closed) has no label: conflicts no longer matter.
export const PULL_REQUEST_MERGE_LABEL: Record<
  Exclude<PullRequestMergeability, "not_applicable">,
  I18nKey
> = {
  clean: I18nKey.CONVERSATION$PR_MERGE_CLEAN,
  conflicting: I18nKey.CONVERSATION$PR_MERGE_CONFLICTING,
  unknown: I18nKey.CONVERSATION$PR_MERGE_UNKNOWN,
};

export const PULL_REQUEST_UNAVAILABLE_LABEL: Record<
  GitProviderAccessProblem,
  I18nKey
> = {
  not_found: I18nKey.CONVERSATION$PR_UNAVAILABLE_NOT_FOUND,
  auth_failed: I18nKey.CONVERSATION$PR_UNAVAILABLE_AUTH_FAILED,
  forbidden: I18nKey.CONVERSATION$PR_UNAVAILABLE_FORBIDDEN,
  rate_limited: I18nKey.CONVERSATION$PR_UNAVAILABLE_RATE_LIMITED,
  error: I18nKey.CONVERSATION$PR_UNAVAILABLE_ERROR,
};
