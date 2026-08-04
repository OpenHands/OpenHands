import {
  Activity,
  CalendarDays,
  FileText,
  GitBranch,
  GitMerge,
  GitPullRequest,
  Inbox,
  Package,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";

export type ChatPromptSuggestionId =
  | "standup-digest"
  | "ship-report"
  | "ci-watchdog"
  | "dependency-audit"
  | "pr-review-digest"
  | "release-notes"
  | "flaky-test-digest"
  | "issue-triage"
  | "stale-branch-cleanup";

export interface ChatPromptSuggestion {
  id: ChatPromptSuggestionId;
  labelKey: I18nKey;
  promptKey: I18nKey;
  icon: LucideIcon;
}

/** Scheduled-task starters aimed at day-to-day developer workflows. */
export const CHAT_PROMPT_SUGGESTIONS: ChatPromptSuggestion[] = [
  {
    id: "standup-digest",
    labelKey: I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST,
    promptKey: I18nKey.AUTOMATIONS$STARTER_STANDUP_DIGEST_PROMPT,
    icon: Sparkles,
  },
  {
    id: "ship-report",
    labelKey: I18nKey.AUTOMATIONS$STARTER_SHIP_REPORT,
    promptKey: I18nKey.AUTOMATIONS$STARTER_SHIP_REPORT_PROMPT,
    icon: CalendarDays,
  },
  {
    id: "ci-watchdog",
    labelKey: I18nKey.AUTOMATIONS$STARTER_CI_WATCHDOG,
    promptKey: I18nKey.AUTOMATIONS$STARTER_CI_WATCHDOG_PROMPT,
    icon: GitBranch,
  },
  {
    id: "dependency-audit",
    labelKey: I18nKey.HOME$SUGGESTION_DEPENDENCY_AUDIT,
    promptKey: I18nKey.HOME$SUGGESTION_DEPENDENCY_AUDIT_PROMPT,
    icon: Package,
  },
  {
    id: "pr-review-digest",
    labelKey: I18nKey.HOME$SUGGESTION_PR_REVIEW_DIGEST,
    promptKey: I18nKey.HOME$SUGGESTION_PR_REVIEW_DIGEST_PROMPT,
    icon: GitPullRequest,
  },
  {
    id: "release-notes",
    labelKey: I18nKey.HOME$SUGGESTION_RELEASE_NOTES,
    promptKey: I18nKey.HOME$SUGGESTION_RELEASE_NOTES_PROMPT,
    icon: FileText,
  },
  {
    id: "flaky-test-digest",
    labelKey: I18nKey.HOME$SUGGESTION_FLAKY_TEST_DIGEST,
    promptKey: I18nKey.HOME$SUGGESTION_FLAKY_TEST_DIGEST_PROMPT,
    icon: Activity,
  },
  {
    id: "issue-triage",
    labelKey: I18nKey.HOME$SUGGESTION_ISSUE_TRIAGE,
    promptKey: I18nKey.HOME$SUGGESTION_ISSUE_TRIAGE_PROMPT,
    icon: Inbox,
  },
  {
    id: "stale-branch-cleanup",
    labelKey: I18nKey.HOME$SUGGESTION_STALE_BRANCH_CLEANUP,
    promptKey: I18nKey.HOME$SUGGESTION_STALE_BRANCH_CLEANUP_PROMPT,
    icon: GitMerge,
  },
];
