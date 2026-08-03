import type { LucideIcon } from "lucide-react";
import {
  ClipboardList,
  GitPullRequest,
  MessageSquareText,
  NotebookPen,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";

export type RecommendedAutomationCard = {
  /** Catalog automation id (matches `@openhands/extensions/automations`). */
  id: string;
  labelKey: I18nKey;
  Icon: LucideIcon;
  /** Lucide stroke color for the card icon. */
  iconColor: string;
  href: string;
};

/**
 * Home recommended-automations rail cards, drawn from real catalog use-cases.
 * Short prompt-style labels; links open the automations page for setup.
 */
export const HOME_RECOMMENDED_AUTOMATION_CARDS: readonly RecommendedAutomationCard[] =
  [
    {
      id: "github-pr-reviewer",
      labelKey: I18nKey.FEATURED_AUTOMATIONS$RECOMMENDED_PR_REVIEW,
      Icon: GitPullRequest,
      iconColor: "#5B9FFF",
      href: "/automations",
    },
    {
      id: "github-repo-monitor",
      labelKey: I18nKey.FEATURED_AUTOMATIONS$RECOMMENDED_GITHUB_MENTIONS,
      Icon: MessageSquareText,
      iconColor: "#A78BFA",
      href: "/automations",
    },
    {
      id: "slack-standup-digest",
      labelKey: I18nKey.FEATURED_AUTOMATIONS$RECOMMENDED_SLACK_STANDUP,
      Icon: NotebookPen,
      iconColor: "#4ADE80",
      href: "/automations",
    },
    {
      id: "linear-triage-assistant",
      labelKey: I18nKey.FEATURED_AUTOMATIONS$RECOMMENDED_LINEAR_TRIAGE,
      Icon: ClipboardList,
      iconColor: "#FB923C",
      href: "/automations",
    },
  ];
