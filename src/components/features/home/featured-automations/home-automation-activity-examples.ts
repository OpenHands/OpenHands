import { AutomationRunStatus } from "#/types/automation";

/**
 * UI-prototype fixtures for the home composer activity list.
 * Not wired to the automation API — replace when backend integration lands.
 */
export interface HomeAutomationActivityExample {
  id: string;
  name: string;
  triggerSummary: string;
  status: AutomationRunStatus;
  /** Relative-looking timestamp label for the prototype (not parsed). */
  whenLabel: string;
  conversationId: string | null;
}

export const HOME_AUTOMATION_ACTIVITY_EXAMPLES: HomeAutomationActivityExample[] =
  [
    {
      id: "example-pr-triage",
      name: "PR Triage Digest",
      triggerSummary: "Every weekday at 09:00",
      status: AutomationRunStatus.RUNNING,
      whenLabel: "2m ago",
      conversationId: "example-conv-pr-triage",
    },
    {
      id: "example-security-pass",
      name: "Nightly Security Pass",
      triggerSummary: "Daily at 02:00",
      status: AutomationRunStatus.PENDING,
      whenLabel: "Queued",
      conversationId: null,
    },
    {
      id: "example-docs-sync",
      name: "Docs Sync on Push",
      triggerSummary: "On push to main",
      status: AutomationRunStatus.COMPLETED,
      whenLabel: "18m ago",
      conversationId: "example-conv-docs-sync",
    },
    {
      id: "example-release-notes",
      name: "Release Notes Generator",
      triggerSummary: "On tag v*",
      status: AutomationRunStatus.COMPLETED,
      whenLabel: "1h ago",
      conversationId: "example-conv-release-notes",
    },
    {
      id: "example-pr-review",
      name: "PR Review on Open",
      triggerSummary: "On pull_request opened",
      status: AutomationRunStatus.FAILED,
      whenLabel: "3h ago",
      conversationId: "example-conv-pr-review",
    },
  ];
