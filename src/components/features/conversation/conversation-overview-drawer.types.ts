export const CONVERSATION_OVERVIEW_DRAWER_SECTION = {
  automations: "automations",
  skills: "skills",
  mcp: "mcp",
  secrets: "secrets",
  pull_requests: "pull_requests",
  issues: "issues",
} as const;

export type ConversationOverviewDrawerSection =
  (typeof CONVERSATION_OVERVIEW_DRAWER_SECTION)[keyof typeof CONVERSATION_OVERVIEW_DRAWER_SECTION];

export interface ConversationOverviewDrawerOpenOptions {
  openAdd?: boolean;
}
