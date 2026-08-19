/**
 * Filter values for the Agents page scope dropdown. Only the levels that the
 * listing endpoint actually returns (built-in, project, user) are offered;
 * `plugin` / `programmatic` agents come from the conversation registry rather
 * than disk discovery and never appear here.
 */
export const AGENT_LEVEL_FILTER_OPTIONS = [
  "all",
  "builtin",
  "project",
  "user",
] as const;

export type AgentLevelFilter = (typeof AGENT_LEVEL_FILTER_OPTIONS)[number];
