import type { ConversationMode } from "#/stores/conversation-store";

/**
 * Modes whose messages run in the planner helper rather than the code agent.
 * Deep Planning is a planning mode with extra phase gating, so anything that
 * routes on "is this plan mode?" must route on this predicate instead of
 * comparing against `"plan"` directly — otherwise deep-plan messages would
 * leak into the code agent.
 */
export const isPlanningMode = (mode: ConversationMode): boolean =>
  mode === "plan" || mode === "deep-plan";
