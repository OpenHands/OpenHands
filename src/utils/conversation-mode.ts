import type { ConversationMode } from "#/stores/conversation-store";
import type { DeepPlanPhaseId } from "#/utils/deep-plan";

/**
 * Modes whose messages run in the planner helper rather than the code agent.
 * Deep Planning is a planning mode with extra phase gating, so anything that
 * routes on "is this plan mode?" must route on this predicate instead of
 * comparing against `"plan"` directly — otherwise deep-plan messages would
 * leak into the code agent.
 *
 * Deep Planning's final phase is the exception: its instruction says "before
 * implementing a task… work one task at a time", i.e. it *executes* the tasks,
 * and the planner's own boundaries forbid exactly that. Routing Implementation
 * to the planner would make the phase impossible, so the chain hands its last
 * phase to the code agent. Pass the active phase; the mode alone cannot tell
 * whether the chain has reached Implementation.
 */
export const isPlanningMode = (
  mode: ConversationMode,
  deepPlanPhase: DeepPlanPhaseId | null = null,
): boolean => {
  if (mode === "plan") return true;
  if (mode === "deep-plan") return deepPlanPhase !== "implementation";
  return false;
};
