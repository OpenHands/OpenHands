import type { Automation } from "#/types/automation";

export const AUTOMATION_STATE_ACTIVE = "ACTIVE";
export const AUTOMATION_STATE_INACTIVE = "INACTIVE";
export const AUTOMATION_STATE_DRAFT = "DRAFT";

export type AutomationLifecycleState = "active" | "inactive" | "draft";

type AutomationStateSource = Pick<Automation, "enabled" | "state">;

export function getAutomationLifecycleState(
  automation: AutomationStateSource,
): AutomationLifecycleState {
  if (automation.state === AUTOMATION_STATE_DRAFT) return "draft";
  if (automation.state === AUTOMATION_STATE_ACTIVE) return "active";
  if (automation.state === AUTOMATION_STATE_INACTIVE) return "inactive";
  return automation.enabled ? "active" : "inactive";
}

export function isDraftAutomation(automation: AutomationStateSource): boolean {
  return getAutomationLifecycleState(automation) === "draft";
}

export const CONTINUE_CONVERSATION_DESTINATION = "continue_conversation";

/**
 * The service keeps the sandbox of a `continue_conversation` automation alive
 * regardless of `keep_alive`, because the next event resumes its conversation.
 */
export function isKeepAliveForced(
  automation: Pick<Automation, "trigger">,
): boolean {
  return (
    automation.trigger.type === "event" &&
    automation.trigger.destination === CONTINUE_CONVERSATION_DESTINATION
  );
}
