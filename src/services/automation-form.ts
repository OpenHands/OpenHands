import { patchAutomationFormSession } from "#/api/automation-form-session";
import type {
  AutomationSetupFormPatch,
  AutomationSetupPatchResult,
} from "#/api/automation-setup-types";
import type { AutomationFormUpdateAction } from "#/types/agent-server/core";

export function handleAutomationFormUpdateAction(
  action: AutomationFormUpdateAction,
  conversationId: string | null,
  eventId?: string | null,
  timestamp?: string,
): AutomationSetupPatchResult {
  if (!conversationId) {
    return { applied: [], skipped: [], duplicate: false };
  }

  return patchAutomationFormSession(
    conversationId,
    action.fields as AutomationSetupFormPatch,
    {
      source: "agent",
      overwriteUserEdits: action.overwrite_user_edits === true,
      eventId,
      updatedAt: timestamp,
    },
  );
}
