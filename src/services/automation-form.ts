import {
  patchAutomationSetupDraft,
  type AutomationSetupFormPatch,
  type AutomationSetupPatchResult,
} from "#/api/automation-setup-draft-store";
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

  return patchAutomationSetupDraft(
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
