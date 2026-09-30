import { useCallback } from "react";
import {
  PENDING_AUTOMATION_SETUP_ID,
  initializeAutomationFormSession,
} from "#/api/automation-form-session";
import { useNavigation } from "#/context/navigation-context";
import type { Automation } from "#/types/automation";
import { setupDraftFromAutomation } from "#/utils/automation-edit-draft";

/**
 * Open an existing automation in the setup page.
 *
 * Edit reuses the collapsed setup form before any conversation exists. The
 * agent conversation starts only if the user sends a prompt from that form.
 */
export function useOpenAutomationEditor() {
  const { navigate } = useNavigation();

  const openEditor = useCallback(
    (automation: Automation) => {
      initializeAutomationFormSession(
        PENDING_AUTOMATION_SETUP_ID,
        setupDraftFromAutomation(automation),
      );
      navigate?.("/automations/setup");
    },
    [navigate],
  );

  return { openEditor };
}
