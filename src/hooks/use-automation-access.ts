import type { Automation } from "#/types/automation";
import { isDraftAutomation } from "#/utils/automation-state";
import {
  useAutomationPermissions,
  useIsAutomationOwner,
} from "./use-automation-permissions";

export interface AutomationAccess {
  /** Turn off, delete, run now: `manage_automations` or the owner. */
  canManage: boolean;
  /**
   * Edit the automation: `manage_automations` or the owner. Admins and Owners
   * may edit automations that run as another person or as a service account.
   */
  canEdit: boolean;
  /** Flip `enabled`; false for draft test artifacts. */
  canToggle: boolean;
}

/**
 * The caller's write access to one automation. One copy of the rules shared by
 * the card, list row, home actions and detail page.
 */
export function useAutomationAccess(automation: Automation): AutomationAccess {
  const { canManage: hasManagePermission } = useAutomationPermissions();
  const isOwner = useIsAutomationOwner(automation);
  const canManage = hasManagePermission || isOwner;
  // Draft test artifacts are finalized through the draft setup flow, not the toggle.
  const canToggle = canManage && !isDraftAutomation(automation);
  return { canManage, canEdit: canManage, canToggle };
}
