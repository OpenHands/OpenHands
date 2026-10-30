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
   * Edit the automation. The service lets the owner edit an automation that
   * runs as them, and `manage_automations` holders edit one that runs as a
   * service account.
   */
  canEdit: boolean;
  /** Flip `enabled`. Turning off needs `canManage`; turning on needs `canEdit`. */
  canToggle: boolean;
}

/**
 * Whether the caller may edit the automation: the owner, or a holder of
 * `manage_automations` when it runs as a service account. Local: always `true`.
 */
export function useCanEditAutomation(automation: Automation): boolean {
  const { canManage: hasManagePermission } = useAutomationPermissions();
  const isOwner = useIsAutomationOwner(automation);
  return (
    isOwner || (automation.run_as_kind === "service" && hasManagePermission)
  );
}

/**
 * The caller's write access to one automation. One copy of the rules shared by
 * the card, list row, home actions and detail page.
 */
export function useAutomationAccess(automation: Automation): AutomationAccess {
  const { canManage: hasManagePermission } = useAutomationPermissions();
  const isOwner = useIsAutomationOwner(automation);
  const canEdit = useCanEditAutomation(automation);
  const canManage = hasManagePermission || isOwner;
  // Draft test artifacts are finalized through the draft setup flow, not the toggle.
  const canToggle = isDraftAutomation(automation)
    ? false
    : automation.enabled
      ? canManage
      : canEdit;
  return { canManage, canEdit, canToggle };
}
