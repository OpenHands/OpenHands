import type { SuperAdminSetupStepId } from "./super-admin-setup-guide.constants";

/** Dispatched when the action behind a setup guide step succeeds in Canvas. */
export const SUPER_ADMIN_SETUP_STEP_EVENT = "oh-super-admin-setup-step";

/** Tell the setup guide that a step's action just succeeded. */
export function notifySuperAdminSetupStep(stepId: SuperAdminSetupStepId) {
  window.dispatchEvent(
    new CustomEvent(SUPER_ADMIN_SETUP_STEP_EVENT, { detail: { id: stepId } }),
  );
}
