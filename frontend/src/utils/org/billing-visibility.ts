import { WebClientConfig } from "#/api/option-service/option.types";

/**
 * Determines whether billing should be hidden based on feature flags and user permissions.
 *
 * Returns true when billing UI should NOT be shown. This is the single source of truth
 * for billing visibility decisions across loaders and hooks.
 *
 * @param config - The application config. When undefined (not yet loaded), billing is
 *   hidden as a safe default to prevent unauthorized access during loading.
 * @param hasViewBillingPermission - Whether the current user has the view_billing permission.
 */
export function isBillingHidden(
  config: WebClientConfig | undefined,
  hasViewBillingPermission: boolean,
): boolean {
  if (!config) return true;
  return !config.feature_flags?.enable_billing || !hasViewBillingPermission;
}
