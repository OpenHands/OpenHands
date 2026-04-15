import { redirect } from "react-router";
import { queryClient } from "#/query-client-config";
import OptionService from "#/api/option-service/option-service.api";
import { WebClientConfig } from "#/api/option-service/option.types";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "#/hooks/query/query-keys";
import { getFirstAvailablePath } from "#/utils/settings-utils";
import { getActiveOrganizationUser } from "./permission-checks";
import { PermissionKey, rolePermissions } from "./permissions";

/**
 * Helper to get config, using fetchQuery for automatic caching and deduplication.
 * Uses the shared query key from QUERY_KEYS to ensure consistency across the app.
 */
async function getConfig(): Promise<WebClientConfig | undefined> {
  return queryClient.fetchQuery({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    ...CONFIG_CACHE_OPTIONS,
  });
}

/**
 * Gets the appropriate fallback path for permission denied scenarios.
 * Respects feature flags to avoid redirecting to hidden pages.
 */
async function getPermissionDeniedFallback(): Promise<string> {
  const config = await getConfig();

  const isSaas = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;

  // Get first available path that respects feature flags
  const fallbackPath = getFirstAvailablePath(isSaas, featureFlags);
  return fallbackPath ?? "/settings";
}

/**
 * Creates a clientLoader guard that checks if the user has the required permission.
 * Redirects to the first available settings page if permission is denied.
 *
 * In OSS mode, permission checks are bypassed since there are no user roles.
 *
 * @param requiredPermission - The permission key to check
 * @param customRedirectPath - Optional custom path to redirect to (will still respect feature flags if not provided)
 * @returns A clientLoader function that can be exported from route files
 */
export const createPermissionGuard =
  (requiredPermission: PermissionKey, customRedirectPath?: string) =>
  async ({ request }: { request: Request }) => {
    // Get config to check app_mode
    const config = await getConfig();

    // In OSS mode, skip permission checks - all settings are accessible
    if (config?.app_mode === "oss") {
      return null;
    }

    const user = await getActiveOrganizationUser();

    const url = new URL(request.url);
    const currentPath = url.pathname;

    // Helper to get redirect response, avoiding infinite loops
    const getRedirectResponse = async () => {
      const redirectPath =
        customRedirectPath ?? (await getPermissionDeniedFallback());
      // Don't redirect to the same path to avoid infinite loops
      if (redirectPath === currentPath) {
        return null;
      }
      return redirect(redirectPath);
    };

    if (!user) {
      return getRedirectResponse();
    }

    const userRole = user.role ?? "member";

    if (!rolePermissions[userRole].includes(requiredPermission)) {
      return getRedirectResponse();
    }

    return null;
  };
