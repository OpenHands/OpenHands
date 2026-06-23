import React from "react";
import { useConfig } from "./query/use-config";
import { useIsAuthed } from "./query/use-is-authed";
import { useUserProviders } from "./use-user-providers";

/**
 * Hook to determine if Git-related features should be shown or enabled
 * based on authentication status and Git provider configuration.
 *
 * Unlike useShouldShowUserFeatures, this hook ALWAYS requires a Git provider
 * to be configured, regardless of app mode. This is because Git features
 * (like fetching user info or suggested tasks) require an actual Git
 * provider token to function.
 *
 * @returns boolean indicating if Git features should be shown
 */
export const useShouldShowGitFeatures = (): boolean => {
  const { data: config } = useConfig();
  const { data: isAuthed } = useIsAuthed();
  const { providers } = useUserProviders();

  return React.useMemo(() => {
    if (!config?.app_mode || !isAuthed) return false;

    // Git features always require a provider to be configured,
    // regardless of app mode (oss, saas, or enterprise)
    return providers.length > 0;
  }, [config?.app_mode, isAuthed, providers.length]);
};
