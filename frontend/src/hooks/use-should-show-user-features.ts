import React from "react";
import { useConfig } from "./query/use-config";
import { useIsAuthed } from "./query/use-is-authed";
import { useUserProviders } from "./use-user-providers";

/**
 * Hook to determine if user-related features should be shown or enabled
 * based on authentication status and provider configuration.
 *
 * @returns boolean indicating if user features should be shown
 */
export const useShouldShowUserFeatures = (): boolean => {
  const { data: config } = useConfig();
  const { data: isAuthed } = useIsAuthed();
  const { providers } = useUserProviders();

  return React.useMemo(() => {
    if (!config?.app_mode || !isAuthed) return false;

    // In OSS mode, only show user features if Git providers are configured
    if (config.app_mode === "oss") {
      return providers.length > 0;
    }

    // In non-OSS modes (saas), always show user features when authenticated
    return true;
  }, [config?.app_mode, isAuthed, providers.length]);
};
