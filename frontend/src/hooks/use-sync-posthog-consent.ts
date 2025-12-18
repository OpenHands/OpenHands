import React from "react";
import { usePostHog } from "posthog-js/react";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";
import { useSettings } from "./query/use-settings";

/**
 * Hook to sync PostHog opt-in/out state with backend setting on mount.
 * This ensures that if the backend setting changes (e.g., via API or different client),
 * the PostHog instance reflects the current user preference.
 */
export const useSyncPostHogConsent = () => {
  const posthog = usePostHog();
  const { data: settings } = useSettings();
  const hasSyncedRef = React.useRef(false);

  React.useEffect(() => {
    // Only run once when both PostHog and settings are available
    if (!posthog || settings === undefined || hasSyncedRef.current) {
      return;
    }

    const backendConsent = settings.user_consents_to_analytics;

    // Only sync if there's a backend preference set
    if (backendConsent !== null) {
      const posthogHasOptedIn = posthog.has_opted_in_capturing();
      const posthogHasOptedOut = posthog.has_opted_out_capturing();

      // Check if PostHog state is out of sync with backend
      const needsSync =
        (backendConsent === true && !posthogHasOptedIn) ||
        (backendConsent === false && !posthogHasOptedOut);

      if (needsSync) {
        handleCaptureConsent(posthog, backendConsent);
      }

      hasSyncedRef.current = true;
    }
  }, [posthog, settings]);
};
