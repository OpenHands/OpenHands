import React from "react";
import { usePostHog } from "posthog-js/react";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";
import { useSaveSettings } from "./mutation/use-save-settings";
import { useConfig } from "./query/use-config";

export const useMigrateUserConsent = () => {
  const posthog = usePostHog();
  const { mutate: saveUserSettings } = useSaveSettings();
  const { data: config } = useConfig();

  /**
   * Migrate user consent to the settings store on the server.
   */
  const migrateUserConsent = React.useCallback(
    async (args?: { handleAnalyticsWasPresentInLocalStorage: () => void }) => {
      const userAnalyticsConsent = localStorage.getItem("analytics-consent");

      if (userAnalyticsConsent) {
        if (!config?.app_mode) return;

        args?.handleAnalyticsWasPresentInLocalStorage();

        if (config.app_mode === "saas") {
          localStorage.removeItem("analytics-consent");
          return;
        }

        saveUserSettings(
          { user_consents_to_analytics: userAnalyticsConsent === "true" },
          {
            onSuccess: () => {
              handleCaptureConsent(posthog, userAnalyticsConsent === "true");
            },
          },
        );

        localStorage.removeItem("analytics-consent");
      }
    },
    [config?.app_mode, posthog, saveUserSettings],
  );

  return { migrateUserConsent };
};
