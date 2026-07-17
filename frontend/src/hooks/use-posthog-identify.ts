import React from "react";
import { usePostHog } from "posthog-js/react";
import { useConfig } from "./query/use-config";
import { useMe } from "./query/use-me";
import { useGitUser } from "./query/use-git-user";
import { useSettings } from "./query/use-settings";

/**
 * Identifies the current user to PostHog using the same distinct_id
 * that the server-side AnalyticsService uses (keycloak user_id in SaaS
 * mode). This ensures cross-domain tracking works: the anonymous
 * distinct_id bootstrapped from the marketing site gets merged with
 * the keycloak user_id that every server-side event uses.
 *
 * In OSS mode, falls back to the Git user login.
 *
 * Identification is gated on analytics consent, matching the agent canvas
 * UI:
 *  - consent === true  → posthog.identify(...)
 *  - consent === false → posthog.reset() (undo a prior identify)
 *  - consent === null / settings loading → no-op (wait for a decision)
 */
export const usePostHogIdentify = () => {
  const posthog = usePostHog();
  const { data: config } = useConfig();
  const { data: me } = useMe();
  const { data: gitUser } = useGitUser();
  const { data: settings } = useSettings();
  const hasIdentifiedRef = React.useRef(false);

  const consent = settings?.user_consents_to_analytics;

  React.useEffect(() => {
    if (!posthog || settings === undefined) return;

    // Reset on explicit denial to undo any prior identify.
    if (consent === false) {
      if (hasIdentifiedRef.current) {
        posthog.reset();
        hasIdentifiedRef.current = false;
      }
      return;
    }

    // Wait for an explicit consent decision before identifying.
    if (consent !== true || hasIdentifiedRef.current) return;

    if (config?.app_mode === "saas" && me?.user_id) {
      posthog.identify(me.user_id, {
        email: me.email,
      });
      hasIdentifiedRef.current = true;
    } else if (config?.app_mode === "oss" && gitUser) {
      posthog.identify(gitUser.login, {
        company: gitUser.company,
        name: gitUser.name,
        email: gitUser.email,
        user: gitUser.login,
        mode: "oss",
      });
      hasIdentifiedRef.current = true;
    }
  }, [posthog, config?.app_mode, me, gitUser, consent, settings]);
};
