import React from "react";
import { PostHogProvider } from "posthog-js/react";
import OptionService from "#/api/option-service/option-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const POSTHOG_BOOTSTRAP_KEY = "posthog_bootstrap";

function getBootstrapIds() {
  // Try to extract from URL hash (e.g. #distinct_id=abc&session_id=xyz)
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const distinctId = params.get("distinct_id");
  const sessionId = params.get("session_id");

  if (distinctId && sessionId) {
    const bootstrap = { distinctID: distinctId, sessionID: sessionId };

    // Persist to sessionStorage so IDs survive full-page OAuth redirects
    sessionStorage.setItem(POSTHOG_BOOTSTRAP_KEY, JSON.stringify(bootstrap));

    // Clean the hash from the URL
    window.history.replaceState(
      null,
      "",
      window.location.pathname + window.location.search,
    );
    return bootstrap;
  }

  // Fallback: check sessionStorage (covers return from OAuth redirect)
  const stored = sessionStorage.getItem(POSTHOG_BOOTSTRAP_KEY);
  if (stored) {
    sessionStorage.removeItem(POSTHOG_BOOTSTRAP_KEY);
    return JSON.parse(stored) as { distinctID: string; sessionID: string };
  }

  return undefined;
}

export function PostHogWrapper({ children }: { children: React.ReactNode }) {
  const [posthogClientKey, setPosthogClientKey] = React.useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = React.useState(true);
  const bootstrapIds = React.useMemo(() => getBootstrapIds(), []);

  React.useEffect(() => {
    (async () => {
      try {
        const config = await OptionService.getConfig();
        setPosthogClientKey(config.POSTHOG_CLIENT_KEY);
      } catch {
        displayErrorToast("Error fetching PostHog client key");
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  if (isLoading || !posthogClientKey) {
    return children;
  }

  return (
    <PostHogProvider
      apiKey={posthogClientKey}
      options={{
        api_host: "https://us.i.posthog.com",
        person_profiles: "identified_only",
        bootstrap: bootstrapIds,
      }}
    >
      {children}
    </PostHogProvider>
  );
}
