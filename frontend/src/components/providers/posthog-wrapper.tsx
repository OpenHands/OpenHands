import React from "react";
import { PostHogProvider } from "posthog-js/react";
import OptionService from "#/api/option-service/option-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

function getBootstrapFromHash() {
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const distinctId = params.get("ph_distinct_id");
  const sessionId = params.get("ph_session_id");

  if (distinctId && sessionId) {
    // Remove the PostHog tracking params from URL hash to keep URL clean
    // replaceState(state, unused, url) - we pass null state, empty title (ignored by browsers), and the clean URL
    window.history.replaceState(
      null,
      "",
      window.location.pathname + window.location.search,
    );
    return { distinctID: distinctId, sessionID: sessionId };
  }
  return undefined;
}

export function PostHogWrapper({ children }: { children: React.ReactNode }) {
  const [posthogClientKey, setPosthogClientKey] = React.useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = React.useState(true);
  const bootstrapIds = React.useMemo(() => getBootstrapFromHash(), []);

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
