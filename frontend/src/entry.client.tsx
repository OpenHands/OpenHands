/* eslint-disable react/react-in-jsx-scope */
/**
 * By default, Remix will handle hydrating your app on the client for you.
 * You are free to delete this file if you'd like to, but if you ever want it revealed again, you can run `npx remix reveal` ✨
 * For more information, see https://remix.run/file-conventions/entry.client
 */

import { HydratedRouter } from "react-router/dom";
import React, { startTransition, StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import { PostHogProvider } from "posthog-js/react";
import "./i18n";
import { QueryClientProvider } from "@tanstack/react-query";
import OptionService from "./api/option-service/option-service.api";
import { displayErrorToast } from "./utils/custom-toast-handlers";
import { queryClient } from "./query-client-config";

function PostHogWrapper({ children }: { children: React.ReactNode }) {
  const [posthogClientKey, setPosthogClientKey] = React.useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = React.useState(true);

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
      }}
    >
      {children}
    </PostHogProvider>
  );
}

async function prepareApp() {
  if (
    process.env.NODE_ENV === "development" &&
    import.meta.env.VITE_MOCK_API === "true"
  ) {
    const { worker } = await import("./mocks/browser");

    await worker.start({
      onUnhandledRequest: "bypass",
    });
  }
}

prepareApp().then(() =>
  startTransition(() => {
    hydrateRoot(
      document,
      <StrictMode>
        <QueryClientProvider client={queryClient}>
          <PostHogWrapper>
            <HydratedRouter />
          </PostHogWrapper>
        </QueryClientProvider>
      </StrictMode>,
    );
  }),
);
