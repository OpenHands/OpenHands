/* eslint-disable react/react-in-jsx-scope */
/**
 * By default, Remix will handle hydrating your app on the client for you.
 * You are free to delete this file if you'd like to, but if you ever want it revealed again, you can run `npx remix reveal` ✨
 * For more information, see https://remix.run/file-conventions/entry.client
 */

import { HydratedRouter } from "react-router/dom";
import { startTransition, StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import "./i18n";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./query-client-config";
import { PostHogWrapper } from "./components/providers/posthog-wrapper";

async function prepareApp() {
  if (
    process.env.NODE_ENV === "development" &&
    import.meta.env.VITE_MOCK_API === "true"
  ) {
    const { worker } = await import("./mocks/browser");

    await worker.start({
      onUnhandledRequest: "bypass",
    });

    // Expose Zustand stores on window so Playwright E2E tests can inject
    // state directly (e.g. to simulate a long conversation) without going
    // through the WebSocket layer.  Only active in mock/dev mode.
    const { useEventStore } = await import("./stores/use-event-store");
    (window as any).__stores = { eventStore: useEventStore }; // eslint-disable-line @typescript-eslint/no-explicit-any
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
