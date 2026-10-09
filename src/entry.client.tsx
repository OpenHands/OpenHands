/**
 * By default, Remix will handle hydrating your app on the client for you.
 * You are free to delete this file if you'd like to, but if you ever want it revealed again, you can run `npx remix reveal` ✨
 * For more information, see https://remix.run/file-conventions/entry.client
 */

import { HydratedRouter } from "react-router/dom";
import { startTransition, StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import {
  AgentServerUIProviders,
  DEFAULT_AGENT_SERVER_ANALYTICS,
} from "./components/providers";
import { installMarsWebBridge } from "./api/mars/mars-web-bridge";
import { waitForI18n } from "./i18n";
import { shouldStartMockWorker } from "./mocks/should-start-mock-worker";

async function prepareApp() {
  await waitForI18n();

  // DigitalOcean Managed Agents in the browser: present only when the server
  // this page came from hosts the MARS bridge (scripts/mars-web-bridge.mjs).
  // Must settle before render — the UI reads `window.marsBridge` synchronously.
  await installMarsWebBridge();

  if (shouldStartMockWorker()) {
    const { worker } = await import("./mocks/browser");

    await worker.start({
      onUnhandledRequest: "bypass",
    });
  }

  if (import.meta.env.DEV) {
    const { installPendingChatPreview } =
      await import("./dev/seed-pending-chat-preview");
    installPendingChatPreview();
  }
}

prepareApp().then(() =>
  startTransition(() => {
    hydrateRoot(
      document,
      <StrictMode>
        <AgentServerUIProviders
          analytics={DEFAULT_AGENT_SERVER_ANALYTICS}
          withStyleRoot={false}
        >
          <HydratedRouter />
        </AgentServerUIProviders>
      </StrictMode>,
    );
  }),
);
