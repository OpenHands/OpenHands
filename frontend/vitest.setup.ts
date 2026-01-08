import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { server } from "#/mocks/node";
import "@testing-library/jest-dom/vitest";

HTMLCanvasElement.prototype.getContext = vi.fn();
HTMLElement.prototype.scrollTo = vi.fn();
window.scrollTo = vi.fn();

// Mock ResizeObserver for test environment
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

// Mock the i18n provider
vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: {
      language: "en",
      exists: () => false,
    },
  }),
}));

vi.mock("#/hooks/use-is-on-tos-page", () => ({
  useIsOnTosPage: () => false,
}));

// Import the Zustand mock to enable automatic store resets
vi.mock("zustand");

// Mock requests during tests
beforeAll(() => {
  server.listen({ onUnhandledRequest: "bypass" });
  vi.stubGlobal("ResizeObserver", MockResizeObserver);

  // Create modal portal element for tests
  if (!document.getElementById("modal-portal-exit")) {
    const portalDiv = document.createElement("div");
    portalDiv.id = "modal-portal-exit";
    document.body.appendChild(portalDiv);
  }
});
afterEach(() => {
  server.resetHandlers();
  // Cleanup the document body after each test
  cleanup();

  // Clear modal portal content after each test
  const portal = document.getElementById("modal-portal-exit");
  if (portal) {
    portal.innerHTML = "";
  }
});
afterAll(() => {
  server.close();
  vi.unstubAllGlobals();
});
