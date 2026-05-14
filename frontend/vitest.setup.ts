import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { server } from "#/mocks/node";
import "@testing-library/jest-dom/vitest";

HTMLCanvasElement.prototype.getContext = vi.fn();
HTMLElement.prototype.scrollTo = vi.fn();
window.scrollTo = vi.fn();

// Some Node builds expose an experimental localStorage object without the
// Storage methods unless --localstorage-file is configured.
if (typeof globalThis.localStorage?.getItem !== "function") {
  const storage = new Map<string, string>();
  const memoryLocalStorage: Storage = {
    get length() {
      return storage.size;
    },
    clear: () => storage.clear(),
    getItem: (key: string) => storage.get(key) ?? null,
    key: (index: number) => Array.from(storage.keys())[index] ?? null,
    removeItem: (key: string) => {
      storage.delete(key);
    },
    setItem: (key: string, value: string) => {
      storage.set(key, value);
    },
  };

  vi.stubGlobal("localStorage", memoryLocalStorage);
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: memoryLocalStorage,
  });
}

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

vi.mock("#/hooks/use-is-on-intermediate-page", () => ({
  useIsOnIntermediatePage: () => false,
}));

// Mock useRevalidator from react-router to allow direct store manipulation
// in tests instead of mocking useSelectedOrganizationId hook
vi.mock("react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router")>()),
  useRevalidator: () => ({
    revalidate: vi.fn(),
  }),
}));

// Import the Zustand mock to enable automatic store resets
vi.mock("zustand");

// Mock requests during tests
beforeAll(() => {
  server.listen({ onUnhandledRequest: "bypass" });
  vi.stubGlobal("ResizeObserver", MockResizeObserver);
});
afterEach(() => {
  server.resetHandlers();
  // Cleanup the document body after each test
  cleanup();
});
afterAll(() => {
  server.close();
  vi.unstubAllGlobals();
});
