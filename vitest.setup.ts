import { afterAll, afterEach, beforeAll, beforeEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { server } from "#/mocks/node";
import "@testing-library/jest-dom/vitest";

// Some modules read env at import time before Vitest's per-test hooks run.
// The beforeEach below restores the same default after tests call
// `vi.unstubAllEnvs()`.
vi.stubEnv("VITE_SESSION_API_KEY", "test-session-key");

if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = vi.fn();
}

if (typeof HTMLElement !== "undefined") {
  HTMLElement.prototype.scrollTo = vi.fn();
}

const windowStub =
  typeof window === "undefined"
    ? ({ event: undefined } as unknown as Window & typeof globalThis)
    : window;

vi.stubGlobal("window", windowStub);
windowStub.scrollTo = vi.fn();

// Node.js 25+ ships a built-in localStorage that requires --localstorage-file
// and is not functional without it. Stub it with a plain in-memory
// implementation so zustand's persist middleware works in tests.
if (
  typeof localStorage === "undefined" ||
  typeof localStorage.setItem !== "function"
) {
  const store: Record<string, string> = {};
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      Object.keys(store).forEach((k) => delete store[k]);
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
  });
}

if (typeof requestAnimationFrame === "undefined") {
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
    setTimeout(() => callback(0), 0),
  );
  vi.stubGlobal(
    "cancelAnimationFrame",
    (timeoutId: ReturnType<typeof setTimeout>) => clearTimeout(timeoutId),
  );
}

// MSW's XMLHttpRequest interceptor references the bare `ProgressEvent`
// global from inside async `respondWith` callbacks (via `createEvent`).
// Vitest's jsdom environment installs `ProgressEvent` as an own accessor on
// `globalThis` while the environment is alive and *deletes* it during
// per-file teardown (it is part of vitest's `LIVING_KEYS`). If an in-flight
// intercepted XHR (e.g. PostHog analytics) resolves its mocked response
// after teardown, the late callback evaluates `ProgressEvent` against a
// torn-down global and throws `ReferenceError: ProgressEvent is not defined`,
// which Vitest reports as an unhandled rejection and fails the whole run.
//
// The robust fix is to drain those pending async response callbacks in
// `afterAll` (which runs *before* jsdom teardown) so they settle while
// `ProgressEvent` is still defined. See `afterAll` below. The getter below
// stashes the live class as a light defense-in-depth for any callback that
// fires before teardown completes; it cannot help after teardown (the
// accessor is deleted there), which is exactly why the `afterAll` drain is
// the real fix.
class MockProgressEvent extends Event {
  readonly lengthComputable: boolean;

  readonly loaded: number;

  readonly total: number;

  constructor(type: string, eventInitDict: ProgressEventInit = {}) {
    super(type, eventInitDict);
    this.lengthComputable = eventInitDict.lengthComputable ?? false;
    this.loaded = eventInitDict.loaded ?? 0;
    this.total = eventInitDict.total ?? 0;
  }
}

// `afterAll` runs while jsdom is still active, so `globalThis.ProgressEvent`
// is jsdom's constructor here. Stash it so the post-teardown getter can
// keep returning the real class even after the accessor is removed.
const _liveProgressEvent =
  typeof globalThis.ProgressEvent !== "undefined"
    ? globalThis.ProgressEvent
    : MockProgressEvent;

Object.defineProperty(globalThis, "ProgressEvent", {
  configurable: true,
  get() {
    return _liveProgressEvent;
  },
});

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

// Mock useRevalidator from react-router to allow direct store manipulation in tests
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

beforeEach(() => {
  vi.stubEnv("VITE_SESSION_API_KEY", "test-session-key");
});

afterEach(async () => {
  server.resetHandlers();
  window.sessionStorage?.removeItem("openhands-active-backend");
  // Cleanup the document body after each test
  cleanup();
  // Drain any queued microtasks before jsdom is torn down between test files.
  // Without this, async state updates queued during render (for example by
  // HeroUI v2 components wrapped in framer-motion's LazyMotion) can resolve
  // after `window` is gone and trigger spurious unhandled rejections in
  // react-dom's `resolveUpdatePriority`. We use `Promise.resolve()` (a
  // microtask) rather than `setTimeout(0)` so this stays compatible with
  // tests that install fake timers.
  await Promise.resolve();
  await Promise.resolve();
});
afterAll(async () => {
  // Drain pending MSW `respondWith` callbacks (and any other queued
  // macrotasks) before jsdom is torn down. MSW resolves intercepted XHR
  // responses asynchronously; if a late callback (e.g. PostHog analytics
  // flushed during the last test) settles after teardown, its `createEvent`
  // call evaluates the bare `ProgressEvent` global against a torn-down
  // jsdom and throws `ReferenceError: ProgressEvent is not defined`. Running
  // a few real-timer ticks here lets those callbacks complete while
  // `ProgressEvent` is still defined. We restore real timers first so a test
  // that left fake timers active can't stall the drain.
  vi.useRealTimers();
  // Reset handlers first so no new intercepted requests start processing
  // during the drain window.
  server.resetHandlers();
  for (let i = 0; i < 30; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  server.close();
  vi.unstubAllGlobals();
});
