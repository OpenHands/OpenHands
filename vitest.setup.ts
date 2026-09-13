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
// global from inside async `respondWith` callbacks (via `createEvent`), and
// the bare `XMLHttpRequestUpload` global in `trigger` (`target instanceof
// XMLHttpRequestUpload`). Vitest's jsdom environment installs these as own
// properties on `globalThis` and removes them during per-file teardown with
// `keys.forEach((key) => delete global[key])`. If an in-flight intercepted
// XHR (e.g. PostHog analytics, or any request that escaped to the real
// network under `onUnhandledRequest: "bypass"` and is still waiting on a
// socket) settles after teardown, its callback evaluates `ProgressEvent`
// against a torn-down global and throws
// `ReferenceError: ProgressEvent is not defined` (or the
// `XMLHttpRequestUpload` equivalent). Vitest reports that as an
// unhandled rejection and fails the whole run even though every test passed.
//
// Two earlier attempts at this (an own-property getter, then the `afterAll`
// drain below) both put the fallback where teardown can reach it, or bounded
// how long a late callback may take. Neither holds: `delete` removes any own
// property regardless of who defined it, and a request stuck on a real socket
// can settle long after 30 macrotask ticks.
//
// `delete` only removes *own* properties, while identifier resolution walks
// the prototype chain. So the fallbacks go on an object inserted into
// `globalThis`'s prototype chain, where teardown cannot delete them: while
// the environment is alive jsdom's own properties shadow them, and once
// teardown removes those own properties, the bare identifiers resolve
// through the prototype to the classes below. Node's `globalThis` does not
// have `Object.prototype` as its direct prototype, so this adds nothing to
// plain objects.
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

// The interceptor only needs `XMLHttpRequestUpload` as an `instanceof`
// operand. While jsdom is alive its own property shadows this fallback, so
// behavior is unchanged; after teardown the check just takes the non-upload
// branch, quietly skipping listener dispatch for callbacks that have already
// outlived their environment instead of crashing the run.
class MockXMLHttpRequestUpload extends EventTarget {}

// Setup files run once per test file, and a worker process is reused across
// files. Without this marker each file would splice another holder into the
// prototype chain, so the chain would grow with every file in the run.
const XHR_GLOBALS_FALLBACK = Symbol.for("agent-canvas.xhr-globals-fallback");

function installXhrGlobalsFallback(fallbacks: Record<string, unknown>) {
  const currentProto = Object.getPrototypeOf(globalThis) as object | null;
  if (currentProto && XHR_GLOBALS_FALLBACK in currentProto) return;

  const holder = Object.create(currentProto) as Record<PropertyKey, unknown>;
  Object.defineProperty(holder, XHR_GLOBALS_FALLBACK, { value: true });
  for (const [name, fallback] of Object.entries(fallbacks)) {
    Object.defineProperty(holder, name, {
      value: fallback,
      configurable: true,
      writable: true,
    });
  }
  Object.setPrototypeOf(globalThis, holder);
}

installXhrGlobalsFallback({
  ProgressEvent: MockProgressEvent,
  XMLHttpRequestUpload: MockXMLHttpRequestUpload,
});

// MSW resolves an intercepted request asynchronously, and `resetHandlers()` does
// not cancel one that is already in flight. Track what the server still owes a
// response to, so the `afterAll` drain can wait for exactly that instead of
// counting a fixed number of event-loop turns. Keyed by request id rather than a
// counter so a duplicate listener registration cannot skew the total.
const inFlightRequestIds = new Set<string>();
server.events.on("request:start", ({ requestId }) => {
  inFlightRequestIds.add(requestId);
});
server.events.on("request:end", ({ requestId }) => {
  inFlightRequestIds.delete(requestId);
});

// Bounds only a request that never settles; the drain exits as soon as the set
// empties, which for most test files is immediately.
const DRAIN_TIMEOUT_MS = 2_000;
const DRAIN_POLL_MS = 5;

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
  // Drain pending MSW `respondWith` callbacks before jsdom is torn down, so a
  // late callback settles against a live jsdom rather than a torn-down one.
  // This is still a best-effort tidy-up, not the guarantee: a request can
  // outlast `DRAIN_TIMEOUT_MS` (one stuck on a real socket, for instance),
  // which is what the prototype-chain XHR-globals fallback above is for.
  // We restore real timers first so a test that left fake timers active can't
  // stall the drain.
  vi.useRealTimers();
  // Reset handlers first so no new intercepted requests start processing
  // during the drain window.
  server.resetHandlers();
  const drainDeadline = Date.now() + DRAIN_TIMEOUT_MS;
  while (inFlightRequestIds.size > 0 && Date.now() < drainDeadline) {
    await new Promise((resolve) => setTimeout(resolve, DRAIN_POLL_MS));
  }
  // `request:end` fires before the interceptor's own continuation runs, so give
  // that continuation one more turn while jsdom's XHR globals still exist.
  await new Promise((resolve) => setTimeout(resolve, 0));
  server.close();
  vi.unstubAllGlobals();
});
