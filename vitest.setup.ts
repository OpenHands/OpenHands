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

// MSW's XMLHttpRequest interceptor references bare identifiers from inside
// async `respondWith` callbacks: `ProgressEvent` (via `createEvent`) and
// `XMLHttpRequestUpload` (`target instanceof XMLHttpRequestUpload` in
// `trigger`). Vitest's jsdom environment installs both as own properties on
// `globalThis` and removes them during per-file teardown with
// `keys.forEach((key) => delete global[key])`. If an in-flight intercepted
// XHR (e.g. PostHog analytics, or any request that escaped to the real
// network under `onUnhandledRequest: "bypass"` and is still waiting on a
// socket) settles after teardown, those callbacks throw
// `ReferenceError: ProgressEvent is not defined` or
// `ReferenceError: XMLHttpRequestUpload is not defined`. Vitest reports
// that as an unhandled rejection and fails the whole run even though every
// test passed.
//
// Two earlier attempts at this (an own-property getter, then the `afterAll`
// drain below) both put the fallback where teardown can reach it, or bounded
// how long a late callback may take. Neither holds: `delete` removes any own
// property regardless of who defined it, and a request stuck on a real socket
// can settle long after 30 macrotask ticks. Keeping only `ProgressEvent` on
// the prototype chain unblocked `createEvent`, after which the same late
// callback died on `instanceof XMLHttpRequestUpload`.
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

// MSW only needs this identifier for `instanceof` — a distinct constructor
// is enough to stop the ReferenceError. Late callbacks then take the
// non-upload listener path, which is fine because the originating test has
// already finished.
class MockXMLHttpRequestUpload extends EventTarget {}

// Setup files run once per test file, and a worker process is reused across
// files. Without this marker each file would splice another holder into the
// prototype chain, so the chain would grow with every file in the run. The
// symbol name is historical (ProgressEvent was the first fallback); it now
// holds every MSW XHR global that teardown would otherwise delete.
const MSW_XHR_FALLBACK = Symbol.for("agent-canvas.progress-event-fallback");

function installMswXhrFallbacks(fallbacks: Record<string, unknown>) {
  const currentProto = Object.getPrototypeOf(globalThis) as object | null;
  let holder: Record<PropertyKey, unknown>;
  if (currentProto && MSW_XHR_FALLBACK in currentProto) {
    holder = currentProto as Record<PropertyKey, unknown>;
  } else {
    holder = Object.create(currentProto) as Record<PropertyKey, unknown>;
    Object.defineProperty(holder, MSW_XHR_FALLBACK, { value: true });
    Object.setPrototypeOf(globalThis, holder);
  }

  for (const [name, fallback] of Object.entries(fallbacks)) {
    if (!Object.prototype.hasOwnProperty.call(holder, name)) {
      Object.defineProperty(holder, name, {
        value: fallback,
        configurable: true,
        writable: true,
      });
    }
  }
}

installMswXhrFallbacks({
  ProgressEvent: MockProgressEvent,
  XMLHttpRequestUpload: MockXMLHttpRequestUpload,
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
  // macrotasks) before jsdom is torn down, so most late callbacks settle
  // against a live jsdom rather than a torn-down one. This is a best-effort
  // tidy-up, not the guarantee: a callback can always outlast the drain
  // window (a bypassed request stuck on a real socket, for instance), which
  // is what the prototype-chain `ProgressEvent` / `XMLHttpRequestUpload`
  // fallbacks above are for. We restore real timers first so a test that
  // left fake timers active can't stall the drain.
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
