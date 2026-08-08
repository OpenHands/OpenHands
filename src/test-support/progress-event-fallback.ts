/**
 * Test-only `ProgressEvent` fallback.
 *
 * MSW's XMLHttpRequest interceptor decides ONCE, at module load, whether the
 * runtime has `ProgressEvent`:
 *
 * ```js
 * const SUPPORTS_PROGRESS_EVENT = typeof ProgressEvent !== "undefined";
 * // ...later, per event:
 * const ProgressEventClass = SUPPORTS_PROGRESS_EVENT ? ProgressEvent : ProgressEventPolyfill;
 * ```
 *
 * The capability check is cached; the class is resolved per call. Under
 * `environment: "jsdom"` the check runs while jsdom's `ProgressEvent` exists,
 * so the interceptor commits to the `SUPPORTS_PROGRESS_EVENT === true` branch
 * for the life of the worker. If a late interceptor callback then fires while
 * Vitest is tearing the environment down between files, evaluating the bare
 * `ProgressEvent` identifier throws `ReferenceError: ProgressEvent is not
 * defined` and Vitest fails the run with an unhandled rejection, even when
 * every test passed.
 *
 * Installing our own class unconditionally means the identifier resolves to
 * something this module owns rather than to a binding the environment
 * teardown removes. It is defined directly on `globalThis` rather than via
 * `vi.stubGlobal()` so `vi.unstubAllGlobals()` cannot take it away before
 * those late callbacks settle.
 */

/** Marker so tests can assert this fallback is the installed class. */
export const PROGRESS_EVENT_FALLBACK = Symbol.for(
  "agent-canvas.progress-event-fallback",
);

export class FallbackProgressEvent extends Event {
  static readonly [PROGRESS_EVENT_FALLBACK] = true;

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

/**
 * Install the fallback on `target`, replacing any existing `ProgressEvent`.
 *
 * Unconditional by design. Guarding on `typeof ProgressEvent === "undefined"`
 * (the previous behaviour) makes this a no-op under jsdom, which defines
 * `ProgressEvent` — so the fallback was only ever installed in the one
 * environment that did not need it, and never in the one that does.
 */
export function installProgressEventFallback(
  target: typeof globalThis = globalThis,
) {
  Object.defineProperty(target, "ProgressEvent", {
    configurable: true,
    writable: true,
    value: FallbackProgressEvent,
  });
}
