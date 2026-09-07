import { describe, expect, it } from "vitest";

/**
 * Regression test for the unhandled rejections that intermittently failed
 * whole CI runs (all tests green, exit code 1):
 * `ReferenceError: ProgressEvent is not defined` and
 * `ReferenceError: XMLHttpRequestUpload is not defined`.
 *
 * MSW's XMLHttpRequest interceptor evaluates those bare identifiers inside
 * async response callbacks (`createEvent` for ProgressEvent;
 * `target instanceof XMLHttpRequestUpload` in `trigger`). Vitest's jsdom
 * teardown does `keys.forEach((key) => delete global[key])`, so any *own*
 * property of those names — jsdom's, or a polyfill a setup file installed —
 * is gone once the environment for a test file is torn down. A callback that
 * settles after that point then throws.
 *
 * `vitest.setup.ts` therefore keeps fallbacks on `globalThis`'s prototype
 * chain, which `delete` cannot reach. This test performs exactly the deletion
 * teardown performs and asserts the identifiers still resolve.
 */
describe("MSW XHR fallbacks in vitest.setup.ts", () => {
  it("resolves ProgressEvent after teardown deletes the own property", () => {
    const live = Object.getOwnPropertyDescriptor(globalThis, "ProgressEvent");
    expect(live).toBeDefined();

    // What vitest's jsdom teardown does to every jsdom key.
    delete (globalThis as { ProgressEvent?: unknown }).ProgressEvent;

    try {
      // Before the fix this is "undefined", and the construction below throws
      // ReferenceError — the exact failure seen in CI.
      expect(typeof ProgressEvent).toBe("function");

      const event = new ProgressEvent("error", {
        lengthComputable: true,
        loaded: 3,
        total: 7,
      });

      expect(event).toBeInstanceOf(Event);
      expect(event.type).toBe("error");
      expect(event.lengthComputable).toBe(true);
      expect(event.loaded).toBe(3);
      expect(event.total).toBe(7);
    } finally {
      if (live) Object.defineProperty(globalThis, "ProgressEvent", live);
    }
  });

  it("resolves XMLHttpRequestUpload after teardown deletes the own property", () => {
    const live = Object.getOwnPropertyDescriptor(
      globalThis,
      "XMLHttpRequestUpload",
    );
    expect(live).toBeDefined();

    delete (globalThis as { XMLHttpRequestUpload?: unknown })
      .XMLHttpRequestUpload;

    try {
      // Before the fix this throws ReferenceError — the exact failure seen
      // in test-and-build (ubuntu) after the ProgressEvent prototype-chain
      // fallback landed: MSW's `trigger` does
      // `target instanceof XMLHttpRequestUpload`.
      expect(typeof XMLHttpRequestUpload).toBe("function");
      expect(() => ({}) instanceof XMLHttpRequestUpload).not.toThrow();
      expect({} instanceof XMLHttpRequestUpload).toBe(false);
    } finally {
      if (live) {
        Object.defineProperty(globalThis, "XMLHttpRequestUpload", live);
      }
    }
  });

  it("prefers jsdom's own constructors while the environment is alive", () => {
    // The own property shadows the prototype fallback, so nothing observes the
    // stand-in until teardown removes jsdom's class.
    expect(
      Object.getOwnPropertyDescriptor(globalThis, "ProgressEvent"),
    ).toBeDefined();
    expect(
      Object.getOwnPropertyDescriptor(globalThis, "XMLHttpRequestUpload"),
    ).toBeDefined();
    expect(new ProgressEvent("progress").type).toBe("progress");
    expect(new XMLHttpRequest().upload).toBeInstanceOf(XMLHttpRequestUpload);
  });

  it("does not add the fallbacks to plain objects", () => {
    // The fallback lives on globalThis's own prototype chain, which in Node is
    // not Object.prototype — so it must not leak onto ordinary objects.
    expect("ProgressEvent" in {}).toBe(false);
    expect("XMLHttpRequestUpload" in {}).toBe(false);
  });
});
