import { describe, expect, it } from "vitest";

/**
 * Regression test for the `ReferenceError: ProgressEvent is not defined`
 * unhandled rejection that intermittently failed whole CI runs
 * (all tests green, exit code 1).
 *
 * MSW's XMLHttpRequest interceptor evaluates the bare `ProgressEvent`
 * identifier inside async response callbacks. Vitest's jsdom teardown does
 * `keys.forEach((key) => delete global[key])`, so any *own* property named
 * `ProgressEvent` — jsdom's, or a polyfill a setup file installed — is gone
 * once the environment for a test file is torn down. A callback that settles
 * after that point then throws.
 *
 * `vitest.setup.ts` therefore keeps a fallback on `globalThis`'s prototype
 * chain, which `delete` cannot reach. This test performs exactly the deletion
 * teardown performs and asserts the identifier still resolves.
 */
describe("ProgressEvent fallback in vitest.setup.ts", () => {
  it("resolves the bare identifier after teardown deletes the own property", () => {
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

  it("prefers jsdom's own ProgressEvent while the environment is alive", () => {
    // The own property shadows the prototype fallback, so nothing observes the
    // stand-in until teardown removes jsdom's class.
    expect(
      Object.getOwnPropertyDescriptor(globalThis, "ProgressEvent"),
    ).toBeDefined();
    expect(new ProgressEvent("progress").type).toBe("progress");
  });

  it("does not add ProgressEvent to plain objects", () => {
    // The fallback lives on globalThis's own prototype chain, which in Node is
    // not Object.prototype — so it must not leak onto ordinary objects.
    expect("ProgressEvent" in {}).toBe(false);
  });
});

describe("XMLHttpRequestUpload fallback in vitest.setup.ts", () => {
  it("resolves the bare identifier after teardown deletes the own property", () => {
    const live = Object.getOwnPropertyDescriptor(
      globalThis,
      "XMLHttpRequestUpload",
    );
    expect(live).toBeDefined();

    delete (globalThis as { XMLHttpRequestUpload?: unknown })
      .XMLHttpRequestUpload;

    try {
      expect(typeof XMLHttpRequestUpload).toBe("function");
    } finally {
      if (live) {
        Object.defineProperty(globalThis, "XMLHttpRequestUpload", live);
      }
    }
  });

  it("does not add XMLHttpRequestUpload to plain objects", () => {
    expect("XMLHttpRequestUpload" in {}).toBe(false);
  });
});
