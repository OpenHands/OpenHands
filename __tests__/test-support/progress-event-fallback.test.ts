import { describe, expect, it } from "vitest";
import {
  FallbackProgressEvent,
  PROGRESS_EVENT_FALLBACK,
  installProgressEventFallback,
} from "#/test-support/progress-event-fallback";

describe("ProgressEvent fallback", () => {
  it("is the class actually installed on the test global", () => {
    // The regression this guards: the fallback used to be wrapped in
    // `if (typeof ProgressEvent === "undefined")`, which is false under jsdom.
    // It therefore installed only where it was not needed, leaving MSW's
    // interceptor to resolve a binding that environment teardown removes.
    expect(
      (globalThis.ProgressEvent as unknown as Record<symbol, unknown>)[
        PROGRESS_EVENT_FALLBACK
      ],
    ).toBe(true);
  });

  it("installs over an existing ProgressEvent rather than deferring to it", () => {
    class SomeoneElsesProgressEvent extends Event {}
    const target = {
      ProgressEvent: SomeoneElsesProgressEvent,
    } as unknown as typeof globalThis;

    installProgressEventFallback(target);

    expect(target.ProgressEvent).toBe(FallbackProgressEvent);
  });

  it("carries the ProgressEvent fields MSW's interceptor constructs", () => {
    // The interceptor builds these with `{ lengthComputable, loaded, total }`.
    const event = new FallbackProgressEvent("progress", {
      lengthComputable: true,
      loaded: 17,
      total: 42,
    });

    expect(event).toBeInstanceOf(Event);
    expect(event.type).toBe("progress");
    expect(event.lengthComputable).toBe(true);
    expect(event.loaded).toBe(17);
    expect(event.total).toBe(42);
  });

  it("defaults the fields the way ProgressEventInit does", () => {
    const event = new FallbackProgressEvent("loadend");

    expect(event.lengthComputable).toBe(false);
    expect(event.loaded).toBe(0);
    expect(event.total).toBe(0);
  });
});
