import { describe, expect, it, vi } from "vitest";

import {
  tryIframeGoBack,
  tryIframeGoForward,
  tryIframeReload,
  tryReadIframeHref,
} from "#/utils/browser-iframe-navigation";

function mockIframe(overrides: {
  href?: string;
  throwOnHref?: boolean;
  throwOnHistory?: boolean;
  throwOnReload?: boolean;
}): HTMLIFrameElement {
  const back = vi.fn();
  const forward = vi.fn();
  const reload = vi.fn();

  const location = {
    get href() {
      if (overrides.throwOnHref) {
        throw new Error("cross-origin");
      }
      return overrides.href ?? "http://localhost:8089/";
    },
    reload: () => {
      if (overrides.throwOnReload) {
        throw new Error("cross-origin");
      }
      reload();
    },
  };

  const history = {
    back: () => {
      if (overrides.throwOnHistory) {
        throw new Error("cross-origin");
      }
      back();
    },
    forward: () => {
      if (overrides.throwOnHistory) {
        throw new Error("cross-origin");
      }
      forward();
    },
  };

  return {
    contentWindow: { history, location },
    // expose spies for assertions
    __spies: { back, forward, reload },
  } as unknown as HTMLIFrameElement & {
    __spies: { back: ReturnType<typeof vi.fn>; forward: ReturnType<typeof vi.fn>; reload: ReturnType<typeof vi.fn> };
  };
}

describe("browser-iframe-navigation", () => {
  it("drives iframe history back/forward/reload when available", () => {
    const iframe = mockIframe({}) as HTMLIFrameElement & {
      __spies: {
        back: ReturnType<typeof vi.fn>;
        forward: ReturnType<typeof vi.fn>;
        reload: ReturnType<typeof vi.fn>;
      };
    };

    expect(tryIframeGoBack(iframe)).toBe(true);
    expect(tryIframeGoForward(iframe)).toBe(true);
    expect(tryIframeReload(iframe)).toBe(true);
    expect(iframe.__spies.back).toHaveBeenCalledOnce();
    expect(iframe.__spies.forward).toHaveBeenCalledOnce();
    expect(iframe.__spies.reload).toHaveBeenCalledOnce();
  });

  it("reads href when same-origin and returns null when blocked", () => {
    expect(tryReadIframeHref(mockIframe({ href: "http://localhost/x" }))).toBe(
      "http://localhost/x",
    );
    expect(tryReadIframeHref(mockIframe({ throwOnHref: true }))).toBeNull();
    expect(tryReadIframeHref(null)).toBeNull();
  });

  it("returns false when history/reload calls throw", () => {
    const iframe = mockIframe({ throwOnHistory: true, throwOnReload: true });
    expect(tryIframeGoBack(iframe)).toBe(false);
    expect(tryIframeGoForward(iframe)).toBe(false);
    expect(tryIframeReload(iframe)).toBe(false);
  });
});
