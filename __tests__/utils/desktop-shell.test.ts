import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isDesktopFullScreen,
  isMacDesktopShell,
  subscribeDesktopFullScreen,
} from "#/utils/desktop-shell";

function setShell(platform: string | undefined) {
  if (platform === undefined) {
    delete window.desktopShell;
    return;
  }
  window.desktopShell = { platform };
}

afterEach(() => {
  setShell(undefined);
});

describe("isMacDesktopShell", () => {
  it("is false in a plain browser tab (no preload bridge)", () => {
    expect(isMacDesktopShell()).toBe(false);
  });

  it("is true when the Electron preload reports darwin", () => {
    setShell("darwin");
    expect(isMacDesktopShell()).toBe(true);
  });

  it.each(["win32", "linux"])(
    "is false on the %s desktop build, which keeps the native title bar",
    (platform) => {
      setShell(platform);
      expect(isMacDesktopShell()).toBe(false);
    },
  );

  it("is false when the bridge exposes no platform", () => {
    window.desktopShell = {};
    expect(isMacDesktopShell()).toBe(false);
  });
});

describe("subscribeDesktopFullScreen", () => {
  it("is a no-op unsubscribe in a browser tab, where there is no window to watch", () => {
    const cb = vi.fn();

    expect(() => subscribeDesktopFullScreen(cb)()).not.toThrow();
    expect(cb).not.toHaveBeenCalled();
  });

  it("forwards fullscreen transitions from the bridge", () => {
    const unsubscribe = vi.fn();
    let emit: ((value: boolean) => void) | undefined;
    window.desktopShell = {
      platform: "darwin",
      onFullScreenChange: (cb) => {
        emit = cb;
        return unsubscribe;
      },
    };
    const cb = vi.fn();

    const stop = subscribeDesktopFullScreen(cb);
    emit?.(true);

    expect(cb).toHaveBeenCalledWith(true);
    stop();
    expect(unsubscribe).toHaveBeenCalled();
  });
});

describe("isDesktopFullScreen", () => {
  it("is false in a browser tab", () => {
    expect(isDesktopFullScreen()).toBe(false);
  });

  it("is false when the bridge exposes no fullscreen state", () => {
    setShell("darwin");
    expect(isDesktopFullScreen()).toBe(false);
  });

  it("reads the state the preload holds, which no transition announces after a reload", () => {
    window.desktopShell = {
      platform: "darwin",
      isFullScreen: () => true,
    };
    expect(isDesktopFullScreen()).toBe(true);
  });
});
