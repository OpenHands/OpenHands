import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isMacDesktopShell,
  subscribeDesktopFullScreen,
} from "#/utils/desktop-shell";

type ShellWindow = Window & {
  desktopShell?: {
    platform?: string;
    onFullScreenChange?: (cb: (v: boolean) => void) => () => void;
  };
};

function setShell(platform: string | undefined) {
  if (platform === undefined) {
    delete (window as ShellWindow).desktopShell;
    return;
  }
  (window as ShellWindow).desktopShell = { platform };
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
    (window as ShellWindow).desktopShell = {};
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
    (window as ShellWindow).desktopShell = {
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
