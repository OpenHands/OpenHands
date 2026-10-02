import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isMacDesktopShell,
  subscribeDesktopFullScreen,
} from "#/utils/desktop-shell";

type ShellWindow = Window & {
  desktopShell?: {
    platform?: string;
    getFullScreen?: () => Promise<boolean>;
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

  it("reads the state the window is already in, which no transition announces", async () => {
    (window as ShellWindow).desktopShell = {
      platform: "darwin",
      getFullScreen: () => Promise.resolve(true),
      onFullScreenChange: () => () => {},
    };
    const cb = vi.fn();

    subscribeDesktopFullScreen(cb);
    await vi.waitFor(() => expect(cb).toHaveBeenCalledWith(true));
  });

  it("lets a transition win over an initial read that resolves after it", async () => {
    let resolveRead: ((value: boolean) => void) | undefined;
    let emit: ((value: boolean) => void) | undefined;
    (window as ShellWindow).desktopShell = {
      platform: "darwin",
      getFullScreen: () =>
        new Promise((resolve) => {
          resolveRead = resolve;
        }),
      onFullScreenChange: (cb) => {
        emit = cb;
        return () => {};
      },
    };
    const cb = vi.fn();

    subscribeDesktopFullScreen(cb);
    emit?.(false);
    resolveRead?.(true);
    await Promise.resolve();

    expect(cb).toHaveBeenCalledExactlyOnceWith(false);
  });

  it("drops an initial read that resolves after unsubscribing", async () => {
    let resolveRead: ((value: boolean) => void) | undefined;
    (window as ShellWindow).desktopShell = {
      platform: "darwin",
      getFullScreen: () =>
        new Promise((resolve) => {
          resolveRead = resolve;
        }),
      onFullScreenChange: () => () => {},
    };
    const cb = vi.fn();

    subscribeDesktopFullScreen(cb)();
    resolveRead?.(true);
    await Promise.resolve();

    expect(cb).not.toHaveBeenCalled();
  });

  it("survives a failed initial read, keeping the transition subscription", async () => {
    const unsubscribe = vi.fn();
    let emit: ((value: boolean) => void) | undefined;
    (window as ShellWindow).desktopShell = {
      platform: "darwin",
      getFullScreen: () => Promise.reject(new Error("no window")),
      onFullScreenChange: (cb) => {
        emit = cb;
        return unsubscribe;
      },
    };
    const cb = vi.fn();

    const stop = subscribeDesktopFullScreen(cb);
    await Promise.resolve();
    emit?.(true);

    expect(cb).toHaveBeenCalledExactlyOnceWith(true);
    stop();
    expect(unsubscribe).toHaveBeenCalled();
  });
});
