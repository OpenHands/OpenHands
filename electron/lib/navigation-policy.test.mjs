import { describe, expect, it, vi } from "vitest";

import {
  attachNavigationGuard,
  attachPopupPolicy,
  mainWindowOpenHandler,
  openExternalUrl,
} from "./navigation-policy.mjs";

/**
 * Minimal stand-in for an Electron `webContents`. Records the listeners the
 * policy registers so a test can fire one, and records the window-open
 * handler so a test can call it.
 */
function fakeWebContents() {
  const listeners = new Map();
  return {
    listeners,
    windowOpenHandler: null,
    on(event, handler) {
      const existing = listeners.get(event) ?? [];
      listeners.set(event, [...existing, handler]);
    },
    setWindowOpenHandler(handler) {
      this.windowOpenHandler = handler;
    },
    listenerCount(event) {
      return (listeners.get(event) ?? []).length;
    },
    /** `did-create-window` hands over the window itself, not (event, url). */
    emitWindow(event, win) {
      (listeners.get(event) ?? []).forEach((handler) => handler(win));
    },
    /** Fire one registered listener and report whether it cancelled. */
    emit(event, url) {
      const handlers = listeners.get(event) ?? [];
      const result = { prevented: false };
      const fakeEvent = {
        preventDefault() {
          result.prevented = true;
        },
      };
      handlers.forEach((handler) => handler(fakeEvent, url));
      return result;
    },
  };
}

function fakePopupWindow() {
  return { webContents: fakeWebContents(), close: vi.fn() };
}

describe("openExternalUrl", () => {
  it.each([
    "https://example.com/",
    "http://example.com/",
    "mailto:a@b.c",
    "tel:+123",
  ])("hands an allowlisted scheme to the OS: %s", (url) => {
    const openExternal = vi.fn();
    openExternalUrl(url, openExternal);
    expect(openExternal).toHaveBeenCalledWith(url);
  });

  it.each([
    "file:///etc/passwd",
    "smb://fileserver/share",
    "custom-handler://do-something",
    "not a url",
  ])("never hands %s to the OS", (url) => {
    const openExternal = vi.fn();
    openExternalUrl(url, openExternal);
    expect(openExternal).not.toHaveBeenCalled();
  });
});

describe("attachNavigationGuard", () => {
  it.each(["will-navigate", "will-redirect"])(
    "cancels %s to a remote host and hands it to the system browser",
    (event) => {
      const openExternal = vi.fn();
      const webContents = fakeWebContents();
      attachNavigationGuard(webContents, openExternal);

      const { prevented } = webContents.emit(event, "https://evil.example/");

      expect(prevented).toBe(true);
      expect(openExternal).toHaveBeenCalledWith("https://evil.example/");
    },
  );

  it.each(["will-navigate", "will-redirect"])(
    "lets the app navigate within the loopback app on %s",
    (event) => {
      const openExternal = vi.fn();
      const webContents = fakeWebContents();
      attachNavigationGuard(webContents, openExternal);

      const { prevented } = webContents.emit(
        event,
        "http://localhost:8000/settings",
      );

      expect(prevented).toBe(false);
      expect(openExternal).not.toHaveBeenCalled();
    },
  );

  it("cancels a loopback look-alike host rather than loading it in the window", () => {
    const openExternal = vi.fn();
    const webContents = fakeWebContents();
    attachNavigationGuard(webContents, openExternal);

    const { prevented } = webContents.emit(
      "will-navigate",
      "http://localhost.evil.example/",
    );

    expect(prevented).toBe(true);
  });

  it("cancels a file: navigation without handing it to the OS", () => {
    const openExternal = vi.fn();
    const webContents = fakeWebContents();
    attachNavigationGuard(webContents, openExternal);

    const { prevented } = webContents.emit(
      "will-navigate",
      "file:///etc/passwd",
    );

    expect(prevented).toBe(true);
    expect(openExternal).not.toHaveBeenCalled();
  });
});

describe("mainWindowOpenHandler", () => {
  it("allows about:blank so the device-flow popup can be opened", () => {
    const openExternal = vi.fn();
    expect(mainWindowOpenHandler("about:blank", openExternal)).toEqual({
      action: "allow",
      overrideBrowserWindowOptions: { width: 800, height: 700 },
    });
    expect(openExternal).not.toHaveBeenCalled();
  });

  it("allows a loopback app URL in an Electron window", () => {
    const openExternal = vi.fn();
    expect(
      mainWindowOpenHandler("http://127.0.0.1:8000/x", openExternal),
    ).toEqual({
      action: "allow",
    });
    expect(openExternal).not.toHaveBeenCalled();
  });

  it("denies a remote URL and sends it to the system browser", () => {
    const openExternal = vi.fn();
    expect(mainWindowOpenHandler("https://example.com/", openExternal)).toEqual(
      {
        action: "deny",
      },
    );
    expect(openExternal).toHaveBeenCalledWith("https://example.com/");
  });

  it("denies a file: URL without handing it to the OS", () => {
    const openExternal = vi.fn();
    expect(mainWindowOpenHandler("file:///etc/passwd", openExternal)).toEqual({
      action: "deny",
    });
    expect(openExternal).not.toHaveBeenCalled();
  });
});

describe("attachPopupPolicy", () => {
  it("lets the popup stay on about:blank", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const { prevented } = popupWin.webContents.emit(
      "will-navigate",
      "about:blank",
    );

    expect(prevented).toBe(false);
    expect(popupWin.close).not.toHaveBeenCalled();
  });

  it("hands the OAuth verification URL to the system browser and closes the popup", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const { prevented } = popupWin.webContents.emit(
      "will-navigate",
      "https://app.all-hands.dev/oauth/verify",
    );

    expect(prevented).toBe(true);
    expect(openExternal).toHaveBeenCalledWith(
      "https://app.all-hands.dev/oauth/verify",
    );
    expect(popupWin.close).toHaveBeenCalled();
  });

  it("applies the same rule to a redirect the popup is sent through", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const { prevented } = popupWin.webContents.emit(
      "will-redirect",
      "https://evil.example/",
    );

    expect(prevented).toBe(true);
    expect(openExternal).toHaveBeenCalledWith("https://evil.example/");
    expect(popupWin.close).toHaveBeenCalled();
  });

  it("denies a window.open() from inside the popup to a remote host", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const decision = popupWin.webContents.windowOpenHandler({
      url: "https://evil.example/",
    });

    expect(decision).toEqual({ action: "deny" });
    expect(openExternal).toHaveBeenCalledWith("https://evil.example/");
  });

  it("allows a window.open() from inside the popup back to the loopback app", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const decision = popupWin.webContents.windowOpenHandler({
      url: "http://localhost:8000/callback",
    });

    expect(decision).toEqual({ action: "allow" });
    expect(openExternal).not.toHaveBeenCalled();
  });

  // The loopback window.open() above is allowed, so Electron creates another
  // window. A window inherits no policy from its opener, so the policy has to
  // be re-applied down the chain or that window is unguarded.
  it("applies the policy to a window the popup opens", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const childWin = fakePopupWindow();
    popupWin.webContents.emitWindow("did-create-window", childWin);

    expect(childWin.webContents.listenerCount("will-navigate")).toBe(1);
    expect(childWin.webContents.listenerCount("will-redirect")).toBe(1);
    expect(childWin.webContents.windowOpenHandler).not.toBeNull();
  });

  it("cancels a remote navigation in a window the popup opened", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const childWin = fakePopupWindow();
    popupWin.webContents.emitWindow("did-create-window", childWin);
    const { prevented } = childWin.webContents.emit(
      "will-navigate",
      "https://evil.example/",
    );

    expect(prevented).toBe(true);
    expect(openExternal).toHaveBeenCalledWith("https://evil.example/");
    expect(childWin.close).toHaveBeenCalled();
  });

  it("keeps following the chain past the first nested window", () => {
    const openExternal = vi.fn();
    const popupWin = fakePopupWindow();
    attachPopupPolicy(popupWin, openExternal);

    const childWin = fakePopupWindow();
    popupWin.webContents.emitWindow("did-create-window", childWin);
    const grandchildWin = fakePopupWindow();
    childWin.webContents.emitWindow("did-create-window", grandchildWin);

    const { prevented } = grandchildWin.webContents.emit(
      "will-redirect",
      "https://evil.example/",
    );

    expect(prevented).toBe(true);
    expect(openExternal).toHaveBeenCalledWith("https://evil.example/");
  });
});
