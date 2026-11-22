// @vitest-environment node
import { describe, expect, it } from "vitest";

import { isTrustedWindowSender } from "../../../electron/lib/window-sender-policy.mjs";

describe("isTrustedWindowSender", () => {
  const webContents = { id: "wc-1" };
  const window = {
    webContents,
    isDestroyed: () => false,
  };

  it("returns true when the sender matches the window's webContents", () => {
    const event = { sender: webContents };
    expect(isTrustedWindowSender(event, () => window)).toBe(true);
  });

  it("returns false for a different window's webContents", () => {
    const otherWindow = {
      webContents: { id: "wc-2" },
      isDestroyed: () => false,
    };
    const event = { sender: webContents };
    expect(isTrustedWindowSender(event, () => otherWindow)).toBe(false);
  });

  it("returns false when the window is destroyed", () => {
    const destroyedWindow = {
      webContents,
      isDestroyed: () => true,
    };
    const event = { sender: webContents };
    expect(isTrustedWindowSender(event, () => destroyedWindow)).toBe(false);
  });

  it("returns false when the event has no sender", () => {
    expect(isTrustedWindowSender({}, () => window)).toBe(false);
  });

  it("returns false when there is no event", () => {
    expect(isTrustedWindowSender(undefined, () => window)).toBe(false);
  });

  it("returns false when getWindow returns null", () => {
    const event = { sender: webContents };
    expect(isTrustedWindowSender(event, () => null)).toBe(false);
  });

  it("returns false when getWindow returns undefined", () => {
    const event = { sender: webContents };
    expect(isTrustedWindowSender(event, () => undefined)).toBe(false);
  });
});
