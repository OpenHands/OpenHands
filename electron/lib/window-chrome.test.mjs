import { describe, expect, it } from "vitest";

import {
  MAC_TITLEBAR_HEIGHT_PX,
  MAC_TRAFFIC_LIGHT_POSITION,
  getMainWindowChrome,
} from "./window-chrome.mjs";

describe("getMainWindowChrome", () => {
  it("uses a hidden-inset title bar and inset traffic lights on macOS", () => {
    expect(getMainWindowChrome("darwin")).toEqual({
      titleBarStyle: "hiddenInset",
      trafficLightPosition: MAC_TRAFFIC_LIGHT_POSITION,
    });
    expect(MAC_TRAFFIC_LIGHT_POSITION.y).toBeLessThan(MAC_TITLEBAR_HEIGHT_PX);
  });

  it("keeps the native title bar on Windows and Linux", () => {
    expect(getMainWindowChrome("win32")).toEqual({
      titleBarStyle: "default",
    });
    expect(getMainWindowChrome("linux")).toEqual({
      titleBarStyle: "default",
    });
  });
});
