import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildAgentCanvasPath,
  getAgentCanvasBasePath,
} from "#/utils/base-path";

afterEach(() => {
  vi.unstubAllEnvs();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_BASE_PATH__;
});

describe("Agent Canvas base path", () => {
  it("defaults to root paths", () => {
    expect(getAgentCanvasBasePath()).toBe("");
    expect(buildAgentCanvasPath("/settings")).toBe("/settings");
  });

  it("uses the build-time VITE_BASE_PATH when configured", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas/");

    expect(getAgentCanvasBasePath()).toBe("/canvas");
    expect(buildAgentCanvasPath("/settings")).toBe("/canvas/settings");
  });

  it("falls back to the runtime-injected base path", () => {
    vi.stubEnv("VITE_BASE_PATH", "");
    (window as unknown as Record<string, unknown>).__AGENT_CANVAS_BASE_PATH__ =
      "canvas";

    expect(getAgentCanvasBasePath()).toBe("/canvas");
    expect(buildAgentCanvasPath("settings")).toBe("/canvas/settings");
  });
});
