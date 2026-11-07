import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildAgentCanvasPath,
  buildAgentCanvasUrl,
  getAgentCanvasBasePath,
  getAgentCanvasBaseUrl,
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

  it("leaves paths that carry their own origin untouched", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    // An absolute or protocol-relative path addresses another document; only
    // app-relative paths belong under the Canvas mount point.
    expect(buildAgentCanvasPath("https://example.com/settings")).toBe(
      "https://example.com/settings",
    );
    expect(buildAgentCanvasPath("//example.com/settings")).toBe(
      "//example.com/settings",
    );
  });
});

describe("Agent Canvas absolute URLs", () => {
  it("joins an app-relative path onto the current origin", () => {
    expect(buildAgentCanvasUrl("/shared/conversations/abc")).toBe(
      `${window.location.origin}/shared/conversations/abc`,
    );
  });

  it("carries the base path so the URL stays inside Canvas", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    expect(buildAgentCanvasUrl("/shared/conversations/abc")).toBe(
      `${window.location.origin}/canvas/shared/conversations/abc`,
    );
  });

  it("uses an explicit origin, such as a cloud backend host", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    expect(
      buildAgentCanvasUrl(
        "/shared/conversations/abc",
        "https://app.example.com",
      ),
    ).toBe("https://app.example.com/canvas/shared/conversations/abc");
  });

  it("does not double-prefix an origin that already ends in the base path", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    expect(
      buildAgentCanvasUrl("/settings", "https://app.example.com/canvas"),
    ).toBe("https://app.example.com/canvas/settings");
  });

  it("returns a bare path when there is no origin available", () => {
    expect(buildAgentCanvasUrl("/settings", "")).toBe("/settings");
  });
});

describe("Agent Canvas base URL", () => {
  it("is the plain origin when Canvas is mounted at the root", () => {
    expect(getAgentCanvasBaseUrl()).toBe(window.location.origin);
  });

  it("includes the base path, without a trailing slash", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    expect(getAgentCanvasBaseUrl()).toBe(`${window.location.origin}/canvas`);
  });

  it("does not add the base path twice when the origin already has it", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    expect(getAgentCanvasBaseUrl("https://app.example.com/canvas")).toBe(
      "https://app.example.com/canvas",
    );
  });

  it("is the base an exported conversation path appends to", () => {
    vi.stubEnv("VITE_BASE_PATH", "/canvas");

    expect(`${getAgentCanvasBaseUrl()}/conversations/abc`).toBe(
      `${window.location.origin}/canvas/conversations/abc`,
    );
  });
});
