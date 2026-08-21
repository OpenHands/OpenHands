import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";

describe("transformVSCodeUrl", () => {
  const originalWindowLocation = window.location;

  beforeEach(() => {
    // Mock window.location
    Object.defineProperty(window, "location", {
      value: {
        hostname: "example.com",
        port: "8081",
        protocol: "https:",
      },
      writable: true,
    });
  });

  afterEach(() => {
    // Restore window.location
    Object.defineProperty(window, "location", {
      value: originalWindowLocation,
      writable: true,
    });
  });

  it("should return null if input is null", () => {
    expect(transformVSCodeUrl(null)).toBeNull();
  });

  it("should replace localhost with current hostname when they differ", () => {
    const input = "http://localhost:8080/?tkn=abc123&folder=/workspace";
    const expected =
      "https://example.com:8081/?tkn=abc123&folder=/workspace";

    expect(transformVSCodeUrl(input)).toBe(expected);
  });

  it("should route localhost VS Code URLs through the reverse proxy port", () => {
    const input =
      "http://localhost:8001/?tkn=abc123&folder=/workspace/project";
    const expected =
      "https://example.com:8081/?tkn=abc123&folder=/workspace/project";

    expect(transformVSCodeUrl(input)).toBe(expected);
  });

  it("should not modify URL if hostname is not localhost", () => {
    const input = "http://otherhost:8080/?tkn=abc123&folder=/workspace";

    expect(transformVSCodeUrl(input)).toBe(input);
  });

  it("should not modify URL if current hostname is also localhost", () => {
    // Change the mocked hostname to localhost
    Object.defineProperty(window, "location", {
      value: {
        hostname: "localhost",
        port: "8081",
        protocol: "http:",
      },
      writable: true,
    });

    const input = "http://localhost:8080/?tkn=abc123&folder=/workspace";

    expect(transformVSCodeUrl(input)).toBe(input);
  });

  it("should handle invalid URLs gracefully", () => {
    const input = "not-a-valid-url";

    expect(transformVSCodeUrl(input)).toBe(input);
  });
});
