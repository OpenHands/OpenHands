import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";

describe("transformVSCodeUrl", () => {
  const originalWindowLocation = window.location;

  beforeEach(() => {
    // Mock window.location
    Object.defineProperty(window, "location", {
      value: {
        hostname: "example.com",
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
    const expected = "http://example.com:8080/?tkn=abc123&folder=/workspace";

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
      },
      writable: true,
    });

    const input = "http://localhost:8080/?tkn=abc123&folder=/workspace";

    expect(transformVSCodeUrl(input)).toBe(input);
  });

  it("should return null for a malformed URL rather than passing it through", () => {
    expect(transformVSCodeUrl("not-a-valid-url")).toBeNull();
  });

  it.each([
    ["javascript:", "javascript:alert(document.domain)"],
    ["data:", "data:text/html,<script>alert(1)</script>"],
    ["vscode:", "vscode://file/etc/passwd"],
    ["file:", "file:///etc/passwd"],
  ])("should return null for a %s URL", (_scheme, input) => {
    expect(transformVSCodeUrl(input)).toBeNull();
  });

  it("should keep https URLs", () => {
    const input = "https://vscode.example.com/?tkn=abc123&folder=/workspace";

    expect(transformVSCodeUrl(input)).toBe(input);
  });
});
