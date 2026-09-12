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

  // Previously this returned the input unchanged. An unparseable string cannot
  // be shown to be a safe thing to hand to `window.open`, so it is rejected.
  it("should return null for unparseable URLs", () => {
    expect(transformVSCodeUrl("not-a-valid-url")).toBeNull();
  });

  it.each([
    ["javascript:", "javascript:alert(document.domain)"],
    ["data:", "data:text/html,<script>alert(1)</script>"],
    ["vbscript:", "vbscript:msgbox(1)"],
    ["file:", "file:///etc/passwd"],
  ])("should reject a %s URL", (_scheme, input) => {
    expect(transformVSCodeUrl(input)).toBeNull();
  });

  it("should allow https URLs", () => {
    const input = "https://vscode.example.com/?tkn=abc123";

    expect(transformVSCodeUrl(input)).toBe(input);
  });

  it("should rewrite a localhost https URL and keep the scheme", () => {
    const input = "https://localhost:8080/?tkn=abc123";

    expect(transformVSCodeUrl(input)).toBe("https://example.com:8080/?tkn=abc123");
  });
});
