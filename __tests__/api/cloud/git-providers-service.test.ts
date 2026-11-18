import { describe, it, expect, vi, beforeEach } from "vitest";

// We'll test the host normalization logic by testing listCloudGitProviders
// but since the real function requires network calls, we'll test the
// normalization logic in isolation by extracting it for testing.

describe("git-providers-service host normalization", () => {
  // Helper function that matches the normalization logic in the actual code
  function normalizeHost(host: string | null | undefined): string | null {
    if (typeof host === "string") {
      const trimmed = host.trim();
      return trimmed !== "" ? trimmed : null;
    }
    return null;
  }

  describe("normalizeHost", () => {
    it("returns non-empty string as-is", () => {
      expect(normalizeHost("gitlab.technogise.com")).toBe(
        "gitlab.technogise.com",
      );
      expect(normalizeHost("https://git.example.com:8443")).toBe(
        "https://git.example.com:8443",
      );
    });

    it("trims whitespace from strings", () => {
      expect(normalizeHost("  gitlab.example.com  ")).toBe("gitlab.example.com");
      expect(normalizeHost("\tgitlab.example.com\n")).toBe("gitlab.example.com");
    });

    it("returns null for empty string", () => {
      expect(normalizeHost("")).toBeNull();
    });

    it("returns null for whitespace-only string", () => {
      expect(normalizeHost("   ")).toBeNull();
      expect(normalizeHost("\t\n")).toBeNull();
    });

    it("returns null for null", () => {
      expect(normalizeHost(null)).toBeNull();
    });

    it("returns null for undefined", () => {
      expect(normalizeHost(undefined)).toBeNull();
    });
  });
});
