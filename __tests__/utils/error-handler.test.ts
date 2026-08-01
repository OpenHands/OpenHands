import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { trackError } from "#/utils/error-handler";
import { trackEvent } from "#/services/telemetry";

vi.mock("#/services/telemetry", () => ({
  trackEvent: vi.fn(),
}));

describe("Error Handler", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("trackError", () => {
    it("should send error to PostHog with basic info", () => {
      const error = {
        source: "test",
      };

      trackError(error);

      expect(trackEvent).toHaveBeenCalledWith("error_outcome", {
        error_source: "test",
        error_kind: "unknown",
        error_telemetry: "diagnostic",
      });
    });

    it("merges metadata while reserving error outcome fields", () => {
      const error = {
        source: "test",
        metadata: {
          extra: "info",
          details: { foo: "bar" },
          error_kind: "spoofed",
          error_telemetry: "outcome",
          error_id: "spoofed",
        },
      };

      trackError(error);

      expect(trackEvent).toHaveBeenCalledWith("error_outcome", {
        error_source: "test",
        error_kind: "unknown",
        error_telemetry: "diagnostic",
        extra: "info",
        details: { foo: "bar" },
      });
    });

    it("keeps a classified error ID without recording the message", () => {
      trackError({
        source: "agent",
        classification: {
          kind: "internal",
          retryable: false,
          user_action: "none",
          error_id: "error-123",
        },
      });

      expect(trackEvent).toHaveBeenCalledWith("error_outcome", {
        error_source: "agent",
        error_kind: "internal",
        error_id: "error-123",
        error_telemetry: "diagnostic",
      });
    });
  });
});
