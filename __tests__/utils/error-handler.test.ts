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
        message: "Test error",
        source: "test",
      };

      trackError(error);

      expect(trackEvent).toHaveBeenCalledWith(
        "error_outcome",
        {
          error_source: "test",
          error_cause: "unknown",
          error_impact: "run_stopped",
          error_blame: "unknown",
          error_telemetry: "diagnostic",
        },
      );
    });

    it("should include additional metadata in PostHog event", () => {
      const error = {
        message: "Test error",
        source: "test",
        metadata: {
          extra: "info",
          details: { foo: "bar" },
        },
      };

      trackError(error);

      expect(trackEvent).toHaveBeenCalledWith(
        "error_outcome",
        {
          error_source: "test",
          error_cause: "unknown",
          error_impact: "run_stopped",
          error_blame: "unknown",
          error_telemetry: "diagnostic",
          extra: "info",
          details: { foo: "bar" },
        },
      );
    });
  });
});
