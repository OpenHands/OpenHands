import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { trackError } from "#/utils/error-handler";
import { trackException } from "#/services/telemetry";

vi.mock("#/services/telemetry", () => ({
  trackException: vi.fn(),
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

      expect(trackException).toHaveBeenCalledWith(
        new Error("Test error"),
        {
          error_source: "test",
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

      expect(trackException).toHaveBeenCalledWith(
        new Error("Test error"),
        {
          error_source: "test",
          extra: "info",
          details: { foo: "bar" },
        },
      );
    });
  });
});
