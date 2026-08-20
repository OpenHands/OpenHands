import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { trackError } from "#/utils/error-handler";
import { trackEvent } from "#/services/telemetry";

vi.mock("#/services/telemetry", () => ({
  trackEvent: vi.fn(),
}));

describe("Error Handler", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/test-error");
  });
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
        current_url: window.location.href,
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
        current_url: window.location.href,
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
        current_url: window.location.href,
        error_source: "agent",
        error_kind: "internal",
        error_id: "error-123",
        error_telemetry: "diagnostic",
      });
    });

    it("promotes a string metadata eventId as the error_id fallback", () => {
      trackError({
        source: "chat",
        metadata: { msgId: "m-1", eventId: "evt-42", extra: "info" },
      });

      expect(trackEvent).toHaveBeenCalledWith("error_outcome", {
        current_url: window.location.href,
        error_source: "chat",
        error_kind: "unknown",
        error_id: "evt-42",
        error_telemetry: "diagnostic",
        msgId: "m-1",
        extra: "info",
      });
    });

    it("lets a classification error_id win over a metadata eventId", () => {
      trackError({
        source: "agent",
        metadata: { msgId: "m-1", eventId: "evt-42" },
        classification: {
          kind: "auth",
          retryable: false,
          user_action: "settings",
          error_id: "cls-7",
        },
      });

      expect(trackEvent).toHaveBeenCalledWith(
        "error_outcome",
        expect.objectContaining({
          current_url: window.location.href,
          error_id: "cls-7",
          error_telemetry: "outcome",
        }),
      );
      expect(trackEvent).not.toHaveBeenCalledWith(
        "error_outcome",
        expect.objectContaining({ eventId: "evt-42" }),
      );
    });

    it("records non-internal classifications as outcomes", () => {
      trackError({
        source: "agent",
        classification: {
          kind: "auth",
          retryable: false,
          user_action: "settings",
        },
      });

      expect(trackEvent).toHaveBeenCalledWith("error_outcome", {
        current_url: window.location.href,
        error_source: "agent",
        error_kind: "auth",
        error_telemetry: "outcome",
      });
    });
  });
});
