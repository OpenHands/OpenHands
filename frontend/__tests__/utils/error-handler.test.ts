import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import posthog from "posthog-js";
import {
  trackError,
  showErrorToast,
  showChatError,
} from "#/utils/error-handler";
import * as Actions from "#/services/actions";
import * as CustomToast from "#/utils/custom-toast-handlers";

vi.mock("posthog-js", () => ({
  default: {
    captureException: vi.fn(),
  },
}));

vi.mock("#/services/actions", () => ({
  handleStatusMessage: vi.fn(),
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
        posthog,
      };

      trackError(error);

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Test error"),
        {
          error_source: "test",
          user_email: null,
        },
      );
    });

    it("should include user_email in PostHog event when userEmail is provided", () => {
      const error = {
        message: "Test error",
        source: "test",
        userEmail: "user@example.com",
        posthog,
      };

      trackError(error);

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Test error"),
        {
          error_source: "test",
          user_email: "user@example.com",
        },
      );
    });

    it("should set user_email to null when userEmail is not provided", () => {
      const error = {
        message: "Test error",
        source: "test",
        posthog,
      };

      trackError(error);

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Test error"),
        {
          error_source: "test",
          user_email: null,
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
        posthog,
      };

      trackError(error);

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Test error"),
        {
          error_source: "test",
          user_email: null,
          extra: "info",
          details: { foo: "bar" },
        },
      );
    });
  });

  describe("showErrorToast", () => {
    const errorToastSpy = vi.spyOn(CustomToast, "displayErrorToast");
    it("should log error and show toast", () => {
      const error = {
        message: "Toast error",
        source: "toast-test",
        posthog,
      };

      showErrorToast(error);

      // Verify PostHog logging
      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Toast error"),
        {
          error_source: "toast-test",
          user_email: null,
        },
      );

      // Verify toast was shown
      expect(errorToastSpy).toHaveBeenCalled();
    });

    it("should include metadata in PostHog event when showing toast", () => {
      const error = {
        message: "Toast error",
        source: "toast-test",
        metadata: { context: "testing" },
        posthog,
      };

      showErrorToast(error);

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Toast error"),
        {
          error_source: "toast-test",
          user_email: null,
          context: "testing",
        },
      );
    });

    it("should log errors from different sources with appropriate metadata", () => {
      // Test agent status error
      showErrorToast({
        message: "Agent error",
        source: "agent-status",
        metadata: { id: "error.agent" },
        posthog,
      });

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Agent error"),
        {
          error_source: "agent-status",
          user_email: null,
          id: "error.agent",
        },
      );

      showErrorToast({
        message: "Server error",
        source: "server",
        metadata: { error_code: 500, details: "Internal error" },
        posthog,
      });

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Server error"),
        {
          error_source: "server",
          user_email: null,
          error_code: 500,
          details: "Internal error",
        },
      );
    });

    it("should forward userEmail to captureException", () => {
      showErrorToast({
        message: "Toast error",
        source: "toast-test",
        userEmail: "toast@example.com",
        posthog,
      });

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Toast error"),
        {
          error_source: "toast-test",
          user_email: "toast@example.com",
        },
      );
    });

    it("should log feedback submission errors with conversation context", () => {
      const error = new Error("Feedback submission failed");
      showErrorToast({
        message: error.message,
        source: "feedback",
        metadata: { conversationId: "123", error },
        posthog,
      });

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Feedback submission failed"),
        {
          error_source: "feedback",
          user_email: null,
          conversationId: "123",
          error,
        },
      );
    });
  });

  describe("showChatError", () => {
    it("should log error and show chat error message", () => {
      const error = {
        message: "Chat error",
        source: "chat-test",
        msgId: "123",
        posthog,
      };

      showChatError(error);

      // Verify PostHog logging
      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Chat error"),
        {
          error_source: "chat-test",
          user_email: null,
        },
      );

      // Verify error message was shown in chat
      expect(Actions.handleStatusMessage).toHaveBeenCalledWith({
        type: "error",
        message: "Chat error",
        id: "123",
        status_update: true,
      });
    });

    it("should forward userEmail to captureException", () => {
      showChatError({
        message: "Chat error",
        source: "chat-test",
        msgId: "456",
        userEmail: "chat@example.com",
        posthog,
      });

      expect(posthog.captureException).toHaveBeenCalledWith(
        new Error("Chat error"),
        {
          error_source: "chat-test",
          user_email: "chat@example.com",
        },
      );
    });
  });
});
