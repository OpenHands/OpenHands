import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FeedbackLauncher } from "#/components/features/feedback/feedback-launcher";

/**
 * `use-tracking` is deliberately NOT mocked. The acceptance criteria name the
 * PostHog submission and identify calls, so the real hook runs and the telemetry
 * service is the seam — otherwise these tests would only prove that one mock
 * called another.
 */
const mocks = vi.hoisted(() => ({
  trackEvent: vi.fn(),
  setTelemetryPersonProperties: vi.fn(),
  setTelemetryBackendContext: vi.fn(),
  isTelemetryEnabled: vi.fn(),
  getLockedCloudHost: vi.fn(),
  backendKind: "local" as "local" | "cloud",
  conversationId: null as string | null,
}));

vi.mock("#/services/telemetry", () => ({
  trackEvent: mocks.trackEvent,
  setTelemetryPersonProperties: mocks.setTelemetryPersonProperties,
  setTelemetryBackendContext: mocks.setTelemetryBackendContext,
  isTelemetryEnabled: mocks.isTelemetryEnabled,
}));

vi.mock("#/api/agent-server-config", () => ({
  getLockedCloudHost: mocks.getLockedCloudHost,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { kind: mocks.backendKind, host: "http://127.0.0.1:8000" },
  }),
}));

vi.mock("#/hooks/query/use-automation-sdk-version", () => ({
  useAutomationSdkVersion: () => null,
}));

vi.mock("#/api/agent-server-compatibility", () => ({
  getCachedAgentServerVersion: () => null,
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: mocks.conversationId }),
}));

const openPanel = async () => {
  const user = userEvent.setup();
  render(<FeedbackLauncher />);
  await user.click(screen.getByTestId("feedback-launcher"));
  return user;
};

const submitWith = async (message: string, email?: string) => {
  const user = await openPanel();
  await user.type(screen.getByTestId("feedback-message"), message);
  if (email) await user.type(screen.getByTestId("feedback-email"), email);
  await user.click(screen.getByTestId("feedback-submit"));
  return user;
};

/** The event properties passed to the real `trackEvent`. */
const capturedProperties = () => mocks.trackEvent.mock.calls[0][1];

beforeEach(() => {
  mocks.trackEvent.mockResolvedValue(undefined);
  mocks.setTelemetryPersonProperties.mockResolvedValue(undefined);
  mocks.isTelemetryEnabled.mockReturnValue(true);
  mocks.getLockedCloudHost.mockReturnValue(null);
  mocks.backendKind = "local";
  mocks.conversationId = null;
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("FeedbackLauncher", () => {
  describe("install gating", () => {
    it("renders on a local install that is not locked to Cloud", () => {
      render(<FeedbackLauncher />);
      expect(screen.getByTestId("feedback-launcher")).toBeInTheDocument();
    });

    it("renders nothing when the app is locked to Cloud", () => {
      mocks.getLockedCloudHost.mockReturnValue("https://app.all-hands.dev");
      render(<FeedbackLauncher />);
      expect(screen.queryByTestId("feedback-launcher")).not.toBeInTheDocument();
    });

    it("renders nothing on a non-local backend", () => {
      // A self-hosted OHE reached on its own domain is not locked to Cloud, so
      // the backend kind is the signal that keeps this off hosted installs.
      mocks.backendKind = "cloud";
      render(<FeedbackLauncher />);
      expect(screen.queryByTestId("feedback-launcher")).not.toBeInTheDocument();
    });
  });

  describe("opening and closing", () => {
    it("does not show the form until the button is clicked", () => {
      render(<FeedbackLauncher />);
      expect(screen.queryByTestId("feedback-panel")).not.toBeInTheDocument();
    });

    it("opens the form from the button and marks the trigger expanded", async () => {
      await openPanel();
      expect(screen.getByTestId("feedback-panel")).toBeInTheDocument();
      expect(screen.getByTestId("feedback-launcher")).toHaveAttribute(
        "aria-expanded",
        "true",
      );
    });

    it("closes on Escape", async () => {
      const user = await openPanel();
      await user.keyboard("{Escape}");
      expect(screen.queryByTestId("feedback-panel")).not.toBeInTheDocument();
    });

    it("does not reopen showing a stale confirmation", async () => {
      const user = await submitWith("done");
      await waitFor(() =>
        expect(screen.getByTestId("feedback-success")).toBeInTheDocument(),
      );

      await user.click(screen.getByTestId("feedback-launcher")); // close
      await user.click(screen.getByTestId("feedback-launcher")); // reopen

      expect(screen.queryByTestId("feedback-success")).not.toBeInTheDocument();
      expect(screen.getByTestId("feedback-message")).toBeInTheDocument();
    });
  });

  describe("what reaches PostHog", () => {
    it("captures the feedback with no email and touches no person property", async () => {
      await submitWith("the sidebar lags");

      await waitFor(() => expect(mocks.trackEvent).toHaveBeenCalledTimes(1));
      expect(mocks.trackEvent.mock.calls[0][0]).toBe(
        "canvas_feedback_submitted",
      );
      expect(capturedProperties()).toMatchObject({
        feedback: "the sidebar lags",
        feedback_length: 16,
        has_email: false,
      });
      expect(mocks.setTelemetryPersonProperties).not.toHaveBeenCalled();
    });

    it("sets the email as a person property rather than an event property", async () => {
      await submitWith("looks good", "dev@example.com");

      await waitFor(() =>
        expect(mocks.setTelemetryPersonProperties).toHaveBeenCalledWith({
          email: "dev@example.com",
        }),
      );
      expect(capturedProperties()).toMatchObject({ has_email: true });
      expect(capturedProperties()).not.toHaveProperty("email");
    });

    it("carries the conversation the user is in", async () => {
      mocks.conversationId = "conv-42";
      await submitWith("in-conversation");

      await waitFor(() => expect(mocks.trackEvent).toHaveBeenCalled());
      expect(capturedProperties()).toMatchObject({
        conversation_id: "conv-42",
      });
    });

    it("omits the conversation id outside a conversation", async () => {
      await submitWith("on the home page");

      await waitFor(() => expect(mocks.trackEvent).toHaveBeenCalled());
      expect(capturedProperties().conversation_id).toBeUndefined();
    });
  });

  describe("validation", () => {
    it("rejects a malformed email and captures nothing", async () => {
      await submitWith("hello", "not-an-email");

      expect(screen.getByTestId("feedback-email-error")).toBeInTheDocument();
      expect(mocks.trackEvent).not.toHaveBeenCalled();
      expect(mocks.setTelemetryPersonProperties).not.toHaveBeenCalled();
    });

    it("cannot be submitted with empty feedback", async () => {
      await openPanel();
      expect(screen.getByTestId("feedback-submit")).toBeDisabled();
    });
  });

  describe("outcome", () => {
    it("confirms a successful submission", async () => {
      await submitWith("nice work");
      await waitFor(() =>
        expect(screen.getByTestId("feedback-success")).toBeInTheDocument(),
      );
    });

    it("keeps what was typed when the capture throws", async () => {
      mocks.trackEvent.mockRejectedValue(new Error("boom"));
      await submitWith("keep this");

      await waitFor(() =>
        expect(screen.getByTestId("feedback-error")).toBeInTheDocument(),
      );
      expect(screen.getByTestId("feedback-message")).toHaveValue("keep this");
    });

    it("reports an error rather than silently dropping feedback without consent", async () => {
      // `trackEvent` resolves without capturing when consent is absent, so a
      // resolved promise would otherwise read as success.
      mocks.isTelemetryEnabled.mockReturnValue(false);
      await submitWith("unheard");

      expect(screen.getByTestId("feedback-error")).toBeInTheDocument();
      expect(mocks.trackEvent).not.toHaveBeenCalled();
    });
  });
});
