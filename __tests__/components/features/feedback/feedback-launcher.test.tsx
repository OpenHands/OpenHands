import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FeedbackLauncher } from "#/components/features/feedback/feedback-launcher";

const mocks = vi.hoisted(() => ({
  trackFeedbackSubmitted: vi.fn(),
  attachFeedbackEmail: vi.fn(),
  getLockedCloudHost: vi.fn(),
  isTelemetryEnabled: vi.fn(),
  conversationId: null as string | null,
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackFeedbackSubmitted: mocks.trackFeedbackSubmitted,
    attachFeedbackEmail: mocks.attachFeedbackEmail,
  }),
}));

vi.mock("#/api/agent-server-config", () => ({
  getLockedCloudHost: mocks.getLockedCloudHost,
}));

vi.mock("#/services/telemetry", () => ({
  isTelemetryEnabled: mocks.isTelemetryEnabled,
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

beforeEach(() => {
  mocks.trackFeedbackSubmitted.mockResolvedValue(undefined);
  mocks.attachFeedbackEmail.mockResolvedValue(undefined);
  mocks.getLockedCloudHost.mockReturnValue(null);
  mocks.isTelemetryEnabled.mockReturnValue(true);
  mocks.conversationId = null;
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("FeedbackLauncher", () => {
  describe("install gating", () => {
    it("renders the control on a non-OHE install", () => {
      render(<FeedbackLauncher />);
      expect(screen.getByTestId("feedback-launcher")).toBeInTheDocument();
    });

    it("renders nothing when the app is locked to Cloud", () => {
      mocks.getLockedCloudHost.mockReturnValue("https://app.all-hands.dev");
      render(<FeedbackLauncher />);
      expect(screen.queryByTestId("feedback-launcher")).not.toBeInTheDocument();
    });
  });

  describe("opening the form", () => {
    it("does not show the form until the button is clicked", () => {
      render(<FeedbackLauncher />);
      expect(screen.queryByTestId("feedback-panel")).not.toBeInTheDocument();
    });

    it("opens the form from the button", async () => {
      await openPanel();
      expect(screen.getByTestId("feedback-panel")).toBeInTheDocument();
      expect(screen.getByTestId("feedback-message")).toBeInTheDocument();
    });
  });

  describe("the email is optional", () => {
    it("submits with no email and does not touch the person", async () => {
      const user = await openPanel();
      await user.type(
        screen.getByTestId("feedback-message"),
        "the sidebar lags",
      );
      await user.click(screen.getByTestId("feedback-submit"));

      await waitFor(() =>
        expect(mocks.trackFeedbackSubmitted).toHaveBeenCalledWith(
          expect.objectContaining({
            feedback: "the sidebar lags",
            hasEmail: false,
          }),
        ),
      );
      expect(mocks.attachFeedbackEmail).not.toHaveBeenCalled();
    });

    it("attaches the email to the person when one is given", async () => {
      const user = await openPanel();
      await user.type(screen.getByTestId("feedback-message"), "looks good");
      await user.type(screen.getByTestId("feedback-email"), "dev@example.com");
      await user.click(screen.getByTestId("feedback-submit"));

      await waitFor(() =>
        expect(mocks.attachFeedbackEmail).toHaveBeenCalledWith(
          "dev@example.com",
        ),
      );
      expect(mocks.trackFeedbackSubmitted).toHaveBeenCalledWith(
        expect.objectContaining({ hasEmail: true }),
      );
    });
  });

  describe("validation", () => {
    it("rejects a malformed email and says so instead of submitting", async () => {
      const user = await openPanel();
      await user.type(screen.getByTestId("feedback-message"), "hello");
      await user.type(screen.getByTestId("feedback-email"), "not-an-email");
      await user.click(screen.getByTestId("feedback-submit"));

      expect(screen.getByTestId("feedback-email-error")).toBeInTheDocument();
      expect(mocks.trackFeedbackSubmitted).not.toHaveBeenCalled();
      expect(mocks.attachFeedbackEmail).not.toHaveBeenCalled();
    });

    it("cannot be submitted with empty feedback", async () => {
      await openPanel();
      expect(screen.getByTestId("feedback-submit")).toBeDisabled();
    });
  });

  describe("outcome", () => {
    it("confirms a successful submission", async () => {
      const user = await openPanel();
      await user.type(screen.getByTestId("feedback-message"), "nice work");
      await user.click(screen.getByTestId("feedback-submit"));

      await waitFor(() =>
        expect(screen.getByTestId("feedback-success")).toBeInTheDocument(),
      );
    });

    it("keeps what was typed when submission fails", async () => {
      mocks.trackFeedbackSubmitted.mockRejectedValue(new Error("network"));
      const user = await openPanel();
      await user.type(screen.getByTestId("feedback-message"), "keep this");
      await user.click(screen.getByTestId("feedback-submit"));

      await waitFor(() =>
        expect(screen.getByTestId("feedback-error")).toBeInTheDocument(),
      );
      expect(screen.getByTestId("feedback-message")).toHaveValue("keep this");
    });

    it("reports an error rather than silently dropping feedback without consent", async () => {
      // `trackEvent` resolves without capturing when consent is absent, so a
      // resolved promise would otherwise read as success.
      mocks.isTelemetryEnabled.mockReturnValue(false);
      const user = await openPanel();
      await user.type(screen.getByTestId("feedback-message"), "unheard");
      await user.click(screen.getByTestId("feedback-submit"));

      expect(screen.getByTestId("feedback-error")).toBeInTheDocument();
      expect(mocks.trackFeedbackSubmitted).not.toHaveBeenCalled();
    });
  });

  describe("session association", () => {
    it("passes the conversation id when the user is in one", async () => {
      mocks.conversationId = "conv-42";
      const user = await openPanel();
      await user.type(
        screen.getByTestId("feedback-message"),
        "in-conversation",
      );
      await user.click(screen.getByTestId("feedback-submit"));

      await waitFor(() =>
        expect(mocks.trackFeedbackSubmitted).toHaveBeenCalledWith(
          expect.objectContaining({ conversationId: "conv-42" }),
        ),
      );
    });

    it("omits the conversation id outside a conversation", async () => {
      const user = await openPanel();
      await user.type(
        screen.getByTestId("feedback-message"),
        "on the home page",
      );
      await user.click(screen.getByTestId("feedback-submit"));

      await waitFor(() =>
        expect(mocks.trackFeedbackSubmitted).toHaveBeenCalledWith(
          expect.objectContaining({ conversationId: undefined }),
        ),
      );
    });
  });
});
