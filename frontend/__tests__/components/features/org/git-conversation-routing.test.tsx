import { screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { GitConversationRouting } from "#/components/features/org/git-conversation-routing";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

describe("GitConversationRouting", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("should render all mock organizations", () => {
    // Arrange & Act
    renderWithProviders(<GitConversationRouting />);

    // Assert
    expect(screen.getByTestId("org-row-1")).toHaveTextContent(
      "GitHub/OpenHands",
    );
    expect(screen.getByTestId("org-row-2")).toHaveTextContent("GitHub/AcmeCo");
    expect(screen.getByTestId("org-row-3")).toHaveTextContent(
      "GitHub/already-claimed",
    );
    expect(screen.getByTestId("org-row-4")).toHaveTextContent(
      "GitLab/OpenHands",
    );
  });

  it("should show pre-claimed org with 'Claimed' label", () => {
    // Arrange & Act
    renderWithProviders(<GitConversationRouting />);

    // Assert
    const claimedButton = screen.getByTestId("claim-button-1");
    expect(claimedButton).toHaveTextContent("ORG$CLAIMED");
  });

  it("should show unclaimed orgs with 'Claim' label", () => {
    // Arrange & Act
    renderWithProviders(<GitConversationRouting />);

    // Assert
    expect(screen.getByTestId("claim-button-2")).toHaveTextContent("ORG$CLAIM");
  });

  it("should claim an organization and show success toast", async () => {
    // Arrange
    const successToastSpy = vi.spyOn(ToastHandlers, "displaySuccessToast");
    renderWithProviders(<GitConversationRouting />);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

    // Act
    await user.click(screen.getByTestId("claim-button-2"));
    // Move pointer away so hover state resets after transition
    await user.unhover(screen.getByTestId("claim-button-2"));
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // Assert
    await waitFor(() => {
      expect(screen.getByTestId("claim-button-2")).toHaveTextContent(
        "ORG$CLAIMED",
      );
    });
    expect(successToastSpy).toHaveBeenCalledWith("ORG$CLAIM_SUCCESS");
  });

  it("should show error toast when claiming an already-claimed org", async () => {
    // Arrange
    const errorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
    renderWithProviders(<GitConversationRouting />);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

    // Act
    await user.click(screen.getByTestId("claim-button-3"));
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // Assert
    await waitFor(() => {
      expect(screen.getByTestId("claim-button-3")).toHaveTextContent(
        "ORG$CLAIM",
      );
    });
    expect(errorToastSpy).toHaveBeenCalledWith("ORG$CLAIM_ERROR");
  });

  it("should disconnect a claimed org and show success toast", async () => {
    // Arrange
    const successToastSpy = vi.spyOn(ToastHandlers, "displaySuccessToast");
    renderWithProviders(<GitConversationRouting />);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

    // Act — disconnect the pre-claimed org (id: 1)
    await user.click(screen.getByTestId("claim-button-1"));
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // Assert
    await waitFor(() => {
      expect(screen.getByTestId("claim-button-1")).toHaveTextContent(
        "ORG$CLAIM",
      );
    });
    expect(successToastSpy).toHaveBeenCalledWith("ORG$DISCONNECT_SUCCESS");
  });

  it("should disable the button during claiming transition", async () => {
    // Arrange
    renderWithProviders(<GitConversationRouting />);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

    // Act
    await user.click(screen.getByTestId("claim-button-2"));

    // Assert — button is disabled while claiming
    expect(screen.getByTestId("claim-button-2")).toBeDisabled();

    // Cleanup — advance timer to complete transition
    act(() => {
      vi.advanceTimersByTime(1000);
    });
  });
});
