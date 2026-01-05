import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WebhookStatusBadge } from "#/components/features/settings/git-settings/webhook-status-badge";

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("WebhookStatusBadge", () => {
  it("should display installed status when webhook is installed", () => {
    // Arrange
    const props = {
      webhookInstalled: true,
    };

    // Act
    render(<WebhookStatusBadge {...props} />);

    // Assert
    const badge = screen.getByText("GITLAB$WEBHOOK_STATUS_INSTALLED");
    expect(badge).toBeInTheDocument();
  });

  it("should display not installed status when webhook is not installed", () => {
    // Arrange
    const props = {
      webhookInstalled: false,
    };

    // Act
    render(<WebhookStatusBadge {...props} />);

    // Assert
    const badge = screen.getByText("GITLAB$WEBHOOK_STATUS_NOT_INSTALLED");
    expect(badge).toBeInTheDocument();
  });

  it("should display installed status when installation result is successful", () => {
    // Arrange
    const props = {
      webhookInstalled: false,
      installationResult: {
        success: true,
        error: null,
      },
    };

    // Act
    render(<WebhookStatusBadge {...props} />);

    // Assert
    const badge = screen.getByText("GITLAB$WEBHOOK_STATUS_INSTALLED");
    expect(badge).toBeInTheDocument();
  });

  it("should display failed status when installation result has error", () => {
    // Arrange
    const props = {
      webhookInstalled: false,
      installationResult: {
        success: false,
        error: "Installation failed",
      },
    };

    // Act
    render(<WebhookStatusBadge {...props} />);

    // Assert
    const badge = screen.getByText("GITLAB$WEBHOOK_STATUS_FAILED");
    expect(badge).toBeInTheDocument();
  });

  it("should show error message when installation fails", () => {
    // Arrange
    const errorMessage = "Permission denied";
    const props = {
      webhookInstalled: false,
      installationResult: {
        success: false,
        error: errorMessage,
      },
    };

    // Act
    render(<WebhookStatusBadge {...props} />);

    // Assert
    const badgeContainer = screen.getByText(
      "GITLAB$WEBHOOK_STATUS_FAILED",
    ).parentElement;
    expect(badgeContainer).toHaveAttribute("title", errorMessage);
  });
});
