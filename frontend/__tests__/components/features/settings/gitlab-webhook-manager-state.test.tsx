import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { GitLabWebhookManagerState } from "#/components/features/settings/git-settings/gitlab-webhook-manager-state";
import { I18nKey } from "#/i18n/declaration";

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("GitLabWebhookManagerState", () => {
  it("should render title and message with translated keys", () => {
    // Arrange
    const props = {
      titleKey: I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE,
      messageKey: I18nKey.GITLAB$WEBHOOK_MANAGER_LOADING,
    };

    // Act
    render(<GitLabWebhookManagerState {...props} />);

    // Assert
    expect(
      screen.getByText(I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.GITLAB$WEBHOOK_MANAGER_LOADING),
    ).toBeInTheDocument();
  });

  it("should apply custom className to container", () => {
    // Arrange
    const customClassName = "custom-container-class";
    const props = {
      titleKey: I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE,
      messageKey: I18nKey.GITLAB$WEBHOOK_MANAGER_LOADING,
      className: customClassName,
    };

    // Act
    const { container } = render(<GitLabWebhookManagerState {...props} />);

    // Assert
    const containerElement = container.firstChild as HTMLElement;
    expect(containerElement).toHaveClass(customClassName);
  });
});
