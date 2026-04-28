import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { SettingsNavLink } from "#/components/features/settings/settings-nav-link";
import { I18nKey } from "#/i18n/declaration";

const mockNavItem = {
  to: "/settings/test",
  icon: <span data-testid="test-icon">Icon</span>,
  text: I18nKey.SETTINGS$NAV_API_KEYS,
};

const renderSettingsNavLink = (
  item = mockNavItem,
  onClick = vi.fn(),
  initialPath = "/",
) =>
  render(
    <MemoryRouter initialEntries={[initialPath]}>
      <SettingsNavLink item={item} onClick={onClick} />
    </MemoryRouter>,
  );

describe("SettingsNavLink", () => {
  it("should render the link with icon and text", () => {
    // Arrange & Act
    renderSettingsNavLink();

    // Assert
    expect(screen.getByRole("link")).toBeInTheDocument();
    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$NAV_API_KEYS")).toBeInTheDocument();
  });

  it("should navigate to the correct route", () => {
    // Arrange & Act
    renderSettingsNavLink();

    // Assert
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/settings/test");
  });

  it("should call onClick when clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onClick = vi.fn();
    renderSettingsNavLink(mockNavItem, onClick);

    // Act
    await user.click(screen.getByRole("link"));

    // Assert
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("should render different text based on item prop", () => {
    // Arrange
    const customItem = {
      to: "/settings/secrets",
      icon: <span>Icon</span>,
      text: I18nKey.SETTINGS$NAV_SECRETS,
    };

    // Act
    renderSettingsNavLink(customItem);

    // Assert
    expect(screen.getByText("SETTINGS$NAV_SECRETS")).toBeInTheDocument();
    expect(screen.getByRole("link")).toHaveAttribute("href", "/settings/secrets");
  });
});
