import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { ContextMenuNavLink } from "#/components/features/context-menu/context-menu-nav-link";
import { I18nKey } from "#/i18n/declaration";

const mockNavItem = {
  to: "/settings/test",
  icon: <span data-testid="test-icon">Icon</span>,
  text: I18nKey.SETTINGS$NAV_API_KEYS,
};

const renderContextMenuNavLink = (item = mockNavItem, onClick = vi.fn()) =>
  render(
    <MemoryRouter>
      <ContextMenuNavLink item={item} onClick={onClick} />
    </MemoryRouter>,
  );

describe("ContextMenuNavLink", () => {
  it("should render the link with icon and text", () => {
    // Arrange & Act
    renderContextMenuNavLink();

    // Assert
    expect(screen.getByRole("link")).toBeInTheDocument();
    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$NAV_API_KEYS")).toBeInTheDocument();
  });

  it("should navigate to the correct route", () => {
    // Arrange & Act
    renderContextMenuNavLink();

    // Assert
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/settings/test");
  });

  it("should call onClick when clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onClick = vi.fn();
    renderContextMenuNavLink(mockNavItem, onClick);

    // Act
    await user.click(screen.getByRole("link"));

    // Assert
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
