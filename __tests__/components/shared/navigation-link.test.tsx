import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { NavigationLink } from "#/components/shared/navigation-link";

function renderNavigationLink(
  currentPath = "/",
  overrides: Partial<NavigationContextValue> = {},
  to = "/settings/mcp",
) {
  const value: NavigationContextValue = {
    currentPath,
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
    ...overrides,
  };

  const result = render(
    <NavigationProvider value={value}>
      <NavigationLink to={to}>MCP</NavigationLink>
    </NavigationProvider>,
  );

  return {
    ...result,
    navigate: value.navigate,
  };
}

describe("NavigationLink", () => {
  it("renders the destination href and active state from navigation context", () => {
    renderNavigationLink("/settings/mcp");

    expect(screen.getByRole("link", { name: "MCP" })).toHaveAttribute(
      "href",
      "/settings/mcp",
    );
    expect(screen.getByRole("link", { name: "MCP" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("uses the injected navigate callback on click", () => {
    const { navigate } = renderNavigationLink();
    const link = screen.getByRole("link", { name: "MCP" });

    fireEvent.click(link);

    expect(navigate).toHaveBeenCalledWith("/settings/mcp", {
      replace: false,
    });
  });

  it("stays active when the destination carries a query string", () => {
    renderNavigationLink(
      "/conversations/abc",
      {},
      "/conversations/abc?backend=local-1",
    );

    const link = screen.getByRole("link", { name: "MCP" });
    expect(link).toHaveAttribute("href", "/conversations/abc?backend=local-1");
    expect(link).toHaveAttribute("aria-current", "page");
  });

  it("does not mark a sibling path active because of a query string", () => {
    renderNavigationLink(
      "/conversations/other",
      {},
      "/conversations/abc?backend=local-1",
    );

    expect(screen.getByRole("link", { name: "MCP" })).not.toHaveAttribute(
      "aria-current",
    );
  });
});
