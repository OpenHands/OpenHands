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
      <NavigationLink to="/settings/mcp">MCP</NavigationLink>
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
});
