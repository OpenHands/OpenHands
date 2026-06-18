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

  it("lets modified clicks use the browser href", () => {
    const { navigate } = renderNavigationLink();
    const link = screen.getByRole("link", { name: "MCP" });

    fireEvent.click(link, { ctrlKey: true });

    expect(navigate).not.toHaveBeenCalled();
    expect(link).toHaveAttribute("href", "/settings/mcp");
  });

  it("matches active state by pathname when the href has query params", () => {
    renderNavigationLink(
      "/conversations/abc",
      {},
      "/conversations/abc?backendId=cloud-prod&orgId=org-2",
    );

    expect(screen.getByRole("link", { name: "MCP" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
