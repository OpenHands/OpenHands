import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, afterEach } from "vitest";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { NavigationLink } from "#/components/shared/navigation-link";

describe("BrandButton", () => {
  const onClickMock = vi.fn();

  it.each([undefined, "text-xs"])(
    "preserves primary foreground and font size when merging %s",
    (className) => {
      render(
        <BrandButton type="button" variant="primary" className={className}>
          Save
        </BrandButton>,
      );
      // A color named text-base is misclassified as a font size by
      // tailwind-merge, dropping text-sm or being dropped by a size override.
      expect(screen.getByRole("button", { name: "Save" })).toHaveClass(
        className ?? "text-sm",
        "text-[var(--oh-accent-foreground)]",
      );
    },
  );

  it("should set a test id", () => {
    render(
      <BrandButton testId="brand-button" type="button" variant="primary">
        Test Button
      </BrandButton>,
    );

    expect(screen.getByTestId("brand-button")).toBeInTheDocument();
  });

  it("should call onClick when clicked", async () => {
    const user = userEvent.setup();
    render(
      <BrandButton type="button" variant="primary" onClick={onClickMock}>
        Test Button
      </BrandButton>,
    );

    await user.click(screen.getByText("Test Button"));
  });

  it("should be disabled if isDisabled is true", () => {
    render(
      <BrandButton type="button" variant="primary" isDisabled>
        Test Button
      </BrandButton>,
    );

    expect(screen.getByText("Test Button")).toBeDisabled();
  });

  it("should pass a start content", () => {
    render(
      <BrandButton
        type="button"
        variant="primary"
        startContent={
          <div data-testid="custom-start-content">Start Content</div>
        }
      >
        Test Button
      </BrandButton>,
    );

    screen.getByTestId("custom-start-content");
  });
});

describe("navigation-link", () => {
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
    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it("prefixes the href with the Canvas base path", () => {
      // A modified click and the browser's own context menu follow `href`
      // directly, so it must resolve inside Canvas rather than against the
      // origin root, which may serve a different app.
      vi.stubEnv("VITE_BASE_PATH", "/canvas");

      renderNavigationLink("/settings/mcp");

      expect(screen.getByRole("link", { name: "MCP" })).toHaveAttribute(
        "href",
        "/canvas/settings/mcp",
      );
    });

    it("keeps the base path and query string together on the href", () => {
      vi.stubEnv("VITE_BASE_PATH", "/canvas");

      renderNavigationLink(
        "/conversations/abc",
        {},
        "/conversations/abc?backend=local-1",
      );

      expect(screen.getByRole("link", { name: "MCP" })).toHaveAttribute(
        "href",
        "/canvas/conversations/abc?backend=local-1",
      );
    });

    it("keeps the base path in the href while still marking the link active", () => {
      vi.stubEnv("VITE_BASE_PATH", "/canvas");

      renderNavigationLink("/settings/mcp");

      // Active state is computed from the router's base-path-stripped pathname
      // (`/settings/mcp`), not the base-prefixed href.
      expect(screen.getByRole("link", { name: "MCP" })).toHaveAttribute(
        "aria-current",
        "page",
      );
    });

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
      expect(link).toHaveAttribute(
        "href",
        "/conversations/abc?backend=local-1",
      );
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

    it("keeps a caller-supplied aria-current when the path does not match", () => {
      // A section link (e.g. the sidebar Customize row) is current on routes
      // other than its own redirecting destination.
      render(
        <NavigationProvider
          value={{
            currentPath: "/mcp",
            conversationId: null,
            isNavigating: false,
            navigate: vi.fn(),
          }}
        >
          <NavigationLink to="/customize" aria-current="page">
            Customize
          </NavigationLink>
        </NavigationProvider>,
      );

      expect(screen.getByRole("link", { name: "Customize" })).toHaveAttribute(
        "aria-current",
        "page",
      );
    });
  });
});
