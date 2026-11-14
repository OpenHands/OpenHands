import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Sidebar } from "#/components/features/sidebar/sidebar";
import { SidebarMobileNavProvider } from "#/components/features/sidebar/sidebar-mobile-nav-context";
import { useSidebarStore } from "#/stores/sidebar-store";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { getPinnedHomeRouteKey } from "#/hooks/use-pinned-home-route";

const EXTENSION_HREF = "/extensions/acme/dashboard";

const { useCanvasExtensionsRuntimeMock } = vi.hoisted(() => ({
  useCanvasExtensionsRuntimeMock: vi.fn(),
}));

vi.mock("react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router")>()),
  Link: ({ children }: React.PropsWithChildren) => children,
  useNavigate: vi.fn(() => vi.fn()),
  useLocation: vi.fn(() => ({ pathname: "/conversations" })),
  useParams: vi.fn(() => ({})),
  useMatch: vi.fn(() => null),
}));

// Fixed local backend so the pin key scopes to ("local", null), matching the
// harness used by the main Sidebar suite.
vi.mock("#/contexts/active-backend-context", () => {
  const active = {
    backend: { id: "local", name: "Local", kind: "local" },
    orgId: null,
  };
  return {
    useActiveBackendContext: () => ({
      backends: [active.backend],
      active,
      setActive: vi.fn(),
    }),
    useActiveBackend: () => active,
  };
});

vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({ children }: { children: unknown }) => children,
}));

vi.mock("#/components/features/conversation-panel/conversation-panel", () => ({
  ConversationPanel: () => null,
}));

// Keep the real provider (Sidebar mounts it) but drive the page list so an
// enabled Canvas Extension page renders in the rail.
vi.mock(
  "#/components/features/canvas-extensions/canvas-extensions-runtime",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("#/components/features/canvas-extensions/canvas-extensions-runtime")
      >();
    return {
      ...actual,
      useCanvasExtensionsRuntime: () => useCanvasExtensionsRuntimeMock(),
    };
  },
);

const renderSidebar = (currentPath: string) => {
  const value: NavigationContextValue = {
    currentPath,
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };

  return render(
    <QueryClientProvider client={new QueryClient()}>
      <NavigationProvider value={value}>
        <SidebarMobileNavProvider>
          <Sidebar />
        </SidebarMobileNavProvider>
      </NavigationProvider>
    </QueryClientProvider>,
  );
};

const ENABLED_PAGE = {
  href: EXTENSION_HREF,
  extension: { name: "acme" },
  contribution: { id: "dashboard", title: "Dashboard" },
};

describe("Pin as Home on Canvas Extension pages (#16995)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
    useCanvasExtensionsRuntimeMock.mockReturnValue({
      pages: [ENABLED_PAGE],
      activating: false,
      errors: new Map(),
    });
  });

  afterEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
    useCanvasExtensionsRuntimeMock.mockReset();
  });

  it("exposes Pin as Home on an enabled extension page and stores its href", () => {
    const pinKey = getPinnedHomeRouteKey("local", null);
    renderSidebar("/conversations");

    const pinToggle = screen.getByTestId(
      "sidebar-pin-home-toggle-extension-acme-dashboard",
    );
    expect(pinToggle).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(pinToggle);

    expect(pinToggle).toHaveAttribute("aria-pressed", "true");
    expect(window.localStorage.getItem(pinKey)).toBe(
      JSON.stringify(EXTENSION_HREF),
    );

    // Unpinning from the same control clears the pin.
    fireEvent.click(pinToggle);
    expect(pinToggle).toHaveAttribute("aria-pressed", "false");
    expect(window.localStorage.getItem(pinKey)).not.toBe(
      JSON.stringify(EXTENSION_HREF),
    );
  });

  it("marks the extension page's pin as active while it is pinned", () => {
    window.localStorage.setItem(
      getPinnedHomeRouteKey("local", null),
      JSON.stringify(EXTENSION_HREF),
    );

    renderSidebar("/conversations");

    expect(
      screen.getByTestId("sidebar-pin-home-toggle-extension-acme-dashboard"),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("does not expose the pin control when the extension page is gone", () => {
    useCanvasExtensionsRuntimeMock.mockReturnValue({
      pages: [],
      activating: false,
      errors: new Map(),
    });

    renderSidebar("/conversations");

    expect(
      screen.queryByTestId("sidebar-pin-home-toggle-extension-acme-dashboard"),
    ).not.toBeInTheDocument();
  });
});
