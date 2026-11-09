import type { ReactNode } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ExtensionsMobileHub } from "#/components/features/skills/extensions-mobile-hub";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "token",
  kind: "cloud",
};

function renderMobileHub(ui: ReactNode) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <MemoryRouter>{ui}</MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("ExtensionsMobileHub", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("renders the Plugins item as an enabled link", () => {
    renderMobileHub(<ExtensionsMobileHub />);

    const hub = screen.getByTestId("extensions-mobile-hub");
    const pluginsItem = within(hub).getByTestId("sidebar-extensions-/plugins");
    expect(pluginsItem).not.toHaveAttribute("aria-disabled");
  });

  it("links Apps to its management page", () => {
    renderMobileHub(<ExtensionsMobileHub />);

    const hub = screen.getByTestId("extensions-mobile-hub");
    expect(within(hub).getByTestId("sidebar-extensions-/apps")).toHaveAttribute(
      "href",
      "/apps",
    );
  });

  describe("cloud backend", () => {
    it("renders the Skills item as an external link to {cloudHost}/settings/skills with the renamed label", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderMobileHub(<ExtensionsMobileHub />);

      const hub = screen.getByTestId("extensions-mobile-hub");
      const skillsItem = within(hub).getByTestId("sidebar-extensions-/skills");
      expect(skillsItem.tagName).toBe("A");
      expect(skillsItem).toHaveAttribute(
        "href",
        "https://app.all-hands.dev/settings/skills",
      );
      expect(skillsItem).toHaveAttribute("target", "_blank");
      expect(skillsItem).toHaveAttribute("rel", "noopener noreferrer");
      expect(skillsItem).toHaveTextContent(
        "SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK",
      );
    });

    it("hides the Plugins and Apps items", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderMobileHub(<ExtensionsMobileHub />);

      const hub = screen.getByTestId("extensions-mobile-hub");
      expect(
        within(hub).queryByTestId("sidebar-extensions-/plugins"),
      ).not.toBeInTheDocument();
      expect(
        within(hub).queryByTestId("sidebar-extensions-/apps"),
      ).not.toBeInTheDocument();
    });

    it("links the MCP Servers item, under its own label, to {cloudHost}/settings/mcp on an Enterprise instance with organization MCP servers", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
      server.use(
        http.get("*/api/v1/web-client/config", () =>
          HttpResponse.json({
            providers_configured: [],
            feature_flags: {
              deployment_mode: "self_hosted",
              enable_org_mcp_servers: true,
            },
          }),
        ),
      );

      renderMobileHub(<ExtensionsMobileHub />);

      const hub = screen.getByTestId("extensions-mobile-hub");
      await waitFor(() =>
        expect(
          within(hub).getByTestId("sidebar-extensions-/mcp"),
        ).toHaveAttribute(
          "href",
          "https://app.all-hands.dev/settings/mcp?org=org-1",
        ),
      );
      const mcpItem = within(hub).getByTestId("sidebar-extensions-/mcp");
      expect(mcpItem).toHaveAttribute("target", "_blank");
      expect(mcpItem).toHaveTextContent("MCP Servers");
    });
  });
});
