import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { __resetMcpHealthStoreForTests } from "#/api/mcp-health/mcp-health-store";
import McpService from "#/api/mcp-service/mcp-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { InstalledServerCard } from "#/components/features/mcp-page/installed-server-card";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import type { ExtendedMCPTestResponse, MCPServerConfig } from "#/types/mcp-server";

const CUSTOM_SERVER: MCPServerConfig = {
  id: "shttp-0",
  type: "shttp",
  name: "custom",
  url: "https://mcp.example.com/mcp",
};

/** Matches the catalog `github` entry (probe spec + docsUrl). */
const GITHUB_SERVER: MCPServerConfig = {
  id: "shttp-0",
  type: "shttp",
  name: "github",
  url: "https://api.githubcopilot.com/mcp/",
  auth: { strategy: "api_key", value: "github_pat_x" },
};

const OAUTH_SERVER: MCPServerConfig = {
  id: "shttp-0",
  type: "shttp",
  name: "my-oauth",
  url: "https://mcp.oauth.example/mcp",
  auth: { strategy: "oauth2" },
};

function renderCard(
  server: MCPServerConfig,
  { onEdit = vi.fn(), onDelete = vi.fn() } = {},
) {
  render(
    <InstalledServerCard server={server} onEdit={onEdit} onDelete={onDelete} />,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          <ActiveBackendProvider>{children}</ActiveBackendProvider>
        </QueryClientProvider>
      ),
    },
  );
  return { onEdit, onDelete };
}

const healthDot = () => screen.getByTestId("mcp-health-dot");
const probeButton = (id: string) =>
  screen.getByTestId(`mcp-health-probe-${id}`);

describe("InstalledServerCard connection health", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    __resetMcpHealthStoreForTests();
    // The backend registry persists to localStorage; wipe it before the
    // reset re-reads storage so each test starts on the default local backend.
    window.localStorage.clear();
    __resetActiveStoreForTests();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
  });

  it("probes on demand: unchecked → checking → healthy with connectivity-only labeling", async () => {
    let resolveProbe!: (value: ExtendedMCPTestResponse) => void;
    vi.spyOn(McpService, "testServer").mockReturnValue(
      new Promise((resolve) => {
        resolveProbe = resolve;
      }),
    );
    renderCard(CUSTOM_SERVER);
    expect(healthDot()).toHaveAttribute("data-status", "unchecked");

    fireEvent.click(probeButton(CUSTOM_SERVER.id));
    expect(healthDot()).toHaveAttribute("data-status", "checking");

    resolveProbe({ ok: true, tools: ["a", "b"] });
    await waitFor(() =>
      expect(healthDot()).toHaveAttribute("data-status", "healthy-connectivity"),
    );
    // The explicit "proves connectivity only" hint must accompany the result.
    expect(
      screen.getByText("MCP$HEALTH_CONNECTIVITY_ONLY_HINT"),
    ).toBeInTheDocument();
  });

  it("offers Retry, Update credentials, and docs on a credentials failure, routing the fix to the editor only", async () => {
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "invalid_auth",
      error_kind: "credentials",
    });
    const { onEdit } = renderCard(GITHUB_SERVER);

    fireEvent.click(probeButton(GITHUB_SERVER.id));
    await waitFor(() =>
      expect(healthDot()).toHaveAttribute("data-status", "failed"),
    );

    // Probing must not have bubbled into the card's edit action.
    expect(onEdit).not.toHaveBeenCalled();
    expect(probeButton(GITHUB_SERVER.id)).toHaveTextContent("MCP$HEALTH_RETRY");
    expect(
      screen.getByRole("link", { name: "MCP$VIEW_DOCS" }),
    ).toHaveAttribute("href", "https://github.com/github/github-mcp-server");

    fireEvent.click(
      screen.getByTestId(`mcp-health-update-credentials-${GITHUB_SERVER.id}`),
    );
    expect(onEdit).toHaveBeenCalledTimes(1);
  });

  it("Retry re-probes and moves the card to healthy in place", async () => {
    vi.spyOn(McpService, "testServer")
      .mockResolvedValueOnce({
        ok: false,
        error: "refused",
        error_kind: "connection",
      })
      .mockResolvedValueOnce({ ok: true, tools: ["a"] });
    renderCard(CUSTOM_SERVER);

    fireEvent.click(probeButton(CUSTOM_SERVER.id));
    await waitFor(() =>
      expect(healthDot()).toHaveAttribute("data-status", "failed"),
    );

    fireEvent.click(probeButton(CUSTOM_SERVER.id));
    await waitFor(() =>
      expect(healthDot()).toHaveAttribute("data-status", "healthy-connectivity"),
    );
  });

  it("renders no health section on cloud backends", () => {
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "k",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1" });

    renderCard(CUSTOM_SERVER);

    expect(
      screen.queryByTestId(`mcp-server-health-${CUSTOM_SERVER.id}`),
    ).not.toBeInTheDocument();
  });

  it("re-authorizes a failed OAuth server and persists the refreshed state", async () => {
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "token expired",
      error_kind: "unknown",
    });
    vi.spyOn(McpService, "authorizeOAuth").mockResolvedValue({
      ok: true,
      tools: [],
      oauth_state: { tokens: { access_token: "fresh-access-token" } },
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    renderCard(OAUTH_SERVER);

    fireEvent.click(probeButton(OAUTH_SERVER.id));
    await waitFor(() =>
      expect(healthDot()).toHaveAttribute("data-status", "failed"),
    );

    fireEvent.click(
      screen.getByTestId(`mcp-health-reauthorize-${OAUTH_SERVER.id}`),
    );

    await waitFor(() =>
      expect(healthDot()).toHaveAttribute("data-status", "healthy-connectivity"),
    );
    expect(McpService.authorizeOAuth).toHaveBeenCalledTimes(1);
    // The refreshed oauth_state is persisted, not dropped on the floor.
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(saveSpy.mock.calls[0][0])).toContain(
      "fresh-access-token",
    );
  });
});
