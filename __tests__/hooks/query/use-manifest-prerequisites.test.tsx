import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { useSetupPrerequisites } from "#/hooks/query/use-manifest-prerequisites";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { server } from "#/mocks/node";
import { createSetupEntry } from "../../manifests/manifest-test-data";
import type { SetupEntry } from "#/manifests/types";

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { getSettings: vi.fn() },
}));

const localBackend: Backend = {
  id: "test-backend",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-backend",
  name: "Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

/** The endpoint the packaged GitHub integration is installed against. */
const GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/";

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

function withIntegrations(
  integrations: SetupEntry["requires"]["integrations"],
): SetupEntry {
  return createSetupEntry({ requires: { integrations } });
}

function renderPrerequisites(entry: SetupEntry) {
  return renderHook(() => useSetupPrerequisites(entry), {
    wrapper: createWrapper(),
  });
}

describe("useSetupPrerequisites", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    // Nothing connected, unless a test says otherwise.
    vi.mocked(SettingsService.getSettings).mockResolvedValue({} as never);
    server.use(
      http.get("*/api/v1/secrets/search", () =>
        HttpResponse.json({ items: [], next_page_id: null }),
      ),
    );
  });

  it("blocks setup while a required integration is not connected", async () => {
    // Arrange
    const entry = withIntegrations({
      github: { message: "Reads pull requests." },
    });

    // Act
    const { result } = renderPrerequisites(entry);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isBlocked).toBe(true);
  });

  it("lets setup continue when an unconnected integration is only advisory", async () => {
    // Arrange
    const entry = withIntegrations({
      notion: { message: "Publishes the result.", required: false },
    });

    // Act
    const { result } = renderPrerequisites(entry);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect({
      isBlocked: result.current.isBlocked,
      warned: result.current.warningIntegrations.map(({ id }) => id),
    }).toEqual({ isBlocked: false, warned: ["notion"] });
  });

  it("has nothing to block on once the integration is connected", async () => {
    // Arrange
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      agent_settings: {
        mcp_config: { mcpServers: { github: { url: GITHUB_MCP_URL } } },
      },
    } as never);
    const entry = withIntegrations({
      github: { message: "Reads pull requests." },
    });

    // Act
    const { result } = renderPrerequisites(entry);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect({
      isBlocked: result.current.isBlocked,
      blocking: result.current.blockingIntegrations,
    }).toEqual({ isBlocked: false, blocking: [] });
  });

  it("recognizes GitLab MCP from flattened cloud settings", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      mcp_config: {
        gitlab: { transport: "http", url: "https://gitlab.com/api/v4/mcp" },
      },
    } as never);
    const entry = withIntegrations({ gitlab: { message: "Reads issues." } });

    const { result } = renderPrerequisites(entry);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.blockingIntegrations).toEqual([]);
  });

  it("recognizes a self-managed GitLab MCP server", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      agent_settings: {
        mcp_config: {
          mcpServers: {
            gitlab: { url: "https://gitlab.example.com/api/v4/mcp" },
          },
        },
      },
    } as never);
    const entry = withIntegrations({ gitlab: { message: "Reads issues." } });

    const { result } = renderPrerequisites(entry);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.blockingIntegrations).toEqual([]);
  });

  it("recognizes an existing GitLab token secret without MCP", async () => {
    server.use(
      http.get("*/api/settings/secrets", () =>
        HttpResponse.json({ secrets: [{ name: "GITLAB_TOKEN" }] }),
      ),
    );
    const entry = withIntegrations({ gitlab: { message: "Reads issues." } });

    const { result } = renderPrerequisites(entry);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.blockingIntegrations).toEqual([]);
  });

  it("recognizes the cloud GitLab sign-in token alias", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    server.use(
      http.get("*/api/v1/secrets/search", () =>
        HttpResponse.json({
          items: [{ name: "gitlab_token" }],
          next_page_id: null,
        }),
      ),
    );
    const entry = withIntegrations({ gitlab: { message: "Reads issues." } });

    const { result } = renderPrerequisites(entry);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.blockingIntegrations).toEqual([]);
  });

  it("recognizes an existing GitLab sign-in in cloud settings", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      provider_tokens_set: { gitlab: null },
    } as never);
    const entry = withIntegrations({ gitlab: { message: "Reads issues." } });

    const { result } = renderPrerequisites(entry);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.blockingIntegrations).toEqual([]);
  });

  it("does not count a disabled GitLab MCP server as connected", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      mcp_config: {
        gitlab: {
          transport: "http",
          url: "https://gitlab.com/api/v4/mcp",
          enabled: false,
        },
      },
    } as never);
    const entry = withIntegrations({ gitlab: { message: "Reads issues." } });

    const { result } = renderPrerequisites(entry);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.blockingIntegrations.map(({ id }) => id)).toEqual([
      "gitlab",
    ]);
  });

  it("has nothing to block on when the integration is connected natively on cloud", async () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    server.use(
      http.get("*/api/v1/web-client/config", () =>
        HttpResponse.json({ providers_configured: ["github"] }),
      ),
    );
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      provider_tokens_set: { github: null },
    } as never);
    const entry = withIntegrations({
      github: { message: "Reads pull requests." },
    });

    // Act
    const { result } = renderPrerequisites(entry);

    // Assert
    await waitFor(() => expect(result.current.isBlocked).toBe(false));
  });
});
