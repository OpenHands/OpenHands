import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";
import SettingsService from "#/api/settings-service/settings-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useRememberedRepositorySelection } from "#/hooks/use-remembered-repository-selection";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { HomeRepositorySelection, useHomeStore } from "#/stores/home-store";

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const cloudBackend: Backend = {
  id: "cloud-ohe",
  name: "OpenHands Enterprise",
  host: "https://ohe.example.com",
  apiKey: "cloud-key",
  kind: "cloud",
};

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );
};

const gitlabConnection = (id: number, overrides = {}) => ({
  id,
  title: `GitLab ${id}`,
  icon: null,
  color: "#fff",
  provider_type: "gitlab",
  connected: true,
  needs_reconnect: false,
  ...overrides,
});

const selectionVia = (gitProviderId?: number): HomeRepositorySelection => ({
  repository: {
    id: "42",
    full_name: "team/app",
    git_provider: "gitlab",
    is_public: false,
    ...(gitProviderId === undefined ? {} : { git_provider_id: gitProviderId }),
  },
  branch: { name: "main", commit_sha: "abc", protected: false },
  provider: "gitlab",
});

function serveCloud({
  oauthV2LoginEnabled,
  connections = [],
}: {
  oauthV2LoginEnabled: boolean;
  connections?: unknown[];
}) {
  vi.mocked(callCloudProxy).mockImplementation((async (req: {
    path: string;
  }) => {
    const url = new URL(`https://ohe.example.com${req.path}`);
    if (url.pathname === "/api/v1/web-client/config") {
      return { feature_flags: { enable_oauth_v2_login: oauthV2LoginEnabled } };
    }
    if (url.pathname === "/oauth/git-providers") return connections;
    if (url.pathname === "/api/v1/git/branches/search") {
      return {
        items: [{ name: "main", commit_sha: "abc", protected: false }],
        next_page_id: null,
      };
    }
    throw new Error(`Unexpected cloud request ${req.path}`);
  }) as never);
}

const branchRequests = () =>
  vi
    .mocked(callCloudProxy)
    .mock.calls.map((call) => (call[0] as { path: string }).path)
    .filter((path) => path.startsWith("/api/v1/git/branches/search"));

const remember = (selection: HomeRepositorySelection) =>
  useHomeStore.getState().setRepositorySelection(cloudBackend.id, selection);

const remembered = () =>
  useHomeStore.getState().repositorySelections[cloudBackend.id];

describe("useRememberedRepositorySelection with git provider connections", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    useHomeStore.setState({ repositorySelections: {} });
    vi.mocked(callCloudProxy).mockReset();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      provider_tokens_set: { gitlab: null },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
  });

  it("keeps a selection made through a linked connection and checks its branch through that connection", async () => {
    serveCloud({
      oauthV2LoginEnabled: true,
      connections: [gitlabConnection(4), gitlabConnection(9)],
    });
    remember(selectionVia(9));

    const { result } = renderHook(
      () => useRememberedRepositorySelection(cloudBackend.id, true),
      { wrapper },
    );

    await waitFor(() => expect(branchRequests()).toHaveLength(1));
    expect(
      new URL(`https://x${branchRequests()[0]}`).searchParams.get(
        "git_provider_id",
      ),
    ).toBe("9");
    expect(result.current.selection?.repository.git_provider_id).toBe(9);
    expect(remembered()).toBeDefined();
  });

  it("forgets a selection whose connection is no longer linked", async () => {
    serveCloud({
      oauthV2LoginEnabled: true,
      connections: [
        gitlabConnection(4),
        gitlabConnection(9, { connected: false }),
      ],
    });
    remember(selectionVia(9));

    renderHook(() => useRememberedRepositorySelection(cloudBackend.id, true), {
      wrapper,
    });

    await waitFor(() => expect(remembered()).toBeUndefined());
    expect(branchRequests()).toHaveLength(0);
  });

  it("forgets a selection that names no connection once connections are listed", async () => {
    serveCloud({
      oauthV2LoginEnabled: true,
      connections: [gitlabConnection(4)],
    });
    remember(selectionVia());

    renderHook(() => useRememberedRepositorySelection(cloudBackend.id, true), {
      wrapper,
    });

    await waitFor(() => expect(remembered()).toBeUndefined());
  });

  it("keeps a selection that names no connection when the instance has none", async () => {
    serveCloud({ oauthV2LoginEnabled: false });
    remember(selectionVia());

    const { result } = renderHook(
      () => useRememberedRepositorySelection(cloudBackend.id, true),
      { wrapper },
    );

    await waitFor(() => expect(branchRequests()).toHaveLength(1));
    expect(
      new URL(`https://x${branchRequests()[0]}`).searchParams.has(
        "git_provider_id",
      ),
    ).toBe(false);
    expect(result.current.selection).not.toBeNull();
  });

  it("forgets a selection tied to a connection when the instance has none", async () => {
    serveCloud({ oauthV2LoginEnabled: false });
    remember(selectionVia(9));

    renderHook(() => useRememberedRepositorySelection(cloudBackend.id, true), {
      wrapper,
    });

    await waitFor(() => expect(remembered()).toBeUndefined());
  });
});
