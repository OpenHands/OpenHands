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
import { useGitProviderOptions } from "#/hooks/use-git-provider-options";
import { DEFAULT_SETTINGS } from "#/services/settings";

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

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "key",
  kind: "local",
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

const connection = (
  id: number,
  title: string,
  overrides: Record<string, unknown> = {},
) => ({
  id,
  title,
  icon: null,
  color: "#fff",
  provider_type: "gitlab",
  connected: true,
  needs_reconnect: false,
  ...overrides,
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
    if (req.path === "/api/v1/web-client/config") {
      return { feature_flags: { enable_oauth_v2_login: oauthV2LoginEnabled } };
    }
    if (req.path === "/oauth/git-providers") return connections;
    throw new Error(`Unexpected cloud request ${req.path}`);
  }) as never);
}

const requestedPaths = () =>
  vi
    .mocked(callCloudProxy)
    .mock.calls.map((call) => (call[0] as { path: string }).path);

describe("useGitProviderOptions", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    vi.mocked(callCloudProxy).mockReset();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      provider_tokens_set: { github: null, gitlab: null },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
  });

  describe("on a cloud instance with enable_oauth_v2_login", () => {
    beforeEach(() => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    });

    it("offers each linked connection, so two of one type stay apart", async () => {
      serveCloud({
        oauthV2LoginEnabled: true,
        connections: [
          connection(4, "Work GitLab"),
          connection(9, "Self-hosted GitLab"),
        ],
      });

      const { result } = renderHook(() => useGitProviderOptions(), {
        wrapper,
      });

      await waitFor(() => expect(result.current.options).toHaveLength(2));
      expect(result.current.isScoped).toBe(true);
      expect(result.current.options).toEqual([
        { provider: "gitlab", gitProviderId: 4, label: "Work GitLab" },
        { provider: "gitlab", gitProviderId: 9, label: "Self-hosted GitLab" },
      ]);
    });

    it("leaves out connections that are not linked or need reconnecting", async () => {
      serveCloud({
        oauthV2LoginEnabled: true,
        connections: [
          connection(1, "Linked"),
          connection(2, "Not linked", { connected: false }),
          connection(3, "Expired", { needs_reconnect: true }),
        ],
      });

      const { result } = renderHook(() => useGitProviderOptions(), {
        wrapper,
      });

      await waitFor(() => expect(result.current.options).toHaveLength(1));
      expect(result.current.options[0].gitProviderId).toBe(1);
    });

    it("drops connections of a type Canvas does not know", async () => {
      serveCloud({
        oauthV2LoginEnabled: true,
        connections: [
          connection(1, "Known"),
          connection(2, "SSO", { provider_type: "enterprise_sso" }),
        ],
      });

      const { result } = renderHook(() => useGitProviderOptions(), {
        wrapper,
      });

      await waitFor(() => expect(result.current.options).toHaveLength(1));
      expect(result.current.options[0].label).toBe("Known");
    });

    it("stays loading until the connections arrive", async () => {
      serveCloud({
        oauthV2LoginEnabled: true,
        connections: [connection(1, "Work GitLab")],
      });

      const { result } = renderHook(() => useGitProviderOptions(), {
        wrapper,
      });

      expect(result.current.isLoading).toBe(true);
      await waitFor(() => expect(result.current.isLoading).toBe(false));
    });
  });

  describe("without enable_oauth_v2_login", () => {
    it("offers one entry per connected provider type, with no id", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id, orgId: null });
      serveCloud({
        oauthV2LoginEnabled: false,
        connections: [connection(4, "Work GitLab")],
      });

      const { result } = renderHook(() => useGitProviderOptions(), {
        wrapper,
      });

      await waitFor(() => expect(result.current.isLoading).toBe(false));
      expect(result.current.isScoped).toBe(false);
      expect(result.current.options).toEqual([
        { provider: "github", gitProviderId: null, label: "GitHub" },
        { provider: "gitlab", gitProviderId: null, label: "GitLab" },
      ]);
      expect(requestedPaths()).not.toContain("/oauth/git-providers");
    });
  });

  describe("on a local backend", () => {
    it("offers the connected provider types without asking a cloud instance", async () => {
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id, orgId: null });

      const { result } = renderHook(() => useGitProviderOptions(), {
        wrapper,
      });

      await waitFor(() => expect(result.current.options).toHaveLength(2));
      expect(result.current.isScoped).toBe(false);
      expect(
        result.current.options.every((option) => option.gitProviderId === null),
      ).toBe(true);
      expect(callCloudProxy).not.toHaveBeenCalled();
    });
  });
});
