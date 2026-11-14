import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsClient } from "@openhands/typescript-client/clients";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { SecretsService } from "#/api/secrets-service";
import {
  useGitProviderHost,
  useRepositoryPullRequests,
} from "#/hooks/query/use-repository-git-items";

const settingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: settingsMock() }),
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "session-key",
  kind: "local",
};

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

function configureLocalSecrets(secrets: Record<string, string>) {
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue(
    Object.keys(secrets).map((name) => ({ name })),
  );
  vi.spyOn(SettingsClient.prototype, "getSecret").mockImplementation(
    async (name: string) => secrets[name],
  );
}

const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  settingsMock.mockReturnValue({ provider_tokens_set: {} });
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(
    new Response("[]", {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  global.fetch = originalFetch;
});

describe("useGitProviderHost", () => {
  it("uses the cloud connection host before anything else", () => {
    settingsMock.mockReturnValue({
      provider_tokens_set: { gitlab: "gitlab.corp.example" },
    });
    configureLocalSecrets({ GITLAB_HOST: "gitlab.example.com" });

    const { result } = renderHook(
      () => useGitProviderHost("gitlab", "gitlab.remote.example"),
      { wrapper: makeWrapper() },
    );

    expect(result.current).toEqual({
      host: "gitlab.corp.example",
      isResolved: true,
    });
    expect(SecretsService.getSecrets).not.toHaveBeenCalled();
  });

  it("uses the workspace remote host when there is no connection host", () => {
    configureLocalSecrets({ GITLAB_HOST: "gitlab.example.com" });

    const { result } = renderHook(
      () => useGitProviderHost("gitlab", "gitlab.remote.example"),
      { wrapper: makeWrapper() },
    );

    expect(result.current).toEqual({
      host: "gitlab.remote.example",
      isResolved: true,
    });
  });

  it("falls back to GITLAB_HOST and reports when the lookup settled", async () => {
    configureLocalSecrets({ GITLAB_HOST: "gitlab.example.com" });

    const { result } = renderHook(() => useGitProviderHost("gitlab", null), {
      wrapper: makeWrapper(),
    });

    expect(result.current.isResolved).toBe(false);
    await waitFor(() =>
      expect(result.current).toEqual({
        host: "https://gitlab.example.com",
        isResolved: true,
      }),
    );
  });

  it("resolves to the public host when nothing is configured", async () => {
    configureLocalSecrets({});

    const { result } = renderHook(() => useGitProviderHost("gitlab", null), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current).toEqual({ host: null, isResolved: true }),
    );
  });
});

describe("useRepositoryPullRequests", () => {
  it("waits for the host before calling the provider API", async () => {
    configureLocalSecrets({ GITLAB_HOST: "gitlab.example.com" });

    const { result } = renderHook(
      () => {
        const { host, isResolved } = useGitProviderHost("gitlab", null);
        return useRepositoryPullRequests(
          "acme/widgets",
          "gitlab",
          host,
          isResolved,
        );
      },
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(urls).toEqual([
      "https://gitlab.example.com/api/v4/projects/acme%2Fwidgets/merge_requests?state=opened&per_page=30",
    ]);
  });
});
