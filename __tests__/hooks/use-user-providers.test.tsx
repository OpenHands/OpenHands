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
import SettingsService from "#/api/settings-service/settings-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useUserProviders } from "#/hooks/use-user-providers";
import { DEFAULT_SETTINGS } from "#/services/settings";

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

describe("useUserProviders", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
    window.localStorage.clear();
  });

  it("lists only git providers when the backend also reports a login-only provider", async () => {
    // Arrange: an SSO login shows up in provider_tokens_set next to the git token
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      provider_tokens_set: Object.fromEntries(
        ["github", "enterprise_sso"].map((provider) => [provider, null]),
      ),
    });

    // Act
    const { result } = renderHook(() => useUserProviders(), { wrapper });

    // Assert
    await waitFor(() => expect(result.current.providers).toEqual(["github"]));
  });
});
