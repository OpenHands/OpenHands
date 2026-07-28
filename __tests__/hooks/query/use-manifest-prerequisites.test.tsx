import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useManifestPrerequisites } from "#/hooks/query/use-manifest-prerequisites";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SecretsService } from "#/api/secrets-service";
import { createManifest } from "../../manifests/manifest-test-data";
import type { ExtensionManifest } from "#/manifests/types";

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { getSettings: vi.fn() },
}));

vi.mock("#/api/secrets-service", () => ({
  SecretsService: { getSecrets: vi.fn() },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "test-backend", kind: "local" },
    orgId: null,
  }),
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function withPrerequisites(
  requires: ExtensionManifest["requires"],
): ExtensionManifest {
  return createManifest({ requires });
}

function renderPrerequisites(manifest: ExtensionManifest) {
  return renderHook(() => useManifestPrerequisites(manifest), {
    wrapper: createWrapper(),
  });
}

describe("useManifestPrerequisites", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Nothing connected and nothing stored, unless a test says otherwise.
    vi.mocked(SettingsService.getSettings).mockResolvedValue({} as never);
    vi.mocked(SecretsService.getSecrets).mockResolvedValue([]);
  });

  it("blocks setup while a required integration is not connected", async () => {
    // Arrange
    const manifest = withPrerequisites({
      integrations: [
        { id: "github", reason: "Reads pull requests.", enforcement: "block" },
      ],
      secrets: [],
      onUnmet: { behavior: "block", message: "Connect GitHub first." },
    });

    // Act
    const { result } = renderPrerequisites(manifest);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isBlocked).toBe(true);
  });

  it("lets setup continue when an unconnected integration is only advisory", async () => {
    // Arrange
    const manifest = withPrerequisites({
      integrations: [
        { id: "notion", reason: "Publishes the result.", enforcement: "warn" },
      ],
      secrets: [],
      onUnmet: { behavior: "block", message: "Connect the accounts." },
      onWarn: { behavior: "continue", message: "Notion is not connected." },
    });

    // Act
    const { result } = renderPrerequisites(manifest);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect({
      isBlocked: result.current.isBlocked,
      warned: result.current.warningIntegrations.map((w) => w.requirement.id),
    }).toEqual({ isBlocked: false, warned: ["notion"] });
  });

  it("blocks setup while a required credential is absent", async () => {
    // Arrange
    const manifest = withPrerequisites({
      integrations: [],
      secrets: [
        {
          key: "API_TOKEN",
          label: "API token",
          help: "Needed to call the service.",
          required: true,
        },
      ],
      onUnmet: { behavior: "block", message: "Provide a token." },
    });

    // Act
    const { result } = renderPrerequisites(manifest);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.missingSecrets.map((s) => s.key)).toEqual([
      "API_TOKEN",
    ]);
  });

  it("treats a credential as satisfied on its name alone, never reading its value", async () => {
    // The host is only allowed to observe readiness. The secrets listing it
    // consults carries names and descriptions, so there is no value to read.

    // Arrange
    vi.mocked(SecretsService.getSecrets).mockResolvedValue([
      { name: "API_TOKEN", description: "API token" },
    ]);
    const manifest = withPrerequisites({
      integrations: [],
      secrets: [
        {
          key: "API_TOKEN",
          label: "API token",
          help: "Needed to call the service.",
          required: true,
        },
      ],
      onUnmet: { behavior: "block", message: "Provide a token." },
    });

    // Act
    const { result } = renderPrerequisites(manifest);

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect({
      missing: result.current.missingSecrets,
      blocked: result.current.isBlocked,
    }).toEqual({ missing: [], blocked: false });
  });

  it("has nothing to check for a manifest that declares no prerequisites", async () => {
    // Act
    const { result } = renderPrerequisites(createManifest());

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isBlocked).toBe(false);
  });
});
