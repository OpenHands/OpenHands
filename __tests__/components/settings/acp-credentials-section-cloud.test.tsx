import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  __resetActiveStoreForTests,
  setRegisteredBackends,
  setActiveSelection,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { AcpCredentialsSection } from "#/components/features/settings/acp-credentials-section";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { fetchCloudSecrets } from "#/api/cloud/secrets-service.api";

// The login-detection probe is gated off on cloud backends (it shells the
// host CLI, which doesn't exist there), so it always reports "unknown" — stub
// it to that so this test isolates the credentials-configured path.
const acpAuthStatusMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

// Mock the cloud secrets boundary, NOT SecretsService — so the real
// SecretsService.getSecrets runs and exercises its cloud branch
// (kind === "cloud" → fetchCloudSecrets), which is the path the host-probe
// can never reach. This is the gap the Docker/cloud-only mocked-getSecrets
// test cannot cover.
vi.mock("#/api/cloud/secrets-service.api", () => ({
  fetchCloudSecrets: vi.fn(),
  createCloudSecret: vi.fn(),
  updateCloudSecret: vi.fn(),
  deleteCloudSecret: vi.fn(),
}));

function Harness({ providerKey }: { providerKey: string }) {
  const form = useAcpCredentialForm(providerKey);
  return <AcpCredentialsSection form={form} providerKey={providerKey} />;
}

function renderSection(providerKey: string) {
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <Harness providerKey={providerKey} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

function activateCloudBackend() {
  setRegisteredBackends([
    {
      id: "cloud-1",
      name: "Cloud",
      host: "https://app.example.com",
      apiKey: "",
      kind: "cloud",
    },
  ]);
  setActiveSelection({ backendId: "cloud-1", orgId: "org-1" });
}

beforeEach(() => {
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  acpAuthStatusMock.mockReturnValue({
    status: "unknown",
    isChecking: false,
    isSupported: false,
  });
  vi.mocked(fetchCloudSecrets).mockResolvedValue([]);
});
afterEach(() => {
  __resetActiveStoreForTests();
});

describe("AcpCredentialsSection on a cloud backend (#1244)", () => {
  it("surfaces 'credentials configured' from the cloud secret store, never 'signed in'", async () => {
    activateCloudBackend();
    // A credential resolved through the cloud secrets API (not the host probe,
    // which is gated off on cloud).
    vi.mocked(fetchCloudSecrets).mockResolvedValue([
      { name: "CLAUDE_CODE_OAUTH_TOKEN" },
    ]);

    renderSection("claude-code");

    expect(
      await screen.findByTestId("settings-acp-auth-configured"),
    ).toBeInTheDocument();
    // The real SecretsService.getSecrets routed through the cloud branch.
    expect(fetchCloudSecrets).toHaveBeenCalled();
    // A stored credential must never overstate itself as a verified login.
    expect(
      screen.queryByTestId("settings-acp-auth-detected"),
    ).not.toBeInTheDocument();
  });

  it("does not show 'configured' when the cloud store has no provider credential", async () => {
    activateCloudBackend();
    vi.mocked(fetchCloudSecrets).mockResolvedValue([]);

    renderSection("claude-code");

    // Wait for the secrets query to settle on a stable element, then assert.
    await screen.findByTestId("settings-acp-secret-CLAUDE_CODE_OAUTH_TOKEN");
    expect(
      screen.queryByTestId("settings-acp-auth-configured"),
    ).not.toBeInTheDocument();
  });
});
