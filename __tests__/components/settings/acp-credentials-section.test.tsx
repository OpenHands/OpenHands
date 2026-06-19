import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { AcpCredentialsSection } from "#/components/features/settings/acp-credentials-section";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { SecretsService } from "#/api/secrets-service";

// The login-detection probe is exercised in its own hook test; here we stub it
// so rendering the section doesn't spin a subprocess and we can drive the auth
// banner states directly.
const acpAuthStatusMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

// The section is presentational: the form lives in the parent (Settings →
// Agent) so a single Save persists the agent spec + credentials together. These
// tests drive the section through the real form hook to cover field rendering,
// conflict warnings, and the auth banner; the save flow is covered in
// __tests__/routes/agent-settings.test.tsx.
function Harness({ providerKey }: { providerKey: string }) {
  const form = useAcpCredentialForm(providerKey);
  return <AcpCredentialsSection form={form} providerKey={providerKey} />;
}

function renderSection(providerKey: string) {
  const user = userEvent.setup();
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
  return { user };
}

beforeEach(() => {
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  acpAuthStatusMock.mockReturnValue({
    status: "unknown",
    isChecking: false,
    isSupported: true,
  });
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
});
afterEach(() => {
  __resetActiveStoreForTests();
});

describe("AcpCredentialsSection", () => {
  it("renders the provider's credential fields (blob as textarea, key as password)", () => {
    renderSection("codex");

    expect(
      screen.getByTestId("settings-acp-secret-CODEX_AUTH_JSON").tagName,
    ).toBe("TEXTAREA");
    expect(
      screen.getByTestId("settings-acp-secret-OPENAI_API_KEY"),
    ).toHaveAttribute("type", "password");
  });

  it("renders nothing for a provider without credential fields", () => {
    renderSection("custom");
    expect(
      screen.queryByTestId("settings-acp-secret-CODEX_AUTH_JSON"),
    ).not.toBeInTheDocument();
  });

  it("warns when the Claude OAuth token and base URL are both set", async () => {
    const { user } = renderSection("claude-code");

    await user.type(
      screen.getByTestId("settings-acp-secret-CLAUDE_CODE_OAUTH_TOKEN"),
      "oauth-token",
    );
    expect(
      screen.queryByTestId("acp-credential-conflict-warning"),
    ).not.toBeInTheDocument();

    await user.type(
      screen.getByTestId("settings-acp-secret-ANTHROPIC_BASE_URL"),
      "https://proxy.example.com",
    );
    expect(
      screen.getByTestId("acp-credential-conflict-warning"),
    ).toBeInTheDocument();
  });

  it("shows the 'already signed in' banner when the login probe detects a session", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "authenticated",
      isChecking: false,
      isSupported: true,
    });
    renderSection("claude-code");
    expect(
      screen.getByTestId("settings-acp-auth-detected"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("settings-acp-auth-checking"),
    ).not.toBeInTheDocument();
  });

  it("shows the checking spinner while the login probe is in flight", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: true,
      isSupported: true,
    });
    renderSection("claude-code");
    expect(
      screen.getByTestId("settings-acp-auth-checking"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("settings-acp-auth-detected"),
    ).not.toBeInTheDocument();
  });

  it("shows no auth banner when there is no detected session", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unauthenticated",
      isChecking: false,
      isSupported: true,
    });
    renderSection("claude-code");
    expect(
      screen.queryByTestId("settings-acp-auth-detected"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("settings-acp-auth-checking"),
    ).not.toBeInTheDocument();
  });

  it("shows the 'credentials configured' banner when a credential is stored but the probe can't confirm a login (Docker/cloud)", async () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
      { name: "ANTHROPIC_API_KEY" },
    ]);
    renderSection("claude-code");
    expect(
      await screen.findByTestId("settings-acp-auth-configured"),
    ).toBeInTheDocument();
    // A stored credential must never overstate itself as a verified login.
    expect(
      screen.queryByTestId("settings-acp-auth-detected"),
    ).not.toBeInTheDocument();
  });

  it("does not treat a stored non-credential field (base URL) as configured", async () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
      { name: "ANTHROPIC_BASE_URL" },
    ]);
    renderSection("claude-code");
    // Wait for the secrets query to settle on a stable element, then assert.
    await screen.findByTestId("settings-acp-secret-ANTHROPIC_API_KEY");
    expect(
      screen.queryByTestId("settings-acp-auth-configured"),
    ).not.toBeInTheDocument();
  });
});
