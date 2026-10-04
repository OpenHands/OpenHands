import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  SetupAcpSecretsStep,
  backendRequiresAcpCredentials,
} from "#/components/features/onboarding/steps/setup-acp-secrets-step";
import { type OnboardingAgentId } from "#/components/features/onboarding/steps/choose-agent-step";
import { SecretsService } from "#/api/secrets-service";

// The login-detection probe is exercised in its own hook test; here we stub it
// so rendering the step doesn't spin a conversation, and so we can drive the
// banner states directly.
const acpAuthStatusMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

function renderStep(
  providerKey: OnboardingAgentId = "claude-code",
  isActive = true,
) {
  const onBack = vi.fn();
  const onNext = vi.fn();
  const user = userEvent.setup();
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <SetupAcpSecretsStep
          providerKey={providerKey}
          isActive={isActive}
          onBack={onBack}
          onNext={onNext}
        />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return { onBack, onNext, user };
}

/**
 * Render the Claude Code step with ANTHROPIC_API_KEY already saved, and wait
 * until that state has loaded (its "already saved" placeholder appears once
 * `useSearchSecrets` resolves). Shared by every test that exercises the
 * existing-secret paths so the fixture lives in one place.
 */
async function renderWithSavedApiKey() {
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
    { name: "ANTHROPIC_API_KEY" },
  ]);
  const handles = renderStep("claude-code");
  const apiKey = screen.getByTestId(
    "onboarding-acp-secret-ANTHROPIC_API_KEY",
  ) as HTMLInputElement;
  await waitFor(() => expect(apiKey.placeholder.length).toBeGreaterThan(0));
  return { ...handles, apiKey };
}

beforeEach(() => {
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  acpAuthStatusMock.mockReturnValue({
    status: "unknown",
    isChecking: false,
    isSupported: false,
  });
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
  vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
});
afterEach(() => {
  // setRegisteredBackends persists to localStorage, which
  // __resetActiveStoreForTests re-reads — clear it so a test's backend
  // registration (e.g. the cloud case) can't leak into the next test.
  localStorage.clear();
  __resetActiveStoreForTests();
});

describe("SetupAcpSecretsStep", () => {
  it("renders the provider's API key and optional base URL fields", () => {
    renderStep("codex");

    expect(
      screen.getByTestId("onboarding-acp-secret-OPENAI_API_KEY"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-acp-secret-OPENAI_BASE_URL"),
    ).toBeInTheDocument();
    // The API key is a password field; the base URL is a plain text input.
    expect(
      screen.getByTestId("onboarding-acp-secret-OPENAI_API_KEY"),
    ).toHaveAttribute("type", "password");
    expect(
      screen.getByTestId("onboarding-acp-secret-OPENAI_BASE_URL"),
    ).toHaveAttribute("type", "text");
  });

  it("flags a credential that already exists as a saved secret", async () => {
    const { apiKey } = await renderWithSavedApiKey();

    // The already-saved field carries a non-empty placeholder hint; a
    // not-yet-saved field (base URL) does not.
    const baseUrl = screen.getByTestId(
      "onboarding-acp-secret-ANTHROPIC_BASE_URL",
    ) as HTMLInputElement;
    expect(apiKey.placeholder.length).toBeGreaterThan(0);
    expect(baseUrl.placeholder).toBe("");
  });

  it("does not write an existing secret when its field is left blank", async () => {
    const { onNext, user } = await renderWithSavedApiKey();

    // Advance without typing: a blank field is a deliberate skip, so the
    // already-saved secret must be left untouched (no overwrite).
    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(SecretsService.createSecret).not.toHaveBeenCalled();
  });

  it("overwrites an existing secret when the user types a replacement", async () => {
    // Key rotation: a credential is already saved, the user types a new value
    // over it. The blank-skip guard must not suppress this — the new value has
    // to be written even though the secret already exists.
    const { onNext, user, apiKey } = await renderWithSavedApiKey();

    await user.type(apiKey, "sk-ant-new-key");
    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));

    await waitFor(() => {
      expect(SecretsService.createSecret).toHaveBeenCalledWith(
        "ANTHROPIC_API_KEY",
        "sk-ant-new-key",
        undefined,
      );
      expect(onNext).toHaveBeenCalledTimes(1);
    });
  });

  it("upserts every filled field as a secret and then advances", async () => {
    const { onNext, user } = renderStep("claude-code");

    await user.type(
      screen.getByTestId("onboarding-acp-secret-ANTHROPIC_API_KEY"),
      "sk-ant-123",
    );
    await user.type(
      screen.getByTestId("onboarding-acp-secret-ANTHROPIC_BASE_URL"),
      "https://proxy.example.com",
    );
    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));

    await waitFor(() => {
      expect(SecretsService.createSecret).toHaveBeenCalledWith(
        "ANTHROPIC_API_KEY",
        "sk-ant-123",
        undefined,
      );
      expect(SecretsService.createSecret).toHaveBeenCalledWith(
        "ANTHROPIC_BASE_URL",
        "https://proxy.example.com",
        undefined,
      );
      expect(onNext).toHaveBeenCalledTimes(1);
    });
  });

  it("does not advance when a secret write fails", async () => {
    vi.spyOn(SecretsService, "createSecret").mockRejectedValue(
      new Error("boom"),
    );
    const { onNext, user } = renderStep("claude-code");

    await user.type(
      screen.getByTestId("onboarding-acp-secret-ANTHROPIC_API_KEY"),
      "sk-ant-123",
    );
    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));

    await waitFor(() =>
      expect(SecretsService.createSecret).toHaveBeenCalledTimes(1),
    );
    expect(onNext).not.toHaveBeenCalled();
  });

  it("runs the login probe scoped to the active step and provider", () => {
    renderStep("claude-code", true);
    expect(acpAuthStatusMock).toHaveBeenCalledWith("claude-code", {
      enabled: true,
    });
  });

  it("disables the login probe when the step is not active", () => {
    renderStep("claude-code", false);
    expect(acpAuthStatusMock).toHaveBeenCalledWith("claude-code", {
      enabled: false,
    });
  });

  it("shows the 'checking' banner while the login probe is in flight", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: true,
      isSupported: true,
    });
    renderStep("claude-code");

    expect(
      screen.getByTestId("onboarding-acp-auth-checking"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-acp-auth-detected"),
    ).not.toBeInTheDocument();
  });

  it("shows the 'already signed in' banner when authenticated, keeping the key fields", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "authenticated",
      isChecking: false,
      isSupported: true,
    });
    renderStep("claude-code");

    expect(
      screen.getByTestId("onboarding-acp-auth-detected"),
    ).toBeInTheDocument();
    // The fields stay visible (now optional) even when already logged in.
    expect(
      screen.getByTestId("onboarding-acp-secret-ANTHROPIC_API_KEY"),
    ).toBeInTheDocument();
  });

  it("renders Gemini's credential fields and the 'signed in' banner together", () => {
    // Gemini's key/base-URL come from the SDK registry like the others, so the
    // step shows the GEMINI_API_KEY field AND the detection banner (its Google
    // login takes precedence, but a key can still be entered).
    acpAuthStatusMock.mockReturnValue({
      status: "authenticated",
      isChecking: false,
      isSupported: true,
    });
    renderStep("gemini-cli");

    expect(
      screen.getByTestId("onboarding-acp-auth-detected"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-acp-secret-GEMINI_API_KEY"),
    ).toBeInTheDocument();
  });

  it("shows no banner when the provider is not authenticated", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unauthenticated",
      isChecking: false,
      isSupported: true,
    });
    renderStep("claude-code");

    expect(
      screen.queryByTestId("onboarding-acp-auth-detected"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-acp-auth-checking"),
    ).not.toBeInTheDocument();
  });

  it("shows the 'credentials configured' banner when a credential is stored but the probe can't confirm a login", async () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
      { name: "ANTHROPIC_API_KEY" },
    ]);
    renderStep("claude-code");

    expect(
      await screen.findByTestId("onboarding-acp-auth-configured"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-acp-auth-detected"),
    ).not.toBeInTheDocument();
  });

  it("renders the Codex subscription blob as a multiline textarea", () => {
    renderStep("codex");

    const blob = screen.getByTestId("onboarding-acp-secret-CODEX_AUTH_JSON");
    expect(blob.tagName).toBe("TEXTAREA");
  });

  it("requires credentials (blocks Next) on a logged-out local backend, then unblocks once one is entered", async () => {
    // local + "unauthenticated" = a fresh container with no host login → the
    // step is required until the user provides a credential.
    acpAuthStatusMock.mockReturnValue({
      status: "unauthenticated",
      isChecking: false,
      isSupported: true,
    });
    const { onNext, user } = renderStep("claude-code");

    expect(
      screen.getByTestId("onboarding-acp-secrets-blocked"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("onboarding-acp-secrets-next")).toBeDisabled();

    await user.type(
      screen.getByTestId("onboarding-acp-secret-CLAUDE_CODE_OAUTH_TOKEN"),
      "oauth-token",
    );

    expect(
      screen.queryByTestId("onboarding-acp-secrets-blocked"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-acp-secrets-next"),
    ).not.toBeDisabled();

    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));
    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
  });

  it("does not block Next when the login probe is unknown (permissive for native dev)", () => {
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: false,
    });
    renderStep("claude-code");

    expect(
      screen.queryByTestId("onboarding-acp-secrets-blocked"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-acp-secrets-next"),
    ).not.toBeDisabled();
  });

  it("stays blocked when only a non-credential field is filled", async () => {
    // GOOGLE_CLOUD_LOCATION (or a base URL) alone can't authenticate anything —
    // only a masked ``secret`` field (blob / token / API key) satisfies a
    // required credential step.
    acpAuthStatusMock.mockReturnValue({
      status: "unauthenticated",
      isChecking: false,
      isSupported: true,
    });
    const { user } = renderStep("gemini-cli");

    await user.type(
      screen.getByTestId("onboarding-acp-secret-GOOGLE_CLOUD_LOCATION"),
      "us-central1",
    );

    expect(
      screen.getByTestId("onboarding-acp-secrets-blocked"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("onboarding-acp-secrets-next")).toBeDisabled();

    await user.type(
      screen.getByTestId("onboarding-acp-secret-GEMINI_API_KEY"),
      "AIza-key",
    );

    expect(
      screen.queryByTestId("onboarding-acp-secrets-blocked"),
    ).not.toBeInTheDocument();
  });

  it("counts a Codex file blob toward the gate on cloud (cloud materialises file secrets)", async () => {
    // Cloud materialises file-content credentials (Codex auth.json, Gemini
    // Vertex SA) from the encrypted secret store via agent_context.secrets at
    // conversation start, so a pasted blob satisfies a required step exactly
    // like an env-var credential does.
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.example.dev",
        apiKey: "key",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1", orgId: null });
    const { user } = renderStep("codex");

    // Required on cloud, so initially blocked until a credential is supplied.
    expect(screen.getByTestId("onboarding-acp-secrets-next")).toBeDisabled();

    await user.click(
      screen.getByTestId("onboarding-acp-secret-CODEX_AUTH_JSON"),
    );
    await user.paste('{"tokens":{}}');

    expect(
      screen.queryByTestId("onboarding-acp-secrets-blocked"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-acp-secrets-next"),
    ).not.toBeDisabled();
  });

  it("holds Next while the login probe is still in flight, without the blocked note", async () => {
    // A fast click must not slip past a gate the probe is about to raise; the
    // "checking login status" banner already explains the wait. A credential
    // typed meanwhile releases the hold.
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: true,
      isSupported: true,
    });
    const { user } = renderStep("claude-code");

    expect(screen.getByTestId("onboarding-acp-secrets-next")).toBeDisabled();
    // Not "blocked" — just pending classification.
    expect(
      screen.queryByTestId("onboarding-acp-secrets-blocked"),
    ).not.toBeInTheDocument();

    await user.type(
      screen.getByTestId("onboarding-acp-secret-CLAUDE_CODE_OAUTH_TOKEN"),
      "oauth-token",
    );

    expect(
      screen.getByTestId("onboarding-acp-secrets-next"),
    ).not.toBeDisabled();
  });

  it("warns when the Claude OAuth token and base URL are both set (bearer-auth conflict)", async () => {
    const { user } = renderStep("claude-code");

    expect(
      screen.queryByTestId("acp-credential-conflict-warning"),
    ).not.toBeInTheDocument();

    await user.type(
      screen.getByTestId("onboarding-acp-secret-CLAUDE_CODE_OAUTH_TOKEN"),
      "oauth-token",
    );
    await user.type(
      screen.getByTestId("onboarding-acp-secret-ANTHROPIC_BASE_URL"),
      "https://proxy.example.com",
    );

    expect(
      screen.getByTestId("acp-credential-conflict-warning"),
    ).toBeInTheDocument();
  });

  it("counts an already-saved secret toward the conflict warning", async () => {
    // A previously saved ANTHROPIC_BASE_URL conflicts just the same as a typed
    // one — the warning must consider the secret store, not just the form.
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
      { name: "ANTHROPIC_BASE_URL" },
    ]);
    const { user } = renderStep("claude-code");
    await waitFor(() =>
      expect(
        (
          screen.getByTestId(
            "onboarding-acp-secret-ANTHROPIC_BASE_URL",
          ) as HTMLInputElement
        ).placeholder.length,
      ).toBeGreaterThan(0),
    );

    await user.type(
      screen.getByTestId("onboarding-acp-secret-CLAUDE_CODE_OAUTH_TOKEN"),
      "oauth-token",
    );

    expect(
      screen.getByTestId("acp-credential-conflict-warning"),
    ).toBeInTheDocument();
  });
});

describe("backendRequiresAcpCredentials", () => {
  it("never requires credentials when a login is already detected", () => {
    expect(backendRequiresAcpCredentials("local", "authenticated")).toBe(false);
    expect(backendRequiresAcpCredentials("cloud", "authenticated")).toBe(false);
  });

  it("always requires credentials on a cloud backend (no host login)", () => {
    expect(backendRequiresAcpCredentials("cloud", "unauthenticated")).toBe(
      true,
    );
    expect(backendRequiresAcpCredentials("cloud", "unknown")).toBe(true);
  });

  it("requires credentials on a logged-out local backend (a fresh container)", () => {
    expect(backendRequiresAcpCredentials("local", "unauthenticated")).toBe(
      true,
    );
  });

  it("stays permissive on a local backend the probe can't classify", () => {
    expect(backendRequiresAcpCredentials("local", "unknown")).toBe(false);
  });
});
