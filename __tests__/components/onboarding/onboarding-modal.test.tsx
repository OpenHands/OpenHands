import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockInstance } from "vitest";

import {
  __resetActiveStoreForTests,
  getActiveSelection,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { OnboardingModal } from "#/components/features/onboarding/onboarding-modal";
import { ONBOARDING_DEFAULT_LLM_MODEL } from "#/components/features/onboarding/steps/setup-llm-step";
import { NavigationProvider } from "#/context/navigation-context";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SecretsService } from "#/api/secrets-service";
import { DEFAULT_SETTINGS } from "#/services/settings";
import * as telemetry from "#/services/telemetry";

const llmSettingsScreenMock = vi.hoisted(() => vi.fn());
const getServerInfoMock = vi.hoisted(() => vi.fn());
let captureMock: MockInstance<typeof telemetry.trackEvent>;

// Both the backend status badge in the embedded edit form and the
// step-1 health probe ride on `useBackendsHealth`, which resolves
// server metadata through `ServerClient`.
vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock(options?: { host?: string }) {
    return {
      getServerInfo: vi.fn(() => getServerInfoMock(options)),
    };
  }),
  // The always-mounted LLM slide initializes settings hooks even though
  // `LlmSettingsScreen` is stubbed, so provide the minimal client it needs.
  SettingsClient: vi.fn(function SettingsClientMock() {
    return {
      getSettings: vi.fn().mockResolvedValue({}),
    };
  }),
}));

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCurrentCloudApiKey: vi.fn().mockResolvedValue({
    orgId: null,
    isLegacyKey: true,
  }),
}));

// The LLM step renders the full `LlmSettingsScreen`, which transitively
// pulls in agent-server config + schema queries we don't need to
// exercise here. Stub it to a marker so we can still verify the LLM
// step is mounted and inspect the onboarding defaults passed to it.
vi.mock("#/routes/llm-settings", async () => {
  const React = await import("react");

  return {
    LlmSettingsScreen: (props: Record<string, unknown>) => {
      llmSettingsScreenMock(props);
      return React.createElement(
        "div",
        { "data-testid": "llm-settings-screen-stub" },
        "llm settings",
      );
    },
  };
});

vi.mock("#/components/features/backends/device-flow-auth", async () => {
  const React = await import("react");

  return {
    DeviceFlowAuth: ({
      onSuccess,
      testIdRoot,
    }: {
      onSuccess: (apiKey: string) => void;
      testIdRoot: string;
    }) =>
      React.createElement(
        "button",
        {
          type: "button",
          "data-testid": `${testIdRoot}-login-button`,
          onClick: () => onSuccess("cloud-session-key"),
        },
        "Login with OpenHands Cloud",
      ),
  };
});

vi.mock(
  "#/components/features/automations/recommended-automations-launcher",
  () => ({
    RecommendedAutomationsLauncher: ({
      onLaunched,
    }: {
      onLaunched?: () => void;
    }) => (
      <div data-testid="recommended-automations-launcher-stub">
        <button type="button" onClick={onLaunched}>
          launch recommended automation
        </button>
      </div>
    ),
  }),
);

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => false,
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: vi.fn(),
    isPending: false,
    isSuccess: false,
  }),
}));

// The ACP credentials slide runs a login-detection probe (calls
// GET /api/acp/auth-status). Stub it here so the modal routing tests don't hit
// the network; the probe itself is covered in use-acp-auth-status.test.tsx.
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: () => ({
    status: "unknown",
    isChecking: false,
    isSupported: false,
  }),
}));

async function waitForConfiguredBackendToBeSkipped() {
  await waitFor(
    () => {
      expect(screen.queryByTestId("onboarding-step-check-backend")).toBeNull();
      expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
        "data-current-step",
        "0",
      );
      expect(
        within(screen.getByTestId("onboarding-slide-0")).getByTestId(
          "onboarding-step-choose-agent",
        ),
      ).toBeInTheDocument();
    },
    { timeout: 3000 },
  );
}

async function completeAgentStep(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId("onboarding-agent-next"));
  await waitFor(
    () =>
      expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
        "data-current-step",
        "1",
      ),
    { timeout: 3000 },
  );
}

function seedCloudBackend() {
  const backend = {
    id: "cloud-backend",
    name: "OpenHands Cloud",
    host: "https://app.all-hands.dev",
    apiKey: "cloud-session-key",
    kind: "cloud" as const,
  };
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id, orgId: null });
  return backend;
}

function renderModal(onClose = vi.fn()) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const navigationValue = {
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <NavigationProvider value={navigationValue}>
          <OnboardingModal onClose={onClose} />
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  captureMock = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
  __resetActiveStoreForTests();
  // Clear accumulated spy/mock call history so per-test assertions (the
  // ACP secret-write checks and the LLM-defaults mock) don't see calls
  // leaked from a prior test. Covers `llmSettingsScreenMock` too.
  vi.clearAllMocks();
  getServerInfoMock.mockReset();
  getServerInfoMock.mockImplementation((options?: { host?: string }) => {
    if (options?.host?.startsWith("https://127.0.0.1:8000")) {
      return Promise.reject(new Error("Failed to fetch"));
    }
    return Promise.resolve({ version: "1.28.0" });
  });
  // ChooseAgentStep's Next button now persists the selection via
  // saveSettings before advancing. Stub it so the rest of the flow
  // (which these tests focus on) isn't gated on a real HTTP call.
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
    ...DEFAULT_SETTINGS,
    agent_settings: {
      ...DEFAULT_SETTINGS.agent_settings,
      llm: {},
    },
  });
  vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
  // The ACP secrets step lists existing secrets to flag "already saved"
  // fields. Stub the fetch so it doesn't reach a real client (none is
  // wired up in this test) and the field placeholders stay in the
  // not-yet-saved state.
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
  vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
});
afterEach(() => {
  captureMock.mockRestore();
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
});

describe("OnboardingModal", () => {
  it("starts no-backend first-run users on the backend step with each slide offset by its index", () => {
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderModal();

    expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
      "data-current-step",
      "0",
    );
    expect(
      screen.getByTestId("onboarding-step-check-backend"),
    ).toBeInTheDocument();

    expect(screen.getByTestId("onboarding-slide-0")).toHaveAttribute(
      "data-active",
      "true",
    );

    // Progress bar reflects step 1 of 4.
    expect(screen.getByTestId("onboarding-progress-step-0")).toHaveAttribute(
      "data-state",
      "current",
    );
    expect(screen.getByTestId("onboarding-progress-step-1")).toHaveAttribute(
      "data-state",
      "upcoming",
    );
  });

  it("skips backend setup when the configured backend is already healthy", async () => {
    renderModal();

    await waitForConfiguredBackendToBeSkipped();
    expect(screen.getByTestId("onboarding-progress-bar")).toHaveAttribute(
      "aria-valuemax",
      "3",
    );
    expect(screen.getByTestId("onboarding-progress-step-2")).toHaveAttribute(
      "data-state",
      "upcoming",
    );
    expect(screen.queryByTestId("onboarding-progress-step-3")).toBeNull();
    expect(screen.queryByTestId("onboarding-agent-back")).toBeNull();
    expect(getServerInfoMock).toHaveBeenCalled();
  });

  it("starts first-run no-backend onboarding as Add a backend without an error banner", () => {
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderModal();

    expect(screen.getByText("BACKEND$ADD_TITLE")).toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-backend-subtitle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-backend-disconnected"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-backend-checking"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("onboarding-backend-cloud-title")).toBeVisible();
    expect(screen.getByTestId("onboarding-backend-login-button")).toBeVisible();
  });

  it("locks no-backend onboarding to Cloud login when VITE_LOCK_TO_CLOUD is set", () => {
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://cloud.example.com");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderModal();

    expect(
      screen.getByText("ONBOARDING$LOGIN_TO_CLOUD_TITLE"),
    ).toBeInTheDocument();
    expect(screen.queryByText("BACKEND$ADD_TITLE")).not.toBeInTheDocument();
    expect(screen.getByTestId("onboarding-backend-cloud-title")).toBeVisible();
    expect(screen.getByTestId("onboarding-backend-login-button")).toBeVisible();
    expect(
      screen.queryByTestId("onboarding-backend-host"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-backend-api-key"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-backend-advanced-toggle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("onboarding-backend-cloud-host"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("onboarding-skip")).not.toBeInTheDocument();
  });

  it("dismisses the onboarding modal immediately after Cloud login in locked-to-Cloud mode without showing the next step", async () => {
    // Regression for hieptl's flicker report on PR #1389: after logging
    // into OpenHands Cloud in locked-to-Cloud mode, the onboarding modal
    // used to advance to the Choose Agent slide (the "next window"),
    // then get torn down by the root first-run gate, then briefly
    // remounted by OnboardingHost — producing a visible flicker. Cloud
    // login IS the onboarding completion in locked mode, so the modal
    // must call `onClose` (dismiss) immediately instead of advancing,
    // so the next slide never shows.
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    const onClose = vi.fn();
    renderModal(onClose);
    const user = userEvent.setup();

    await user.click(screen.getByTestId("onboarding-backend-login-button"));

    // Cloud login must dismiss the modal (not advance to Choose Agent).
    expect(onClose).toHaveBeenCalledTimes(1);
    // The stale-org reset still runs when replacing a mismatched host;
    // here there was no prior backend, so nothing to assert beyond
    // dismissal. The key contract: no slide advancement happened via
    // the Cloud login path. The backend step is what was visible at
    // click time, and the modal is now dismissing.
  });

  it("keeps the backend step visible for a reachable stale Local backend in locked-to-Cloud mode", async () => {
    // Regression for PR #1389 review: in locked-to-Cloud mode a reachable
    // stale Local backend (one persisted from a previous non-locked
    // session) must NOT skip `CheckBackendStep`. The user has to stay on
    // the backend slide so they can log into the locked Cloud host and
    // replace the stale backend, rather than continuing as Local.
    window.localStorage.clear();
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    const staleLocal = {
      id: "stale-local",
      name: "Local",
      host: "http://127.0.0.1:8000",
      apiKey: "stale-key",
      kind: "local" as const,
    };
    setRegisteredBackends([staleLocal]);
    setActiveSelection({ backendId: staleLocal.id, orgId: null });
    __resetActiveStoreForTests();

    renderModal();

    // The health probe succeeds for the stale Local backend, but the
    // backend slide must remain the active step (not skipped to agent
    // selection) because the backend is not the locked Cloud host.
    await waitFor(() => {
      expect(
        screen.getByTestId("onboarding-step-check-backend"),
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
      "data-current-step",
      "0",
    );
    expect(screen.getByTestId("onboarding-progress-step-0")).toHaveAttribute(
      "data-state",
      "current",
    );
    // Progress bar keeps all 4 steps (backend slide still in the flow).
    expect(screen.getByTestId("onboarding-progress-bar")).toHaveAttribute(
      "aria-valuemax",
      "4",
    );
    // The locked Cloud login UI must be presented for replacement, and
    // the connected-backend "Next" shortcut (which would let the user
    // advance with the stale Local backend still active) must NOT be
    // offered. Keeping the slide mounted is not enough on its own.
    expect(
      screen.getByText("ONBOARDING$LOGIN_TO_CLOUD_TITLE"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("onboarding-backend-cloud-title")).toBeVisible();
    expect(screen.getByTestId("onboarding-backend-login-button")).toBeVisible();
    expect(
      screen.queryByTestId("onboarding-backend-show-configuration"),
    ).toBeNull();
    expect(screen.queryByTestId("onboarding-backend-next")).toBeNull();
    // The misleading "Connected" banner for the stale backend should
    // not render either; the user is being told to log into Cloud.
    expect(screen.queryByTestId("onboarding-backend-subtitle")).toBeNull();
  });

  it("keeps the backend step visible for a reachable Cloud backend on a different host in locked-to-Cloud mode", async () => {
    // A Cloud backend pointing at a host other than the locked Cloud host
    // must also keep `CheckBackendStep` visible — `kind === "cloud"` alone
    // is not enough; the host must match the locked host (normalized).
    window.localStorage.clear();
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    const otherCloud = {
      id: "other-cloud",
      name: "Other Cloud",
      host: "https://other-cloud.example.com",
      apiKey: "other-token",
      kind: "cloud" as const,
    };
    setRegisteredBackends([otherCloud]);
    setActiveSelection({ backendId: otherCloud.id, orgId: null });
    __resetActiveStoreForTests();

    renderModal();

    await waitFor(() => {
      expect(
        screen.getByTestId("onboarding-step-check-backend"),
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
      "data-current-step",
      "0",
    );
    expect(screen.getByTestId("onboarding-progress-bar")).toHaveAttribute(
      "aria-valuemax",
      "4",
    );
  });

  it("clears the stale active org_id when Cloud login replaces a mismatched Cloud backend", async () => {
    // Regression for PR #1389 review: replacing a mismatched Cloud
    // backend updates the backend row's host/apiKey, but the persisted
    // active org_id is keyed to the OLD host's org list. Leaving it in
    // place causes subsequent Cloud API calls to send an invalid
    // X-Org-Id to the locked Cloud host. After Cloud login completes,
    // active.orgId must be reset to null.
    window.localStorage.clear();
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    const otherCloud = {
      id: "other-cloud",
      name: "Other Cloud",
      host: "https://other-cloud.example.com",
      apiKey: "other-token",
      kind: "cloud" as const,
    };
    setRegisteredBackends([otherCloud]);
    setActiveSelection({
      backendId: otherCloud.id,
      orgId: "stale-org-from-other-host",
    });
    __resetActiveStoreForTests();
    expect(getActiveSelection()?.orgId).toBe("stale-org-from-other-host");

    renderModal();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(
        screen.getByTestId("onboarding-backend-login-button"),
      ).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("onboarding-backend-login-button"));

    await waitFor(() => {
      expect(getActiveSelection()?.orgId).toBeNull();
    });
    // The backend row was updated rather than added: still a single
    // entry, but now pointed at the locked Cloud host.
    expect(getActiveSelection()?.backendId).toBe(otherCloud.id);
  });

  it("skips the backend step in locked-to-Cloud mode when the active backend IS the locked Cloud host", async () => {
    // Positive control: when the active backend is the locked Cloud host
    // (and healthy), the backend slide is skipped exactly as before.
    window.localStorage.clear();
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    const lockedCloud = {
      id: "locked-cloud",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev/",
      apiKey: "cloud-token",
      kind: "cloud" as const,
    };
    setRegisteredBackends([lockedCloud]);
    setActiveSelection({ backendId: lockedCloud.id, orgId: null });
    __resetActiveStoreForTests();

    renderModal();

    // Trailing slash on the stored host must normalize-match the locked
    // host, so the backend slide is skipped.
    await waitForConfiguredBackendToBeSkipped();
    expect(screen.getByTestId("onboarding-progress-bar")).toHaveAttribute(
      "aria-valuemax",
      "3",
    );
  });

  it("keeps users on Choose Agent after Cloud login in standard (non-locked) mode", async () => {
    // Regression for #1389 review feedback: in standard mode (no
    // VITE_LOCK_TO_CLOUD), the onboarding backend slide shows both the
    // manual column and the Cloud column. Completing Cloud login from
    // there used to land the user on the Set Up LLM slide because the
    // slide-rail renumbered when `skipBackendStep` flipped before the
    // post-flip step-decrement effect ran. Phase-based state must keep
    // them on Choose Agent regardless of the renumber.
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderModal();
    const user = userEvent.setup();

    // Both columns should be visible in standard mode.
    expect(screen.getByTestId("onboarding-backend-host")).toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-backend-login-button"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("onboarding-backend-login-button"));

    await waitFor(() => {
      expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
        "data-current-step",
        "0",
      );
      expect(
        within(screen.getByTestId("onboarding-slide-0")).getByTestId(
          "onboarding-step-choose-agent",
        ),
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId("onboarding-slide-0")).toHaveAttribute(
      "data-active",
      "true",
    );
    // The Set Up LLM slide is always mounted (transform-translated off-screen),
    // but it must not be the active slide after Cloud login completes.
    expect(screen.getByTestId("onboarding-slide-1")).toHaveAttribute(
      "data-active",
      "false",
    );
  });

  it("does not render an 'Or' divider between manual and Cloud columns", () => {
    // Regression for #1389 review feedback: the "Or" label between
    // BackendConnectionOptions' manual and Cloud columns is visually
    // redundant given the columns are already clearly separated and
    // both have prominent titles. It must not render.
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderModal();

    expect(screen.queryByText("BACKEND$LOGIN_OR")).toBeNull();
  });

  it("shows a connection error when saving an unreachable backend", async () => {
    window.localStorage.clear();
    vi.stubEnv("VITE_BACKEND_BASE_URL", "");
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    __resetActiveStoreForTests();

    renderModal();
    const user = userEvent.setup();

    await user.clear(screen.getByTestId("onboarding-backend-host"));
    await user.type(
      screen.getByTestId("onboarding-backend-host"),
      "https://127.0.0.1:8000",
    );
    await user.clear(screen.getByTestId("onboarding-backend-api-key"));
    await user.type(
      screen.getByTestId("onboarding-backend-api-key"),
      "session-key",
    );
    await user.click(screen.getByTestId("onboarding-backend-next"));

    expect(
      await screen.findByTestId("onboarding-backend-error"),
    ).toHaveTextContent("BACKEND$CONNECTION_TEST_FAILED");
    expect(screen.getByTestId("onboarding-backend-error")).toHaveTextContent(
      "Disconnected",
    );
  });

  it("pre-fills the LLM step with OpenAI GPT-5.5", () => {
    renderModal();

    expect(llmSettingsScreenMock).toHaveBeenCalledTimes(1);
    expect(llmSettingsScreenMock).toHaveBeenCalledWith(
      expect.objectContaining({
        initialValueOverrides: {
          "llm.model": ONBOARDING_DEFAULT_LLM_MODEL,
        },
      }),
    );
  });

  it("does not show backend configuration when the configured backend is healthy", async () => {
    renderModal();

    await waitForConfiguredBackendToBeSkipped();
    expect(screen.queryByTestId("onboarding-backend-connected")).toBeNull();
    expect(
      screen.queryByTestId("onboarding-backend-configuration-fields"),
    ).toBeNull();
  });

  it("advances each step via the per-step Next button and reframes slide offsets", async () => {
    renderModal();
    const user = userEvent.setup();

    // Healthy configured backends are skipped, so agent selection is step 0.
    await waitForConfiguredBackendToBeSkipped();
    expect(screen.getByTestId("onboarding-slide-0")).toHaveAttribute(
      "data-active",
      "true",
    );

    // Step 0 → 1. ChooseAgentStep does an async save before advancing.
    await completeAgentStep(user);
    expect(screen.getByTestId("onboarding-slide-1")).toHaveAttribute(
      "data-active",
      "true",
    );

    // Step 1 → 2
    await user.click(screen.getByTestId("onboarding-llm-next"));
    expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
      "data-current-step",
      "2",
    );
    expect(screen.getByTestId("onboarding-slide-2")).toHaveAttribute(
      "data-active",
      "true",
    );
  });

  it("Skip immediately closes the modal", async () => {
    const onClose = vi.fn();
    renderModal(onClose);

    const user = userEvent.setup();
    await user.click(screen.getByTestId("onboarding-skip"));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("stays open when the user clicks outside it or presses Escape", async () => {
    // Arrange
    const onClose = vi.fn();
    renderModal(onClose);
    const user = userEvent.setup();

    // Act: errant interactions outside the modal box — click the dark
    // backdrop overlay, then press Escape.
    const backdrop = screen.getByRole("dialog")
      .firstElementChild as HTMLElement;
    await user.click(backdrop);
    await user.keyboard("{Escape}");

    // Assert: neither dismisses the flow nor marks onboarding completed
    // (https://github.com/OpenHands/agent-canvas/issues/1085); the modal
    // only closes via explicit actions (Skip / launch).
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByTestId("onboarding-modal")).toBeInTheDocument();
  });

  it("wraps the slide rail in a dedicated scroll region so the modal chrome stays put", () => {
    // Arrange + act: render the modal once.
    renderModal();

    // Assert: the slide rail lives inside the scroll region. Long step
    // content overflows this region rather than the modal itself, so
    // the progress bar above it never scrolls away. Skip sits below the modal.
    const scrollArea = screen.getByTestId("onboarding-scroll-area");
    const rail = screen.getByTestId("onboarding-slide-rail");
    expect(scrollArea.contains(rail)).toBe(true);
  });

  it("keeps the LLM step heading and Back/Next outside the scrollable settings body", async () => {
    // Arrange: render the modal and walk through to the LLM step.
    renderModal();
    const user = userEvent.setup();
    await waitForConfiguredBackendToBeSkipped();
    await completeAgentStep(user);
    // Wait for the LLM slide to become the active one before querying
    // by role — otherwise the heading is `aria-hidden` from inside a
    // not-yet-active slide and getByRole filters it out.
    await waitFor(
      () =>
        expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
          "data-current-step",
          "1",
        ),
      { timeout: 3000 },
    );

    // Act: locate the step's scrollable settings wrapper and the chrome
    // around it that the user expects to remain visible.
    const step = screen.getByTestId("onboarding-step-setup-llm");
    const settings = within(step).getByTestId("onboarding-llm-settings");
    const heading = within(step).getByRole("heading", { level: 2 });
    const back = within(step).getByTestId("onboarding-llm-back");
    const next = within(step).getByTestId("onboarding-llm-next");

    // Assert: heading and footer buttons are siblings of the settings
    // body, not descendants. Anything moved inside the settings wrapper

    // would scroll out of view on the All tab — this is the invariant
    // the fix relies on.
    expect(settings.contains(heading)).toBe(false);
    expect(settings.contains(back)).toBe(false);
    expect(settings.contains(next)).toBe(false);
  });

  it("hides the Say Hello OR separator when recommended automations are unavailable on Cloud", () => {
    seedCloudBackend();

    renderModal();

    expect(screen.queryByTestId("onboarding-hello-or-separator")).toBeNull();
    expect(
      screen.queryByTestId("onboarding-recommended-automations"),
    ).toBeNull();
  });

  it("shows the setup slide with Gemini's credential fields", async () => {
    renderModal();
    const user = userEvent.setup();

    // Pick Gemini CLI: its key/base-URL come from the SDK registry like the
    // other providers, so the slide shows the GEMINI_API_KEY field.
    await waitForConfiguredBackendToBeSkipped();
    await user.click(screen.getByTestId("onboarding-agent-option-gemini-cli"));
    await completeAgentStep(user);

    // Lands on the setup slide (the ACP step) — not jumped past to Say Hello.
    await waitFor(
      () =>
        expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
          "data-current-step",
          "1",
        ),
      { timeout: 3000 },
    );
    expect(screen.getByTestId("onboarding-slide-1")).toHaveAttribute(
      "data-active",
      "true",
    );
    expect(
      screen.getByTestId("onboarding-step-setup-acp-secrets"),
    ).toBeInTheDocument();
    // Gemini exposes credential fields (GEMINI_API_KEY), derived from the SDK
    // registry like Claude Code / Codex.
    expect(
      screen.getByTestId("onboarding-acp-secret-GEMINI_API_KEY"),
    ).toBeInTheDocument();

    // The flow skips backend setup and keeps three progress segments.
    expect(
      screen.getByTestId("onboarding-progress-step-2"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("onboarding-progress-step-3")).toBeNull();
    expect(screen.getByTestId("onboarding-progress-step-1")).toHaveAttribute(
      "data-state",
      "current",
    );
  });

  it("shows the ACP credentials step for Claude Code and saves entered keys as secrets", async () => {
    renderModal();
    const user = userEvent.setup();

    // Pick Claude Code after configuring the backend.
    await waitForConfiguredBackendToBeSkipped();
    await user.click(screen.getByTestId("onboarding-agent-option-claude-code"));
    await completeAgentStep(user);

    // The setup slide is the ACP credentials step — not Say Hello — after
    // skipping the already healthy backend.
    await waitFor(
      () =>
        expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
          "data-current-step",
          "1",
        ),
      { timeout: 3000 },
    );
    expect(screen.getByTestId("onboarding-slide-1")).toHaveAttribute(
      "data-active",
      "true",
    );
    expect(
      screen.getByTestId("onboarding-step-setup-acp-secrets"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-progress-step-2"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("onboarding-progress-step-3")).toBeNull();

    // Both Anthropic credentials are offered; the optional base URL too.
    const apiKeyField = screen.getByTestId(
      "onboarding-acp-secret-ANTHROPIC_API_KEY",
    );
    expect(apiKeyField).toBeInTheDocument();
    expect(
      screen.getByTestId("onboarding-acp-secret-ANTHROPIC_BASE_URL"),
    ).toBeInTheDocument();

    // Fill the API key and advance: the value is upserted as a global
    // secret of the same name, then the flow moves on to Say Hello.
    await user.type(apiKeyField, "sk-ant-test");
    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));

    await waitFor(() => {
      expect(SecretsService.createSecret).toHaveBeenCalledWith(
        "ANTHROPIC_API_KEY",
        "sk-ant-test",
        undefined,
      );
    });
    await waitFor(
      () =>
        expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
          "data-current-step",
          "2",
        ),
      { timeout: 3000 },
    );
  });

  it("skips the secret write when the ACP credentials step is left blank", async () => {
    renderModal();
    const user = userEvent.setup();

    await waitForConfiguredBackendToBeSkipped();
    await user.click(screen.getByTestId("onboarding-agent-option-codex"));
    await completeAgentStep(user);
    await waitFor(
      () =>
        expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
          "data-current-step",
          "1",
        ),
      { timeout: 3000 },
    );

    // Leaving every field empty is a deliberate skip — no secret is
    // written, and the user still advances to Say Hello.
    await user.click(screen.getByTestId("onboarding-acp-secrets-next"));
    await waitFor(
      () =>
        expect(screen.getByTestId("onboarding-modal")).toHaveAttribute(
          "data-current-step",
          "2",
        ),
      { timeout: 3000 },
    );
    expect(SecretsService.createSecret).not.toHaveBeenCalled();
  });

  it("pre-fills the say-hello input with the default greeting on the final step", async () => {
    renderModal();
    const user = userEvent.setup();

    await waitForConfiguredBackendToBeSkipped();
    await completeAgentStep(user);
    await user.click(screen.getByTestId("onboarding-llm-next"));

    const helloInput = screen.getByTestId(
      "onboarding-hello-input",
    ) as HTMLInputElement;
    // Translation is mocked to return the key; the default-message
    // hook still pre-fills with whatever t() returns, which here is
    // the I18nKey itself. The contract under test is that the input
    // is non-empty and matches the resolved default message.
    expect(helloInput.value).toBe("ONBOARDING$HELLO_DEFAULT_MESSAGE");
  });

  it("shows recommended automations below the Say Hello input", async () => {
    const onClose = vi.fn();
    renderModal(onClose);
    const user = userEvent.setup();

    await waitForConfiguredBackendToBeSkipped();
    await completeAgentStep(user);
    await waitFor(() =>
      expect(screen.getByTestId("onboarding-slide-1")).toHaveAttribute(
        "data-active",
        "true",
      ),
    );
    await user.click(screen.getByTestId("onboarding-llm-next"));
    await waitFor(() =>
      expect(screen.getByTestId("onboarding-slide-2")).toHaveAttribute(
        "data-active",
        "true",
      ),
    );

    const helloInput = screen.getByTestId("onboarding-hello-input");
    const recommendations = screen.getByTestId(
      "onboarding-recommended-automations",
    );
    expect(
      helloInput.compareDocumentPosition(recommendations) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      within(recommendations).getByTestId(
        "recommended-automations-launcher-stub",
      ),
    ).toBeInTheDocument();

    expect(recommendations.closest("form")).toBeNull();

    await user.click(
      within(recommendations).getByRole("button", {
        name: "launch recommended automation",
      }),
    );
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  describe("analytics events", () => {
    // Surface only the captures for a given event; a session emits several.
    const eventCalls = (event: string) =>
      captureMock.mock.calls.filter(([name]) => name === event);

    beforeEach(() => {
      // Grant analytics consent so events pass useTracking's consent gate;
      // the outer beforeEach seeds settings with consent withheld.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        ...DEFAULT_SETTINGS,
        agent_settings: { ...DEFAULT_SETTINGS.agent_settings, llm: {} },
        user_consents_to_analytics: true,
      });
    });

    it("captures onboarding_started once per session even across a remount", async () => {
      // Arrange + act: open onboarding and let the first mount settle (a Step
      // Viewed capture confirms consent resolved and backend health settled).
      const firstMount = renderModal();
      await waitFor(() =>
        expect(
          eventCalls("onboarding_step_viewed").length,
        ).toBeGreaterThanOrEqual(1),
      );

      // Act: tear the modal down and mount it again in the same session. The
      // remounted modal re-emits Step Viewed from a fresh ref, so waiting for
      // the second one proves the Started effect had its chance to re-fire.
      firstMount.unmount();
      renderModal();
      await waitFor(() =>
        expect(
          eventCalls("onboarding_step_viewed").length,
        ).toBeGreaterThanOrEqual(2),
      );

      // Assert: the sessionStorage guard kept Started at a single capture.
      expect(eventCalls("onboarding_started")).toHaveLength(1);
    });

    it("captures no onboarding events while analytics consent is not granted", async () => {
      // Arrange: withhold consent.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        ...DEFAULT_SETTINGS,
        agent_settings: { ...DEFAULT_SETTINGS.agent_settings, llm: {} },
        user_consents_to_analytics: false,
      });

      // Act: open onboarding and let it settle on the first step.
      renderModal();
      await waitForConfiguredBackendToBeSkipped();

      // Assert: nothing reached PostHog.
      expect(captureMock).not.toHaveBeenCalled();
    });

    it("captures onboarding_step_viewed for each step as the user advances", async () => {
      // Arrange: the healthy backend is skipped, so the user lands on the
      // agent step (index 0 of the 3-step flow).
      renderModal();
      const user = userEvent.setup();
      await waitForConfiguredBackendToBeSkipped();

      // Assert: the entry step carries the full funnel identity.
      await waitFor(() =>
        expect(captureMock).toHaveBeenCalledWith(
          "onboarding_step_viewed",
          expect.objectContaining({
            step: "agent",
            step_index: 0,
            total_steps: 3,
            agent: "openhands",
          }),
        ),
      );

      // Act: advance to the next step.
      await completeAgentStep(user);

      // Assert: the new step is captured too — tracking follows transitions,
      // not just the initial mount.
      await waitFor(() =>
        expect(captureMock).toHaveBeenCalledWith(
          "onboarding_step_viewed",
          expect.objectContaining({
            step: "setup",
            step_index: 1,
            total_steps: 3,
          }),
        ),
      );
    });

    it("captures onboarding_completed when a conversation is launched from the final step", async () => {
      // Arrange: walk through to the final Say Hello step.
      renderModal();
      const user = userEvent.setup();
      await waitForConfiguredBackendToBeSkipped();
      await completeAgentStep(user);
      await user.click(screen.getByTestId("onboarding-llm-next"));
      await waitFor(() =>
        expect(screen.getByTestId("onboarding-slide-2")).toHaveAttribute(
          "data-active",
          "true",
        ),
      );

      // Act: launch from the final step. The stubbed recommended-automation
      // launcher invokes the same onLaunched completion callback as Say Hello.
      const recommendations = screen.getByTestId(
        "onboarding-recommended-automations",
      );
      await user.click(
        within(recommendations).getByRole("button", {
          name: "launch recommended automation",
        }),
      );

      // Assert.
      expect(captureMock).toHaveBeenCalledWith(
        "onboarding_completed",
        expect.objectContaining({ agent: "openhands" }),
      );
    });

    it("captures onboarding_skipped with the current step when the user skips", async () => {
      // Arrange: healthy backend skipped → user is on the agent step (0 of 3).
      renderModal();
      const user = userEvent.setup();
      await waitForConfiguredBackendToBeSkipped();

      // Act: skip out of onboarding.
      await user.click(screen.getByTestId("onboarding-skip"));

      // Assert.
      expect(captureMock).toHaveBeenCalledWith(
        "onboarding_skipped",
        expect.objectContaining({
          step: "agent",
          step_index: 0,
          total_steps: 3,
          agent: "openhands",
        }),
      );
    });
  });
});
