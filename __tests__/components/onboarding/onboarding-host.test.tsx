import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { OnboardingHost } from "#/components/features/onboarding/onboarding-host";
import { ONBOARDING_COMPLETED_STORAGE_KEY } from "#/components/features/onboarding/use-onboarding-completion";
import SettingsService from "#/api/settings-service/settings-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { NavigationProvider } from "#/context/navigation-context";

// We don't need to exercise the modal's internals here; just verify
// whether OnboardingHost mounts it at all.
vi.mock("#/components/features/onboarding/onboarding-modal", () => ({
  OnboardingModal: () => (
    <div data-testid="onboarding-modal-stub">onboarding modal</div>
  ),
}));

function renderHost() {
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
      <MemoryRouter initialEntries={["/"]}>
        <ActiveBackendProvider>
          <NavigationProvider value={navigationValue}>
            <OnboardingHost />
          </NavigationProvider>
        </ActiveBackendProvider>
      </MemoryRouter>
    </QueryClientProvider>,
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

function seedUserAddedLocalBackend() {
  // A Local backend the user explicitly added via "Add Backend" — its id
  // is NOT the launcher-seeded SEEDED_DEFAULT_BACKEND_ID, so the
  // "pre-configured server" skip is allowed to fire for it.
  const backend = {
    id: "user-added-local",
    name: "My Agent Server",
    host: "http://localhost:9000",
    apiKey: "session-key",
    kind: "local" as const,
  };
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id, orgId: null });
  return backend;
}

beforeEach(() => {
  window.localStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
  __resetActiveStoreForTests();
  vi.clearAllMocks();
});

afterEach(() => {
  window.localStorage.clear();
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
});

describe("OnboardingHost", () => {
  it("renders the onboarding modal for a fresh install with no configured LLM", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_set: false,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "" },
      },
    });

    renderHost();

    expect(
      await screen.findByTestId("onboarding-modal-stub"),
    ).toBeInTheDocument();
  });

  it("skips the modal when the active Cloud backend has a configured LLM", async () => {
    seedCloudBackend();
    const getSettings = vi
      .spyOn(SettingsService, "getSettings")
      .mockResolvedValue({
        ...DEFAULT_SETTINGS,
        llm_api_key_set: true,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "anthropic/claude-sonnet-4-5", api_key: "stored" },
        },
      });

    renderHost();

    await waitFor(() => {
      expect(getSettings).toHaveBeenCalledOnce();
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("skips the modal for a configured Cloud backend that does not match the locked host", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://other-cloud.example.com");
    seedCloudBackend();
    const getSettings = vi
      .spyOn(SettingsService, "getSettings")
      .mockResolvedValue({
        ...DEFAULT_SETTINGS,
        llm_api_key_set: true,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "anthropic/claude-sonnet-4-5", api_key: "stored" },
        },
      });

    renderHost();

    await waitFor(() => {
      expect(getSettings).toHaveBeenCalledOnce();
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("skips the modal when the active Cloud backend uses subscription auth", async () => {
    seedCloudBackend();
    const getSettings = vi
      .spyOn(SettingsService, "getSettings")
      .mockResolvedValue({
        ...DEFAULT_SETTINGS,
        llm_api_key_set: false,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "openai/gpt-5.5", auth_type: "subscription" },
        },
      });

    renderHost();

    await waitFor(() => {
      expect(getSettings).toHaveBeenCalledOnce();
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("still shows the modal for a Cloud user when an API key is set but no model is configured", async () => {
    seedCloudBackend();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_set: true,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "" },
      },
    });

    renderHost();

    expect(
      await screen.findByTestId("onboarding-modal-stub"),
    ).toBeInTheDocument();
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("shows the modal for a user-added Local backend that already has an LLM configured", async () => {
    seedUserAddedLocalBackend();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_is_set: true,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "openai/zai-org/GLM-5.2", api_key: "**********" },
      },
    });

    renderHost();

    expect(
      await screen.findByTestId("onboarding-modal-stub"),
    ).toBeInTheDocument();
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("still shows the modal for a launcher-seeded default-local backend even when the agent-server reports a configured LLM", async () => {
    // Regression for the mock-LLM E2E fresh-install / happy-path tests:
    // the shared agent-server retains a previously-configured LLM across
    // browser sessions, so keying first-run onboarding off the server's
    // LLM state would suppress the modal for a genuinely fresh browser
    // install. The launcher-seeded default-local backend (id
    // SEEDED_DEFAULT_BACKEND_ID) must not trigger the skip — the
    // `openhands-onboarded` localStorage flag stays the source of truth
    // for first-run detection there.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_is_set: true,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "openai/zai-org/GLM-5.2", api_key: "**********" },
      },
    });

    renderHost();

    expect(
      await screen.findByTestId("onboarding-modal-stub"),
    ).toBeInTheDocument();
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("still shows the modal for a fresh Local agent-server with no API key set", async () => {
    // The default agent-server schema returns a model name (e.g.
    // "gpt-5.5") but llm_api_key_is_set === false until the user
    // configures one. The modal must keep running the LLM step.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_is_set: false,
      llm_api_key_set: false,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "gpt-5.5", api_key: null },
      },
    });

    renderHost();

    expect(
      await screen.findByTestId("onboarding-modal-stub"),
    ).toBeInTheDocument();
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });
});
