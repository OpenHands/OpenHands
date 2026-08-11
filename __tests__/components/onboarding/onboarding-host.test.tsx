import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { OnboardingHost } from "#/components/features/onboarding/onboarding-host";
import { ONBOARDING_COMPLETED_STORAGE_KEY } from "#/components/features/onboarding/use-onboarding-completion";
import { ONBOARDING_DISMISSED_SESSION_KEY_PREFIX } from "#/components/features/onboarding/use-onboarding-dismissal";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import SettingsService from "#/api/settings-service/settings-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import OptionService from "#/api/option-service/option-service.api";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";
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
  OnboardingModal: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="onboarding-modal-stub">
      onboarding modal
      <button type="button" data-testid="dismiss-onboarding" onClick={onClose}>
        dismiss
      </button>
    </div>
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
  window.sessionStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
  __resetActiveStoreForTests();
  vi.restoreAllMocks();
  vi.spyOn(OptionService, "getConfig").mockResolvedValue(
    createMockWebClientConfig(),
  );
  vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
    profiles: [],
    active_profile: null,
  });
  vi.spyOn(AgentProfilesService, "listProfiles").mockResolvedValue({
    profiles: [],
    active_agent_profile_id: null,
  });
});

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
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

  it("skips the modal for a user-added Local backend that already has a usable LLM profile", async () => {
    seedUserAddedLocalBackend();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_is_set: true,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "openai/zai-org/GLM-5.2", api_key: "**********" },
      },
    });
    const listProfiles = vi
      .spyOn(ProfilesService, "listProfiles")
      .mockResolvedValue({
        profiles: [
          {
            name: "default",
            model: "openai/zai-org/GLM-5.2",
            base_url: null,
            api_key_set: true,
          },
        ],
        active_profile: "default",
      });

    renderHost();

    await waitFor(() => {
      expect(listProfiles).toHaveBeenCalledOnce();
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();
  });

  it("reevaluates onboarding when the active backend changes", async () => {
    const configuredBackend = seedUserAddedLocalBackend();
    const unconfiguredBackend = {
      ...configuredBackend,
      id: "unconfigured-local",
      name: "Unconfigured Agent Server",
    };
    setRegisteredBackends([configuredBackend, unconfiguredBackend]);
    setActiveSelection({ backendId: configuredBackend.id, orgId: null });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        agent_kind: "openhands",
      },
    });
    vi.mocked(ProfilesService.listProfiles)
      .mockResolvedValueOnce({
        profiles: [
          {
            name: "default",
            model: "openai/gpt-5.5",
            base_url: null,
            api_key_set: true,
          },
        ],
        active_profile: "default",
      })
      .mockResolvedValueOnce({ profiles: [], active_profile: null });

    renderHost();

    await waitFor(() => {
      expect(ProfilesService.listProfiles).toHaveBeenCalledOnce();
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });

    act(() => {
      setActiveSelection({ backendId: unconfiguredBackend.id, orgId: null });
    });

    expect(
      await screen.findByTestId("onboarding-modal-stub"),
    ).toBeInTheDocument();
    expect(ProfilesService.listProfiles).toHaveBeenCalledTimes(2);
  });

  it("keeps Skip for now scoped to the active backend and browser session", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_set: false,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        agent_kind: "openhands",
      },
    });
    const firstRender = renderHost();

    fireEvent.click(await screen.findByTestId("dismiss-onboarding"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
    expect(
      window.sessionStorage.getItem(
        `${ONBOARDING_DISMISSED_SESSION_KEY_PREFIX}:${SEEDED_DEFAULT_BACKEND_ID}`,
      ),
    ).toBe("1");
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).toBeNull();

    firstRender.unmount();
    renderHost();

    await waitFor(() => {
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
  });

  it("skips the modal for a configured launcher-seeded shared local backend", async () => {
    // Every browser attached to the npm launcher shares backend readiness,
    // so a fresh browser profile must not repeat setup for this backend.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      llm_api_key_is_set: true,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        llm: { model: "openai/zai-org/GLM-5.2", api_key: "**********" },
      },
    });
    const listProfiles = vi
      .spyOn(ProfilesService, "listProfiles")
      .mockResolvedValue({
        profiles: [
          {
            name: "default",
            model: "openai/zai-org/GLM-5.2",
            base_url: null,
            api_key_set: true,
          },
        ],
        active_profile: "default",
      });

    renderHost();

    await waitFor(() => {
      expect(listProfiles).toHaveBeenCalledOnce();
      expect(
        screen.queryByTestId("onboarding-modal-stub"),
      ).not.toBeInTheDocument();
    });
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
