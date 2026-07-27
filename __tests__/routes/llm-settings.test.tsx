import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
// Import the named export LlmSettingsScreen directly for testing the form component.
// The default export now renders LlmSettingsLocalView (the profiles manager view).
import LlmSettingsRoute, { LlmSettingsScreen } from "#/routes/llm-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";
import * as activeBackendContext from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import * as useLlmProfilesHook from "#/hooks/query/use-llm-profiles";
import LLMSubscriptionService from "#/api/llm-subscription-service";

vi.mock("#/hooks/query/use-llm-profiles");
// The profile manager gates mutate controls on this hook; default to a user
// who can manage so the manager renders its full (editable) surface.
vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => true,
}));

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings_schema:
      overrides.agent_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema,
    agent_settings:
      overrides.agent_settings ?? MOCK_DEFAULT_USER_SETTINGS.agent_settings,
  };
}

function renderLlmSettingsScreen(
  props: Parameters<typeof LlmSettingsScreen>[0] = {},
) {
  return render(<LlmSettingsScreen {...props} />, {
    wrapper: ({ children }) => (
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: { queries: { retry: false } },
            })
          }
        >
          {children}
        </QueryClientProvider>
      </MemoryRouter>
    ),
  });
}

function renderLlmSettingsRoute() {
  return render(<LlmSettingsRoute />, {
    wrapper: ({ children }) => (
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: { queries: { retry: false } },
            })
          }
        >
          {children}
        </QueryClientProvider>
      </MemoryRouter>
    ),
  });
}

const mockLocalBackend: Backend = {
  id: "local-1",
  name: "Local Backend",
  host: "http://localhost:18000",
  apiKey: "",
  kind: "local",
};

const mockCloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud Backend",
  host: "https://app.all-hands.dev",
  apiKey: "test-key",
  kind: "cloud",
};

/**
 * Helper to create properly typed mock return values for useLlmProfiles.
 */
function createMockLlmProfilesReturn(
  overrides: Partial<ReturnType<typeof useLlmProfilesHook.useLlmProfiles>> = {},
): ReturnType<typeof useLlmProfilesHook.useLlmProfiles> {
  return {
    data: { profiles: [], active_profile: null },
    isLoading: false,
    error: null,
    isError: false,
    isFetching: false,
    isSuccess: true,
    refetch: vi.fn(),
    ...overrides,
  } as ReturnType<typeof useLlmProfilesHook.useLlmProfiles>;
}

describe("LlmSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the OSS LLM settings form from the SDK schema fallback", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        llm_api_key_set: true,
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "openai/gpt-4o",
            api_key: null,
            base_url: "",
          },
        },
      }),
    );

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-settings-screen");

    expect(screen.getByTestId("llm-provider-input")).toBeInTheDocument();
    expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
  });

  it("shows the API key as set on the global settings page when a key exists", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_model: "openai/gpt-4o", llm_api_key_set: true }),
    );

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-settings-screen");

    expect(screen.getByTestId("set-indicator")).toBeInTheDocument();
    expect(screen.getByTestId("llm-api-key-input")).toHaveValue("");
  });

  it("does not clear an existing base URL on Basic save without a model change", async () => {
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        llm_base_url: "https://custom.example/v1",
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "openai/gpt-4o",
            api_key: null,
            base_url: "https://custom.example/v1",
          },
        },
      }),
    );

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-settings-screen");
    fireEvent.click(screen.getByTestId("sdk-section-basic-toggle"));
    fireEvent.change(screen.getByTestId("llm-api-key-input"), {
      target: { value: "test-api-key" },
    });
    fireEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => expect(saveSettingsSpy).toHaveBeenCalled());
    const payload = saveSettingsSpy.mock.calls[0][0] as Record<string, unknown>;
    const llmPayload = (payload.agent_settings_diff as Record<string, unknown>)
      .llm as Record<string, unknown>;
    expect(llmPayload.api_key).toBe("test-api-key");
    expect(llmPayload).not.toHaveProperty("base_url");
  });

  it("does not show a 'key set' indicator for a brand-new embedded profile even when a global key exists (bug #640)", async () => {
    // A global key exists, but a fresh profile form must look unset so the user
    // knows they have to enter one — otherwise the profile saves with no key.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_model: "openai/gpt-4o", llm_api_key_set: true }),
    );

    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      initialValueOverrides: {
        "llm.model": "",
        "llm.api_key": "",
        "llm.base_url": "",
      },
    });

    await screen.findByTestId("llm-settings-screen");

    expect(screen.getByTestId("llm-api-key-input")).toHaveValue("");
    expect(screen.queryByTestId("set-indicator")).not.toBeInTheDocument();
  });

  it("renders ChatGPT subscription settings without API key fields", async () => {
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockResolvedValue({
      vendor: "openai",
      connected: false,
      accountEmail: null,
      expiresAt: null,
    });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "gpt-5.2-codex",
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "gpt-5.2-codex",
            auth_type: "subscription",
            subscription_vendor: "openai",
          },
        },
      }),
    );

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");

    expect(
      screen.getByTestId("openai-subscription-auth-card"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    expect(screen.queryByTestId("base-url-input")).not.toBeInTheDocument();
  });

  it("disables subscription model controls while models are loading", async () => {
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockResolvedValue({
      vendor: "openai",
      connected: true,
      accountEmail: "graham@example.com",
      expiresAt: null,
    });
    vi.spyOn(LLMSubscriptionService, "getOpenAIModels").mockReturnValue(
      new Promise(() => {}),
    );
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "gpt-5.2-codex",
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "gpt-5.2-codex",
            auth_type: "subscription",
            subscription_vendor: "openai",
          },
        },
      }),
    );

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");

    expect(screen.getByTestId("llm-auth-type-input")).toBeDisabled();
    expect(screen.getByTestId("llm-subscription-model-input")).toBeDisabled();
  });

  it("auto-polls the ChatGPT subscription device login after opening verification", async () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockResolvedValue({
      vendor: "openai",
      connected: false,
      accountEmail: null,
      expiresAt: null,
    });
    vi.spyOn(
      LLMSubscriptionService,
      "startOpenAIDeviceLogin",
    ).mockResolvedValue({
      deviceCode: "device-code",
      userCode: "USER-CODE",
      verificationUri: "https://auth.openai.com/activate",
      verificationUriComplete:
        "https://auth.openai.com/activate?user_code=USER-CODE",
      expiresAt: null,
      intervalSeconds: 1,
    });
    vi.spyOn(LLMSubscriptionService, "getOpenAIModels").mockResolvedValue([
      "gpt-5.2-codex",
    ]);
    const pollLogin = vi
      .spyOn(LLMSubscriptionService, "pollOpenAIDeviceLogin")
      .mockResolvedValue({
        vendor: "openai",
        connected: true,
        accountEmail: "graham@example.com",
        expiresAt: null,
      });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "gpt-5.2-codex",
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "gpt-5.2-codex",
            auth_type: "subscription",
            subscription_vendor: "openai",
          },
        },
      }),
    );

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");
    fireEvent.click(screen.getByTestId("subscription-connect"));
    const userCode = await screen.findByTestId("subscription-user-code");
    expect(userCode).toHaveTextContent("USER-CODE");
    expect(userCode.parentElement).toHaveClass("text-white");

    expect(openSpy).toHaveBeenCalledWith(
      "https://auth.openai.com/activate?user_code=USER-CODE",
      "_blank",
      "noopener,noreferrer",
    );

    await waitFor(
      () => {
        expect(pollLogin).toHaveBeenCalled();
      },
      { timeout: 2500 },
    );
    expect(pollLogin.mock.calls[0]?.[0]).toBe("device-code");
  });
});

describe("LlmSettingsRoute - backend mode rendering", () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    // Default to local backend
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockLocalBackend,
      orgId: null,
    });

    // Mock useLlmProfiles for local mode tests
    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue(
      createMockLlmProfilesReturn(),
    );
  });

  it("renders LlmSettingsLocalView (profile manager) for local backends", async () => {
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockLocalBackend,
      orgId: null,
    });

    renderLlmSettingsRoute();

    // Local mode shows the "Add LLM Profile" button from LlmProfilesManager
    await screen.findByTestId("add-llm-profile");
    expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
  });

  it("renders LlmSettingsLocalView (profile manager) for cloud backends", async () => {
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockCloudBackend,
      orgId: "org-123",
    });

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        llm_api_key_set: true,
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "openai/gpt-4o",
            api_key: null,
            base_url: "",
          },
        },
      }),
    );

    renderLlmSettingsRoute();

    // Cloud now manages the LLM through profiles too (app-server
    // /api/v1/settings/profiles), so the route renders the profile manager
    // — same view as local — rather than the plain settings form.
    await screen.findByTestId("add-llm-profile");
    expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
  });
});
