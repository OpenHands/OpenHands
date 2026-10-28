import { createServer } from "node:http";
import { once } from "node:events";
import { http, HttpResponse, passthrough } from "msw";
import { HttpError } from "@openhands/typescript-client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
// Import the named export LlmSettingsScreen directly for testing the form component.
// The default export now renders LlmSettingsLocalView (the profiles manager view).
import LlmSettingsRoute, { LlmSettingsScreen } from "#/routes/llm-settings";
import ConfigService from "#/api/config-service/config-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { server } from "#/mocks/node";
import {
  OPENAI_SUBSCRIPTION_STATUS_PATH,
  OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
} from "#/constants/llm-subscription";
import { useFreeModelsStore } from "#/stores/free-models-store";
import { Settings } from "#/types/settings";
import * as activeBackendContext from "#/contexts/active-backend-context";
import {
  getActiveSelection,
  getRegisteredBackends,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { OpenAISubscriptionAuthCard } from "#/components/features/settings/llm-settings/openai-subscription-auth-card";
import * as useLlmProfilesHook from "#/hooks/query/use-llm-profiles";
import LLMSubscriptionService from "#/api/llm-subscription-service";
import ProviderConnectionsService, {
  type ProviderConnection,
} from "#/api/provider-connections-service/provider-connections-service.api";
import type { SdkSectionSaveControl } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { I18nKey } from "#/i18n/declaration";

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

describe("subscription login on a Cloud backend", () => {
  it.each([
    [404, I18nKey.SETTINGS$SUBSCRIPTION_STATUS_CLOUD_UNAVAILABLE],
    [403, I18nKey.SETTINGS$SUBSCRIPTION_STATUS_CLOUD_UNAUTHORIZED],
    [500, I18nKey.SETTINGS$SUBSCRIPTION_STATUS_FAILED],
  ])("shows safe Cloud guidance for HTTP %i", async (code, message) => {
    const originalBackends = getRegisteredBackends();
    const originalSelection = getActiveSelection();
    setRegisteredBackends([mockCloudBackend]);
    setActiveSelection({ backendId: mockCloudBackend.id });
    server.use(
      http.get(
        `${mockCloudBackend.host}${OPENAI_SUBSCRIPTION_STATUS_PATH}`,
        () =>
          HttpResponse.json(
            { detail: "sensitive-example-token" },
            { status: code },
          ),
      ),
    );
    const view = render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <activeBackendContext.ActiveBackendProvider>
          <OpenAISubscriptionAuthCard isCloud />
        </activeBackendContext.ActiveBackendProvider>
      </QueryClientProvider>,
    );
    try {
      await waitFor(() =>
        expect(screen.getByTestId("subscription-status")).toHaveTextContent(
          message,
        ),
      );
      expect(screen.getByTestId("subscription-status")).not.toHaveTextContent(
        "sensitive-example-token",
      );
      expect(
        screen.getByTestId("subscription-status-detail"),
      ).toHaveTextContent(`HTTP ${code}`);
    } finally {
      view.unmount();
      setRegisteredBackends(originalBackends);
      setActiveSelection(originalSelection);
    }
  });
});

describe("subscription login on a local backend", () => {
  it("requests a device challenge on click and displays its code", async () => {
    const requests: string[] = [];
    const endpoint = createServer((request, response) => {
      requests.push(`${request.method} ${request.url}`);
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(
        JSON.stringify(
          request.url === OPENAI_SUBSCRIPTION_STATUS_PATH
            ? { connected: false }
            : {
                device_code: "device-code",
                user_code: "USER-CODE",
                verification_uri: "https://auth.openai.com/activate",
                interval: 60,
              },
        ),
      );
    });
    endpoint.listen(0, "127.0.0.1");
    await once(endpoint, "listening");
    const address = endpoint.address();
    if (!address || typeof address === "string") throw new Error("No port");
    const host = `http://127.0.0.1:${address.port}`;
    server.use(http.all(`${host}/*`, () => passthrough()));
    const originalBackends = getRegisteredBackends();
    const originalSelection = getActiveSelection();
    const backend = { ...mockLocalBackend, host };
    setRegisteredBackends([backend]);
    setActiveSelection({ backendId: backend.id });
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    const view = render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <activeBackendContext.ActiveBackendProvider>
          <OpenAISubscriptionAuthCard />
        </activeBackendContext.ActiveBackendProvider>
      </QueryClientProvider>,
    );
    try {
      await waitFor(() =>
        expect(screen.getByTestId("subscription-connect")).toBeEnabled(),
      );
      await userEvent.click(screen.getByTestId("subscription-connect"));
      expect(
        await screen.findByTestId("subscription-user-code"),
      ).toHaveTextContent("USER-CODE");
      expect(open).toHaveBeenCalledWith(
        "https://auth.openai.com/activate",
        "_blank",
        "noopener,noreferrer",
      );
      expect(requests).toEqual([
        `GET ${OPENAI_SUBSCRIPTION_STATUS_PATH}`,
        `POST ${OPENAI_SUBSCRIPTION_DEVICE_START_PATH}`,
      ]);
    } finally {
      view.unmount();
      open.mockRestore();
      setRegisteredBackends(originalBackends);
      setActiveSelection(originalSelection);
      await new Promise<void>((resolve, reject) =>
        endpoint.close((error) => (error ? reject(error) : resolve())),
      );
    }
  });
});

describe("LlmSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useFreeModelsStore.getState().setFlags({
      freeModels: new Set(["openhands/kimi-k3"]),
      defaultModel: "openhands/kimi-k3",
    });
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

  const mockSubscriptionSettings = () =>
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

  it("keeps the upgrade message only when subscription endpoints are missing", async () => {
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockRejectedValue(
      new HttpError(404, "Not Found", { detail: "Not Found" }),
    );
    mockSubscriptionSettings();

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");
    await waitFor(() =>
      expect(screen.getByTestId("subscription-status")).toHaveTextContent(
        I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNAVAILABLE,
      ),
    );
    expect(screen.getByTestId("subscription-status")).not.toHaveTextContent(
      I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNREACHABLE,
    );
    expect(screen.getByTestId("subscription-status")).not.toHaveTextContent(
      I18nKey.SETTINGS$SUBSCRIPTION_STATUS_FAILED,
    );
    expect(screen.getByTestId("subscription-status-detail")).toHaveTextContent(
      "HTTP 404",
    );
  });

  it("names a server-side failure instead of blaming the version", async () => {
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockRejectedValue(
      new HttpError(500, "Internal Server Error", {
        detail: "PermissionError: chmod",
      }),
    );
    mockSubscriptionSettings();

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");
    await waitFor(() =>
      expect(screen.getByTestId("subscription-status")).toHaveTextContent(
        I18nKey.SETTINGS$SUBSCRIPTION_STATUS_FAILED,
      ),
    );
    expect(screen.getByTestId("subscription-status")).not.toHaveTextContent(
      I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNAVAILABLE,
    );
    expect(screen.getByTestId("subscription-status-detail")).toHaveTextContent(
      "HTTP 500",
    );
    expect(
      screen.getByTestId("subscription-status-detail"),
    ).not.toHaveTextContent("PermissionError: chmod");
  });

  it("names an unreachable agent server for transport failures", async () => {
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockRejectedValue(
      new Error("Request failed", { cause: new TypeError("fetch failed") }),
    );
    mockSubscriptionSettings();

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");
    await waitFor(() =>
      expect(screen.getByTestId("subscription-status")).toHaveTextContent(
        I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNREACHABLE,
      ),
    );
    expect(screen.getByTestId("subscription-status")).not.toHaveTextContent(
      I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNAVAILABLE,
    );
  });

  it("retries the status check when the retry button is clicked", async () => {
    const getStatus = vi
      .spyOn(LLMSubscriptionService, "getOpenAIStatus")
      .mockRejectedValue(new TypeError("fetch failed"));
    mockSubscriptionSettings();

    renderLlmSettingsScreen();

    await screen.findByTestId("llm-subscription-settings");
    const retry = await screen.findByTestId("subscription-status-retry");
    await userEvent.click(retry);
    await waitFor(() => expect(getStatus).toHaveBeenCalledTimes(2));
  });

  it("exposes canonical subscription metadata through the save control", async () => {
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockResolvedValue({
      vendor: "openai",
      connected: true,
      accountEmail: "graham@example.com",
      expiresAt: null,
    });
    vi.spyOn(LLMSubscriptionService, "getOpenAIModels").mockResolvedValue([
      "gpt-5.6-luna",
    ]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "gpt-5.6-luna",
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "gpt-5.6-luna",
            auth_type: "subscription",
            subscription_vendor: "openai",
          },
        },
      }),
    );
    const saveControlRef: { current: SdkSectionSaveControl | null } = {
      current: null,
    };

    renderLlmSettingsScreen({
      onSaveControlChange: (control) => {
        saveControlRef.current = control;
      },
    });

    await screen.findByTestId("llm-subscription-settings");
    await waitFor(() => expect(saveControlRef.current).not.toBeNull());
    expect(saveControlRef.current?.getSavePayload()).toEqual({
      agent_settings_diff: {
        llm: {
          auth_type: "subscription",
          model: "gpt-5.6-luna",
          subscription_vendor: "openai",
          temperature: null,
        },
      },
    });
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

  it("throws translated error when saving subscription profile before models load", async () => {
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
        llm_model: "gpt-5.6-luna",
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          llm: {
            model: "gpt-5.6-luna",
            auth_type: "subscription",
            subscription_vendor: "openai",
          },
        },
      }),
    );
    const saveControlRef: { current: SdkSectionSaveControl | null } = {
      current: null,
    };

    renderLlmSettingsScreen({
      onSaveControlChange: (control) => {
        saveControlRef.current = control;
      },
    });

    await screen.findByTestId("llm-subscription-settings");
    await waitFor(() => expect(saveControlRef.current).not.toBeNull());
    expect(() => saveControlRef.current?.getSavePayload()).toThrow(
      I18nKey.SETTINGS$SUBSCRIPTION_MODELS_NOT_LOADED,
    );
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
    await waitFor(() =>
      expect(screen.getByTestId("subscription-connect")).toBeEnabled(),
    );
    fireEvent.click(screen.getByTestId("subscription-connect"));
    const userCode = await screen.findByTestId("subscription-user-code");
    expect(userCode).toHaveTextContent("USER-CODE");
    expect(userCode.parentElement).toHaveClass("text-contrast");

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

  it("does not open or poll a login started for a previous workspace", async () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const backendSpy = vi.spyOn(activeBackendContext, "useActiveBackend");
    backendSpy.mockReturnValue({ backend: mockCloudBackend, orgId: "first" });
    vi.spyOn(LLMSubscriptionService, "getOpenAIStatus").mockResolvedValue({
      vendor: "openai",
      connected: false,
      accountEmail: null,
      expiresAt: null,
    });
    vi.spyOn(LLMSubscriptionService, "getOpenAIModels").mockResolvedValue([
      "gpt-5.2-codex",
    ]);
    let finishLogin!: (
      challenge: Awaited<
        ReturnType<typeof LLMSubscriptionService.startOpenAIDeviceLogin>
      >,
    ) => void;
    vi.spyOn(LLMSubscriptionService, "startOpenAIDeviceLogin").mockReturnValue(
      new Promise((resolve) => {
        finishLogin = resolve;
      }),
    );
    const poll = vi.spyOn(LLMSubscriptionService, "pollOpenAIDeviceLogin");
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
    const view = renderLlmSettingsScreen();
    await screen.findByTestId("subscription-connect");
    fireEvent.click(screen.getByTestId("subscription-connect"));
    await waitFor(() => expect(finishLogin).toBeDefined());
    backendSpy.mockReturnValue({ backend: mockCloudBackend, orgId: "second" });
    view.rerender(<LlmSettingsScreen />);
    finishLogin({
      deviceCode: "old-device",
      userCode: "OLD-CODE",
      verificationUri: "https://auth.openai.com/activate",
      verificationUriComplete: null,
      expiresAt: null,
      intervalSeconds: 1,
    });
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(openSpy).not.toHaveBeenCalled();
    expect(screen.queryByTestId("subscription-user-code")).toBeNull();
    expect(poll).not.toHaveBeenCalled();
  });
});

describe("LlmSettingsScreen - provider connection selector", () => {
  const connection: ProviderConnection = {
    id: "conn-1",
    display_name: "My OpenAI",
    provider: "openai",
    base_url: null,
    created_at: 1,
    updated_at: 2,
    api_key_set: true,
  };

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_model: "openai/gpt-4o", llm_api_key_set: true }),
    );
  });

  it("hides the selector on cloud even when a connection is linked", async () => {
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockCloudBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    const listSpy = vi
      .spyOn(ProviderConnectionsService, "list")
      .mockResolvedValue([connection]);

    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      // showProviderConnection omitted → cloud path
      initialValueOverrides: {
        "llm.model": "openai/gpt-4o",
        "llm.provider_connection_id": "conn-1",
      },
    });

    await screen.findByTestId("llm-settings-screen");
    expect(
      screen.queryByTestId("llm-provider-connection-input"),
    ).not.toBeInTheDocument();
    expect(listSpy).not.toHaveBeenCalled();
  });

  it("hides the API key / base URL inputs when linked to a connection", async () => {
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockLocalBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    vi.spyOn(ProviderConnectionsService, "list").mockResolvedValue([
      connection,
    ]);

    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      showProviderConnection: true,
      initialValueOverrides: {
        "llm.model": "openai/gpt-4o",
        "llm.provider_connection_id": "conn-1",
      },
    });

    await screen.findByTestId("llm-settings-screen");
    await screen.findByTestId("llm-provider-connection-input");
    expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    expect(screen.queryByTestId("base-url-input")).not.toBeInTheDocument();
  });

  it("updates form state when a linked provider connection is changed to None", async () => {
    const user = userEvent.setup();
    let latestValues: Record<string, string | boolean> = {};

    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockLocalBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    vi.spyOn(ProviderConnectionsService, "list").mockResolvedValue([
      connection,
    ]);

    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      showProviderConnection: true,
      initialValueOverrides: {
        "llm.model": "openai/gpt-4o",
        "llm.provider_connection_id": "conn-1",
      },
      onSaveControlChange: (control) => {
        latestValues = control.values;
      },
    });

    await screen.findByTestId("llm-settings-screen");
    const selector = await screen.findByTestId("llm-provider-connection-input");
    expect(selector).toHaveValue("My OpenAI");

    await user.click(selector);
    await user.click(await screen.findByText("SETTINGS$MCP_AUTH_MODE_NONE"));

    await waitFor(() => {
      expect(latestValues["llm.provider_connection_id"]).toBe("");
    });
  });

  it("still renders the selector for an orphaned link when no connections load", async () => {
    // Regression: a profile linked to a since-deleted connection would hide the
    // API key / base URL inputs while also hiding the selector, leaving no way
    // to recover the credential or unlink.
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockLocalBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    vi.spyOn(ProviderConnectionsService, "list").mockResolvedValue([]);

    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      showProviderConnection: true,
      initialValueOverrides: {
        "llm.model": "openai/gpt-4o",
        "llm.provider_connection_id": "conn-gone",
      },
    });

    await screen.findByTestId("llm-settings-screen");
    expect(
      await screen.findByTestId("llm-provider-connection-input"),
    ).toBeInTheDocument();
  });
});

describe("LlmSettingsScreen - OpenHands provider on cloud", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useFreeModelsStore.getState().setFlags({
      freeModels: new Set(["openhands/kimi-k3"]),
      defaultModel: "openhands/kimi-k3",
    });
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockCloudBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_model: "openhands/kimi-k3", llm_api_key_set: true }),
    );
    vi.spyOn(ConfigService, "searchProviders").mockResolvedValue({
      items: [{ name: "openhands", verified: true }],
      next_page_id: null,
    });
    vi.spyOn(ConfigService, "searchModels").mockResolvedValue({
      items: [
        {
          provider: "openhands",
          name: "kimi-k3",
          verified: true,
          free: true,
          default: true,
        },
      ],
      next_page_id: null,
    });
  });

  it("hides the inline API key and base URL inputs for an OpenHands provider model", async () => {
    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      initialValueOverrides: {
        "llm.model": "openhands/kimi-k3",
        "llm.api_key": "",
        "llm.base_url": "",
      },
    });

    await screen.findByTestId("llm-settings-screen");

    expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    expect(screen.queryByTestId("base-url-input")).not.toBeInTheDocument();
    // The free-models note is still surfaced so the user understands the model.
    expect(
      await screen.findByTestId("openhands-free-models-note"),
    ).toBeInTheDocument();
  });

  it("hides the OpenHands API key help link on cloud (the key is server-minted)", async () => {
    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      initialValueOverrides: {
        "llm.model": "openhands/kimi-k3",
        "llm.api_key": "",
        "llm.base_url": "",
      },
    });

    await screen.findByTestId("llm-settings-screen");

    expect(
      screen.queryByTestId("openhands-api-key-help"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("openhands-api-key-help-2"),
    ).not.toBeInTheDocument();
  });

  it("strips api_key and base_url from the saved settings payload", async () => {
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    renderLlmSettingsScreen({
      initialValueOverrides: {
        "llm.model": "openhands/kimi-k3",
        "llm.api_key": "should-not-be-sent",
        "llm.base_url": "https://should-not-be-sent.example/v1",
      },
    });

    await screen.findByTestId("llm-settings-screen");
    fireEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => expect(saveSettingsSpy).toHaveBeenCalled());
    const payload = saveSettingsSpy.mock.calls[0][0] as Record<string, unknown>;
    const llmPayload = (payload.agent_settings_diff as Record<string, unknown>)
      .llm as Record<string, unknown>;
    expect(llmPayload.model).toBe("openhands/kimi-k3");
    expect(llmPayload).not.toHaveProperty("api_key");
    expect(llmPayload).not.toHaveProperty("base_url");
  });

  it("still shows the API key input for the OpenHands provider on a local backend", async () => {
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockLocalBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      initialValueOverrides: {
        "llm.model": "openhands/kimi-k3",
        "llm.api_key": "",
        "llm.base_url": "",
      },
    });

    await screen.findByTestId("llm-settings-screen");

    // Local mode still collects an inline key for the OpenHands provider.
    expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
    const openHandsHelp = screen.getByTestId("openhands-api-key-help");
    expect(openHandsHelp).toBeInTheDocument();
    // The OpenHands provider help uses the split TEXT/LINK/SUFFIX keys (not
    // the stale single-string "API Keys tab" key) and links to OpenHands
    // Cloud's API Keys page, where the OpenHands LLM Key section lives.
    expect(openHandsHelp).toHaveTextContent(
      "SETTINGS$OPENHANDS_API_KEY_HELP_TEXT",
    );
    expect(openHandsHelp).toHaveTextContent(
      "SETTINGS$OPENHANDS_API_KEY_HELP_LINK",
    );
    expect(openHandsHelp).toHaveTextContent(
      "SETTINGS$OPENHANDS_API_KEY_HELP_SUFFIX",
    );
    expect(openHandsHelp).toHaveTextContent(
      "SETTINGS$SEE_HERE_FOR_MORE_DETAILS",
    );
    expect(openHandsHelp).not.toHaveTextContent("SETTINGS$NAV_API_KEYS");
    const helpLinks = within(openHandsHelp).getAllByRole("link");
    expect(helpLinks).toHaveLength(2);
    expect(helpLinks[0]).toHaveAttribute(
      "href",
      "https://app.all-hands.dev/settings/api-keys",
    );
    expect(helpLinks[1]).toHaveAttribute(
      "href",
      "https://docs.openhands.dev/usage/local-setup#getting-an-api-key",
    );
  });

  it("still shows the API key input for a non-OpenHands provider on cloud", async () => {
    renderLlmSettingsScreen({
      embedded: true,
      hideSaveButton: true,
      initialValueOverrides: {
        "llm.model": "openai/gpt-4o",
        "llm.api_key": "",
        "llm.base_url": "",
      },
    });

    await screen.findByTestId("llm-settings-screen");

    expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
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
