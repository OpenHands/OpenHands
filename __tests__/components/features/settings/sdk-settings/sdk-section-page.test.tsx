import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import SettingsService from "#/api/settings-service/settings-service.api";
import {
  SdkSectionPage,
  type SdkSectionSaveControl,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

const mockUseSearchParams = vi.fn();
vi.mock("react-router", async () => {
  const actual =
    await vi.importActual<typeof import("react-router")>("react-router");
  return {
    ...actual,
    useSearchParams: () => mockUseSearchParams(),
    useRevalidator: () => ({ revalidate: vi.fn() }),
  };
});

const mockUseConfig = vi.fn();
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
    agent_settings_schema:
      overrides.agent_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema,
    conversation_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
      ...overrides.conversation_settings,
    },
    conversation_settings_schema:
      overrides.conversation_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.conversation_settings_schema,
  };
}

function buildSavableSettings(): Settings {
  return buildSettings({
    agent_settings_schema: {
      model_name: "AgentSettings",
      sections: [
        {
          key: "llm",
          label: "LLM",
          fields: [
            {
              key: "llm.endpoint",
              label: "Endpoint",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: "https://api.example.com",
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
          ],
        },
      ],
    },
    agent_settings: {
      "llm.endpoint": "https://api.example.com",
    },
  });
}

function renderSdkSectionPage(
  props: React.ComponentProps<typeof SdkSectionPage>,
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  mockUseConfig.mockReturnValue({
    data: {},
    isLoading: false,
  });
  mockUseSearchParams.mockReturnValue([{ get: () => null }, vi.fn()]);

  return render(React.createElement(SdkSectionPage, props), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
  mockUseConfig.mockReturnValue({
    data: {},
    isLoading: false,
  });
  mockUseSearchParams.mockReturnValue([{ get: () => null }, vi.fn()]);
});

describe("SdkSectionPage", () => {
  it("renders advanced-only fields when a custom initial view is provided", async () => {
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "llm",
          label: "LLM",
          fields: [
            {
              key: "llm.model",
              label: "Model",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: "openai/gpt-4o",
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
            {
              key: "llm.api_version",
              label: "API Version",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: null,
              choices: [],
              depends_on: [],
              prominence: "major",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: schema,
        agent_settings: {
          "llm.model": "openai/gpt-4o",
        },
      }),
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
      getInitialView: () => "advanced",
    });

    expect(
      await screen.findByTestId("sdk-settings-llm.api_version"),
    ).toBeInTheDocument();
  });

  it("preserves the selected view when parent rerenders with the same settings", async () => {
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "llm",
          label: "LLM",
          fields: [
            {
              key: "llm.model",
              label: "Model",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: "openhands/claude-opus-4-5-20251101",
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
            {
              key: "llm.base_url",
              label: "Base URL",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: null,
              choices: [],
              depends_on: [],
              prominence: "major",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: schema,
        agent_settings: {
          "llm.model": "openhands/claude-opus-4-5-20251101",
        },
      }),
    );

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });

    function Wrapper() {
      const [externalValue, setExternalValue] = React.useState("");

      return (
        <SdkSectionPage
          settingsSources={[
            { settingsSource: "agent_settings", sectionKeys: ["llm"] },
          ]}
          header={() => (
            <input
              data-testid="external-state-input"
              value={externalValue}
              onChange={(event) => setExternalValue(event.target.value)}
            />
          )}
        />
      );
    }

    render(<Wrapper />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

    await screen.findByTestId("sdk-section-advanced-toggle");
    await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));
    await screen.findByTestId("sdk-settings-llm.base_url");

    await userEvent.type(screen.getByTestId("external-state-input"), "a");

    await waitFor(() => {
      expect(
        screen.getByTestId("sdk-settings-llm.base_url"),
      ).toBeInTheDocument();
    });
  });

  it("resets from advanced to the inferred basic view after saving when advanced settings match defaults", async () => {
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "llm",
          label: "LLM",
          fields: [
            {
              key: "llm.endpoint",
              label: "Endpoint",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: "https://api.example.com",
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
            {
              key: "llm.api_version",
              label: "API Version",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: null,
              choices: [],
              depends_on: [],
              prominence: "major",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    let persistedSettings = buildSettings({
      agent_settings_schema: schema,
      agent_settings: {
        llm: {
          endpoint: "https://api.example.com",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    vi.spyOn(SettingsService, "saveSettings").mockImplementation(
      async (payload) => {
        const agentSettings = payload.agent_settings_diff as Record<
          string,
          unknown
        >;
        const llmSettings = (agentSettings.llm ?? {}) as Record<
          string,
          unknown
        >;

        persistedSettings = buildSettings({
          agent_settings_schema: schema,
          agent_settings: {
            llm: {
              endpoint:
                typeof llmSettings.endpoint === "string"
                  ? llmSettings.endpoint
                  : "https://api.example.com",
            },
          },
        });

        return true;
      },
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
    });

    await screen.findByTestId("sdk-section-advanced-toggle");
    await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));
    await screen.findByTestId("sdk-settings-llm.api_version");

    const endpointInput = await screen.findByTestId(
      "sdk-settings-llm.endpoint",
    );
    await userEvent.clear(endpointInput);
    await userEvent.type(endpointInput, "https://api.changed.example.com");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(
        screen.queryByTestId("sdk-settings-llm.api_version"),
      ).not.toBeInTheDocument();
    });
  });

  it("resets from all to the inferred basic view after saving when detailed settings match defaults", async () => {
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "llm",
          label: "LLM",
          fields: [
            {
              key: "llm.endpoint",
              label: "Endpoint",
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: "https://api.example.com",
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
            {
              key: "llm.timeout",
              label: "Timeout",
              section: "llm",
              section_label: "LLM",
              value_type: "integer",
              default: 30,
              choices: [],
              depends_on: [],
              prominence: "minor",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    let persistedSettings = buildSettings({
      agent_settings_schema: schema,
      agent_settings: {
        llm: {
          endpoint: "https://api.example.com",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    vi.spyOn(SettingsService, "saveSettings").mockImplementation(
      async (payload) => {
        const agentSettings = payload.agent_settings_diff as Record<
          string,
          unknown
        >;
        const llmSettings = (agentSettings.llm ?? {}) as Record<
          string,
          unknown
        >;

        persistedSettings = buildSettings({
          agent_settings_schema: schema,
          agent_settings: {
            llm: {
              endpoint:
                typeof llmSettings.endpoint === "string"
                  ? llmSettings.endpoint
                  : "https://api.example.com",
            },
          },
        });

        return true;
      },
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
    });

    await screen.findByTestId("sdk-section-all-toggle");
    await userEvent.click(screen.getByTestId("sdk-section-all-toggle"));
    await screen.findByTestId("sdk-settings-llm.timeout");

    const endpointInput = await screen.findByTestId(
      "sdk-settings-llm.endpoint",
    );
    await userEvent.clear(endpointInput);
    await userEvent.type(endpointInput, "https://api.changed.example.com");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(
        screen.queryByTestId("sdk-settings-llm.timeout"),
      ).not.toBeInTheDocument();
    });
  });

  it("shows the advanced toggle when it is forced for a critical-only schema", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSavableSettings(),
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
      forceShowAdvancedView: true,
    });

    await screen.findByTestId("sdk-section-basic-toggle");
    expect(
      screen.getByTestId("sdk-section-advanced-toggle"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-section-all-toggle"),
    ).not.toBeInTheDocument();
  });

  it("shows the all toggle instead of an empty advanced tier for minor-only schemas", async () => {
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "condenser",
          label: "Condenser",
          fields: [
            {
              key: "condenser.enabled",
              label: "Enable memory condensation",
              section: "condenser",
              section_label: "Condenser",
              value_type: "boolean",
              default: true,
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
            {
              key: "condenser.max_size",
              label: "Max size",
              section: "condenser",
              section_label: "Condenser",
              value_type: "integer",
              default: 240,
              choices: [],
              depends_on: ["condenser.enabled"],
              prominence: "minor",
              secret: false,
              required: true,
            },
          ],
        },
      ],
    };

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: schema,
        agent_settings: {
          "condenser.enabled": true,
          "condenser.max_size": 240,
        },
      }),
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["condenser"] },
      ],
    });

    await screen.findByTestId("sdk-section-basic-toggle");
    expect(
      screen.queryByTestId("sdk-section-advanced-toggle"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("sdk-section-all-toggle")).toBeInTheDocument();
  });

  it("renders URL-like schema fields as url inputs", async () => {
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "verification",
          label: "Verification",
          fields: [
            {
              key: "verification.critic_enabled",
              label: "Enable critic",
              section: "verification",
              section_label: "Verification",
              value_type: "boolean",
              default: true,
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: true,
            },
            {
              key: "verification.critic_server_url",
              label: "Critic server URL",
              section: "verification",
              section_label: "Verification",
              value_type: "string",
              default: null,
              choices: [],
              depends_on: ["verification.critic_enabled"],
              prominence: "minor",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: schema,
        agent_settings: {
          verification: {
            critic_enabled: true,
            critic_server_url: "https://critic.example.com",
          },
        },
      }),
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["verification"] },
      ],
      getInitialView: () => "all",
    });

    expect(
      await screen.findByTestId("sdk-settings-verification.critic_server_url"),
    ).toHaveAttribute("type", "url");
  });

  it("shows a success toast after saving settings", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSavableSettings(),
    );
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    const displaySuccessToastSpy = vi.spyOn(
      ToastHandlers,
      "displaySuccessToast",
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
    });

    const endpointInput = await screen.findByTestId(
      "sdk-settings-llm.endpoint",
    );
    await userEvent.clear(endpointInput);
    await userEvent.type(endpointInput, "https://api.changed.example.com");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(displaySuccessToastSpy).toHaveBeenCalled();
    });
  });

  it("saves dirty fields from multiple settings sources into separate diffs", async () => {
    const agentSchema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "verification",
          label: "Verification",
          fields: [
            {
              key: "verification.critic_enabled",
              label: "Enable critic",
              section: "verification",
              section_label: "Verification",
              value_type: "boolean",
              default: false,
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };
    const conversationSchema: NonNullable<
      Settings["conversation_settings_schema"]
    > = {
      model_name: "ConversationSettings",
      sections: [
        {
          key: "verification",
          label: "Verification",
          fields: [
            {
              key: "confirmation_mode",
              label: "Confirmation mode",
              section: "verification",
              section_label: "Verification",
              value_type: "boolean",
              default: false,
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: agentSchema,
        conversation_settings_schema: conversationSchema,
        agent_settings: {
          verification: {
            critic_enabled: false,
          },
        },
        conversation_settings: {
          confirmation_mode: false,
        },
      }),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSdkSectionPage({
      settingsSources: [
        {
          settingsSource: "conversation_settings",
          sectionKeys: ["verification"],
        },
        {
          settingsSource: "agent_settings",
          sectionKeys: ["verification"],
        },
      ],
    });

    const confirmationInput = await screen.findByTestId(
      "sdk-settings-confirmation_mode",
    );
    const criticInput = await screen.findByTestId(
      "sdk-settings-verification.critic_enabled",
    );
    await userEvent.click(confirmationInput.closest("label")!);
    await userEvent.click(criticInput.closest("label")!);
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith({
        conversation_settings_diff: {
          confirmation_mode: true,
        },
        agent_settings_diff: {
          verification: {
            critic_enabled: true,
          },
        },
      });
    });
  });

  it("shows an error toast when saving settings fails", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSavableSettings(),
    );
    vi.spyOn(SettingsService, "saveSettings").mockRejectedValue(
      new AxiosError("Request failed"),
    );
    const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
    });

    const endpointInput = await screen.findByTestId(
      "sdk-settings-llm.endpoint",
    );
    await userEvent.clear(endpointInput);
    await userEvent.type(endpointInput, "https://api.changed.example.com");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(displayErrorToastSpy).toHaveBeenCalled();
    });
  });

  it("renders the schema-unavailable fallback instead of crashing when the schema is malformed", async () => {
    // Simulates the production failure mode we hit on Vercel previews:
    // the frontend points at a host that does not serve
    // `/api/settings/agent-schema`, so the schema query resolves with a
    // truthy object that nevertheless has no `sections` array. The page
    // must surface this as the existing "schema unavailable" message
    // instead of throwing
    // `Cannot read properties of undefined (reading 'filter')` and
    // letting React Router escalate to a full-screen error.
    const malformedSchema = {
      model_name: "AgentSettings",
      // `sections` deliberately omitted to mimic an SPA shell that
      // happened to parse into a non-schema object.
    } as unknown as NonNullable<Settings["agent_settings_schema"]>;

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ agent_settings_schema: malformedSchema }),
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
    });

    expect(
      await screen.findByText("SETTINGS$SDK_SCHEMA_UNAVAILABLE"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-section-settings-screen"),
    ).not.toBeInTheDocument();
  });

  it("allows saving custom payloads when only external state is dirty", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
      extraDirty: true,
      buildPayload: (payload) => ({
        ...payload,
        search_api_key: "external-search-key",
      }),
    });

    await userEvent.click(await screen.findByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({ search_api_key: "external-search-key" }),
      );
    });
  });

  it("exposes the active view and a coerced, dirty-only payload on the save control", async () => {
    // Arrange — a basic-tier schema with a single editable field.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSavableSettings(),
    );
    let latestControl: SdkSectionSaveControl | null = null;

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
      onSaveControlChange: (control) => {
        latestControl = control;
      },
    });

    // Act — change one field so it becomes dirty.
    const endpointInput = await screen.findByTestId(
      "sdk-settings-llm.endpoint",
    );
    await userEvent.clear(endpointInput);
    await userEvent.type(endpointInput, "https://new.example.com");

    // Assert — the control reports the basic view and returns only the dirty
    // field, nested under its section. Consumers (e.g. the local profile
    // editor) rely on both to drive a custom save.
    await waitFor(() => {
      expect(latestControl?.view).toBe("basic");
      expect(latestControl?.getDirtyPayload()).toEqual({
        llm: { endpoint: "https://new.example.com" },
      });
    });
  });
});
