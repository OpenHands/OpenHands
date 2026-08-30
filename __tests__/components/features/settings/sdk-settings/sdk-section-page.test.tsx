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
  { strict = false }: { strict?: boolean } = {},
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

  const result = render(React.createElement(SdkSectionPage, props), {
    wrapper: ({ children }) => {
      const tree = (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      );
      return strict ? <React.StrictMode>{tree}</React.StrictMode> : tree;
    },
  });
  // Exposed so a test can drive a background refetch, which is otherwise
  // unreachable from outside the component.
  return { ...result, queryClient };
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

  it("renders each field once when the schema has duplicate sections for a key", async () => {
    // The combined AgentSettings schema emits an "llm" section for each agent
    // variant ("openhands" and "acp") with identical field keys. Only the first
    // is used; without de-duplication every field would render twice (and React
    // would warn about duplicate keys).
    const llmSection = {
      key: "llm",
      label: "LLM",
      fields: [
        {
          key: "llm.api_version",
          label: "API Version",
          section: "llm",
          section_label: "LLM",
          value_type: "string" as const,
          default: null,
          choices: [],
          depends_on: [],
          prominence: "minor" as const,
          secret: false,
          required: false,
        },
      ],
    };
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [llmSection, { ...llmSection }],
    };

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: schema,
        agent_settings: {},
      }),
    );

    renderSdkSectionPage({
      settingsSources: [
        { settingsSource: "agent_settings", sectionKeys: ["llm"] },
      ],
      getInitialView: () => "all",
    });

    const fields = await screen.findAllByTestId("sdk-settings-llm.api_version");
    expect(fields).toHaveLength(1);
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

  it("floors a critical-less schema at advanced under StrictMode", async () => {
    // StrictMode double-invokes state updaters, so the hydration updater must
    // stay pure: an impure one takes its already-hydrated branch on the second
    // (kept) call and pins the page to the empty basic tier (#16097).
    const schema: NonNullable<Settings["agent_settings_schema"]> = {
      model_name: "AgentSettings",
      sections: [
        {
          key: "agent_context",
          label: "Agent Context",
          fields: [
            {
              key: "agent_context.load_memory",
              label: "Persistent memory",
              section: "agent_context",
              section_label: "Agent Context",
              value_type: "boolean",
              default: false,
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
        agent_settings: { "agent_context.load_memory": false },
      }),
    );

    renderSdkSectionPage(
      {
        settingsSources: [
          { settingsSource: "agent_settings", sectionKeys: ["agent_context"] },
        ],
      },
      { strict: true },
    );

    expect(
      await screen.findByTestId("sdk-settings-agent_context.load_memory"),
    ).toBeInTheDocument();
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

  describe("unsaved edits across background refetches", () => {
    /**
     * Two fields so a refetch can change one while the user is editing the
     * other — the case that separates a real baseline/overlay split from
     * simply refusing to re-hydrate.
     */
    function buildTwoFieldSettings(
      llm: { endpoint?: string; model?: string } = {},
    ): Settings {
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
                {
                  key: "llm.model",
                  label: "Model",
                  section: "llm",
                  section_label: "LLM",
                  value_type: "string",
                  default: "gpt-4",
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
          llm: {
            endpoint: "https://api.example.com",
            model: "gpt-4",
            ...llm,
          },
        },
      });
    }

    const AGENT_LLM_SOURCE = [
      { settingsSource: "agent_settings" as const, sectionKeys: ["llm"] },
    ];

    it("keeps an unsaved edit when a settings refetch lands", async () => {
      // The bug: the hydration effect replaced local values on every refetch,
      // so a background refresh silently reverted what the user had typed
      // while the form stayed on screen.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
      });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://typed.example.com");

      // A refetch resolves with different server state.
      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({ model: "gpt-5-from-server" }),
      );
      await queryClient.invalidateQueries();

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "gpt-5-from-server",
        );
      });
      // The untouched field took the server's new value; the edited one did not.
      expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
        "https://typed.example.com",
      );
    });

    it("still shows the edit as dirty after the refetch", async () => {
      // Preserving the value but clearing the flag would leave Save disabled
      // on a form that visibly holds unsaved input.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
      });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://typed.example.com");

      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({ model: "gpt-5-from-server" }),
      );
      await queryClient.invalidateQueries();

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "gpt-5-from-server",
        );
      });
      expect(screen.getByTestId("save-button")).not.toBeDisabled();
    });

    it("submits the edited value, not the value the refetch brought in", async () => {
      // The payload is built from the merged view, so an overlay that
      // displayed correctly but saved the baseline would still lose the edit.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
      });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://typed.example.com");

      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({
          endpoint: "https://server-moved-on.example.com",
        }),
      );
      await queryClient.invalidateQueries();
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            agent_settings_diff: expect.objectContaining({
              llm: expect.objectContaining({
                endpoint: "https://typed.example.com",
              }),
            }),
          }),
        );
      });
    });

    it("keeps the saved value on screen after a successful save", async () => {
      // Clearing the overlay without folding the save into the baseline would
      // flick the field back to its pre-save value until the refetch landed.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      vi.spyOn(SettingsService, "saveSettings").mockImplementation(async () => {
        getSettingsSpy.mockResolvedValue(
          buildTwoFieldSettings({ endpoint: "https://saved.example.com" }),
        );
        return true;
      });

      renderSdkSectionPage({ settingsSources: AGENT_LLM_SOURCE });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://saved.example.com");
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(screen.getByTestId("save-button")).toBeDisabled();
      });
      expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
        "https://saved.example.com",
      );
    });

    it("preserves the edit when the save fails", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildTwoFieldSettings(),
      );
      vi.spyOn(SettingsService, "saveSettings").mockRejectedValue(
        new AxiosError("nope"),
      );

      renderSdkSectionPage({ settingsSources: AGENT_LLM_SOURCE });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://failed.example.com");
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(screen.getByTestId("save-button")).not.toBeDisabled();
      });
      expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
        "https://failed.example.com",
      );
    });

    it("discards edits when the source selection changes", async () => {
      // Guard, not a bug repro: a scope or source change is a different form,
      // so edits made against the previous one must not carry over and get
      // submitted somewhere they were never typed. Both go through the same
      // reset effect, but only the source half is reachable: `getSettingsQueryFn`
      // throws `Unsupported settings scope` for anything but "personal"
      // (`src/hooks/query/use-settings.ts`), so no scope change exists to test
      // anywhere in the product today.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildTwoFieldSettings(),
      );

      const { rerender } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
      });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://personal.example.com");
      expect(screen.getByTestId("save-button")).not.toBeDisabled();

      rerender(
        React.createElement(SdkSectionPage, {
          settingsSources: [
            {
              settingsSource: "agent_settings" as const,
              sectionKeys: ["llm"],
              excludeKeys: new Set<string>(),
            },
          ],
        }),
      );

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
          "https://api.example.com",
        );
      });
      expect(screen.getByTestId("save-button")).toBeDisabled();
    });

    it("keeps initial overrides dirty across a refetch", async () => {
      // Overrides are prefilled on the user's behalf and must stay savable
      // untouched — a refetch must not quietly clean them.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
        initialValueOverrides: { "llm.model": "prefilled-model" },
      });

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "prefilled-model",
        );
      });
      expect(screen.getByTestId("save-button")).not.toBeDisabled();

      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({ endpoint: "https://moved.example.com" }),
      );
      await queryClient.invalidateQueries();

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
          "https://moved.example.com",
        );
      });
      expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
        "prefilled-model",
      );
      expect(screen.getByTestId("save-button")).not.toBeDisabled();
    });

    it("keeps an unsaved edit when the refetch brings a changed schema", async () => {
      // The other half of the same acceptance criterion. Note this drives the
      // settings query carrying a new inline schema, not the schema query:
      // `useSettingsSchema` short-circuits to its fallback whenever
      // `settings.agent_settings_schema` is present, which it always is here.
      // Same hydration path either way, since initial values are derived from
      // the schema as well as the settings.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
      });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://typed.example.com");

      // Same values, but the schema gains a field — a new schema identity.
      const withExtraField = buildTwoFieldSettings();
      withExtraField.agent_settings_schema!.sections[0].fields.push({
        key: "llm.temperature",
        label: "Temperature",
        section: "llm",
        section_label: "LLM",
        value_type: "string",
        default: "0.5",
        choices: [],
        depends_on: [],
        prominence: "critical",
        secret: false,
        required: false,
      });
      getSettingsSpy.mockResolvedValue(withExtraField);
      await queryClient.invalidateQueries();

      await waitFor(() => {
        expect(
          screen.getByTestId("sdk-settings-llm.temperature"),
        ).toBeInTheDocument();
      });
      expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
        "https://typed.example.com",
      );
    });

    it("keeps per-source edits separate when a refetch lands on a multi-source page", async () => {
      // Two sources that both expose a "verification" section. An overlay
      // keyed by source must not let one source's edit leak into the other's
      // diff, and a refetch must not merge them.
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
      const build = () =>
        buildSettings({
          agent_settings_schema: agentSchema,
          conversation_settings_schema: conversationSchema,
          agent_settings: { verification: { critic_enabled: false } },
          conversation_settings: { confirmation_mode: false },
        });

      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(build());
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { queryClient } = renderSdkSectionPage({
        settingsSources: [
          {
            settingsSource: "conversation_settings",
            sectionKeys: ["verification"],
          },
          { settingsSource: "agent_settings", sectionKeys: ["verification"] },
        ],
      });

      const criticInput = await screen.findByTestId(
        "sdk-settings-verification.critic_enabled",
      );
      await userEvent.click(criticInput.closest("label")!);

      // Must differ in content: an identical payload is structurally shared by
      // react-query, so `settings` keeps its identity and hydration never runs
      // — the refetch this test exists to exercise would not happen.
      const moved = build();
      (
        moved.conversation_settings as Record<string, unknown>
      ).confirmation_mode = true;
      const callsBefore = getSettingsSpy.mock.calls.length;
      getSettingsSpy.mockResolvedValue(moved);
      await queryClient.invalidateQueries();
      await waitFor(() => {
        expect(getSettingsSpy.mock.calls.length).toBeGreaterThan(callsBefore);
      });

      await userEvent.click(screen.getByTestId("save-button"));

      // Only the agent source is dirty, so only its diff is submitted.
      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalledWith({
          agent_settings_diff: { verification: { critic_enabled: true } },
        });
      });
    });

    it("does not strand the form when a refetch drops an edited field", async () => {
      // An overlay entry for a field the schema no longer defines can never be
      // rendered or submitted, but it still counted as dirty — leaving Save
      // enabled over a payload that builds empty and silently no-ops.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
      });

      const modelInput = await screen.findByTestId("sdk-settings-llm.model");
      await userEvent.clear(modelInput);
      await userEvent.type(modelInput, "edited-then-removed");

      // The refetch returns a schema without `llm.model`.
      const withoutModel = buildTwoFieldSettings();
      withoutModel.agent_settings_schema!.sections[0].fields =
        withoutModel.agent_settings_schema!.sections[0].fields.filter(
          (f) => f.key !== "llm.model",
        );
      getSettingsSpy.mockResolvedValue(withoutModel);
      await queryClient.invalidateQueries();

      await waitFor(() => {
        expect(
          screen.queryByTestId("sdk-settings-llm.model"),
        ).not.toBeInTheDocument();
      });
      expect(screen.getByTestId("save-button")).toBeDisabled();
      expect(saveSettingsSpy).not.toHaveBeenCalled();
    });

    it("does not let a prefill overwrite the value the user saved", async () => {
      // Overrides are a local prefill, so they belong in the overlay. Merging
      // them into the baseline let the prefill outrank the server copy — and
      // because `useSaveSettings` invalidates on success, the refetch that
      // confirms a save would immediately revert the saved value and leave it
      // looking clean. That is this issue's own bug, on the save path.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings({ model: "server-original" }));
      vi.spyOn(SettingsService, "saveSettings").mockImplementation(async () => {
        getSettingsSpy.mockResolvedValue(
          buildTwoFieldSettings({ model: "user-chose" }),
        );
        return true;
      });

      renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
        initialValueOverrides: { "llm.model": "prefilled" },
      });

      const modelInput = await screen.findByTestId("sdk-settings-llm.model");
      await waitFor(() => expect(modelInput).toHaveValue("prefilled"));

      await userEvent.clear(modelInput);
      await userEvent.type(modelInput, "user-chose");

      const callsBefore = getSettingsSpy.mock.calls.length;
      await userEvent.click(screen.getByTestId("save-button"));

      // The save invalidates the settings query, so wait for that confirming
      // read to land before asserting — the rebase alone shows the right value.
      await waitFor(() => {
        expect(screen.getByTestId("save-button")).toBeDisabled();
      });
      await waitFor(() => {
        expect(getSettingsSpy.mock.calls.length).toBeGreaterThan(callsBefore);
      });

      expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
        "user-chose",
      );
    });

    it("survives the refetch that a successful save triggers", async () => {
      // Guard rather than a bug repro: `useSaveSettings` invalidates the
      // settings query on success, so a refetch always follows a save, and the
      // saved value must survive it. This passes on main too — main never
      // reset values either — but it pins the contract the rebase has to keep.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      vi.spyOn(SettingsService, "saveSettings").mockImplementation(async () => {
        // The write lands, so every later read reflects it.
        getSettingsSpy.mockResolvedValue(
          buildTwoFieldSettings({ endpoint: "https://saved.example.com" }),
        );
        return true;
      });

      renderSdkSectionPage({ settingsSources: AGENT_LLM_SOURCE });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://saved.example.com");
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(screen.getByTestId("save-button")).toBeDisabled();
      });
      await waitFor(() => {
        expect(getSettingsSpy.mock.calls.length).toBeGreaterThan(1);
      });
      expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
        "https://saved.example.com",
      );
    });

    it("keeps an edit across a refetch under StrictMode", async () => {
      // StrictMode double-invokes effects and updaters; the overlay must not
      // be seeded or pruned twice.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());

      const { queryClient } = renderSdkSectionPage(
        {
          settingsSources: AGENT_LLM_SOURCE,
          initialValueOverrides: { "llm.model": "prefilled" },
        },
        { strict: true },
      );

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://strict.example.com");

      const callsBefore = getSettingsSpy.mock.calls.length;
      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({ endpoint: "https://server.example.com" }),
      );
      await queryClient.invalidateQueries();
      // Sync on the refetch itself. `llm.model` is already "prefilled" before
      // it lands, so waiting on that value would let the test finish without
      // the refetch ever happening.
      await waitFor(() => {
        expect(getSettingsSpy.mock.calls.length).toBeGreaterThan(callsBefore);
      });
      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "prefilled",
        );
      });
      expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
        "https://strict.example.com",
      );
    });

    it("shows a changed prefill instead of the one seeded before it", async () => {
      // Seeding spread the previous overlay last, so an already-seeded value
      // outranked the new override and a changed prefill never reached the
      // form — while the effect dutifully recorded the new signature.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildTwoFieldSettings(),
      );

      const { rerender } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
        initialValueOverrides: { "llm.model": "first-prefill" },
      });

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "first-prefill",
        );
      });

      rerender(
        React.createElement(SdkSectionPage, {
          settingsSources: AGENT_LLM_SOURCE,
          initialValueOverrides: { "llm.model": "second-prefill" },
        }),
      );

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "second-prefill",
        );
      });
    });

    it("still prunes a dropped schema field even when it is also a prefill", async () => {
      // Sparing every override key from the prune brings back the stranded
      // form: a field the schema stops defining is unreachable, so it must be
      // dropped whether or not the caller also prefilled it.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
        initialValueOverrides: { "llm.model": "prefilled" },
      });

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "prefilled",
        );
      });

      const withoutModel = buildTwoFieldSettings();
      withoutModel.agent_settings_schema!.sections[0].fields =
        withoutModel.agent_settings_schema!.sections[0].fields.filter(
          (f) => f.key !== "llm.model",
        );
      const callsBefore = getSettingsSpy.mock.calls.length;
      getSettingsSpy.mockResolvedValue(withoutModel);
      await queryClient.invalidateQueries();
      await waitFor(() => {
        expect(getSettingsSpy.mock.calls.length).toBeGreaterThan(callsBefore);
      });

      await waitFor(() => {
        expect(
          screen.queryByTestId("sdk-settings-llm.model"),
        ).not.toBeInTheDocument();
      });
      expect(screen.getByTestId("save-button")).toBeDisabled();
      expect(saveSettingsSpy).not.toHaveBeenCalled();
    });

    it("keeps a caller-driven value the schema never defined", async () => {
      // `llm-settings-local-view` prefills `llm.provider_connection_id`, which
      // is deliberately outside the schema, and reads it back off
      // `saveControl.values`. Pruning it on a refetch would drop the link and
      // silently unlink the profile on the next save.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      let latestControl: SdkSectionSaveControl | null = null;

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
        initialValueOverrides: { "llm.provider_connection_id": "conn-123" },
        onSaveControlChange: (control) => {
          latestControl = control;
        },
      });

      await screen.findByTestId("sdk-settings-llm.endpoint");

      const callsBefore = getSettingsSpy.mock.calls.length;
      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({ model: "server-moved-on" }),
      );
      await queryClient.invalidateQueries();
      await waitFor(() => {
        expect(getSettingsSpy.mock.calls.length).toBeGreaterThan(callsBefore);
      });
      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "server-moved-on",
        );
      });

      await waitFor(() => {
        expect(latestControl?.values["llm.provider_connection_id"]).toBe(
          "conn-123",
        );
      });
    });

    it("keeps an edit made while the save was in flight", async () => {
      // A successful save consumes only the edits it actually submitted. A
      // field typed into after Save was pressed was never part of that
      // request, so clearing the whole overlay would discard it — the same
      // class of silent loss this issue is about.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      let release: () => void = () => {};
      const inFlight = new Promise<void>((resolve) => {
        release = resolve;
      });
      vi.spyOn(SettingsService, "saveSettings").mockImplementation(async () => {
        await inFlight;
        getSettingsSpy.mockResolvedValue(
          buildTwoFieldSettings({ endpoint: "https://submitted.example.com" }),
        );
        return true;
      });

      renderSdkSectionPage({ settingsSources: AGENT_LLM_SOURCE });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://submitted.example.com");
      await userEvent.click(screen.getByTestId("save-button"));

      // Typed while the request is still open, so it is not in the payload.
      const modelInput = screen.getByTestId("sdk-settings-llm.model");
      await userEvent.clear(modelInput);
      await userEvent.type(modelInput, "typed-during-save");

      release();

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.endpoint")).toHaveValue(
          "https://submitted.example.com",
        );
      });
      expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
        "typed-during-save",
      );
      expect(screen.getByTestId("save-button")).not.toBeDisabled();
    });

    it("keeps getDirtyPayload dirty-only after a refetch", async () => {
      // A refetch changing an untouched field must not make it look edited.
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue(buildTwoFieldSettings());
      let latestControl: SdkSectionSaveControl | null = null;

      const { queryClient } = renderSdkSectionPage({
        settingsSources: AGENT_LLM_SOURCE,
        onSaveControlChange: (control) => {
          latestControl = control;
        },
      });

      const endpointInput = await screen.findByTestId(
        "sdk-settings-llm.endpoint",
      );
      await userEvent.clear(endpointInput);
      await userEvent.type(endpointInput, "https://typed.example.com");

      getSettingsSpy.mockResolvedValue(
        buildTwoFieldSettings({ model: "gpt-5-from-server" }),
      );
      await queryClient.invalidateQueries();

      await waitFor(() => {
        expect(screen.getByTestId("sdk-settings-llm.model")).toHaveValue(
          "gpt-5-from-server",
        );
      });
      // Asserted inside `waitFor` like the other save-control test: outside a
      // closure the control-flow analysis narrows `latestControl` to `null`.
      await waitFor(() => {
        expect(latestControl?.getDirtyPayload()).toEqual({
          llm: { endpoint: "https://typed.example.com" },
        });
      });
    });
  });
});
