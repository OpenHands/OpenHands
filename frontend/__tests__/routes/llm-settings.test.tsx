import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { organizationService } from "#/api/organization-service/organization-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MOCK_DEFAULT_USER_SETTINGS,
  resetTestHandlersMockSettings,
} from "#/mocks/handlers";
import LlmSettingsScreen, { clientLoader } from "#/routes/llm-settings";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { Organization, OrganizationMember } from "#/types/org";
import { Settings, SettingsValue } from "#/types/settings";

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
  };
}

function buildOrganizationMember(
  overrides: Partial<OrganizationMember> = {},
): OrganizationMember {
  return {
    org_id: "1",
    user_id: "99",
    email: "owner@example.com",
    role: "owner",
    status: "active",
    llm_api_key: "",
    max_iterations: 20,
    llm_model: "",
    llm_base_url: "",
    ...overrides,
  };
}

function buildOrganization(
  overrides: Partial<Organization> = {},
): Organization {
  return {
    id: overrides.id ?? "1",
    name: overrides.name ?? "Example Org",
    contact_name: overrides.contact_name ?? "Example Contact",
    contact_email: overrides.contact_email ?? "contact@example.com",
    conversation_expiration: overrides.conversation_expiration ?? 30,
    remote_runtime_resource_factor:
      overrides.remote_runtime_resource_factor ?? 1,
    billing_margin: overrides.billing_margin ?? 0,
    enable_proactive_conversation_starters:
      overrides.enable_proactive_conversation_starters ?? false,
    sandbox_base_container_image:
      overrides.sandbox_base_container_image ??
      "ghcr.io/all-hands-ai/runtime:latest",
    sandbox_runtime_container_image:
      overrides.sandbox_runtime_container_image ??
      "ghcr.io/all-hands-ai/runtime:latest",
    org_version: overrides.org_version ?? 1,
    search_api_key: overrides.search_api_key ?? null,
    sandbox_api_key: overrides.sandbox_api_key ?? null,
    max_budget_per_task: overrides.max_budget_per_task ?? 0,
    enable_solvability_analysis: overrides.enable_solvability_analysis ?? false,
    v1_enabled: overrides.v1_enabled ?? true,
    credits: overrides.credits ?? 0,
    is_personal: overrides.is_personal,
    ...overrides,
  };
}

function buildSettingsWithAdvancedToggle(
  overrides: Partial<Settings> = {},
): Settings {
  const schema = structuredClone(
    overrides.agent_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
  );
  const llmSection = schema.sections.find((section) => section.key === "llm");

  if (
    llmSection &&
    !llmSection.fields.some((field) => field.key === "llm.timeout")
  ) {
    llmSection.fields.push({
      key: "llm.timeout",
      label: "Timeout",
      section: "llm",
      section_label: "LLM",
      value_type: "integer",
      default: null,
      choices: [],
      depends_on: [],
      prominence: "major",
      secret: false,
      required: false,
    });
  }

  return buildSettings({ ...overrides, agent_settings_schema: schema });
}

async function selectProvider(providerLabel: "OpenHands" | "OpenAI") {
  const providerInput = screen.getByTestId("llm-provider-input");
  await userEvent.click(providerInput);
  await userEvent.click(await screen.findByText(providerLabel));
  await waitFor(() => {
    expect(providerInput).toHaveValue(providerLabel);
  });
  return providerInput;
}

async function selectModel(modelLabel: string) {
  const modelInput = screen.getByTestId("llm-model-input");
  await userEvent.click(modelInput);
  await userEvent.click(await screen.findByText(modelLabel));
  await waitFor(() => {
    expect(modelInput).toHaveValue(modelLabel);
  });
  return modelInput;
}

function renderLlmSettingsScreen({
  appMode = "oss",
  organizationId = "1",
  meData,
  organizations,
  scope = "personal",
}: {
  appMode?: "oss" | "saas";
  organizationId?: string;
  meData?: OrganizationMember;
  organizations?: Organization[];
  scope?: "personal" | "org";
} = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  useSelectedOrganizationStore.setState({ organizationId });
  mockUseConfig.mockReturnValue({
    data: { app_mode: appMode },
    isLoading: false,
  });

  if (appMode === "saas") {
    queryClient.setQueryData(["user", "authenticated", appMode], true);
    queryClient.setQueryData(
      ["organizations", organizationId, "me"],
      meData ?? buildOrganizationMember({ org_id: organizationId }),
    );
    queryClient.setQueryData(["organizations"], {
      items: organizations ?? [
        buildOrganization({ id: organizationId, is_personal: false }),
      ],
      currentOrgId: organizationId,
    });
  }

  return render(<LlmSettingsScreen scope={scope} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
  resetTestHandlersMockSettings();
  mockUseSearchParams.mockReturnValue([{ get: () => null }, vi.fn()]);
  mockUseConfig.mockReturnValue({
    data: { app_mode: "oss" },
    isLoading: false,
  });
  useSelectedOrganizationStore.setState({ organizationId: "1" });
});

describe("LlmSettingsScreen", () => {
  it("renders the schema-driven basic LLM form in OSS mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-screen");
    expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
    expect(screen.getByTestId("llm-provider-input")).toBeInTheDocument();
    expect(screen.getByTestId("llm-model-input")).toBeInTheDocument();
    expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
    expect(screen.getByTestId("save-button")).toBeInTheDocument();
  });

  it("opens advanced view when a custom advanced LLM base URL is already set", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        llm_base_url: "https://custom.example/v1",
        agent_settings: {
          llm: {
            model: "openai/gpt-4o",
            base_url: "https://custom.example/v1",
          },
        },
      }),
    );

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    expect(screen.getByTestId("llm-custom-model-input")).toBeInTheDocument();
    expect(screen.getByTestId("base-url-input")).toBeInTheDocument();
  });

  it("uses schema defaults for custom-rendered advanced fields", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const llmSection = schema.sections.find((section) => section.key === "llm");
    const baseUrlField = llmSection?.fields.find(
      (field) => field.key === "llm.base_url",
    );

    if (!baseUrlField) {
      throw new Error("Expected llm.base_url field in test schema");
    }

    baseUrlField.default = "https://schema.default/v1";
    llmSection?.fields.push({
      key: "llm.timeout",
      label: "Timeout",
      section: "llm",
      section_label: "LLM",
      value_type: "integer",
      default: 30,
      choices: [],
      depends_on: [],
      prominence: "major",
      secret: false,
      required: false,
    });

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_base_url: "",
        agent_settings: {
          llm: {
            model: "openai/gpt-4o",
          },
        },
        agent_settings_schema: schema,
      }),
    );

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

    expect(screen.getByTestId("base-url-input")).toHaveValue(
      "https://schema.default/v1",
    );
  });

  it("does not render the agent field even when the schema includes it", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );

    schema.sections.push({
      key: "general",
      label: "General",
      fields: [
        {
          key: "agent",
          label: "Agent",
          section: "general",
          section_label: "General",
          value_type: "string",
          default: "CodeActAgent",
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
      ],
    });

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        llm_base_url: "https://custom.example/v1",
        agent_settings_schema: schema,
        agent_settings: {
          agent: "BrowsingAgent",
          llm: {
            model: "openai/gpt-4o",
            base_url: "https://custom.example/v1",
          },
        },
      }),
    );

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");

    expect(screen.queryByTestId("agent-input")).not.toBeInTheDocument();
  });

  it("uses the docs.openhands.dev domain for the API key help link", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        agent_settings: {
          llm: {
            model: "openai/gpt-4o",
          },
        },
      }),
    );

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-screen");

    const helpLink = within(
      screen.getByTestId("llm-settings-form-basic"),
    ).getByTestId("llm-api-key-help-anchor");

    expect(helpLink.querySelector("a")).toHaveAttribute(
      "href",
      "https://docs.openhands.dev/usage/local-setup#getting-an-api-key",
    );
  });

  it("defaults to basic view on first visit when org settings use a bare OpenAI model with the default base URL", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const llmSection = schema.sections.find((section) => section.key === "llm");

    if (!llmSection) {
      throw new Error("Expected llm section in test schema");
    }

    llmSection.fields.push({
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
    });

    schema.sections.push({
      key: "general",
      label: "General",
      fields: [
        {
          key: "agent",
          label: "Agent",
          section: "general",
          section_label: "General",
          value_type: "string",
          default: "CodeActAgent",
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
      ],
    });

    vi.spyOn(
      organizationService,
      "getOrganizationAgentSettings",
    ).mockResolvedValue(
      buildSettings({
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        agent_settings_schema: schema,
        agent_settings: {
          agent: "CodeActAgent",
          llm: {
            model: "gpt-4",
            base_url: "https://api.openai.com",
          },
        },
      }),
    );

    renderLlmSettingsScreen({ appMode: "saas", scope: "org" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("sdk-settings-llm.timeout"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("defaults to basic view on first personal SaaS visit even when effective settings include inherited org-only LLM fields", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const llmSection = schema.sections.find((section) => section.key === "llm");

    if (!llmSection) {
      throw new Error("Expected llm section in test schema");
    }

    llmSection.fields.push({
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
    });

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        agent_settings_schema: schema,
        agent_settings: {
          agent: "CodeActAgent",
          llm: {
            model: "gpt-4",
            base_url: "https://api.openai.com",
            timeout: 60,
          },
        },
      }),
    );

    renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("sdk-settings-llm.timeout"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("hides the API key input for OpenHands provider in SaaS mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-screen");
    expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    expect(screen.getByTestId("openhands-api-key-help")).toBeInTheDocument();
  });

  it("shows the API key input for non-OpenHands providers in SaaS mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        agent_settings: { llm: { model: "openai/gpt-4o" } },
      }),
    );

    renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-screen");
    expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
  });

  it("keeps personal settings editable for team members in SaaS mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderLlmSettingsScreen({
      appMode: "saas",
      meData: buildOrganizationMember({ role: "member" }),
    });

    await screen.findByTestId("llm-settings-screen");
    expect(screen.getByTestId("save-button")).toBeInTheDocument();
  });

  describe("Contextual info messages", () => {
    it("should show admin info message for admin user in team organization", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({
        appMode: "saas",
        organizationId: "3",
        meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        organizations: [buildOrganization({ id: "3", is_personal: false })],
      });

      expect(
        await screen.findByTestId("llm-settings-info-message"),
      ).toBeInTheDocument();
    });

    it("should show member info message for member user in team organization", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({
        appMode: "saas",
        organizationId: "2",
        meData: buildOrganizationMember({ org_id: "2", role: "member" }),
        organizations: [buildOrganization({ id: "2", is_personal: false })],
      });

      expect(
        await screen.findByTestId("llm-settings-info-message"),
      ).toBeInTheDocument();
    });

    it("shows the personal info message for personal workspace in SaaS mode", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({
        appMode: "saas",
        organizationId: "1",
        meData: buildOrganizationMember({ org_id: "1", role: "owner" }),
        organizations: [buildOrganization({ id: "1", is_personal: true })],
      });

      expect(
        await screen.findByTestId("llm-settings-info-message"),
      ).toBeInTheDocument();
    });

    it("should not show info message in OSS mode", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({ appMode: "oss" });

      await screen.findByTestId("llm-settings-screen");
      expect(
        screen.queryByTestId("llm-settings-info-message"),
      ).not.toBeInTheDocument();
    });
  });

  it("submits basic form values through SDK setting keys", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        agent_settings: { llm: { model: "openai/gpt-4o" } },
      }),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderLlmSettingsScreen({ appMode: "oss" });

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({ api_key: "test-api-key" }),
          }),
        }),
      );
    });
  });

  it("resets hidden advanced and all settings back to defaults when saving basic view", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const llmSection = schema.sections.find((section) => section.key === "llm");

    if (!llmSection) {
      throw new Error("Expected llm section in test schema");
    }

    const baseUrlField = llmSection.fields.find(
      (field) => field.key === "llm.base_url",
    );
    if (!baseUrlField) {
      throw new Error("Expected llm.base_url field in test schema");
    }
    baseUrlField.default = "https://schema.default/v1";

    llmSection.fields.push({
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
    });

    schema.sections.push({
      key: "general",
      label: "General",
      fields: [
        {
          key: "agent",
          label: "Agent",
          section: "general",
          section_label: "General",
          value_type: "string",
          default: "CodeActAgent",
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
      ],
    });

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        llm_base_url: "https://custom.example/v1",
        agent_settings_schema: schema,
        agent_settings: {
          agent: "BrowsingAgent",
          llm: {
            model: "openai/gpt-4o",
            base_url: "https://custom.example/v1",
            timeout: 90,
          },
        },
      }),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    await userEvent.click(screen.getByTestId("sdk-section-basic-toggle"));

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({
              api_key: "test-api-key",
              base_url: "https://schema.default/v1",
              timeout: 30,
            }),
          }),
        }),
      );
    });

    const payload = saveSettingsSpy.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(
      (payload.agent_settings as Record<string, unknown>) ?? {},
    ).not.toHaveProperty("agent");
  });

  it("preserves existing MCP settings when saving the LLM page", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const existingMcpConfig = {
      mcpServers: {
        tavily: {
          transport: "http",
          url: "https://example.com/mcp",
        },
      },
    };

    schema.sections.push({
      key: "general",
      label: "General",
      fields: [
        {
          key: "agent",
          label: "Agent",
          section: "general",
          section_label: "General",
          value_type: "string",
          default: "CodeActAgent",
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
        {
          key: "mcp_config",
          label: "MCP Config",
          section: "general",
          section_label: "General",
          value_type: "object",
          default: null,
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: false,
        },
      ],
    });

    let persistedSettings = buildSettingsWithAdvancedToggle({
      llm_model: "openai/gpt-4o",
      agent_settings_schema: schema,
      agent_settings: {
        agent: "BrowsingAgent",
        llm: {
          model: "openai/gpt-4o",
        },
        mcp_config: existingMcpConfig,
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        const payloadAgentSettings = (payload.agent_settings ?? {}) as Record<
          string,
          unknown
        >;

        const nextAgentSettings: NonNullable<Settings["agent_settings"]> = {
          ...(persistedSettings.agent_settings ?? {}),
          ...(payloadAgentSettings as Record<string, SettingsValue>),
          llm: {
            ...((persistedSettings.agent_settings?.llm as Record<
              string,
              SettingsValue
            >) ?? {}),
            ...((payloadAgentSettings.llm as Record<string, SettingsValue>) ?? {}),
          },
        };

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          agent_settings_schema: schema,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledTimes(1);
    });

    const payload = saveSettingsSpy.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(
      (payload.agent_settings as Record<string, unknown>) ?? {},
    ).not.toHaveProperty("mcp_config");

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    expect(persistedSettings.agent_settings?.mcp_config).toEqual(
      existingMcpConfig,
    );
  });

  it("does not include search API key updates when saving basic LLM settings", async () => {
    let persistedSettings = buildSettingsWithAdvancedToggle({
      llm_model: "openai/gpt-4o",
      search_api_key: "tavily-key",
      search_api_key_set: true,
      agent_settings: {
        llm: {
          model: "openai/gpt-4o",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        const nextAgentSettings = {
          ...persistedSettings.agent_settings,
        } as NonNullable<Settings["agent_settings"]>;

        Object.entries(payload).forEach(([key, value]) => {
          if (key.includes(".") || key === "agent" || key === "mcp_config") {
            nextAgentSettings[key] = value as SettingsValue;
          }
        });

        persistedSettings = buildSettings({
          ...persistedSettings,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({ api_key: "test-api-key" }),
          }),
        }),
      );
    });

    const payload = saveSettingsSpy.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(payload).not.toHaveProperty("search_api_key");

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps the basic view after save on SaaS personal settings when an inherited org search API key remains set on refetch", async () => {
    let persistedSettings = buildSettingsWithAdvancedToggle({
      llm_model: "openai/gpt-4o",
      search_api_key_set: true,
      agent_settings: {
        llm: {
          model: "openai/gpt-4o",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        const nextAgentSettings = {
          ...persistedSettings.agent_settings,
        } as NonNullable<Settings["agent_settings"]>;

        Object.entries(payload).forEach(([key, value]) => {
          if (key.includes(".") || key === "agent" || key === "mcp_config") {
            nextAgentSettings[key] = value as SettingsValue;
          }
        });

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          search_api_key: "",
          search_api_key_set: true,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-form-basic");

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({ api_key: "test-api-key" }),
          }),
        }),
      );
    });

    const payload = saveSettingsSpy.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(payload).not.toHaveProperty("search_api_key");

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("does not clear the hidden search API key on SaaS org settings when saving basic view", async () => {
    let persistedSettings = buildSettingsWithAdvancedToggle({
      llm_model: "openai/gpt-4o",
      llm_base_url: "https://custom.example/v1",
      search_api_key: "****1234",
      agent_settings: {
        llm: {
          model: "openai/gpt-4o",
          base_url: "https://custom.example/v1",
        },
      },
    });

    const getOrganizationSettingsSpy = vi
      .spyOn(organizationService, "getOrganizationAgentSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveOrganizationSettingsSpy = vi
      .spyOn(organizationService, "saveOrganizationAgentSettings")
      .mockImplementation(async (payload) => {
        const nextAgentSettings = {
          ...persistedSettings.agent_settings,
        } as NonNullable<Settings["agent_settings"]>;

        Object.entries(payload).forEach(([key, value]) => {
          if (key.includes(".") || key === "agent" || key === "mcp_config") {
            nextAgentSettings[key] = value as SettingsValue;
          }
        });

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          llm_base_url: "",
          search_api_key: "****1234",
          agent_settings: nextAgentSettings,
        });

        return persistedSettings;
      });

    renderLlmSettingsScreen({ appMode: "saas", scope: "org" });

    await screen.findByTestId("llm-settings-form-advanced");
    await userEvent.click(screen.getByTestId("sdk-section-basic-toggle"));

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveOrganizationSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({
              api_key: "test-api-key",
              base_url: null,
            }),
          }),
        }),
      );
    });

    const payload = saveOrganizationSettingsSpy.mock.calls[0]?.at(0) as Record<
      string,
      unknown
    >;
    expect(payload).not.toHaveProperty("search_api_key");

    await waitFor(() => {
      expect(getOrganizationSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps the basic view after save when a stale legacy base URL lingers on refetch", async () => {
    let persistedSettings = buildSettingsWithAdvancedToggle({
      llm_base_url: "https://stale.example/v1",
      agent_settings: {
        llm: {
          base_url: "https://stale.example/v1",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        const nextAgentSettings = {
          ...persistedSettings.agent_settings,
        } as NonNullable<Settings["agent_settings"]>;

        Object.entries(payload).forEach(([key, value]) => {
          if (key.includes(".") || key === "agent" || key === "mcp_config") {
            nextAgentSettings[key] = value as SettingsValue;
          }
        });

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    await userEvent.click(screen.getByTestId("sdk-section-basic-toggle"));

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({
              api_key: "test-api-key",
              base_url: null,
            }),
          }),
        }),
      );
    });

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps the basic view after saving a basic model change when refetch includes a provider base URL", async () => {
    let persistedSettings = buildSettingsWithAdvancedToggle();

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        const nextAgentSettings = {
          ...persistedSettings.agent_settings,
        } as NonNullable<Settings["agent_settings"]>;

        Object.entries(payload).forEach(([key, value]) => {
          if (key.includes(".") || key === "agent" || key === "mcp_config") {
            nextAgentSettings[key] = value as SettingsValue;
          }
        });

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          llm_model: "openai/gpt-4o",
          llm_base_url: "https://api.openai.com/v1",
          agent_settings: {
            llm: {
              model: "openai/gpt-4o",
              base_url: "https://api.openai.com/v1",
            },
          },
        });

        return true;
      });

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    await selectProvider("OpenAI");
    await selectModel("gpt-4o");
    await userEvent.type(
      await screen.findByTestId("llm-api-key-input"),
      "test-api-key",
    );
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({
              model: "openai/gpt-4o",
              api_key: "test-api-key",
            }),
          }),
        }),
      );
    });

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("does not render the search API key input in advanced LLM settings", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithAdvancedToggle(),
    );

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

    await waitFor(() => {
      expect(
        screen.getByTestId("llm-settings-form-advanced"),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("search-api-key-input"),
      ).not.toBeInTheDocument();
    });
  });

  it("does not reveal all-only fields after save when refetch includes an MCP-owned search API key", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const llmSection = schema.sections.find((section) => section.key === "llm");

    if (!llmSection) {
      throw new Error("Expected llm section in test schema");
    }

    llmSection.fields.push({
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
    });

    schema.sections.push({
      key: "general",
      label: "General",
      fields: [
        {
          key: "agent",
          label: "Agent",
          section: "general",
          section_label: "General",
          value_type: "string",
          default: "CodeActAgent",
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
      ],
    });

    let persistedSettings = buildSettings({
      agent_settings_schema: schema,
      search_api_key: "",
      search_api_key_set: false,
      agent_settings: {
        llm: {
          model: "openhands/claude-opus-4-5-20251101",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    vi.spyOn(SettingsService, "saveSettings").mockImplementation(
      async (payload) => {
        persistedSettings = buildSettings({
          agent_settings_schema: schema,
          search_api_key: "tavily-key",
          search_api_key_set: true,
          agent_settings: {
            llm: {
              model: "openhands/claude-opus-4-5-20251101",
            },
          },
        });

        expect(payload).not.toHaveProperty("search_api_key");
        return true;
      },
    );

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    await userEvent.click(screen.getByTestId("sdk-section-all-toggle"));
    expect(screen.getByTestId("sdk-settings-llm.timeout")).toBeInTheDocument();

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
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

  it("does not reveal all-only fields after save when refetch returns a litellm_proxy model with the managed proxy base URL", async () => {
    const schema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const llmSection = schema.sections.find((section) => section.key === "llm");

    if (!llmSection) {
      throw new Error("Expected llm section in test schema");
    }

    llmSection.fields.push({
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
    });

    let persistedSettings = buildSettings({
      agent_settings_schema: schema,
      agent_settings: {
        llm: {
          model: "openhands/claude-opus-4-5-20251101",
        },
      },
    });

    const getSettingsSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    vi.spyOn(SettingsService, "saveSettings").mockImplementation(async () => {
      persistedSettings = buildSettings({
        agent_settings_schema: schema,
        agent_settings: {
          llm: {
            model: "litellm_proxy/claude-opus-4-5-20251101",
            base_url: "https://llm-proxy.app.all-hands.dev",
          },
        },
      });

      return true;
    });

    renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("sdk-settings-llm.timeout"),
    ).not.toBeInTheDocument();

    await userEvent.type(
      await screen.findByTestId("llm-api-key-input"),
      "test-api-key",
    );
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

  it("submits advanced form values through SDK setting keys", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        agent_settings: {
          llm: {
            model: "openai/gpt-4o",
            base_url: "https://custom.example/v1",
          },
        },
      }),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderLlmSettingsScreen({ appMode: "oss" });

    const baseUrlInput = await screen.findByTestId("base-url-input");
    await userEvent.type(baseUrlInput, "/extra");

    await waitFor(() => {
      expect(baseUrlInput).toHaveValue("https://custom.example/v1/extra");
      expect(screen.getByTestId("save-button")).not.toBeDisabled();
    });

    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings: expect.objectContaining({
            llm: expect.objectContaining({
              base_url: "https://custom.example/v1/extra",
            }),
          }),
        }),
      );
    });
  });

  describe("API key visibility in Basic Settings", () => {
    it("should hide API key input when SaaS mode is enabled and OpenHands provider is selected", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({ appMode: "saas" });
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const providerInput = within(basicForm).getByTestId("llm-provider-input");

      await waitFor(() => {
        expect(providerInput).toHaveValue("OpenHands");
      });

      expect(
        within(basicForm).queryByTestId("llm-api-key-input"),
      ).not.toBeInTheDocument();
      expect(
        within(basicForm).queryByTestId("llm-api-key-help-anchor"),
      ).not.toBeInTheDocument();
    });

    it("should show API key input when SaaS mode is enabled and non-OpenHands provider is selected", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      renderLlmSettingsScreen({ appMode: "saas" });
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const providerInput = within(basicForm).getByTestId("llm-provider-input");

      await waitFor(() => {
        expect(providerInput).toHaveValue("OpenAI");
      });

      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });

    it("should show API key input when OSS mode is enabled and OpenHands provider is selected", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({ appMode: "oss" });
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const providerInput = within(basicForm).getByTestId("llm-provider-input");

      await waitFor(() => {
        expect(providerInput).toHaveValue("OpenHands");
      });

      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });

    it("should show API key input when OSS mode is enabled and non-OpenHands provider is selected", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      renderLlmSettingsScreen({ appMode: "oss" });
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const providerInput = within(basicForm).getByTestId("llm-provider-input");

      await waitFor(() => {
        expect(providerInput).toHaveValue("OpenAI");
      });

      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });

    it("should hide API key input when switching from non-OpenHands to OpenHands provider in SaaS mode", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      renderLlmSettingsScreen({ appMode: "saas" });
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      await waitFor(() => {
        expect(
          within(basicForm).getByTestId("llm-api-key-input"),
        ).toBeInTheDocument();
      });

      await selectProvider("OpenHands");

      expect(
        within(basicForm).queryByTestId("llm-api-key-input"),
      ).not.toBeInTheDocument();
      expect(
        within(basicForm).queryByTestId("llm-api-key-help-anchor"),
      ).not.toBeInTheDocument();
    });

    it("should show API key input when switching from OpenHands to non-OpenHands provider in SaaS mode", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      renderLlmSettingsScreen({ appMode: "saas" });
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      expect(
        within(basicForm).queryByTestId("llm-api-key-input"),
      ).not.toBeInTheDocument();

      await selectProvider("OpenAI");

      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });
  });

  describe("Role-based permissions", () => {
    describe("Member role (personal overrides allowed)", () => {
      it("should keep all input fields enabled in basic view", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "2",
          meData: buildOrganizationMember({ org_id: "2", role: "member" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const basicForm = screen.getByTestId("llm-settings-form-basic");
        const providerInput =
          within(basicForm).getByTestId("llm-provider-input");
        const modelInput = within(basicForm).getByTestId("llm-model-input");
        const apiKeyInput = within(basicForm).getByTestId("llm-api-key-input");

        await waitFor(() => {
          expect(providerInput).toBeEnabled();
          expect(modelInput).toBeEnabled();
          expect(apiKeyInput).toBeEnabled();
        });
      });

      it("should render the submit button", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings(),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "2",
          meData: buildOrganizationMember({ org_id: "2", role: "member" }),
        });

        await screen.findByTestId("llm-settings-screen");
        expect(screen.getByTestId("save-button")).toBeInTheDocument();
      });

      it("should keep the advanced/basic toggle enabled for members", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettingsWithAdvancedToggle(),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "2",
          meData: buildOrganizationMember({ org_id: "2", role: "member" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const basicToggle = screen.getByTestId("sdk-section-basic-toggle");
        const advancedToggle = screen.getByTestId(
          "sdk-section-advanced-toggle",
        );

        expect(basicToggle).toBeEnabled();
        expect(advancedToggle).toBeEnabled();
        expect(
          screen.getByTestId("llm-settings-form-basic"),
        ).toBeInTheDocument();
      });
    });

    describe("Owner role (full access)", () => {
      it("should enable all input fields in basic view", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "1",
          meData: buildOrganizationMember({ org_id: "1", role: "owner" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const basicForm = screen.getByTestId("llm-settings-form-basic");
        const providerInput =
          within(basicForm).getByTestId("llm-provider-input");
        const modelInput = within(basicForm).getByTestId("llm-model-input");
        const apiKeyInput = within(basicForm).getByTestId("llm-api-key-input");

        await waitFor(() => {
          expect(providerInput).not.toBeDisabled();
          expect(modelInput).not.toBeDisabled();
          expect(apiKeyInput).not.toBeDisabled();
        });
      });

      it("should enable all input fields in advanced view", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettingsWithAdvancedToggle({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "1",
          meData: buildOrganizationMember({ org_id: "1", role: "owner" }),
        });

        await screen.findByTestId("llm-settings-screen");
        await userEvent.click(
          screen.getByTestId("sdk-section-advanced-toggle"),
        );

        const advancedForm = screen.getByTestId("llm-settings-form-advanced");
        const customModelInput = within(advancedForm).getByTestId(
          "llm-custom-model-input",
        );
        const baseUrlInput = within(advancedForm).getByTestId("base-url-input");
        const apiKeyInput =
          within(advancedForm).getByTestId("llm-api-key-input");

        await waitFor(() => {
          expect(customModelInput).not.toBeDisabled();
          expect(baseUrlInput).not.toBeDisabled();
          expect(apiKeyInput).not.toBeDisabled();
        });
      });

      it("should enable submit button when form is dirty", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "1",
          meData: buildOrganizationMember({ org_id: "1", role: "owner" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const submitButton = screen.getByTestId("save-button");
        expect(submitButton).toBeDisabled();

        await userEvent.type(
          screen.getByTestId("llm-api-key-input"),
          "test-api-key",
        );

        await waitFor(() => {
          expect(submitButton).not.toBeDisabled();
        });
      });

      it("should allow submitting form changes", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );
        const saveSettingsSpy = vi
          .spyOn(SettingsService, "saveSettings")
          .mockResolvedValue(true);

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "1",
          meData: buildOrganizationMember({ org_id: "1", role: "owner" }),
        });

        await screen.findByTestId("llm-settings-screen");
        await userEvent.type(
          screen.getByTestId("llm-api-key-input"),
          "test-api-key",
        );
        await userEvent.click(screen.getByTestId("save-button"));

        await waitFor(() => {
          expect(saveSettingsSpy).toHaveBeenCalled();
        });
      });
    });

    describe("Admin role (full access)", () => {
      it("should enable all input fields in basic view", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "3",
          meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const basicForm = screen.getByTestId("llm-settings-form-basic");
        const providerInput =
          within(basicForm).getByTestId("llm-provider-input");
        const modelInput = within(basicForm).getByTestId("llm-model-input");
        const apiKeyInput = within(basicForm).getByTestId("llm-api-key-input");

        await waitFor(() => {
          expect(providerInput).not.toBeDisabled();
          expect(modelInput).not.toBeDisabled();
          expect(apiKeyInput).not.toBeDisabled();
        });
      });

      it("should enable all input fields in advanced view", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettingsWithAdvancedToggle({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "3",
          meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        });

        await screen.findByTestId("llm-settings-screen");
        await userEvent.click(
          screen.getByTestId("sdk-section-advanced-toggle"),
        );

        const advancedForm = screen.getByTestId("llm-settings-form-advanced");
        const customModelInput = within(advancedForm).getByTestId(
          "llm-custom-model-input",
        );
        const baseUrlInput = within(advancedForm).getByTestId("base-url-input");
        const apiKeyInput =
          within(advancedForm).getByTestId("llm-api-key-input");

        await waitFor(() => {
          expect(customModelInput).not.toBeDisabled();
          expect(baseUrlInput).not.toBeDisabled();
          expect(apiKeyInput).not.toBeDisabled();
        });
      });

      it("should enable submit button when form is dirty", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "3",
          meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const submitButton = screen.getByTestId("save-button");
        expect(submitButton).toBeDisabled();

        await userEvent.type(
          screen.getByTestId("llm-api-key-input"),
          "test-api-key",
        );

        await waitFor(() => {
          expect(submitButton).not.toBeDisabled();
        });
      });

      it("should allow submitting form changes", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );
        const saveSettingsSpy = vi
          .spyOn(SettingsService, "saveSettings")
          .mockResolvedValue(true);

        renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "3",
          meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        });

        await screen.findByTestId("llm-settings-screen");
        await userEvent.type(
          screen.getByTestId("llm-api-key-input"),
          "test-api-key",
        );
        await userEvent.click(screen.getByTestId("save-button"));

        await waitFor(() => {
          expect(saveSettingsSpy).toHaveBeenCalled();
        });
      });
    });

    describe("clientLoader permission checks", () => {
      it("should export a clientLoader for route protection", () => {
        expect(clientLoader).toBeTypeOf("function");
      });
    });
  });
});
