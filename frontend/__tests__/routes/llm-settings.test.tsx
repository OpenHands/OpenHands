import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { http, HttpResponse } from "msw";

import OrgProfilesService from "#/api/organization-service/org-profiles-service.api";
import { organizationService } from "#/api/organization-service/organization-service.api";
import ProfilesService, {
  LlmProfileSummary,
} from "#/api/settings-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MOCK_DEFAULT_USER_SETTINGS,
  resetTestHandlersMockSettings,
} from "#/mocks/handlers";
import { server } from "#/mocks/node";
import LlmSettingsScreen, { clientLoader } from "#/routes/llm-settings";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { Organization, OrganizationMember } from "#/types/org";
import { Settings, SettingsValue } from "#/types/settings";

// ProfilesService is mocked because the LLM screen's auto-profile flow
// calls save + activate after every successful settings save. The default
// resolved values are reapplied in beforeEach — the existing suite-wide
// ``vi.restoreAllMocks()`` would otherwise wipe them between tests.
vi.mock("#/api/settings-service/profiles-service.api", () => ({
  default: {
    listProfiles: vi.fn(),
    saveProfile: vi.fn(),
    deleteProfile: vi.fn(),
    activateProfile: vi.fn(),
    renameProfile: vi.fn(),
  },
}));

vi.mock("#/api/organization-service/org-profiles-service.api", () => ({
  default: {
    listProfiles: vi.fn(),
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
    deleteProfile: vi.fn(),
    activateProfile: vi.fn(),
    renameProfile: vi.fn(),
  },
}));

function resetProfilesServiceDefaults() {
  vi.mocked(ProfilesService.listProfiles)
    .mockReset()
    .mockResolvedValue({ profiles: [], active_profile: null });
  vi.mocked(ProfilesService.saveProfile)
    .mockReset()
    .mockResolvedValue(undefined);
  vi.mocked(ProfilesService.deleteProfile)
    .mockReset()
    .mockResolvedValue(undefined);
  vi.mocked(ProfilesService.activateProfile)
    .mockReset()
    .mockResolvedValue(undefined);
  vi.mocked(ProfilesService.renameProfile)
    .mockReset()
    .mockResolvedValue(undefined);
}

function resetOrgProfilesServiceDefaults() {
  vi.mocked(OrgProfilesService.listProfiles)
    .mockReset()
    .mockResolvedValue({ profiles: [], active_profile: null });
  vi.mocked(OrgProfilesService.getProfile)
    .mockReset()
    .mockResolvedValue({
      name: "openai_gpt-4o",
      llm: { model: "openai/gpt-4o" },
    });
  vi.mocked(OrgProfilesService.saveProfile)
    .mockReset()
    .mockResolvedValue(undefined);
  vi.mocked(OrgProfilesService.deleteProfile)
    .mockReset()
    .mockResolvedValue(undefined);
  vi.mocked(OrgProfilesService.activateProfile)
    .mockReset()
    .mockResolvedValue(undefined);
  vi.mocked(OrgProfilesService.renameProfile)
    .mockReset()
    .mockResolvedValue(undefined);
}

// Stub the profile mutation hooks so auto-save doesn't invalidate the
// settings query — tests in this file pin exact getSettings call counts.
// The mutateAsync implementations forward to the service mock so the
// auto-profile tests can still assert what was called.
vi.mock("#/hooks/mutation/use-save-llm-profile", () => ({
  useSaveLlmProfile: () => ({
    mutateAsync: (vars: { name: string; request?: unknown }) =>
      ProfilesService.saveProfile(vars.name, vars.request as never),
    isPending: false,
  }),
}));
vi.mock("#/hooks/mutation/use-activate-llm-profile", () => ({
  useActivateLlmProfile: () => ({
    mutateAsync: (name: string) => ProfilesService.activateProfile(name),
    isPending: false,
  }),
}));

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

function getPayloadAgentSettings(
  payload: Record<string, unknown>,
): Record<string, unknown> {
  return (payload.agent_settings_diff as Record<string, unknown>) ?? {};
}

// Read the mocked settings through the spy's implementation directly so the
// helper can mirror them into the seeded profile without bumping the
// getSettings call counts that several tests pin.
async function readMockedActiveSettings(
  scope: "personal" | "org",
): Promise<Settings | null> {
  const settingsFn =
    scope === "org"
      ? organizationService.getOrganizationSettings
      : SettingsService.getSettings;
  if (!vi.isMockFunction(settingsFn)) return null;
  const impl = settingsFn.getMockImplementation();
  if (!impl) return null;
  try {
    return (
      ((await (impl as (orgId?: string) => Promise<unknown>)(
        "1",
      )) as Settings) ?? null
    );
  } catch {
    return null;
  }
}

async function renderLlmSettingsScreen({
  appMode = "oss",
  organizationId = "1",
  meData,
  organizations,
  scope = "personal",
  view = "form",
  profile,
  featureFlags,
}: {
  appMode?: "oss" | "saas";
  organizationId?: string;
  meData?: OrganizationMember;
  organizations?: Organization[];
  scope?: "personal" | "org";
  // Profile-enabled scopes land on the Available Models list by default; set
  // ``view`` to ``"form"`` (the default) to auto-click into the SDK form via
  // Edit on a seeded active profile. The seeded profile mirrors the mocked
  // settings, so the (now profile-hydrated) edit form keeps showing the same
  // values as before and existing form-oriented assertions keep working.
  // ``"create"`` clicks Add Profile instead (blank create form), and
  // ``"profiles"`` tests the list view itself.
  view?: "form" | "create" | "profiles";
  // Override fields on the seeded edit profile when a test needs it to
  // *differ* from the active settings.
  profile?: Partial<LlmProfileSummary>;
  // Web-client feature flags merged into the mocked /config payload, e.g.
  // ``{ allow_user_llm_configuration: false }`` for managed OHE installs.
  featureFlags?: Record<string, unknown>;
} = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  useSelectedOrganizationStore.setState({ organizationId });
  mockUseConfig.mockReturnValue({
    data: {
      app_mode: appMode,
      ...(featureFlags ? { feature_flags: featureFlags } : {}),
    },
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

  if (view === "form") {
    // One active profile to Edit into, mirroring the mocked settings so
    // the profile-hydrated edit form shows the same values the
    // settings-hydrated form used to. The name stays fixed so auto-profile
    // assertions that expect a derived name keep working.
    const activeSettings = await readMockedActiveSettings(scope);
    const activeLlm = (activeSettings?.agent_settings?.llm ?? {}) as Record<
      string,
      unknown
    >;
    const seededProfile: LlmProfileSummary = {
      name: "openai_gpt-4o",
      model:
        ((activeLlm.model as string | undefined) ??
          activeSettings?.llm_model) ||
        null,
      base_url:
        ((activeLlm.base_url as string | undefined) ??
          activeSettings?.llm_base_url) ||
        null,
      api_key_set: activeSettings?.llm_api_key_set ?? false,
      ...profile,
    };
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [seededProfile],
      active_profile: seededProfile.name,
    });
    vi.mocked(OrgProfilesService.listProfiles).mockResolvedValue({
      profiles: [seededProfile],
      active_profile: seededProfile.name,
    });
  }

  const rendered = render(<LlmSettingsScreen scope={scope} />, {
    wrapper: ({ children }) => (
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      </MemoryRouter>
    ),
  });

  if (view === "form") {
    // Edit the seeded active profile: Add Profile now opens a blank create
    // form, while these tests exercise the form against persisted settings.
    await userEvent.click(await screen.findByTestId("profile-menu-trigger"));
    await userEvent.click(await screen.findByTestId("profile-edit"));
  } else if (view === "create") {
    await userEvent.click(await screen.findByTestId("add-llm-profile"));
  }

  return rendered;
}

beforeEach(() => {
  vi.restoreAllMocks();
  resetProfilesServiceDefaults();
  resetOrgProfilesServiceDefaults();
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

    await renderLlmSettingsScreen({ appMode: "oss" });

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

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    expect(screen.getByTestId("llm-custom-model-input")).toBeInTheDocument();
    expect(screen.getByTestId("base-url-input")).toBeInTheDocument();
  });

  it("defaults to basic view when an OpenHands managed model has no base URL", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithAdvancedToggle({
        llm_model: "openhands/claude-opus-4-5-20251101",
        llm_base_url: "",
        agent_settings: {
          llm: {
            model: "openhands/claude-opus-4-5-20251101",
          },
        },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("opens basic view when an OpenHands managed model has a server-filled base URL", async () => {
    // Managed openhands/* profiles always come back with a base_url — the
    // backend fills it with the managed LiteLLM endpoint on save. That must
    // not be mistaken for a user customization that opens advanced/all.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithAdvancedToggle({
        llm_model: "openhands/claude-opus-4-5-20251101",
        llm_base_url: "http://openhands-litellm:4000",
        agent_settings: {
          llm: {
            model: "openhands/claude-opus-4-5-20251101",
            base_url: "http://openhands-litellm:4000",
          },
        },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("opens basic view when editing an org-scope OpenHands managed profile with a server-filled base URL", async () => {
    // Org scope is not covered by the personal-SaaS first-mount-basic rule,
    // so this exercises the base-URL inference path directly.
    vi.spyOn(organizationService, "getOrganizationSettings").mockResolvedValue(
      buildSettingsWithAdvancedToggle({
        llm_model: "openhands/claude-opus-4-5-20251101",
        llm_base_url: "http://openhands-litellm:4000",
        agent_settings: {
          llm: {
            model: "openhands/claude-opus-4-5-20251101",
            base_url: "http://openhands-litellm:4000",
          },
        },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "saas", scope: "org" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("opens basic view even when an OpenHands model has a non-managed base URL", async () => {
    // Accepted edge case: the server owns base_url for openhands/* models,
    // so editing always starts on basic — even if the stored URL was set
    // deliberately in advanced mode. The value is preserved and the
    // advanced view remains one click away.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithAdvancedToggle({
        llm_model: "openhands/claude-opus-4-5-20251101",
        llm_base_url: "https://custom.example/v1",
        agent_settings: {
          llm: {
            model: "openhands/claude-opus-4-5-20251101",
            base_url: "https://custom.example/v1",
          },
        },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("treats a litellm_proxy model with the managed proxy URL as an explicit custom endpoint", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithAdvancedToggle({
        llm_model: "litellm_proxy/claude-opus-4-5-20251101",
        llm_base_url: "https://llm-proxy.app.all-hands.dev",
        agent_settings: {
          llm: {
            model: "litellm_proxy/claude-opus-4-5-20251101",
            base_url: "https://llm-proxy.app.all-hands.dev",
          },
        },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    expect(screen.getByTestId("llm-custom-model-input")).toHaveValue(
      "litellm_proxy/claude-opus-4-5-20251101",
    );
    expect(
      screen.queryByTestId("openhands-api-key-help-2"),
    ).not.toBeInTheDocument();
  });

  it("shows Advanced and All toggles in OSS mode for the default LLM route schema", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-screen");
    expect(
      screen.getByTestId("sdk-section-advanced-toggle"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("sdk-section-all-toggle")).toBeInTheDocument();
  });

  it("keeps Advanced visible but hides All in SaaS mode for the default LLM route schema", async () => {
    vi.spyOn(organizationService, "getOrganizationSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          llm: {
            model: "openai/gpt-4o",
          },
        },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "saas", scope: "org" });

    await screen.findByTestId("llm-settings-screen");
    expect(
      screen.getByTestId("sdk-section-advanced-toggle"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-section-all-toggle"),
    ).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

    expect(
      screen.getByTestId("llm-settings-form-advanced"),
    ).toBeInTheDocument();
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

    await renderLlmSettingsScreen({ appMode: "oss" });

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

    await renderLlmSettingsScreen({ appMode: "oss" });

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

    await renderLlmSettingsScreen({ appMode: "oss" });

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

    vi.spyOn(organizationService, "getOrganizationSettings").mockResolvedValue(
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

    await renderLlmSettingsScreen({ appMode: "saas", scope: "org" });

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

    await renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-form-basic");
    expect(
      screen.queryByTestId("sdk-settings-llm.timeout"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("llm-settings-form-advanced"),
    ).not.toBeInTheDocument();
  });

  it("hides the API key input for OpenHands provider on enterprise cloud", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    await renderLlmSettingsScreen({
      appMode: "saas",
      featureFlags: { deployment_mode: "cloud" },
    });

    await screen.findByTestId("llm-settings-screen");
    expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    expect(screen.getByTestId("openhands-api-key-help")).toBeInTheDocument();
  });

  it("hides the OpenHands API key help on self-hosted OHE", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    await renderLlmSettingsScreen({
      appMode: "saas",
      featureFlags: { deployment_mode: "self_hosted" },
    });

    await screen.findByTestId("llm-settings-screen");
    expect(
      screen.queryByTestId("openhands-api-key-help"),
    ).not.toBeInTheDocument();
  });

  it("shows the API key input for non-OpenHands providers in SaaS mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_model: "openai/gpt-4o",
        agent_settings: { llm: { model: "openai/gpt-4o" } },
      }),
    );

    await renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-screen");
    expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
  });

  it("keeps personal settings editable for team members in SaaS mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    await renderLlmSettingsScreen({
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

      await renderLlmSettingsScreen({
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

      await renderLlmSettingsScreen({
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

      await renderLlmSettingsScreen({
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

      await renderLlmSettingsScreen({ appMode: "oss" });

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

    await renderLlmSettingsScreen({ appMode: "oss" });

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({ api_key: "test-api-key" }),
          }),
        }),
      );
    });
  });

  it("preserves the existing base URL when saving basic view without changing the model", async () => {
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

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    await userEvent.click(screen.getByTestId("sdk-section-basic-toggle"));

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              api_key: "test-api-key",
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
    const llmPayload = getPayloadAgentSettings(payload).llm as Record<
      string,
      unknown
    >;
    // Edit-save now persists the profile-hydrated base_url explicitly
    // (instead of omitting it), preserving the existing custom value.
    expect(llmPayload.base_url).toBe("https://custom.example/v1");
    expect(getPayloadAgentSettings(payload)).not.toHaveProperty("agent");
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
        const payloadAgentSettings = getPayloadAgentSettings(payload);

        const nextAgentSettings: NonNullable<Settings["agent_settings"]> = {
          ...(persistedSettings.agent_settings ?? {}),
          ...(payloadAgentSettings as Record<string, SettingsValue>),
          llm: {
            ...((persistedSettings.agent_settings?.llm as Record<
              string,
              SettingsValue
            >) ?? {}),
            ...((payloadAgentSettings.llm as Record<string, SettingsValue>) ??
              {}),
          },
        };

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          agent_settings_schema: schema,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    await renderLlmSettingsScreen({ appMode: "oss" });

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
    expect(getPayloadAgentSettings(payload)).not.toHaveProperty("mcp_config");

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

        const payloadAgentSettings = getPayloadAgentSettings(payload);
        Object.assign(
          nextAgentSettings,
          payloadAgentSettings as Record<string, SettingsValue>,
        );
        nextAgentSettings.llm = {
          ...((persistedSettings.agent_settings?.llm as Record<
            string,
            SettingsValue
          >) ?? {}),
          ...((payloadAgentSettings.llm as Record<string, SettingsValue>) ??
            {}),
        };

        persistedSettings = buildSettings({
          ...persistedSettings,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-basic");

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
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

    // Personal scope flips to the Available Models list after a successful
    // save (``handleSaveSuccess`` → ``setShowProfiles(true)``).
    await waitFor(() => {
      expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-basic"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("returns to the profiles list after save on SaaS personal settings even when an inherited org search API key remains set on refetch", async () => {
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

        const payloadAgentSettings = getPayloadAgentSettings(payload);
        Object.assign(
          nextAgentSettings,
          payloadAgentSettings as Record<string, SettingsValue>,
        );
        nextAgentSettings.llm = {
          ...((persistedSettings.agent_settings?.llm as Record<
            string,
            SettingsValue
          >) ?? {}),
          ...((payloadAgentSettings.llm as Record<string, SettingsValue>) ??
            {}),
        };

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          search_api_key: "",
          search_api_key_set: true,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    await renderLlmSettingsScreen({ appMode: "saas" });

    await screen.findByTestId("llm-settings-form-basic");

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
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

    // After save on personal SaaS, the screen returns to the Available
    // Models list — not the advanced form — even though the refetched
    // settings still carry an inherited search_api_key flag.
    await waitFor(() => {
      expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });

    // Re-entering the form via Add Profile must land on basic, not get
    // bumped into advanced by the lingering search_api_key on refetch.
    await userEvent.click(screen.getByTestId("add-llm-profile"));
    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("does not clear hidden search API key or existing base URL on SaaS org settings when saving basic view", async () => {
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
      .spyOn(organizationService, "getOrganizationSettings")
      .mockImplementation(async () => structuredClone(persistedSettings));
    const saveOrganizationSettingsSpy = vi
      .spyOn(organizationService, "saveOrganizationSettings")
      .mockImplementation(async ({ settings }) => {
        const nextAgentSettings = {
          ...persistedSettings.agent_settings,
        } as NonNullable<Settings["agent_settings"]>;

        const agentSettingsDiff = settings.agent_settings_diff as
          | Settings["agent_settings"]
          | undefined;
        if (agentSettingsDiff) {
          Object.assign(nextAgentSettings, agentSettingsDiff);
        }

        Object.entries(settings).forEach(([key, value]) => {
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

    await renderLlmSettingsScreen({ appMode: "saas", scope: "org" });

    await screen.findByTestId("llm-settings-form-advanced");
    await userEvent.click(screen.getByTestId("sdk-section-basic-toggle"));

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveOrganizationSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          settings: expect.objectContaining({
            agent_settings_diff: expect.objectContaining({
              llm: expect.objectContaining({
                api_key: "test-api-key",
              }),
            }),
          }),
        }),
      );
    });

    const payload = saveOrganizationSettingsSpy.mock.calls[0]?.at(0) as {
      settings: Record<string, unknown>;
    };
    const llmPayload = (
      payload.settings.agent_settings_diff as Record<string, unknown>
    ).llm as Record<string, unknown>;
    // Edit-save now persists the profile-hydrated base_url explicitly
    // (instead of omitting it), preserving the existing custom value.
    expect(llmPayload.base_url).toBe("https://custom.example/v1");
    expect(payload.settings).not.toHaveProperty("search_api_key");

    await waitFor(() => {
      expect(getOrganizationSettingsSpy).toHaveBeenCalledTimes(3);
    });

    await waitFor(() => {
      expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-basic"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });

    await userEvent.click(screen.getByTestId("add-llm-profile"));
    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps an existing custom base URL when saving basic view without a model change", async () => {
    // Use a non-managed model: openhands/* models treat any base_url as
    // server-owned, which would keep this scenario from opening advanced.
    let persistedSettings = buildSettingsWithAdvancedToggle({
      llm_model: "openai/gpt-4o",
      llm_base_url: "https://stale.example/v1",
      agent_settings: {
        llm: {
          model: "openai/gpt-4o",
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

        const payloadAgentSettings = getPayloadAgentSettings(payload);
        Object.assign(
          nextAgentSettings,
          payloadAgentSettings as Record<string, SettingsValue>,
        );
        nextAgentSettings.llm = {
          ...((persistedSettings.agent_settings?.llm as Record<
            string,
            SettingsValue
          >) ?? {}),
          ...((payloadAgentSettings.llm as Record<string, SettingsValue>) ??
            {}),
        };

        persistedSettings = buildSettingsWithAdvancedToggle({
          ...persistedSettings,
          agent_settings: nextAgentSettings,
        });

        return true;
      });

    await renderLlmSettingsScreen({ appMode: "oss" });

    await screen.findByTestId("llm-settings-form-advanced");
    await userEvent.click(screen.getByTestId("sdk-section-basic-toggle"));

    const apiKeyInput = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKeyInput, "test-api-key");
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              api_key: "test-api-key",
            }),
          }),
        }),
      );
    });

    const payload = saveSettingsSpy.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    const llmPayload = getPayloadAgentSettings(payload).llm as Record<
      string,
      unknown
    >;
    // Edit-save now persists the profile-hydrated base_url explicitly
    // (instead of omitting it), preserving the existing custom value.
    expect(llmPayload.base_url).toBe("https://stale.example/v1");

    await waitFor(() => {
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });

    // Personal scope returns to Available Models after save.
    await waitFor(() => {
      expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });

    // The persisted settings still carry the custom base_url instead of
    // being silently cleared during the previous save.
    expect(
      (persistedSettings.agent_settings?.llm as Record<string, SettingsValue>)
        ?.base_url,
    ).toBe("https://stale.example/v1");

    // Re-entering via Add Profile starts a blank create form on the basic
    // view regardless of stored values; the preserved base_url shows when
    // editing the existing configuration instead.
    await userEvent.click(screen.getByTestId("add-llm-profile"));
    await waitFor(() => {
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });
  });

  it("returns to the profiles list after saving a basic model change and re-enters the form in basic view even when refetch includes a provider base URL", async () => {
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

        const payloadAgentSettings = getPayloadAgentSettings(payload);
        Object.assign(
          nextAgentSettings,
          payloadAgentSettings as Record<string, SettingsValue>,
        );
        nextAgentSettings.llm = {
          ...((persistedSettings.agent_settings?.llm as Record<
            string,
            SettingsValue
          >) ?? {}),
          ...((payloadAgentSettings.llm as Record<string, SettingsValue>) ??
            {}),
        };

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

    await renderLlmSettingsScreen({ appMode: "oss" });

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
          agent_settings_diff: expect.objectContaining({
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

    // Personal scope returns to Available Models after save.
    await waitFor(() => {
      expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });

    // Re-entering the form must land on basic — the provider-default
    // base_url that came back on refetch must not pop us into advanced.
    await userEvent.click(screen.getByTestId("add-llm-profile"));
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

    await renderLlmSettingsScreen({ appMode: "oss" });

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

    await renderLlmSettingsScreen({ appMode: "oss" });

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

  it("does not reveal all-only fields after save when refetch returns an OpenHands managed model", async () => {
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
            model: "openhands/claude-opus-4-5-20251101",
          },
        },
      });

      return true;
    });

    await renderLlmSettingsScreen({ appMode: "oss" });

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

    await renderLlmSettingsScreen({ appMode: "oss" });

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
          agent_settings_diff: expect.objectContaining({
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

      await renderLlmSettingsScreen({ appMode: "saas" });
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

      await renderLlmSettingsScreen({ appMode: "saas" });
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

      await renderLlmSettingsScreen({ appMode: "oss" });
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

      await renderLlmSettingsScreen({ appMode: "oss" });
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

      await renderLlmSettingsScreen({ appMode: "saas" });
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

      await renderLlmSettingsScreen({ appMode: "saas" });
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
    describe("Org default profiles", () => {
      it("shows org profiles and management controls for admins", async () => {
        vi.mocked(OrgProfilesService.listProfiles).mockResolvedValue({
          profiles: [
            {
              name: "sonnet",
              model: "openhands/claude-sonnet-4-5-20250929",
              base_url: null,
              api_key_set: true,
            },
          ],
          active_profile: "sonnet",
        });

        await renderLlmSettingsScreen({
          appMode: "saas",
          scope: "org",
          organizationId: "3",
          meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
          view: "profiles",
        });

        expect(await screen.findByText("sonnet")).toBeInTheDocument();
        expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
        expect(screen.getByTestId("profile-menu-trigger")).toBeInTheDocument();
      });

      it("shows org profiles without management controls for members", async () => {
        vi.mocked(OrgProfilesService.listProfiles).mockResolvedValue({
          profiles: [
            {
              name: "sonnet",
              model: "openhands/claude-sonnet-4-5-20250929",
              base_url: null,
              api_key_set: true,
            },
          ],
          active_profile: "sonnet",
        });

        await renderLlmSettingsScreen({
          appMode: "saas",
          scope: "org",
          organizationId: "2",
          meData: buildOrganizationMember({ org_id: "2", role: "member" }),
          view: "profiles",
        });

        expect(await screen.findByText("sonnet")).toBeInTheDocument();
        expect(screen.queryByTestId("add-llm-profile")).not.toBeInTheDocument();
        expect(
          screen.queryByTestId("profile-menu-trigger"),
        ).not.toBeInTheDocument();
      });
    });

    describe("Member role (personal overrides allowed)", () => {
      it("should keep all input fields enabled in basic view", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        await renderLlmSettingsScreen({
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

        await renderLlmSettingsScreen({
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

        await renderLlmSettingsScreen({
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

        await renderLlmSettingsScreen({
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

        await renderLlmSettingsScreen({
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

      it("keeps the submit button enabled in the profile form even when pristine", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        await renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "1",
          meData: buildOrganizationMember({ org_id: "1", role: "owner" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const submitButton = screen.getByTestId("save-button");
        // The profile form snapshots the current config as a profile (the
        // name is optional — it falls back to a model-derived default), so
        // Save is available without first dirtying a field. This is what lets
        // you save a profile in SaaS managed mode, where the model is fixed
        // and there's no editable API key to make the form dirty.
        expect(submitButton).not.toBeDisabled();

        // Editing a field keeps it enabled.
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

        await renderLlmSettingsScreen({
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

        await renderLlmSettingsScreen({
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

        await renderLlmSettingsScreen({
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

      it("keeps the submit button enabled in the profile form even when pristine", async () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          buildSettings({
            llm_model: "openai/gpt-4o",
            agent_settings: { llm: { model: "openai/gpt-4o" } },
          }),
        );

        await renderLlmSettingsScreen({
          appMode: "saas",
          organizationId: "3",
          meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        });

        await screen.findByTestId("llm-settings-screen");
        const submitButton = screen.getByTestId("save-button");
        // The profile form snapshots the current config as a profile (the
        // name is optional — it falls back to a model-derived default), so
        // Save is available without first dirtying a field. This is what lets
        // you save a profile in SaaS managed mode, where the model is fixed
        // and there's no editable API key to make the form dirty.
        expect(submitButton).not.toBeDisabled();

        // Editing a field keeps it enabled.
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

        await renderLlmSettingsScreen({
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

  // ── Auto-profile flow ───────────────────────────────────────────────
  //
  // After a successful LLM settings save the screen should snapshot the
  // just-saved agent_settings.llm into a profile named after the model
  // and activate it — that's how Profiles tab populates in the first
  // place. These tests pin that chain.

  describe("auto-profile on save", () => {
    it("saves + activates a profile named after the model after a personal-scope save", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);

      await renderLlmSettingsScreen({ appMode: "oss" });

      await userEvent.type(
        await screen.findByTestId("llm-api-key-input"),
        "test-api-key",
      );
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
          "openai_gpt-4o",
          { include_secrets: true, preserve_existing_api_key: false },
        );
      });
      await waitFor(() => {
        expect(ProfilesService.activateProfile).toHaveBeenCalledWith(
          "openai_gpt-4o",
        );
      });
    });

    it("asks the backend to preserve the stored profile key when no key is typed", async () => {
      // A save with a blank key field must not snapshot the active settings'
      // api_key into the profile — that silently swaps the profile's own key.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);

      await renderLlmSettingsScreen({ appMode: "oss" });

      await screen.findByTestId("llm-api-key-input");
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
          "openai_gpt-4o",
          { include_secrets: true, preserve_existing_api_key: true },
        );
      });
    });

    it("asks the backend to preserve the stored org profile key when no key is typed", async () => {
      vi.spyOn(
        organizationService,
        "getOrganizationSettings",
      ).mockResolvedValue(
        buildSettings({
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(
        organizationService,
        "saveOrganizationSettings",
      ).mockResolvedValue({
        agent_settings: {},
        conversation_settings: {},
        search_api_key: undefined,
        llm_api_key_set: false,
      });

      await renderLlmSettingsScreen({
        appMode: "saas",
        scope: "org",
        organizationId: "3",
        meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
      });

      const orgProfileNameInput = await screen.findByTestId(
        "llm-profile-name-input",
      );
      await userEvent.clear(orgProfileNameInput);
      await userEvent.type(orgProfileNameInput, "team-profile");
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(OrgProfilesService.saveProfile).toHaveBeenCalledWith(
          "3",
          "team-profile",
          { include_secrets: true, preserve_existing_api_key: true },
        );
      });
    });

    it("saves + activates an org profile on the org-default settings screen", async () => {
      vi.spyOn(
        organizationService,
        "getOrganizationSettings",
      ).mockResolvedValue(
        buildSettings({
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(
        organizationService,
        "saveOrganizationSettings",
      ).mockResolvedValue({
        agent_settings: {},
        conversation_settings: {},
        search_api_key: undefined,
        llm_api_key_set: false,
      });

      await renderLlmSettingsScreen({
        appMode: "saas",
        scope: "org",
        organizationId: "3",
        meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
      });

      const orgProfileNameInput = await screen.findByTestId(
        "llm-profile-name-input",
      );
      await userEvent.clear(orgProfileNameInput);
      await userEvent.type(orgProfileNameInput, "team-profile");
      await userEvent.type(
        await screen.findByTestId("llm-api-key-input"),
        "test-api-key",
      );
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(organizationService.saveOrganizationSettings).toHaveBeenCalled();
      });
      await waitFor(() => {
        expect(OrgProfilesService.saveProfile).toHaveBeenCalledWith(
          "3",
          "team-profile",
          { include_secrets: true, preserve_existing_api_key: false },
        );
      });
      await waitFor(() => {
        expect(OrgProfilesService.activateProfile).toHaveBeenCalledWith(
          "3",
          "team-profile",
        );
      });
      expect(ProfilesService.saveProfile).not.toHaveBeenCalled();
      expect(ProfilesService.activateProfile).not.toHaveBeenCalled();
    });

    it("uses the user-typed profile name instead of the model-derived default", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);

      await renderLlmSettingsScreen({ appMode: "oss" });

      const profileNameInput = await screen.findByTestId(
        "llm-profile-name-input",
      );
      await userEvent.clear(profileNameInput);
      await userEvent.type(profileNameInput, "my-custom-name");
      await userEvent.type(
        await screen.findByTestId("llm-api-key-input"),
        "test-api-key",
      );
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
          "my-custom-name",
          { include_secrets: true, preserve_existing_api_key: false },
        );
      });
      await waitFor(() => {
        expect(ProfilesService.activateProfile).toHaveBeenCalledWith(
          "my-custom-name",
        );
      });
    });

    it("falls back to the derived name when the user-typed name fails the regex", async () => {
      // "has space" is invalid (PROFILE_NAME_PATTERN forbids whitespace).
      // The helper text turns red but save proceeds with the derived name —
      // we don't want a settings save to silently succeed while the profile
      // step blows up server-side with a 422.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);

      await renderLlmSettingsScreen({ appMode: "oss" });

      const profileNameInput = await screen.findByTestId(
        "llm-profile-name-input",
      );
      await userEvent.clear(profileNameInput);
      await userEvent.type(profileNameInput, "has space");
      await userEvent.type(
        await screen.findByTestId("llm-api-key-input"),
        "test-api-key",
      );
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
          "openai_gpt-4o",
          { include_secrets: true, preserve_existing_api_key: false },
        );
      });
    });

    it("renders the profile-name input on the org-default profile form for admins", async () => {
      vi.spyOn(
        organizationService,
        "getOrganizationSettings",
      ).mockResolvedValue(
        buildSettings({
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      await renderLlmSettingsScreen({
        appMode: "saas",
        scope: "org",
        organizationId: "3",
        meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
      });

      await screen.findByTestId("llm-api-key-input");
      expect(screen.getByTestId("llm-profile-name-input")).toBeInTheDocument();
    });

    it("swallows profile-save failures so the user still sees the settings-saved toast", async () => {
      // If the profiles endpoint is down (e.g. hit the MAX_PROFILES_PER_USER
      // cap), the settings save itself must still be treated as succeeded.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
      vi.mocked(ProfilesService.saveProfile).mockRejectedValueOnce(
        new Error("409 conflict"),
      );

      await renderLlmSettingsScreen({ appMode: "oss" });

      await userEvent.type(
        await screen.findByTestId("llm-api-key-input"),
        "test-api-key",
      );
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalled();
      });
      // Activate must NOT run when save already failed — otherwise we'd
      // activate a profile that doesn't exist on the backend.
      expect(ProfilesService.activateProfile).not.toHaveBeenCalled();
    });
  });

  // ── Add Profile (create flow) ───────────────────────────────────────
  //
  // Adding a new profile starts from a clean slate: the form opens on the
  // basic view with blank fields instead of inheriting the active/org
  // settings (which previously dropped users with a custom model or base
  // URL straight into a pre-filled advanced form).

  describe("Add Profile (create flow)", () => {
    const settingsWithCustomModelAndBaseUrl = () =>
      buildSettings({
        llm_model: "litellm_proxy/custom-model",
        llm_base_url: "https://custom.example/v1",
        agent_settings: {
          llm: {
            model: "litellm_proxy/custom-model",
            base_url: "https://custom.example/v1",
          },
        },
      });

    it("opens the create form on basic view even when the active settings have a custom model and base URL", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        settingsWithCustomModelAndBaseUrl(),
      );

      await renderLlmSettingsScreen({ appMode: "oss", view: "create" });

      await screen.findByTestId("llm-settings-form-basic");
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });

    it("starts the create form blank instead of pre-filling from the active settings", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        settingsWithCustomModelAndBaseUrl(),
      );

      await renderLlmSettingsScreen({ appMode: "oss", view: "create" });

      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.getByTestId("llm-provider-input")).toHaveValue("");
      expect(screen.getByTestId("llm-model-input")).toHaveValue("");
      // No provider selected yet, so the API key input isn't rendered (it
      // appears once a key-taking provider is chosen).
      expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();

      await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

      expect(screen.getByTestId("llm-custom-model-input")).toHaveValue("");
      expect(screen.getByTestId("base-url-input")).toHaveValue("");
      expect(screen.getByTestId("llm-api-key-input")).toHaveValue("");
    });

    it("does not preselect the active provider when creating a profile in SaaS mode", async () => {
      // Default settings use an OpenHands model; previously the create form
      // inherited the provider selection.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      await renderLlmSettingsScreen({ appMode: "saas", view: "create" });

      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.getByTestId("llm-provider-input")).toHaveValue("");
    });

    it("only shows the API key input once a key-taking provider is selected", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      await renderLlmSettingsScreen({ appMode: "saas", view: "create" });

      // Blank create form: no provider selected, so no API key input —
      // rendering it only to remove it again when a managed provider is
      // picked was jarring.
      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();

      // Picking a key-taking provider reveals the input.
      await selectProvider("OpenAI");
      expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();

      // The managed OpenHands provider keeps it hidden in SaaS mode (keys
      // are auto-provisioned).
      await selectProvider("OpenHands");
      expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    });

    it("keeps Save disabled in the create form until a model is chosen", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      await renderLlmSettingsScreen({ appMode: "oss", view: "create" });

      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.getByTestId("save-button")).toBeDisabled();

      // Picking a provider alone isn't a model yet.
      await selectProvider("OpenAI");
      expect(screen.getByTestId("save-button")).toBeDisabled();

      await selectModel("gpt-4o");
      await waitFor(() => {
        expect(screen.getByTestId("save-button")).not.toBeDisabled();
      });
    });

    it("saves and activates a profile built from the blank create form", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openhands/claude-opus-4-5-20251101",
          agent_settings: {
            llm: { model: "openhands/claude-opus-4-5-20251101" },
          },
        }),
      );
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      await renderLlmSettingsScreen({ appMode: "oss", view: "create" });

      await screen.findByTestId("llm-settings-form-basic");
      await selectProvider("OpenAI");
      await selectModel("gpt-4o");
      await userEvent.type(
        screen.getByTestId("llm-api-key-input"),
        "new-api-key",
      );
      await userEvent.click(screen.getByTestId("save-button"));

      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            agent_settings_diff: expect.objectContaining({
              llm: expect.objectContaining({
                model: "openai/gpt-4o",
                api_key: "new-api-key",
              }),
            }),
          }),
        );
      });
      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
          "openai_gpt-4o",
          { include_secrets: true, preserve_existing_api_key: false },
        );
      });
      await waitFor(() => {
        expect(ProfilesService.activateProfile).toHaveBeenCalledWith(
          "openai_gpt-4o",
        );
      });
    });

    it("opens a blank org create form on basic view for admins even when org defaults have custom fields", async () => {
      vi.spyOn(
        organizationService,
        "getOrganizationSettings",
      ).mockResolvedValue(settingsWithCustomModelAndBaseUrl());

      await renderLlmSettingsScreen({
        appMode: "saas",
        scope: "org",
        organizationId: "3",
        meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        view: "create",
      });

      await screen.findByTestId("llm-settings-form-basic");
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();

      await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

      expect(screen.getByTestId("llm-custom-model-input")).toHaveValue("");
      expect(screen.getByTestId("base-url-input")).toHaveValue("");
    });

    it("hydrates the edit form from a non-active profile instead of the active settings", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openhands/claude-opus-4-5-20251101",
          agent_settings: {
            llm: { model: "openhands/claude-opus-4-5-20251101" },
          },
        }),
      );
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);
      vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
        profiles: [
          {
            name: "active-profile",
            model: "openhands/claude-opus-4-5-20251101",
            base_url: null,
            api_key_set: false,
          },
          {
            name: "other-profile",
            model: "openai/gpt-4o-mini",
            base_url: null,
            api_key_set: false,
          },
        ],
        active_profile: "active-profile",
      });

      await renderLlmSettingsScreen({ appMode: "oss", view: "profiles" });

      const triggers = await screen.findAllByTestId("profile-menu-trigger");
      await userEvent.click(triggers[1]);
      await userEvent.click(await screen.findByTestId("profile-edit"));

      await screen.findByTestId("llm-settings-form-basic");
      await waitFor(() => {
        expect(screen.getByTestId("llm-provider-input")).toHaveValue("OpenAI");
        expect(screen.getByTestId("llm-model-input")).toHaveValue(
          "gpt-4o-mini",
        );
      });

      // Saving without changes must persist the edited profile's values
      // instead of snapshotting the active settings over it.
      await userEvent.click(screen.getByTestId("save-button"));
      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            agent_settings_diff: expect.objectContaining({
              llm: expect.objectContaining({ model: "openai/gpt-4o-mini" }),
            }),
          }),
        );
      });
      await waitFor(() => {
        expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
          "other-profile",
          // No key typed on this pristine edit-save, so the profile's
          // stored key must survive the snapshot.
          { include_secrets: true, preserve_existing_api_key: true },
        );
      });
    });

    it("hydrates the edit form with the canonical model when the profile stores a hidden alias", async () => {
      // Legacy profile model "openhands/custom-llm" is a hidden alias that
      // the catalog maps to the visible claude-sonnet model.
      server.use(
        http.get("/api/v1/config/models/search", () =>
          HttpResponse.json({
            items: [
              {
                provider: "openhands",
                name: "claude-sonnet-4-5-20250929",
                verified: true,
              },
              {
                provider: "openhands",
                name: "custom-llm",
                verified: false,
                hidden: true,
                canonical: "claude-sonnet-4-5-20250929",
              },
            ],
            next_page_id: null,
          }),
        ),
      );
      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      await renderLlmSettingsScreen({
        appMode: "oss",
        profile: {
          name: "legacy-profile",
          model: "openhands/custom-llm",
          base_url: null,
        },
      });

      await screen.findByTestId("llm-settings-form-basic");
      await waitFor(() => {
        expect(screen.getByTestId("llm-provider-input")).toHaveValue(
          "OpenHands",
        );
        expect(screen.getByTestId("llm-model-input")).toHaveValue(
          "claude-sonnet-4-5-20250929",
        );
      });
      expect(
        screen.queryByTestId("model-unavailable-warning"),
      ).not.toBeInTheDocument();

      // A pristine save persists the canonical id — deliberately migrating
      // the profile off the legacy alias (both names route on the proxy).
      await userEvent.click(screen.getByTestId("save-button"));
      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            agent_settings_diff: expect.objectContaining({
              llm: expect.objectContaining({
                model: "openhands/claude-sonnet-4-5-20250929",
              }),
            }),
          }),
        );
      });
    });

    it("keeps the stored alias when a hidden model has no canonical mapping", async () => {
      server.use(
        http.get("/api/v1/config/models/search", () =>
          HttpResponse.json({
            items: [
              {
                provider: "openhands",
                name: "claude-sonnet-4-5-20250929",
                verified: true,
              },
              // Hidden but with no canonical mapping — nothing to translate.
              {
                provider: "openhands",
                name: "orphan-alias",
                verified: false,
                hidden: true,
              },
            ],
            next_page_id: null,
          }),
        ),
      );

      await renderLlmSettingsScreen({
        appMode: "oss",
        profile: {
          name: "orphan-profile",
          model: "openhands/orphan-alias",
          base_url: null,
        },
      });

      await screen.findByTestId("llm-settings-form-basic");
      await waitFor(() => {
        expect(screen.getByTestId("llm-provider-input")).toHaveValue(
          "OpenHands",
        );
      });
      // Unchanged today-behavior: hidden aliases are never dropdown options,
      // so the selection renders empty — and no unavailable badge, since the
      // proxy still serves the alias.
      expect(screen.getByTestId("llm-model-input")).toHaveValue("");
      expect(
        screen.queryByTestId("model-unavailable-warning"),
      ).not.toBeInTheDocument();
    });

    it("opens edit on advanced with the profile's custom base URL while the active settings are plain", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          llm_model: "openhands/claude-opus-4-5-20251101",
          agent_settings: {
            llm: { model: "openhands/claude-opus-4-5-20251101" },
          },
        }),
      );

      await renderLlmSettingsScreen({
        appMode: "oss",
        profile: {
          name: "proxy-profile",
          model: "litellm_proxy/custom-model",
          base_url: "https://proxy.example/v1",
          api_key_set: true,
        },
      });

      await screen.findByTestId("llm-settings-form-advanced");
      expect(screen.getByTestId("llm-custom-model-input")).toHaveValue(
        "litellm_proxy/custom-model",
      );
      expect(screen.getByTestId("base-url-input")).toHaveValue(
        "https://proxy.example/v1",
      );
      // The set-but-hidden key indicator mirrors the profile, not the
      // active settings (whose llm_api_key_set is false here).
      expect(screen.getByTestId("llm-api-key-input")).toHaveAttribute(
        "placeholder",
        "<hidden>",
      );
    });

    it("opens edit on basic for a plain profile while the active settings have a custom base URL", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        settingsWithCustomModelAndBaseUrl(),
      );

      await renderLlmSettingsScreen({
        appMode: "oss",
        profile: {
          name: "plain-profile",
          model: "openai/gpt-4o",
          base_url: null,
        },
      });

      await screen.findByTestId("llm-settings-form-basic");
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
      await waitFor(() => {
        expect(screen.getByTestId("llm-provider-input")).toHaveValue("OpenAI");
        expect(screen.getByTestId("llm-model-input")).toHaveValue("gpt-4o");
      });
    });

    it("hydrates the org edit form from a non-active org profile", async () => {
      vi.spyOn(
        organizationService,
        "getOrganizationSettings",
      ).mockResolvedValue(
        buildSettings({
          llm_model: "openhands/claude-opus-4-5-20251101",
          agent_settings: {
            llm: { model: "openhands/claude-opus-4-5-20251101" },
          },
        }),
      );
      vi.mocked(OrgProfilesService.listProfiles).mockResolvedValue({
        profiles: [
          {
            name: "org-active",
            model: "openhands/claude-opus-4-5-20251101",
            base_url: null,
            api_key_set: false,
          },
          {
            name: "org-other",
            model: "openai/gpt-4o-mini",
            base_url: null,
            api_key_set: false,
          },
        ],
        active_profile: "org-active",
      });

      await renderLlmSettingsScreen({
        appMode: "saas",
        scope: "org",
        organizationId: "3",
        meData: buildOrganizationMember({ org_id: "3", role: "admin" }),
        view: "profiles",
      });

      const triggers = await screen.findAllByTestId("profile-menu-trigger");
      await userEvent.click(triggers[1]);
      await userEvent.click(await screen.findByTestId("profile-edit"));

      await screen.findByTestId("llm-settings-form-basic");
      await waitFor(() => {
        expect(screen.getByTestId("llm-provider-input")).toHaveValue("OpenAI");
        expect(screen.getByTestId("llm-model-input")).toHaveValue(
          "gpt-4o-mini",
        );
      });
    });

    it("still opens the edit form on advanced view with stored values for a profile with custom fields", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        settingsWithCustomModelAndBaseUrl(),
      );

      await renderLlmSettingsScreen({ appMode: "oss", view: "form" });

      await screen.findByTestId("llm-settings-form-advanced");
      expect(screen.getByTestId("llm-custom-model-input")).toHaveValue(
        "litellm_proxy/custom-model",
      );
      expect(screen.getByTestId("base-url-input")).toHaveValue(
        "https://custom.example/v1",
      );
      expect(screen.getByTestId("llm-profile-name-input")).toHaveValue(
        "openai_gpt-4o",
      );
    });
  });

  describe("managed LLM configuration gating (allow_user_llm_configuration=false)", () => {
    const MANAGED_FLAGS = { allow_user_llm_configuration: false };

    it("hides BYOK inputs and lands on the model dropdown even when a custom base URL is saved", async () => {
      // A saved custom base URL normally opens the advanced form with the
      // custom-model + base-URL inputs. With BYOK disallowed the user lands
      // on the managed dropdown instead — the saved settings themselves are
      // untouched and keep working at runtime.
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

      await renderLlmSettingsScreen({
        appMode: "oss",
        featureFlags: MANAGED_FLAGS,
      });

      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.getByTestId("llm-provider-input")).toBeInTheDocument();
      expect(screen.getByTestId("llm-model-input")).toBeInTheDocument();
      expect(screen.getByTestId("save-button")).toBeInTheDocument();

      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-custom-model-input"),
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("base-url-input")).not.toBeInTheDocument();
      expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
      // No BYOK and no schema-driven advanced fields → no Advanced toggle.
      expect(
        screen.queryByTestId("sdk-section-advanced-toggle"),
      ).not.toBeInTheDocument();
    });

    it("suppresses the Advanced toggle when BYOK is off, even with schema-driven advanced fields", async () => {
      // Without BYOK the LLM Advanced view (custom model + base URL) has
      // nothing to show, so Basic and Advanced would be identical. The
      // Advanced toggle is hidden entirely — even when the schema carries
      // advanced fields that would otherwise surface it.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettingsWithAdvancedToggle({
          llm_model: "openai/gpt-4o",
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      await renderLlmSettingsScreen({
        appMode: "oss",
        featureFlags: MANAGED_FLAGS,
      });

      await screen.findByTestId("llm-settings-form-basic");

      expect(
        screen.queryByTestId("sdk-section-advanced-toggle"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-custom-model-input"),
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("base-url-input")).not.toBeInTheDocument();
      expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    });

    it("hides BYOK inputs on the org-defaults form in SaaS mode", async () => {
      vi.spyOn(
        organizationService,
        "getOrganizationSettings",
      ).mockResolvedValue(
        buildSettings({
          agent_settings: { llm: { model: "openai/gpt-4o" } },
        }),
      );

      await renderLlmSettingsScreen({
        appMode: "saas",
        scope: "org",
        featureFlags: MANAGED_FLAGS,
      });

      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.getByTestId("llm-provider-input")).toBeInTheDocument();
      expect(screen.getByTestId("llm-model-input")).toBeInTheDocument();
      expect(
        screen.queryByTestId("sdk-section-advanced-toggle"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-custom-model-input"),
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("base-url-input")).not.toBeInTheDocument();
      expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();
    });

    it("keeps BYOK inputs when the flag is explicitly true", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings(),
      );

      await renderLlmSettingsScreen({
        appMode: "oss",
        featureFlags: { allow_user_llm_configuration: true },
      });

      await screen.findByTestId("llm-settings-form-basic");
      expect(screen.getByTestId("llm-api-key-input")).toBeInTheDocument();
      expect(
        screen.getByTestId("sdk-section-advanced-toggle"),
      ).toBeInTheDocument();

      await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

      expect(
        screen.getByTestId("llm-settings-form-advanced"),
      ).toBeInTheDocument();
      expect(screen.getByTestId("llm-custom-model-input")).toBeInTheDocument();
      expect(screen.getByTestId("base-url-input")).toBeInTheDocument();
    });
  });
});
