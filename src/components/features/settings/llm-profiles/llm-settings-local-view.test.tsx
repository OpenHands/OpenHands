import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Settings, SettingsSchema } from "#/types/settings";
import { LlmSettingsLocalView } from "./llm-settings-local-view";

const PROFILE_NAME = "my-profile";
const PROFILE_MODEL = "gpt-4o";
// The encrypted token the backend returns for a saved key: much longer than a
// real key. It must never be rendered back into the password field.
const ENCRYPTED_KEY = "sk-encrypted-very-long-token-that-should-never-render";

const llmSchema: SettingsSchema = {
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
          default: "gpt-4o",
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: true,
        },
        {
          key: "llm.api_key",
          label: "API Key",
          section: "llm",
          section_label: "LLM",
          value_type: "string",
          default: null,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: true,
          required: false,
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
          prominence: "critical",
          secret: false,
          required: false,
        },
      ],
    },
  ],
};

const baseSettings: Settings = {
  llm_model: "",
  llm_base_url: "",
  agent: "default",
  language: "en",
  llm_api_key: null,
  llm_api_key_set: false,
  search_api_key_set: false,
  confirmation_mode: false,
  security_analyzer: null,
  max_iterations: null,
  remote_runtime_resource_factor: null,
  provider_tokens_set: {},
  enable_default_condenser: false,
  condenser_max_size: null,
  enable_sound_notifications: false,
  enable_proactive_conversation_starters: false,
  enable_solvability_analysis: false,
  user_consents_to_analytics: null,
  max_budget_per_task: null,
  agent_settings_schema: llmSchema,
  agent_settings: {
    llm: {
      model: PROFILE_MODEL,
      api_key: null,
      base_url: null,
    },
  },
  conversation_settings_schema: null,
  conversation_settings: {},
};

const { profilesServiceMock, hooksMock, toastMock } = vi.hoisted(() => ({
  profilesServiceMock: {
    listProfiles: vi.fn(),
    getProfile: vi.fn(),
    renameProfile: vi.fn(),
    saveProfile: vi.fn(),
  },
  hooksMock: {
    profilesData: {
      profiles: [] as Array<{ name: string; model?: string | null }>,
      active_profile: null as string | null,
    },
    settingsData: null as Settings | null,
    schemaData: null as SettingsSchema | null,
    conversationSchemaData: null as SettingsSchema | null,
    setHideSectionHeader: vi.fn(),
  },
  toastMock: {
    displayErrorToast: vi.fn(),
    displaySuccessToast: vi.fn(),
  },
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: profilesServiceMock,
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => ({
    data: hooksMock.profilesData,
    isLoading: false,
    error: null,
  }),
}));

vi.mock("#/hooks/mutation/use-save-llm-profile", () => ({
  useSaveLlmProfile: () => ({
    mutateAsync: profilesServiceMock.saveProfile,
    isPending: false,
  }),
}));

vi.mock("#/hooks/mutation/use-activate-llm-profile", () => ({
  useActivateLlmProfile: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: hooksMock.settingsData,
    isLoading: false,
    isFetching: false,
  }),
}));

vi.mock("#/hooks/query/use-agent-settings-schema", () => ({
  useAgentSettingsSchema: () => ({
    data: hooksMock.schemaData,
    error: null,
    isLoading: false,
    isFetching: false,
  }),
  useConversationSettingsSchema: () => ({
    data: hooksMock.conversationSchemaData,
    error: null,
    isLoading: false,
    isFetching: false,
  }),
}));

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("#/hooks/query/use-llm-subscription-models", () => ({
  useOpenAISubscriptionModels: () => ({
    data: [],
    isLoading: false,
    isFetching: false,
  }),
}));

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({ data: [] }),
}));

vi.mock("#/hooks/query/use-provider-models", () => ({
  useProviderModels: () => ({ data: [], isLoading: false, error: null }),
}));

vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => true,
}));

vi.mock("#/contexts/settings-section-header-context", () => ({
  useSettingsSectionHeader: () => ({
    setHideSectionHeader: hooksMock.setHideSectionHeader,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: toastMock.displayErrorToast,
  displaySuccessToast: toastMock.displaySuccessToast,
}));

// HeroUI-heavy controls are not under test here; keep the embedded editor light.
vi.mock("#/components/shared/modals/settings/model-selector", () => ({
  ModelSelector: () => null,
}));

vi.mock("#/components/features/settings/settings-dropdown-input", () => ({
  SettingsDropdownInput: () => null,
}));

function renderView() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <LlmSettingsLocalView />
    </QueryClientProvider>,
  );
}

async function openEditor(profileConfig: Record<string, unknown>) {
  hooksMock.profilesData = {
    profiles: [{ name: PROFILE_NAME, model: PROFILE_MODEL }],
    active_profile: PROFILE_NAME,
  };
  profilesServiceMock.getProfile.mockResolvedValue({
    name: PROFILE_NAME,
    config: profileConfig,
  });

  renderView();

  fireEvent.click(await screen.findByTestId("profile-menu-trigger"));
  fireEvent.click(await screen.findByTestId("profile-edit"));

  return screen.findByTestId("llm-api-key-input");
}

describe("LlmSettingsLocalView API key masking (issue #15706)", () => {
  beforeEach(() => {
    hooksMock.settingsData = baseSettings;
    hooksMock.schemaData = llmSchema;
    hooksMock.conversationSchemaData = null;
    vi.clearAllMocks();
  });

  it("keeps the API key field blank with a <hidden> placeholder when editing a profile that has a stored key", async () => {
    const input = await openEditor({
      model: PROFILE_MODEL,
      api_key: ENCRYPTED_KEY,
      base_url: null,
    });

    // The stored (encrypted) key must not be rendered back into the field.
    expect(input).toHaveValue("");
    expect(input).toHaveAttribute("placeholder", "<hidden>");
    expect(screen.getByTestId("set-indicator")).toBeInTheDocument();
    expect(screen.queryByDisplayValue(ENCRYPTED_KEY)).not.toBeInTheDocument();
    expect(screen.queryByText(ENCRYPTED_KEY)).not.toBeInTheDocument();
  });

  it("keeps the field blank without a key-set indicator when the profile has no stored key", async () => {
    const input = await openEditor({
      model: PROFILE_MODEL,
      base_url: null,
    });

    expect(input).toHaveValue("");
    expect(input).toHaveAttribute("placeholder", "");
    expect(screen.queryByTestId("set-indicator")).not.toBeInTheDocument();
  });

  it("lets the user type a replacement key while editing a profile that has a stored key", async () => {
    const input = await openEditor({
      model: PROFILE_MODEL,
      api_key: ENCRYPTED_KEY,
      base_url: null,
    });

    fireEvent.change(input, { target: { value: "sk-new-replacement" } });

    expect(input).toHaveValue("sk-new-replacement");
    expect(screen.getByTestId("set-indicator")).toBeInTheDocument();
  });
});
