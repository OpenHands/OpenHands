import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { DefaultLlmProfileSelector } from "#/components/features/settings/llm-profiles/default-llm-profile-selector";
import AgentProfilesService, {
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { I18nKey } from "#/i18n/declaration";

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "backend-1", kind: "local" },
    orgId: null,
  }),
}));

const connection: ProviderConnection = {
  id: "conn-personal",
  display_name: "OpenHands Personal",
  provider: "openhands",
  base_url: null,
  created_at: 1,
  updated_at: 1,
  api_key_set: true,
};

const profiles: ProfileInfo[] = [
  {
    name: "glm",
    model: "openhands/glm-5.2",
    base_url: null,
    api_key_set: true,
    provider_connection_id: "conn-personal",
  },
  {
    name: "do-sonnet",
    model: "digitalocean/anthropic-claude-sonnet-5.5",
    base_url: null,
    api_key_set: true,
  },
  { name: "terra", model: "gpt-5.6-terra", base_url: null, api_key_set: true },
];

const agentProfile = (
  overrides: Partial<AgentProfileListResponse["profiles"][number]>,
) => ({
  id: "agent-1",
  name: "default",
  agent_kind: "openhands" as const,
  revision: 1,
  llm_profile_ref: null,
  mcp_server_refs: null,
  ...overrides,
});

const renderSelector = ({
  activeProfile = "glm",
  canManage = true,
  onSelect = vi.fn(),
  agents = {
    profiles: [agentProfile({})],
    active_agent_profile_id: "agent-1",
  } as AgentProfileListResponse,
} = {}) => {
  vi.spyOn(AgentProfilesService, "listProfiles").mockResolvedValue(agents);
  renderWithProviders(
    <DefaultLlmProfileSelector
      profiles={profiles}
      activeProfile={activeProfile}
      connections={[connection]}
      canManage={canManage}
      isActivating={false}
      onSelect={onSelect}
    />,
  );
  return { onSelect };
};

describe("DefaultLlmProfileSelector", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the default profile with its provider connection and model", () => {
    renderSelector();

    expect(screen.getByRole("combobox")).toHaveValue("glm");
    expect(screen.getByTestId("default-llm-profile-details")).toHaveTextContent(
      "OpenHands Personal · openhands/glm-5.2",
    );
  });

  it("names each option's provider and model, and selects the picked profile", async () => {
    const user = userEvent.setup();
    const { onSelect } = renderSelector();

    await user.click(screen.getByRole("combobox"));
    const listbox = await screen.findByRole("listbox");

    // A linked profile names its connection; an unlinked one falls back to
    // its model's provider prefix, then to "Unlinked".
    expect(
      within(listbox).getByRole("option", { name: /^do-sonnet/ }),
    ).toHaveTextContent(
      "digitalocean · digitalocean/anthropic-claude-sonnet-5.5",
    );
    const terra = within(listbox).getByRole("option", { name: /^terra/ });
    expect(terra).toHaveTextContent(
      `${I18nKey.SETTINGS$PROFILE_UNLINKED_PROVIDER} · gpt-5.6-terra`,
    );

    await user.click(terra);
    expect(onSelect).toHaveBeenCalledWith("terra");
  });

  it("explains where the default profile is used from its info icon", async () => {
    const user = userEvent.setup();
    renderSelector();

    // The test i18n returns keys, so the label and tooltip read as their keys.
    await user.hover(
      screen.getByRole("button", {
        name: I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_INFO_LABEL,
      }),
    );

    expect(
      await screen.findByText(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_INFO),
    ).toBeInTheDocument();
  });

  it("is read-only for view-only members", () => {
    renderSelector({ canManage: false });

    expect(screen.getByRole("combobox")).toBeDisabled();
  });

  it("explains when the active agent profile starts conversations with another LLM", async () => {
    renderSelector({
      agents: {
        profiles: [
          agentProfile({}),
          agentProfile({
            id: "agent-2",
            name: "Reviewer",
            llm_profile_ref: "do-sonnet",
          }),
        ],
        active_agent_profile_id: "agent-2",
      } as AgentProfileListResponse,
    });

    // The test i18n returns keys, so assert the override copy is the one shown.
    expect(
      await screen.findByTestId("default-llm-profile-agent-note"),
    ).toHaveTextContent(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_AGENT_OVERRIDE);
  });

  it("shows no note when the default agent profile is active", async () => {
    renderSelector();

    await waitFor(() => {
      expect(AgentProfilesService.listProfiles).toHaveBeenCalled();
    });
    expect(
      screen.queryByTestId("default-llm-profile-agent-note"),
    ).not.toBeInTheDocument();
  });
});
