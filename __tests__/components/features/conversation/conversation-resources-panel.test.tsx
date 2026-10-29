import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { createInstance } from "i18next";
import { I18nextProvider, initReactI18next } from "react-i18next";
import type { OpenHandsAgentProfile } from "@openhands/typescript-client";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import ConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import SettingsService from "#/api/settings-service/settings-service.api";
import SkillsService from "#/api/skills-service";
import { SecretsService } from "#/api/secrets-service";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { NavigationProvider } from "#/context/navigation-context";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import {
  BACKENDS_STORAGE_KEY,
  ACTIVE_BACKEND_STORAGE_KEY,
} from "#/api/backend-registry/storage";
import { ConversationResourcesPanel } from "#/components/features/conversation/conversation-resources-panel";
import { DEFAULT_SETTINGS } from "#/services/settings";
import {
  setMcpServerHealth,
  __resetMcpHealthStoreForTests,
} from "#/api/mcp-health/mcp-health-store";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import translations from "#/i18n/translation.json";

vi.unmock("react-i18next");

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: "en",
  resources: {
    en: {
      openhands: Object.fromEntries(
        Object.entries(translations).map(([key, value]) => [key, value.en]),
      ),
    },
  },
  defaultNS: "openhands",
  interpolation: { escapeValue: false },
});
let conversation: AppConversation;
let profile: OpenHandsAgentProfile;

function showPanel(kind: "local" | "cloud" = "local") {
  localStorage.setItem(
    BACKENDS_STORAGE_KEY,
    JSON.stringify([
      {
        id: "test",
        name: "Test",
        kind,
        host: "http://localhost:8000",
        apiKey: "dummy",
      },
    ]),
  );
  localStorage.setItem(
    ACTIVE_BACKEND_STORAGE_KEY,
    JSON.stringify({
      backendId: "test",
      orgId: kind === "cloud" ? "org" : null,
    }),
  );
  __resetActiveStoreForTests();
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <I18nextProvider i18n={i18n}>
        <ActiveBackendProvider>
          <NavigationProvider
            value={{
              conversationId: "conversation",
              currentPath: "/conversations/conversation",
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            <MemoryRouter>
              <ConversationResourcesPanel />
            </MemoryRouter>
          </NavigationProvider>
        </ActiveBackendProvider>
      </I18nextProvider>
    </QueryClientProvider>,
  );
}

async function expand(label: string) {
  await userEvent.click(screen.getByText(label, { selector: "summary" }));
  return within(
    screen.getByText(label, { selector: "summary" }).closest("details")!,
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  __resetMcpHealthStoreForTests();
  localStorage.clear();
  conversation = {
    id: "conversation",
    llm_model: "conversation-model",
    agent_kind: "openhands",
    sandbox_status: "RUNNING",
    launched_agent_profile: {
      agent_profile_id: "profile-id",
      revision: 3,
      secret_refs: ["ALLOWED_KEY"],
    },
  } as AppConversation;
  profile = {
    agent: "Agent",
    system_message_suffix: null,
    tools: null,
    tool_concurrency_limit: 4,
    id: "profile-id",
    name: "Research agent",
    revision: 3,
    agent_kind: "openhands",
    llm_profile_ref: "llm",
    public_skills: [],
    disabled_skills: ["hidden-skill"],
    mcp_server_refs: ["allowed-server"],
    condenser: {
      condenser_kind: "llm_summarizing",
      api_key: "never-render-this",
    },
    verification: {
      critic_enabled: true,
      enable_iterative_refinement: false,
      critic_mode: "all",
      critic_threshold: 0.5,
      max_refinement_iterations: 3,
      critic_server_url: null,
      critic_model_name: null,
    },
  } as OpenHandsAgentProfile;
  vi.spyOn(ConversationService, "batchGetAppConversations").mockImplementation(
    async () => [conversation],
  );
  vi.spyOn(AgentProfilesService, "listProfiles").mockImplementation(
    async () => ({
      profiles: [profile],
      active_agent_profile_id: "other-profile",
    }),
  );
  vi.spyOn(AgentProfilesService, "getProfile").mockImplementation(async () => ({
    name: profile.name,
    profile,
  }));
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
    ...DEFAULT_SETTINGS,
    llm_model: "global-model",
    agent_settings: {
      ...DEFAULT_SETTINGS.agent_settings,
      mcp_config: {
        "allowed-server": { transport: "http", url: "https://allowed.invalid" },
        "hidden-server": { transport: "http", url: "https://hidden.invalid" },
      },
    },
  });
  vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
    {
      name: "project-skill",
      type: "knowledge",
      source: "/workspace/.agents/skills/project-skill/SKILL.md",
    },
    {
      name: "hidden-skill",
      type: "knowledge",
      source: "/workspace/.agents/skills/hidden-skill/SKILL.md",
    },
  ]);
  vi.spyOn(SkillsService, "getConversationSkills").mockResolvedValue([
    { name: "cloud-skill", type: "knowledge", source: "public" },
  ]);
  vi.spyOn(SecretsService, "getSecretsOrThrow").mockResolvedValue([
    { name: "SAVED_KEY", description: "" },
  ]);
});

describe("conversation resources", () => {
  it("updates the displayed MCP health from the existing store", async () => {
    showPanel();
    await screen.findByText("Research agent");
    const mcp = await expand("MCP servers");
    const server = {
      id: "allowed-server",
      name: "allowed-server",
      type: "shttp" as const,
      url: "https://allowed.invalid",
    };
    act(() =>
      setMcpServerHealth(getMcpServerHealthKey(server), {
        status: "healthy",
        verification: "connectivity-only",
        toolCount: 2,
        checkedAt: Date.now(),
      }),
    );
    expect(
      mcp.getByText(i18n.t("MCP$HEALTH_STATUS_CONNECTIVITY_ONLY")),
    ).toBeInTheDocument();
    act(() =>
      setMcpServerHealth(getMcpServerHealthKey(server), {
        status: "failed",
        kind: "unknown",
        error: "private diagnostic",
        checkedAt: Date.now(),
      }),
    );
    expect(mcp.getByText(i18n.t("ERROR$GENERIC"))).toBeInTheDocument();
    expect(mcp.queryByText("private diagnostic")).not.toBeInTheDocument();
  });

  it("reports unavailable profile resources after a failed profile lookup", async () => {
    vi.spyOn(AgentProfilesService, "getProfile").mockRejectedValue(
      new Error("lookup failed"),
    );
    showPanel();
    await screen.findByText("Research agent");
    await waitFor(() =>
      expect(AgentProfilesService.getProfile).toHaveBeenCalled(),
    );
    expect(
      (await expand("Condenser")).getByText("Unavailable"),
    ).toBeInTheDocument();
  });

  it("lists saved secret names for an unrestricted launch without rendering values or descriptions", async () => {
    conversation.launched_agent_profile!.secret_refs = null;
    const secrets = [
      {
        name: "SAVED_KEY",
        description: "private description",
        value: "private value",
      },
    ];
    vi.spyOn(SecretsService, "getSecretsOrThrow").mockResolvedValue(secrets);
    showPanel();
    const section = await expand("Secrets");
    expect(await section.findByText("SAVED_KEY")).toBeInTheDocument();
    expect(section.queryByText("private value")).not.toBeInTheDocument();
    expect(section.queryByText("private description")).not.toBeInTheDocument();
  });

  it("shows the launched profile, scoped resources and reported model without secret values", async () => {
    showPanel();
    await screen.findByText("Research agent");
    expect(screen.getByText("Launch revision: 3")).toBeInTheDocument();
    expect(
      (await expand("Model")).getByText("conversation-model"),
    ).toBeInTheDocument();
    const skills = await expand("Skills");
    expect(await skills.findByText("project-skill")).toBeInTheDocument();
    expect(skills.queryByText("hidden-skill")).not.toBeInTheDocument();
    const mcp = await expand("MCP servers");
    expect(mcp.getByText("allowed-server")).toBeInTheDocument();
    expect(mcp.queryByText("hidden-server")).not.toBeInTheDocument();
    expect(
      (await expand("Secrets")).getByText("ALLOWED_KEY"),
    ).toBeInTheDocument();
    expect(
      (await expand("Condenser")).getByText("llm_summarizing"),
    ).toBeInTheDocument();
    expect(screen.queryByText("never-render-this")).not.toBeInTheDocument();
    expect(screen.queryByText("global-model")).not.toBeInTheDocument();
    expect(SecretsService.getSecretsOrThrow).not.toHaveBeenCalled();
    expect(
      (await expand("Verification")).getByText("Enabled"),
    ).toBeInTheDocument();
  });

  it("warns when the profile changed and does not attribute its new configuration to the old conversation", async () => {
    profile.revision = 4;
    showPanel();
    expect(
      await screen.findByText(/Profile changed since launch/),
    ).toBeInTheDocument();
    expect(
      (await expand("Condenser")).getByText("Unavailable"),
    ).toBeInTheDocument();
    expect(screen.queryByText("llm_summarizing")).not.toBeInTheDocument();
  });

  it("does not substitute global defaults when no launch profile was recorded", async () => {
    conversation.launched_agent_profile = null;
    showPanel();
    await screen.findByText(/No launch profile recorded/);
    expect(
      (await expand("MCP servers")).getByText("Unavailable"),
    ).toBeInTheDocument();
    expect(AgentProfilesService.listProfiles).not.toHaveBeenCalled();
    expect(screen.queryByText("global-model")).not.toBeInTheDocument();
  });

  it("gates OpenHands-only categories for ACP while keeping its reported model", async () => {
    conversation.agent_kind = "acp";
    showPanel();
    await screen.findByText("Research agent");
    expect(
      (await expand("Skills")).getByText("Not available on this backend"),
    ).toBeInTheDocument();
    expect(
      (await expand("Verification")).getByText("Not available on this backend"),
    ).toBeInTheDocument();
    expect(
      (await expand("Model")).getByText("conversation-model"),
    ).toBeInTheDocument();
  });

  it("uses cloud conversation skills and does not expose unsupported profile secret scope", async () => {
    showPanel("cloud");
    await screen.findByText("Research agent");
    expect(
      await (await expand("Skills")).findByText("cloud-skill"),
    ).toBeInTheDocument();
    expect(
      (await expand("Secrets")).getByText("Not available on this backend"),
    ).toBeInTheDocument();
    expect(SecretsService.getSecretsOrThrow).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(SkillsService.getConversationSkills).toHaveBeenCalledWith(
        "conversation",
      ),
    );
  });
});
