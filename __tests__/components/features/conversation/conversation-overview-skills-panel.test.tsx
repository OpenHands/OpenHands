import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationOverviewSkillsPanel } from "#/components/features/conversation/conversation-overview-skills-panel";
import SettingsService from "#/api/settings-service/settings-service.api";
import SkillsService from "#/api/skills-service";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import type { SkillInfo } from "#/types/settings";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { selected_workspace: "/workspace/project/demo" },
  }),
}));

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "deno",
    type: "knowledge",
    source: "/Users/test/.openhands/cache/skills/public-skills/skills/deno/SKILL.md",
    description: "Use this skill for Deno projects.",
    triggers: ["deno"],
    version: "1.0.0",
    license: "Apache-2.0",
    compatibility: null,
    metadata: null,
    allowed_tools: null,
    is_agentskills_format: true,
    disable_model_invocation: false,
    ...overrides,
  };
}

function renderPanel(openAdd = false) {
  return render(<ConversationOverviewSkillsPanel openAdd={openAdd} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

describe("ConversationOverviewSkillsPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
  });

  it("opens the add skill modal when openAdd is true", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([buildSkill()]);

    renderPanel(true);

    expect(await screen.findByTestId("add-skill-modal")).toBeInTheDocument();
  });

  it("shows the empty state without an inline add skill button", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    renderPanel();

    expect(
      await screen.findByTestId("conversation-overview-skills-empty"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-skills-add-skill-button"),
    ).not.toBeInTheDocument();
  });

  it("defaults to this-project scope and can show all skills", async () => {
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({
        name: "project-skill",
        source: "/workspace/project/demo/.openhands/skills/project/SKILL.md",
      }),
      buildSkill({ name: "public-skill" }),
    ]);

    renderPanel();

    expect(
      await screen.findByTestId("conversation-overview-skills-scope"),
    ).toBeInTheDocument();
    expect(await screen.findByText("project-skill")).toBeInTheDocument();
    expect(screen.queryByText("public-skill")).not.toBeInTheDocument();

    await user.click(
      screen.getByTestId("conversation-overview-skills-scope-option-all"),
    );

    expect(await screen.findByText("public-skill")).toBeInTheDocument();
    expect(screen.getByText("project-skill")).toBeInTheDocument();
  });
});
