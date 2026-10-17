import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  render,
  screen,
  within,
  fireEvent,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SkillsSettingsScreen from "#/routes/skills-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import SkillsService from "#/api/skills-service";
import {
  ADD_SKILL_DOCS_URL,
  ADD_SKILL_EXAMPLE_COMMAND,
} from "#/constants/skills-docs";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings, SkillInfo } from "#/types/settings";
import * as activeBackendContext from "#/contexts/active-backend-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

const navigateMock = vi.fn();

const ACTIVE_PROFILE = {
  id: "profile-default",
  name: "default",
  revision: 1,
  agent_kind: "openhands",
  llm_profile_ref: "default",
  mcp_server_refs: null,
};

/** The skill selection the active agent profile stores. */
function withActiveProfile(fields: Record<string, unknown>) {
  const profile = { ...ACTIVE_PROFILE, ...fields };
  vi.spyOn(AgentProfilesService, "listProfiles").mockResolvedValue({
    profiles: [profile],
    active_agent_profile_id: profile.id,
  } as never);
  vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
    name: profile.name,
    profile,
  } as never);
  return vi.spyOn(AgentProfilesService, "saveProfile").mockResolvedValue({
    name: profile.name,
    message: "saved",
  });
}

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: navigateMock,
    currentPath: "/skills",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    default_profile_migrated: true,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
  };
}

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "deno",
    type: "knowledge",
    source:
      "/Users/test/.openhands/cache/skills/public-skills/skills/deno/SKILL.md",
    description:
      "If the project uses deno, use this skill to initialize Deno projects.",
    triggers: ["deno", "deno.json", "deno.lock"],
    version: "1.0.0",
    license: "Apache-2.0",
    compatibility: "Requires Deno 1.40+",
    metadata: null,
    allowed_tools: ["bash"],
    is_agentskills_format: true,
    disable_model_invocation: false,
    ...overrides,
  };
}

function renderSkillsSettingsScreen(initialEntry = "/skills") {
  const router = createMemoryRouter(
    [
      {
        path: "/skills",
        Component: () => (
          <ActiveBackendProvider>
            <SkillsSettingsScreen />
          </ActiveBackendProvider>
        ),
      },
    ],
    { initialEntries: [initialEntry] },
  );

  render(<RouterProvider router={router} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });

  return router;
}

describe("SkillsSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    navigateMock.mockReset();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    withActiveProfile({ public_skills: [], disabled_skills: [] });
  });

  it("renders the description text inside the description badge", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    renderSkillsSettingsScreen();

    const description = await screen.findByTestId(
      "skills-settings-description",
    );
    expect(description).toHaveTextContent("SETTINGS$SKILLS_PAGE_DESCRIPTION");
    expect(screen.getByText("NAV$CUSTOMIZE")).toBeInTheDocument();
    expect(screen.getByTestId("sidebar-extensions-/skills")).toHaveTextContent(
      "Skills",
    );
    expect(screen.getByTestId("sidebar-extensions-/plugins")).toHaveTextContent(
      "Plugins",
    );
    expect(screen.getByTestId("sidebar-extensions-/mcp")).toHaveTextContent(
      "MCP Servers",
    );
  });

  it("shows card subtitle text from skill content when description is omitted", async () => {
    const skill = buildSkill({
      name: "SSH Microagent",
      description: null,
      content: `---
description: Connect and run commands on remote machines over SSH.
---
# SSH Microagent

Full skill body.`,
      triggers: ["ssh"],
    });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    expect(
      within(card).getByTestId(`skill-description-${skill.name}`),
    ).toHaveTextContent(
      "Connect and run commands on remote machines over SSH.",
    );
  });

  it("surfaces the YAML description under the card title with the source path beneath it", async () => {
    const skill = buildSkill();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    expect(
      within(card).getByTestId(`skill-description-${skill.name}`),
    ).toHaveTextContent(skill.description!);
    expect(
      within(card).getByTestId(`skill-source-${skill.name}`),
    ).toHaveTextContent(skill.source!);
    expect(
      within(card).getByTestId(`skill-icon-${skill.name}`),
    ).toBeInTheDocument();
    expect(
      within(card).getByTestId("skill-type-badge-knowledge"),
    ).toHaveTextContent("SETTINGS$SKILLS_TYPE_KNOWLEDGE");
  });

  it("copies the source path when the copy button is clicked", async () => {
    const user = userEvent.setup();
    const skill = buildSkill();
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(writeText);
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    await user.click(
      within(card).getByTestId(`skill-copy-source-${skill.name}`),
    );

    expect(writeText).toHaveBeenCalledWith(skill.source);
  });

  it("hides the copy button when the source is a scope label instead of a path", async () => {
    const skill = buildSkill({ name: "add_repo_inst", source: "global" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    expect(
      within(card).getByTestId(`skill-source-${skill.name}`),
    ).toHaveTextContent("global");
    expect(
      within(card).queryByTestId(`skill-copy-source-${skill.name}`),
    ).not.toBeInTheDocument();
  });

  it("filters skills by name, description, or trigger via the search input", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: "deno", description: "Deno runtime helper" }),
      buildSkill({
        name: "vercel",
        description: "Preview deployment helper",
        triggers: ["vercel", "preview deployment"],
        source: "/skills/vercel/SKILL.md",
      }),
    ]);

    renderSkillsSettingsScreen();
    await screen.findByTestId("skill-card-deno");

    fireEvent.change(screen.getByTestId("skills-search-input"), {
      target: { value: "preview" },
    });

    expect(screen.queryByTestId("skill-card-deno")).not.toBeInTheDocument();
    expect(screen.getByTestId("skill-card-vercel")).toBeInTheDocument();
  });

  it("narrows the visible skills when a facet row is selected", async () => {
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: "deno", type: "knowledge" }),
      buildSkill({
        name: "global-rules",
        type: "repo",
        triggers: [],
        source: "/skills/global-rules.md",
      }),
    ]);

    const router = renderSkillsSettingsScreen();
    await screen.findByTestId("skill-card-deno");

    await user.click(screen.getByTestId("skill-facet-type-repo"));

    expect(screen.queryByTestId("skill-card-deno")).not.toBeInTheDocument();
    expect(screen.getByTestId("skill-card-global-rules")).toBeInTheDocument();
    expect(router.state.location.search).toBe("?type=repo");
  });

  it("seeds the filter state from the URL on load", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: "deno", category: "environment" }),
      buildSkill({ name: "prd", category: "writing", triggers: [] }),
    ]);

    renderSkillsSettingsScreen("/skills?category=writing");

    await screen.findByTestId("skill-card-prd");
    expect(screen.queryByTestId("skill-card-deno")).not.toBeInTheDocument();
  });

  it("restores the search text when history navigation changes the query", async () => {
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: "deno", type: "knowledge", description: "A helper" }),
      buildSkill({
        name: "global-rules",
        type: "repo",
        description: "A helper",
        triggers: [],
        source: "/skills/global-rules.md",
      }),
    ]);

    const router = renderSkillsSettingsScreen("/skills?q=helper");
    await screen.findByTestId("skill-card-deno");
    const search = screen.getByTestId("skills-search-input");

    // Push a second entry, then drop the query from it, so going back lands on a URL whose `q` differs from the input.
    await user.click(screen.getByTestId("skill-facet-type-repo"));
    fireEvent.change(search, { target: { value: "" } });
    await waitFor(() =>
      expect(router.state.location.search).toBe("?type=repo"),
    );

    await act(() => router.navigate(-1));

    await waitFor(() => expect(search).toHaveValue("helper"));
    expect(router.state.location.search).toBe("?q=helper");
  });

  it("filters through the mobile filters modal", async () => {
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: "deno", type: "knowledge" }),
      buildSkill({
        name: "global-rules",
        type: "repo",
        triggers: [],
        source: "/skills/global-rules.md",
      }),
    ]);

    renderSkillsSettingsScreen();
    await screen.findByTestId("skill-card-deno");

    await user.click(screen.getByTestId("skills-filters-button"));
    const modal = await screen.findByTestId("skill-filters-modal");
    await user.click(within(modal).getByTestId("skill-facet-type-repo"));

    expect(screen.queryByTestId("skill-card-deno")).not.toBeInTheDocument();
    expect(screen.getByTestId("skill-card-global-rules")).toBeInTheDocument();
  });

  it("opens a detail modal with full metadata when a skill card is clicked", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({
      name: "rich",
      license: "MIT",
      compatibility: "Requires Python 3.11+",
      allowed_tools: ["bash", "execute_bash"],
      source: "/skills/rich/SKILL.md",
    });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    await user.click(card);

    const modal = await screen.findByTestId("skill-detail-modal");
    expect(modal).toHaveAttribute("data-skill-name", skill.name);
    expect(
      within(modal).getByTestId(`skill-modal-pill-${skill.name}-license`),
    ).toHaveTextContent("MIT");
    expect(
      within(modal).getByTestId(`skill-modal-pill-${skill.name}-compatibility`),
    ).toHaveTextContent("Requires Python 3.11+");
    expect(
      within(modal).getByTestId(`skill-modal-pill-${skill.name}-tool-bash`),
    ).toHaveTextContent("bash");
    expect(
      within(modal).getByTestId(
        `skill-modal-pill-${skill.name}-tool-execute_bash`,
      ),
    ).toHaveTextContent("execute_bash");
    expect(
      within(modal).getByTestId(`skill-modal-toggle-${skill.name}`),
    ).toBeInTheDocument();
  });

  it("toggles a skill from the detail modal", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "toggle-me" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);
    await user.click(card);

    const modal = await screen.findByTestId("skill-detail-modal");
    expect(
      within(modal).getByText("SETTINGS$SKILLS_ENABLED"),
    ).toBeInTheDocument();

    await user.click(
      within(modal).getByTestId(`skill-modal-toggle-${skill.name}`),
    );

    expect(card).not.toHaveClass("opacity-70");
    expect(
      within(card).getByTestId(`skill-toggle-${skill.name}`),
    ).toHaveAttribute("aria-checked", "false");
    expect(
      within(modal).getByText("SETTINGS$SKILLS_DISABLED"),
    ).toBeInTheDocument();
  });

  it("saves a user skill toggled off to the active profile's deny-list", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "save-me" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);
    const saveProfile = withActiveProfile({ public_skills: [] });

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    await waitFor(() =>
      expect(saveProfile).toHaveBeenCalledWith(
        "default",
        expect.objectContaining({
          public_skills: [],
          disabled_skills: [skill.name],
        }),
      ),
    );
  });
  it("names the profile the page edits", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    renderSkillsSettingsScreen();

    expect(
      await screen.findByTestId("active-profile-scope-notice"),
    ).toHaveTextContent("SETTINGS$BEHAVIOR_ACTIVE_PROFILE");
  });
  it("toggles a skill from the card without opening the modal", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "card-toggle" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    expect(card).not.toHaveClass("opacity-70");
    expect(
      within(card).getByTestId(`skill-toggle-${skill.name}`),
    ).toHaveAttribute("aria-checked", "false");
    expect(screen.queryByTestId("skill-detail-modal")).not.toBeInTheDocument();
  });

  it("shows an empty-state message when no skills match the current filters", async () => {
    const skill = buildSkill();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);

    renderSkillsSettingsScreen();
    await screen.findByTestId(`skill-card-${skill.name}`);

    fireEvent.change(screen.getByTestId("skills-search-input"), {
      target: { value: "no-such-skill-xyz" },
    });

    expect(screen.getByTestId("skills-no-match")).toBeInTheDocument();
  });

  it("opens the add skill modal with docs link and closes it", async () => {
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    renderSkillsSettingsScreen();
    await screen.findByTestId("skills-add-skill-button");

    await user.click(screen.getByTestId("skills-add-skill-button"));

    const modal = await screen.findByTestId("add-skill-modal");
    expect(modal).toBeInTheDocument();
    expect(screen.getByTestId("add-skill-modal-example")).toHaveTextContent(
      "/add-skill https://github.com/OpenHands/extensions/tree/main/skills/code-review",
    );
    expect(screen.getByTestId("add-skill-modal-docs-link")).toHaveAttribute(
      "href",
      ADD_SKILL_DOCS_URL,
    );

    await user.click(screen.getByTestId("add-skill-modal-dismiss"));

    expect(screen.queryByTestId("add-skill-modal")).not.toBeInTheDocument();
  });

  it("copies the example command from the add skill modal", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(writeText);
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    renderSkillsSettingsScreen();
    await user.click(await screen.findByTestId("skills-add-skill-button"));
    await screen.findByTestId("add-skill-modal");

    await user.click(screen.getByTestId("add-skill-modal-example-copy"));

    expect(writeText).toHaveBeenCalledWith(ADD_SKILL_EXAMPLE_COMMAND);
  });
  // A skill from the bundled `@openhands/extensions` catalog: those are
  // governed by the `enabled_skills` allow-list, not the deny-list.
  const CATALOG_RECOMMENDED = "add-skill";
  const CATALOG_OPTIONAL = "add-javadoc";

  it("shows the notice that skill changes only reach new conversations", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    renderSkillsSettingsScreen();

    expect(
      await screen.findByTestId("skills-new-conversation-notice"),
    ).toHaveTextContent("SETTINGS$SKILLS_NEW_CONVERSATION_NOTICE");
  });

  it("badges a catalog skill the catalog recommends", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: CATALOG_RECOMMENDED }),
      buildSkill({ name: CATALOG_OPTIONAL }),
    ]);

    renderSkillsSettingsScreen();
    await screen.findByTestId(`skill-card-${CATALOG_RECOMMENDED}`);

    expect(
      screen.getByTestId(`skill-recommended-${CATALOG_RECOMMENDED}`),
    ).toHaveTextContent("SETTINGS$SKILLS_RECOMMENDED");
    expect(
      screen.queryByTestId(`skill-recommended-${CATALOG_OPTIONAL}`),
    ).not.toBeInTheDocument();
  });

  it("renders a catalog skill the profile does not select as off", async () => {
    // The reported bug: everything in the catalog arrived switched on.
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: CATALOG_OPTIONAL }),
    ]);
    withActiveProfile({
      public_skills: [CATALOG_RECOMMENDED],
      disabled_skills: [],
    });

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_OPTIONAL}`);

    await waitFor(() =>
      expect(
        within(card).getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`),
      ).toHaveAttribute("aria-checked", "false"),
    );
  });
  it("writes nothing back when the user toggles nothing", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: CATALOG_OPTIONAL }),
    ]);
    const saveProfile = withActiveProfile({ public_skills: null });

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_OPTIONAL}`);
    // A profile without a selection loads the whole catalog.
    await waitFor(() =>
      expect(
        within(card).getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`),
      ).toHaveAttribute("aria-checked", "true"),
    );

    expect(saveProfile).not.toHaveBeenCalled();
  });
  it("saves a catalog toggle to the profile's allow-list rather than its deny-list", async () => {
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: CATALOG_RECOMMENDED }),
    ]);
    const saveProfile = withActiveProfile({
      public_skills: [CATALOG_RECOMMENDED],
      disabled_skills: [],
    });

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_RECOMMENDED}`);
    await waitFor(() =>
      expect(
        within(card).getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
      ).toHaveAttribute("aria-checked", "true"),
    );

    await user.click(
      within(card).getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
    );

    await waitFor(() =>
      expect(saveProfile).toHaveBeenCalledWith(
        "default",
        expect.objectContaining({ public_skills: [], disabled_skills: [] }),
      ),
    );
  });
  it("drops a re-enabled catalog skill from the profile's deny-list", async () => {
    // A stale deny entry would otherwise veto the allow-list entry we just
    // wrote, leaving the toggle on and the skill still absent.
    const user = userEvent.setup();
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: CATALOG_RECOMMENDED }),
    ]);
    const saveProfile = withActiveProfile({
      public_skills: null,
      disabled_skills: [CATALOG_RECOMMENDED],
    });

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_RECOMMENDED}`);
    await waitFor(() =>
      expect(
        within(card).getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
      ).toHaveAttribute("aria-checked", "false"),
    );

    await user.click(
      within(card).getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
    );

    await waitFor(() =>
      expect(saveProfile).toHaveBeenCalledWith(
        "default",
        expect.objectContaining({
          public_skills: expect.arrayContaining([CATALOG_RECOMMENDED]),
          disabled_skills: [],
        }),
      ),
    );
  });

  it("refuses toggles while an ACP profile is active", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "save-me" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);
    const saveProfile = withActiveProfile({
      agent_kind: "acp",
      acp_server: "claude-code",
      llm_profile_ref: null,
    });
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);
    expect(
      await screen.findByTestId("active-profile-scope-notice"),
    ).toHaveTextContent("SETTINGS$BEHAVIOR_ACP_PROFILE");

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    expect(toast).toHaveBeenCalledWith("SETTINGS$BEHAVIOR_ACP_PROFILE");
    expect(saveProfile).not.toHaveBeenCalled();
  });

  it("says when the profile can't load and refuses toggles until it does", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "save-me" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);
    const saveProfile = withActiveProfile({});
    vi.mocked(AgentProfilesService.getProfile).mockRejectedValue(
      new Error("unreachable"),
    );
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);
    await screen.findByTestId("active-profile-load-error");

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    expect(toast).toHaveBeenCalledWith("SETTINGS$BEHAVIOR_PROFILE_LOAD_FAILED");
    expect(saveProfile).not.toHaveBeenCalled();
  });

  it("saves a toggle to the account's deny-list on Cloud", async () => {
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "test-key",
        kind: "cloud",
      },
      orgId: "org-1",
    });
    const user = userEvent.setup();
    const skill = buildSkill({ name: "save-me" });
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([skill]);
    const saveProfile = withActiveProfile({});
    const saveSettings = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true as never);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);
    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    await waitFor(() =>
      expect(saveSettings).toHaveBeenCalledWith(
        expect.objectContaining({ disabled_skills: [skill.name] }),
      ),
    );
    expect(saveProfile).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("active-profile-scope-notice"),
    ).not.toBeInTheDocument();
  });
});
