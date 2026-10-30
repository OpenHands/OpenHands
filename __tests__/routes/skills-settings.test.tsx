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
import { SkillsSettingsScreen } from "#/routes/skills-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import SkillsService from "#/api/skills-service";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { HttpError } from "@openhands/typescript-client";
import {
  ADD_SKILL_DOCS_URL,
  ADD_SKILL_EXAMPLE_COMMAND,
} from "#/constants/skills-docs";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings, SkillInfo } from "#/types/settings";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

const navigateMock = vi.fn();
let migrationState: {
  isFetching: boolean;
  isSuccess: boolean;
  error: unknown;
} = { isFetching: false, isSuccess: true, error: null };
vi.mock(
  "#/hooks/use-migrate-default-agent-profile",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("#/hooks/use-migrate-default-agent-profile")
    >()),
    useMigrateDefaultAgentProfile: () => migrationState,
  }),
);
const updateDefaultAgentProfileMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/mutation/sync-agent-profiles", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/mutation/sync-agent-profiles")
  >()),
  updateDefaultAgentProfile: updateDefaultAgentProfileMock,
}));

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
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
  };
}

// The local page lists the server catalog; Cloud lists its own skills.
function mockSkillList(skills: SkillInfo[]) {
  vi.spyOn(SkillsService, "getSkills").mockResolvedValue(skills);
  vi.spyOn(SkillsService, "getSkillCatalog").mockResolvedValue(skills);
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

function renderSkillsSettingsScreen(
  initialEntry = "/skills",
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  }),
) {
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
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });

  return router;
}

describe("SkillsSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    navigateMock.mockReset();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    migrationState = { isFetching: false, isSuccess: true, error: null };
    // No OpenHands `default`: the page keeps its lists in settings.
    vi.spyOn(AgentProfilesService, "getProfile").mockRejectedValue(
      new HttpError(404, "Not Found", {}, "missing"),
    );
  });

  it("renders the description text inside the description badge", async () => {
    mockSkillList([]);

    renderSkillsSettingsScreen();

    const description = await screen.findByTestId(
      "skills-settings-description",
    );
    expect(description).toHaveTextContent("SETTINGS$SKILLS_PAGE_DESCRIPTION");
    expect(screen.getByText("NAV$CUSTOMIZE")).toBeInTheDocument();
    expect(
      screen.queryByTestId("sidebar-extensions-/skills"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("sidebar-extensions-/plugins")).toHaveTextContent(
      "Plugins",
    );
    expect(
      screen.queryByTestId("sidebar-extensions-/mcp"),
    ).not.toBeInTheDocument();
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
    mockSkillList([skill]);

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
    mockSkillList([skill]);

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
    mockSkillList([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    await user.click(
      within(card).getByTestId(`skill-copy-source-${skill.name}`),
    );

    expect(writeText).toHaveBeenCalledWith(skill.source);
  });

  it("hides the copy button when the source is a scope label instead of a path", async () => {
    const skill = buildSkill({ name: "add_repo_inst", source: "global" });
    mockSkillList([skill]);

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
    mockSkillList([
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
    mockSkillList([
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
    mockSkillList([
      buildSkill({ name: "deno", category: "environment" }),
      buildSkill({ name: "prd", category: "writing", triggers: [] }),
    ]);

    renderSkillsSettingsScreen("/skills?category=writing");

    await screen.findByTestId("skill-card-prd");
    expect(screen.queryByTestId("skill-card-deno")).not.toBeInTheDocument();
  });

  it("restores the search text when history navigation changes the query", async () => {
    const user = userEvent.setup();
    mockSkillList([
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
    mockSkillList([
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
    mockSkillList([skill]);

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
    mockSkillList([skill]);

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

  it("saves disabled_skills to the server when a skill is toggled off and settings has no prior disabled_skills field", async () => {
    // Reproduces the bug where disabled_skills is absent from settings (undefined),
    // causing hasHydratedInitialSettings to never become true and the save to be silently skipped.
    const user = userEvent.setup();
    const skill = buildSkill({ name: "save-me" });
    mockSkillList([skill]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ disabled_skills: undefined }),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    await screen.findByTestId(`skill-card-${skill.name}`);
    const card = screen.getByTestId(`skill-card-${skill.name}`);

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    await waitFor(() =>
      expect(saveSpy).toHaveBeenCalledWith(
        expect.objectContaining({ disabled_skills: [skill.name] }),
      ),
    );
  });

  it("saves an updated disabled list when a skill is toggled off and settings already has disabled_skills", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "another-skill" });
    mockSkillList([skill]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ disabled_skills: [] }),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    await screen.findByTestId(`skill-card-${skill.name}`);
    const card = screen.getByTestId(`skill-card-${skill.name}`);

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    await waitFor(() =>
      expect(saveSpy).toHaveBeenCalledWith(
        expect.objectContaining({ disabled_skills: [skill.name] }),
      ),
    );
  });

  it("toggles a skill from the card without opening the modal", async () => {
    const user = userEvent.setup();
    const skill = buildSkill({ name: "card-toggle" });
    mockSkillList([skill]);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${skill.name}`);

    await user.click(within(card).getByTestId(`skill-toggle-${skill.name}`));

    expect(card).not.toHaveClass("opacity-70");
    expect(
      within(card).getByTestId(`skill-toggle-${skill.name}`),
    ).toHaveAttribute("aria-checked", "false");
    expect(screen.queryByTestId("skill-detail-modal")).not.toBeInTheDocument();
  });

  it("keeps same-name local and bundled skills distinct while search and facets change", async () => {
    const user = userEvent.setup();
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    mockSkillList([
      buildSkill({
        name: "skill-creator",
        type: "knowledge",
        source: "/Users/test/.openhands/skills/skill-creator/SKILL.md",
      }),
      buildSkill({
        name: "skill-creator",
        type: "repo",
        triggers: [],
        source: "public",
      }),
      buildSkill({
        name: "deno",
        source: "/skills/deno/SKILL.md",
      }),
    ]);

    renderSkillsSettingsScreen();
    await waitFor(() =>
      expect(screen.getAllByTestId("skill-card-skill-creator")).toHaveLength(2),
    );

    const search = screen.getByTestId("skills-search-input");
    for (let round = 0; round < 3; round += 1) {
      fireEvent.change(search, { target: { value: "creator" } });
      await waitFor(() =>
        expect(screen.queryByTestId("skill-card-deno")).not.toBeInTheDocument(),
      );
      expect(screen.getAllByTestId("skill-card-skill-creator")).toHaveLength(2);

      await user.click(screen.getByTestId("skill-facet-type-repo"));
      await waitFor(() =>
        expect(screen.getAllByTestId("skill-card-skill-creator")).toHaveLength(
          1,
        ),
      );

      await user.click(screen.getByTestId("skill-facet-type-repo"));
      await waitFor(() =>
        expect(screen.getAllByTestId("skill-card-skill-creator")).toHaveLength(
          2,
        ),
      );

      fireEvent.change(search, { target: { value: "" } });
      await screen.findByTestId("skill-card-deno");
      expect(screen.getAllByTestId("skill-card-skill-creator")).toHaveLength(2);
    }

    const duplicateKeyWarnings = consoleError.mock.calls.filter((call) =>
      call.some((arg) => typeof arg === "string" && arg.includes("same key")),
    );
    expect(duplicateKeyWarnings).toHaveLength(0);
  });

  it("shows an empty-state message when no skills match the current filters", async () => {
    const skill = buildSkill();
    mockSkillList([skill]);

    renderSkillsSettingsScreen();
    await screen.findByTestId(`skill-card-${skill.name}`);

    fireEvent.change(screen.getByTestId("skills-search-input"), {
      target: { value: "no-such-skill-xyz" },
    });

    expect(screen.getByTestId("skills-no-match")).toBeInTheDocument();
  });

  it("opens the add skill modal with docs link and closes it", async () => {
    const user = userEvent.setup();
    mockSkillList([]);

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
    mockSkillList([]);

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

  it("lists the skills agent profiles can select on a local backend", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      buildSkill({ name: "repo-only" }),
    ]);
    vi.spyOn(SkillsService, "getSkillCatalog").mockResolvedValue([
      buildSkill({ name: "from-catalog" }),
    ]);

    renderSkillsSettingsScreen();

    expect(
      await screen.findByTestId("skill-card-from-catalog"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("skill-card-repo-only"),
    ).not.toBeInTheDocument();
  });

  it("says the selection is the default list for new agent profiles", async () => {
    mockSkillList([]);

    renderSkillsSettingsScreen();

    expect(
      await screen.findByTestId("skills-default-list-notice"),
    ).toHaveTextContent("SETTINGS$SKILLS_DEFAULT_LIST_NOTICE");
  });

  it("badges a catalog skill the catalog recommends", async () => {
    mockSkillList([
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

  it("renders an unlisted catalog skill as off without it being denied", async () => {
    // The reported bug: everything in the catalog arrived switched on.
    mockSkillList([buildSkill({ name: CATALOG_OPTIONAL })]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ enabled_skills: undefined, disabled_skills: [] }),
    );

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_OPTIONAL}`);

    expect(
      within(card).getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`),
    ).toHaveAttribute("aria-checked", "false");
  });

  it("writes nothing back when the user toggles nothing", async () => {
    // Persisting the hydrated state on mount would race the one-shot migration
    // and could narrow a workspace it had just preserved.
    mockSkillList([buildSkill({ name: CATALOG_OPTIONAL })]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ enabled_skills: undefined, disabled_skills: [] }),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    await screen.findByTestId(`skill-card-${CATALOG_OPTIONAL}`);
    await waitFor(() =>
      expect(screen.getByTestId("skills-result-summary")).toBeInTheDocument(),
    );

    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("saves a catalog toggle to enabled_skills rather than the deny-list", async () => {
    const user = userEvent.setup();
    mockSkillList([buildSkill({ name: CATALOG_RECOMMENDED })]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ enabled_skills: [CATALOG_RECOMMENDED] }),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_RECOMMENDED}`);

    await user.click(
      within(card).getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
    );

    await waitFor(() =>
      expect(saveSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          enabled_skills: [],
          disabled_skills: [],
        }),
      ),
    );
  });

  it("shows and saves the default agent profile's own list", async () => {
    const user = userEvent.setup();
    updateDefaultAgentProfileMock.mockReset().mockResolvedValue(undefined);
    mockSkillList([
      buildSkill({ name: CATALOG_RECOMMENDED }),
      buildSkill({ name: CATALOG_OPTIONAL }),
    ]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ enabled_skills: [CATALOG_RECOMMENDED] }),
    );
    vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
      name: "default",
      profile: {
        id: "p-default",
        name: "default",
        revision: 1,
        agent_kind: "openhands",
        llm_profile_ref: "llm",
        public_skills: [CATALOG_OPTIONAL],
        disabled_skills: [],
      },
    } as never);
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    const optional = await screen.findByTestId(
      `skill-card-${CATALOG_OPTIONAL}`,
    );
    await waitFor(() =>
      expect(
        within(optional).getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`),
      ).toHaveAttribute("aria-checked", "true"),
    );
    expect(
      screen.getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
    ).toHaveAttribute("aria-checked", "false");

    await user.click(
      within(optional).getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`),
    );

    await waitFor(() =>
      expect(updateDefaultAgentProfileMock).toHaveBeenCalledOnce(),
    );
    const [, change] = updateDefaultAgentProfileMock.mock.calls[0];
    expect(
      change({
        agent_kind: "openhands",
        public_skills: [CATALOG_OPTIONAL],
        disabled_skills: [],
      }),
    ).toEqual({ public_skills: [], disabled_skills: [] });
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("keeps a second quick toggle while the first write's refetch lands", async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const releases: (() => void)[] = [];
    updateDefaultAgentProfileMock.mockReset().mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          releases.push(resolve);
        }),
    );
    mockSkillList([
      buildSkill({ name: CATALOG_RECOMMENDED }),
      buildSkill({ name: CATALOG_OPTIONAL }),
    ]);
    const stored = (publicSkills: string[]) =>
      ({
        name: "default",
        profile: {
          id: "p-default",
          name: "default",
          revision: 1,
          agent_kind: "openhands",
          llm_profile_ref: "llm",
          public_skills: publicSkills,
          disabled_skills: [],
        },
      }) as never;
    const getProfile = vi
      .spyOn(AgentProfilesService, "getProfile")
      .mockResolvedValue(stored([]));

    renderSkillsSettingsScreen("/skills", queryClient);
    const first = await screen.findByTestId(
      `skill-toggle-${CATALOG_RECOMMENDED}`,
    );
    await waitFor(() => expect(first).toHaveAttribute("aria-checked", "false"));
    await user.click(first);
    await user.click(screen.getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`));

    // The first write lands and its refetch returns only the first toggle.
    getProfile.mockResolvedValue(stored([CATALOG_RECOMMENDED]));
    await act(() => queryClient.invalidateQueries());

    expect(
      screen.getByTestId(`skill-toggle-${CATALOG_OPTIONAL}`),
    ).toHaveAttribute("aria-checked", "true");
    await act(async () => releases.forEach((release) => release()));
  });

  it("writes nowhere when the default profile cannot be read", async () => {
    const user = userEvent.setup();
    updateDefaultAgentProfileMock.mockReset().mockResolvedValue(undefined);
    mockSkillList([buildSkill({ name: CATALOG_RECOMMENDED })]);
    vi.spyOn(AgentProfilesService, "getProfile").mockRejectedValue(
      new HttpError(500, "Server Error", {}, "boom"),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    const toggle = await screen.findByTestId(
      `skill-toggle-${CATALOG_RECOMMENDED}`,
    );
    await waitFor(() => expect(toggle).toBeDisabled());
    await user.click(toggle);

    expect(saveSpy).not.toHaveBeenCalled();
    expect(updateDefaultAgentProfileMock).not.toHaveBeenCalled();
  });

  it("locks the switches while the default profile's migration has failed", async () => {
    const user = userEvent.setup();
    updateDefaultAgentProfileMock.mockReset().mockResolvedValue(undefined);
    migrationState = {
      isFetching: false,
      isSuccess: false,
      error: new Error("offline"),
    };
    mockSkillList([buildSkill({ name: CATALOG_RECOMMENDED })]);
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    const toggle = await screen.findByTestId(
      `skill-toggle-${CATALOG_RECOMMENDED}`,
    );
    await waitFor(() => expect(toggle).toBeDisabled());
    await user.click(toggle);

    expect(saveSpy).not.toHaveBeenCalled();
    expect(updateDefaultAgentProfileMock).not.toHaveBeenCalled();
  });

  it("drops a re-enabled catalog skill from an unmigrated deny-list", async () => {
    // A stale deny entry would otherwise veto the allow-list entry we just
    // wrote, leaving the toggle on and the skill still absent.
    const user = userEvent.setup();
    mockSkillList([buildSkill({ name: CATALOG_RECOMMENDED })]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        enabled_skills: undefined,
        disabled_skills: [CATALOG_RECOMMENDED],
      }),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderSkillsSettingsScreen();
    const card = await screen.findByTestId(`skill-card-${CATALOG_RECOMMENDED}`);

    await user.click(
      within(card).getByTestId(`skill-toggle-${CATALOG_RECOMMENDED}`),
    );

    await waitFor(() =>
      expect(saveSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          enabled_skills: expect.arrayContaining([CATALOG_RECOMMENDED]),
          disabled_skills: [],
        }),
      ),
    );
  });
});
