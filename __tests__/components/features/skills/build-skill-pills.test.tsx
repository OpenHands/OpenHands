import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, afterEach, beforeEach } from "vitest";
import { type TFunction } from "i18next";
import { buildSkillPills } from "#/components/features/skills/build-skill-pills";
import { type SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import { type SkillInfo } from "#/types/settings";
import { type ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { type Backend } from "#/api/backend-registry/types";
import { ExtensionsMobileHub } from "#/components/features/skills/extensions-mobile-hub";
import { getSkillCardDescription } from "#/components/features/skills/get-skill-card-description";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import {
  ADD_SKILL_EXAMPLE_COMMAND,
  ADD_SKILL_SKILL_NAME,
} from "#/constants/skills-docs";
import { getSkillChatLaunchMessage } from "#/components/features/skills/get-skill-chat-launch-message";
import { isCopyableSkillSource } from "#/components/features/skills/is-copyable-skill-source";

describe("build-skill-pills", () => {
  /** Returns its key unchanged, for assertions that do not care about copy. */
  const translate = ((key: string) => key) as TFunction;

  function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
    return {
      name: "deno",
      type: "knowledge",
      source: "public",
      category: "environment",
      triggers: [],
      ...overrides,
    };
  }

  /** Mounts the built pills so their rendered output can be asserted. */
  function renderPills(pills: SkillCardPill[]) {
    render(
      <div>
        {pills.map((pill) => (
          <span key={pill.id}>{pill.node}</span>
        ))}
      </div>,
    );
  }

  describe("buildSkillPills category pill", () => {
    it("renders the catalog category", () => {
      renderPills(
        buildSkillPills(buildSkill({ category: "environment" }), translate),
      );

      expect(screen.getByTestId("skill-category-deno")).toHaveTextContent(
        "SETTINGS$SKILLS_CATEGORY_ENVIRONMENT",
      );
    });

    it("omits the pill when the skill is uncategorized", () => {
      renderPills(buildSkillPills(buildSkill({ category: null }), translate));

      expect(
        screen.queryByTestId("skill-category-deno"),
      ).not.toBeInTheDocument();
    });

    it("places the category pill directly after the type badge", () => {
      // Order is asserted on pill ids, so no rendering is needed here.
      const pills = buildSkillPills(buildSkill(), translate);

      expect(pills[0]!.id).toBe("type-knowledge");
      expect(pills[1]!.id).toBe("category-environment");
    });
  });
});

describe("extensions-mobile-hub", () => {
  const cloudBackend: Backend = {
    id: "cloud-1",
    name: "OpenHands Cloud",
    host: "https://app.all-hands.dev",
    apiKey: "token",
    kind: "cloud",
  };

  function renderMobileHub(ui: ReactNode) {
    return render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ActiveBackendProvider>
          <MemoryRouter>{ui}</MemoryRouter>
        </ActiveBackendProvider>
      </QueryClientProvider>,
    );
  }

  describe("ExtensionsMobileHub", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("renders the Plugins item as an enabled link", () => {
      renderMobileHub(<ExtensionsMobileHub />);

      const hub = screen.getByTestId("extensions-mobile-hub");
      const pluginsItem = within(hub).getByTestId(
        "sidebar-extensions-/plugins",
      );
      expect(pluginsItem).not.toHaveAttribute("aria-disabled");
    });

    it("links Apps to its management page", () => {
      renderMobileHub(<ExtensionsMobileHub />);

      const hub = screen.getByTestId("extensions-mobile-hub");
      expect(
        within(hub).getByTestId("sidebar-extensions-/apps"),
      ).toHaveAttribute("href", "/apps");
    });

    describe("cloud backend", () => {
      it("renders the Skills item as an external link to {cloudHost}/settings/skills with the renamed label", () => {
        setRegisteredBackends([cloudBackend]);
        setActiveSelection({ backendId: cloudBackend.id });

        renderMobileHub(<ExtensionsMobileHub />);

        const hub = screen.getByTestId("extensions-mobile-hub");
        const skillsItem = within(hub).getByTestId(
          "sidebar-extensions-/skills",
        );
        expect(skillsItem.tagName).toBe("A");
        expect(skillsItem).toHaveAttribute(
          "href",
          "https://app.all-hands.dev/settings/skills",
        );
        expect(skillsItem).toHaveAttribute("target", "_blank");
        expect(skillsItem).toHaveAttribute("rel", "noopener noreferrer");
        expect(skillsItem).toHaveTextContent(
          "SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK",
        );
      });

      it("hides the Plugins and Apps items", () => {
        setRegisteredBackends([cloudBackend]);
        setActiveSelection({ backendId: cloudBackend.id });

        renderMobileHub(<ExtensionsMobileHub />);

        const hub = screen.getByTestId("extensions-mobile-hub");
        expect(
          within(hub).queryByTestId("sidebar-extensions-/plugins"),
        ).not.toBeInTheDocument();
        expect(
          within(hub).queryByTestId("sidebar-extensions-/apps"),
        ).not.toBeInTheDocument();
      });
    });
  });
});

describe("get-skill-card-description", () => {
  function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
    return {
      name: "test",
      type: "knowledge",
      source: null,
      ...overrides,
    };
  }

  describe("getSkillCardDescription", () => {
    it("prefers the API description field when present", () => {
      expect(
        getSkillCardDescription(
          buildSkill({ description: "From API", content: "ignored body" }),
        ),
      ).toBe("From API");
    });

    it("reads description from YAML frontmatter in content", () => {
      const content = `---
description: SSH helper for remote hosts
name: SSH Microagent
---
# SSH Microagent

Body paragraph that should not be used first.
`;

      expect(getSkillCardDescription(buildSkill({ content }))).toBe(
        "SSH helper for remote hosts",
      );
    });

    it("falls back to the first body paragraph when no description exists", () => {
      const content = `# My Skill

First paragraph for the card.

Second paragraph.`;

      expect(getSkillCardDescription(buildSkill({ content }))).toBe(
        "First paragraph for the card.",
      );
    });

    it("returns an empty string when neither description nor content exists", () => {
      expect(getSkillCardDescription(buildSkill())).toBe("");
    });
  });
});

describe("getSkillChatLaunchMessage", () => {
  it("returns the full add-skill example command for the add-skill skill", () => {
    expect(getSkillChatLaunchMessage({ name: ADD_SKILL_SKILL_NAME })).toBe(
      ADD_SKILL_EXAMPLE_COMMAND,
    );
  });

  it("points the add-skill example at a skill in the bundled extensions catalog", () => {
    const skillName = ADD_SKILL_EXAMPLE_COMMAND.split("/skills/").at(-1);

    expect(SKILLS_CATALOG.map((entry) => entry.name)).toContain(skillName);
  });

  it("returns a slash-command prefix for other skills", () => {
    expect(getSkillChatLaunchMessage({ name: "agent-builder" })).toBe(
      "/agent-builder ",
    );
  });
});

describe("isCopyableSkillSource", () => {
  it("returns false for scope labels and empty values", () => {
    expect(isCopyableSkillSource(null)).toBe(false);
    expect(isCopyableSkillSource("")).toBe(false);
    expect(isCopyableSkillSource("global")).toBe(false);
    expect(isCopyableSkillSource("agent")).toBe(false);
  });

  it("returns true for filesystem paths and URLs", () => {
    expect(isCopyableSkillSource("/skills/deno/SKILL.md")).toBe(true);
    expect(isCopyableSkillSource(".agents/skills/foo/SKILL.md")).toBe(true);
    expect(isCopyableSkillSource("https://github.com/example/skills")).toBe(
      true,
    );
  });
});
