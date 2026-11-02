import { describe, expect, it, vi } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { Activity, Bot } from "lucide-react";
import type { RecommendedAutomation } from "@openhands/extensions/automations";
import {
  getAutomationIcon,
  getAutomationLaunchPrompt,
  getAutomationAvailability,
  rankByAvailability,
  resolveAutomationImpactStatement,
} from "#/utils/automation-catalog";
import type { Automation } from "#/types/automation";

// Entries carrying an `impact` — valid and malformed — and one declaring none
// are appended to the real catalog, so what a given package release declares
// never decides these assertions.
vi.mock("@openhands/extensions/automations", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@openhands/extensions/automations")>();
  const syntheticEntries = [
    { id: "impact-absent" },
    {
      id: "impact-valid",
      impact: {
        basis: "completed-runs",
        one: "1 widget check completed",
        other: "{{count}} widget checks completed",
      },
    },
    {
      id: "impact-unknown-basis",
      impact: { basis: "run-counter", one: "1 widget", other: "{{count}} widgets" },
    },
    {
      id: "impact-markup",
      impact: {
        basis: "completed-runs",
        one: "<b>1 check</b>",
        other: "{{count}} <b>checks</b>",
      },
    },
    {
      id: "impact-no-count",
      impact: { basis: "completed-runs", one: "one check", other: "many checks" },
    },
    {
      id: "impact-foreign-placeholder",
      impact: {
        basis: "completed-runs",
        one: "1 check",
        other: "{{count}} checks for {{form.repo}}",
      },
    },
  ];
  return {
    ...actual,
    AUTOMATION_CATALOG: [
      ...actual.AUTOMATION_CATALOG,
      ...syntheticEntries,
    ] as typeof actual.AUTOMATION_CATALOG,
  };
});

const automationById = (id: string) =>
  AUTOMATION_CATALOG.find((automation) => automation.id === id)!;

function createInstalledAutomation(templateId?: string): Automation {
  return {
    id: "automation-1",
    name: "Widget Monitor",
    trigger: { type: "cron" },
    enabled: true,
    prompt: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...(templateId
      ? {
          preset_metadata: {
            template: { id: templateId, version: "1.0.0", config: {} },
          },
        }
      : {}),
  };
}

describe("getAutomationLaunchPrompt", () => {
  it("resolves the command from the skill that implements the automation", () => {
    // Arrange / Act / Assert — an entry whose skill is its own id, and one
    // that names a different skill, both resolve to that skill's command.
    expect(
      getAutomationLaunchPrompt(automationById("github-pr-reviewer")),
    ).toBe("/pr-reviewer:setup");
    expect(
      getAutomationLaunchPrompt(
        automationById("incident-retrospective-drafter"),
      ),
    ).toBe("/incident-retro:setup");
  });

  it("spells the request out when the skill declares no command", () => {
    // Arrange / Act / Assert — `jira-issue-to-pr` is invoked by description,
    // so there is no trigger to resolve.
    expect(getAutomationLaunchPrompt(automationById("jira-issue-to-pr"))).toBe(
      "Set up the Jira issue to GitHub PR automation",
    );
  });
});

describe("getAutomationIcon", () => {
  it("resolves the glyph an entry declares", () => {
    // Arrange / Act / Assert — the two entries that name one, each a
    // different slug, so the lookup is not passing by returning a constant.
    expect(getAutomationIcon(automationById("news-digest"))).toBe(Activity);
    expect(
      getAutomationIcon(automationById("github-agents-md-maintainer")),
    ).toBe(Bot);
  });

  it("returns null for an entry that declares none", () => {
    // Its integrations' logos identify it instead.
    expect(getAutomationIcon(automationById("github-pr-reviewer"))).toBeNull();
  });

  it("returns null for a slug this host has no artwork for", () => {
    // The catalog is published elsewhere, so an unknown slug must fall back to
    // the logos rather than render nothing.
    expect(
      getAutomationIcon({
        ...automationById("news-digest"),
        icon: "not-a-real-slug",
      } as unknown as Parameters<typeof getAutomationIcon>[0]),
    ).toBeNull();
  });
});

describe("resolveAutomationImpactStatement", () => {
  it("renders the phrase for the count, singular and plural", () => {
    // Arrange
    const automation = createInstalledAutomation("impact-valid");

    // Act / Assert
    expect(resolveAutomationImpactStatement(automation, 1)).toBe(
      "1 widget check completed",
    );
    expect(resolveAutomationImpactStatement(automation, 4)).toBe(
      "4 widget checks completed",
    );
  });

  it("says nothing for a zero or unknown count", () => {
    // Arrange — zero must never surface as zero-value copy, and null is an
    // older service whose lifetime count is unknowable.
    const automation = createInstalledAutomation("impact-valid");

    // Act / Assert
    expect(resolveAutomationImpactStatement(automation, 0)).toBeNull();
    expect(resolveAutomationImpactStatement(automation, null)).toBeNull();
  });

  it("says nothing without a template to join back to", () => {
    // Arrange / Act / Assert — no provenance at all, and provenance naming an
    // entry the catalog does not hold.
    expect(
      resolveAutomationImpactStatement(createInstalledAutomation(), 4),
    ).toBeNull();
    expect(
      resolveAutomationImpactStatement(
        createInstalledAutomation("not-in-catalog"),
        4,
      ),
    ).toBeNull();
  });

  it("says nothing when the entry declares no impact this host honors", () => {
    // Arrange / Act / Assert — an entry without the field, and one whose
    // basis this host does not know how to compute.
    expect(
      resolveAutomationImpactStatement(
        createInstalledAutomation("impact-absent"),
        4,
      ),
    ).toBeNull();
    expect(
      resolveAutomationImpactStatement(
        createInstalledAutomation("impact-unknown-basis"),
        4,
      ),
    ).toBeNull();
  });

  it("rejects a malformed impact declaration", () => {
    // Arrange / Act / Assert — markup, a plural without the count, and a
    // placeholder from another namespace each invalidate the declaration.
    for (const templateId of [
      "impact-markup",
      "impact-no-count",
      "impact-foreign-placeholder",
    ]) {
      expect(
        resolveAutomationImpactStatement(
          createInstalledAutomation(templateId),
          4,
        ),
      ).toBeNull();
    }
  });
});

/** A catalog entry with just what the hiding rule reads. */
function template(
  id: string,
  requires: Record<string, unknown>,
  setup?: Record<string, unknown>,
): RecommendedAutomation {
  return { id, requires, setup } as unknown as RecommendedAutomation;
}

const message = { message: "Used for the work." };
const githubPrReviewer = template(
  "github-reviewer",
  { integrations: {} },
  {
    form: {
      args: {
        repositories: { type: "repo-picker", provider: "github" },
      },
    },
  },
);
const jiraToBitbucket = template("jira-to-bitbucket", {
  integrations: { "atlassian-rovo": message, bitbucket: message },
});
const jiraToGithub = template("jira-to-github", {
  integrations: { "atlassian-rovo": message, github: message },
});
const optionalGitlab = template("optional-gitlab", {
  integrations: { gitlab: { ...message, required: false } },
});
const newsDigest = template("news-digest", { integrations: {} });
const dataCenterReviewer = template("bitbucket-dc-reviewer", {
  integrations: {},
  builtInIntegrations: ["bitbucket_data_center"],
});
const dataCenterIssueToPr = template("jira-dc-to-bitbucket-dc", {
  integrations: {},
  builtInIntegrations: ["jira_dc", "bitbucket_data_center"],
});
const catalog = [
  githubPrReviewer,
  jiraToBitbucket,
  jiraToGithub,
  optionalGitlab,
  newsDigest,
  dataCenterReviewer,
  dataCenterIssueToPr,
];

describe("getAutomationAvailability", () => {
  it("ranks what an Enterprise instance on Data Center does not report last, without hiding it", () => {
    // Arrange
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["bitbucket_data_center", "jira_dc"]),
    };

    // Act
    const { hidden, unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(catalog.filter((t) => hidden.has(t.id))).toEqual([]);
    expect(Object.fromEntries(unlikely)).toEqual({
      "github-reviewer": ["github"],
      "jira-to-bitbucket": ["bitbucket", "jira"],
      "jira-to-github": ["github", "jira"],
    });
  });

  it("names Jira Cloud on an Enterprise instance using Jira Data Center alongside a git provider the template needs", () => {
    // Arrange
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["github", "jira_dc"]),
    };

    // Act
    const { unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(unlikely.get("jira-to-github")).toEqual(["jira"]);
    expect(unlikely.has("github-reviewer")).toBe(false);
  });

  it("does not rank Jira Cloud templates down on an Enterprise instance that also enables Jira Cloud", () => {
    // Arrange
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["github", "jira", "jira_dc"]),
    };

    // Act
    const { unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(unlikely.has("jira-to-github")).toBe(false);
  });

  it("never lets an integration a template can start without count against it", () => {
    // Arrange
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["github"]),
    };

    // Act
    const { unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(unlikely.has("optional-gitlab")).toBe(false);
    expect(unlikely.has("news-digest")).toBe(false);
  });

  it("offers a Data Center template on an Enterprise instance missing part of it, ranked last", () => {
    // Arrange
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["bitbucket_data_center"]),
    };

    // Act
    const { hidden, unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(hidden.has("jira-dc-to-bitbucket-dc")).toBe(false);
    expect(unlikely.get("jira-dc-to-bitbucket-dc")).toEqual(["jira_dc"]);
    expect(unlikely.has("bitbucket-dc-reviewer")).toBe(false);
  });

  it("leaves OpenHands Cloud's catalog alone apart from hiding Data Center templates", () => {
    // Arrange
    const instance = {
      deploymentMode: "cloud",
      builtInIntegrations: new Set(["github", "gitlab", "bitbucket", "jira"]),
    };

    // Act
    const { hidden, unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect([...hidden]).toEqual([
      "bitbucket-dc-reviewer",
      "jira-dc-to-bitbucket-dc",
    ]);
    expect(unlikely.size).toBe(0);
  });

  it("hides only Data Center templates while the instance's integrations are unknown", () => {
    // Arrange — a local backend, or a cloud instance whose config is loading.
    const instance = { deploymentMode: null, builtInIntegrations: null };

    // Act
    const { hidden, unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect([...hidden]).toEqual([
      "bitbucket-dc-reviewer",
      "jira-dc-to-bitbucket-dc",
    ]);
    expect(unlikely.size).toBe(0);
  });

  it("keeps hiding what an Enterprise instance never offers", () => {
    // Arrange
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["github"]),
    };

    // Act
    const { hidden } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(hidden.has("custom-automation")).toBe(true);
  });
});

describe("rankByAvailability", () => {
  it("lists the templates an instance looks ready for first, each group in its existing order", () => {
    // Arrange
    const automations = [{ id: "a" }, { id: "b" }, { id: "c" }, { id: "d" }];
    const unlikely = new Map([
      ["a", ["github"]],
      ["c", ["jira"]],
    ]);

    // Act
    const ranked = rankByAvailability(automations, unlikely);

    // Assert
    expect(ranked.map(({ id }) => id)).toEqual(["b", "d", "a", "c"]);
  });
});
