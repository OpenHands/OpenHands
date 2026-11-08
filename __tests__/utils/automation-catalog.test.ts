import { describe, expect, it, vi } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { Activity, Bot } from "lucide-react";
import type { RecommendedAutomation } from "@openhands/extensions/automations";
import {
  getAutomationIcon,
  getAutomationLaunchPrompt,
  getAutomationAvailability,
  getTemplateLogoIds,
  rankByAvailability,
  resolveAutomationImpactStatement,
  splitBySuggestion,
  SUGGESTED_TEMPLATE_LIMIT,
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
      impact: {
        basis: "run-counter",
        one: "1 widget",
        other: "{{count}} widgets",
      },
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
      impact: {
        basis: "completed-runs",
        one: "one check",
        other: "many checks",
      },
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

describe("getTemplateLogoIds", () => {
  it("names the repository provider of a template that declares no integration", () => {
    expect(getTemplateLogoIds(automationById("github-pr-reviewer"))).toEqual([
      "github",
    ]);
  });

  it("lists a provider once when it is both declared and picked", () => {
    expect(getTemplateLogoIds(automationById("github-repo-monitor"))).toEqual([
      "github",
    ]);
  });

  it("stands a built-in Data Center integration in with its product's logo", () => {
    // Arrange — no integration and no picker, like the Data Center templates.
    const entry = {
      ...automationById("news-digest"),
      requires: {
        integrations: {},
        builtInIntegrations: ["jira_dc", "bitbucket_data_center"],
      },
    } as unknown as RecommendedAutomation;

    // Act / Assert
    expect(getTemplateLogoIds(entry)).toEqual(["jira", "bitbucket"]);
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

  it("ranks a Data Center template last only when the instance uses the other kind", () => {
    // Arrange — Bitbucket Data Center with Jira Cloud rather than Jira Data Center.
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["bitbucket_data_center", "jira"]),
    };

    // Act
    const { hidden, unlikely, suggested } = getAutomationAvailability(
      instance,
      catalog,
    );

    // Assert
    expect(hidden.has("jira-dc-to-bitbucket-dc")).toBe(false);
    expect(unlikely.get("jira-dc-to-bitbucket-dc")).toEqual(["jira_dc"]);
    expect(suggested.get("bitbucket-dc-reviewer")).toEqual([
      "bitbucket_data_center",
    ]);
  });

  it("suggests a Data Center template on an instance reporting part of it, without evidence against the rest", () => {
    // Arrange — Bitbucket Data Center, no Jira of either kind reported.
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["bitbucket_data_center"]),
    };

    // Act
    const { unlikely, suggested } = getAutomationAvailability(
      instance,
      catalog,
    );

    // Assert
    expect(unlikely.has("jira-dc-to-bitbucket-dc")).toBe(false);
    expect(suggested.has("jira-dc-to-bitbucket-dc")).toBe(true);
  });

  it("leaves OpenHands Cloud's catalog alone, Data Center templates included", () => {
    // Arrange
    const instance = {
      deploymentMode: "cloud",
      builtInIntegrations: new Set(["github", "gitlab", "bitbucket", "jira"]),
    };

    // Act
    const { hidden, unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(hidden.size).toBe(0);
    expect(unlikely.size).toBe(0);
  });

  it("hides nothing while the instance's integrations are unknown", () => {
    // Arrange — a local backend, or a cloud instance whose config is loading.
    const instance = { deploymentMode: null, builtInIntegrations: null };

    // Act
    const { hidden, unlikely } = getAutomationAvailability(instance, catalog);

    // Assert
    expect(hidden.size).toBe(0);
    expect(unlikely.size).toBe(0);
  });

  it("keeps every template in its default place on an Enterprise instance that reports no git provider", () => {
    // Arrange — sign-up by password, nothing else set up.
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set<string>(),
    };

    // Act
    const { unlikely, suggested } = getAutomationAvailability(
      instance,
      catalog,
    );

    // Assert
    expect(unlikely.has("github-reviewer")).toBe(false);
    expect(unlikely.has("jira-to-github")).toBe(false);
    expect(unlikely.size).toBe(0);
    expect(suggested.size).toBe(0);
  });

  it("suggests the templates using an integration the user connected, on any backend", () => {
    // Arrange — a local backend with the Atlassian Rovo MCP server installed.
    const instance = {
      deploymentMode: null,
      builtInIntegrations: null,
      connectedIntegrations: new Set(["atlassian-rovo"]),
    };

    // Act
    const { suggested, unlikely } = getAutomationAvailability(
      instance,
      catalog,
    );

    // Assert
    expect(Object.fromEntries(suggested)).toEqual({
      "jira-to-bitbucket": ["atlassian-rovo"],
      "jira-to-github": ["atlassian-rovo"],
    });
    expect(unlikely.size).toBe(0);
  });

  it("does not rank a template last for an integration the user connected themselves", () => {
    // Arrange — Bitbucket Data Center instance, user installed the GitHub MCP.
    const instance = {
      deploymentMode: "self_hosted",
      builtInIntegrations: new Set(["bitbucket_data_center"]),
      connectedIntegrations: new Set(["github"]),
    };

    // Act
    const { suggested, unlikely } = getAutomationAvailability(
      instance,
      catalog,
    );

    // Assert
    expect(unlikely.has("github-reviewer")).toBe(false);
    expect(suggested.get("github-reviewer")).toEqual(["github"]);
    expect(unlikely.get("jira-to-bitbucket")).toEqual(["bitbucket"]);
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

describe("splitBySuggestion", () => {
  it("leads with the matched templates, fills with featured ones and stops at the limit", () => {
    // Arrange
    const automations = Array.from({ length: 12 }, (_, i) => ({
      id: `t${i}`,
    }));
    const suggested = new Map(
      ["t9", "t10", "t11"].map((id) => [id, ["github"]] as const),
    );

    // Act
    const groups = splitBySuggestion(automations, () => true, { suggested });

    // Assert
    expect(groups.suggested).toHaveLength(SUGGESTED_TEMPLATE_LIMIT);
    expect(groups.suggested.map(({ id }) => id)).toEqual([
      "t9",
      "t10",
      "t11",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
    ]);
    expect(groups.more.map(({ id }) => id)).toEqual(["t5", "t6", "t7", "t8"]);
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
