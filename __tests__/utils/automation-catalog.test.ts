import { describe, expect, it, vi } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { Activity, Bot } from "lucide-react";
import {
  getAutomationIcon,
  getAutomationLaunchPrompt,
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
