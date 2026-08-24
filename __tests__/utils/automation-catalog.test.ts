import { describe, expect, it } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { Activity, Bot } from "lucide-react";
import {
  getAutomationIcon,
  getAutomationLaunchPrompt,
} from "#/utils/automation-catalog";

const automationById = (id: string) =>
  AUTOMATION_CATALOG.find((automation) => automation.id === id)!;

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
