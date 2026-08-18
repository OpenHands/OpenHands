import { describe, expect, it } from "vitest";
import type { RecommendedAutomation } from "@openhands/extensions/automations";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import {
  getAutomationLaunchPrompt,
  getIntegrationIds,
  getRequiredIntegrationIds,
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

describe("integration id readers tolerate malformed entries", () => {
  // A catalog entry that violates the type by omitting `requires` (e.g. an
  // older/mismatched @openhands/extensions build) must not crash the Automate
  // tab; it should simply contribute no integrations.
  const withoutRequires = {
    id: "broken-entry",
    name: "Broken entry",
  } as unknown as RecommendedAutomation;

  const withoutIntegrations = {
    id: "empty-requires",
    name: "Empty requires",
    requires: {},
  } as unknown as RecommendedAutomation;

  it("getIntegrationIds returns [] when requires/integrations are missing", () => {
    expect(getIntegrationIds(withoutRequires)).toEqual([]);
    expect(getIntegrationIds(withoutIntegrations)).toEqual([]);
  });

  it("getRequiredIntegrationIds returns [] when requires/integrations are missing", () => {
    expect(getRequiredIntegrationIds(withoutRequires)).toEqual([]);
    expect(getRequiredIntegrationIds(withoutIntegrations)).toEqual([]);
  });
});
