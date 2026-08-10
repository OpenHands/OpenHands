import { describe, expect, it } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { getAutomationLaunchPrompt } from "#/utils/automation-catalog";

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
