import { describe, expect, it } from "vitest";
import type { Automation } from "#/types/automation";
import type { SkillInfo } from "#/types/settings";
import {
  CONVERSATION_OVERVIEW_PROJECT_SCOPE,
  countAutomationsForRepository,
  countSkillsForProject,
  filterAutomationsByProjectScope,
  filterSkillsByProjectScope,
  isAutomationForRepository,
  sortAutomationsByProjectRelevance,
  sortSkillsByProjectRelevance,
} from "#/utils/conversation-overview-project-scope";

function buildAutomation(
  overrides: Partial<Automation> & Pick<Automation, "id" | "name">,
): Automation {
  return {
    trigger: { type: "cron", schedule: "0 * * * *" },
    enabled: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    prompt: null,
    ...overrides,
  };
}

function buildSkill(
  overrides: Partial<SkillInfo> & Pick<SkillInfo, "name">,
): SkillInfo {
  return {
    type: "knowledge",
    source: "public",
    description: null,
    triggers: [],
    license: null,
    compatibility: null,
    metadata: null,
    allowed_tools: null,
    is_agentskills_format: true,
    disable_model_invocation: false,
    ...overrides,
  };
}

describe("conversation overview project scope", () => {
  it("matches automations to the open repository case-insensitively", () => {
    expect(
      isAutomationForRepository(
        buildAutomation({
          id: "1",
          name: "A",
          repository: "OpenHands/Agent-Canvas.git",
        }),
        "openhands/agent-canvas",
      ),
    ).toBe(true);
    expect(
      isAutomationForRepository(
        buildAutomation({ id: "2", name: "B", repository: "other/repo" }),
        "openhands/agent-canvas",
      ),
    ).toBe(false);
    expect(
      isAutomationForRepository(
        buildAutomation({ id: "3", name: "C" }),
        "openhands/agent-canvas",
      ),
    ).toBe(false);
  });

  it("filters and sorts automations for project vs all scopes", () => {
    const automations = [
      buildAutomation({ id: "1", name: "Zeta", repository: "other/repo" }),
      buildAutomation({
        id: "2",
        name: "Alpha",
        repository: "openhands/agent-canvas",
      }),
      buildAutomation({
        id: "3",
        name: "Beta",
        repository: "openhands/agent-canvas",
      }),
    ];

    expect(
      filterAutomationsByProjectScope(
        automations,
        CONVERSATION_OVERVIEW_PROJECT_SCOPE.project,
        "openhands/agent-canvas",
      ).map((automation) => automation.id),
    ).toEqual(["2", "3"]);

    expect(
      sortAutomationsByProjectRelevance(
        automations,
        "openhands/agent-canvas",
      ).map((automation) => automation.name),
    ).toEqual(["Alpha", "Beta", "Zeta"]);

    expect(
      countAutomationsForRepository(automations, "openhands/agent-canvas"),
    ).toBe(2);
  });

  it("filters and sorts skills with project skills first", () => {
    const skills = [
      buildSkill({ name: "public-z", source: "public" }),
      buildSkill({ name: "project-a", source: "project" }),
      buildSkill({ name: "personal-b", source: "user" }),
    ];

    expect(
      filterSkillsByProjectScope(
        skills,
        CONVERSATION_OVERVIEW_PROJECT_SCOPE.project,
        "/workspace/demo",
      ).map((skill) => skill.name),
    ).toEqual(["project-a"]);

    expect(
      sortSkillsByProjectRelevance(skills, "/workspace/demo").map(
        (skill) => skill.name,
      ),
    ).toEqual(["project-a", "personal-b", "public-z"]);

    expect(countSkillsForProject(skills, "/workspace/demo")).toBe(1);
  });
});
