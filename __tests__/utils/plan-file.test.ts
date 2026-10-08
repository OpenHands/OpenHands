import { describe, expect, it } from "vitest";

import {
  AGENTS_TMP_DIR,
  DEEP_PLAN_PHASE_TAG_KEY,
  LOCAL_PLANNER_PARENT_TAG_KEY,
  buildPhasePlanPath,
  buildPlanPath,
  findPhasePlannerConversationId,
  plannerPhaseOf,
} from "#/utils/plan-file";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

/** Minimal planner conversation carrying only the tags under test. */
function makePlanner(
  id: string,
  tags: Record<string, string> | null,
): AppConversation {
  return { id, tags } as AppConversation;
}

describe("buildPhasePlanPath", () => {
  it("pins each phase to its own document under .agents_tmp", () => {
    expect(buildPhasePlanPath("/workspace/project", "requirements")).toBe(
      `/workspace/project/${AGENTS_TMP_DIR}/requirements.md`,
    );
    expect(buildPhasePlanPath("/workspace/project", "database")).toBe(
      `/workspace/project/${AGENTS_TMP_DIR}/database-design.md`,
    );
  });

  it("falls back to PLAN.md for phases without an output document", () => {
    expect(buildPhasePlanPath("/workspace/project", "analysis")).toBe(
      buildPlanPath("/workspace/project"),
    );
    expect(buildPhasePlanPath("/workspace/project", "implementation")).toBe(
      buildPlanPath("/workspace/project"),
    );
  });

  it("normalizes a trailing slash so the path is not doubled", () => {
    expect(buildPhasePlanPath("/workspace/project/", "tasks")).toBe(
      `/workspace/project/${AGENTS_TMP_DIR}/tasks.md`,
    );
  });
});

describe("plannerPhaseOf", () => {
  it("reads the phase from the phase tag", () => {
    expect(
      plannerPhaseOf(
        makePlanner("p1", { [DEEP_PLAN_PHASE_TAG_KEY]: "backend" }),
      ),
    ).toBe("backend");
  });

  it("returns null for a plain planner or an unknown phase tag", () => {
    expect(plannerPhaseOf(makePlanner("p1", {}))).toBeNull();
    expect(plannerPhaseOf(makePlanner("p1", null))).toBeNull();
    expect(
      plannerPhaseOf(
        makePlanner("p1", { [DEEP_PLAN_PHASE_TAG_KEY]: "not-a-phase" }),
      ),
    ).toBeNull();
  });
});

describe("findPhasePlannerConversationId", () => {
  const parent = "parent-1";
  const parentTag = { [LOCAL_PLANNER_PARENT_TAG_KEY]: parent };

  it("returns the planner tagged with the requested phase", () => {
    const planners = [
      makePlanner("plain", parentTag),
      makePlanner("req", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "requirements",
      }),
      makePlanner("db", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "database",
      }),
    ];
    expect(findPhasePlannerConversationId(planners, parent, "database")).toBe(
      "db",
    );
  });

  it("returns the untagged planner when no phase is requested, never a phase one", () => {
    const planners = [
      makePlanner("req", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "requirements",
      }),
      makePlanner("plain", parentTag),
    ];
    expect(findPhasePlannerConversationId(planners, parent, null)).toBe(
      "plain",
    );
  });

  it("returns null when the requested phase has no planner yet", () => {
    const planners = [
      makePlanner("req", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "requirements",
      }),
    ];
    expect(
      findPhasePlannerConversationId(planners, parent, "frontend"),
    ).toBeNull();
  });

  it("ignores a planner tagged for a different parent", () => {
    const planners = [
      makePlanner("other", {
        [LOCAL_PLANNER_PARENT_TAG_KEY]: "other-parent",
        [DEEP_PLAN_PHASE_TAG_KEY]: "requirements",
      }),
    ];
    expect(
      findPhasePlannerConversationId(planners, parent, "requirements"),
    ).toBeNull();
  });
});
