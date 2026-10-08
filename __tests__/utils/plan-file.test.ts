import { describe, expect, it } from "vitest";

import {
  AGENTS_TMP_DIR,
  DEEP_PLAN_PHASE_TAG_KEY,
  LOCAL_PLANNER_PARENT_TAG_KEY,
  buildPhasePlanPath,
  buildPlanPath,
  findPhasePlannerConversationId,
  isFallbackPlannerId,
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

describe("isFallbackPlannerId", () => {
  const parent = "parent-1";
  const parentTag = { [LOCAL_PLANNER_PARENT_TAG_KEY]: parent };

  it("accepts a store id whose fetched child is tagged for the same phase", () => {
    const planners = [
      makePlanner("req", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "requirements",
      }),
    ];
    expect(
      isFallbackPlannerId(
        planners,
        parent,
        "req",
        "requirements",
        "requirements",
      ),
    ).toBe(true);
  });

  it("rejects a store id whose child is tagged for a different phase", () => {
    // The store says "requirements" but the fetched child pins "database" —
    // the tag wins, so the mismatched planner is never adopted.
    const planners = [
      makePlanner("db", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "database",
      }),
    ];
    expect(
      isFallbackPlannerId(
        planners,
        parent,
        "db",
        "requirements",
        "requirements",
      ),
    ).toBe(false);
  });

  it("rejects a bare store id for a phase when no metadata recorded that phase", () => {
    // Right after a conversation switch the store id belongs to the previous
    // conversation and no child proves it; a deep-plan phase must be proven.
    expect(
      isFallbackPlannerId([], parent, "stale-id", "requirements", null),
    ).toBe(false);
  });

  it("trusts a plain store id only when nothing contradicts it", () => {
    expect(isFallbackPlannerId([], parent, "plain-id", null, null)).toBe(true);
    // A fetched phase planner proves the phase is already owned, so the stored
    // id cannot stand in for it.
    const planners = [
      makePlanner("req", {
        ...parentTag,
        [DEEP_PLAN_PHASE_TAG_KEY]: "requirements",
      }),
    ];
    expect(
      isFallbackPlannerId(
        planners,
        parent,
        "stale-id",
        "requirements",
        "requirements",
      ),
    ).toBe(false);
  });

  it("rejects when the parent or id is unknown", () => {
    expect(isFallbackPlannerId([], null, "id", null, null)).toBe(false);
    expect(isFallbackPlannerId([], parent, null, null, null)).toBe(false);
  });
});
