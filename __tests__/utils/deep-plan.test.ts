import { describe, expect, it } from "vitest";
import {
  matchDeepPlanDocumentFile,
  nextDeepPlanPhase,
  upstreamPhasesOf,
} from "#/utils/deep-plan";

describe("matchDeepPlanDocumentFile", () => {
  it("maps a phase output file to its phase by basename", () => {
    expect(matchDeepPlanDocumentFile("/workspace/requirements.md")).toBe(
      "requirements",
    );
    expect(matchDeepPlanDocumentFile("/workspace/database-design.md")).toBe(
      "database",
    );
    expect(matchDeepPlanDocumentFile("backend-design.md")).toBe("backend");
    expect(matchDeepPlanDocumentFile("/workspace/frontend-design.md")).toBe(
      "frontend",
    );
    expect(matchDeepPlanDocumentFile("/workspace/tasks.md")).toBe("tasks");
  });

  it("matches regardless of case and Windows separators", () => {
    expect(matchDeepPlanDocumentFile("C:\\workspace\\REQUIREMENTS.MD")).toBe(
      "requirements",
    );
  });

  it("does not match PLAN.md or an unrelated file", () => {
    // PLAN.md belongs to plain Plan mode; the deep-plan chain has no document
    // for the conversation-only analysis/implementation phases.
    expect(matchDeepPlanDocumentFile("/workspace/PLAN.md")).toBeNull();
    expect(matchDeepPlanDocumentFile("/workspace/notes.md")).toBeNull();
    expect(matchDeepPlanDocumentFile(null)).toBeNull();
    expect(matchDeepPlanDocumentFile(undefined)).toBeNull();
  });
});

describe("phase ordering", () => {
  it("treats only earlier phases as upstream", () => {
    expect(upstreamPhasesOf("database")).toEqual(["analysis", "requirements"]);
  });

  it("has no successor after the final phase", () => {
    expect(nextDeepPlanPhase("implementation")).toBeNull();
    expect(nextDeepPlanPhase("analysis")).toBe("requirements");
  });
});
