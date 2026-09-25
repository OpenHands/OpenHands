import { describe, expect, it } from "vitest";
import {
  buildRcaContextPrompt,
  composeTaskWithRcaContext,
  decodeRcaParam,
  parseRcaContext,
  type RcaContext,
} from "#/utils/rca-context";
import { buildRcaLaunchPath } from "#/utils/rca-launch-url";

const fullRca: RcaContext = {
  summary: "Connection pool exhaustion under load",
  evidence: ["pg pool max=5 reached", "502s spiked at 14:03 UTC"],
  suspectedComponents: ["db-pool", "api-gateway"],
  suspectedFiles: ["src/db/pool.ts", "src/api/gateway.ts"],
  recommendedAction: "Raise the pool limit and add backpressure",
  source: "holmesgpt",
};

describe("parseRcaContext", () => {
  it("parses a camelCase payload into an RcaContext", () => {
    expect(parseRcaContext(fullRca)).toEqual(fullRca);
  });

  it("accepts snake_case keys from external systems", () => {
    const parsed = parseRcaContext({
      summary: "Pool exhausted",
      evidence: "single evidence line",
      suspected_components: ["db-pool"],
      suspected_files: ["src/db/pool.ts"],
      recommended_action: "Increase pool size",
      source: "holmesgpt",
    });

    expect(parsed).toEqual({
      summary: "Pool exhausted",
      evidence: ["single evidence line"],
      suspectedComponents: ["db-pool"],
      suspectedFiles: ["src/db/pool.ts"],
      recommendedAction: "Increase pool size",
      source: "holmesgpt",
    });
  });

  it("keeps only a summary when optional fields are absent", () => {
    expect(parseRcaContext({ summary: "Just a summary" })).toEqual({
      summary: "Just a summary",
      evidence: [],
      suspectedComponents: [],
      suspectedFiles: [],
      recommendedAction: undefined,
      source: undefined,
    });
  });

  it("drops non-string and blank list entries", () => {
    const parsed = parseRcaContext({
      summary: "s",
      evidence: ["keep", 42, "   ", null],
      suspected_files: [" ", "src/a.ts"],
    });

    expect(parsed?.evidence).toEqual(["keep"]);
    expect(parsed?.suspectedFiles).toEqual(["src/a.ts"]);
  });

  it.each([
    null,
    undefined,
    "summary text",
    42,
    ["array"],
    {},
    { summary: "   " },
    { summary: 42 },
  ])("returns null for unusable input %j", (input) => {
    expect(parseRcaContext(input)).toBeNull();
  });
});

describe("decodeRcaParam", () => {
  it("round-trips a base64-encoded RcaContext", () => {
    const decoded = decodeRcaParam(btoa(JSON.stringify(fullRca)));
    expect(decoded).toEqual(fullRca);
  });

  it("returns null for malformed base64 or JSON", () => {
    expect(decodeRcaParam("not-valid-base64!!!")).toBeNull();
    expect(decodeRcaParam(btoa("not json"))).toBeNull();
    expect(
      decodeRcaParam(btoa(JSON.stringify({ noSummary: true }))),
    ).toBeNull();
  });
});

describe("buildRcaContextPrompt", () => {
  it("renders every section with the source attribution", () => {
    const prompt = buildRcaContextPrompt(fullRca);

    expect(prompt).toContain("## Root Cause Analysis Context");
    expect(prompt).toContain("provided by holmesgpt");
    expect(prompt).toContain("### Summary");
    expect(prompt).toContain("Connection pool exhaustion under load");
    expect(prompt).toContain("### Evidence");
    expect(prompt).toContain("- pg pool max=5 reached");
    expect(prompt).toContain("### Suspected Components");
    expect(prompt).toContain("- api-gateway");
    expect(prompt).toContain("### Suspected Files");
    expect(prompt).toContain("- src/db/pool.ts");
    expect(prompt).toContain("### Recommended Action");
    expect(prompt).toContain("Raise the pool limit");
  });

  it("instructs the agent to verify the analysis against the repository first", () => {
    const prompt = buildRcaContextPrompt(fullRca);
    expect(prompt).toMatch(/analyze the repository/i);
    expect(prompt).toMatch(/verify the evidence|verify/i);
  });

  it("omits empty optional sections and falls back to a generic attribution", () => {
    const prompt = buildRcaContextPrompt({
      summary: "Only a summary",
      evidence: [],
      suspectedComponents: [],
      suspectedFiles: [],
    });

    expect(prompt).toContain("provided by an external system");
    expect(prompt).not.toContain("### Evidence");
    expect(prompt).not.toContain("### Suspected Components");
    expect(prompt).not.toContain("### Suspected Files");
    expect(prompt).not.toContain("### Recommended Action");
  });
});

describe("composeTaskWithRcaContext", () => {
  it("places the RCA block ahead of the task text", () => {
    const composed = composeTaskWithRcaContext("Fix the bug", fullRca);
    expect(composed).toBeDefined();
    const rcaIndex = composed!.indexOf("## Root Cause Analysis Context");
    const taskIndex = composed!.indexOf("Fix the bug");
    expect(rcaIndex).toBeGreaterThanOrEqual(0);
    expect(taskIndex).toBeGreaterThan(rcaIndex);
  });

  it("returns just the RCA block when there is no task text", () => {
    const composed = composeTaskWithRcaContext(undefined, fullRca);
    expect(composed).toContain("## Root Cause Analysis Context");
  });

  it("returns the task text unchanged when no RCA is attached", () => {
    expect(composeTaskWithRcaContext("Fix the bug", null)).toBe("Fix the bug");
  });

  it("returns undefined when both inputs are empty", () => {
    expect(composeTaskWithRcaContext(undefined, null)).toBeUndefined();
    expect(composeTaskWithRcaContext("   ", undefined)).toBeUndefined();
  });
});

describe("buildRcaLaunchPath", () => {
  it("produces a launch path whose params decode back to the inputs", () => {
    const path = buildRcaLaunchPath(
      fullRca,
      [{ source: "github:owner/repo", ref: "main", repo_path: null }],
      "Investigate this",
    );
    const params = new URLSearchParams(path.split("?")[1]);

    expect(path.startsWith("/launch?")).toBe(true);
    expect(decodeRcaParam(params.get("rca")!)).toEqual(fullRca);
    expect(JSON.parse(atob(params.get("plugins")!))).toEqual([
      { source: "github:owner/repo", ref: "main", repo_path: null },
    ]);
    expect(params.get("message")).toBe("Investigate this");
  });

  it("omits the message param when none is given", () => {
    const path = buildRcaLaunchPath(fullRca, [{ source: "github:owner/repo" }]);
    expect(new URLSearchParams(path.split("?")[1]).get("message")).toBeNull();
  });
});
