import { describe, it, expect } from "vitest";
import { ActionEvent, SecurityRisk } from "#/types/agent-server/core";
import { resolveVisualizerBody } from "#/components/features/chat/tool-visualizers/dispatcher";
import { bashAction, grepObservation, terminalObservation } from "./test-utils";

describe("resolveVisualizerBody", () => {
  it("returns a body for a migrated action kind", () => {
    expect(resolveVisualizerBody(bashAction("ls"))).not.toBeNull();
  });

  it("returns a body for a migrated observation kind", () => {
    expect(
      resolveVisualizerBody(grepObservation({ pattern: "x" })),
    ).not.toBeNull();
  });

  it("returns a body for the terminal tool (shares the bash visualizer)", () => {
    expect(resolveVisualizerBody(terminalObservation("ok", 0))).not.toBeNull();
  });

  it("returns null for an unmigrated tool so the markdown fallback runs", () => {
    const thinkAction: ActionEvent = {
      id: "t1",
      timestamp: "2024-01-01T00:00:00Z",
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: { kind: "ThinkAction", thought: "hmm" },
      tool_name: "think",
      tool_call_id: "c1",
      tool_call: {
        id: "c1",
        type: "function",
        function: { name: "think", arguments: "{}" },
      },
      llm_response_id: "r1",
      security_risk: SecurityRisk.LOW,
    };
    expect(resolveVisualizerBody(thinkAction)).toBeNull();
  });
});
