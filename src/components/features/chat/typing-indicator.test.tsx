import { describe, expect, it } from "vitest";
import { deriveLiveActivity } from "./typing-indicator";
import type { ActionEvent, ObservationEvent } from "#/types/agent-server/core";
import type { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import type { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";
import { SecurityRisk } from "#/types/agent-server/core/base/common";

const terminalAction: ActionEvent<ExecuteBashAction> = {
  id: "action-1",
  timestamp: "2026-08-21T12:00:00.000Z",
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "npm test",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "terminal",
  tool_call_id: "tool-1",
  tool_call: {
    id: "tool-1",
    type: "function",
    function: { name: "terminal", arguments: "{}" },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.LOW,
  summary: "Run tests",
};

const terminalObservation: ObservationEvent<ExecuteBashObservation> = {
  id: "observation-1",
  timestamp: "2026-08-21T12:00:01.000Z",
  source: "environment",
  tool_name: "terminal",
  tool_call_id: "tool-1",
  action_id: "action-1",
  observation: {
    kind: "ExecuteBashObservation",
    command: "npm test",
    content: [{ type: "text", text: "Tests passed" }],
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 123,
      username: "openhands",
      hostname: "sandbox",
      working_dir: "/workspace/project",
      py_interpreter_path: null,
      prefix: "",
      suffix: "",
    },
  },
};

describe("deriveLiveActivity", () => {
  it("does not invent Thinking when there are no live events", () => {
    expect(deriveLiveActivity([])).toBeNull();
  });

  it("does not invent Thinking after the latest action has resolved", () => {
    expect(
      deriveLiveActivity([terminalAction, terminalObservation]),
    ).toBeNull();
  });
});
