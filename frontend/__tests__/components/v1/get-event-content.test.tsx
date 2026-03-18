import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getEventContent } from "#/components/v1/chat";
import { ActionEvent, ObservationEvent, SecurityRisk } from "#/types/v1/core";

const terminalActionEvent: ActionEvent = {
  id: "action-1",
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "Checking repository status." }],
  thinking_blocks: [],
  action: {
    kind: "TerminalAction",
    command: "git status",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "terminal",
  tool_call_id: "tool-1",
  tool_call: {
    id: "tool-1",
    type: "function",
    function: {
      name: "terminal",
      arguments: '{"command":"git status"}',
    },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.LOW,
  summary: "Check repository status",
};

const terminalObservationEvent: ObservationEvent = {
  id: "obs-1",
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "terminal",
  tool_call_id: "tool-1",
  action_id: "action-1",
  observation: {
    kind: "TerminalObservation",
    content: [{ type: "text", text: "On branch main" }],
    command: "git status",
    exit_code: 0,
    is_error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 1,
      username: "openhands",
      hostname: "sandbox",
      prefix: "",
      suffix: "",
      working_dir: "/workspace/project/OpenHands",
      py_interpreter_path: null,
    },
  },
};

describe("getEventContent", () => {
  it("uses the action summary as the full action title", () => {
    const { title } = getEventContent(terminalActionEvent);

    render(<>{title}</>);

    expect(screen.getByText("Check repository status")).toBeInTheDocument();
    expect(screen.queryByText("$ git status")).not.toBeInTheDocument();
  });

  it("falls back to command-based title when summary is missing", () => {
    const actionWithoutSummary = { ...terminalActionEvent, summary: undefined };
    const { title } = getEventContent(actionWithoutSummary);

    render(<>{title}</>);

    // Without i18n loaded, the translation key renders as the raw key
    expect(screen.getByText("ACTION_MESSAGE$RUN")).toBeInTheDocument();
    expect(
      screen.queryByText("Check repository status"),
    ).not.toBeInTheDocument();
  });

  it("reuses the action summary as the full paired observation title", () => {
    const { title } = getEventContent(
      terminalObservationEvent,
      terminalActionEvent,
    );

    render(<>{title}</>);

    expect(screen.getByText("Check repository status")).toBeInTheDocument();
    expect(screen.queryByText("$ git status")).not.toBeInTheDocument();
  });
});
