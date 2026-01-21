import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ReasoningContent } from "#/components/v1/chat/event-message-components/reasoning-content";
import { ActionEvent } from "#/types/v1/core";
import { SecurityRisk } from "#/types/v1/core/base/common";

describe("ReasoningContent", () => {
  it("should not render anything when there is no reasoning content or thinking blocks", () => {
    const mockActionEvent: ActionEvent = {
      id: "test-event",
      timestamp: new Date().toISOString(),
      source: "agent",
      action: {
        kind: "ExecuteBashAction",
        command: "echo hello",
        is_input: false,
        timeout: null,
        reset: false,
      },
      thought: [],
      reasoning_content: null,
      thinking_blocks: [],
      tool_name: "ExecuteBashAction",
      tool_call_id: "test-tool-call",
      tool_call: {
        id: "test-tool-call",
        function: {
          name: "ExecuteBashAction",
          arguments: '{"command": "echo hello"}',
        },
        type: "function",
      },
      llm_response_id: "test-response-id",
      security_risk: SecurityRisk.LOW,
    };

    const { container } = renderWithProviders(
      <ReasoningContent event={mockActionEvent} />,
    );

    expect(container.firstChild).toBeNull();
  });

  it("should render reasoning content when available", () => {
    const mockActionEvent: ActionEvent = {
      id: "test-event",
      timestamp: new Date().toISOString(),
      source: "agent",
      action: {
        kind: "ExecuteBashAction",
        command: "echo hello",
        is_input: false,
        timeout: null,
        reset: false,
      },
      thought: [],
      reasoning_content: "This is the reasoning content from the AI model.",
      thinking_blocks: [],
      tool_name: "ExecuteBashAction",
      tool_call_id: "test-tool-call",
      tool_call: {
        id: "test-tool-call",
        function: {
          name: "ExecuteBashAction",
          arguments: '{"command": "echo hello"}',
        },
        type: "function",
      },
      llm_response_id: "test-response-id",
      security_risk: SecurityRisk.LOW,
    };

    renderWithProviders(<ReasoningContent event={mockActionEvent} />);

    expect(screen.getByText("Reasoning")).toBeInTheDocument();
    expect(
      screen.queryByText("This is the reasoning content"),
    ).not.toBeInTheDocument();
  });

  it("should render thinking blocks when available", () => {
    const mockActionEvent: ActionEvent = {
      id: "test-event",
      timestamp: new Date().toISOString(),
      source: "agent",
      action: {
        kind: "ExecuteBashAction",
        command: "echo hello",
        is_input: false,
        timeout: null,
        reset: false,
      },
      thought: [],
      reasoning_content: null,
      thinking_blocks: [
        {
          type: "thinking",
          thinking: "This is the first thinking block content.",
          signature: "test-signature-1",
        },
        {
          type: "redacted_thinking",
          data: "redacted-data",
        },
      ],
      tool_name: "ExecuteBashAction",
      tool_call_id: "test-tool-call",
      tool_call: {
        id: "test-tool-call",
        function: {
          name: "ExecuteBashAction",
          arguments: '{"command": "echo hello"}',
        },
        type: "function",
      },
      llm_response_id: "test-response-id",
      security_risk: SecurityRisk.LOW,
    };

    renderWithProviders(<ReasoningContent event={mockActionEvent} />);

    expect(screen.getByText("Reasoning")).toBeInTheDocument();
    expect(screen.queryByText("Thinking Block 1")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Redacted Thinking Block 2"),
    ).not.toBeInTheDocument();
  });

  it("should expand and show content when clicked", async () => {
    const mockActionEvent: ActionEvent = {
      id: "test-event",
      timestamp: new Date().toISOString(),
      source: "agent",
      action: {
        kind: "ExecuteBashAction",
        command: "echo hello",
        is_input: false,
        timeout: null,
        reset: false,
      },
      thought: [],
      reasoning_content: "This is the reasoning content from the AI model.",
      thinking_blocks: [
        {
          type: "thinking",
          thinking: "This is thinking block content.",
          signature: "test-signature",
        },
      ],
      tool_name: "ExecuteBashAction",
      tool_call_id: "test-tool-call",
      tool_call: {
        id: "test-tool-call",
        function: {
          name: "ExecuteBashAction",
          arguments: '{"command": "echo hello"}',
        },
        type: "function",
      },
      llm_response_id: "test-response-id",
      security_risk: SecurityRisk.LOW,
    };

    const user = userEvent.setup();
    renderWithProviders(<ReasoningContent event={mockActionEvent} />);

    await user.click(screen.getByText("Reasoning"));

    expect(screen.getByText("Reasoning Content")).toBeInTheDocument();
    expect(
      screen.getByText("This is the reasoning content from the AI model."),
    ).toBeInTheDocument();
    expect(screen.getByText("Thinking Blocks")).toBeInTheDocument();
    expect(screen.getByText("Thinking Block 1")).toBeInTheDocument();
    expect(screen.getByText("This is thinking block content.")).toBeInTheDocument();
  });

  it("should show redacted thinking content correctly", async () => {
    const mockActionEvent: ActionEvent = {
      id: "test-event",
      timestamp: new Date().toISOString(),
      source: "agent",
      action: {
        kind: "ExecuteBashAction",
        command: "echo hello",
        is_input: false,
        timeout: null,
        reset: false,
      },
      thought: [],
      reasoning_content: null,
      thinking_blocks: [
        {
          type: "redacted_thinking",
          data: "redacted-data",
        },
      ],
      tool_name: "ExecuteBashAction",
      tool_call_id: "test-tool-call",
      tool_call: {
        id: "test-tool-call",
        function: {
          name: "ExecuteBashAction",
          arguments: '{"command": "echo hello"}',
        },
        type: "function",
      },
      llm_response_id: "test-response-id",
      security_risk: SecurityRisk.LOW,
    };

    const user = userEvent.setup();
    renderWithProviders(<ReasoningContent event={mockActionEvent} />);

    await user.click(screen.getByText("Reasoning"));

    expect(screen.getByText("[Redacted thinking content]")).toBeInTheDocument();
  });

  it("should render both reasoning content and thinking blocks", async () => {
    const mockActionEvent: ActionEvent = {
      id: "test-event",
      timestamp: new Date().toISOString(),
      source: "agent",
      action: {
        kind: "ExecuteBashAction",
        command: "echo hello",
        is_input: false,
        timeout: null,
        reset: false,
      },
      thought: [],
      reasoning_content: "Main reasoning content",
      thinking_blocks: [
        {
          type: "thinking",
          thinking: "Detailed thinking",
          signature: "test-signature",
        },
      ],
      tool_name: "ExecuteBashAction",
      tool_call_id: "test-tool-call",
      tool_call: {
        id: "test-tool-call",
        function: {
          name: "ExecuteBashAction",
          arguments: '{"command": "echo hello"}',
        },
        type: "function",
      },
      llm_response_id: "test-response-id",
      security_risk: SecurityRisk.LOW,
    };

    const user = userEvent.setup();
    renderWithProviders(<ReasoningContent event={mockActionEvent} />);

    await user.click(screen.getByText("Reasoning"));

    expect(screen.getByText("Reasoning Content")).toBeInTheDocument();
    expect(screen.getByText("Main reasoning content")).toBeInTheDocument();
    expect(screen.getByText("Thinking Blocks")).toBeInTheDocument();
    expect(screen.getByText("Thinking Block 1")).toBeInTheDocument();
  });
});
