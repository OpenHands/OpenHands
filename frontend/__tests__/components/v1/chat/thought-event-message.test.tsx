import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ThoughtEventMessage } from "#/components/v1/chat/event-message-components/thought-event-message";
import { ActionEvent } from "#/types/v1/core";
import { SecurityRisk } from "#/types/v1/core/base/common";

describe("ThoughtEventMessage", () => {
  it("should not render anything when there is no thought content, reasoning content, or thinking blocks", () => {
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
      <ThoughtEventMessage event={mockActionEvent} />,
    );

    expect(container.firstChild).toBeNull();
  });

  it("should render thought content when available", () => {
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
      thought: [
        {
          type: "text",
          text: "I need to run a command to help the user.",
        },
      ],
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

    renderWithProviders(<ThoughtEventMessage event={mockActionEvent} />);

    expect(
      screen.getByText("I need to run a command to help the user."),
    ).toBeInTheDocument();
  });

  it("should render reasoning content when there is no thought but reasoning exists", () => {
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
      reasoning_content: "This is reasoning from an AI model.",
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

    renderWithProviders(<ThoughtEventMessage event={mockActionEvent} />);

    expect(screen.getByText("Reasoning")).toBeInTheDocument();
  });

  it("should render both thought and reasoning content", () => {
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
      thought: [
        {
          type: "text",
          text: "I need to run a command to help the user.",
        },
      ],
      reasoning_content: "This is reasoning from an AI model.",
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

    renderWithProviders(<ThoughtEventMessage event={mockActionEvent} />);

    expect(
      screen.getByText("I need to run a command to help the user."),
    ).toBeInTheDocument();
    expect(screen.getByText("Reasoning")).toBeInTheDocument();
  });

  it("should render thinking blocks when there is no thought but thinking blocks exist", () => {
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
          thinking: "This is thinking content.",
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

    renderWithProviders(<ThoughtEventMessage event={mockActionEvent} />);

    expect(screen.getByText("Reasoning")).toBeInTheDocument();
  });

  it("should handle empty reasoning content gracefully", () => {
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
      thought: [
        {
          type: "text",
          text: "I need to run a command to help the user.",
        },
      ],
      reasoning_content: "",
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

    renderWithProviders(<ThoughtEventMessage event={mockActionEvent} />);

    expect(
      screen.getByText("I need to run a command to help the user."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Reasoning")).not.toBeInTheDocument();
  });

  it("should handle whitespace-only reasoning content", () => {
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
      thought: [
        {
          type: "text",
          text: "I need to run a command to help the user.",
        },
      ],
      reasoning_content: "   ",
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

    renderWithProviders(<ThoughtEventMessage event={mockActionEvent} />);

    expect(
      screen.getByText("I need to run a command to help the user."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Reasoning")).not.toBeInTheDocument();
  });
});
