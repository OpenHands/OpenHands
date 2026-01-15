import { toSocketIo } from "@mswjs/socket.io-binding";
import { AgentState } from "#/types/agent-state";
import {
  AssistantMessageAction,
  UserMessageAction,
} from "#/types/core/actions";
import { AgentStateChangeObservation } from "#/types/core/observations";
import { MessageEvent } from "#/types/v1/core";
import { AgentErrorEvent } from "#/types/v1/core/events/observation-event";
import { ConversationErrorEvent } from "#/types/v1/core/events/conversation-state-event";
import { MockSessionMessaage } from "./session-history.mock";

export const generateAgentStateChangeObservation = (
  state: AgentState,
): AgentStateChangeObservation => ({
  id: 1,
  cause: 0,
  message: "AGENT_STATE_CHANGE_MESSAGE",
  source: "agent",
  timestamp: new Date().toISOString(),
  observation: "agent_state_changed",
  content: "AGENT_STATE_CHANGE_MESSAGE",
  extras: { agent_state: state },
});

export const generateAssistantMessageAction = (
  message: string,
): AssistantMessageAction => ({
  id: 2,
  message: "USER_MESSAGE",
  source: "agent",
  timestamp: new Date().toISOString(),
  action: "message",
  args: {
    thought: message,
    image_urls: [],
    file_urls: [],
    wait_for_response: false,
  },
});

export const generateUserMessageAction = (
  message: string,
): UserMessageAction => ({
  id: 3,
  message: "USER_MESSAGE",
  source: "user",
  timestamp: new Date().toISOString(),
  action: "message",
  args: {
    content: message,
    image_urls: [],
    file_urls: [],
  },
});

export const emitAssistantMessage = (
  io: ReturnType<typeof toSocketIo>,
  message: string,
) => io.client.emit("oh_event", generateAssistantMessageAction(message));

export const emitUserMessage = (
  io: ReturnType<typeof toSocketIo>,
  message: string,
) => io.client.emit("oh_event", generateUserMessageAction(message));

export const emitMessages = (
  io: ReturnType<typeof toSocketIo>,
  messages: MockSessionMessaage[],
) => {
  messages.forEach(({ source, message }) => {
    if (source === "assistant") {
      emitAssistantMessage(io, message);
    } else {
      emitUserMessage(io, message);
    }
  });
};

// V1 Event Mock Factories for WebSocket Testing

/**
 * Creates a mock MessageEvent for testing WebSocket event handling
 */
export const createMockMessageEvent = (
  overrides: Partial<MessageEvent> = {},
): MessageEvent => ({
  id: "test-event-123",
  timestamp: new Date().toISOString(),
  source: "agent",
  llm_message: {
    role: "assistant",
    content: [{ type: "text", text: "Hello from agent" }],
  },
  activated_microagents: [],
  extended_content: [],
  ...overrides,
});

/**
 * Creates a mock user MessageEvent for testing WebSocket event handling
 */
export const createMockUserMessageEvent = (
  overrides: Partial<MessageEvent> = {},
): MessageEvent => ({
  id: "user-message-123",
  timestamp: new Date().toISOString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Hello from user" }],
  },
  activated_microagents: [],
  extended_content: [],
  ...overrides,
});

/**
 * Creates a mock AgentErrorEvent for testing error handling
 */
export const createMockAgentErrorEvent = (
  overrides: Partial<AgentErrorEvent> = {},
): AgentErrorEvent => ({
  id: "error-event-123",
  timestamp: new Date().toISOString(),
  source: "agent",
  tool_name: "str_replace_editor",
  tool_call_id: "tool-call-456",
  error: "Failed to execute command: Permission denied",
  ...overrides,
});

/**
 * Creates a mock ExecuteBashAction event for testing terminal command handling
 */
export const createMockExecuteBashActionEvent = (
  command: string = "ls -la",
) => ({
  id: "bash-action-123",
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "Executing bash command" }],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command,
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "ExecuteBashAction",
  tool_call_id: "bash-call-456",
  tool_call: {
    id: "bash-call-456",
    type: "function",
    function: {
      name: "ExecuteBashAction",
      arguments: JSON.stringify({ command }),
    },
  },
  llm_response_id: "llm-response-789",
  security_risk: { level: "low" },
});

/**
 * Creates a mock ExecuteBashObservation event for testing terminal output handling
 */
export const createMockExecuteBashObservationEvent = (
  content: string = "total 24\ndrwxr-xr-x  5 user  staff  160 Jan 10 12:00 .",
  command: string = "ls -la",
) => ({
  id: "bash-obs-123",
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "ExecuteBashAction",
  tool_call_id: "bash-call-456",
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: content }],
    command,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: { cwd: "/home/user" },
  },
  action_id: "bash-action-123",
});

/**
 * Creates a mock BrowserObservation event for testing browser state handling
 */
export const createMockBrowserObservationEvent = (
  screenshotData: string | null = "base64-screenshot-data",
  output: string = "Browser action completed",
  error: string | null = null,
) => ({
  id: "browser-obs-123",
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "browser_navigate",
  tool_call_id: "browser-call-456",
  observation: {
    kind: "BrowserObservation",
    output,
    error,
    screenshot_data: screenshotData,
  },
  action_id: "browser-action-123",
});

/**
 * Creates a mock BrowserNavigateAction event for testing browser URL extraction
 */
export const createMockBrowserNavigateActionEvent = (
  url: string = "https://example.com",
) => ({
  id: "browser-action-123",
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "Navigating to URL" }],
  thinking_blocks: [],
  action: {
    kind: "BrowserNavigateAction",
    url,
    new_tab: false,
  },
  tool_name: "browser_navigate",
  tool_call_id: "browser-call-456",
  tool_call: {
    id: "browser-call-456",
    type: "function",
    function: {
      name: "browser_navigate",
      arguments: JSON.stringify({ url, new_tab: false }),
    },
  },
  llm_response_id: "llm-response-789",
  security_risk: { level: "low" },
});

/**
 * Creates a mock ConversationErrorEvent for testing conversation-level error handling
 * These are infrastructure/authentication errors that should show error banners
 */
export const createMockConversationErrorEvent = (
  overrides: Partial<ConversationErrorEvent> = {},
): ConversationErrorEvent => ({
  id: "conversation-error-123",
  timestamp: new Date().toISOString(),
  source: "environment",
  kind: "ConversationErrorEvent",
  code: "AuthenticationError",
  detail: "Your session has expired. Please log in again.",
  ...overrides,
});
