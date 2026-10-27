import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { SecretsService } from "#/api/secrets-service";
import { Messages } from "#/components/conversation-events/chat/messages";
import {
  type SecretRequest,
  useSecretRequestStore,
} from "#/stores/secret-request-store";
import {
  ActionEvent,
  MessageEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";

const CONVERSATION_ID = "test-conversation-id";

const makeBashAction = (id: string): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: `echo ${id}`,
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: `call_${id}`,
  tool_call: {
    id: `call_${id}`,
    type: "function",
    function: {
      name: "execute_bash",
      arguments: JSON.stringify({ command: `echo ${id}` }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const agentMessage: MessageEvent = {
  id: "agent-msg-1",
  timestamp: new Date().toISOString(),
  source: "agent",
  llm_message: {
    role: "assistant",
    content: [{ type: "text", text: "Meanwhile, I ran the tests." }],
  },
  activated_skills: [],
  extended_content: [],
};

const makeRequest = (toolCallId: string, name: string): SecretRequest => ({
  conversationId: CONVERSATION_ID,
  toolCallId,
  name,
  reason: null,
});

const cardNames = () =>
  screen
    .getAllByTestId("secret-request-card")
    .map((card) => card.querySelector("input")?.value);

describe("Messages secret requests", () => {
  beforeEach(() => {
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
  });

  it("shows a pending request right after its tool call, above later messages", () => {
    const action = makeBashAction("action-1");
    useSecretRequestStore.setState({
      requests: [makeRequest(action.tool_call_id, "GITHUB_TOKEN")],
    });

    renderWithProviders(
      <Messages
        messages={[action, agentMessage]}
        allEvents={[action, agentMessage]}
      />,
    );

    const card = screen.getByTestId("secret-request-card");
    const laterMessage = screen.getByText("Meanwhile, I ran the tests.");
    expect(
      card.compareDocumentPosition(laterMessage) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("falls back to the end of the chat when the tool call is not loaded", () => {
    const action = makeBashAction("action-1");
    useSecretRequestStore.setState({
      requests: [
        makeRequest("call_unknown", "NPM_TOKEN"),
        makeRequest(action.tool_call_id, "GITHUB_TOKEN"),
      ],
    });

    renderWithProviders(
      <Messages
        messages={[action, agentMessage]}
        allEvents={[action, agentMessage]}
      />,
    );

    expect(cardNames()).toEqual(["GITHUB_TOKEN", "NPM_TOKEN"]);
  });
});
