import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createUserMessageEvent, renderWithProviders } from "test-utils";
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
  type OpenHandsEvent,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";

const CONVERSATION_ID = "test-conversation-id";

// `SecretRequestCard` reports completion through `completeSecretRequest`; stub it
// so the test can assert the call without driving the conversation-secret and
// client-tool-result network paths. Completing a request removes it from the
// queue, like the real service does.
const { mockCompleteSecretRequest } = vi.hoisted(() => ({
  mockCompleteSecretRequest: vi.fn(),
}));
vi.mock("#/services/secret-request", () => ({
  completeSecretRequest: mockCompleteSecretRequest,
}));

// Mirrors the virtualization harness: a real scroll element so the list can
// cross the threshold.
function Harness({ events }: { events: OpenHandsEvent[] }) {
  const [scrollElement, setScrollElement] =
    React.useState<HTMLDivElement | null>(null);
  return (
    <div
      ref={setScrollElement}
      data-testid="scroll-parent"
      style={{ height: 600, overflowY: "auto" }}
    >
      <Messages
        messages={events}
        allEvents={events}
        scrollParent={scrollElement}
      />
    </div>
  );
}

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
    vi.restoreAllMocks();
    mockCompleteSecretRequest
      .mockReset()
      .mockImplementation((request: SecretRequest) =>
        useSecretRequestStore.getState().dismiss(request.toolCallId),
      );
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    // jsdom has no layout; give the scroll container a measurable viewport so
    // the virtualizer can compute a range.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
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

  it("completes a secret request saved while the list crosses into virtualization", async () => {
    // The reported bug: with 149 rows, a save is started; an incoming event
    // raises the count to 150 before the save resolves. Crossing the threshold
    // replaces the plain list with the virtual shell, so `SecretRequestCard`
    // remounts in the trailing content. The per-call `onSuccess` that reports
    // the result lived on the unmounted observer, so the save succeeded but
    // `completeSecretRequest` never ran and the agent waited forever.
    const action = makeBashAction("action-1");
    const request = makeRequest(action.tool_call_id, "GITHUB_TOKEN");
    useSecretRequestStore.setState({ requests: [request] });

    const user = userEvent.setup();
    let resolveSave: () => void = () => {};
    vi.spyOn(SecretsService, "createSecret").mockReturnValue(
      new Promise<void>((resolve) => {
        resolveSave = resolve;
      }),
    );

    const initial = [
      action,
      ...Array.from({ length: 148 }, (_, i) =>
        createUserMessageEvent(`message-${i}`),
      ),
    ];
    const { rerender } = renderWithProviders(<Harness events={initial} />);
    expect(
      screen.queryByTestId("virtualized-message-list"),
    ).not.toBeInTheDocument();

    await user.type(screen.getByTestId("value-input"), "ghp_example");
    await user.click(screen.getByTestId("submit-button"));

    // The agent streams one more event: 150 rows, now virtualized.
    rerender(
      <Harness
        events={[...initial, createUserMessageEvent("message-crossing")]}
      />,
    );
    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();
    // The card remounted in the trailing content; its draft is gone, which is
    // what makes this a fresh observer.
    expect(mockCompleteSecretRequest).not.toHaveBeenCalled();

    await act(async () => {
      resolveSave();
      await Promise.resolve();
    });

    await waitFor(() =>
      expect(mockCompleteSecretRequest).toHaveBeenCalledWith(request, {
        name: "GITHUB_TOKEN",
        description: undefined,
      }),
    );
    expect(useSecretRequestStore.getState().requests).toHaveLength(0);
  });
});
