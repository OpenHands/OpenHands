import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { HttpError } from "@openhands/typescript-client";
import { Messages } from "#/components/conversation-events/chat/messages";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { useEventStore } from "#/stores/use-event-store";
import { ASK_USER_ANSWER_PREFIX } from "#/utils/decision-closures";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import {
  SecurityRisk,
  type ActionEvent,
  type ObservationEvent,
  type MessageEvent,
} from "#/types/agent-server/core";
import type { AskUserAction } from "#/types/agent-server/core/base/action";
import type { AskUserObservation } from "#/types/agent-server/core/base/observation";

const question: ActionEvent<AskUserAction> = {
  id: "question-1",
  timestamp: "2026-11-15T12:00:00Z",
  source: "agent",
  thought: [],
  thinking_blocks: [],
  tool_name: "ask_user",
  tool_call_id: "call-1",
  tool_call: {
    id: "call-1",
    type: "function",
    function: { name: "ask_user", arguments: "{}" },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.UNKNOWN,
  action: {
    kind: "AskUserAction",
    question: "Choose where to store the data?",
    options: [
      { label: "SQLite", consequence: "Keep data on this device." },
      { label: "PostgreSQL", consequence: "Share data across servers." },
    ],
    recommended: 1,
  },
};
const result: ObservationEvent<AskUserObservation> = {
  id: "result-1",
  timestamp: "2026-11-15T12:00:01Z",
  source: "environment",
  tool_name: "ask_user",
  tool_call_id: "call-1",
  action_id: question.id,
  observation: {
    kind: "AskUserObservation",
    content: [{ type: "text", text: "The question is shown." }],
    is_error: false,
  },
};
const reply = (sender?: string): MessageEvent => ({
  id: "reply-1",
  timestamp: "2026-11-15T12:00:03Z",
  source: "user",
  sender,
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Use a file instead" }],
  },
  activated_skills: [],
  extended_content: [],
});
const delivery = (id: string, label: string): MessageEvent => ({
  ...reply(),
  id,
  llm_message: {
    role: "user",
    content: [
      {
        type: "text",
        text: `[Answer to your question: "${question.action.question}"]\n${label}`,
      },
    ],
  },
});
function Transcript({ readOnly = false }: { readOnly?: boolean }) {
  const events = useEventStore((s) => s.events);
  const messages = useEventStore((s) => s.uiEvents);
  return (
    <Messages
      messages={messages.filter(shouldRenderEvent)}
      allEvents={events}
      readOnly={readOnly}
    />
  );
}
beforeEach(() => {
  useEventStore.getState().clearEventsForConversation("test-conversation-id");
  useEventStore.getState().addEvents([question, result]);
  vi.spyOn(
    experimentalFeaturesService,
    "listExperimentalFeatures",
  ).mockResolvedValue([
    {
      key: "ask_user",
      label: "Decision cards",
      description: "",
      enabled: true,
    },
  ]);
});
afterEach(() => vi.restoreAllMocks());

it("hydrates the answered card without repeating the machine answer as a chat bubble", async () => {
  const machineText =
    '[Answer to your question: "Choose where to store the data?"]\nPostgreSQL\nUse the EU region';
  useEventStore.getState().addEvents([
    {
      id: "answer-1",
      timestamp: "2026-11-15T12:00:02Z",
      kind: "AskUserAnswerEvent",
      source: "user",
      ask_id: question.id,
      option_index: 1,
      option_label: "PostgreSQL",
      text: "Use the EU region",
    },
    {
      ...reply(),
      llm_message: {
        role: "user",
        content: [{ type: "text", text: machineText }],
      },
    } satisfies MessageEvent,
    {
      ...reply(),
      id: "ordinary-chat",
      llm_message: {
        role: "user",
        content: [{ type: "text", text: "Keep the backups too." }],
      },
    } satisfies MessageEvent,
  ]);
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  expect(within(card).getByText("Use the EU region")).toBeVisible();
  expect(
    within(card).getByRole("img", { name: "ASK_USER$ANSWERED" }).parentElement,
  ).toHaveTextContent("PostgreSQL");
  expect(within(card).getByRole("status")).toHaveTextContent(
    "ASK_USER$ANSWERED",
  );
  expect(
    screen.queryByText(/\[Answer to your question:/),
  ).not.toBeInTheDocument();
  expect(screen.getByText("Keep the backups too.")).toBeVisible();
});

it("keeps an undelivered answer retryable after hydration and settles on the delivered retry", async () => {
  useEventStore.getState().addEvent({
    id: "orphan-answer",
    kind: "AskUserAnswerEvent",
    timestamp: "2026-11-15T12:00:02Z",
    source: "user",
    ask_id: question.id,
    option_index: 0,
    option_label: "SQLite",
    text: "Lost before delivery",
  });
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  await waitFor(() =>
    expect(
      within(card).getByRole("button", { name: /PostgreSQL/ }),
    ).toBeEnabled(),
  );
  act(() => {
    useEventStore.getState().addEvents([
      {
        id: "delivered-answer",
        kind: "AskUserAnswerEvent",
        timestamp: "2026-11-15T12:00:03Z",
        source: "user",
        ask_id: question.id,
        option_index: 1,
        option_label: "PostgreSQL",
        text: "Now delivered",
      },
      {
        ...delivery("delivery-1", "PostgreSQL"),
        timestamp: "2026-11-15T12:00:04Z",
      },
    ]);
  });
  expect(await within(card).findByText("Now delivered")).toBeVisible();
  expect(
    within(card).getByRole("img", { name: "ASK_USER$ANSWERED" }).parentElement,
  ).toHaveTextContent("PostgreSQL");
  expect(
    within(card).queryByText("Lost before delivery"),
  ).not.toBeInTheDocument();
});

it("does not offer to resend an answer already saved when run capacity is full", async () => {
  vi.spyOn(ConversationClient.prototype, "answerAskUser").mockRejectedValue(
    new HttpError(429, "Too Many Requests"),
  );
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  await userEvent.click(
    await within(card).findByRole("button", { name: /SQLite/ }),
  );
  expect(await within(card).findByRole("alert")).toHaveTextContent(
    "ASK_USER$SAVED_NOT_RUNNING",
  );
  expect(within(card).queryByRole("button")).not.toBeInTheDocument();
  act(() => useEventStore.getState().addEvent(reply()));
  expect(within(card).getByRole("alert")).toHaveTextContent(
    "ASK_USER$SAVED_NOT_RUNNING",
  );
});

it("keeps the decision outside collapsed activity and submits a non-default option with words once", async () => {
  let finish!: (
    value: Awaited<ReturnType<ConversationClient["answerAskUser"]>>,
  ) => void;
  const answer = vi
    .spyOn(ConversationClient.prototype, "answerAskUser")
    .mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  const user = userEvent.setup();
  const option = await within(card).findByRole("button", {
    name: /PostgreSQL/,
  });
  await user.type(within(card).getByRole("textbox"), "Use the EU region");
  await user.dblClick(option);
  expect(answer).toHaveBeenCalledOnce();
  expect(answer).toHaveBeenCalledWith("test-conversation-id", question.id, {
    optionIndex: 1,
    text: "Use the EU region",
  });
  expect(option).toBeDisabled();
  await act(async () =>
    finish({
      id: "answer-1",
      timestamp: "2026-11-15T12:00:02Z",
      kind: "AskUserAnswerEvent",
      source: "user",
      ask_id: question.id,
      option_index: 1,
      option_label: "PostgreSQL",
      text: "Use the EU region",
    }),
  );
  expect(await within(card).findByText("ASK_USER$ANSWERED")).toBeVisible();
  expect(within(card).queryByRole("button")).not.toBeInTheDocument();
});

it("allows words alone, preserves them after a failed request, and permits retry", async () => {
  const answer = vi
    .spyOn(ConversationClient.prototype, "answerAskUser")
    .mockRejectedValue(new Error("offline"));
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  const user = userEvent.setup();
  const send = await within(card).findByRole("button", {
    name: "ASK_USER$SEND",
  });
  expect(send).toBeDisabled();
  await user.type(within(card).getByRole("textbox"), "Neither, use JSON");
  await user.click(send);
  expect(await within(card).findByRole("alert")).toHaveTextContent(
    "ASK_USER$SEND_FAILED",
  );
  expect(within(card).getByRole("textbox")).toHaveValue("Neither, use JSON");
  expect(answer).toHaveBeenCalledWith("test-conversation-id", question.id, {
    text: "Neither, use JSON",
  });
  expect(send).toBeEnabled();
  act(() => useEventStore.getState().addEvent(reply()));
  expect(
    await within(card).findByText("ASK_USER$ANSWERED_IN_CHAT"),
  ).toBeVisible();
  expect(within(card).queryByRole("alert")).not.toBeInTheDocument();
});

it("ignores relayed messages but settles on the person's chat reply, including history hydration", async () => {
  useEventStore.getState().addEvent(reply("another-conversation"));
  const { unmount } = renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  await waitFor(() =>
    expect(within(card).getByRole("button", { name: /SQLite/ })).toBeEnabled(),
  );
  act(() =>
    useEventStore.getState().addEvent({ ...reply(), id: "person-reply" }),
  );
  expect(
    await within(card).findByText("ASK_USER$ANSWERED_IN_CHAT"),
  ).toBeVisible();
  expect(within(card).queryByRole("button")).not.toBeInTheDocument();
  unmount();
  renderWithProviders(<Transcript />);
  expect(await screen.findByText("ASK_USER$ANSWERED_IN_CHAT")).toBeVisible();
});

it("keeps existing decisions readable with the feature off", async () => {
  vi.mocked(
    experimentalFeaturesService.listExperimentalFeatures,
  ).mockResolvedValue([]);
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  expect(await within(card).findByText("ASK_USER$DISABLED")).toBeVisible();
  expect(within(card).getByText("Share data across servers.")).toBeVisible();
  expect(within(card).getByText("Keep data on this device.")).toBeVisible();
  expect(
    within(card).getByText("ASK_USER$RECOMMENDED").parentElement,
  ).toHaveTextContent("PostgreSQL");
  expect(within(card).queryByRole("button")).not.toBeInTheDocument();
});

it("keeps the first closure for each decision and ignores planner replies in the main conversation", async () => {
  useEventStore.getState().addEvents([
    {
      ...question,
      id: "question-2",
      tool_call_id: "call-2",
      tool_call: { ...question.tool_call, id: "call-2" },
      action: { ...question.action, question: "Choose the backup location?" },
    },
    {
      ...result,
      id: "result-2",
      action_id: "question-2",
      tool_call_id: "call-2",
    },
    { ...reply(), isFromPlanningAgent: true },
  ]);
  renderWithProviders(<Transcript />);
  const earlierCard = await screen.findByRole("region", {
    name: question.action.question,
  });
  const currentCard = await screen.findByRole("region", {
    name: "Choose the backup location?",
  });
  expect(within(earlierCard).getByText("ASK_USER$SUPERSEDED")).toBeVisible();
  await waitFor(() =>
    expect(
      within(currentCard).getByRole("button", { name: /SQLite/ }),
    ).toBeEnabled(),
  );
  act(() =>
    useEventStore.getState().addEvent({ ...reply(), id: "main-reply" }),
  );
  expect(
    await within(currentCard).findByText("ASK_USER$ANSWERED_IN_CHAT"),
  ).toBeVisible();
  expect(within(earlierCard).getByText("ASK_USER$SUPERSEDED")).toBeVisible();
});

it("never sends a planner answer to the main conversation with the feature enabled", async () => {
  const answer = vi.spyOn(ConversationClient.prototype, "answerAskUser");
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  await waitFor(() =>
    expect(within(card).getByRole("button", { name: /SQLite/ })).toBeEnabled(),
  );
  act(() => {
    useEventStore.getState().clearEventsForConversation("test-conversation-id");
    useEventStore.getState().addEvents([
      { ...question, isFromPlanningAgent: true },
      { ...result, isFromPlanningAgent: true },
    ]);
  });
  const plannerCard = await screen.findByRole("region", {
    name: question.action.question,
  });
  expect(within(plannerCard).getByText("ASK_USER$PLANNER")).toBeVisible();
  expect(within(plannerCard).queryByRole("button")).not.toBeInTheDocument();
  expect(answer).not.toHaveBeenCalled();
});

it("preserves the agent's narration and reasoning after the question result replaces its action", async () => {
  useEventStore.getState().clearEventsForConversation("test-conversation-id");
  useEventStore.getState().addEvents([
    {
      ...question,
      thought: [{ type: "text", text: "The storage choice affects sharing." }],
      reasoning_content: "Consider whether several servers need access.",
    } satisfies ActionEvent<AskUserAction>,
    result,
  ]);
  renderWithProviders(<Transcript />);
  expect(
    await screen.findByText("The storage choice affects sharing."),
  ).toBeVisible();
  await userEvent.click(screen.getByRole("button", { expanded: false }));
  expect(
    await screen.findByText("Consider whether several servers need access."),
  ).toBeVisible();
  expect(
    screen.getByRole("region", { name: question.action.question }),
  ).toBeVisible();
});

it("reads an ordinary reply after an undelivered answer as an answer in the chat", async () => {
  useEventStore.getState().addEvents([
    {
      id: "orphan-answer",
      kind: "AskUserAnswerEvent",
      timestamp: "2026-11-15T12:00:02Z",
      source: "user",
      ask_id: question.id,
      option_index: 0,
      option_label: "SQLite",
      text: "Lost before delivery",
    },
    reply(),
  ]);
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  expect(
    await within(card).findByText("ASK_USER$ANSWERED_IN_CHAT"),
  ).toBeVisible();
  expect(
    within(card).queryByRole("img", { name: "ASK_USER$ANSWERED" }),
  ).not.toBeInTheDocument();
  expect(
    within(card).queryByText("Lost before delivery"),
  ).not.toBeInTheDocument();
  expect(screen.getByText("Use a file instead")).toBeVisible();
});

it("takes no answer in a read-only view", async () => {
  const answer = vi.spyOn(ConversationClient.prototype, "answerAskUser");
  renderWithProviders(<Transcript readOnly />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  expect(await within(card).findByText("ASK_USER$WAITING")).toBeVisible();
  expect(within(card).queryByRole("button")).not.toBeInTheDocument();
  expect(within(card).queryByRole("textbox")).not.toBeInTheDocument();
  expect(answer).not.toHaveBeenCalled();
});

it("shows the agent's reaction on the hidden answer message on the card", async () => {
  useEventStore.getState().addEvents([
    {
      id: "answer-1",
      timestamp: "2026-11-15T12:00:02Z",
      kind: "AskUserAnswerEvent",
      source: "user",
      ask_id: question.id,
      option_index: 1,
      option_label: "PostgreSQL",
      text: null,
    },
    delivery("delivery-1", "PostgreSQL"),
    {
      id: "reaction-1",
      timestamp: "2026-11-15T12:00:05Z",
      kind: "ReactionEvent",
      source: "agent",
      target_event_id: "delivery-1",
      emoji: "eyes",
      removed: false,
    },
  ]);
  renderWithProviders(<Transcript />);
  const card = await screen.findByRole("region", {
    name: question.action.question,
  });
  expect(await within(card).findByText("👀")).toBeVisible();
  expect(
    screen.queryByText(/\[Answer to your question:/),
  ).not.toBeInTheDocument();
});

it("hides the answer message by the SDK's ANSWER_HEADER_PREFIX", () => {
  // Must match ANSWER_HEADER_PREFIX in openhands/sdk/event/ask_user.py.
  expect(ASK_USER_ANSWER_PREFIX).toBe("[Answer to your question");
  expect(shouldRenderEvent(delivery("delivery-1", "SQLite"))).toBe(false);
  expect(shouldRenderEvent(reply())).toBe(true);
});
