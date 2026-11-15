import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { renderWithProviders } from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { isActionEvent } from "#/types/agent-server/type-guards";
import {
  answer,
  answerMessage,
  askAction,
  askShown,
  QUESTION,
  userMessage,
} from "../../features/chat/ask-user/ask-user-fixtures";

const flags = vi.hoisted(() => ({ askUser: true }));
vi.mock("#/hooks/query/use-experimental-features", () => ({
  useIsExperimentalFeatureEnabled: (key: string) =>
    key === "ask_user" ? flags.askUser : false,
}));
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
  useConversationId: () => ({ conversationId: "conv-1" }),
}));

// What the chat renders: the UI list swaps an action for its observation and
// drops events it has no row for, such as the answer event.
const show = (events: OpenHandsEvent[]) =>
  renderWithProviders(
    <Messages
      messages={events.filter(
        (event) => shouldRenderEvent(event) && !isActionEvent(event),
      )}
      allEvents={events}
    />,
  );

// The UI list swaps an action for its observation.
const asked = () => [askShown()];
const history = () => [askAction(), askShown()];

describe("Messages decision cards", () => {
  beforeEach(() => {
    flags.askUser = true;
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a waiting question as a card with its options, below the agent's words", () => {
    renderWithProviders(<Messages messages={asked()} allEvents={history()} />);

    const card = screen.getByTestId("ask-user-card");
    expect(card).toHaveAttribute("data-status", "pending");
    expect(within(card).getByText(QUESTION)).toBeInTheDocument();
    expect(
      within(card)
        .getAllByTestId("ask-user-option")
        .map((option) => option.textContent),
    ).toEqual([
      "PostgresASK_USER$RECOMMENDEDNeeds a managed database.",
      "SQLiteOne file, no server.",
    ]);
    expect(screen.getByText("This one is your call.")).toBeInTheDocument();
  });

  it("sends a tap, with the note, to the answer route", async () => {
    const answerAskUser = vi
      .spyOn(ConversationClient.prototype, "answerAskUser")
      .mockResolvedValue(answer());
    renderWithProviders(<Messages messages={asked()} allEvents={history()} />);

    await userEvent.type(screen.getByTestId("ask-user-note"), "use port 5433");
    await userEvent.click(screen.getAllByTestId("ask-user-option")[1]);

    await waitFor(() =>
      expect(answerAskUser).toHaveBeenCalledWith("conv-1", "ask-1", {
        optionIndex: 1,
        text: "use port 5433",
      }),
    );
    // One answer per card: the options lock while it is on its way.
    expect(screen.getAllByTestId("ask-user-option")[0]).toBeDisabled();
  });

  it("folds an answered card to the choice and hides the message the tap sent", () => {
    const all = [
      askAction(),
      askShown(),
      answer("ask-1", 3, { text: "use port 5433" }),
      answerMessage(),
    ];
    show(all);

    const card = screen.getByTestId("ask-user-card");
    expect(card).toHaveAttribute("data-status", "answered");
    expect(within(card).getByTestId("ask-user-outcome")).toHaveTextContent(
      "SQLiteuse port 5433",
    );
    expect(screen.queryByTestId("ask-user-option")).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Answer to your question/),
    ).not.toBeInTheDocument();
  });

  it("marks a question answered in the chat and keeps the user's message", () => {
    const reply = userMessage("chat-1", "Postgres, with DATABASE_URL.", 3);
    renderWithProviders(
      <Messages
        messages={[askShown(), reply]}
        allEvents={[askAction(), askShown(), reply]}
      />,
    );

    expect(screen.getByTestId("ask-user-card")).toHaveAttribute(
      "data-status",
      "answered_in_chat",
    );
    expect(
      screen.getByText("Postgres, with DATABASE_URL."),
    ).toBeInTheDocument();
  });

  it("changes nothing while the feature is off", () => {
    flags.askUser = false;
    const all = [askAction(), askShown(), answer(), answerMessage()];
    show(all);

    expect(screen.queryByTestId("ask-user-card")).not.toBeInTheDocument();
    expect(screen.getByText(/Answer to your question/)).toBeInTheDocument();
  });
});
