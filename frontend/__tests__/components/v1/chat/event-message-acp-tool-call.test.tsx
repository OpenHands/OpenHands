import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EventMessage } from "#/components/v1/chat/event-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { ACPToolCallEvent } from "#/types/v1/core/events/acp-tool-call-event";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: { APP_MODE: "saas" } }),
}));
vi.mock("#/hooks/use-agent-state");
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

const makeEvent = (
  overrides: Partial<ACPToolCallEvent> = {},
): ACPToolCallEvent => ({
  kind: "ACPToolCallEvent",
  id: "evt-1",
  timestamp: "2026-04-16T19:32:29.828069",
  source: "agent",
  tool_call_id: "toolu_123",
  title: "gh pr diff 490",
  tool_kind: "execute",
  status: "completed",
  raw_input: { command: "gh pr diff 490" },
  raw_output: "diff output here",
  content: null,
  is_error: false,
  ...overrides,
});

describe("EventMessage - ACPToolCallEvent dispatch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
      executionStatus: null,
      isArchived: false,
    });
  });

  it("renders an ACP card through the same GenericEventMessage wrapper as observations", () => {
    renderWithProviders(
      <EventMessage
        event={makeEvent()}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The test-utils i18n instance doesn't load the real translation bundle,
    // so createTitleFromKey falls back to the key literal. Assert on the
    // key — the integration case (rendered string) is covered by a Storybook
    // story + manual verification listed in the PR description.
    expect(screen.getByText("ACTION_MESSAGE$ACP_RUN")).toBeInTheDocument();
  });

  it("shows the success check mark for completed tool calls", () => {
    renderWithProviders(
      <EventMessage
        event={makeEvent()}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // Same status-icon testid as regular successful observations.
    expect(screen.getByTestId("status-icon")).toBeInTheDocument();
  });

  it("omits the status icon while a call is in progress", () => {
    renderWithProviders(
      <EventMessage
        event={makeEvent({ status: "in_progress", raw_output: null })}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // getACPToolCallResult returns undefined for in_progress, so
    // SuccessIndicator renders no icon.
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
  });

  it("expands details on click and shows the Command: + Output: blocks", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EventMessage
        event={makeEvent()}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Expand" }));

    // Markdown renderer wraps code blocks but the plain text survives.
    expect(screen.getByText(/gh pr diff 490/)).toBeInTheDocument();
    expect(screen.getByText(/diff output here/)).toBeInTheDocument();
  });
});
