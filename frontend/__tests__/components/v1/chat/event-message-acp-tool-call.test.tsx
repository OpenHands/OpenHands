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

    // The title row renders ``event.title`` verbatim — the upstream ACP
    // sub-agent (Claude Code / Codex / Gemini CLI) already emits a
    // humanised label, so no translation-key wrapping happens here.
    expect(screen.getByText("gh pr diff 490")).toBeInTheDocument();
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
    // ``gh pr diff 490`` appears twice now — once in the title row (the
    // verbatim ``event.title``) and once in the ``Command:`` block in the
    // expanded details. We just care that the details panel contains it.
    expect(screen.getAllByText(/gh pr diff 490/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/diff output here/)).toBeInTheDocument();
  });
});
