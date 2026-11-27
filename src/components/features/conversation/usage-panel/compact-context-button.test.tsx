import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AgentState } from "#/types/agent-state";
import { CompactContextButton } from "./compact-context-button";

const mutateMock = vi.fn();

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conv-1",
      conversation_url: null,
      session_api_key: null,
    },
  }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: AgentState.AWAITING_USER_INPUT }),
}));

vi.mock("#/hooks/mutation/use-condense-conversation", () => ({
  useCondenseConversation: () => ({
    mutate: mutateMock,
    isPending: false,
  }),
}));

vi.mock("#/hooks/use-await-context-compaction", () => ({
  useAwaitContextCompaction: vi.fn(),
  buildContextCompactionResult: vi.fn(),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

function renderButton(fillPercent: number) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <CompactContextButton fillPercent={fillPercent} />
    </QueryClientProvider>,
  );
}

describe("CompactContextButton", () => {
  beforeEach(() => {
    mutateMock.mockClear();
  });

  it("puts the explanation in an info control beside a content-sized button", () => {
    renderButton(40);

    expect(
      screen.queryByText("CONVERSATION$COMPACT_CONTEXT_DESCRIPTION"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("CONVERSATION$CONTEXT_FILLING_UP"),
    ).not.toBeInTheDocument();

    const info = screen.getByTestId("compact-context-info");
    expect(info).toHaveAttribute(
      "aria-label",
      "CONVERSATION$COMPACT_CONTEXT_DESCRIPTION",
    );

    const button = screen.getByTestId("compact-context-button");
    expect(button).toHaveClass("w-fit");
    expect(button).not.toHaveClass("w-full");
  });

  it("adds a high-fill warning once the context passes the warning threshold", () => {
    renderButton(75);

    expect(
      screen.getByText("CONVERSATION$CONTEXT_FILLING_UP"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("compact-context-button")).toHaveTextContent(
      "CONVERSATION$COMPACT_CONTEXT",
    );
  });
});
