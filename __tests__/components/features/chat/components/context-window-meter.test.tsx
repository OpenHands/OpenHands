import type { ReactNode } from "react";
import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import useMetricsStore from "#/stores/metrics-store";

const navigateToTabMock = vi.fn();

vi.mock("#/hooks/use-select-conversation-tab", () => ({
  useSelectConversationTab: () => ({
    navigateToTab: navigateToTabMock,
    selectTab: vi.fn(),
    isTabActive: vi.fn(),
    onTabChange: vi.fn(),
    selectedTab: null,
    isRightPanelShown: false,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: undefined }),
}));

vi.mock("#/hooks/query/use-conversation-metrics", () => ({
  useConversationMetrics: () => ({ data: undefined }),
}));

vi.mock("#/hooks/use-compact-context-action", () => ({
  useCompactContextAction: () => ({
    handleCompact: vi.fn(),
    isCompacting: false,
    isDisabled: false,
    description: "CONVERSATION$COMPACT_CONTEXT_DESCRIPTION",
  }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: "awaiting_user_input" }),
}));

// HeroUI Tooltip only mounts content on real-DOM hover; surface it eagerly.
vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({
    content,
    children,
  }: {
    content: ReactNode;
    children: ReactNode;
  }) => (
    <>
      {children}
      <span data-testid="styled-tooltip-content">{content}</span>
    </>
  ),
}));

// eslint-disable-next-line import/first
import { ContextWindowMeter } from "#/components/features/chat/components/context-window-meter";

describe("ContextWindowMeter", () => {
  afterEach(() => {
    navigateToTabMock.mockClear();
    useMetricsStore.setState({
      cost: null,
      max_budget_per_task: null,
      usage: null,
    });
  });

  it("does not render when context window metrics are unavailable", () => {
    renderWithProviders(<ContextWindowMeter />);

    expect(
      screen.queryByTestId("context-window-meter"),
    ).not.toBeInTheDocument();
  });

  it("shows a Show Context tooltip on the ring", () => {
    useMetricsStore.setState({
      cost: null,
      max_budget_per_task: null,
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        context_window: 1_000_000,
        per_turn_token: 198_500,
      },
    });

    renderWithProviders(<ContextWindowMeter />);

    expect(screen.getByTestId("styled-tooltip-content")).toHaveTextContent(
      "CHAT_INTERFACE$SHOW_CONTEXT",
    );
  });

  it("opens a popover with compact usage details", () => {
    useMetricsStore.setState({
      cost: null,
      max_budget_per_task: null,
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        context_window: 1_000_000,
        per_turn_token: 198_500,
      },
    });

    renderWithProviders(<ContextWindowMeter />);

    fireEvent.click(screen.getByTestId("context-window-meter"));

    expect(
      screen.getByTestId("context-window-meter-popover"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("20% CONVERSATION$USED (80% CONVERSATION$LEFT)"),
    ).toBeInTheDocument();
    expect(screen.getByText("198.5k / 1.0M")).toBeInTheDocument();
    expect(
      screen.getByTestId("context-window-compact-button"),
    ).toHaveTextContent("CONVERSATION$COMPACT_CONTEXT");
    expect(
      screen.getByText("CONVERSATION$CONTEXT_WINDOW"),
    ).toBeInTheDocument();
  });

  it("opens the Usage drawer from the popover meter", () => {
    useMetricsStore.setState({
      cost: 1.25,
      max_budget_per_task: null,
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        context_window: 1_000_000,
        per_turn_token: 198_500,
      },
    });

    renderWithProviders(<ContextWindowMeter />);

    fireEvent.click(screen.getByTestId("context-window-meter"));
    fireEvent.click(screen.getByTestId("context-window-meter-bar-button"));

    expect(navigateToTabMock).toHaveBeenCalledWith("usage");
  });

  it("opens the Usage drawer from the usage row", () => {
    useMetricsStore.setState({
      cost: 1.25,
      max_budget_per_task: null,
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        context_window: 1_000_000,
        per_turn_token: 198_500,
      },
    });

    renderWithProviders(<ContextWindowMeter />);

    fireEvent.click(screen.getByTestId("context-window-meter"));
    fireEvent.click(screen.getByTestId("context-window-plan-usage"));

    expect(navigateToTabMock).toHaveBeenCalledWith("usage");
  });
});
