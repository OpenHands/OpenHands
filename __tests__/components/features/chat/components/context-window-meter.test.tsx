import React from "react";
import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import useMetricsStore from "#/stores/metrics-store";

vi.mock("#/components/features/conversation/metrics-modal/metrics-modal", () => ({
  MetricsModal: ({
    isOpen,
    onOpenChange,
  }: {
    isOpen: boolean;
    onOpenChange: (open: boolean) => void;
  }) =>
    isOpen ? (
      <div data-testid="metrics-modal-stub">
        <button
          type="button"
          data-testid="close-metrics-modal"
          onClick={() => onOpenChange(false)}
        >
          Close
        </button>
      </div>
    ) : null,
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: undefined }),
}));

vi.mock("#/hooks/query/use-conversation-metrics", () => ({
  useConversationMetrics: () => ({ data: undefined }),
}));

// eslint-disable-next-line import/first
import { ContextWindowMeter } from "#/components/features/chat/components/context-window-meter";

describe("ContextWindowMeter", () => {
  afterEach(() => {
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
    expect(screen.getByText("198.5k / 1.0M (20%)")).toBeInTheDocument();
  });

  it("opens the metrics modal from the plan usage row", () => {
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

    expect(screen.getByTestId("metrics-modal-stub")).toBeInTheDocument();
  });
});
