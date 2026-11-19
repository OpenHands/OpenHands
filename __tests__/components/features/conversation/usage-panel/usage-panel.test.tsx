import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { UsagePanel } from "#/components/features/conversation/usage-panel/usage-panel";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type {
  RuntimeConversationInfo,
  RuntimeMetrics,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import LLMBalanceService from "#/api/llm-balance-service";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { I18nKey } from "#/i18n/declaration";
import useMetricsStore from "#/stores/metrics-store";
import { buildUsageBreakdown } from "#/utils/conversation-metrics";

const useActiveConversationMock = vi.fn();

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const CONVERSATION_ID = "conv-usage";

const llmMetrics = (
  modelName: string,
  cost: number,
  promptTokens: number,
  completionTokens: number,
): RuntimeMetrics => ({
  model_name: modelName,
  accumulated_cost: cost,
  max_budget_per_task: null,
  accumulated_token_usage: {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    context_window: 200_000,
    per_turn_token: promptTokens,
  },
  costs: [],
  response_latencies: [],
  token_usages: [],
});

const runtimeInfo = (
  usageToMetrics: Record<string, RuntimeMetrics>,
): RuntimeConversationInfo => ({
  id: CONVERSATION_ID,
  title: "Usage",
  created_at: "2026-11-13T00:00:00Z",
  updated_at: "2026-11-13T00:00:00Z",
  status: ExecutionStatus.IDLE,
  available_models: [],
  metrics: null,
  max_budget_per_run: null,
  stats: { usage_to_metrics: usageToMetrics },
});

function renderPanel(
  usageToMetrics: Record<string, RuntimeMetrics>,
  agentKind = "openhands",
) {
  useActiveConversationMock.mockReturnValue({
    data: {
      id: CONVERSATION_ID,
      conversation_url: `http://localhost:8000/api/conversations/${CONVERSATION_ID}`,
      session_api_key: "key",
      agent_kind: agentKind,
    },
  });
  vi.spyOn(
    AgentServerConversationService,
    "getRuntimeConversation",
  ).mockResolvedValue(runtimeInfo(usageToMetrics));
  return renderWithProviders(<UsagePanel />);
}

describe("UsagePanel cost transparency", () => {
  beforeEach(() => {
    useMetricsStore.getState().resetMetrics();
    vi.spyOn(LLMBalanceService, "getBalance").mockRejectedValue(
      new Error("no balance"),
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("labels a priced total as an estimate and attributes it per model", async () => {
    renderPanel({
      agent: llmMetrics("anthropic/claude-sonnet-4-5", 1.25, 10_000, 500),
      condenser: llmMetrics("openai/gpt-5-mini", 0.05, 2_000, 100),
    });

    expect(await screen.findByTestId("total-cost-value")).toHaveTextContent(
      "$1.3000",
    );
    expect(screen.getByTestId("cost-estimate-note")).toHaveTextContent(
      I18nKey.CONVERSATION$COST_ESTIMATE_NOTE,
    );
    const breakdown = screen.getByTestId("usage-model-breakdown");
    const rows = within(breakdown).getAllByText(/^(anthropic|openai)\//);
    expect(rows.map((row) => row.textContent)).toEqual([
      "anthropic/claude-sonnet-4-5",
      "openai/gpt-5-mini",
    ]);
    expect(within(breakdown).getByText("$1.2500")).toBeInTheDocument();
    expect(within(breakdown).getByText("$0.0500")).toBeInTheDocument();
  });

  it("says pricing is missing instead of showing $0.0000 for unpriced tokens", async () => {
    renderPanel({
      agent: llmMetrics("openai/acme-internal-coder", 0, 1_250_000, 4_200),
      condenser: llmMetrics("openai/acme-internal-coder", 0, 0, 0),
    });

    expect(await screen.findByTestId("total-cost-value")).toHaveTextContent(
      "—",
    );
    expect(screen.queryByText("$0.0000")).not.toBeInTheDocument();
    expect(screen.getByTestId("cost-unpriced-note")).toHaveTextContent(
      I18nKey.CONVERSATION$NO_COST_DATA,
    );
    expect(screen.queryByTestId("cost-estimate-note")).not.toBeInTheDocument();
    const breakdown = screen.getByTestId("usage-model-breakdown");
    expect(
      within(breakdown).getAllByText("openai/acme-internal-coder"),
    ).toHaveLength(1);
  });

  it("keeps the ACP plan-usage note instead of the generic estimate note", async () => {
    renderPanel({ agent: llmMetrics("claude-code", 0.4, 1_000, 50) }, "acp");

    expect(await screen.findByTestId("total-cost-value")).toHaveTextContent(
      "$0.4000",
    );
    expect(
      screen.getByText(I18nKey.CONVERSATION$PLAN_USAGE_NOTE),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("cost-estimate-note")).not.toBeInTheDocument();
  });

  it("shows the breakdown from live stats the same way as the REST snapshot", async () => {
    const live = {
      agent: llmMetrics("anthropic/claude-sonnet-4-5", 2, 20_000, 800),
      condenser: llmMetrics("openai/gpt-5-mini", 0.5, 4_000, 200),
    };
    renderPanel({});
    useMetricsStore.getState().setMetrics(
      {
        cost: 2.5,
        max_budget_per_task: null,
        usage: {
          prompt_tokens: 24_000,
          completion_tokens: 1_000,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          context_window: 200_000,
          per_turn_token: 20_000,
        },
        usage_breakdown: buildUsageBreakdown(live),
      },
      CONVERSATION_ID,
    );

    await waitFor(() =>
      expect(screen.getByTestId("total-cost-value")).toHaveTextContent(
        "$2.5000",
      ),
    );
    const breakdown = screen.getByTestId("usage-model-breakdown");
    expect(within(breakdown).getByText("$2.0000")).toBeInTheDocument();
    expect(within(breakdown).getByText("$0.5000")).toBeInTheDocument();
  });
});
