import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import GraphService from "#/api/graph-service/graph-service.api";
import type {
  GraphConfig,
  GraphIndexStatus,
  GraphQueryResult,
} from "#/api/graph-service/graph-types";
import { GRAPH_SOURCE } from "#/api/graph-service/graph-constants";
import { GraphSettings } from "#/components/features/graph/graph-settings";
import { IndexerStatusCard } from "#/components/features/graph/indexer-status-card";
import { QueryConsole } from "#/components/features/graph/query-console";
import { GraphPage } from "#/components/features/graph/graph-page";
import { RelevantFilesCard } from "#/components/features/graph/relevant-files-card";
import { I18nKey } from "#/i18n/declaration";

const STATUS: GraphIndexStatus = {
  running: false,
  files_indexed: 4,
  symbols: 6,
  edges: 8,
  languages_used: ["python"],
  coverage: 1,
  last_full_index_at: "2026-01-01T00:00:00+00:00",
  last_error: null,
};

const CONFIG: GraphConfig = {
  enabled: true,
  languages: ["python", "javascript"],
  graph_budget_lines: 400,
  max_context_files: 12,
  strict: false,
  stale_after_minutes: 60,
  imported_project_path: null,
};

const RESULT: GraphQueryResult = {
  query: "callers",
  symbol: "greet",
  result: [
    {
      id: "function:app.py:main:4",
      kind: "function",
      name: "main",
      path: "app.py",
      start_line: 4,
      end_line: 6,
      external: false,
    },
  ],
  source: GRAPH_SOURCE,
  status: "ok",
  last_full_index_at: STATUS.last_full_index_at,
  coverage: 1,
};

describe("IndexerStatusCard", () => {
  it("renders idle status metrics", () => {
    renderWithProviders(
      <IndexerStatusCard
        status={STATUS}
        onClear={vi.fn()}
        onRetrigger={vi.fn()}
      />,
    );
    expect(screen.getByTestId("graph-indexer-running")).toHaveTextContent(
      I18nKey.GRAPH$IDLE,
    );
    expect(screen.getByTestId("graph-files-indexed")).toHaveTextContent("4");
    expect(screen.getByTestId("graph-coverage")).toHaveTextContent("100%");
  });

  it("confirms clear and retrigger", async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    const onRetrigger = vi.fn();
    renderWithProviders(
      <IndexerStatusCard
        status={STATUS}
        onClear={onClear}
        onRetrigger={onRetrigger}
      />,
    );
    await user.click(screen.getByTestId("graph-clear"));
    expect(screen.getByTestId("confirmation-modal")).toHaveTextContent(
      I18nKey.GRAPH$CLEAR_CONFIRM,
    );
    await user.click(screen.getByTestId("confirm-button"));
    expect(onClear).toHaveBeenCalled();

    await user.click(screen.getByTestId("graph-retrigger"));
    expect(screen.getByTestId("confirmation-modal")).toHaveTextContent(
      I18nKey.GRAPH$REINDEX_CONFIRM,
    );
    await user.click(screen.getByTestId("confirm-button"));
    expect(onRetrigger).toHaveBeenCalled();
  });
});

describe("QueryConsole", () => {
  it("shows results and source graph", () => {
    renderWithProviders(<QueryConsole onQuery={vi.fn()} result={RESULT} />);
    expect(screen.getByTestId("graph-query-source")).toHaveTextContent(
      GRAPH_SOURCE,
    );
    expect(screen.getByTestId("graph-query-results")).toHaveTextContent("main");
    expect(screen.queryByTestId("graph-stale-banner")).not.toBeInTheDocument();
  });

  it("shows a staleness banner", () => {
    renderWithProviders(
      <QueryConsole
        onQuery={vi.fn()}
        result={{ ...RESULT, status: "stale" }}
      />,
    );
    expect(screen.getByTestId("graph-stale-banner")).toHaveTextContent(
      I18nKey.GRAPH$STALE,
    );
  });
});

describe("GraphSettings", () => {
  it("puts config patches", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<GraphSettings config={CONFIG} onChange={onChange} />);
    await user.click(
      screen.getByRole("switch", { name: I18nKey.GRAPH$ENABLED }),
    );
    expect(onChange).toHaveBeenCalledWith({ enabled: false });
  });
});

describe("RelevantFilesCard", () => {
  it("lists files, reasons, budget, and toggle", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    renderWithProviders(
      <RelevantFilesCard
        files={[
          {
            file: "helper.py",
            relevance: 400,
            reason: "defined here",
            lines: 3,
          },
        ]}
        budgetLines={400}
        usedLines={3}
        enabled
        defaultEnabled
        onToggle={onToggle}
      />,
    );
    expect(screen.getByTestId("graph-relevant-row-helper.py")).toHaveTextContent(
      "helper.py",
    );
    expect(screen.getByTestId("graph-relevant-row-helper.py")).toHaveTextContent(
      "defined here",
    );
    expect(screen.getByTestId("graph-budget-usage")).toHaveTextContent("3/400");
    expect(screen.getByTestId("graph-relevant-default")).toHaveAttribute(
      "data-default-enabled",
      "true",
    );
    await user.click(
      screen.getByRole("switch", { name: I18nKey.GRAPH$CONTEXT_TOGGLE }),
    );
    expect(onToggle).toHaveBeenCalled();
  });

  it("shows a stale-skipped banner", () => {
    renderWithProviders(
      <RelevantFilesCard
        files={[]}
        budgetLines={400}
        usedLines={0}
        enabled={false}
        defaultEnabled
        staleSkipped
        onToggle={vi.fn()}
      />,
    );
    expect(screen.getByTestId("graph-stale-skipped")).toHaveTextContent(
      I18nKey.GRAPH$STALE_SKIPPED,
    );
  });
});

describe("GraphPage", () => {
  it("wires indexer status and config PUT without a query console", async () => {
    const user = userEvent.setup();
    vi.spyOn(GraphService, "getStatus").mockResolvedValue(STATUS);
    vi.spyOn(GraphService, "getConfig").mockResolvedValue(CONFIG);
    const put = vi
      .spyOn(GraphService, "putConfig")
      .mockResolvedValue({ ...CONFIG, strict: true });

    renderWithProviders(<GraphPage />);
    await screen.findByTestId("graph-indexer-controls");
    await screen.findByTestId("graph-indexer-status");
    expect(screen.queryByTestId("graph-query-submit")).not.toBeInTheDocument();
    expect(screen.queryByTestId("graph-relevant-files")).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("switch", { name: I18nKey.GRAPH$STRICT }),
    );
    await waitFor(() => {
      expect(put).toHaveBeenCalledWith({ strict: true });
    });
  });
});
