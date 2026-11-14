import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelBreakdown } from "#/components/features/conversation/usage-panel/model-breakdown";
import { I18nKey } from "#/i18n/declaration";
import type { UsageBreakdownEntry } from "#/api/conversation-service/agent-server-conversation-service.types";

// The global vitest setup mocks useTranslation to return the key itself, so
// these tests assert on raw I18nKey strings.

const entry = (
  overrides: Partial<UsageBreakdownEntry> = {},
): UsageBreakdownEntry => ({
  modelName: "anthropic/claude-sonnet-4-5",
  accumulatedCost: 0.003,
  usage: {
    prompt_tokens: 19_747_245,
    completion_tokens: 415_535,
    cache_read_tokens: 18_989_568,
    cache_write_tokens: 0,
  },
  ...overrides,
});

describe("ModelBreakdown", () => {
  it("renders nothing for an empty breakdown", () => {
    const { container } = render(<ModelBreakdown entries={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows each model with its own accumulated cost", () => {
    render(
      <ModelBreakdown
        entries={[
          entry({ modelName: "m/b", accumulatedCost: 1.5 }),
          entry({ modelName: "m/a", accumulatedCost: 0.003 }),
        ]}
      />,
    );
    expect(screen.getByText(I18nKey.CONVERSATION$BY_MODEL)).toBeInTheDocument();
    expect(screen.getByText("m/b")).toBeInTheDocument();
    expect(screen.getByText("$1.5000")).toBeInTheDocument();
    expect(screen.getByText("m/a")).toBeInTheDocument();
    expect(screen.getByText("$0.0030")).toBeInTheDocument();
  });

  it("summarizes per-model input and output tokens", () => {
    render(<ModelBreakdown entries={[entry()]} />);
    expect(
      screen.getByText(
        `${I18nKey.CONVERSATION$INPUT} 19,747,245 · ${I18nKey.CONVERSATION$OUTPUT} 415,535`,
      ),
    ).toBeInTheDocument();
  });

  it("marks a model that used tokens but reported no cost as unpriced", () => {
    render(
      <ModelBreakdown
        entries={[entry({ modelName: "acme/coder", accumulatedCost: 0 })]}
      />,
    );
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("$0.0000")).not.toBeInTheDocument();
  });
});
