import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, it, expect, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { CriticResultDisplay } from "#/components/v1/chat/event-message-components/critic-result-display";
import type { CriticResult } from "#/types/v1/core/base/critic";

const mockUseSettings = vi.hoisted(() =>
  vi.fn(() => ({
    data: {
      agent_settings: {
        verification: {
          enable_iterative_refinement: true,
        },
      },
    },
  })),
);

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockUseSettings(),
}));

const makeCriticResult = (
  overrides: Partial<CriticResult> = {},
): CriticResult => ({
  score: 0.85,
  message: null,
  metadata: null,
  ...overrides,
});

beforeEach(() => {
  mockUseSettings.mockReturnValue({
    data: {
      agent_settings: {
        verification: {
          enable_iterative_refinement: true,
        },
      },
    },
  });
});

describe("CriticResultDisplay", () => {
  it("renders score as percentage", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult({ score: 0.72 })} />,
    );

    expect(screen.getByText("(72.0%)")).toBeInTheDocument();
  });

  it("renders 5 stars for a perfect score", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult({ score: 1.0 })} />,
    );

    expect(screen.getByText("★★★★★")).toBeInTheDocument();
  });

  it("renders 0 stars for a zero score", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult({ score: 0 })} />,
    );

    expect(screen.getByText("☆☆☆☆☆")).toBeInTheDocument();
  });

  it("renders green color for high score", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult({ score: 0.8 })} />,
    );

    const stars = screen.getByText("★★★★☆");
    expect(stars.className).toContain("text-green-400");
  });

  it("renders yellow color for medium score", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult({ score: 0.5 })} />,
    );

    const stars = screen.getByText("★★★☆☆");
    expect(stars.className).toContain("text-yellow-400");
  });

  it("renders red color for low score", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult({ score: 0.2 })} />,
    );

    const stars = screen.getByText("★☆☆☆☆");
    expect(stars.className).toContain("text-red-400");
  });

  it("renders label text", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult()} />,
    );

    expect(
      screen.getByText("CRITIC$SUCCESS_LIKELIHOOD_LABEL"),
    ).toBeInTheDocument();
  });

  it("does not render expand button without features", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult()} />,
    );

    expect(
      screen.queryByLabelText("Expand details"),
    ).not.toBeInTheDocument();
  });

  it("prompts users to enable iterative refinement when it is disabled", () => {
    mockUseSettings.mockReturnValue({
      data: {
        agent_settings: {
          verification: {
            enable_iterative_refinement: false,
          },
        },
      },
    });

    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult()} />,
    );

    expect(
      screen.getByTestId("critic-iterative-refinement-hint"),
    ).toHaveTextContent("CRITIC$ITERATIVE_REFINEMENT_HINT");
  });

  it("does not prompt users when iterative refinement is enabled", () => {
    renderWithProviders(
      <CriticResultDisplay criticResult={makeCriticResult()} />,
    );

    expect(
      screen.queryByTestId("critic-iterative-refinement-hint"),
    ).not.toBeInTheDocument();
  });

  it("renders expand button when features are present", () => {
    const result = makeCriticResult({
      metadata: {
        categorized_features: {
          agent_behavioral_issues: [
            {
              name: "insufficient_testing",
              display_name: "Insufficient Testing",
              probability: 0.75,
            },
          ],
        },
      },
    });

    renderWithProviders(<CriticResultDisplay criticResult={result} />);

    expect(screen.getByLabelText("Expand details")).toBeInTheDocument();
  });

  it("expands features on click", async () => {
    const user = userEvent.setup();
    const result = makeCriticResult({
      metadata: {
        categorized_features: {
          agent_behavioral_issues: [
            {
              name: "insufficient_testing",
              display_name: "Insufficient Testing",
              probability: 0.75,
            },
          ],
        },
      },
    });

    renderWithProviders(<CriticResultDisplay criticResult={result} />);

    expect(
      screen.queryByText("Insufficient Testing"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Expand details"));

    expect(screen.getByText("Insufficient Testing")).toBeInTheDocument();
    expect(screen.getByText("(75%)")).toBeInTheDocument();
    expect(screen.getByText("CRITIC$POTENTIAL_ISSUES")).toBeInTheDocument();
  });

  it("collapses features on second click", async () => {
    const user = userEvent.setup();
    const result = makeCriticResult({
      metadata: {
        categorized_features: {
          agent_behavioral_issues: [
            {
              name: "loop_behavior",
              display_name: "Loop Behavior",
              probability: 0.6,
            },
          ],
        },
      },
    });

    renderWithProviders(<CriticResultDisplay criticResult={result} />);

    await user.click(screen.getByLabelText("Expand details"));
    expect(screen.getByText("Loop Behavior")).toBeInTheDocument();

    await user.click(screen.getByLabelText("Collapse details"));
    expect(screen.queryByText("Loop Behavior")).not.toBeInTheDocument();
  });

  it("renders multiple categories of features", async () => {
    const user = userEvent.setup();
    const result = makeCriticResult({
      metadata: {
        categorized_features: {
          agent_behavioral_issues: [
            {
              name: "incomplete_changes",
              display_name: "Incomplete Changes",
              probability: 0.8,
            },
          ],
          infrastructure_issues: [
            {
              name: "build_failure",
              display_name: "Build Failure",
              probability: 0.4,
            },
          ],
          user_followup_patterns: [
            {
              name: "will_ask_refinement",
              display_name: "Will Ask Refinement",
              probability: 0.55,
            },
          ],
        },
      },
    });

    renderWithProviders(<CriticResultDisplay criticResult={result} />);

    await user.click(screen.getByLabelText("Expand details"));

    expect(screen.getByText("CRITIC$POTENTIAL_ISSUES")).toBeInTheDocument();
    expect(screen.getByText("Incomplete Changes")).toBeInTheDocument();
    expect(screen.getByText("CRITIC$INFRASTRUCTURE")).toBeInTheDocument();
    expect(screen.getByText("Build Failure")).toBeInTheDocument();
    expect(screen.getByText("CRITIC$LIKELY_FOLLOWUP")).toBeInTheDocument();
    expect(screen.getByText("Will Ask Refinement")).toBeInTheDocument();
  });
});
