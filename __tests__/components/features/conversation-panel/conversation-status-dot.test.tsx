import type { ReactNode } from "react";
import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "test-utils";
import { ConversationStatusDot } from "#/components/features/conversation-panel/conversation-status-dot";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({
    children,
    content,
  }: {
    children: ReactNode;
    content: string;
  }) => (
    <div data-testid="styled-tooltip" data-content={content}>
      {children}
    </div>
  ),
}));

describe("ConversationStatusDot", () => {
  it.each([
    [ExecutionStatus.FINISHED, "conversation-status-check", "COMMON$FINISHED"],
    [ExecutionStatus.RUNNING, "conversation-status-working", "COMMON$WORKING"],
    [ExecutionStatus.PAUSED, "conversation-status-paused", "COMMON$PAUSED"],
    [ExecutionStatus.IDLE, "conversation-status-paused", "COMMON$PAUSED"],
    [
      ExecutionStatus.WAITING_FOR_CONFIRMATION,
      "conversation-status-paused",
      "COMMON$PAUSED",
    ],
    [ExecutionStatus.ERROR, "conversation-status-error", "COMMON$ERROR"],
    [ExecutionStatus.STUCK, "conversation-status-error", "COMMON$ERROR"],
  ])("renders %s as %s", (status, testId, tooltipLabel) => {
    renderWithProviders(<ConversationStatusDot executionStatus={status} />);

    expect(screen.getByTestId(testId)).toBeInTheDocument();
    expect(screen.getByTestId("styled-tooltip")).toHaveAttribute(
      "data-content",
      tooltipLabel,
    );
  });

  it("renders the paused state for missing execution status", () => {
    renderWithProviders(<ConversationStatusDot executionStatus={undefined} />);

    expect(
      screen.getByTestId("conversation-status-paused"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("styled-tooltip")).toHaveAttribute(
      "data-content",
      "COMMON$PAUSED",
    );
  });

  it("renders the paused state for null execution status (never-messaged conversation, #16752)", () => {
    renderWithProviders(<ConversationStatusDot executionStatus={null} />);

    // Never-messaged conversations must never render the flashing
    // `conversation-status-working` dot — that's the actively-running state.
    expect(
      screen.queryByTestId("conversation-status-working"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-status-paused"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("styled-tooltip")).toHaveAttribute(
      "data-content",
      "COMMON$PAUSED",
    );
  });

  it("renders archive icon when sandbox is MISSING (archived)", () => {
    renderWithProviders(
      <ConversationStatusDot
        executionStatus={ExecutionStatus.PAUSED}
        sandboxStatus="MISSING"
      />,
    );

    expect(
      screen.getByTestId("conversation-status-archived"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("styled-tooltip")).toHaveAttribute(
      "data-content",
      "COMMON$ARCHIVED",
    );
  });
});
