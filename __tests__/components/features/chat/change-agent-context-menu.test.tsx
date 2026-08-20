import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChangeAgentContextMenu } from "#/components/features/chat/change-agent-context-menu";

describe("ChangeAgentContextMenu", () => {
  it("marks the active mode with hover background and a checkmark", () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="code"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(codeOption).toHaveClass("bg-[var(--oh-interactive-hover)]");
    expect(planOption).not.toHaveClass("bg-[var(--oh-interactive-hover)]");
    expect(codeOption.querySelectorAll("svg")).toHaveLength(2);
    expect(planOption.querySelectorAll("svg")).toHaveLength(1);
  });

  it("shows plan as active when conversation mode is plan", () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(planOption).toHaveClass("bg-[var(--oh-interactive-hover)]");
    expect(codeOption).not.toHaveClass("bg-[var(--oh-interactive-hover)]");
    expect(planOption.querySelectorAll("svg")).toHaveLength(2);
  });

  it("calls onCodeClick and onClose when code is selected", () => {
    const onClose = vi.fn();
    const onCodeClick = vi.fn();

    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={onClose}
        onCodeClick={onCodeClick}
        onPlanClick={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("code-option"));

    expect(onCodeClick).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
