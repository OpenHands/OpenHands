import { describe, expect, it, vi } from "vitest";
import { toRunSummaryState } from "#/components/features/automations/to-latest-run-state";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";

describe("to-latest-run-state", () => {
  function createRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
    return {
      id: "run-1",
      status: AutomationRunStatus.COMPLETED,
      conversation_id: null,
      bash_command_id: null,
      error_detail: null,
      started_at: "2026-01-02T00:00:00Z",
      completed_at: "2026-01-02T00:03:00Z",
      ...overrides,
    };
  }

  describe("toRunSummaryState", () => {
    it("summarizes recent runs for the dashboard stats footer", () => {
      const completed = createRun();
      const failed = createRun({
        id: "run-2",
        status: AutomationRunStatus.FAILED,
        started_at: "2026-01-02T00:10:00Z",
        completed_at: "2026-01-02T00:12:00Z",
      });

      const state = toRunSummaryState({
        latestRun: completed,
        recentRuns: [completed, failed],
        total: 8,
        isLoading: false,
        isError: false,
      });

      expect(state.summary?.total).toBe(8);
      expect(state.summary?.recentSuccessRate).toBe(0.5);
      expect(state.summary?.averageDurationMs).toBe(150_000);
    });

    it("keeps an empty loading state from showing fake totals", () => {
      const state = toRunSummaryState({
        latestRun: null,
        recentRuns: [],
        isLoading: true,
        isError: false,
      });

      expect(state.summary).toBeNull();
      expect(state.isLoading).toBe(true);
    });
  });
});

describe("ToggleSwitch", () => {
  it("calls onToggle when clicked", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(
      <ToggleSwitch enabled={false} label="Toggle test" onToggle={onToggle} />,
    );

    await user.click(screen.getByRole("switch"));

    expect(onToggle).toHaveBeenCalledOnce();
  });

  it("reflects enabled state via aria-checked", () => {
    render(<ToggleSwitch enabled label="Toggle test" onToggle={vi.fn()} />);

    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
  });

  it("ignores pointer and keyboard activation when disabled", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(
      <ToggleSwitch
        enabled={false}
        label="Toggle test"
        onToggle={onToggle}
        disabled
      />,
    );

    const toggle = screen.getByRole("switch");
    await user.click(toggle);
    toggle.focus();
    await user.keyboard(" ");

    expect(toggle).toBeDisabled();
    expect(toggle).not.toHaveFocus();
    expect(onToggle).not.toHaveBeenCalled();
  });

  it("stops event propagation on click", async () => {
    const parentClick = vi.fn();
    const user = userEvent.setup();
    render(
      <div onClick={parentClick} onKeyDown={parentClick} role="presentation">
        <ToggleSwitch enabled={false} label="Toggle test" onToggle={vi.fn()} />
      </div>,
    );

    await user.click(screen.getByRole("switch"));

    expect(parentClick).not.toHaveBeenCalled();
  });
});
