import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import LoopService from "#/api/loop-service/loop-service.api";
import type {
  LoopDefinition,
  LoopRun,
  LoopTrigger,
} from "#/api/loop-service/loop-types";
import { LoopsOverview } from "#/components/features/loops/loops-overview";
import { I18nKey } from "#/i18n/declaration";

const DEFINITION: LoopDefinition = {
  id: "def-commit",
  name: "commit-loop",
  project_id: "proj-1",
  stages: [{ name: "lint", cmd: null, iterative: true }],
  max_iterations: 10,
  max_cost_usd: 5,
  on_failure: "auto_fix",
  config: {},
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const TRIGGER: LoopTrigger = {
  id: "trigger-1",
  project_id: "proj-1",
  loop_definition_id: "def-commit",
  trigger_type: "manual",
  schedule_type: null,
  cron_expr: null,
  interval_seconds: null,
  payload: {},
  enabled: true,
  last_fired_at: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const RUN: LoopRun = {
  id: "run-1",
  definition_id: "def-commit",
  project_id: "proj-1",
  session_id: null,
  worktree_dir: null,
  status: "passed",
  current_stage: null,
  iteration: 1,
  total_cost_usd: 0.01,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  stages: [],
};

describe("LoopsOverview", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(LoopService, "listDefinitions").mockResolvedValue([DEFINITION]);
    vi.spyOn(LoopService, "listTriggers").mockResolvedValue([TRIGGER]);
    vi.spyOn(LoopService, "listEvents").mockResolvedValue([]);
    vi.spyOn(LoopService, "listRuns").mockResolvedValue([]);
  });

  it("patches enabled when the trigger toggle is clicked", async () => {
    const user = userEvent.setup();
    const updateTrigger = vi
      .spyOn(LoopService, "updateTrigger")
      .mockResolvedValue({ ...TRIGGER, enabled: false });

    renderWithProviders(<LoopsOverview />);

    expect(
      await screen.findByTestId("loop-trigger-trigger-1"),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("switch", { name: I18nKey.LOOPS$ENABLE }),
    );

    await waitFor(() => {
      expect(updateTrigger).toHaveBeenCalledWith("trigger-1", {
        enabled: false,
      });
    });
  });

  it("fires the trigger when Run now is clicked", async () => {
    const user = userEvent.setup();
    const fireTrigger = vi.spyOn(LoopService, "fireTrigger").mockResolvedValue({
      event: {
        id: "event-1",
        trigger_id: "trigger-1",
        loop_run_id: "run-1",
        fired_at: "2026-01-01T00:00:00Z",
        status: "fired",
        reason: "manual",
      },
      run: RUN,
    });

    renderWithProviders(<LoopsOverview />);

    await user.click(
      await screen.findByTestId("loop-trigger-run-now-trigger-1"),
    );

    await waitFor(() => {
      expect(fireTrigger).toHaveBeenCalledWith("trigger-1");
    });
  });
});
