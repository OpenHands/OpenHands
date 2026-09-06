import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import type { LoopDefinition } from "#/api/loop-service/loop-types";
import {
  TriggerForm,
  validateTriggerForm,
} from "#/components/features/loops/trigger-form";
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

describe("validateTriggerForm", () => {
  it("requires a cron expression or interval for scheduled triggers", () => {
    expect(
      validateTriggerForm({
        project_id: "proj-1",
        loop_definition_id: "def-commit",
        trigger_type: "scheduled",
        schedule_type: "cron",
        cron_expr: "",
        interval_seconds: "",
      }),
    ).toBe(I18nKey.LOOPS$VALIDATION_SCHEDULE);
    expect(
      validateTriggerForm({
        project_id: "proj-1",
        loop_definition_id: "def-commit",
        trigger_type: "scheduled",
        schedule_type: "interval",
        cron_expr: "",
        interval_seconds: "30",
      }),
    ).toBeNull();
  });

  it("does not require a schedule for manual triggers", () => {
    expect(
      validateTriggerForm({
        project_id: "proj-1",
        loop_definition_id: "def-commit",
        trigger_type: "manual",
        schedule_type: "",
        cron_expr: "",
        interval_seconds: "",
      }),
    ).toBeNull();
  });
});

describe("TriggerForm", () => {
  it("hides schedule fields for manual triggers", () => {
    renderWithProviders(
      <TriggerForm
        definitions={[DEFINITION]}
        onSubmit={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("loop-trigger-form")).toBeInTheDocument();
    expect(
      screen.queryByTestId("loop-trigger-schedule-type"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("loop-trigger-cron")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("loop-trigger-interval"),
    ).not.toBeInTheDocument();
  });

  it("blocks scheduled submit until cron or interval is set", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWithProviders(
      <TriggerForm
        definitions={[DEFINITION]}
        onSubmit={onSubmit}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByLabelText(I18nKey.LOOPS$TYPE));
    await user.click(await screen.findByText(I18nKey.LOOPS$TYPE_SCHEDULED));
    await user.click(screen.getByTestId("loop-trigger-save"));

    expect(screen.getByTestId("loop-trigger-schedule-error")).toHaveTextContent(
      I18nKey.LOOPS$VALIDATION_SCHEDULE,
    );
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
