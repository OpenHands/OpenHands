import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import type { LoopTrigger, TriggerEvent } from "#/api/loop-service/loop-types";
import { TriggerEventsFeed } from "#/components/features/loops/trigger-events-feed";
import { I18nKey } from "#/i18n/declaration";

const TRIGGER: LoopTrigger = {
  id: "trigger-1",
  project_id: "proj-1",
  loop_definition_id: "def-commit",
  trigger_type: "on_commit",
  schedule_type: null,
  cron_expr: null,
  interval_seconds: null,
  payload: {},
  enabled: true,
  last_fired_at: "2026-01-01T00:00:00Z",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const EVENT: TriggerEvent = {
  id: "event-1",
  trigger_id: "trigger-1",
  loop_run_id: "run-1",
  fired_at: "2026-01-01T00:00:00Z",
  status: "fired",
  reason: "on_commit",
};

describe("TriggerEventsFeed", () => {
  it("renders recent trigger events with type, status, and run link", () => {
    renderWithProviders(
      <TriggerEventsFeed events={[EVENT]} triggers={[TRIGGER]} />,
    );

    expect(screen.getByTestId("loop-events-feed")).toBeInTheDocument();
    expect(screen.getByTestId("loop-event-event-1")).toHaveTextContent(
      I18nKey.LOOPS$TYPE_ON_COMMIT,
    );
    expect(screen.getByTestId("loop-event-status-event-1")).toHaveTextContent(
      I18nKey.LOOPS$STATUS_FIRED,
    );
    expect(screen.getByTestId("loop-event-run-event-1")).toHaveTextContent(
      "run-1",
    );
  });

  it("renders an empty state when there are no events", () => {
    renderWithProviders(<TriggerEventsFeed events={[]} triggers={[]} />);

    expect(screen.getByTestId("loop-events-empty")).toHaveTextContent(
      I18nKey.LOOPS$EMPTY_EVENTS,
    );
  });
});
