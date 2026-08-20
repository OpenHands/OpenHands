import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { HomeAutomationRunTooltip } from "#/components/features/home/featured-automations/home-automation-run-tooltip";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { language: "en" },
  }),
}));

const automation: Automation = {
  id: "auto-1",
  name: "Release notes drafter",
  prompt: "Draft release notes",
  trigger: { type: "cron", schedule: "0 12 * * 5" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function makeRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.RUNNING,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-08-01T10:00:00Z",
    completed_at: null,
    ...overrides,
  };
}

function makeState(run: AutomationRun | null): LatestAutomationRunState {
  return {
    latestRun: run,
    recentRuns: run ? [run] : [],
    isLoading: false,
    isError: false,
  };
}

// The row that opens this hovercard clips a long phase; the hovercard is the
// place where the whole phase has to be readable.
const LONG_LABEL =
  "Rendering the changelog for 37 merged pull requests across 4 repositories";

describe("HomeAutomationRunTooltip — phase", () => {
  it("shows the phase of a running run in full", () => {
    render(
      <HomeAutomationRunTooltip
        automation={automation}
        runState={makeState(
          makeRun({ phase_code: "drafting_notes", phase_label: LONG_LABEL }),
        )}
      />,
    );

    expect(screen.getByTestId("run-phase-row")).toHaveTextContent(LONG_LABEL);
  });

  it("translates a phase code the frontend knows instead of printing it raw", () => {
    render(
      <HomeAutomationRunTooltip
        automation={automation}
        runState={makeState(
          makeRun({ phase_code: "bundle_upload", phase_label: null }),
        )}
      />,
    );

    expect(screen.getByTestId("run-phase-row")).toHaveTextContent(
      "AUTOMATIONS$DETAIL$PHASE_BUNDLE_UPLOAD",
    );
  });

  it("omits the phase row for a finished run, matching every other surface", () => {
    render(
      <HomeAutomationRunTooltip
        automation={automation}
        runState={makeState(
          makeRun({
            status: AutomationRunStatus.COMPLETED,
            completed_at: "2026-08-01T10:05:00Z",
            phase_code: "running_agent",
            phase_label: null,
          }),
        )}
      />,
    );

    expect(screen.queryByTestId("run-phase-row")).not.toBeInTheDocument();
  });

  it("omits the phase row when the run never reported one", () => {
    render(
      <HomeAutomationRunTooltip
        automation={automation}
        runState={makeState(makeRun())}
      />,
    );

    expect(screen.queryByTestId("run-phase-row")).not.toBeInTheDocument();
  });
});
