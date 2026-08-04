import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ActivityLogSection } from "#/components/features/automations/detail/activity-log-section";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import { downloadActivityLogExport } from "#/utils/automation-activity-log-export";

const { trackAutomationActivityLogExported } = vi.hoisted(() => ({
  trackAutomationActivityLogExported: vi.fn(),
}));

vi.mock("#/utils/automation-activity-log-export", () => ({
  downloadActivityLogExport: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackAutomationActivityLogExported,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "default-local", kind: "local" },
    orgId: null,
  }),
}));

vi.mock("#/hooks/query/use-automation-detail", () => ({
  useAutomationRuns: vi.fn(),
}));

import { useAutomationRuns } from "#/hooks/query/use-automation-detail";

const automation: Automation = {
  id: "a1",
  name: "Test Activity Log",
  trigger: { type: "cron", schedule: "0 9 * * 1-5", timezone: "UTC" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  prompt: "hello",
};

const run: AutomationRun = {
  id: "r1",
  status: AutomationRunStatus.COMPLETED,
  conversation_id: "c1",
  bash_command_id: "b1",
  error_detail: null,
  started_at: "2026-01-01T09:00:00Z",
  completed_at: "2026-01-01T09:01:00Z",
};

function renderSection() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActivityLogSection automation={automation} />
    </QueryClientProvider>,
  );
}

describe("ActivityLogSection export", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: [run], total: 1 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);
  });

  it("exports JSON via the activity-log export helper and tracks it", async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("activity-log-export-json"));

    await waitFor(() => {
      expect(downloadActivityLogExport).toHaveBeenCalledWith({
        automation,
        format: "json",
        conversationBaseUrl: window.location.origin,
      });
    });
    expect(trackAutomationActivityLogExported).toHaveBeenCalledWith({
      backendKind: "local",
      format: "json",
    });
  });

  it("exports CSV via the activity-log export helper and tracks it", async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("activity-log-export-csv"));

    await waitFor(() => {
      expect(downloadActivityLogExport).toHaveBeenCalledWith({
        automation,
        format: "csv",
        conversationBaseUrl: window.location.origin,
      });
    });
    expect(trackAutomationActivityLogExported).toHaveBeenCalledWith({
      backendKind: "local",
      format: "csv",
    });
  });

  it("disables export when there are no runs", () => {
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: [], total: 0 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);

    renderSection();

    expect(screen.getByTestId("activity-log-export-json")).toBeDisabled();
    expect(screen.getByTestId("activity-log-export-csv")).toBeDisabled();
  });
});
