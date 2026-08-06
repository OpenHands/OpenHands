import { describe, it, expect, vi, beforeEach } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  getActivityLogExportFilename,
  serializeActivityLogRowsCsv,
  mapAutomationRunToExportRow,
  fetchAllActivityLogExportRows,
  downloadActivityLogExport,
} from "#/utils/automation-activity-log-export";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
  type AutomationRunExportRow,
} from "#/types/automation";
import { downloadBlob } from "#/utils/utils";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    listAutomationRuns: vi.fn(),
  },
}));

vi.mock("#/utils/utils", () => ({
  downloadBlob: vi.fn(),
}));

const automation: Pick<Automation, "id" | "name" | "trigger"> = {
  id: "a1",
  name: "Test",
  trigger: { type: "cron", schedule: "0 9 * * *" },
};

const sampleRun = (overrides: Partial<AutomationRun> = {}): AutomationRun => ({
  id: "r1",
  status: AutomationRunStatus.FAILED,
  conversation_id: "c1",
  bash_command_id: null,
  error_detail: "boom",
  started_at: "2026-01-01T09:00:00Z",
  completed_at: "2026-01-01T09:01:00Z",
  ...overrides,
});

const sampleRow = (
  overrides: Partial<AutomationRunExportRow> = {},
): AutomationRunExportRow => ({
  run_id: "r1",
  automation_id: "a1",
  automation_name: "Test",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  start_time: "2026-01-01T09:00:00Z",
  end_time: "2026-01-01T09:01:00Z",
  duration_seconds: 60,
  status: AutomationRunStatus.FAILED,
  conversation_id: "c1",
  conversation_url: "http://localhost:8000/conversations/c1",
  error: "boom",
  cost: null,
  ...overrides,
});

describe("automation-activity-log-export", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("builds a slug filename", () => {
    expect(
      getActivityLogExportFilename(
        { id: "abc", name: "Test Activity Log" },
        "json",
      ),
    ).toBe("test-activity-log.activity-log.json");
  });

  it("falls back to automation id when name has no slug chars", () => {
    expect(
      getActivityLogExportFilename({ id: "abc-123", name: "!!!" }, "csv"),
    ).toBe("abc-123.activity-log.csv");
  });

  it("maps a list run into an export row with conversation URL and duration", () => {
    expect(
      mapAutomationRunToExportRow(
        sampleRun(),
        automation,
        "http://localhost:8000",
      ),
    ).toEqual(sampleRow());
  });

  it("carries the run's accumulated cost into the export row", () => {
    // Arrange
    const run = sampleRun({ cost: 0.4213 });

    // Act
    const row = mapAutomationRunToExportRow(run, automation);

    // Assert
    expect(row.cost).toBe(0.4213);
  });

  it("serializes the cost as a raw number so spreadsheets can total it", () => {
    // Arrange
    const rows: AutomationRunExportRow[] = [sampleRow({ cost: 0.4213 })];

    // Act
    const csv = serializeActivityLogRowsCsv(rows);

    // Assert
    expect(csv.split("\n")[0]).toContain("cost");
    expect(csv.split("\n")[1]).toContain("0.4213");
  });

  it("serializes CSV with conversation URL and escaped fields", () => {
    const rows: AutomationRunExportRow[] = [
      sampleRow({
        error: 'said "boom", then failed',
      }),
    ];

    const csv = serializeActivityLogRowsCsv(rows);
    expect(csv.split("\n")[0]).toContain("conversation_url");
    expect(csv).toContain("http://localhost:8000/conversations/c1");
    expect(csv).toContain("FAILED");
    expect(csv).toContain('"said ""boom"", then failed"');
    expect(csv).toContain('"{""type"":""cron"",""schedule"":""0 9 * * *""}"');
  });

  it("pages listAutomationRuns until complete", async () => {
    vi.mocked(AutomationService.listAutomationRuns)
      .mockResolvedValueOnce({
        runs: [sampleRun({ id: "r1" })],
        total: 2,
      })
      .mockResolvedValueOnce({
        runs: [sampleRun({ id: "r2", conversation_id: "c2" })],
        total: 2,
      });

    const rows = await fetchAllActivityLogExportRows(
      automation,
      "http://localhost:8000",
    );

    expect(rows).toHaveLength(2);
    expect(rows[0].run_id).toBe("r1");
    expect(rows[1].run_id).toBe("r2");
    expect(rows[1].conversation_url).toBe(
      "http://localhost:8000/conversations/c2",
    );
    expect(AutomationService.listAutomationRuns).toHaveBeenCalledTimes(2);
    expect(AutomationService.listAutomationRuns).toHaveBeenNthCalledWith(1, "a1", {
      limit: 100,
      offset: 0,
    });
    expect(AutomationService.listAutomationRuns).toHaveBeenNthCalledWith(2, "a1", {
      limit: 100,
      offset: 1,
    });
  });

  it("downloads JSON after paging list runs", async () => {
    vi.mocked(AutomationService.listAutomationRuns).mockResolvedValueOnce({
      runs: [
        sampleRun({
          status: AutomationRunStatus.COMPLETED,
          error_detail: null,
        }),
      ],
      total: 1,
    });

    await downloadActivityLogExport({
      automation,
      format: "json",
      conversationBaseUrl: "http://localhost:8000",
    });

    expect(AutomationService.listAutomationRuns).toHaveBeenCalledWith("a1", {
      limit: 100,
      offset: 0,
    });
    expect(downloadBlob).toHaveBeenCalledWith(
      expect.any(Blob),
      "test.activity-log.json",
    );
    const blob = vi.mocked(downloadBlob).mock.calls[0][0] as Blob;
    expect(blob.type).toBe("application/json");
  });

  it("downloads CSV after paging list runs", async () => {
    vi.mocked(AutomationService.listAutomationRuns).mockResolvedValueOnce({
      runs: [sampleRun()],
      total: 1,
    });

    await downloadActivityLogExport({
      automation,
      format: "csv",
      conversationBaseUrl: "http://localhost:8000",
    });

    expect(AutomationService.listAutomationRuns).toHaveBeenCalledWith("a1", {
      limit: 100,
      offset: 0,
    });
    expect(downloadBlob).toHaveBeenCalledWith(
      expect.any(Blob),
      "test.activity-log.csv",
    );
    const blob = vi.mocked(downloadBlob).mock.calls[0][0] as Blob;
    expect(blob.type).toBe("text/csv;charset=utf-8");
  });
});
