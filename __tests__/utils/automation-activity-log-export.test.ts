import { describe, it, expect, vi, beforeEach } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  getActivityLogExportFilename,
  serializeActivityLogRowsCsv,
  fetchAllActivityLogExportRows,
  downloadActivityLogExport,
} from "#/utils/automation-activity-log-export";
import {
  AutomationRunStatus,
  type AutomationRunExportRow,
} from "#/types/automation";
import { downloadBlob } from "#/utils/utils";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    exportAutomationRuns: vi.fn(),
  },
}));

vi.mock("#/utils/utils", () => ({
  downloadBlob: vi.fn(),
}));

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

  it("pages the export endpoint until empty", async () => {
    const page1 = sampleRow({ run_id: "r1" });
    const page2 = sampleRow({ run_id: "r2", conversation_id: "c2" });

    vi.mocked(AutomationService.exportAutomationRuns)
      .mockResolvedValueOnce({
        runs: [page1],
        total: 2,
        limit: 500,
        offset: 0,
      })
      .mockResolvedValueOnce({
        runs: [page2],
        total: 2,
        limit: 500,
        offset: 1,
      });

    const rows = await fetchAllActivityLogExportRows(
      "a1",
      "http://localhost:8000",
    );

    expect(rows).toEqual([page1, page2]);
    expect(AutomationService.exportAutomationRuns).toHaveBeenCalledTimes(2);
    expect(AutomationService.exportAutomationRuns).toHaveBeenNthCalledWith(
      1,
      "a1",
      expect.objectContaining({
        limit: 500,
        offset: 0,
        conversation_base_url: "http://localhost:8000",
      }),
    );
    expect(AutomationService.exportAutomationRuns).toHaveBeenNthCalledWith(
      2,
      "a1",
      expect.objectContaining({
        offset: 1,
      }),
    );
  });

  it("downloads JSON after paging the export endpoint", async () => {
    vi.mocked(AutomationService.exportAutomationRuns).mockResolvedValueOnce({
      runs: [sampleRow({ status: AutomationRunStatus.COMPLETED, error: null })],
      total: 1,
      limit: 500,
      offset: 0,
    });

    await downloadActivityLogExport({
      automation: { id: "a1", name: "Test" },
      format: "json",
      conversationBaseUrl: "http://localhost:8000",
    });

    expect(AutomationService.exportAutomationRuns).toHaveBeenCalledWith(
      "a1",
      expect.objectContaining({
        conversation_base_url: "http://localhost:8000",
      }),
    );
    expect(downloadBlob).toHaveBeenCalledWith(
      expect.any(Blob),
      "test.activity-log.json",
    );
    const blob = vi.mocked(downloadBlob).mock.calls[0][0] as Blob;
    expect(blob.type).toBe("application/json");
  });

  it("downloads CSV after paging the export endpoint", async () => {
    vi.mocked(AutomationService.exportAutomationRuns).mockResolvedValueOnce({
      runs: [sampleRow()],
      total: 1,
      limit: 500,
      offset: 0,
    });

    await downloadActivityLogExport({
      automation: { id: "a1", name: "Test" },
      format: "csv",
      conversationBaseUrl: "http://localhost:8000",
    });

    expect(AutomationService.exportAutomationRuns).toHaveBeenCalledWith(
      "a1",
      expect.objectContaining({
        conversation_base_url: "http://localhost:8000",
      }),
    );
    expect(downloadBlob).toHaveBeenCalledWith(
      expect.any(Blob),
      "test.activity-log.csv",
    );
    const blob = vi.mocked(downloadBlob).mock.calls[0][0] as Blob;
    expect(blob.type).toBe("text/csv;charset=utf-8");
  });
});
