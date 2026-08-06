import type {
  ActivityLogExportFormat,
  Automation,
  AutomationRun,
  AutomationRunExportRow,
} from "#/types/automation";
import { downloadBlob } from "#/utils/utils";
import AutomationService from "#/api/automation-service/automation-service.api";

const EXPORT_PAGE_SIZE = 100;

const CSV_COLUMNS = [
  "run_id",
  "automation_id",
  "automation_name",
  "trigger",
  "start_time",
  "end_time",
  "duration_seconds",
  "status",
  "conversation_id",
  "conversation_url",
  "error",
  // Appended rather than grouped with the other run metrics so existing
  // consumers that read the CSV by column position keep working.
  "cost",
] as const;

export function getActivityLogExportFilename(
  automation: Pick<Automation, "id" | "name">,
  format: ActivityLogExportFormat,
): string {
  const slug = automation.name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || automation.id}.activity-log.${format}`;
}

function csvEscape(value: string): string {
  if (/[",\n\r]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

export function serializeActivityLogRowsCsv(
  rows: AutomationRunExportRow[],
): string {
  const lines: string[] = [CSV_COLUMNS.join(",")];
  for (const row of rows) {
    const values = CSV_COLUMNS.map((key) => {
      const raw = row[key];
      if (raw == null) return "";
      if (key === "trigger") {
        return csvEscape(JSON.stringify(raw));
      }
      return csvEscape(String(raw));
    });
    lines.push(values.join(","));
  }
  return `${lines.join("\n")}\n`;
}

export function buildConversationUrl(
  conversationId: string | null,
  conversationBaseUrl?: string,
): string | null {
  if (!conversationId) return null;
  const path = `/conversations/${conversationId}`;
  if (!conversationBaseUrl) return path;
  return `${conversationBaseUrl.replace(/\/$/, "")}${path}`;
}

export function mapAutomationRunToExportRow(
  run: AutomationRun,
  automation: Pick<Automation, "id" | "name" | "trigger">,
  conversationBaseUrl?: string,
): AutomationRunExportRow {
  const start = run.started_at || null;
  const end = run.completed_at;
  let duration_seconds: number | null = null;
  if (start && end) {
    duration_seconds =
      (new Date(end).getTime() - new Date(start).getTime()) / 1000;
  }

  return {
    run_id: run.id,
    automation_id: automation.id,
    automation_name: automation.name,
    trigger: automation.trigger,
    start_time: start,
    end_time: end,
    duration_seconds,
    status: run.status,
    conversation_id: run.conversation_id,
    conversation_url: buildConversationUrl(
      run.conversation_id,
      conversationBaseUrl,
    ),
    error: run.error_detail,
    // Exports carry the raw number and leave formatting to the consumer; the
    // Activity Log row is the only place that renders it as USD.
    cost: run.cost ?? null,
  };
}

/**
 * Page ``GET /v1/{id}/runs`` until complete, then project export rows locally.
 */
export async function fetchAllActivityLogExportRows(
  automation: Pick<Automation, "id" | "name" | "trigger">,
  conversationBaseUrl?: string,
): Promise<AutomationRunExportRow[]> {
  const rows: AutomationRunExportRow[] = [];
  let offset = 0;
  let total = Number.POSITIVE_INFINITY;

  while (offset < total) {
    const page = await AutomationService.listAutomationRuns(automation.id, {
      limit: EXPORT_PAGE_SIZE,
      offset,
    });

    for (const run of page.runs) {
      rows.push(
        mapAutomationRunToExportRow(run, automation, conversationBaseUrl),
      );
    }
    total = page.total;
    offset += page.runs.length;
    if (page.runs.length === 0) break;
  }

  return rows;
}

/**
 * Page the runs list endpoint and download one CSV or JSON file.
 */
export async function downloadActivityLogExport(options: {
  automation: Pick<Automation, "id" | "name" | "trigger">;
  format: ActivityLogExportFormat;
  conversationBaseUrl?: string;
}): Promise<void> {
  const { automation, format, conversationBaseUrl } = options;
  const rows = await fetchAllActivityLogExportRows(
    automation,
    conversationBaseUrl,
  );

  if (format === "json") {
    downloadBlob(
      new Blob(
        [`${JSON.stringify({ runs: rows, total: rows.length }, null, 2)}\n`],
        { type: "application/json" },
      ),
      getActivityLogExportFilename(automation, "json"),
    );
    return;
  }

  downloadBlob(
    new Blob([serializeActivityLogRowsCsv(rows)], {
      type: "text/csv;charset=utf-8",
    }),
    getActivityLogExportFilename(automation, "csv"),
  );
}
