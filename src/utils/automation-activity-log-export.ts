import type {
  ActivityLogExportFormat,
  Automation,
  AutomationRunExportRow,
} from "#/types/automation";
import { downloadBlob } from "#/utils/utils";
import AutomationService from "#/api/automation-service/automation-service.api";

const EXPORT_PAGE_SIZE = 500;

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

/**
 * Page the automation ``/runs/export`` JSON endpoint until complete.
 * Does not assemble from the UI list endpoint.
 */
export async function fetchAllActivityLogExportRows(
  id: string,
  conversationBaseUrl?: string,
): Promise<AutomationRunExportRow[]> {
  const runs: AutomationRunExportRow[] = [];
  let offset = 0;
  let total = Number.POSITIVE_INFINITY;

  while (offset < total) {
    const page = await AutomationService.exportAutomationRuns(id, {
      limit: EXPORT_PAGE_SIZE,
      offset,
      conversation_base_url: conversationBaseUrl,
    });

    runs.push(...page.runs);
    total = page.total;
    offset += page.runs.length;
    if (page.runs.length === 0) break;
  }

  return runs;
}

/**
 * Page the Activity Log export endpoint and download one CSV or JSON file.
 */
export async function downloadActivityLogExport(options: {
  automation: Pick<Automation, "id" | "name">;
  format: ActivityLogExportFormat;
  conversationBaseUrl?: string;
}): Promise<void> {
  const { automation, format, conversationBaseUrl } = options;
  const rows = await fetchAllActivityLogExportRows(
    automation.id,
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
