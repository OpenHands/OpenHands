import { downloadBlob } from "#/utils/utils";
import type { SecurityFindingViewModel } from "#/utils/security-findings-view";

export type SecurityFindingsExportFormat = "csv" | "excel" | "pdf";

export interface SecurityFindingsExportLabels {
  title: string;
  tool: string;
  severity: string;
  reference: string;
  description: string;
  location: string;
  toolSast: string;
  toolSca: string;
  severityHigh: string;
  severityMedium: string;
  severityLow: string;
  severityInfo: string;
}

const EXPORT_COLUMNS = [
  "tool",
  "severity",
  "reference",
  "description",
  "location",
] as const;

function csvEscape(value: string): string {
  if (/[",\n\r]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function xmlEscape(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function htmlEscape(value: string): string {
  return xmlEscape(value).replace(/'/g, "&#39;");
}

function toolLabel(
  tool: SecurityFindingViewModel["tool"],
  labels: SecurityFindingsExportLabels,
): string {
  return tool === "sast" ? labels.toolSast : labels.toolSca;
}

function severityLabel(
  bucket: SecurityFindingViewModel["severityBucket"],
  labels: SecurityFindingsExportLabels,
): string {
  switch (bucket) {
    case "high":
      return labels.severityHigh;
    case "medium":
      return labels.severityMedium;
    case "low":
      return labels.severityLow;
    default:
      return labels.severityInfo;
  }
}

function rowValues(
  row: SecurityFindingViewModel,
  labels: SecurityFindingsExportLabels,
): Record<(typeof EXPORT_COLUMNS)[number], string> {
  return {
    tool: toolLabel(row.tool, labels),
    severity: severityLabel(row.severityBucket, labels),
    reference: row.reference,
    description: row.description,
    location: row.location,
  };
}

export function getSecurityFindingsExportFilename(
  format: SecurityFindingsExportFormat,
  now = new Date(),
): string {
  const stamp = now.toISOString().slice(0, 19).replace(/[:T]/g, "-");
  if (format === "excel") return `security-findings-${stamp}.xls`;
  if (format === "pdf") return `security-findings-${stamp}.pdf`;
  return `security-findings-${stamp}.csv`;
}

export function serializeSecurityFindingsCsv(
  rows: readonly SecurityFindingViewModel[],
  labels: SecurityFindingsExportLabels,
): string {
  const header = [
    labels.tool,
    labels.severity,
    labels.reference,
    labels.description,
    labels.location,
  ]
    .map(csvEscape)
    .join(",");
  const lines = [header];
  for (const row of rows) {
    const values = rowValues(row, labels);
    lines.push(
      EXPORT_COLUMNS.map((key) => csvEscape(values[key])).join(","),
    );
  }
  // UTF-8 BOM so Excel on Windows opens Portuguese accents correctly.
  return `\uFEFF${lines.join("\n")}\n`;
}

/**
 * SpreadsheetML 2003 XML — opens in Excel / LibreOffice without an xlsx lib.
 */
export function serializeSecurityFindingsExcelXml(
  rows: readonly SecurityFindingViewModel[],
  labels: SecurityFindingsExportLabels,
): string {
  const headerCells = [
    labels.tool,
    labels.severity,
    labels.reference,
    labels.description,
    labels.location,
  ]
    .map(
      (value) =>
        `<Cell><Data ss:Type="String">${xmlEscape(value)}</Data></Cell>`,
    )
    .join("");

  const bodyRows = rows
    .map((row) => {
      const values = rowValues(row, labels);
      const cells = EXPORT_COLUMNS.map(
        (key) =>
          `<Cell><Data ss:Type="String">${xmlEscape(values[key])}</Data></Cell>`,
      ).join("");
      return `<Row>${cells}</Row>`;
    })
    .join("");

  return (
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<?mso-application progid="Excel.Sheet"?>\n` +
    `<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" ` +
    `xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n` +
    `<Worksheet ss:Name="Findings"><Table>\n` +
    `<Row>${headerCells}</Row>\n` +
    `${bodyRows}\n` +
    `</Table></Worksheet></Workbook>\n`
  );
}

function buildPrintableHtml(
  rows: readonly SecurityFindingViewModel[],
  labels: SecurityFindingsExportLabels,
): string {
  const header = [
    labels.tool,
    labels.severity,
    labels.reference,
    labels.description,
    labels.location,
  ]
    .map((value) => `<th>${htmlEscape(value)}</th>`)
    .join("");

  const body = rows
    .map((row) => {
      const values = rowValues(row, labels);
      const cells = EXPORT_COLUMNS.map(
        (key) => `<td>${htmlEscape(values[key])}</td>`,
      ).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `<!DOCTYPE html>
<html lang="pt">
<head>
  <meta charset="utf-8" />
  <title>${htmlEscape(labels.title)}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #111; }
    h1 { font-size: 18px; margin: 0 0 16px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; }
    @media print { body { margin: 12px; } }
  </style>
</head>
<body>
  <h1>${htmlEscape(labels.title)}</h1>
  <table>
    <thead><tr>${header}</tr></thead>
    <tbody>${body}</tbody>
  </table>
</body>
</html>`;
}

/**
 * Opens a UTF-8 HTML report and triggers the browser print dialog so the
 * user can save as PDF with full Portuguese accent support.
 */
export function openSecurityFindingsPrintPdf(
  rows: readonly SecurityFindingViewModel[],
  labels: SecurityFindingsExportLabels,
): void {
  const html = buildPrintableHtml(rows, labels);
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const popup = window.open(url, "_blank", "noopener,noreferrer");
  if (!popup) {
    // Popup blocked — fall back to downloading the HTML report.
    downloadBlob(blob, getSecurityFindingsExportFilename("pdf").replace(/\.pdf$/, ".html"));
    URL.revokeObjectURL(url);
    return;
  }
  const revoke = () => URL.revokeObjectURL(url);
  popup.addEventListener("load", () => {
    try {
      popup.focus();
      popup.print();
    } finally {
      window.setTimeout(revoke, 60_000);
    }
  });
  // Safari sometimes never fires load for blob URLs — still revoke later.
  window.setTimeout(revoke, 120_000);
}

export function exportSecurityFindings(
  rows: readonly SecurityFindingViewModel[],
  format: SecurityFindingsExportFormat,
  labels: SecurityFindingsExportLabels,
): void {
  if (format === "pdf") {
    openSecurityFindingsPrintPdf(rows, labels);
    return;
  }
  if (format === "excel") {
    const xml = serializeSecurityFindingsExcelXml(rows, labels);
    downloadBlob(
      new Blob([xml], {
        type: "application/vnd.ms-excel;charset=utf-8",
      }),
      getSecurityFindingsExportFilename("excel"),
    );
    return;
  }
  const csv = serializeSecurityFindingsCsv(rows, labels);
  downloadBlob(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
    getSecurityFindingsExportFilename("csv"),
  );
}
