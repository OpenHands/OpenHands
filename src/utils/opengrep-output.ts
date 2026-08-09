import type {
  SecurityFinding,
  SecurityFindingSeverity,
  SecurityScanResult,
} from "#/types/security-scan";

interface OpengrepPosition {
  line: number;
  col: number;
}

interface OpengrepMatchExtra {
  message?: string;
  severity?: string;
  lines?: string;
}

interface OpengrepMatch {
  check_id: string;
  path: string;
  start: OpengrepPosition;
  end: OpengrepPosition;
  extra?: OpengrepMatchExtra;
}

interface OpengrepCliOutput {
  results?: OpengrepMatch[];
}

const VALID_SEVERITIES: ReadonlySet<SecurityFindingSeverity> = new Set([
  "CRITICAL",
  "HIGH",
  "ERROR",
  "WARNING",
  "MEDIUM",
  "LOW",
  "INFO",
  "EXPERIMENT",
  "INVENTORY",
]);

function normalizeSeverity(value: string | undefined): SecurityFindingSeverity {
  const upper = String(value ?? "INFO").toUpperCase();
  if (VALID_SEVERITIES.has(upper as SecurityFindingSeverity)) {
    return upper as SecurityFindingSeverity;
  }
  return "INFO";
}

function buildFindingId(match: OpengrepMatch, index: number): string {
  return `${match.check_id}:${match.path}:${match.start.line}:${match.start.col}:${index}`;
}

export function parseOpengrepJsonOutput(raw: string): SecurityScanResult {
  let parsed: OpengrepCliOutput;
  try {
    parsed = JSON.parse(raw) as OpengrepCliOutput;
  } catch {
    throw new Error("invalid_output");
  }

  const results = Array.isArray(parsed.results) ? parsed.results : [];
  const findings: SecurityFinding[] = results.map((match, index) => ({
    id: buildFindingId(match, index),
    ruleId: match.check_id,
    message: match.extra?.message?.trim() || match.check_id,
    severity: normalizeSeverity(match.extra?.severity),
    filePath: match.path,
    startLine: match.start.line,
    startCol: match.start.col,
    endLine: match.end.line,
    endCol: match.end.col,
  }));

  return {
    findings,
    scannedAt: new Date().toISOString(),
    tool: "opengrep",
  };
}

export const OPENGREP_SCAN_COMMAND = [
  'OPENGREP_BIN=""',
  'if command -v opengrep >/dev/null 2>&1; then',
  '  OPENGREP_BIN="opengrep"',
  'elif [ -x "$HOME/.opengrep/cli/latest/opengrep" ]; then',
  '  OPENGREP_BIN="$HOME/.opengrep/cli/latest/opengrep"',
  "fi",
  'if [ -z "$OPENGREP_BIN" ]; then',
  "  curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash -s -- 2>/dev/null || true",
  '  if [ -x "$HOME/.opengrep/cli/latest/opengrep" ]; then',
  '    OPENGREP_BIN="$HOME/.opengrep/cli/latest/opengrep"',
  "  fi",
  "fi",
  'if [ -z "$OPENGREP_BIN" ]; then',
  '  echo "opengrep_not_installed" >&2',
  "  exit 127",
  "fi",
  '"$OPENGREP_BIN" scan --json --config auto .',
].join("\n");
