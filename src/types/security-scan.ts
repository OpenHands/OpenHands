export type SecurityFindingSeverity =
  | "CRITICAL"
  | "HIGH"
  | "ERROR"
  | "WARNING"
  | "MEDIUM"
  | "LOW"
  | "INFO"
  | "EXPERIMENT"
  | "INVENTORY";

export interface SecurityFinding {
  id: string;
  ruleId: string;
  message: string;
  severity: SecurityFindingSeverity;
  filePath: string;
  startLine: number;
  startCol: number;
  endLine: number;
  endCol: number;
}

export interface SecurityScanResult {
  findings: SecurityFinding[];
  scannedAt: string;
  tool: "opengrep";
}

export type SecurityScanErrorCode =
  | "opengrep_not_installed"
  | "scan_failed"
  | "invalid_output"
  | "runtime_unavailable";

export interface SecurityScanError {
  code: SecurityScanErrorCode;
  message: string;
}
