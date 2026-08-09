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

export interface ScaFinding {
  id: string;
  packageName: string;
  packageVersion: string;
  purl: string;
  cveId: string;
  severity: SecurityFindingSeverity;
  description: string;
}

export interface ScaScanResult {
  findings: ScaFinding[];
  scannedAt: string;
  tool: "dependency-track";
}

export type SecurityScanErrorCode =
  | "opengrep_not_installed"
  | "syft_not_installed"
  | "dependency_track_not_configured"
  | "bom_upload_failed"
  | "bom_processing_failed"
  | "scan_failed"
  | "invalid_output"
  | "runtime_unavailable";

export type ScaScanErrorCode = SecurityScanErrorCode;

export interface SecurityScanError {
  code: SecurityScanErrorCode;
  message: string;
}

export type ScaScanError = SecurityScanError;
