import type { ExtendedMCPTestFailureKind } from "#/types/mcp-server";

/** How strongly the last successful check proved the server works. */
export type McpHealthVerification =
  // A representative read-only tool call succeeded — credentials exercised.
  | "verified"
  // Connect + tools/list only — proves connectivity, not credentials.
  | "connectivity-only";

export type McpServerHealth =
  | { status: "unchecked" }
  | { status: "checking"; checkId: number }
  | {
      status: "healthy";
      verification: McpHealthVerification;
      toolCount: number;
      checkedAt: number;
    }
  | {
      status: "failed";
      kind: ExtendedMCPTestFailureKind;
      /** Redacted, display-safe error detail. */
      error: string;
      checkedAt: number;
    };

export const UNCHECKED_MCP_HEALTH: McpServerHealth = { status: "unchecked" };
