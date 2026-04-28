export type AcpServerKind = "claude-code" | "codex" | "gemini-cli";

export const ACP_SERVER_DISPLAY_NAMES: Record<AcpServerKind | string, string> =
  {
    "claude-code": "Claude Code",
    codex: "Codex",
    "gemini-cli": "Gemini CLI",
  };

export const VALID_ACP_SERVERS = new Set<string>(
  Object.keys(ACP_SERVER_DISPLAY_NAMES),
);

export function isAcpServerKind(value: unknown): value is AcpServerKind {
  return typeof value === "string" && VALID_ACP_SERVERS.has(value);
}
