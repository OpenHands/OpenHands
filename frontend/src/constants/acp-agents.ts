/** Known ACP backend servers the GUI can select from. Mirrors SDK's ACPServerKind. */
export type AcpServerKind = "claude-code" | "codex" | "gemini-cli";

export const ACP_SERVER_DISPLAY_NAMES = {
  "claude-code": "Claude Code",
  codex: "Codex",
  "gemini-cli": "Gemini CLI",
} satisfies Record<AcpServerKind, string>;

export const VALID_ACP_SERVERS = new Set<string>(
  Object.keys(ACP_SERVER_DISPLAY_NAMES),
);

export function isAcpServerKind(value: unknown): value is AcpServerKind {
  return typeof value === "string" && VALID_ACP_SERVERS.has(value);
}

export function getAcpServerDisplayName(value: unknown): string | undefined {
  return isAcpServerKind(value) ? ACP_SERVER_DISPLAY_NAMES[value] : undefined;
}

/** Provider API key label per ACP server, shown in the Basic tab. */
export const ACP_API_KEY_LABELS: Partial<Record<AcpServerKind, string>> = {
  "claude-code": "Anthropic API Key",
  codex: "OpenAI API Key",
  "gemini-cli": "Google API Key",
};

/** Provider base URL label per ACP server, shown in the Basic tab alongside the API key. */
export const ACP_BASE_URL_LABELS: Partial<Record<AcpServerKind, string>> = {
  "claude-code": "Anthropic Base URL",
  codex: "OpenAI Base URL",
  "gemini-cli": "Google Base URL",
};

/** Default subprocess command per ACP server. Mirrors SDK's _DEFAULT_ACP_COMMANDS. */
export const ACP_DEFAULT_COMMANDS: Partial<Record<AcpServerKind, string[]>> = {
  "claude-code": ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
  codex: ["npx", "-y", "@zed-industries/codex-acp"],
  "gemini-cli": ["npx", "-y", "@google/gemini-cli", "--acp"],
};
