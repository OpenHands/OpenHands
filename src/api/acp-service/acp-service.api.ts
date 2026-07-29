import { BashClient } from "@openhands/typescript-client/clients";
import type { BashOutput } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";

export type AcpAuthStatus = "authenticated" | "unauthenticated" | "unknown";

// Hard cap (seconds) for a single detection command. These are fast local
// status checks (no npx download), so 10s is ample for a healthy CLI while
// keeping the onboarding spinner short before falling back to "unknown".
const PROBE_TIMEOUT_SECONDS = 10;

interface AcpAuthProbe {
  /** Shell command run on the (local) agent-server host to detect login. */
  command: string;
  /** Classify the command's output into an auth status. */
  classify: (out: BashOutput) => AcpAuthStatus;
}

/** Combined stdout+stderr — CLIs are inconsistent about which stream they use
 * (e.g. ``codex login status`` writes to stderr). */
function streams(out: BashOutput): string {
  return `${out.stdout ?? ""}\n${out.stderr ?? ""}`;
}

// Claude Code: ``claude auth status --json`` prints {"loggedIn": bool, …}. The
// CLI exits non-zero when logged out, so we read the JSON, not the exit code.
// No parseable ``loggedIn`` (e.g. the CLI isn't installed → "command not
// found", empty stdout) ⇒ unknown, so onboarding shows the API-key fields
// rather than guessing.
function classifyClaude(out: BashOutput): AcpAuthStatus {
  let parsed: unknown;
  try {
    parsed = JSON.parse((out.stdout ?? "").trim());
  } catch {
    return "unknown";
  }
  const loggedIn = (parsed as { loggedIn?: unknown } | null)?.loggedIn;
  if (typeof loggedIn === "boolean") {
    return loggedIn ? "authenticated" : "unauthenticated";
  }
  return "unknown";
}

// Codex: ``codex login status`` prints "Logged in using …" / "Not logged in"
// (to stderr), so check both streams. Match "not logged in" first since it
// contains the "logged in" substring. Neither phrase (e.g. CLI missing) ⇒
// unknown.
function classifyCodex(out: BashOutput): AcpAuthStatus {
  const text = streams(out).toLowerCase();
  if (text.includes("not logged in")) return "unauthenticated";
  if (text.includes("logged in")) return "authenticated";
  return "unknown";
}

// Gemini CLI signs in via Google OAuth and has no status command, so we check
// its credentials file. The command echoes present/absent and exits 0 either
// way; anything else (a shell failure) ⇒ unknown.
function classifyGemini(out: BashOutput): AcpAuthStatus {
  // The command echoes exactly `present` / `absent` to stdout, so match trimmed
  // stdout exactly. Reading only stdout (not stderr) means a stray shell
  // warning can't turn a real result into `unknown`.
  const text = (out.stdout ?? "").trim();
  if (text === "present") return "authenticated";
  if (text === "absent") return "unauthenticated";
  return "unknown";
}

// Per-provider login detection, keyed by ``acp_server`` / OnboardingAgentId.
// Providers absent here (OpenHands, custom, unknown) report ``unknown``.
const ACP_AUTH_PROBES: Record<string, AcpAuthProbe> = {
  "claude-code": {
    command: "claude auth status --json",
    classify: classifyClaude,
  },
  codex: {
    command: "codex login status",
    classify: classifyCodex,
  },
  "gemini-cli": {
    command:
      'test -f "$HOME/.gemini/oauth_creds.json" && echo present || echo absent',
    classify: classifyGemini,
  },
};

/**
 * Detects whether the selected ACP provider is already logged in — entirely
 * client-side, with **no dedicated agent-server endpoint**. It runs the
 * provider's own status command (or, for Gemini, a credentials-file check)
 * through the existing agent-server bash endpoint and classifies the output.
 *
 * Gated by the caller to **local backends**: the command runs wherever the
 * agent-server runs, and on a user's own machine the provider CLIs and
 * credential files live at their standard paths. No prompt is sent, so no model
 * tokens are spent. A provider that can't be classified — CLI not installed,
 * unexpected output, or the bash call failing — comes back as ``unknown`` so
 * onboarding falls back to the API-key fields rather than a misleading banner.
 */
class AcpService {
  static async getAuthStatus(server: string): Promise<AcpAuthStatus> {
    const probe = ACP_AUTH_PROBES[server];
    if (!probe) return "unknown";
    const out = await new BashClient(
      getAgentServerClientOptions(),
    ).executeCommand(probe.command, undefined, PROBE_TIMEOUT_SECONDS);
    return probe.classify(out);
  }
}

export default AcpService;
