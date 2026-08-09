/**
 * Upstream Agent Canvas update helpers.
 *
 * Heimdall fork: version polling against OpenHands npm is disabled. Keep the
 * constants for any leftover docs/tests; `fetchLatestAgentCanvasVersion` is a
 * hard no-op so nothing can hit the registry by accident.
 */

export const AGENT_CANVAS_RELEASE_NOTES_URL =
  "https://github.com/OpenHands/OpenHands/releases";

/** Literal shell commands — intentionally not localized. */
export const AGENT_CANVAS_UPDATE_COMMANDS = {
  npm: "npm install -g @openhands/agent-canvas@latest",
  docker: "docker pull ghcr.io/openhands/agent-canvas:latest",
} as const;

export async function fetchLatestAgentCanvasVersion(
  _signal?: AbortSignal,
): Promise<string> {
  throw new Error(
    "Upstream Agent Canvas version checks are disabled in this fork",
  );
}
