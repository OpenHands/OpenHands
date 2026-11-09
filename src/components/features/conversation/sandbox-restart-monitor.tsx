import { useSandboxRestartWarning } from "#/hooks/use-sandbox-restart-warning";

/** Renders nothing. Mounted inside the WebSocket provider to watch for sandbox restarts. */
export function SandboxRestartMonitor() {
  useSandboxRestartWarning();
  return null;
}
