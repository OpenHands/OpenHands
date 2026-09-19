import type { AcpAuthStatus } from "#/hooks/query/use-acp-auth-status";

/**
 * Resolved display state for the ACP auth banner.
 *
 * - ``"signed-in"`` — the host-login probe confirmed a session (native backend).
 * - ``"checking"`` — the first probe is still in flight.
 * - ``"configured"`` — the probe can't confirm a login (e.g. a container with no
 *   interactive CLI, or a cloud backend where the probe doesn't run), but a
 *   credential for the provider exists in the active backend's secret store.
 * - ``"none"`` — no confirmed login and no stored credential; show nothing.
 */
export type AcpAuthDisplay = "signed-in" | "checking" | "configured" | "none";

interface AcpAuthDisplayInput {
  /** Result of the host-login probe (only meaningful on a native backend). */
  status: AcpAuthStatus;
  /** True while the first probe for this provider is in flight. */
  isChecking: boolean;
  /**
   * Whether a credential for the provider exists in the active backend's secret
   * store. Unlike {@link status}, this signal is available on Docker and cloud
   * backends, so it can convey an accurate state where the probe goes silent.
   */
  credentialsConfigured: boolean;
}

/**
 * Decide what the ACP auth banner should show, given the (host-only) login
 * probe and the backend-truthful "a credential is stored" signal.
 *
 * Precedence preserves the existing banner behavior — a probe-confirmed login
 * wins, then the in-flight spinner — and only adds ``"configured"`` in the gap
 * that previously rendered nothing. A stored credential never reports as
 * ``"signed-in"``: only the probe can confirm an actual host login. See #1244.
 */
export function resolveAcpAuthDisplay({
  status,
  isChecking,
  credentialsConfigured,
}: AcpAuthDisplayInput): AcpAuthDisplay {
  if (status === "authenticated") return "signed-in";
  if (isChecking) return "checking";
  if (credentialsConfigured) return "configured";
  return "none";
}
