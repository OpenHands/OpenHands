import { useQuery } from "@tanstack/react-query";
import AcpService, {
  type AcpAuthStatus,
} from "#/api/acp-service/acp-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export type { AcpAuthStatus };

/**
 * Probe whether the selected ACP provider is already authenticated on the
 * (local) agent-server — by a subscription login (Claude Pro/Max, ChatGPT,
 * Google) or a pre-set API key.
 *
 * Detection is entirely client-side: {@link AcpService.getAuthStatus} runs the
 * provider's own status command (Claude: ``claude auth status``; Codex:
 * ``codex login status``; Gemini: a credentials-file check) through the
 * agent-server bash endpoint and classifies the output — no dedicated
 * endpoint, no prompt, no model tokens. Anything it can't classify (CLI not
 * installed, unexpected output, the bash call failing) is ``unknown``.
 */
async function probeAcpAuth(providerKey: string): Promise<AcpAuthStatus> {
  try {
    return await AcpService.getAuthStatus(providerKey);
  } catch {
    // The bash endpoint is unreachable or errored: fall back to "unknown" so
    // the caller shows the API-key fields rather than falsely claiming
    // "not logged in".
    return "unknown";
  }
}

interface UseAcpAuthStatusOptions {
  /**
   * Gate the probe to when the consuming surface is actually visible — the
   * onboarding modal mounts every slide at once, so without this the probe
   * would fire (and spin a subprocess) before the user reaches the step and
   * before the backend is confirmed connected. Defaults to ``true``.
   */
  enabled?: boolean;
}

/**
 * React Query wrapper around {@link probeAcpAuth}.
 *
 * Gated to **local backends only**: the detection command runs wherever the
 * agent-server runs, and a provider CLI / credentials file is only reliably
 * present on the user's own machine. On a remote/cloud backend they're ~never
 * there, so we skip the probe, return ``"unknown"``, and let the caller fall
 * back to the (already optional) API-key fields.
 *
 * Eligibility is intentionally *not* tied to whether the provider has API-key
 * fields: subscription/OAuth providers (e.g. Gemini) are detectable too, and an
 * unknown ``providerKey`` simply classifies as ``"unknown"``. The caller renders
 * this hook only for ACP providers, so any local backend is probeable.
 *
 * The probe runs a subprocess on the agent-server, so it is not polled on an
 * interval — but the result is no longer cached for the whole session: the
 * query is stale immediately and refetches on mount and on window focus, so
 * signing in or out via the provider CLI outside Canvas shows up without a
 * page reload (#19730). React Query keeps the previous status visible while a
 * refetch is in flight, so the banner never flashes back to "checking".
 */
export function useAcpAuthStatus(
  providerKey: string | null | undefined,
  options: UseAcpAuthStatusOptions = {},
) {
  const { enabled = true } = options;
  const active = useActiveBackend();
  const isLocal = active.backend.kind === "local";
  const isSupported = isLocal;
  const queryEnabled = enabled && isSupported && !!providerKey;

  const query = useQuery<AcpAuthStatus, Error>({
    // ``providerKey`` both discriminates the cache (so switching providers
    // re-probes) and parameterizes the probe — ``queryEnabled`` guarantees it
    // is non-empty whenever the query runs.
    queryKey: ["acp-auth-status", active.backend.id, providerKey],
    queryFn: () => probeAcpAuth(providerKey as string),
    enabled: queryEnabled,
    // Always stale: remounts (Settings → Agent closed and reopened, the
    // onboarding credentials step re-entered) and window focus re-probe, so
    // the banner tracks CLI sign-in/sign-out done outside Canvas. The
    // subprocess is cheap and only runs on those user-visible transitions.
    staleTime: 0,
    gcTime: 1000 * 60 * 15,
    // ``probeAcpAuth`` always resolves (it catches internally → "unknown"), so
    // this is redundant today — kept as a guard so the probe still never retries
    // if that inner catch is ever removed.
    retry: false,
  });

  return {
    status: query.data ?? "unknown",
    /** True while the first probe for this provider is in flight. */
    isChecking: queryEnabled && query.isFetching && query.data === undefined,
    /** Whether a probe can run at all on this backend (local backends only). */
    isSupported,
  };
}
