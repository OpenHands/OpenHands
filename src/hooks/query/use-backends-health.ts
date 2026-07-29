import React from "react";
import axios from "axios";
import { useQueries } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import {
  ServerClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import {
  getCloudOrganizations,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import {
  assertAgentServerVersionIsSupported,
  isSdkHttpStatusError,
} from "#/api/agent-server-compatibility";
import type { Backend } from "#/api/backend-registry/types";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  isCorsOrNetworkError,
  isCorsOrNetworkErrorMessage,
} from "#/utils/user-facing-error";
import {
  getHealthSnapshot,
  recordBackendFailure,
  recordBackendSuccess,
  subscribeBackendHealth,
} from "#/api/backend-registry/health-store";
import { MAX_CONSECUTIVE_FAILURES } from "#/api/backend-registry/health-storage";

const REFRESH_INTERVAL_MS = 10000;
const PROBE_TIMEOUT_MS = 4000;
export const INVALID_BACKEND_API_KEY_ERROR = "Invalid API key";
export const MISSING_BACKEND_API_KEY_ERROR = "API key required";
export const CLOUD_BACKEND_API_KEY_OR_NETWORK_ERROR =
  "Cloud API key or network issue";
export const CLOUD_BACKEND_LOGGED_OUT_ERROR = "Logged out";

export function isInvalidBackendApiKeyHealthError(
  error: string | null | undefined,
): boolean {
  return error === INVALID_BACKEND_API_KEY_ERROR;
}

export function isMissingBackendApiKeyHealthError(
  error: string | null | undefined,
): boolean {
  return error === MISSING_BACKEND_API_KEY_ERROR;
}

export function isCloudBackendApiKeyOrNetworkHealthError(
  error: string | null | undefined,
): boolean {
  return error === CLOUD_BACKEND_API_KEY_OR_NETWORK_ERROR;
}

function hasMissingBackendApiKey(backend: Backend): boolean {
  return (
    backend.kind === "cloud" &&
    backend.authMode !== "cookie" &&
    !backend.apiKey.trim()
  );
}

export function isCloudBackendLoggedOutHealthError(
  error: string | null | undefined,
): boolean {
  return error === CLOUD_BACKEND_LOGGED_OUT_ERROR;
}

/**
 * Probe a single backend for connectivity. The probe path differs by
 * backend kind:
 *
 *  - Local agent-server: GET `/api/settings`, then `/server_info` via the
 *    typescript-client. The settings call validates the configured session
 *    API key; the server info call validates the version compatibility floor.
 *  - Cloud: GET `/api/keys/current` directly against the cloud host. That
 *    endpoint is lightweight, requires auth, and `getCurrentCloudApiKey`
 *    already absorbs the legacy-key 400 fallback so we treat that as
 *    "connected" too. Missing / rejected keys are reported explicitly instead
 *    of falling through to the browser's opaque CORS/network error shape.
 *
 * Throws on failure so React Query marks the query as errored — the
 * dropdown reads `isSuccess` to flip the indicator green.
 */
async function probeBackend(backend: Backend): Promise<true> {
  if (backend.kind === "cloud") {
    if (backend.authMode !== "cookie" && !backend.apiKey?.trim()) {
      throw new Error(MISSING_BACKEND_API_KEY_ERROR);
    }

    try {
      if (backend.authMode === "cookie") {
        await getCloudOrganizations(backend);
      } else {
        await getCurrentCloudApiKey(backend);
      }
    } catch (error) {
      if (
        (axios.isAxiosError(error) && error.response?.status === 401) ||
        (error instanceof HttpError && error.status === 401)
      ) {
        throw new Error(CLOUD_BACKEND_LOGGED_OUT_ERROR);
      }
      if (isCorsOrNetworkError(error)) {
        throw new Error(CLOUD_BACKEND_API_KEY_OR_NETWORK_ERROR);
      }
      throw error;
    }
    return true;
  }

  try {
    const clientOptions = getAgentServerClientOptions({
      host: backend.host,
      sessionApiKey: backend.apiKey || null,
      timeout: PROBE_TIMEOUT_MS,
    });

    await new SettingsClient(clientOptions).getSettings();
    const serverInfo = await new ServerClient(clientOptions).getServerInfo();
    assertAgentServerVersionIsSupported(serverInfo);
  } catch (error) {
    if (isSdkHttpStatusError(error, 401)) {
      throw new Error(INVALID_BACKEND_API_KEY_ERROR);
    }
    throw error;
  }
  return true;
}

/**
 * How many extra times to re-run a failed probe before giving up, and how
 * long to wait between attempts.
 */
const PROBE_RETRY_ATTEMPTS = 2;
const PROBE_RETRY_DELAY_MS = 300;

/**
 * Probe a backend, retrying a couple of times on failure before giving up.
 *
 * The connectivity indicator (and the onboarding "backend connected" banner)
 * only flips green once a probe succeeds. With `retry: false` at the query
 * level and a REFRESH_INTERVAL_MS (10s) poll, a single transient first-probe
 * miss — the agent-server still warming up right after navigation, a momentary
 * proxy 5xx, a dropped connection — would otherwise leave the banner stuck for
 * a full 10s until the next scheduled refetch. That is the root cause of the
 * flaky onboarding e2e "backend health probe should report connected" timeout.
 *
 * Retrying here, inside the query function rather than via React Query's
 * `retry`, is deliberate: the success/failure recording below runs once per
 * settled query, so a single logical probe still records exactly one outcome.
 * Retrying at the query level instead would call `recordBackendFailure` on
 * every internal attempt and reach the disabled cap several times too fast.
 *
 * Definitive auth failures (logged out, invalid/missing key) are NOT retried:
 * they are a decided server response, not a transient miss, so retrying only
 * delays surfacing the correct "disconnected" state (and the recovery UI).
 */
function isRetryableProbeError(error: unknown): boolean {
  if (!(error instanceof Error)) return true;
  return (
    error.message !== INVALID_BACKEND_API_KEY_ERROR &&
    error.message !== MISSING_BACKEND_API_KEY_ERROR &&
    error.message !== CLOUD_BACKEND_LOGGED_OUT_ERROR
  );
}

async function probeBackendWithQuickRetry(backend: Backend): Promise<true> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await probeBackend(backend);
    } catch (error) {
      if (attempt >= PROBE_RETRY_ATTEMPTS || !isRetryableProbeError(error)) {
        throw error;
      }
      await new Promise((resolve) => {
        setTimeout(resolve, PROBE_RETRY_DELAY_MS);
      });
    }
  }
}

export interface BackendHealth {
  /** `null` while the first probe is in flight; then `true` / `false`. */
  isConnected: boolean | null;
  /** Number of consecutive failed probes since the last success. */
  consecutiveFailures: number;
  /** Last error message captured from a failed probe, if any. */
  lastError: string | null;
  /**
   * `true` once `consecutiveFailures` reaches the cap. While disabled,
   * ordinary background polling stops and survives a page refresh in
   * that state.
   */
  disabled: boolean;
}

export interface UseBackendsHealthOptions {
  /**
   * Re-probe disabled backends once when the hook mounts. Used by the
   * Manage Backends modal so a recovered backend can clear its stale
   * persisted error state without forcing the user to edit the config.
   */
  probeDisabledOnce?: boolean;
}

/**
 * Poll every backend in `backends` once every 10s and report a simple
 * connected / disconnected verdict per backend id.
 *
 * The query key includes `host` and `apiKey` so editing a backend's
 * connection details re-keys the query and triggers an immediate
 * refetch instead of waiting for the next tick.
 *
 * After `MAX_CONSECUTIVE_FAILURES` failures in a row, ordinary polling
 * stops for that backend until the user updates its host / apiKey.
 * Callers can still opt into a one-shot recheck for disabled backends
 * (for example when the user explicitly opens Manage Backends). The
 * failure count and last error live in localStorage so a page refresh
 * does not silently re-arm polling against a backend that's known to
 * be unreachable.
 */
export function useBackendsHealth(
  backends: Backend[],
  options: UseBackendsHealthOptions = {},
): Record<string, BackendHealth> {
  const { probeDisabledOnce = false } = options;
  const healthMap = React.useSyncExternalStore(
    subscribeBackendHealth,
    getHealthSnapshot,
    getHealthSnapshot,
  );

  const results = useQueries({
    queries: backends.map((b) => {
      const entry = healthMap[b.id];
      const hasMissingCloudApiKey = hasMissingBackendApiKey(b);
      const isDisabled = entry?.disabled === true;
      const shouldReprobeStaleCloudNetworkError =
        isDisabled &&
        b.kind === "cloud" &&
        isCorsOrNetworkErrorMessage(entry?.lastError);
      const shouldProbe =
        !hasMissingCloudApiKey &&
        (!isDisabled ||
          probeDisabledOnce ||
          shouldReprobeStaleCloudNetworkError);
      return {
        queryKey: [
          "backend-health",
          b.id,
          b.kind,
          b.host,
          b.apiKey ?? "",
        ] as const,
        queryFn: async () => {
          try {
            const result = await probeBackendWithQuickRetry(b);
            recordBackendSuccess(b.id);
            return result;
          } catch (err) {
            recordBackendFailure(b.id, err);
            throw err;
          }
        },
        enabled: shouldProbe,
        refetchInterval:
          isDisabled || hasMissingCloudApiKey
            ? (false as const)
            : REFRESH_INTERVAL_MS,
        refetchIntervalInBackground: false,
        refetchOnMount: isDisabled && probeDisabledOnce ? "always" : true,
        refetchOnReconnect: !isDisabled && !hasMissingCloudApiKey,
        refetchOnWindowFocus: !isDisabled && !hasMissingCloudApiKey,
        retry: false,
        // Keep the previous verdict visible while the next probe is in
        // flight so the indicator doesn't flicker on routine polling.
        staleTime: isDisabled ? 0 : REFRESH_INTERVAL_MS,
        meta: { disableToast: true },
      };
    }),
  });

  const out: Record<string, BackendHealth> = {};
  backends.forEach((b, i) => {
    const r = results[i];
    const entry = healthMap[b.id];
    const hasMissingCloudApiKey = hasMissingBackendApiKey(b);
    const disabled = hasMissingCloudApiKey ? false : entry?.disabled === true;
    const consecutiveFailures = hasMissingCloudApiKey
      ? 0
      : (entry?.consecutiveFailures ?? 0);
    const lastError = hasMissingCloudApiKey
      ? MISSING_BACKEND_API_KEY_ERROR
      : (entry?.lastError ?? null);

    let isConnected: boolean | null;
    if (hasMissingCloudApiKey) {
      isConnected = false;
    } else if (disabled) {
      // Polling stopped after hitting the cap — treat as disconnected
      // so existing consumers (dot, badge) render red without needing
      // to know about the new fields.
      isConnected = false;
    } else if (r.isSuccess) isConnected = true;
    else if (r.isError) isConnected = false;
    else isConnected = null;

    out[b.id] = { isConnected, consecutiveFailures, lastError, disabled };
  });
  return out;
}

export { MAX_CONSECUTIVE_FAILURES };
